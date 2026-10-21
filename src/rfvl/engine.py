import json
import hashlib
import math
import random
import time
from pathlib import Path
import numpy as np
import torch
from .codec import spatial_loss
from .data import FeatureDataset
from .model import FaithfulVideoModel


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(cfg, device, checkpoint=None):
    seed_all(cfg.get("seed", 13))
    model = FaithfulVideoModel(cfg).to(device)
    state = None
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["config"]["backbone"] != cfg["backbone"] or state["config"].get("tiny", False) != cfg.get("tiny", False):
            raise ValueError("checkpoint backbone does not match config")
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        allowed = lambda key: key.startswith("backbone.text.") and "lora_" not in key or key.startswith("backbone.lm_head.")
        if unexpected or any(not allowed(key) for key in missing):
            raise ValueError(f"checkpoint mismatch: {missing}, {unexpected}")
    return model, state


def save_checkpoint(path, model, optimizer=None, step=0, stage="joint", scheduler=None):
    keep = lambda key: model.cfg.get("tiny", False) or not key.startswith(("backbone.text.", "backbone.lm_head.")) or "lora_" in key
    payload = {"config": model.cfg, "model": {k: v.detach().cpu() for k, v in model.state_dict().items() if keep(k)},
               "step": step, "stage": stage, "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
               "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
               "optimizer": optimizer.state_dict() if optimizer else None, "scheduler": scheduler.state_dict() if scheduler else None}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def datasets_from(manifests, cache, tokenizer):
    return [FeatureDataset(path, cache, tokenizer) for path in manifests]


def subset_sample(sample, budget):
    if not budget or len(sample["times"]) <= budget:
        return sample
    required = {int(random.choice(region["indices"].tolist())) for region in sample["regions"]}
    remaining = list(set(range(len(sample["times"]))) - required)
    selection = sorted(required | set(random.sample(remaining, min(max(0, budget - len(required)), len(remaining)))))
    mapping = {old: new for new, old in enumerate(selection)}
    result = {**sample}
    for key in ["times", "spatial", "semantic"]:
        result[key] = sample[key][selection]
    result["sizes"] = [sample["sizes"][i] for i in selection]
    result["regions"] = []
    for region in sample["regions"]:
        valid = [j for j, old in enumerate(region["indices"].tolist()) if old in mapping]
        result["regions"].append({"indices": torch.tensor([mapping[int(region["indices"][j])] for j in valid]),
                                  "indicators": region["indicators"][valid],
                                  "annotation": {k: v[valid] for k, v in region["annotation"].items()}})
    return result


def train(cfg, manifests, cache, output, stage="joint", checkpoint=None, resume=False, device="cuda", steps=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model, state = build_model(cfg, device, checkpoint)
    datasets = datasets_from(manifests, cache, model.backbone.tokenizer)
    if stage != "codec":
        if not bool(model.codec.calibrated):
            raise ValueError("train and calibrate the codec before unified training")
        model.codec.freeze()
    else:
        model.requires_grad_(False)
        model.codec.requires_grad_(True)
    by_task = {}
    seen = set()
    for ds_index, dataset in enumerate(datasets):
        for row_index, row in enumerate(dataset.records):
            if stage == "pretrain" and row["dataset"] != "howtoground1m":
                raise ValueError("pretrain expects HowToGround1M only")
            if stage == "transfer" and row["dataset"] != "activitynet_entities":
                raise ValueError("transfer expects ActivityNet-Entities only")
            if row["split"] != "train":
                raise ValueError("training accepts only split=train")
            signature = (row["video_id"], row["task"], row["text"], tuple(r["track_id"] for r in row["regions"])) if row["dataset"].startswith("mevis") else row["id"]
            if signature in seen:
                continue
            seen.add(signature)
            by_task.setdefault(row["task"], []).append((ds_index, row_index))
    groups = [{"params": [p for n, p in model.named_parameters() if p.requires_grad and "lora_" in n], "lr": cfg.get("lora_lr", 2e-5)},
              {"params": [p for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n], "lr": cfg.get("new_lr", 1e-4)}]
    if stage == "transfer":
        for group in groups:
            group["lr"] = cfg.get("transfer_lr", 2e-5)
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.get("weight_decay", 0.01))
    accumulation = cfg.get("accumulation", 32)
    exposure_limit = None
    if steps is None:
        if stage in {"pretrain", "transfer"}:
            epochs = cfg.get("pretrain_epochs", 2) if stage == "pretrain" else cfg.get("transfer_epochs", 20)
            exposure_limit = sum(map(len, by_task.values())) * epochs
            steps = math.ceil(exposure_limit / accumulation)
        else:
            steps = cfg.get("codec_steps", 30000) if stage == "codec" else cfg.get("joint_steps", 60000)
    warmup = max(1, round(steps * cfg.get("warmup", 0.03)))
    schedule = lambda s: min((s + 1) / warmup, 1) if s < warmup else 0.5 * (1 + math.cos(math.pi * min(1, (s - warmup) / max(1, steps - warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    start = 0
    if resume:
        if not state or state["stage"] != stage or state["optimizer"] is None:
            raise ValueError("--resume requires a same-stage training checkpoint")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        restore_rng(state["rng"])
        start = state["step"]
    tasks = sorted(by_task)
    if not tasks:
        raise ValueError("no training samples")
    elapsed = time.monotonic()
    flat_samples = [pair for task in tasks for pair in by_task[task]]
    epoch_number, epoch_order = None, None
    model.train()
    if stage != "codec":
        model.codec.eval()
    with (output / "train.jsonl").open("a", encoding="utf-8") as log:
        for step in range(start, steps):
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            micro_batches = min(accumulation, exposure_limit - step * accumulation) if exposure_limit is not None else accumulation
            for micro in range(micro_batches):
                if stage in {"pretrain", "transfer"}:
                    epoch, cursor = divmod(step * accumulation + micro, len(flat_samples))
                    if epoch != epoch_number:
                        epoch_order = list(flat_samples)
                        random.Random(cfg.get("seed", 13) + epoch).shuffle(epoch_order)
                        epoch_number = epoch
                    ds, index = epoch_order[cursor]
                    task = datasets[ds].records[index]["task"]
                else:
                    task = random.choice(tasks)
                    ds, index = random.choice(by_task[task])
                sample = subset_sample(datasets[ds][index], cfg.get("spatial_frames", 16))
                if stage == "codec":
                    losses = []
                    for region in sample["regions"]:
                        features, evidence = model.encode_region(sample, region)
                        losses.append(spatial_loss(model.codec.decode(features, evidence), region["annotation"], cfg.get("spatial_weights")))
                    loss = torch.stack(losses).mean()
                else:
                    loss = model.training_loss(sample)["loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite loss at step {step}, task {task}")
                (loss / micro_batches).backward()
                total += float(loss.detach()) / micro_batches
            norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.get("clip_grad", 1.0), error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            record = {"step": step + 1, "stage": stage, "loss": total, "grad_norm": float(norm), "seconds": time.monotonic() - elapsed}
            if str(device).startswith("cuda"):
                record["peak_memory_gb"] = torch.cuda.max_memory_allocated() / 1024**3
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
            if (step + 1) % cfg.get("save_every", 500) == 0:
                save_checkpoint(output / f"step_{step + 1}.pt", model, optimizer, step + 1, stage, scheduler)
            save_checkpoint(output / "last.pt", model, optimizer, step + 1, stage, scheduler) if cfg.get("save_every", 500) == 1 else None
    save_checkpoint(output / "last.pt", model, optimizer, steps, stage, scheduler)
    return output / "last.pt"


@torch.no_grad()
def calibrate(cfg, manifests, cache, checkpoint, output, device="cuda"):
    model, _ = build_model(cfg, device, checkpoint)
    model.eval()
    count, total, square = 0, torch.zeros(model.codec.dim, dtype=torch.float64), torch.zeros(model.codec.dim, dtype=torch.float64)
    seen = set()
    for dataset in datasets_from(manifests, cache, model.backbone.tokenizer):
        for index, record in enumerate(dataset.records):
            if record["split"] != "train":
                raise ValueError("calibration must only use training videos")
            sample = dataset[index]
            for region_index, region in enumerate(sample["regions"]):
                region_key = hashlib.sha256(json.dumps(record["regions"][region_index], sort_keys=True).encode()).hexdigest()
                signature = (record["dataset"], record["video_id"], record.get("segment_id"), region_key)
                if signature in seen:
                    continue
                seen.add(signature)
                indices = region["indices"].to(device)
                features = model.codec.fuse(sample["spatial"].to(device).float()[indices], sample["semantic"].to(device).float()[indices])
                values = model.codec.encode(features, region["indicators"].to(device), standardize=False).double().cpu()
                total += values.sum(0)
                square += values.square().sum(0)
                count += len(values)
    model.codec.set_statistics(count, total, square)
    save_checkpoint(output, model, stage="calibrated")
    return {"frames": count, "checkpoint": str(output)}
