import json
from pathlib import Path
import torch
from torch.nn import functional as F
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from .data import cache_key, load_frames, read_jsonl


def align_times(values, source, target):
    source, target = source.to(values.device), target.to(values.device)
    if len(source) == 1:
        return values[:1].expand(len(target), *values.shape[1:])
    upper = torch.searchsorted(source, target).clamp(1, len(source) - 1)
    lower = upper - 1
    weight = ((target - source[lower]) / (source[upper] - source[lower])).clamp(0, 1)
    return torch.lerp(values[lower], values[upper], weight.reshape(-1, *([1] * (values.ndim - 1))))


@torch.no_grad()
def extract_features(manifest, output, cfg, sam_checkpoint, device="cuda"):
    from sam2.build_sam import build_sam2
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"backbone": cfg["backbone"], "sam_checkpoint": str(Path(sam_checkpoint).resolve()),
                "semantic_frames": cfg.get("semantic_frames", 16), "semantic_size": cfg.get("semantic_size", 448),
                "spatial_size": cfg.get("spatial_size", 512), "semantic_grid": cfg.get("semantic_grid", 4),
                "sampling": "independent image encodings; timestamp interpolation", "spatial_grid": cfg.get("spatial_grid", 32)}
    meta_path = output / "feature_config.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != metadata:
        raise ValueError("cache configuration changed; choose a new cache directory")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    processor = AutoProcessor.from_pretrained(cfg["backbone"])
    full = Qwen2_5_VLForConditionalGeneration.from_pretrained(cfg["backbone"], torch_dtype=dtype)
    visual = full.model.visual.to(device).eval()
    del full
    sam = build_sam2(cfg.get("sam_config", "configs/sam2.1/sam2.1_hiera_b+.yaml"), sam_checkpoint, device=device).eval()
    for row in read_jsonl(manifest):
        path = output / f"{cache_key(row)}.pt"
        if path.exists():
            continue
        frames = load_frames(row)
        times = torch.tensor(row["times"], dtype=torch.float32)
        count = min(cfg.get("semantic_frames", 16), len(frames))
        selection = torch.linspace(0, len(frames) - 1, count).round().long().unique()
        grids = []
        size = cfg.get("semantic_size", 448)
        pool_size = cfg.get("semantic_grid", 4)
        for i in selection:
            batch = processor.image_processor(images=[frames[int(i)].resize((size, size))], return_tensors="pt", min_pixels=size**2, max_pixels=size**2)
            grid = batch["image_grid_thw"].to(device)
            value = visual(batch["pixel_values"].to(device, dtype), grid_thw=grid)
            merge = visual.config.spatial_merge_size
            h, w = int(grid[0, 1]) // merge, int(grid[0, 2]) // merge
            value = value.reshape(h, w, -1).permute(2, 0, 1)
            grids.append(F.adaptive_avg_pool2d(value.float(), (pool_size, pool_size)).permute(1, 2, 0).cpu())
        semantic = torch.stack(grids)
        aligned = align_times(semantic, times[selection], times)
        spatial = []
        import numpy as np
        spa_size = cfg.get("spatial_size", 512)
        for image in frames:
            array = torch.tensor(np.array(image.resize((spa_size, spa_size))), device=device).permute(2, 0, 1).float()[None] / 255
            mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
            std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
            encoded = sam.forward_image((array - mean) / std)["backbone_fpn"][-1]
            spatial.append(F.adaptive_avg_pool2d(encoded.float(), (cfg.get("spatial_grid", 32),) * 2)[0].cpu())
        torch.save({"visual": semantic.reshape(-1, semantic.shape[-1]).half(),
                    "visual_times": times[selection].repeat_interleave(pool_size**2), "semantic": aligned.half(),
                    "spatial": torch.stack(spatial).half(), "times": times, "sizes": [list(x.size) for x in frames]}, path)
        print(f"cached {row['video_id']} -> {path.name}", flush=True)
