import json
import pickle
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from .data import FeatureDataset
from .engine import build_model


def parse_entities(ids, regions, model, terminated):
    phrases, bindings = [], []
    inside, phrase_tokens, active, has_block = False, [], None, False
    valid = terminated
    decoded_positions = {region["token_position"] for region in regions}
    for position, token in enumerate(ids):
        if token == model.entity_start:
            if inside or active is not None and not has_block:
                valid = False
            inside, phrase_tokens, active = True, [], None
        elif token == model.entity_end:
            if not inside:
                valid = False
            phrase = model.backbone.tokenizer.decode(phrase_tokens, skip_special_tokens=True).strip()
            if not phrase:
                valid = False
            phrases.append(phrase)
            active, inside, has_block = len(phrases) - 1, False, False
        elif token == model.region_start:
            if active is None or inside or position not in decoded_positions:
                valid = False
            if position + 1 >= len(ids) or ids[position + 1] != model.region_end:
                valid = False
            bindings.append(active)
            has_block = True
        elif token == model.region_end:
            if position == 0 or ids[position - 1] != model.region_start:
                valid = False
        elif inside:
            phrase_tokens.append(token)
    valid = valid and not inside and active is not None and has_block and bool(phrases)
    return {"valid": bool(valid), "phrases": phrases, "bindings": bindings}


@torch.no_grad()
def infer(cfg, manifest, cache, checkpoint, output, device="cuda", limit=None):
    model, _ = build_model(cfg, device, checkpoint)
    if not bool(model.codec.calibrated):
        raise ValueError("checkpoint codec has not been calibrated")
    model.eval()
    dataset = FeatureDataset(manifest, cache, model.backbone.tokenizer)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    groups = set()
    for index, row in enumerate(dataset.records):
        if limit is not None and len(records) >= limit:
            break
        if row.get("group_id"):
            if row["group_id"] in groups:
                continue
            groups.add(row["group_id"])
        sample = dataset[index]
        generator = torch.Generator(device=device).manual_seed(cfg.get("inference_seed", 101))
        cap = cfg.get("max_expression_tokens", 64) if row["task"] == "expression" else cfg.get("max_caption_tokens", 128)
        generated = model.generate(sample, max_tokens=cap, max_regions=cfg.get("max_regions", 16), generator=generator,
                                   force_regions=1 if row["task"] == "comprehension" else 0)
        protocol = parse_entities(generated["ids"], generated["regions"], model, generated["terminated"]) if row["task"] == "grounded_caption" else None
        artifact = f"sample_{index:07d}.pt"
        torch.save(generated, output / artifact)
        regions = []
        for number, region in enumerate(generated["regions"]):
            regions.append({"boxes": region["boxes"].tolist(), "presence": region["presence_logits"].sigmoid().tolist(),
                            "phrase_index": protocol["bindings"][number] if protocol and number < len(protocol["bindings"]) else None})
        result = {"id": row["id"], "video_id": row["video_id"], "task": row["task"], "text": generated["text"], "regions": regions,
                  "terminated": generated["terminated"], "protocol": protocol, "artifact": artifact, "sizes": sample["sizes"]}
        records.append(result)
        print(f"generated {row['id']}", flush=True)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for result in records:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    return output / "predictions.jsonl"


def export_predictions(manifest, prediction_file, output, kind, center=False):
    from .data import read_jsonl
    rows = {r["id"]: r for r in read_jsonl(manifest)}
    predictions = list(read_jsonl(prediction_file))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result, captions = {}, []
    for prediction in predictions:
        row = rows[prediction["id"]]
        expected = {"mevis": {"mevis_v1", "mevis_v2"}, "iground": {"iground", "howtoground1m"},
                    "activitynet": {"activitynet_entities"}, "videorefer": {"videorefer_d"}}
        if row["dataset"] not in expected[kind]:
            continue
        valid = prediction["protocol"] is None or prediction["protocol"]["valid"]
        regions = prediction["regions"] if valid else []
        if kind == "mevis":
            if row["task"] != "comprehension":
                continue
            artifact = torch.load(Path(prediction_file).parent / prediction["artifact"], map_location="cpu", weights_only=True)
            expression = row["id"].split("/")[-2]
            folder = output / row["video_id"] / expression
            folder.mkdir(parents=True, exist_ok=True)
            for index, frame in enumerate(row["frames"]):
                width, height = prediction["sizes"][index]
                mask = np.zeros((height, width), dtype=np.uint8)
                if artifact["regions"]:
                    region = artifact["regions"][0]
                    if region["presence_logits"][index] > 0:
                        mask = (F.interpolate(region["mask_logits"][index][None, None], (height, width), mode="bilinear", align_corners=False)[0, 0] > 0).numpy().astype(np.uint8) * 255
                Image.fromarray(mask).save(folder / (Path(frame).stem + ".png"))
        elif kind == "iground":
            phrases = prediction["protocol"]["phrases"] if valid else []
            boxes_per_frame, labels_per_frame = [], []
            for index, (width, height) in enumerate(prediction["sizes"]):
                boxes, labels = [], []
                for region in regions:
                    phrase = region["phrase_index"]
                    if phrase is not None and phrase < len(phrases) and region["presence"][index] >= .5:
                        boxes.append((np.array(region["boxes"][index]) * [width, height, width, height]).astype(np.float32))
                        labels.append(phrases[phrase])
                boxes_per_frame.append(np.array(boxes, dtype=np.float32).reshape(-1, 4))
                labels_per_frame.append(labels)
            if center:
                middle = len(boxes_per_frame) // 2
                boxes_per_frame, labels_per_frame = boxes_per_frame[middle:middle + 1], labels_per_frame[middle:middle + 1]
            result[row["source_clip_id"]] = {"pred_bboxes": boxes_per_frame, "pred_labels": labels_per_frame,
                                             "pred_phrases": phrases, "pred_text": prediction["text"]}
        elif kind == "activitynet":
            phrases = prediction["protocol"]["phrases"] if valid else []
            names, indices, boxes = [], [], []
            for region in regions:
                phrase_index = region["phrase_index"]
                if phrase_index is None or phrase_index >= len(phrases):
                    continue
                phrase = phrases[phrase_index]
                start = prediction["text"].find(phrase)
                names.append(phrase)
                indices.append(len(prediction["text"][:start].split()))
                sizes = np.asarray([row["annotation_size"]] * len(row["times"])) if "annotation_size" in row else np.asarray(prediction["sizes"])
                values = np.asarray(region["boxes"]) * sizes[:, [0, 1, 0, 1]]
                values[np.asarray(region["presence"]) < .5] = 0
                boxes.append(values.tolist())
            result.setdefault(row["video_id"], {})[row["segment_id"]] = {"clss": names, "idx_in_sent": indices, "bbox_for_all_frames": boxes}
        elif kind == "videorefer":
            captions.append({"id": row.get("source_id", row["id"]), "video": row.get("video", row["video_id"]),
                             "caption": row["text"], "pred": prediction["text"], "answer": prediction["text"]})
    if kind == "iground":
        with (output / "predictions.pkl").open("wb") as handle:
            pickle.dump(result, handle)
    elif kind == "activitynet":
        (output / "submission_gen.json").write_text(json.dumps({"results": result, "eval_mode": "gen", "external_data": {"used": True, "details": "Qwen2.5-VL, SAM2.1, HowToGround1M"}}), encoding="utf-8")
    elif kind == "videorefer":
        (output / "predictions.json").write_text(json.dumps(captions, ensure_ascii=False), encoding="utf-8")
    return str(output)
