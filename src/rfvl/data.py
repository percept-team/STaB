import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image

DATASETS = {"mevis_v1", "mevis_v2", "howtoground1m", "iground", "videorefer_d", "activitynet_entities"}
TASKS = {"comprehension", "expression", "grounded_caption", "region_description"}


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            validate_record(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def validate_record(record):
    if record["dataset"] not in DATASETS or record["task"] not in TASKS:
        raise ValueError("unknown dataset/task")
    times = record["times"]
    if not times or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("times must be nonempty and strictly increasing")
    if not record.get("frames") and not record.get("video"):
        raise ValueError("frames or video is required")
    if record.get("frames") and len(record["frames"]) != len(times):
        raise ValueError("frame/time length mismatch")
    if record["task"] != "grounded_caption" and len(record.get("regions", [])) != 1:
        raise ValueError("this task needs exactly one target object/set")
    for region in record.get("regions", []):
        if not region["annotations"]:
            raise ValueError("region has no annotated times")
        seen = set()
        for ann in region["annotations"]:
            index = ann["index"]
            if index in seen or not 0 <= index < len(times):
                raise ValueError("invalid/duplicate annotated index")
            seen.add(index)
            if "present" not in ann:
                raise ValueError("presence must be explicit")
            if "box" in ann:
                x1, y1, x2, y2 = ann["box"]
                if not 0 <= x1 <= x2 <= 1 or not 0 <= y1 <= y2 <= 1:
                    raise ValueError("boxes must be normalized xyxy")
        if record["task"] == "grounded_caption":
            start, end = region["span"]
            if not 0 <= start < end <= len(record["text"]):
                raise ValueError("invalid phrase span")


def audit_manifests(paths):
    videos, ids, counts = {}, set(), {}
    for path in paths:
        for row in read_jsonl(path):
            validate_record(row)
            if row["id"] in ids:
                raise ValueError(f"duplicate sample ID: {row['id']}")
            ids.add(row["id"])
            key = ("mevis" if row["dataset"].startswith("mevis") else row["dataset"], row["video_id"])
            split = row["split"]
            if key in videos and videos[key] != split:
                raise ValueError(f"video leaks across splits: {key}")
            videos[key] = split
            label = f"{row['dataset']}/{split}/{row['task']}"
            counts[label] = counts.get(label, 0) + 1
    return counts


def cache_key(record):
    payload = {k: record.get(k) for k in ["dataset", "video_id", "frames", "video", "times", "segment"]}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def load_frames(record):
    if record.get("frames"):
        return [Image.open(path).convert("RGB") for path in record["frames"]]
    import imageio.v2 as iio
    reader = iio.get_reader(record["video"])
    try:
        fps = reader.get_meta_data()["fps"]
        return [Image.fromarray(reader.get_data(round(time * fps))).convert("RGB") for time in record["times"]]
    finally:
        reader.close()


def decode_mask(annotation, height, width):
    if "mask_path" in annotation:
        return np.asarray(Image.open(annotation["mask_path"])) > 0
    if "mask_rles" in annotation:
        from pycocotools import mask as mask_utils
        result = np.zeros((height, width), dtype=bool)
        for rle in annotation["mask_rles"]:
            if not rle:
                continue
            if isinstance(rle.get("counts"), list):
                rle = mask_utils.frPyObjects(rle, height, width)
            result |= mask_utils.decode(rle).astype(bool)
        return result
    return None


def materialize_region(region, sizes, grid_size=64):
    annotations = sorted(region["annotations"], key=lambda x: x["index"])
    indices, masks, indicators, boxes, present = [], [], [], [], []
    kind = region.get("type", "box")
    for ann in annotations:
        index = ann["index"]
        width, height = sizes[index]
        indices.append(index)
        present.append(float(ann["present"]))
        mask = decode_mask(ann, height, width)
        if kind == "mask":
            if mask is None:
                if ann["present"]:
                    raise ValueError("present mask missing")
                mask = np.zeros((height, width), dtype=bool)
            small = np.asarray(Image.fromarray(mask).resize((grid_size, grid_size), Image.Resampling.NEAREST), dtype=np.float32).copy()
            masks.append(small)
            indicators.append(small)
        else:
            box = ann.get("box", [0, 0, 0, 0])
            boxes.append(box)
            indicator = np.zeros((grid_size, grid_size), dtype=np.float32)
            x1, y1, x2, y2 = np.array(box) * grid_size
            if ann["present"]:
                indicator[int(y1):int(np.ceil(y2)), int(x1):int(np.ceil(x2))] = 1
            indicators.append(indicator)
    target = {"present": torch.tensor(present)}
    target["masks" if kind == "mask" else "boxes"] = torch.tensor(np.stack(masks if kind == "mask" else boxes)).float()
    return {"indices": torch.tensor(indices), "indicators": torch.tensor(np.stack(indicators)), "annotation": target}


def make_sample(record, features, tokenizer):
    task = record["task"]
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)
    instructions = {"comprehension": f"Find the referred object or object set: {record['text']}\n",
                    "expression": "Describe the indicated object so it can be distinguished by its motion.\n",
                    "grounded_caption": "Describe the video and ground each annotated entity.\n",
                    "region_description": "Describe the indicated object, its appearance and actions.\n"}
    special = tokenizer.tokens
    events = []
    if task == "comprehension":
        events = [{"kind": "region", "region": 0}]
    elif task in {"expression", "region_description"}:
        events = [{"kind": "region", "region": 0, "supervised": False}, {"kind": "text", "ids": encode(record["text"])}]
    else:
        groups = {}
        for index, region in enumerate(record["regions"]):
            groups.setdefault(tuple(region["span"]), []).append(index)
        position = 0
        for (start, end), indices in sorted(groups.items()):
            if start < position:
                raise ValueError("overlapping phrase spans need explicit disambiguation")
            events.append({"kind": "text", "ids": encode(record["text"][position:start]) + [special["<|entity_start|>"]] + encode(record["text"][start:end]) + [special["<|entity_end|>"]]})
            for index in sorted(indices, key=lambda i: str(record["regions"][i]["track_id"])):
                events.append({"kind": "region", "region": index})
            position = end
        events.append({"kind": "text", "ids": encode(record["text"][position:])})
    events = [e for e in events if e["kind"] != "text" or e["ids"]]
    prompt = f"<|im_start|>user\n{instructions[task]}<|im_end|>\n<|im_start|>assistant\n"
    return {**features, "task": task, "prompt_ids": encode(prompt), "events": events,
            "regions": [materialize_region(r, features["sizes"]) for r in record["regions"]],
            "condition_region": 0 if task in {"expression", "region_description"} else None}


class FeatureDataset(torch.utils.data.Dataset):
    def __init__(self, manifest, cache_dir, tokenizer):
        self.records = list(read_jsonl(manifest))
        self.cache_dir = Path(cache_dir)
        self.tokenizer = tokenizer
        if not self.records:
            raise ValueError("manifest is empty")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        features = torch.load(self.cache_dir / f"{cache_key(row)}.pt", map_location="cpu", weights_only=True)
        return make_sample(row, features, self.tokenizer)
