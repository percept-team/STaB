import json
import pickle
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .data import cache_key, write_jsonl


def create_fixtures(output):
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    frames, mask_paths = [], []
    for index in range(4):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[8:20, 4 + index:16 + index] = [200, 80, 40]
        path = root / f"frame_{index}.png"
        Image.fromarray(image).save(path)
        frames.append(str(path))
        mask = (image[:, :, 0] > 0).astype(np.uint8) * 255
        mask_path = root / f"mask_{index}.png"
        Image.fromarray(mask).save(mask_path)
        mask_paths.append(str(mask_path))
    mask_region = {"track_id": "target", "type": "mask", "annotations": [{"index": i, "present": True, "mask_path": path} for i, path in enumerate(mask_paths)]}
    box_region = {"track_id": "0", "type": "box", "span": [0, 3], "annotations": [{"index": i, "present": True, "box": [.125 + i / 32, .25, .5 + i / 32, .625]} for i in [0, 2]]}
    records = []
    for dataset, tasks in [("mevis_v1", ["comprehension", "expression"]), ("mevis_v2", ["comprehension", "expression"]),
                           ("howtoground1m", ["grounded_caption"]), ("iground", ["grounded_caption"]),
                           ("activitynet_entities", ["grounded_caption"]), ("videorefer_d", ["region_description"])]:
        for task in tasks:
            row = {"id": f"{dataset}/train/smoke/{task}", "dataset": dataset, "video_id": dataset + "_smoke", "split": "train",
                   "task": task, "frames": frames, "times": [0., 1., 2., 3.], "text": "cat moves", "regions": [box_region if task == "grounded_caption" else mask_region],
                   "source_clip_id": "smoke", "segment_id": "0", "source_id": 0}
            records.append(row)
        write_jsonl(root / f"{dataset}.jsonl", [r for r in records if r["dataset"] == dataset])
    write_jsonl(root / "all.jsonl", records)
    cache = root / "cache"
    cache.mkdir(exist_ok=True)
    generator = torch.Generator().manual_seed(13)
    for row in records:
        path = cache / f"{cache_key(row)}.pt"
        if path.exists():
            continue
        torch.save({"visual": torch.randn(4, 64, generator=generator), "visual_times": torch.arange(4).float(),
                    "semantic": torch.randn(4, 2, 2, 64, generator=generator), "spatial": torch.randn(4, 16, 8, 8, generator=generator),
                    "times": torch.arange(4).float(), "sizes": [[32, 32]] * 4}, path)
    return root
