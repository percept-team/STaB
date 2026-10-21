import importlib.util
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from .data import decode_mask, read_jsonl


def mask_metrics(manifest, predictions, official_metrics):
    spec = importlib.util.spec_from_file_location("mevis_official_metrics", official_metrics)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    preds = {r["id"]: r for r in read_jsonl(predictions)}
    rows = [r for r in read_jsonl(manifest) if r["task"] == "comprehension"]
    values, presence_decisions = [], []
    for row in rows:
        result = preds.get(row["id"])
        generated = torch.load(Path(predictions).parent / result["artifact"], map_location="cpu", weights_only=True) if result else {"regions": []}
        js, fs, has_target, has_prediction = [], [], False, False
        for ann in row["regions"][0]["annotations"]:
            index = ann["index"]
            if result:
                width, height = result["sizes"][index]
            else:
                from PIL import Image
                width, height = Image.open(row["frames"][index]).size
            target = decode_mask(ann, height, width)
            if target is None:
                target = np.zeros((height, width), dtype=bool)
            has_target |= bool(target.any())
            predicted = np.zeros((height, width), dtype=bool)
            if generated["regions"]:
                region = generated["regions"][0]
                if region["presence_logits"][index] > 0:
                    predicted = (torch.nn.functional.interpolate(region["mask_logits"][index][None, None], (height, width), mode="bilinear", align_corners=False)[0, 0] > 0).numpy()
            has_prediction |= bool(predicted.any())
            js.append(float(np.asarray(module.db_eval_iou(target, predicted)).mean()))
            fs.append(float(np.asarray(module.db_eval_boundary(target, predicted)).mean()))
        values.append((np.mean(js), np.mean(fs)))
        presence_decisions.append((has_target, has_prediction))
    if not values:
        raise ValueError("no comprehension records")
    j, f = np.mean(values, axis=0) * 100
    positive = [pred for target, pred in presence_decisions if target]
    negative = [not pred for target, pred in presence_decisions if not target]
    return {"J": float(j), "F": float(f), "J&F": float((j + f) / 2), "samples": len(values),
            "clip_target_accuracy_diagnostic": float(np.mean(positive) * 100) if positive else None,
            "clip_no_target_accuracy_diagnostic": float(np.mean(negative) * 100) if negative else None,
            "metric_source": str(Path(official_metrics).resolve())}


def caption_metrics(manifest, predictions):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.meteor.meteor import Meteor
    rows = {r["id"]: r for r in read_jsonl(manifest)}
    gts, res = {}, {}
    for i, pred in enumerate(read_jsonl(predictions)):
        row = rows[pred["id"]]
        if row["task"] == "comprehension":
            continue
        references = row.get("references", [row["text"]])
        gts[i] = [{"caption": text} for text in references]
        res[i] = [{"caption": pred["text"]}]
    if not gts:
        raise ValueError("no predictions")
    tokenizer = PTBTokenizer()
    gt, prediction = tokenizer.tokenize(gts), tokenizer.tokenize(res)
    cider, _ = Cider().compute_score(gt, prediction)
    meteor_scorer = Meteor()
    meteor, _ = meteor_scorer.compute_score(gt, prediction)
    return {"CIDEr": float(cider * 100), "METEOR": float(meteor * 100), "samples": len(gts), "scale": "COCO score x 100"}


def protocol_metrics(predictions):
    items = [p for p in read_jsonl(predictions) if p["task"] == "grounded_caption"]
    if not items:
        raise ValueError("no grounded-caption predictions")
    by_video = {}
    for item in items:
        by_video.setdefault(item["video_id"], []).append(bool(item["protocol"]["valid"]))
    return {"PVR": 100 * sum(all(x) for x in by_video.values()) / len(by_video), "videos": len(by_video)}


def official_run(repository, benchmark, prediction, annotations, output, split="val", split_file=None, loc_mode="all"):
    repository, output = Path(repository).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if benchmark == "iground":
        command = [sys.executable, str(repository / "eval_iground.py"), "--split", split, "--pred_file_path", str(Path(prediction).resolve()),
                   "--gt_file_path", str(Path(annotations).resolve()), "--save_dir", str(output / "coco"), "--evaluation_mode", "flattening"]
    elif benchmark == "activitynet":
        command = [sys.executable, str(repository / "eval_anet.py"), "--submission", str(Path(prediction).resolve()),
                   "--reference", str(Path(annotations).resolve()), "--eval_mode", "gen", "--loc_mode", loc_mode]
        if split_file:
            command += ["--split_file", str(Path(split_file).resolve()), "--split", split]
    else:
        raise ValueError("use VideoRefer's own GPT scoring command after exporting predictions")
    (output / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
    with (output / "official.log").open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=repository, stdout=handle, stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"official evaluator failed; see {output / 'official.log'}")
    return {"status": "completed", "log": str(output / "official.log")}


def roundtrip_inputs(manifest, predictions, output):
    rows = {r["id"]: r for r in read_jsonl(manifest)}
    result = []
    for pred in read_jsonl(predictions):
        row = rows[pred["id"]]
        if row["task"] != "expression":
            continue
        result.append({**row, "id": row["id"] + "/roundtrip", "task": "comprehension", "reference_expression": row["text"], "text": pred["text"]})
    from .data import write_jsonl
    write_jsonl(output, result)
    return {"samples": len(result), "output": str(output), "next": "Run an independent frozen listener; evaluate its output using the original mask targets."}


def prepare_center(annotations, output):
    import pickle
    with open(annotations, "rb") as handle:
        records = pickle.load(handle)
    centered = {}
    for key, row in records.items():
        middle = len(row["bboxes"]) // 2
        centered[key] = {**row, "bboxes": [row["bboxes"][middle]], "labels": [row["labels"][middle]]}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as handle:
        pickle.dump(centered, handle)
    return {"clips": len(centered), "output": str(output)}
