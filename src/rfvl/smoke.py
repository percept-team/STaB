import json
from pathlib import Path
import torch
from .data import audit_manifests
from .engine import calibrate, train
from .fixtures import create_fixtures
from .inference import export_predictions, infer
from .evaluation import protocol_metrics


def run_smoke(cfg, output):
    if not cfg.get("tiny"):
        raise ValueError("smoke uses the offline tiny configuration")
    torch.set_num_threads(2)
    output = Path(output).resolve()
    fixtures = create_fixtures(output / "fixtures")
    manifest, cache = str(fixtures / "all.jsonl"), str(fixtures / "cache")
    coverage = audit_manifests([manifest])
    codec = train(cfg, [manifest], cache, output / "codec", stage="codec", device="cpu", steps=2)
    calibrated = output / "calibrated.pt"
    calibration = calibrate(cfg, [manifest], cache, codec, calibrated, "cpu")
    pretrain = train(cfg, [str(fixtures / "howtoground1m.jsonl")], cache, output / "pretrain", stage="pretrain", checkpoint=calibrated, device="cpu", steps=1)
    joint = train(cfg, [manifest], cache, output / "joint", stage="joint", checkpoint=pretrain, device="cpu", steps=2)
    joint = train(cfg, [manifest], cache, output / "joint", stage="joint", checkpoint=joint, resume=True, device="cpu", steps=3)
    transfer = train(cfg, [str(fixtures / "activitynet_entities.jsonl")], cache, output / "transfer", stage="transfer", checkpoint=joint, device="cpu", steps=1)
    predictions = infer(cfg, manifest, cache, joint, output / "inference", device="cpu")
    exports = {}
    for kind in ["mevis", "iground", "activitynet", "videorefer"]:
        exports[kind] = export_predictions(manifest, predictions, output / "exports" / kind, kind)
    report = {"status": "passed", "backbone": "Transformers Qwen2_5_VLTextModel, random tiny configuration + PEFT LoRA",
              "coverage": coverage, "calibration": calibration, "exports": exports, "pvr": protocol_metrics(predictions),
              "transfer_checkpoint": str(transfer), "limitations": ["synthetic images and cached random features", "not pretrained Q3B or SAM", "not a benchmark result"]}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
