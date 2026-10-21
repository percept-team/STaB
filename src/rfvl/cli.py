import argparse
import json
import platform
from pathlib import Path
import torch
import yaml


def load_config(path):
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if "extends" in config:
        config = {**load_config(path.parent / config.pop("extends")), **config}
    return config


def main():
    parser = argparse.ArgumentParser(prog="rfvl")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    fixtures = commands.add_parser("fixtures")
    fixtures.add_argument("--output", required=True)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--config", default="configs/tiny_cpu.yaml")
    smoke.add_argument("--output", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("manifests", nargs="+")
    center = commands.add_parser("prepare-center")
    center.add_argument("--annotations", required=True)
    center.add_argument("--output", required=True)
    convert = commands.add_parser("convert")
    from .data import DATASETS
    convert.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    convert.add_argument("--root", required=True)
    convert.add_argument("--annotations")
    convert.add_argument("--split", default="train")
    convert.add_argument("--fps", type=float, default=1)
    convert.add_argument("--track-map")
    convert.add_argument("--infer-tracks", action="store_true", help="opt-in IoU tracking when raw annotation lacks track IDs")
    convert.add_argument("--sizes")
    convert.add_argument("--video-ids")
    convert.add_argument("--fps-map")
    convert.add_argument("--mode", choices=["single", "multi"], default="multi")
    convert.add_argument("--output", required=True)
    for name in ["cache", "train", "calibrate", "infer"]:
        sub = commands.add_parser(name)
        sub.add_argument("--config", required=True)
        sub.add_argument("--manifest", nargs="+" if name in {"train", "calibrate"} else None, required=True)
        sub.add_argument("--device", default="cuda")
        sub.add_argument("--output", required=True)
        if name == "cache":
            sub.add_argument("--sam-checkpoint", required=True)
        else:
            sub.add_argument("--cache", required=True)
            sub.add_argument("--checkpoint", required=name != "train")
        if name == "train":
            sub.add_argument("--stage", choices=["codec", "pretrain", "joint", "transfer"], default="joint")
            sub.add_argument("--resume", action="store_true")
            sub.add_argument("--steps", type=int)
        if name == "infer":
            sub.add_argument("--limit", type=int)
    export = commands.add_parser("export")
    export.add_argument("--manifest", required=True)
    export.add_argument("--predictions", required=True)
    export.add_argument("--kind", choices=["mevis", "iground", "activitynet", "videorefer"], required=True)
    export.add_argument("--center", action="store_true")
    export.add_argument("--output", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--metric", choices=["mask", "caption", "pvr", "roundtrip-inputs", "videorefer-scores"], required=True)
    evaluate.add_argument("--manifest")
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--official-metrics")
    evaluate.add_argument("--output", required=True)
    official = commands.add_parser("official-eval")
    official.add_argument("--benchmark", choices=["iground", "activitynet"], required=True)
    official.add_argument("--repository", required=True)
    official.add_argument("--predictions", required=True)
    official.add_argument("--annotations", required=True)
    official.add_argument("--split", default="val")
    official.add_argument("--split-file")
    official.add_argument("--loc-mode", choices=["all", "loc"], default="all")
    official.add_argument("--output", required=True)
    judge = commands.add_parser("videorefer-judge")
    judge.add_argument("--predictions", required=True)
    judge.add_argument("--system-prompt", required=True)
    judge.add_argument("--output", required=True)
    args = parser.parse_args()
    command = args.command
    if command == "doctor":
        import transformers
        print(json.dumps({"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
                          "cuda": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None}, indent=2))
        return
    if command == "fixtures":
        from .fixtures import create_fixtures
        result = str(create_fixtures(args.output))
    elif command == "smoke":
        from .smoke import run_smoke
        result = run_smoke(load_config(args.config), args.output)
    elif command == "audit":
        from .data import audit_manifests
        result = audit_manifests(args.manifests)
    elif command == "prepare-center":
        from .evaluation import prepare_center
        result = prepare_center(args.annotations, args.output)
    elif command == "convert":
        from .converters import convert as run_convert
        result = {"records": run_convert(args), "output": args.output}
    elif command == "cache":
        from .features import extract_features
        result = extract_features(args.manifest, args.output, load_config(args.config), args.sam_checkpoint, args.device)
    elif command == "train":
        from .engine import train
        result = str(train(load_config(args.config), args.manifest, args.cache, args.output, args.stage, args.checkpoint, args.resume, args.device, args.steps))
    elif command == "calibrate":
        from .engine import calibrate
        result = calibrate(load_config(args.config), args.manifest, args.cache, args.checkpoint, args.output, args.device)
    elif command == "infer":
        from .inference import infer
        result = str(infer(load_config(args.config), args.manifest, args.cache, args.checkpoint, args.output, args.device, args.limit))
    elif command == "export":
        from .inference import export_predictions
        result = export_predictions(args.manifest, args.predictions, args.output, args.kind, args.center)
    elif command == "official-eval":
        from .evaluation import official_run
        result = official_run(args.repository, args.benchmark, args.predictions, args.annotations, args.output, args.split, args.split_file, args.loc_mode)
    elif command == "videorefer-judge":
        from .judge import judge_videorefer
        result = judge_videorefer(args.predictions, args.system_prompt, args.output)
    else:
        from . import evaluation
        if args.metric == "mask":
            if not args.official_metrics:
                parser.error("mask evaluation requires --official-metrics /path/to/MeViS/tools/metrics.py")
            result = evaluation.mask_metrics(args.manifest, args.predictions, args.official_metrics)
        elif args.metric == "caption":
            result = evaluation.caption_metrics(args.manifest, args.predictions)
        elif args.metric == "pvr":
            result = evaluation.protocol_metrics(args.predictions)
        elif args.metric == "videorefer-scores":
            from .judge import aggregate_scores
            result = aggregate_scores(args.predictions)
        else:
            result = evaluation.roundtrip_inputs(args.manifest, args.predictions, args.output)
        if args.metric != "roundtrip-inputs":
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
