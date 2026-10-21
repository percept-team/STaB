import json
import pickle
import re
from pathlib import Path
import numpy as np
from PIL import Image
from .data import write_jsonl


def resolve_video(root, name):
    path = Path(root) / name
    if path.is_file():
        return str(path.resolve())
    for suffix in [".mp4", ".mkv", ".webm", ".avi", ".mov"]:
        candidate = path.with_suffix(suffix)
        if candidate.is_file():
            return str(candidate.resolve())
    raise FileNotFoundError(path)


def normalized_box(box, width, height):
    value = np.asarray(box, dtype=float) / [width, height, width, height]
    if np.any(value < -0.02) or np.any(value > 1.02):
        raise ValueError("box exceeds image bounds; check annotation dimensions")
    return value.clip(0, 1).tolist()


def mevis_records(root, split, version, fps=1.0):
    root = Path(root)
    videos = json.loads((root / "meta_expressions.json").read_text(encoding="utf-8"))["videos"]
    mask_path = root / "mask_dict.json"
    masks = json.loads(mask_path.read_text(encoding="utf-8")) if mask_path.exists() else None
    for video_id, video in videos.items():
        names = sorted(video["frames"])
        frames = [str((root / "JPEGImages" / video_id / f"{name}.jpg").resolve()) for name in names]
        grouped = {}
        for expression_id, exp in video["expressions"].items():
            raw_ids = exp.get("anno_id", [])
            if not isinstance(raw_ids, list):
                raw_ids = [raw_ids]
            anno_ids = sorted(str(x) for x in raw_ids if x is not None and str(x) != "-1")
            annotations = []
            for index, name in enumerate(names):
                if masks is not None:
                    rles = [masks[key][index] for key in anno_ids]
                    from pycocotools import mask as coco_mask
                    present = False
                    for rle in rles:
                        if rle:
                            encoded = coco_mask.frPyObjects(rle, *rle["size"]) if isinstance(rle["counts"], list) else rle
                            present |= bool(coco_mask.area(encoded).sum() > 0)
                    annotations.append({"index": index, "present": present, "mask_rles": rles})
                else:
                    path = root / "Annotations" / video_id / str(expression_id) / f"{name}.png"
                    if not path.exists():
                        raise FileNotFoundError(f"MeViS labels missing: {path}; use labeled validation release")
                    annotations.append({"index": index, "present": bool(np.asarray(Image.open(path)).any()), "mask_path": str(path.resolve())})
            base = {"id": f"{version}/{split}/{video_id}/{expression_id}", "dataset": version, "split": split,
                    "video_id": video_id, "frames": frames, "times": [int(name) / fps for name in names], "text": exp["exp"],
                    "regions": [{"track_id": "+".join(anno_ids) or "empty", "type": "mask", "annotations": annotations}]}
            yield {**base, "id": base["id"] + "/comprehension", "task": "comprehension"}
            if any(a["present"] for a in annotations):
                key = tuple(anno_ids) if anno_ids else (str(expression_id),)
                grouped.setdefault(key, []).append(base)
        for group in grouped.values():
            references = list(dict.fromkeys(row["text"] for row in group))
            # Each reference is a training target; evaluation groups by object set.
            for row in group:
                yield {**row, "id": row["id"] + "/expression", "task": "expression", "references": references,
                       "group_id": f"{version}/{split}/{video_id}/" + row["regions"][0]["track_id"]}


def infer_box_tracks(boxes_per_frame, labels_per_frame, threshold=.3, max_gap=2):
    from scipy.optimize import linear_sum_assignment
    active, next_id, output = {}, 0, []
    for frame, (boxes, labels) in enumerate(zip(boxes_per_frame, labels_per_frame)):
        ids = [None] * len(boxes)
        for label in sorted(set(str(x).lower() for x in labels)):
            current = [i for i, value in enumerate(labels) if str(value).lower() == label]
            previous = [key for key, value in active.items() if value[0] == label and frame - value[1] <= max_gap]
            if previous and current:
                a = np.array([active[key][2] for key in previous], dtype=float)
                b = np.array([boxes[i] for i in current], dtype=float)
                lo, hi = np.maximum(a[:, None, :2], b[None, :, :2]), np.minimum(a[:, None, 2:], b[None, :, 2:])
                inter = np.clip(hi - lo, 0, None).prod(-1)
                union = np.clip(a[:, 2:] - a[:, :2], 0, None).prod(-1)[:, None] + np.clip(b[:, 2:] - b[:, :2], 0, None).prod(-1)[None] - inter
                iou = inter / np.maximum(union, 1e-8)
                rows, columns = linear_sum_assignment(1 - iou)
                for row, col in zip(rows, columns):
                    if iou[row, col] >= threshold:
                        ids[current[col]] = previous[row]
            for local in current:
                if ids[local] is None:
                    ids[local], next_id = next_id, next_id + 1
                active[ids[local]] = (label, frame, boxes[local])
        output.append(ids)
    return output


def grove_records(annotation_path, video_root, split, dataset, track_map=None, infer_tracks=False):
    def entries():
        path = Path(annotation_path)
        files = sorted(path.glob("*.pkl")) if path.is_dir() else [path]
        for file in files:
            with file.open("rb") as handle:
                value = pickle.load(handle)
            if "video_fname" in value:
                yield file.stem, value
            else:
                yield from value.items()
    for clip_id, ann in entries():
        required = ["video_fname", "fps", "clip_start_frame", "clip_end_frame", "width", "height", "caption", "tagged_caption", "phrases", "bboxes", "labels"]
        missing = set(required) - ann.keys()
        if missing:
            raise ValueError(f"{clip_id}: missing GROVE fields {sorted(missing)}")
        if infer_tracks and "track_ids" not in ann:
            ann = {**ann, "track_ids": infer_box_tracks(ann["bboxes"], ann["labels"])}
        tagged = ann["tagged_caption"].strip('"').strip()
        caption = re.sub(r"</?p>", "", tagged)
        spans = {}
        for match in re.finditer(r"<p>(.*?)</p>", tagged, re.DOTALL):
            prefix = re.sub(r"</?p>", "", tagged[:match.start()])
            phrase = match.group(1)
            spans.setdefault(phrase.lower(), []).append([len(prefix), len(prefix) + len(phrase)])
        fps = float(ann["fps"])
        start, end = int(ann["clip_start_frame"]), int(ann["clip_end_frame"])
        count = len(ann["bboxes"])
        if count != len(ann["labels"]) or count < 1:
            raise ValueError("invalid GROVE frame labels")
        frame_indices = ann.get("frame_indices")
        if frame_indices is None:
            stride = max(1, int(fps / float(ann.get("annotation_fps", 5))))
            frame_indices = list(range(start, end + 1, stride))
            if len(frame_indices) > count:
                frame_indices = list(range(start, end, stride))
            if len(frame_indices) != count:
                raise ValueError(f"{clip_id}: supply explicit frame_indices; annotation grid does not match GROVE's 5 FPS rule")
        times = [float(i) / fps for i in frame_indices]
        tracks = {}
        for index, (boxes, labels) in enumerate(zip(ann["bboxes"], ann["labels"])):
            ids = ann.get("track_ids", [None] * count)[index]
            if track_map is not None and str(clip_id) in track_map:
                ids = track_map[str(clip_id)][index]
            lowered = [str(x).lower() for x in labels]
            if len(lowered) != len(set(lowered)) and ids is None:
                raise ValueError(f"{clip_id}: multiple instances need track_ids or --track-map; refusing to invent tracks")
            for local, (box, label) in enumerate(zip(boxes, lowered)):
                track_id = str(ids[local]) if ids is not None else label
                key = (label, track_id)
                tracks.setdefault(key, []).append({"index": index, "present": True, "box": normalized_box(box, ann["width"], ann["height"])})
        regions = []
        for (label, track_id), items in sorted(tracks.items()):
            candidates = spans.get(label)
            if not candidates or len(candidates) != 1:
                raise ValueError(f"{clip_id}: ambiguous tagged occurrence of {label!r}; use canonical manifest with explicit spans")
            # Only explicit annotations enter the target domain.
            regions.append({"track_id": track_id, "span": candidates[0], "type": "box", "annotations": items})
        yield {"id": f"{dataset}/{split}/{clip_id}", "dataset": dataset, "split": split, "task": "grounded_caption",
               "video_id": str(ann.get("video_id", Path(ann["video_fname"]).stem)), "video": resolve_video(video_root, ann["video_fname"]),
               "times": times, "text": caption, "regions": regions, "source_clip_id": str(clip_id),
               "segment": [start / fps, end / fps], "track_source": "iou_inference" if infer_tracks else "provided_or_unique_phrase"}


def activitynet_records(annotation_path, video_root, split, video_ids=None, sizes=None):
    data = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    for video_id, video in data["annotations"].items():
        if video_ids is not None and video_id not in video_ids:
            continue
        width = video.get("rwidth", video.get("width"))
        height = video.get("rheight", video.get("height"))
        if sizes and video_id in sizes:
            width, height = sizes[video_id]
        if width is None or height is None:
            raise ValueError(f"{video_id}: supply --sizes with dimensions in annotation coordinates (often resized width 720)")
        for segment_id, segment in video["segments"].items():
            tokens = segment["tokens"]
            caption = " ".join(tokens)
            offsets, position = [], 0
            for token in tokens:
                offsets.append((position, position + len(token)))
                position += len(token) + 1
            start, end = segment["timestamps"]
            times = [start + (i + 0.5) * (end - start) / 10 for i in range(10)]
            groups = {}
            for j, (box, frame) in enumerate(zip(segment["process_bnd_box"], segment["frame_ind"])):
                word_indices = segment["process_idx"][j]
                if not isinstance(word_indices, list):
                    word_indices = [word_indices]
                span = [offsets[min(word_indices)][0], offsets[max(word_indices)][1]]
                track = segment.get("track_ids", [None] * len(segment["frame_ind"]))[j]
                key = (tuple(span), str(track) if track is not None else "0")
                groups.setdefault(key, []).append({"index": int(frame), "present": True, "box": normalized_box(box, width, height)})
            regions = []
            for (span, track), items in groups.items():
                if len({x["index"] for x in items}) != len(items):
                    raise ValueError(f"{video_id}/{segment_id}: repeated entity at same time needs explicit track_ids")
                regions.append({"track_id": track, "span": list(span), "type": "box", "annotations": items})
            yield {"id": f"activitynet_entities/{split}/{video_id}/{segment_id}", "dataset": "activitynet_entities", "split": split,
                   "task": "grounded_caption", "video_id": video_id, "video": resolve_video(video_root, video_id), "times": times,
                   "segment": [start, end], "segment_id": str(segment_id), "annotation_size": [width, height], "text": caption, "regions": regions}


def videorefer_records(annotation_path, video_root, mode="multi", fps_map=None):
    import imageio.v2 as iio
    for ann in json.loads(Path(annotation_path).read_text(encoding="utf-8")):
        video = resolve_video(video_root, ann["video"])
        if fps_map and ann["video"] in fps_map:
            fps = fps_map[ann["video"]]
        else:
            reader = iio.get_reader(video)
            fps = reader.get_meta_data()["fps"]
            reader.close()
        tracks = ann["annotation"]
        frame_ids = sorted({int(key) for track in tracks for key in track})
        # Semantic context still covers the full clip in single-frame prompt mode.
        reader = iio.get_reader(video)
        duration = reader.get_meta_data().get("duration")
        reader.close()
        if duration is None or not np.isfinite(duration):
            raise ValueError("video duration unavailable; remux the video before conversion")
        context = np.linspace(0, max(0, duration - 1 / fps), 16).tolist()
        times = sorted(set([x / fps for x in frame_ids] + context))
        indices = {frame: times.index(frame / fps) for frame in frame_ids}
        regions = []
        if len(tracks) != 1:
            raise ValueError("description sample must identify one object track")
        for track_id, track in enumerate(tracks):
            items = []
            for key, value in track.items():
                if mode == "single" and str(key) != str(ann["frame_idx"]):
                    continue
                rle = value["segmentation"]
                items.append({"index": indices[int(key)], "present": bool(rle), "mask_rles": [rle]})
            regions.append({"track_id": str(track_id), "type": "mask", "annotations": items})
        yield {"id": f"videorefer_d/test/{ann['id']}/{mode}", "dataset": "videorefer_d", "split": "test",
               "task": "region_description", "video_id": Path(video).stem, "video": video, "times": times,
               "text": ann["caption"], "regions": regions, "prompt_mode": mode, "source_id": ann["id"]}


def convert(args):
    optional_json = lambda path: json.loads(Path(path).read_text(encoding="utf-8")) if path else None
    if args.dataset.startswith("mevis"):
        records = mevis_records(args.root, args.split, args.dataset, args.fps)
    elif args.dataset in {"iground", "howtoground1m"}:
        records = grove_records(args.annotations, args.root, args.split, args.dataset, optional_json(args.track_map), args.infer_tracks)
    elif args.dataset == "activitynet_entities":
        records = activitynet_records(args.annotations, args.root, args.split, optional_json(args.video_ids), optional_json(args.sizes))
    else:
        records = videorefer_records(args.annotations, args.root, args.mode, optional_json(args.fps_map))
    return write_jsonl(args.output, records)
