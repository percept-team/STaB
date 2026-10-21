import json
import pickle
import numpy as np
import pytest
import torch
from PIL import Image
from pycocotools import mask as coco_mask
from rfvl.converters import mevis_records, grove_records, activitynet_records, videorefer_records
from rfvl.data import audit_manifests, make_sample, materialize_region, validate_record, write_jsonl
from rfvl.backbone import ByteTokenizer
from rfvl.features import align_times


def rle(mask):
    value = coco_mask.encode(np.asfortranarray(mask.astype(np.uint8)))
    value["counts"] = value["counts"].decode()
    return value


def test_mevis_union_absence_and_expression_grouping(tmp_path):
    root = tmp_path
    (root / "JPEGImages/v").mkdir(parents=True)
    for i in range(3):
        Image.new("RGB", (16, 16)).save(root / f"JPEGImages/v/{i:05}.jpg")
    first, second, empty = np.zeros((16, 16)), np.zeros((16, 16)), np.zeros((16, 16))
    first[1:3, 1:3], second[8:10, 8:10] = 1, 1
    (root / "mask_dict.json").write_text(json.dumps({"1": [rle(first), rle(empty), rle(first)], "2": [rle(second)] * 3}))
    expressions = {"0": {"exp": "two cats", "anno_id": [1, 2]}, "1": {"exp": "cats together", "anno_id": [1, 2]}, "2": {"exp": "no dog", "anno_id": []}}
    (root / "meta_expressions.json").write_text(json.dumps({"videos": {"v": {"frames": [f"{i:05}" for i in range(3)], "expressions": expressions}}}))
    rows = list(mevis_records(root, "train", "mevis_v2"))
    assert len(rows) == 5
    absent = next(x for x in rows if x["text"] == "no dog")
    assert not any(a["present"] for a in absent["regions"][0]["annotations"])
    region = materialize_region(rows[0]["regions"][0], [[16, 16]] * 3, 16)
    assert region["annotation"]["masks"][0].sum() == 8
    generated = [x for x in rows if x["task"] == "expression"]
    assert generated[0]["references"] == ["two cats", "cats together"]


def test_grove_sparse_and_multiple_instances(tmp_path):
    (tmp_path / "v.mp4").write_bytes(b"unused converter fixture")
    ann = {"video_fname": "v.mp4", "fps": 10, "clip_start_frame": 0, "clip_end_frame": 4,
           "width": 100, "height": 50, "caption": "a cat moves", "tagged_caption": "a <p>cat</p> moves",
           "phrases": ["cat"], "bboxes": [[[1, 2, 30, 40], [40, 2, 80, 40]], [], [[2, 2, 32, 40]]],
           "labels": [["cat", "cat"], [], ["cat"]], "track_ids": [[10, 20], [], [10]]}
    path = tmp_path / "raw.pkl"
    path.write_bytes(pickle.dumps({"v_0": ann}))
    rows = list(grove_records(path, tmp_path, "train", "iground"))
    assert rows[0]["times"] == [0, .2, .4]
    assert len(rows[0]["regions"]) == 2
    assert rows[0]["regions"][0]["span"] == [2, 5]
    assert len(rows[0]["regions"][0]["annotations"]) == 2
    del ann["track_ids"]
    path.write_bytes(pickle.dumps({"v_0": ann}))
    with pytest.raises(ValueError, match="track_ids"):
        list(grove_records(path, tmp_path, "train", "iground"))
    inferred = next(grove_records(path, tmp_path, "train", "iground", infer_tracks=True))
    assert sum(len(r["annotations"]) for r in inferred["regions"]) == 3
    assert inferred["track_source"] == "iou_inference"


def test_activitynet_sparse_targets(tmp_path):
    (tmp_path / "v.mp4").write_bytes(b"unused")
    ann = {"annotations": {"v": {"width": 100, "height": 100, "segments": {"0": {"timestamps": [2, 12], "tokens": ["a", "cat", "moves"],
           "process_clss": [["cat"], ["cat"]], "process_idx": [[1], [1]], "frame_ind": [1, 8], "process_bnd_box": [[10, 10, 30, 30], [20, 10, 40, 30]]}}}}}
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(ann))
    row = next(activitynet_records(path, tmp_path, "val"))
    assert len(row["times"]) == 10
    assert len(row["regions"][0]["annotations"]) == 2
    assert row["times"][1] == 3.5
    validate_record(row)


def test_videorefer_single_multi_preserve_video_context(tmp_path):
    import imageio.v2 as iio
    video = tmp_path / "v.mp4"
    writer = iio.get_writer(video, fps=4)
    for i in range(8):
        writer.append_data(np.full((32, 32, 3), i * 20, dtype=np.uint8))
    writer.close()
    value = rle(np.ones((32, 32)))
    ann = [{"id": 0, "video": "v.mp4", "caption": "cat", "frame_idx": "2", "annotation": [{"2": {"segmentation": value}, "6": {"segmentation": value}}]}]
    path = tmp_path / "annotations.json"
    path.write_text(json.dumps(ann))
    single = next(videorefer_records(path, tmp_path, "single"))
    multi = next(videorefer_records(path, tmp_path, "multi"))
    assert single["times"] == multi["times"]
    assert len(single["regions"][0]["annotations"]) == 1
    assert len(multi["regions"][0]["annotations"]) == 2


def test_split_leakage_is_rejected(tmp_path):
    from rfvl.fixtures import create_fixtures
    root = create_fixtures(tmp_path / "fixtures")
    from rfvl.data import read_jsonl
    rows = list(read_jsonl(root / "mevis_v1.jsonl"))
    rows[1]["split"] = "val"
    write_jsonl(root / "bad.jsonl", rows)
    with pytest.raises(ValueError, match="leaks"):
        audit_manifests([root / "bad.jsonl"])


def test_temporal_interpolation_and_endpoint_clamping():
    values = torch.tensor([[0.], [10.]])
    result = align_times(values, torch.tensor([0., 2.]), torch.tensor([-1., 1., 3.]))
    assert torch.equal(result, torch.tensor([[0.], [5.], [10.]]))
