import json
from pathlib import Path
import torch
from rfvl.engine import build_model, calibrate, save_checkpoint, train
from rfvl.fixtures import create_fixtures
from rfvl.inference import infer, export_predictions, parse_entities
from rfvl.judge import judge_videorefer, parse_scores
from rfvl.data import FeatureDataset
from test_core import config, sample


def test_checkpoint_and_resume_reproduce_same_updates(tmp_path):
    torch.set_num_threads(2)
    cfg = {**config(), "accumulation": 1, "save_every": 1, "spatial_frames": 3, "new_lr": .001, "lora_lr": .001}
    root = create_fixtures(tmp_path / "data")
    model, _ = build_model(cfg, "cpu")
    model.codec.calibrated.fill_(True)
    initial = tmp_path / "initial.pt"
    save_checkpoint(initial, model)
    final = train(cfg, [root / "mevis_v1.jsonl"], root / "cache", tmp_path / "first", checkpoint=initial, device="cpu", steps=3)
    resumed = train(cfg, [root / "mevis_v1.jsonl"], root / "cache", tmp_path / "resumed", checkpoint=tmp_path / "first/step_2.pt", resume=True, device="cpu", steps=3)
    a, b = [torch.load(p, weights_only=False)["model"] for p in [final, resumed]]
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_future_text_cannot_change_endpoint_prefix():
    torch.set_num_threads(2)
    model, _ = build_model(config(), "cpu")
    model.eval()
    embeddings = torch.randn(7, 64)
    blocks = torch.tensor([0, 0, 1, 2, 3, 3, 4])
    with torch.no_grad():
        original, _ = model.backbone(embeddings, blocks)
        changed = embeddings.clone()
        changed[4:] += 50
        perturbed, _ = model.backbone(changed, blocks)
    assert torch.allclose(original[:4], perturbed[:4], atol=1e-6)


def test_small_sample_loss_reduces():
    torch.set_num_threads(2)
    model, _ = build_model(config(), "cpu")
    model.codec.freeze()
    data = sample("expression")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    losses = []
    for _ in range(16):
        optimizer.zero_grad()
        loss = model.training_loss(data)["loss"]
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0] * .7


def test_protocol_parser_keeps_multiple_tracks_and_rejects_orphans():
    model, _ = build_model(config(), "cpu")
    ids = [model.entity_start, 99, 97, 116, model.entity_end, model.region_start, model.region_end, model.region_start, model.region_end, model.eos]
    result = parse_entities(ids, [{"token_position": 5}, {"token_position": 7}], model, True)
    assert result == {"valid": True, "phrases": ["cat"], "bindings": [0, 0]}
    assert not parse_entities([model.region_start, model.region_end], [{"token_position": 0}], model, True)["valid"]
    assert not parse_entities(ids, [{"token_position": 5}, {"token_position": 7}], model, False)["valid"]


def test_judge_request_and_resume_without_network(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    source = tmp_path / "prediction.json"
    source.write_text(json.dumps([{"id": 1, "caption": "cat", "pred": "cat"}]))
    prompt = tmp_path / "system.txt"
    prompt.write_text("official prompt fixture")
    text = "Subject Correspondence: 5\nAppearance Description: 4\nTemporal Description: 3\nHallucination Detection: 2"
    calls = []
    class Response:
        status_code = 200
        def raise_for_status(self):
            pass
        def json(self):
            return {"choices": [{"message": {"content": text}}], "model": "gpt-4o"}
    def post(url, **kwargs):
        calls.append(kwargs["json"])
        return Response()
    destination = tmp_path / "scores.json"
    assert judge_videorefer(source, prompt, destination, post)["Avg"] == 3.5
    judge_videorefer(source, prompt, destination, post)
    assert len(calls) == 1
    assert calls[0]["messages"][0]["content"] == "official prompt fixture"
