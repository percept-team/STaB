import copy
import torch
from rfvl.backbone import block_mask
from rfvl.bridge import BrownianBridge
from rfvl.codec import RegionCodec, spatial_loss
from rfvl.model import FaithfulVideoModel


def config():
    return dict(backbone="random-Qwen2.5-VL-TextModel", tiny=True, semantic_dim=64, spatial_dim=16,
                evidence_dim=16, codec_heads=4, lora_rank=4, lora_alpha=8, lora_dropout=0, seed=13)


def sample(task="comprehension", box=False):
    torch.manual_seed(7)
    annotation = {"present": torch.tensor([1., 0., 1.])}
    annotation["boxes" if box else "masks"] = torch.tensor([[.1, .1, .8, .8], [0, 0, 0, 0], [.2, .2, .6, .6]]) if box else torch.ones(3, 8, 8)
    region = {"indices": torch.tensor([0, 1, 2]), "indicators": torch.ones(3, 8, 8), "annotation": annotation}
    events = [{"kind": "region", "region": 0}]
    condition = None
    if task in {"expression", "region_description"}:
        events[0]["supervised"] = False
        events.append({"kind": "text", "ids": [65, 66]})
        condition = 0
    if task == "grounded_caption":
        events = [{"kind": "text", "ids": [258, 65, 259]}, *events, {"kind": "text", "ids": [66, 67]}]
    return {"visual": torch.randn(4, 64), "visual_times": torch.tensor([0., 0., 1., 1.]),
            "semantic": torch.randn(3, 2, 2, 64), "spatial": torch.randn(3, 16, 8, 8), "times": torch.tensor([0., 1., 2.]),
            "sizes": [[32, 24]] * 3, "task": task, "prompt_ids": [70, 71], "events": events, "regions": [region], "condition_region": condition}


def test_bridge_endpoints_and_weighted_estimator():
    bridge = BrownianBridge()
    x, endpoint = torch.randn(5, 3), torch.randn(5, 3)
    assert torch.equal(bridge.corrupt(x, endpoint, 7), endpoint)
    assert torch.allclose(bridge.probabilities.sum(), torch.tensor(1.))
    calls = []
    result = bridge.sample(endpoint, lambda state, tau: calls.append(float(tau)) or x)
    assert len(calls) == 8 and torch.equal(result, x)
    mean, variance = bridge.reverse_parameters(endpoint, x, 7)
    assert torch.allclose(mean, x / 8 + endpoint * 7 / 8)
    assert abs(float(variance) - 7 / 64) < 1e-6


def test_bridge_reverse_monte_carlo():
    torch.manual_seed(13)
    bridge = BrownianBridge()
    state, prediction = torch.zeros(20000, 1), torch.ones(20000, 1)
    mean, var = bridge.reverse_parameters(state, prediction, 3)
    sampled = mean + var.sqrt() * torch.randn_like(state)
    assert abs(float(sampled.mean()) - 0.25) < .015
    assert abs(float(sampled.var()) - float(var)) < .01


def test_block_causality():
    blocks = torch.tensor([0, 0, 1, 2, 2, 3])
    mask = block_mask(blocks)[0, 0]
    assert mask[3, 4] == 0 and mask[4, 3] == 0
    assert mask[2, 3] < -1e10 and mask[0, 2] < -1e10


def test_codec_gradient_through_frozen_decoder():
    codec = RegionCodec(16, 64, 16, heads=4)
    data = sample()
    features = codec.fuse(data["spatial"], data["semantic"])
    evidence = codec.encode(features, torch.ones(3, 8, 8)).detach().requires_grad_()
    codec.freeze()
    prediction = codec.decode(features.detach(), evidence)
    loss = spatial_loss(prediction, data["regions"][0]["annotation"])
    loss.backward()
    assert evidence.grad.abs().sum() > 0
    assert all(p.grad is None for p in codec.parameters())
    assert (prediction["boxes"][:, 2:] >= prediction["boxes"][:, :2]).all()


def test_all_tasks_forward_backward_and_generation():
    torch.set_num_threads(2)
    torch.manual_seed(13)
    model = FaithfulVideoModel(config())
    model.codec.freeze()
    for task in ["comprehension", "expression", "grounded_caption", "region_description"]:
        model.zero_grad(set_to_none=True)
        model.train()
        result = model.training_loss(sample(task, box=task == "grounded_caption"))
        assert torch.isfinite(result["loss"])
        result["loss"].backward()
        assert model.backbone.special_head.weight.grad is not None
        if task in {"comprehension", "grounded_caption"}:
            assert model.continuous_head.weight.grad.abs().sum() > 0
            assert model.anchor_head[-1].weight.grad.abs().sum() > 0
        assert all(p.grad is None for p in model.codec.parameters())
        model.eval()
        generated = model.generate(sample(task), max_tokens=4, max_regions=2, force_regions=1 if task in {"comprehension", "grounded_caption"} else 0)
        if task in {"comprehension", "grounded_caption"}:
            assert generated["regions"][0]["boxes"].shape == (3, 4)


def test_cached_attention_matches_full_and_noisy_cache_isolated():
    torch.manual_seed(13)
    model = FaithfulVideoModel(config()).eval()
    values = torch.randn(6, 64)
    blocks = torch.tensor([0, 0, 1, 2, 2, 3])
    with torch.no_grad():
        full, _ = model.backbone(values, blocks)
        _, cache = model.backbone(values[:3], blocks[:3], use_cache=True)
        noisy = copy.deepcopy(cache)
        model.backbone(values[3:5] + 9, blocks[3:5], past=noisy, key_blocks=blocks[:5], offset=3, use_cache=True)
        assert cache.get_seq_length() == 3
        clean, cache = model.backbone(values[3:5], blocks[3:5], past=cache, key_blocks=blocks[:5], offset=3, use_cache=True)
        last, _ = model.backbone(values[5:], blocks[5:], past=cache, key_blocks=blocks, offset=5, use_cache=True)
    assert torch.allclose(clean, full[3:5], atol=1e-5)
    assert torch.allclose(last, full[5:], atol=1e-5)


def test_sparse_multiple_evidence_blocks():
    torch.set_num_threads(2)
    model = FaithfulVideoModel(config())
    model.codec.freeze()
    data = sample("grounded_caption", box=True)
    region = copy.deepcopy(data["regions"][0])
    region["indices"] = torch.tensor([0, 2])
    region["indicators"] = region["indicators"][[0, 2]]
    region["annotation"] = {k: v[[0, 2]] for k, v in region["annotation"].items()}
    data["regions"].append(region)
    data["events"].insert(2, {"kind": "region", "region": 1})
    loss = model.training_loss(data)["loss"]
    loss.backward()
    assert model.continuous_head.weight.grad.abs().sum() > 0
    generated = model.generate(data, max_tokens=2, force_regions=2)
    assert len(generated["regions"]) == 2
