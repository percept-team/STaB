import copy
import torch
from torch import nn
from torch.nn import functional as F
from .backbone import QwenBackbone
from .bridge import BrownianBridge
from .codec import RegionCodec, spatial_loss, time_encoding


class FaithfulVideoModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = QwenBackbone(cfg)
        h = self.backbone.hidden_size
        d = cfg.get("evidence_dim", 256)
        self.codec = RegionCodec(cfg.get("spatial_dim", 256), cfg.get("semantic_dim", h), d, cfg.get("codec_depth", 2), cfg.get("codec_heads", 8))
        self.visual_project = nn.Linear(cfg.get("semantic_dim", h), h)
        self.evidence_project = nn.Linear(d, h)
        self.endpoint_project = nn.Linear(d, h)
        self.video_time = nn.Linear(d, h)
        self.diffusion_time = nn.Sequential(nn.Linear(d, h), nn.SiLU(), nn.Linear(h, h))
        self.anchor_attention = nn.MultiheadAttention(h, self.backbone.text.config.num_attention_heads, batch_first=True)
        self.anchor_head = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, d))
        self.continuous_head = nn.Linear(h, d)
        self.bridge = BrownianBridge(cfg.get("bridge_steps", 8), cfg.get("sigma", 1), cfg.get("gamma", 1))
        self.region_start = self.backbone.tokenizer.tokens["<|region_start|>"]
        self.region_end = self.backbone.tokenizer.tokens["<|region_end|>"]
        self.entity_start = self.backbone.tokenizer.tokens["<|entity_start|>"]
        self.entity_end = self.backbone.tokenizer.tokens["<|entity_end|>"]
        self.eos = self.backbone.tokenizer.tokens["<|rfvl_end|>"]

    @property
    def device(self):
        return self.visual_project.weight.device

    def token(self, ids):
        return self.backbone.embed(torch.as_tensor(ids, device=self.device)).float()

    def clean_region(self, evidence, times):
        return self.evidence_project(evidence) + self.video_time(time_encoding(times, self.codec.dim))

    def visual(self, sample):
        features = sample["visual"].to(self.device).float()
        times = sample["visual_times"].to(self.device).float()
        h = self.backbone.hidden_size
        return self.visual_project(features) + time_encoding(times, h)

    def endpoint(self, marker_hidden, visual, times):
        h = self.backbone.hidden_size
        queries = marker_hidden[None] + time_encoding(times, h)
        attended = self.anchor_attention(queries[None], visual[None], visual[None], need_weights=False)[0][0]
        return self.anchor_head(torch.cat([marker_hidden.expand(len(times), -1), attended], -1))

    def encode_region(self, sample, region):
        indices = region["indices"].to(self.device)
        features = self.codec.fuse(sample["spatial"].to(self.device).float()[indices], sample["semantic"].to(self.device).float()[indices])
        evidence = self.codec.encode(features, region["indicators"].to(self.device).float())
        return features, evidence

    def append(self, embeddings, blocks, value, together=False):
        start = int(blocks[-1]) + 1
        added = torch.full((len(value),), start, device=self.device, dtype=torch.long) if together else torch.arange(start, start + len(value), device=self.device)
        return torch.cat([embeddings, value]), torch.cat([blocks, added])

    def training_loss(self, sample):
        visual = self.visual(sample)
        embeddings = visual
        blocks = torch.zeros(len(visual), device=self.device, dtype=torch.long)
        embeddings, blocks = self.append(embeddings, blocks, self.token(sample["prompt_ids"]))
        ar_positions, ar_targets, losses = [], [], []

        def append_text(ids, supervised):
            nonlocal embeddings, blocks
            begin = len(embeddings)
            embeddings, blocks = self.append(embeddings, blocks, self.token(ids))
            if supervised:
                ar_positions.extend(range(begin - 1, begin + len(ids) - 1))
                ar_targets.extend(ids)

        for event in sample["events"]:
            if event["kind"] == "text":
                append_text(event["ids"], event.get("supervised", True))
                continue
            region = sample["regions"][event["region"]]
            features, target = self.encode_region(sample, region)
            times = sample["times"].to(self.device)[region["indices"].to(self.device)].float()
            output = event.get("supervised", True)
            append_text([self.region_start], output)
            if output:
                prefix_hidden, _ = self.backbone(embeddings, blocks)
                endpoint = self.endpoint(prefix_hidden[-1], visual, times)
                index = self.bridge.sample_index()
                state = self.bridge.corrupt(target.detach(), endpoint, index)
                tau = self.bridge.grid[index + 1].to(state)
                noisy = self.clean_region(state, times) + self.endpoint_project(endpoint) + self.diffusion_time(time_encoding(tau.reshape(1), self.codec.dim))
                all_embeddings, all_blocks = self.append(embeddings, blocks, noisy, together=True)
                hidden, _ = self.backbone(all_embeddings, all_blocks)
                prediction = self.continuous_head(hidden[-len(times):])
                weights = self.cfg.get("loss_weights", {})
                losses.append(weights.get("bridge", 1) * F.mse_loss(prediction, target.detach())
                              + weights.get("anchor", 1) * F.mse_loss(endpoint, target.detach())
                              + weights.get("spatial", 1) * spatial_loss(self.codec.decode(features, prediction), region["annotation"], self.cfg.get("spatial_weights")))
            embeddings, blocks = self.append(embeddings, blocks, self.clean_region(target.detach(), times), together=True)
            append_text([self.region_end], output)
        append_text([self.eos], True)
        hidden, _ = self.backbone(embeddings, blocks)
        ar = F.cross_entropy(self.backbone.logits(hidden[ar_positions]), torch.tensor(ar_targets, device=self.device))
        region_loss = torch.stack(losses).mean() if losses else ar * 0
        loss = self.cfg.get("loss_weights", {}).get("text", 1) * ar + region_loss
        return {"loss": loss, "text": ar.detach(), "region": region_loss.detach()}

    @torch.no_grad()
    def generate(self, sample, max_tokens=128, max_regions=16, generator=None, force_regions=0):
        self.eval()
        visual = self.visual(sample)
        embeddings = visual
        blocks = torch.zeros(len(visual), device=self.device, dtype=torch.long)
        embeddings, blocks = self.append(embeddings, blocks, self.token(sample["prompt_ids"]))
        if sample.get("condition_region") is not None:
            region = sample["regions"][sample["condition_region"]]
            _, evidence = self.encode_region(sample, region)
            times = sample["times"].to(self.device)[region["indices"].to(self.device)].float()
            embeddings, blocks = self.append(embeddings, blocks, self.token([self.region_start]))
            embeddings, blocks = self.append(embeddings, blocks, self.clean_region(evidence, times), together=True)
            embeddings, blocks = self.append(embeddings, blocks, self.token([self.region_end]))
        hidden, cache = self.backbone(embeddings, blocks, use_cache=True)
        ids, regions = [], []
        forced = force_regions
        terminated = False
        for _ in range(max_tokens):
            next_id = self.region_start if forced else int(self.backbone.logits(hidden[-1]).argmax())
            forced = max(0, forced - 1)
            ids.append(next_id)
            if next_id == self.eos:
                terminated = True
                break
            value = self.token([next_id])
            old_length = len(embeddings)
            embeddings, blocks = self.append(embeddings, blocks, value)
            hidden, cache = self.backbone(value, blocks[-1:], past=cache, key_blocks=blocks, offset=old_length, use_cache=True)
            if next_id != self.region_start:
                continue
            if len(regions) >= max_regions:
                break
            times = sample["times"].to(self.device).float()
            endpoint = self.endpoint(hidden[-1], visual, times)
            prefix_cache, prefix_blocks, prefix_length = cache, blocks, len(embeddings)
            current_block = torch.full((len(times),), int(blocks[-1]) + 1, device=self.device, dtype=torch.long)
            keys = torch.cat([blocks, current_block])

            def predict(state, tau):
                value = self.clean_region(state, times) + self.endpoint_project(endpoint) + self.diffusion_time(time_encoding(tau.reshape(1), self.codec.dim))
                # Each noisy pass owns its cache; only clean evidence is committed.
                out, _ = self.backbone(value, current_block, past=copy.deepcopy(prefix_cache), key_blocks=keys, offset=prefix_length, use_cache=True)
                return self.continuous_head(out)

            evidence = self.bridge.sample(endpoint, predict, generator)
            features = self.codec.fuse(sample["spatial"].to(self.device).float(), sample["semantic"].to(self.device).float())
            decoded = self.codec.decode(features, evidence)
            regions.append({"token_position": len(ids) - 1, "evidence": evidence.cpu(), **{k: v.cpu() for k, v in decoded.items()}})
            clean = self.clean_region(evidence, times)
            embeddings, blocks = self.append(embeddings, prefix_blocks, clean, together=True)
            hidden, cache = self.backbone(clean, current_block, past=prefix_cache, key_blocks=blocks, offset=prefix_length, use_cache=True)
            old_length = len(embeddings)
            end = self.token([self.region_end])
            embeddings, blocks = self.append(embeddings, blocks, end)
            hidden, cache = self.backbone(end, blocks[-1:], past=cache, key_blocks=blocks, offset=old_length, use_cache=True)
            ids.append(self.region_end)
            if sample["task"] == "comprehension":
                terminated = True
                break
        return {"ids": ids, "regions": regions, "terminated": terminated,
                "text": self.backbone.tokenizer.decode([i for i in ids if i not in self.backbone.special_ids], skip_special_tokens=True)}
