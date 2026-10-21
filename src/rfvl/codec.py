import math
import torch
from torch import nn
from torch.nn import functional as F


def time_encoding(times, dim):
    half = dim // 2
    scales = torch.exp(torch.arange(half, device=times.device, dtype=torch.float32) * (-math.log(10000) / max(half - 1, 1)))
    angles = times.float().unsqueeze(-1) * scales
    return F.pad(torch.cat([angles.sin(), angles.cos()], -1), (0, dim % 2))


class RegionCodec(nn.Module):
    def __init__(self, spatial_dim=256, semantic_dim=2048, dim=256, depth=2, heads=8):
        super().__init__()
        self.dim = dim
        self.semantic = nn.Linear(semantic_dim, dim)
        self.spatial = nn.Conv2d(spatial_dim, dim, 1)
        layers = [nn.Conv2d(dim + 3, dim, 3, padding=1), nn.GELU()]
        for _ in range(depth - 1):
            layers += [nn.Conv2d(dim, dim, 3, padding=1), nn.GELU()]
        self.encoder = nn.Sequential(*layers)
        self.query = nn.Parameter(torch.randn(1, 1, dim) / dim**0.5)
        self.compress = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.decode_layers = nn.ModuleList([nn.MultiheadAttention(dim, heads, batch_first=True) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(depth)])
        self.mask_query = nn.Linear(dim, dim)
        self.mask_features = nn.Conv2d(dim, dim, 1)
        self.box = nn.Linear(dim, 4)
        self.presence = nn.Linear(dim, 1)
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.register_buffer("calibrated", torch.tensor(False))

    def fuse(self, spatial, semantic):
        # Features are cached before trainable fusion.
        sem = self.semantic(semantic).permute(0, 3, 1, 2)
        return self.spatial(spatial) + F.interpolate(sem, spatial.shape[-2:], mode="bilinear", align_corners=False)

    def encode(self, features, indicators, standardize=True):
        t, _, h, w = features.shape
        y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=features.device), torch.linspace(-1, 1, w, device=features.device), indexing="ij")
        xy = torch.stack([x, y]).to(features).expand(t, -1, -1, -1)
        indicator = F.interpolate(indicators[:, None].to(features), (h, w), mode="nearest")
        memory = self.encoder(torch.cat([features, indicator, xy], 1)).flatten(2).transpose(1, 2)
        evidence = self.compress(self.query.expand(t, -1, -1), memory, memory, need_weights=False)[0][:, 0]
        return (evidence - self.mean) / self.std if standardize else evidence

    def decode(self, features, evidence):
        query = (evidence * self.std + self.mean).unsqueeze(1)
        memory = features.flatten(2).transpose(1, 2)
        for attn, norm in zip(self.decode_layers, self.norms):
            query = norm(query + attn(query, memory, memory, need_weights=False)[0])
        state = query[:, 0]
        mask = torch.einsum("td,tdhw->thw", self.mask_query(state), self.mask_features(features)) / self.dim**0.5
        raw = self.box(state).sigmoid()
        boxes = torch.cat([raw[:, :2], raw[:, :2] + (1 - raw[:, :2]) * raw[:, 2:]], -1)
        return {"mask_logits": mask, "boxes": boxes, "presence_logits": self.presence(state)[:, 0]}

    @torch.no_grad()
    def set_statistics(self, count, total, square_total):
        if count < 2:
            raise ValueError("at least two annotated frames required for calibration")
        mean = total / count
        variance = (square_total / count - mean.square()).clamp_min(1e-6)
        self.mean.copy_(mean.to(self.mean))
        self.std.copy_(variance.sqrt().to(self.std))
        self.calibrated.fill_(True)

    def freeze(self):
        self.requires_grad_(False)
        self.eval()


def generalized_iou(boxes, targets):
    lo, hi = torch.maximum(boxes[:, :2], targets[:, :2]), torch.minimum(boxes[:, 2:], targets[:, 2:])
    intersection = (hi - lo).clamp_min(0).prod(-1)
    area = (boxes[:, 2:] - boxes[:, :2]).clamp_min(0).prod(-1)
    target_area = (targets[:, 2:] - targets[:, :2]).clamp_min(0).prod(-1)
    union = area + target_area - intersection
    enclosing = (torch.maximum(boxes[:, 2:], targets[:, 2:]) - torch.minimum(boxes[:, :2], targets[:, :2])).clamp_min(0).prod(-1)
    return intersection / union.clamp_min(1e-8) - (enclosing - union) / enclosing.clamp_min(1e-8)


def spatial_loss(prediction, annotation, weights=None):
    weights = weights or {}
    logits = prediction["presence_logits"]
    present = annotation["present"].to(logits)
    loss = weights.get("presence", 1) * F.binary_cross_entropy_with_logits(logits, present)
    foreground = present.bool()
    if "masks" in annotation:
        masks = F.interpolate(annotation["masks"][:, None].to(logits), prediction["mask_logits"].shape[-2:], mode="nearest")[:, 0]
        pred, target = prediction["mask_logits"], masks
        prob = pred.sigmoid().flatten(1)
        flat = target.flatten(1)
        dice = 1 - (2 * (prob * flat).sum(-1) + 1) / (prob.sum(-1) + flat.sum(-1) + 1)
        loss = loss + weights.get("bce", 1) * F.binary_cross_entropy_with_logits(pred, target) + weights.get("dice", 1) * dice.mean()
    if "boxes" in annotation and foreground.any():
        boxes, target = prediction["boxes"][foreground], annotation["boxes"].to(logits)[foreground]
        loss = loss + weights.get("l1", 1) * F.l1_loss(boxes, target) + weights.get("giou", 1) * (1 - generalized_iou(boxes, target)).mean()
    return loss
