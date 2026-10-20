# RFVL

Official implementation of **Referentially Faithful Video-Language Generation with Autoregressive and Brownian-Bridge Diffusion**.

## Features

- **Referring motion comprehension**: video and expression → object masks.
- **Referring motion expression generation**: video and object regions → discriminative referring expressions.
- **Spatially grounded video captioning**: interleaved generation of text and object regions.

## Datasets

| Dataset | Usage |
|---|---|
| MeViSv1 / MeViSv2 | Referring motion comprehension and expression generation, including multi-object and no-target samples |
| HowToGround1M | Pretraining for spatially grounded video captioning |
| iGround | Training and evaluation for spatially grounded video captioning |
| VideoRefer-Bench D | Region-conditioned description with single-frame / multi-frame prompts |
| ActivityNet-Entities | Transfer learning and evaluation with sparse region annotations |

## Quick Start

Run the commands from the project root containing `pyproject.toml`. Python 3.10–3.12 is recommended.

### GPU Environment

```bash
bash scripts/bootstrap_autodl.sh
python -m rfvl.cli doctor
```

## Training and Inference

The pretrained weights are from [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) and [SAM 2.1 Hiera-Base+](https://huggingface.co/facebook/sam2.1-hiera-base-plus). Follow the [Usage Guide](docs/USAGE.md) to download the weights, convert the data, and prepare the manifests first.

```bash
bash scripts/train_4090.sh

python -m rfvl.cli infer --config configs/q3b_4090.yaml \
  --manifest data/manifests/mevis_v1_val.jsonl --cache cache/q3b \
  --checkpoint runs/q3b/joint/last.pt --output runs/q3b/mevis_v1_val
```

## Reproduction Notes

This implementation builds on [Transformers](https://github.com/huggingface/transformers), [PEFT](https://github.com/huggingface/peft), and [SAM 2](https://github.com/facebookresearch/sam2). The reproduction documentation draws on [PaperHarness](https://github.com/plll4zzx/paper-harness) and [fxxk-coming-soon](https://github.com/Nitrogen216/fxxk-coming-soon).
