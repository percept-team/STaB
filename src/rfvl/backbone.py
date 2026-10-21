import torch
from torch import nn
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLTextModel

SPECIAL = ["<|region_start|>", "<|region_end|>", "<|entity_start|>", "<|entity_end|>", "<|rfvl_end|>"]


class ByteTokenizer:
    """Offline tokenizer used only by the CPU smoke configuration."""
    def __init__(self):
        self.base_size = 256
        self.tokens = {s: 256 + i for i, s in enumerate(SPECIAL)}

    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))

    def decode(self, ids, **kwargs):
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")


def block_mask(query_blocks, key_blocks=None, dtype=torch.float32):
    key_blocks = query_blocks if key_blocks is None else key_blocks
    visible = key_blocks[None, :] <= query_blocks[:, None]
    return torch.zeros(1, 1, len(query_blocks), len(key_blocks), device=query_blocks.device, dtype=dtype).masked_fill(~visible[None, None], torch.finfo(dtype).min)


class QwenBackbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tiny = cfg.get("tiny", False)
        if self.tiny:
            self.tokenizer = ByteTokenizer()
            config = Qwen2_5_VLTextConfig(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                                        rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3]})
            config._attn_implementation = "eager"
            self.text = Qwen2_5_VLTextModel(config)
            self.lm_head = nn.Linear(64, 256, bias=False)
        else:
            name = cfg["backbone"]
            self.tokenizer = AutoTokenizer.from_pretrained(name)
            dtype = torch.bfloat16 if cfg.get("precision", "bf16") == "bf16" else torch.float32
            full = Qwen2_5_VLForConditionalGeneration.from_pretrained(name, torch_dtype=dtype, attn_implementation=cfg.get("attention", "sdpa"))
            self.text = full.model.language_model
            self.lm_head = full.lm_head
            self.tokenizer.base_size = self.lm_head.out_features
            self.tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL})
            self.tokenizer.tokens = {s: self.tokenizer.convert_tokens_to_ids(s) for s in SPECIAL}
            # Qwen's vocabulary may contain unused IDs beyond the tokenizer length.
            self.special_ids = [self.tokenizer.tokens[s] for s in SPECIAL]
            del full
        self.base_size = self.lm_head.out_features
        if self.tiny:
            self.special_ids = list(self.tokenizer.tokens.values())
        self.text.requires_grad_(False)
        self.lm_head.requires_grad_(False)
        if cfg.get("lora", True):
            from peft import LoraConfig, get_peft_model
            self.text = get_peft_model(self.text, LoraConfig(r=cfg.get("lora_rank", 16), lora_alpha=cfg.get("lora_alpha", 32),
                lora_dropout=cfg.get("lora_dropout", 0.05), target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        self.hidden_size = self.text.config.hidden_size
        self.special_embeddings = nn.Embedding(len(SPECIAL), self.hidden_size)
        self.special_head = nn.Linear(self.hidden_size, len(SPECIAL), bias=False)
        if cfg.get("gradient_checkpointing", False):
            self.text.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    @property
    def dtype(self):
        return self.text.get_input_embeddings().weight.dtype

    def embed(self, ids):
        result = self.text.get_input_embeddings()(ids.clamp_max(self.base_size - 1))
        for index, token in enumerate(self.special_ids):
            result = torch.where((ids == token)[..., None], self.special_embeddings.weight[index].to(result), result)
        return result

    def logits(self, hidden):
        logits = self.lm_head(hidden.to(self.lm_head.weight.dtype)).float()
        maximum = max(self.base_size, max(self.special_ids) + 1)
        if maximum > self.base_size:
            logits = torch.nn.functional.pad(logits, (0, maximum - self.base_size), value=-1e9)
        extra = self.special_head(hidden.to(self.special_head.weight.dtype)).float()
        logits[..., self.special_ids] = extra
        return logits

    def forward(self, embeddings, blocks, past=None, key_blocks=None, offset=0, use_cache=False):
        count = embeddings.shape[0]
        positions = torch.arange(offset, offset + count, device=embeddings.device)[None, None].expand(3, 1, -1)
        output = self.text(inputs_embeds=embeddings[None].to(self.dtype),
                           attention_mask=block_mask(blocks, key_blocks, self.dtype), position_ids=positions,
                           past_key_values=past, use_cache=use_cache, return_dict=True)
        return output.last_hidden_state[0].float(), output.past_key_values
