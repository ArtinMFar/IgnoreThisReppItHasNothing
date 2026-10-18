"""Exports a trained RecurrentQwen checkpoint as a self-contained Hugging Face repo (trust_remote_code)."""

import json
import os
import shutil

import torch

import recurrent_qwen

AUTO_MAP = {
    "AutoConfig": "configuration_recurrent_qwen.RecurrentQwenConfig",
    "AutoModel": "modeling_recurrent_qwen.RecurrentQwenModel",
    "AutoModelForImageTextToText": "modeling_recurrent_qwen.RecurrentQwenForConditionalGeneration",
    "AutoModelForCausalLM": "modeling_recurrent_qwen.RecurrentQwenForConditionalGeneration",
}
CODE_FILES = ("configuration_recurrent_qwen.py", "modeling_recurrent_qwen.py")


def export(checkpoint, out_dir, base=None, dtype=torch.bfloat16):
    from recurrent_qwen.modeling_recurrent_qwen import RecurrentQwenForConditionalGeneration

    shutil.rmtree(out_dir, ignore_errors=True)
    model = RecurrentQwenForConditionalGeneration.from_pretrained(checkpoint, dtype=dtype)
    for layer in model.model.language_model.layers:
        if layer.block_type == "linear_attention":
            layer.linear_attn.record_skip_logits = False
    model.save_pretrained(out_dir, safe_serialization=True)

    src = os.path.dirname(recurrent_qwen.__file__)
    for name in CODE_FILES:
        shutil.copy(os.path.join(src, name), os.path.join(out_dir, name))
    cfg_path = os.path.join(out_dir, "config.json")
    cfg = json.load(open(cfg_path))
    cfg["auto_map"] = AUTO_MAP
    cfg["architectures"] = ["RecurrentQwenForConditionalGeneration"]
    for key in ("_attn_implementation", "attn_implementation"):
        cfg.pop(key, None)
        cfg.get("text_config", {}).pop(key, None)
    json.dump(cfg, open(cfg_path, "w"), indent=2)

    if base is not None:
        from transformers import AutoProcessor, GenerationConfig

        AutoProcessor.from_pretrained(base).save_pretrained(out_dir)
        try:
            GenerationConfig.from_pretrained(base).save_pretrained(out_dir)
        except OSError:
            pass
    return out_dir
