# coding=utf-8
"""RecurrentQwen configuration.

RecurrentQwen is Qwen3.5 converted into a RecurrentGemma-style model:

* The 8 full-attention layers become local (sliding-window) attention with a fixed
  2048-token window, so the KV cache stops growing after 2048 tokens.
* The 24 Gated-DeltaNet (linear-attention / recurrent) layers keep a fixed-size state
  and carry everything older than the window. They get two Mamba-3-inspired upgrades:
    - a *priority gate* (selective skip): for low-value / repetitive tokens the layer
      neither decays nor writes its state, so floods of repetitive content (e.g. NPC
      movement commands) cannot wash out important information.
    - a *data-dependent state rotation* (Mamba-3's complex-valued state, implemented
      with the "RoPE trick"): keys/queries are rotated by a cumulative, input-dependent
      angle, which gives the recurrent state rotational dynamics for state tracking.
"""

from huggingface_hub.dataclasses import strict

from transformers.configuration_utils import SubConfigSpec
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)


@strict
class RecurrentQwenTextConfig(Qwen3_5TextConfig):
    r"""
    sliding_window (`int`, *optional*, defaults to 2048):
        Attention window of the (former full-attention) local attention layers. `None` keeps full attention.
    priority_gate (`bool`, *optional*, defaults to `True`):
        Adds a per-head, input-dependent skip gate to every recurrent layer. When the gate closes on a token, the
        recurrent state is neither decayed nor written for that token.
    priority_gate_bias_init (`float`, *optional*, defaults to -5.0):
        Initial bias of the skip gate logits (sigmoid(-5) ~ 0.7% skip), so the converted model starts very close
        to the original one.
    state_rotation_pairs (`int`, *optional*, defaults to 16):
        Number of key-dimension pairs (per key head) rotated by the data-dependent, cumulative angle. 0 disables it.
    """

    model_type = "recurrent_qwen_text"

    sliding_window: int | None = 2048
    priority_gate: bool = True
    priority_gate_bias_init: float = -5.0
    state_rotation_pairs: int = 16

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)
        if self.sliding_window is not None:
            self.layer_types = [
                "sliding_attention" if layer_type == "full_attention" else layer_type
                for layer_type in self.layer_types
            ]


@strict
class RecurrentQwenConfig(Qwen3_5Config):
    model_type = "recurrent_qwen"
    sub_configs_defaults = {
        "vision_config": SubConfigSpec(config_class=Qwen3_5VisionConfig),
        "text_config": SubConfigSpec(config_class=RecurrentQwenTextConfig),
    }


__all__ = ["RecurrentQwenConfig", "RecurrentQwenTextConfig"]
