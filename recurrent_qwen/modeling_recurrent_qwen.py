# coding=utf-8
"""RecurrentQwen: Qwen3.5 converted to a RecurrentGemma-style local-attention + recurrent model.

See `configuration_recurrent_qwen.py` for a description of the architecture changes. All new parameters are
initialised so that the converted model starts (almost exactly) where the original Qwen3.5 model was.
"""

import torch
import torch.nn.functional as F
from torch import nn

from transformers import initialization as init
from transformers.cache_utils import Cache, DynamicCache
from transformers.integrations.accelerate import force_accelerate_hooks
from transformers.masking_utils import (
    create_causal_mask,
    create_recurrent_attention_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from .configuration_recurrent_qwen import RecurrentQwenConfig, RecurrentQwenTextConfig


def rotate_pairs(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotates the first `2 * P` channels of `x` ([..., D]) as P (x_i, x_{P+i}) pairs; cos/sin are [..., P]."""
    num_pairs = cos.shape[-1]
    x1 = x[..., :num_pairs].float()
    x2 = x[..., num_pairs : 2 * num_pairs].float()
    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(x.dtype)
    return torch.cat([rotated, x[..., 2 * num_pairs :]], dim=-1)


class RecurrentQwenGatedDeltaNet(qwen.Qwen3_5GatedDeltaNet):
    """Gated DeltaNet with a priority (skip) gate and a Mamba-3 style data-dependent state rotation.

    Priority gate: keep_t = 1 - sigmoid(w_skip . x_t + b_skip) per value head. The step size dt_t (which sets both the
    decay exp(-A dt_t) and the rotation angle) and the write strength beta_t are multiplied by keep_t. A token with
    keep_t ~ 0 is invisible to the recurrent state: nothing is forgotten and nothing is written. This is the
    "Delta -> 0 ignores the input" selectivity of Mamba made explicit, and is what lets the model give repetitive,
    low-information tokens low priority.

    State rotation: Mamba-3 shows that a complex-valued SSM equals a real SSM whose B/C (here: keys/queries) are rotated
    by the cumulative sum of data-dependent angles. We rotate `state_rotation_pairs` channel pairs of every key head by
    phase_t = sum_{s<=t} dt_s * theta(x_s), so that reading the state applies the relative rotation accumulated
    between write and read time.
    """

    def __init__(self, config: RecurrentQwenTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.priority_gate = config.priority_gate
        self.priority_gate_bias_init = config.priority_gate_bias_init
        self.rot_pairs = config.state_rotation_pairs
        if self.priority_gate:
            self.in_proj_skip = nn.Linear(self.hidden_size, self.num_v_heads, bias=True)
        if self.rot_pairs:
            self.in_proj_theta = nn.Linear(self.hidden_size, self.num_k_heads * self.rot_pairs, bias=False)
        # Training hook: when True, the skip-gate logits of the last forward are kept in `last_skip_logits`
        self.record_skip_logits = False
        self.last_skip_logits = None

    @force_accelerate_hooks("conv1d")
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Cache | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        hidden_states = qwen.apply_mask_to_padding_states(hidden_states, attention_mask)

        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = cache_params is not None and cache_params.has_previous_state(
            self.layer_idx, state_idx=0
        )
        layer_cache = cache_params.layers[self.layer_idx] if cache_params is not None else None

        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        z = self.in_proj_z(hidden_states)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        if use_precomputed_states and seq_len == 1 and not layer_cache.record_past:
            conv_state = layer_cache.conv_states[0]
            mixed_qkv = qwen.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            if cache_params is not None:
                mixed_qkv = cache_params.update_conv_state(
                    mixed_qkv, self.layer_idx, conv_kernel_size=self.conv_kernel_size
                )

            mixed_qkv = qwen.causal_conv1d_fn(
                mixed_qkv,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                activation=self.activation,
                **kwargs,
            )

            if cache_params is not None:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        dt = F.softplus(a.float() + self.dt_bias)
        if self.priority_gate:
            skip_logits = self.in_proj_skip(hidden_states).float()
            if self.record_skip_logits:
                self.last_skip_logits = skip_logits
            keep = torch.sigmoid(-skip_logits)
            dt = dt * keep
            beta = (beta.float() * keep).to(beta.dtype)
        g = -self.A_log.float().exp() * dt

        if self.rot_pairs:
            heads_per_key = self.num_v_heads // self.num_k_heads
            theta = self.in_proj_theta(hidden_states).float().view(batch_size, seq_len, self.num_k_heads, -1)
            key_dt = dt.view(batch_size, seq_len, self.num_k_heads, heads_per_key).mean(-1, keepdim=True)
            phase = (theta * key_dt).cumsum(dim=1)
            if use_precomputed_states and getattr(layer_cache, "rotation_phase", None) is not None:
                phase = phase + layer_cache.rotation_phase[:, None]
            if cache_params is not None:
                layer_cache.rotation_phase = phase[:, -1].detach()
            cos, sin = phase.cos(), phase.sin()
            query = rotate_pairs(query, cos, sin)
            key = rotate_pairs(key, cos, sin)

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        recurrent_state = layer_cache.recurrent_states[0] if use_precomputed_states else None
        if use_precomputed_states and seq_len == 1:
            core_attn_out, last_recurrent_state = qwen.torch_recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=kwargs.pop("cu_seq_lens_q", None),
                **kwargs,
            )
        else:
            core_attn_out, last_recurrent_state = qwen.torch_chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=kwargs.pop("cu_seq_lens_q", None),
                **kwargs,
            )

        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        return self.out_proj(core_attn_out)


class RecurrentQwenAttention(qwen.Qwen3_5Attention):
    """Qwen3.5 gated attention restricted to a local window of `config.sliding_window` tokens."""

    def __init__(self, config: RecurrentQwenTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if is_sliding else None

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        return super().forward(
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_values,
            sliding_window=self.sliding_window,
            **kwargs,
        )


class RecurrentQwenDecoderLayer(qwen.Qwen3_5DecoderLayer):
    def __init__(self, config: RecurrentQwenTextConfig, layer_idx: int):
        GradientCheckpointingLayer.__init__(self)
        self.hidden_size = config.hidden_size
        self.block_type = config.layer_types[layer_idx]
        if self.block_type == "linear_attention":
            self.linear_attn = RecurrentQwenGatedDeltaNet(config, layer_idx)
        else:
            self.self_attn = RecurrentQwenAttention(config, layer_idx)
        self.mlp = qwen.Qwen3_5MLP(config, config.intermediate_size)
        self.input_layernorm = qwen.Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = qwen.Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.block_type == "linear_attention":
            hidden_states = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                **kwargs,
            )
        else:
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class RecurrentQwenPreTrainedModel(qwen.Qwen3_5PreTrainedModel):
    config: RecurrentQwenConfig
    config_class = RecurrentQwenConfig
    _no_split_modules = ["RecurrentQwenDecoderLayer", "Qwen3_5VisionBlock"]
    _can_record_outputs = {
        "hidden_states": RecurrentQwenDecoderLayer,
        "attentions": RecurrentQwenAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, RecurrentQwenGatedDeltaNet):
            if module.priority_gate:
                init.zeros_(module.in_proj_skip.weight)
                init.constant_(module.in_proj_skip.bias, module.priority_gate_bias_init)
            if module.rot_pairs:
                init.zeros_(module.in_proj_theta.weight)


class RecurrentQwenTextModel(RecurrentQwenPreTrainedModel, qwen.Qwen3_5TextModel):
    config: RecurrentQwenTextConfig
    config_class = RecurrentQwenTextConfig

    def __init__(self, config: RecurrentQwenTextConfig):
        RecurrentQwenPreTrainedModel.__init__(self, config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [RecurrentQwenDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = qwen.Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = qwen.Qwen3_5TextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> qwen.BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }
            mask_builders = {
                "full_attention": create_causal_mask,
                "sliding_attention": create_sliding_window_causal_mask,
                "linear_attention": create_recurrent_attention_mask,
            }
            causal_mask_mapping = {
                layer_type: mask_builders[layer_type](**mask_kwargs) for layer_type in set(self.config.layer_types)
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return qwen.Qwen3_5ModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class RecurrentQwenModel(RecurrentQwenPreTrainedModel, qwen.Qwen3_5Model):
    def __init__(self, config: RecurrentQwenConfig):
        RecurrentQwenPreTrainedModel.__init__(self, config)
        self.visual = qwen.Qwen3_5VisionModel._from_config(config.vision_config)
        self.language_model = RecurrentQwenTextModel._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()


class RecurrentQwenForConditionalGeneration(RecurrentQwenPreTrainedModel, qwen.Qwen3_5ForConditionalGeneration):
    def __init__(self, config: RecurrentQwenConfig):
        RecurrentQwenPreTrainedModel.__init__(self, config)
        self.model = RecurrentQwenModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()


__all__ = [
    "RecurrentQwenForConditionalGeneration",
    "RecurrentQwenModel",
    "RecurrentQwenPreTrainedModel",
    "RecurrentQwenTextModel",
]
