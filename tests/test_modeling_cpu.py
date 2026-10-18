"""CPU sanity tests for RecurrentQwen on a tiny random config (no downloads needed)."""

import torch

from transformers import DynamicCache
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

from recurrent_qwen.configuration_recurrent_qwen import RecurrentQwenConfig
from recurrent_qwen.modeling_recurrent_qwen import RecurrentQwenForConditionalGeneration

TEXT = dict(
    hidden_size=64, num_hidden_layers=8, intermediate_size=128, num_attention_heads=4, num_key_value_heads=2,
    head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
    vocab_size=512, rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                                     "mrope_section": [1, 1, 0], "mrope_interleaved": True},
)
VISION = dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64, patch_size=16)


def build(window, gate_bias=-5.0, rot_pairs=4, seed=0):
    torch.manual_seed(seed)
    base = Qwen3_5ForConditionalGeneration(Qwen3_5Config(text_config=TEXT, vision_config=VISION)).eval()
    cfg = RecurrentQwenConfig(
        text_config=dict(TEXT, sliding_window=window, priority_gate_bias_init=gate_bias, state_rotation_pairs=rot_pairs),
        vision_config=VISION,
    )
    model = RecurrentQwenForConditionalGeneration(cfg).eval()
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected, unexpected
    assert all(("in_proj_skip" in k) or ("in_proj_theta" in k) for k in missing), missing
    for layer in model.model.language_model.layers:
        if layer.block_type == "linear_attention":
            torch.nn.init.zeros_(layer.linear_attn.in_proj_skip.weight)
            torch.nn.init.constant_(layer.linear_attn.in_proj_skip.bias, gate_bias)
            torch.nn.init.zeros_(layer.linear_attn.in_proj_theta.weight)
    return base, model


def test_identity_at_init():
    base, model = build(window=64, gate_bias=-60.0)
    ids = torch.randint(0, 500, (2, 40))
    with torch.no_grad():
        a = base(input_ids=ids).logits
        b = model(input_ids=ids).logits
    err = (a - b).abs().max().item()
    print("identity (seq<window, gate closed, no rotation) max err:", err)
    assert err < 1e-4


def test_window_changes_long_context():
    base, model = build(window=8, gate_bias=-60.0)
    ids = torch.randint(0, 500, (1, 40))
    with torch.no_grad():
        a = base(input_ids=ids).logits
        b = model(input_ids=ids).logits
    early = (a[:, :8] - b[:, :8]).abs().max().item()
    late = (a[:, 20:] - b[:, 20:]).abs().max().item()
    print(f"window=8: first-8 err {early:.2e}, late err {late:.2e}")
    assert early < 1e-4 and late > 1e-4


def test_incremental_decoding_matches_full_forward():
    _, model = build(window=8)
    torch.manual_seed(1)
    with torch.no_grad():  # make the new parameters non-trivial
        for layer in model.model.language_model.layers:
            if layer.block_type == "linear_attention":
                layer.linear_attn.in_proj_skip.weight.normal_(0, 0.5)
                layer.linear_attn.in_proj_skip.bias.fill_(0.0)
                layer.linear_attn.in_proj_theta.weight.normal_(0, 0.5)
    ids = torch.randint(0, 500, (2, 30))
    with torch.no_grad():
        full = model(input_ids=ids).logits
        cache = DynamicCache(config=model.config.get_text_config())
        out = model(input_ids=ids[:, :13], past_key_values=cache, use_cache=True)
        step_logits = [out.logits]
        for t in range(13, 30):
            out = model(input_ids=ids[:, t : t + 1], past_key_values=cache, use_cache=True)
            step_logits.append(out.logits)
        inc = torch.cat(step_logits, dim=1)
    err = (full - inc).abs().max().item()
    print("incremental vs full max err:", err)
    assert err < 1e-3
    kv = [layer.keys.shape[-2] for layer in cache.layers if hasattr(layer, "keys") and layer.keys is not None]
    print("cached KV length per sliding layer after 30 tokens (window 8):", kv)
    assert all(k <= 8 for k in kv)


def test_generate_and_skip_logits():
    _, model = build(window=8)
    for layer in model.model.language_model.layers:
        if layer.block_type == "linear_attention":
            layer.linear_attn.record_skip_logits = True
    ids = torch.randint(0, 500, (1, 12))
    out = model.generate(input_ids=ids, max_new_tokens=20, do_sample=False)
    print("generated shape:", tuple(out.shape))
    logits = model.model.language_model.layers[0].linear_attn.last_skip_logits
    assert logits is not None and logits.shape[-1] == 4


def test_save_load_roundtrip(tmp="/tmp/rq_tiny"):
    _, model = build(window=8)
    model.save_pretrained(tmp)
    reloaded = RecurrentQwenForConditionalGeneration.from_pretrained(tmp).eval()
    ids = torch.randint(0, 500, (1, 20))
    with torch.no_grad():
        err = (model(input_ids=ids).logits - reloaded(input_ids=ids).logits).abs().max().item()
    print("save/load max err:", err, "| layer types:", reloaded.config.text_config.layer_types[:4])
    assert err < 1e-5



def test_multiturn_cache_reuse_matches_recompute():
    """Keeping the cache between chat turns gives the same greedy outputs as re-reading the whole history."""
    _, model = build(window=8)
    torch.manual_seed(2)
    with torch.no_grad():
        for layer in model.model.language_model.layers:
            if layer.block_type == "linear_attention":
                layer.linear_attn.in_proj_skip.weight.normal_(0, 0.3)
                layer.linear_attn.in_proj_theta.weight.normal_(0, 0.3)
    turns = [torch.randint(0, 500, (1, n)) for n in (11, 7, 9)]
    cache = DynamicCache(config=model.config.get_text_config())
    ids, cached_replies, fresh_replies = None, [], []
    for turn in turns:
        ids = turn if ids is None else torch.cat([ids, turn], dim=1)
        out = model.generate(input_ids=ids, past_key_values=cache, max_new_tokens=6, do_sample=False)
        cached_replies.append(out[:, ids.shape[1]:])
        fresh = model.generate(input_ids=ids, max_new_tokens=6, do_sample=False)
        fresh_replies.append(fresh[:, ids.shape[1]:])
        ids = out
    print("cached:", [r.tolist() for r in cached_replies])
    print("fresh: ", [r.tolist() for r in fresh_replies])
    assert all(torch.equal(a, b) for a, b in zip(cached_replies, fresh_replies))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
