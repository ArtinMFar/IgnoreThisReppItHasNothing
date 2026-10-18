"""Builds the Hugging Face model card from training metadata and evaluation results."""

import json


def _scaling_table(teacher, student):
    rows = ["| Context so far (tokens) | Cache: Qwen3.5-4B | Cache: RecurrentQwen-4b | "
            "Next-turn TTFT (cached): Qwen3.5-4B | Next-turn TTFT (cached): RecurrentQwen-4b | "
            "Decode ms/token: Qwen3.5-4B | Decode ms/token: RecurrentQwen-4b | "
            "Cold prefill s: Qwen3.5-4B | Cold prefill s: RecurrentQwen-4b |",
            "|---|---|---|---|---|---|---|---|---|"]
    for t, s in zip(teacher, student):
        rows.append(f"| {t['context']:,} | {t['cache_mb']:.0f} MB | {s['cache_mb']:.0f} MB | "
                    f"{1000 * t['next_turn_ttft_s']:.0f} ms | {1000 * s['next_turn_ttft_s']:.0f} ms | "
                    f"{t['decode_ms_per_token']:.1f} | {s['decode_ms_per_token']:.1f} | "
                    f"{t['prefill_s']:.2f} | {s['prefill_s']:.2f} |")
    return "\n".join(rows)


def _recall_table(results):
    keys = sorted({k for r in results.values() for k in r}, key=lambda k: int(k.split("_")[0]))
    rows = ["| Model | " + " | ".join(k.replace("_tokens", " tok") for k in keys) + " |",
            "|---|" + "---|" * len(keys)]
    for name, r in results.items():
        rows.append(f"| {name} | " + " | ".join(f"{100 * r.get(k, 0):.0f}%" for k in keys) + " |")
    return "\n".join(rows)


def _nll_table(results):
    keys = list(next(iter(results.values())).keys())
    rows = ["| Model | " + " | ".join(k.replace("nll_", "pos ") for k in keys) + " |", "|---|" + "---|" * len(keys)]
    for name, r in results.items():
        rows.append(f"| {name} | " + " | ".join(f"{r[k]:.3f}" for k in keys) + " |")
    return "\n".join(rows)


def build_card(teacher_eval, fresh_eval, student_eval, train_info):
    recall = {"Qwen3.5-4B (original, full attention)": teacher_eval["memory_recall"],
              "Converted, before training": fresh_eval["memory_recall"],
              "**RecurrentQwen-4b**": student_eval["memory_recall"]}
    nll = {"Qwen3.5-4B (original)": teacher_eval["long_nll"],
           "Converted, before training": fresh_eval["long_nll"],
           "**RecurrentQwen-4b**": student_eval["long_nll"]}
    gate = student_eval.get("gate", {})
    return f"""---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
library_name: transformers
pipeline_tag: image-text-to-text
tags:
- recurrent
- hybrid
- linear-attention
- gated-deltanet
- sliding-window-attention
- mamba3
- distillation
- vision-language
---

# RecurrentQwen-4b

**RecurrentQwen-4b** is [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) converted into a
[RecurrentGemma](https://arxiv.org/abs/2404.07839)-style model: **local attention over a fixed 2,048-token window +
recurrent layers with a fixed-size state**. Once the context is longer than 2,048 tokens, the cache stops growing
and the cost of generating each new token stays flat, no matter how long the conversation gets.
It keeps Qwen3.5's vision encoder, so it still understands images.

## What changed vs. Qwen3.5-4B

Qwen3.5-4B is already a hybrid: 24 Gated DeltaNet (linear-attention / recurrent) layers and 8 full-attention
layers (every 4th layer). RecurrentQwen-4b changes:

1. **Full attention → 2,048-token sliding-window attention** in the 8 attention layers (like RecurrentGemma's
   local attention). The KV cache is capped at 2,047 tokens per attention layer, so total cache memory is constant
   after 2k tokens. Everything older than the window must be carried by the recurrent layers.
2. **Priority gate (selective skip), inspired by Mamba's selectivity.** Every recurrent layer gets a per-head,
   input-dependent gate `keep = 1 - sigmoid(w·x + b)`. It scales the step size `dt` (which controls both decay and
   the rotation below) and the write strength `beta` of the delta rule. When the gate closes on a token, the state is
   **neither decayed nor written**: the token is invisible to long-term memory. During training, half of the heads
   ("memory heads") of every recurrent layer after the first attention layer were explicitly taught to close the gate
   on repetitive, low-information content (game-NPC movement commands, game ticks, telemetry, heartbeat logs, chat spam,
   progress bars, price ticks...). So floods of repetitive output don't wash out real conversation, while the other
   heads are free to track it. Recent repetitive content is still visible through the 2k attention window.
   Measured on held-out samples, memory heads skip {100 * gate.get('skip_prob_filler', 0):.0f}% of the time on repetitive
   tokens and {100 * gate.get('skip_prob_other', 0):.0f}% of the time on other tokens.
3. **Data-dependent state rotation (Mamba-3's complex-valued state).** [Mamba-3](https://arxiv.org/abs/2603.15569)
   shows that a complex-valued SSM equals a real one whose B/C projections are rotated by a cumulative,
   data-dependent angle (the "RoPE trick"), which gives the state rotational dynamics and improves state tracking.
   We rotate 16 channel pairs of every key/query head by `phase_t = Σ dt_s · θ(x_s)`; the phase is carried in the
   cache during generation.
4. Kept from Gated DeltaNet: the **delta rule** (only the part of a new value that the state does not already predict
   gets written, which is the same "store the most useful information" idea Mamba-3 emphasises), gated decay, and
   the short convolution. Mamba-3's exponential-trapezoidal discretization amounts to an implicit convolution on
   the input, a role the existing short convolution already fills here. Mamba-3's MIMO variant (more FLOPs per
   decode step) was not applied: it changes the state layout and would need pretraining.

All new parameters are initialised to (almost exactly) reproduce the original model, then the whole language model
is trained.

## Training

- **Full-parameter distillation** from the original Qwen3.5-4B (teacher: full attention, frozen) using its token
  probabilities: forward KL(teacher ‖ student) on every position, plus cross-entropy on ground-truth answers of
  synthetic memory and multi-document QA tasks, plus the priority-gate auxiliary loss.
- All language-model weights trained (no LoRA). The vision encoder was kept frozen and identical to the teacher's,
  which preserves vision; image–text data was included so the language model keeps using it.
- Data ({train_info['tokens_m']:.0f}M tokens, sequences up to 16k tokens):
  [smoltalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) SFT (thinking and non-thinking chats,
  reasoning, multilingual, tool calls, long-context LongAlign),
  [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) long documents,
  [PG19](https://huggingface.co/datasets/emozilla/pg19) books, multi-document QA built from
  [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa), synthetic long-memory tasks with repetitive filler,
  and 21 high-quality subsets of [FineVision](https://huggingface.co/datasets/HuggingFaceM4/FineVision) (images).
- Hardware: {train_info['hardware']}, {train_info['steps']} optimizer steps, AdamW (lr {train_info['lr']} for
  pretrained weights, {train_info['new_lr']} for new gate weights), bf16 autocast with fp32 master weights.

## Results

### Memory and speed vs. context length (1x B200, bf16, batch 1, prompt prefilled in 4k chunks)

{_scaling_table(teacher_eval['scaling'], student_eval['scaling'])}

- **Cache memory and per-token decode cost** are flat beyond 2k tokens.
- **Next-turn TTFT (cached)**: in a chat or game loop, keep the cache (`past_key_values`) between turns. A new 512-token
  message then only touches the fixed-size recurrent state and the last 2,047 tokens of keys/values, so its time to
  first token does not grow with the length of the conversation.
- **Cold prefill** (re-reading a whole long prompt from scratch, e.g. a new conversation that starts with a 100k-token
  document) has to look at every token once, so it grows linearly with prompt length. There's no quadratic attention
  term.

### Recall of facts buried under repetitive content (held-out synthetic tasks, greedy, exact match)

{_recall_table(recall)}

### Long-text modeling: next-token NLL by position (PG19 test books, 32k tokens, lower is better)

{_nll_table(nll)}

### Vision (ChartQA test subset, relaxed accuracy)

| Qwen3.5-4B | Converted, before training | RecurrentQwen-4b |
|---|---|---|
| {100 * teacher_eval['chartqa_relaxed_acc']:.0f}% | {100 * fresh_eval['chartqa_relaxed_acc']:.0f}% | {100 * student_eval['chartqa_relaxed_acc']:.0f}% |

## Usage

Needs `transformers>=5.20` and `trust_remote_code=True`. Install
[flash-linear-attention](https://github.com/fla-org/flash-linear-attention) for fast recurrent kernels.

```python
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

repo = "ArtinMFar/RecurrentQwen-4b"
processor = AutoProcessor.from_pretrained(repo)
model = AutoModelForImageTextToText.from_pretrained(repo, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda")

messages = [{{"role": "user", "content": [
    {{"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"}},
    {{"type": "text", "text": "Describe this image."}},
]}}]
inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                       return_tensors="pt", enable_thinking=False).to(model.device)
out = model.generate(**inputs, max_new_tokens=256)
print(processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

**Keep the cache between turns** (chat apps, game NPCs): pass the previous `past_key_values` back in with only the new
message, so each turn costs the same no matter how long the conversation is. The default `DynamicCache` built from this
config keeps only the last 2,047 keys/values in the attention layers plus the fixed-size recurrent states (~64 MB total
in bf16, batch 1):

```python
import torch
from transformers import DynamicCache

tok = processor.tokenizer
im_end = tok.convert_tokens_to_ids("<|im_end|>")
cache = DynamicCache(config=model.config.get_text_config())
ids = tok("<|im_start|>system\nYou are Garrick, the village blacksmith.<|im_end|>", return_tensors="pt").input_ids
ids = ids.to(model.device)

for user_msg in ["Hi! My name is Lyra.", "[tick 1] player_pos=(3, 4)", "What's my name?"]:
    turn = f"\n<|im_start|>user\n{{user_msg}}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    ids = torch.cat([ids, tok(turn, return_tensors="pt").input_ids.to(model.device)], dim=1)
    out = model.generate(input_ids=ids, past_key_values=cache, max_new_tokens=128, do_sample=False)
    print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
    ids = out if out[0, -1] == im_end else torch.cat([out, out.new_tensor([[im_end]])], dim=1)
```

Append each new turn to the running token stream (don't re-render the whole chat template: it rewrites earlier
turns, and a recurrent / sliding-window cache can't be rolled back). `generate` only runs the tokens that are not in
the cache yet. For very long one-shot prompts, prefill in chunks of e.g. 4,096 tokens with a shared cache to keep peak
memory flat as well.

## Limitations

- Custom architecture: runs with 🤗 Transformers (`trust_remote_code`). vLLM / SGLang / llama.cpp don't support it.
- Information older than 2,048 tokens is only available through the fixed-size recurrent state. That state is
  lossy, so exact long-range copying (long verbatim quotes, many distinct facts) is weaker than with full attention.
- Qwen3.5's multi-token-prediction (MTP) head was not carried over.
- Beam search does not reorder the rotation phase in the cache; use greedy or sampling.
- Prefill time still grows linearly with prompt length.

## Training code

https://github.com/ArtinMFar/IgnoreThisReppItHasNothing (branch `claude/recurrent-qwen-4b-training-q2ta43`)

<details><summary>Raw evaluation results</summary>

```json
{json.dumps({"teacher": teacher_eval, "converted_before_training": fresh_eval, "recurrentqwen": student_eval}, indent=1)}
```
</details>
"""
