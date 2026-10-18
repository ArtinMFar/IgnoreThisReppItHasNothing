"""Evaluation of RecurrentQwen against the original Qwen3.5-4B.

  * scaling: KV/state cache bytes, prefill time and per-token decode latency vs. context length
  * memory recall: held-out synthetic "facts buried under repetitive content" tasks (exact match)
  * long-text NLL by position bucket on held-out PG19 test books
  * vision: ChartQA relaxed accuracy on a test subset
  * priority gate: mean skip probability of memory heads on filler vs. non-filler tokens
"""

import argparse
import json
import random
import re
import time

import torch
import torch.nn.functional as F

from rq_data.formatting import FLAG_FILLER, chat_segments, encode_segments
from rq_data.synthetic_memory import generate_sample


def cache_bytes(cache):
    total = 0
    for layer in cache.layers:
        for name in ("keys", "values"):
            t = getattr(layer, name, None)
            if isinstance(t, torch.Tensor):
                total += t.numel() * t.element_size()
        for name in ("conv_states", "recurrent_states"):
            for t in getattr(layer, name, None) or []:
                if isinstance(t, torch.Tensor):
                    total += t.numel() * t.element_size()
        phase = getattr(layer, "rotation_phase", None)
        if isinstance(phase, torch.Tensor):
            total += phase.numel() * phase.element_size()
    return total


@torch.no_grad()
def prefill(model, ids, cache, chunk=4096):
    out = None
    for s in range(0, ids.shape[1], chunk):
        out = model(input_ids=ids[:, s : s + chunk], past_key_values=cache, use_cache=True, logits_to_keep=1)
    return out


@torch.no_grad()
def scaling(model, lengths, device, decode_steps=32):
    from transformers import DynamicCache

    results = []
    for L in lengths:
        ids = torch.randint(1000, 100000, (1, L), device=device)
        cache = DynamicCache(config=model.config.get_text_config())
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        out = prefill(model, ids, cache)
        torch.cuda.synchronize()
        t_prefill = time.time() - t0
        nxt = out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(decode_steps):
            out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
            nxt = out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()
        results.append({
            "context": L, "cache_mb": cache_bytes(cache) / 2**20, "prefill_s": t_prefill,
            "decode_ms_per_token": (time.time() - t0) / decode_steps * 1000,
            "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
        })
        print(json.dumps(results[-1]), flush=True)
        del cache, out
        torch.cuda.empty_cache()
    return results


def _norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


@torch.no_grad()
def memory_recall(model, tokenizer, device, n=60, seed=987654):
    """Generates held-out synthetic tasks; a sample is correct if every key value of the reference is produced."""
    from transformers import DynamicCache

    rng = random.Random(seed)
    buckets = {}
    for i in range(n):
        target = [3000, 6000, 12000, 24000][i % 4]
        ids, flags = generate_sample(tokenizer, rng, target)
        text = tokenizer.decode(ids)
        prompt_text, answer = text.rsplit("<think>\n\n</think>\n\n", 1)
        answer = answer.replace("<|im_end|>", "").strip()
        prompt = tokenizer(prompt_text + "<think>\n\n</think>\n\n", add_special_tokens=False,
                           return_tensors="pt")["input_ids"].to(device)
        cache = DynamicCache(config=model.config.get_text_config())
        out = prefill(model, prompt, cache)
        nxt = out.logits[:, -1:].argmax(-1)
        gen = [nxt.item()]
        for _ in range(48):
            out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
            nxt = out.logits[:, -1:].argmax(-1)
            gen.append(nxt.item())
            if gen[-1] in (tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id):
                break
        pred = tokenizer.decode(gen, skip_special_tokens=True)
        keys = re.findall(r"(?:is now|is|are|named|under the|on|bring|name is|allergic to) ([^,.]+)", answer)
        ok = all(_norm(k) in _norm(pred) for k in keys) if keys else _norm(answer) in _norm(pred)
        b = buckets.setdefault(target, [0, 0])
        b[0] += int(ok)
        b[1] += 1
        if i < 4:
            print(f"[{prompt.shape[1]} tok] ref: {answer!r} | pred: {pred!r} | ok={ok}", flush=True)
    return {f"{k}_tokens": v[0] / v[1] for k, v in sorted(buckets.items())}


@torch.no_grad()
def long_nll(model, tokenizer, device, n_books=6, length=32768):
    from datasets import load_dataset

    from transformers import DynamicCache

    edges = [0, 2048, 8192, 16384, length]
    sums = [0.0] * (len(edges) - 1)
    counts = [0] * (len(edges) - 1)
    ds = load_dataset("emozilla/pg19", split="test", streaming=True)
    used = 0
    for row in ds:
        ids = tokenizer(row["text"][:200_000], add_special_tokens=False)["input_ids"]
        if len(ids) < length + 1:
            continue
        ids = torch.tensor(ids[: length + 1], device=device)[None]
        cache = DynamicCache(config=model.config.get_text_config())
        pos = 0
        for s in range(0, length, 4096):
            chunk = ids[:, s : s + 4096]
            out = model(input_ids=chunk, past_key_values=cache, use_cache=True)
            nll = F.cross_entropy(out.logits[0].float(), ids[0, s + 1 : s + 1 + chunk.shape[1]], reduction="none")
            for j, v in enumerate(nll.tolist()):
                p = pos + j
                for b in range(len(edges) - 1):
                    if edges[b] <= p < edges[b + 1]:
                        sums[b] += v
                        counts[b] += 1
            pos += chunk.shape[1]
        used += 1
        if used >= n_books:
            break
    return {f"nll_{edges[b]}-{edges[b + 1]}": sums[b] / max(counts[b], 1) for b in range(len(edges) - 1)}


@torch.no_grad()
def chartqa(model, processor, device, n=100):
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceM4/ChartQA", split="test", streaming=True)
    correct = total = 0
    for row in ds:
        messages = [{"role": "user", "content": [{"type": "image"},
                                                  {"type": "text", "text": row["query"] + "\nAnswer with a single word or number."}]}]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
        inputs = processor(text=[text], images=[row["image"]], return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        pred = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        label = row["label"][0] if isinstance(row["label"], list) else row["label"]
        try:
            ok = abs(float(pred.rstrip("%")) - float(label.rstrip("%"))) <= 0.05 * abs(float(label.rstrip("%"))) + 1e-9
        except ValueError:
            ok = _norm(pred) == _norm(label)
        correct += int(ok)
        total += 1
        if total <= 3:
            print(f"ChartQA q={row['query']!r} label={label!r} pred={pred!r}", flush=True)
        if total >= n:
            break
    return {"chartqa_relaxed_acc": correct / total}


@torch.no_grad()
def gate_stats(model, tokenizer, device, n=8, seed=4242):
    layers = [l for l in model.model.language_model.layers if l.block_type == "linear_attention"]
    for l in layers:
        l.linear_attn.record_skip_logits = True
    rng = random.Random(seed)
    filler_p, other_p = [], []
    for _ in range(n):
        ids, flags = generate_sample(tokenizer, rng, 6000)
        flags = torch.tensor(flags, device=device)
        model.model(input_ids=torch.tensor(ids, device=device)[None], use_cache=False)
        probs = torch.stack([torch.sigmoid(l.linear_attn.last_skip_logits[0, :, 1::2]) for l in layers[3:]])
        is_filler = (flags & FLAG_FILLER) > 0
        filler_p.append(probs[:, is_filler].mean().item())
        other_p.append(probs[:, ~is_filler].mean().item())
    for l in layers:
        l.linear_attn.record_skip_logits = False
        l.linear_attn.last_skip_logits = None
    return {"skip_prob_filler": sum(filler_p) / n, "skip_prob_other": sum(other_p) / n}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="path or 'teacher'")
    p.add_argument("--base", default="Qwen/Qwen3.5-4B")
    p.add_argument("--out", default=None)
    p.add_argument("--skip", default="")
    args = p.parse_args()
    device = torch.device("cuda")
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    from recurrent_qwen.modeling_recurrent_qwen import RecurrentQwenForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.base)
    tok = processor.tokenizer
    if args.model == "teacher":
        model = Qwen3_5ForConditionalGeneration.from_pretrained(args.base, dtype=torch.bfloat16)
    else:
        model = RecurrentQwenForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16,
                                                                      attn_implementation="sdpa")
    model.to(device).eval()
    res = {"model": args.model}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if "scaling" not in args.skip:
            res["scaling"] = scaling(model, [1024, 4096, 16384, 65536, 131072], device)
        if "recall" not in args.skip:
            res["memory_recall"] = memory_recall(model, tok, device)
            print(json.dumps(res["memory_recall"]), flush=True)
        if "nll" not in args.skip:
            res["long_nll"] = long_nll(model, tok, device)
            print(json.dumps(res["long_nll"]), flush=True)
        if "chartqa" not in args.skip:
            res.update(chartqa(model, processor, device))
            print("chartqa", res["chartqa_relaxed_acc"], flush=True)
        if "gate" not in args.skip and args.model != "teacher":
            res["gate"] = gate_stats(model, tok, device)
            print(json.dumps(res["gate"]), flush=True)
    print(json.dumps(res, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
