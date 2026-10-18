"""GPU sanity checks and attention benchmark before a training run.

1. Kernel check: the Gated DeltaNet ops resolve to flash-linear-attention, not the slow PyTorch fallback.
2. Conversion check: KL(teacher || fresh student) is ~0 inside the window and grows past 2048 tokens.
3. Sliding-window attention: flex-attention output matches SDPA with an explicit mask; timing and memory of a
   training forward/backward at 8k and 16k tokens for both.
"""

import glob
import time

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from transformers import Qwen3_5ForConditionalGeneration
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

from rq_train.train import build_student, register_flex_attention


def kl_per_position(t_logits, s_logits):
    t = F.log_softmax(t_logits.float(), -1)
    s = F.log_softmax(s_logits.float(), -1)
    return (t.exp() * (t - s)).sum(-1)


@torch.no_grad()
def logits_chunked(model, ids, chunk=2048, **kw):
    h = model.model(input_ids=ids, use_cache=False, **kw).last_hidden_state[0]
    return torch.cat([model.lm_head(h[i : i + chunk]).float().cpu() for i in range(0, h.shape[0], chunk)])


def teacher_memory(base="Qwen/Qwen3.5-4B"):
    """Peak memory / time of a no-grad teacher forward at 16k tokens with SDPA vs. flex attention."""
    from rq_train.train import attn_config

    register_flex_attention()
    dev = torch.device("cuda")
    ids = torch.randint(0, 200000, (1, 16384), device=dev)
    for impl in ("sdpa", "rq_flex"):
        teacher = Qwen3_5ForConditionalGeneration.from_pretrained(base, dtype=torch.bfloat16,
                                                                  attn_implementation=attn_config(impl)).to(dev).eval()
        mask = {"full_attention": None, "linear_attention": None} if impl == "rq_flex" else None
        outs = []
        for _ in range(2):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base_mem = torch.cuda.memory_allocated()
            t0 = time.time()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                h = teacher.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            torch.cuda.synchronize()
            outs.append(h.float())
        print(f"teacher {impl:8s} 16k fwd {time.time() - t0:.2f}s, transient peak "
              f"{(torch.cuda.max_memory_allocated() - base_mem) / 1e9:.1f} GB", flush=True)
        if impl == "sdpa":
            ref = outs[-1]
        else:
            print("teacher flex vs sdpa hidden max diff:", (outs[-1] - ref).abs().max().item(),
                  "rel:", ((outs[-1] - ref).norm() / ref.norm()).item(), flush=True)
        del teacher, h, outs
        torch.cuda.empty_cache()


def main(base="Qwen/Qwen3.5-4B", data="/data/rows"):
    dev = torch.device("cuda")
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
    for fn in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn"):
        impl = getattr(qwen, fn)
        print(f"{fn} -> {getattr(impl, '__wrapped__', impl).__module__}")
    register_flex_attention()

    files = sorted(glob.glob(f"{data}/fineweb-edu-long/part-*.parquet")) + sorted(glob.glob(f"{data}/pg19/part-*.parquet"))
    rows = pq.read_table(files[-1]).to_pylist()
    long_ids = max(rows, key=lambda r: r["n_tokens"])["input_ids"]
    print("sample row tokens:", len(long_ids))

    teacher = Qwen3_5ForConditionalGeneration.from_pretrained(base, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev).eval()
    student, _ = build_student(base, 2048, "rq_flex", None)
    student.to(dev, torch.bfloat16).eval()

    T = min(8192, len(long_ids))
    ids = torch.tensor(long_ids[:T], device=dev)[None]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        t_logits = logits_chunked(teacher, ids)
        s_logits = logits_chunked(student, ids, attention_mask={"sliding_attention": None, "linear_attention": None})
    kl = kl_per_position(t_logits, s_logits)
    for a, b in ((0, 2048), (2048, 4096), (4096, T)):
        print(f"fresh student KL vs teacher, positions [{a}, {b}): {kl[a:b].mean():.4f}")
    nll_t = F.cross_entropy(t_logits[:-1], ids[0, 1:].cpu())
    nll_s = F.cross_entropy(s_logits[:-1], ids[0, 1:].cpu())
    print(f"NLL teacher {nll_t:.4f} | fresh student {nll_s:.4f}")

    student.config._attn_implementation = "sdpa"
    for layer in student.model.language_model.layers:
        if layer.block_type != "linear_attention":
            layer.self_attn.config._attn_implementation = "sdpa"
    with torch.autocast("cuda", dtype=torch.bfloat16):
        s_logits_sdpa = logits_chunked(student, ids)
    print("flex vs sdpa max |dlogit|:", (s_logits - s_logits_sdpa).abs().max().item(),
          "mean:", (s_logits - s_logits_sdpa).abs().mean().item())
    del teacher, t_logits, s_logits, s_logits_sdpa
    torch.cuda.empty_cache()

    student.to(torch.float32).train()
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    for impl in ("rq_flex", "sdpa"):
        student.config._attn_implementation = impl
        for layer in student.model.language_model.layers:
            if layer.block_type != "linear_attention":
                layer.self_attn.config._attn_implementation = impl
        for T in (8192, 16384):
            ids = torch.randint(0, 200000, (1, T), device=dev)
            mask = {"sliding_attention": None, "linear_attention": None} if impl == "rq_flex" else None
            times = []
            torch.cuda.reset_peak_memory_stats()
            for it in range(3):
                torch.cuda.synchronize()
                t0 = time.time()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    h = student.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
                    loss = h.float().pow(2).mean()
                loss.backward()
                torch.cuda.synchronize()
                times.append(time.time() - t0)
            g = student.model.language_model.layers[0].linear_attn.in_proj_theta.weight.grad
            print(f"{impl:8s} T={T:6d} fwd+bwd {min(times):.2f}s ({T / min(times):.0f} tok/s, model only) "
                  f"peak mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB | theta grad norm {g.norm():.3e}")
            student.zero_grad(set_to_none=True)


if __name__ == "__main__":
    main()
