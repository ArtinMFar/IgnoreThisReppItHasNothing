"""Peak memory of a student training forward/backward (flex sliding attention) for different padded lengths."""

import time

import torch

from rq_train.train import build_student, register_flex_attention


def main(base="Qwen/Qwen3.5-4B", lengths=(8192, 9216, 10240, 11264, 12288, 13312, 14336, 15360, 16384)):
    register_flex_attention()
    dev = torch.device("cuda")
    student, _ = build_student(base, 2048, "rq_flex", None)
    student.to(dev).train()
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    mask = {"sliding_attention": None, "linear_attention": None}
    for T in lengths:
        ids = torch.randint(0, 200000, (1, T), device=dev)
        for it in range(2):
            torch.cuda.synchronize()
            base_mem = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = student.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
                loss = h.float().pow(2).mean()
            loss.backward()
            torch.cuda.synchronize()
            student.zero_grad(set_to_none=True)
            del h, loss
        print(f"T={T:6d} activations peak {(torch.cuda.max_memory_allocated() - base_mem) / 1e9:6.1f} GB "
              f"time {time.time() - t0:.2f}s", flush=True)
