"""Full-parameter distillation of Qwen3.5-4B (teacher, full attention) into RecurrentQwen (student).

Losses (all token-weighted over the global batch):
  * KL(teacher || student) on every next-token distribution (the teacher sees the full context, the student only a
    2048-token attention window plus its recurrent state, so it must learn to carry long-range information).
  * Cross-entropy on ground-truth answer tokens of the synthetic memory / multi-document QA tasks.
  * Priority-gate auxiliary BCE: on synthetic memory rows, half of the recurrent heads ("memory heads") of every
    recurrent layer after the first attention layer are taught to skip repetitive filler tokens.

Launched with torchrun from `modal_train.py`.
"""

import argparse
import contextlib
import json
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from recurrent_qwen.configuration_recurrent_qwen import RecurrentQwenConfig
from recurrent_qwen.modeling_recurrent_qwen import RecurrentQwenForConditionalGeneration
from rq_data.formatting import FLAG_CE, FLAG_FILLER
from rq_train.data import assign_files, list_files, make_loader

NEW_PARAM_KEYS = ("in_proj_skip", "in_proj_theta")


# ---------------------------------------------------------------------------------------------------------------
# Sliding-window attention for training: flex attention only computes the 2048-wide band (O(T * window)).
# ---------------------------------------------------------------------------------------------------------------
_FLEX = {}


def register_flex_attention():
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface

    compiled = torch.compile(flex_attention, dynamic=False)

    def sliding_flex_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0,
                               sliding_window=None, **kwargs):
        q_len, kv_len = query.shape[2], key.shape[2]
        cache_key = (q_len, kv_len, sliding_window, query.device)
        if cache_key not in _FLEX:
            offset = kv_len - q_len

            def mask_mod(b, h, q_idx, kv_idx):
                q_pos = q_idx + offset
                causal = q_pos >= kv_idx
                if sliding_window is None:
                    return causal
                return causal & (q_pos - kv_idx < sliding_window)

            _FLEX[cache_key] = create_block_mask(mask_mod, B=None, H=None, Q_LEN=q_len, KV_LEN=kv_len,
                                                 device=query.device)
        out = compiled(query, key, value, block_mask=_FLEX[cache_key], scale=scaling, enable_gqa=True)
        return out.transpose(1, 2).contiguous(), None

    AttentionInterface.register("rq_flex", sliding_flex_attention)
    AttentionMaskInterface.register("rq_flex", lambda *args, **kwargs: None)


# ---------------------------------------------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------------------------------------------
def _distill_chunk(h, th, w_s, w_t, kl_mask, ce_target, ce_mask):
    s_logp = F.log_softmax(F.linear(h, w_s).float(), dim=-1)
    with torch.no_grad():
        t_logp = F.log_softmax(F.linear(th, w_t).float(), dim=-1)
    kl = (t_logp.exp() * (t_logp - s_logp)).sum(-1)
    ce = -s_logp.gather(-1, ce_target[:, None]).squeeze(-1)
    return (kl * kl_mask).sum(), (ce * ce_mask).sum(), kl.detach()


class DistillStudent(nn.Module):
    """Wraps the student so DDP sees a single forward that returns the loss sums."""

    def __init__(self, student, aux_layers, memory_heads, chunk=1024, window=2048):
        super().__init__()
        self.student = student
        self.aux_layers = aux_layers
        self.register_buffer("memory_heads", torch.tensor(memory_heads), persistent=False)
        self.chunk = chunk
        self.window = window

    def forward(self, input_ids, position_ids, image_embeds, teacher_hidden, w_t, kl_mask, ce_target, ce_mask,
                filler, use_aux, attention_mask=None):
        mm = None
        if image_embeds is not None:
            from transformers.modeling_outputs import BaseModelOutputWithPooling

            mm = {"image": BaseModelOutputWithPooling(pooler_output=(image_embeds,))}
        out = self.student.model(input_ids=input_ids, position_ids=position_ids, mm_encoder_outputs=mm,
                                 attention_mask=attention_mask, use_cache=False)
        h = out.last_hidden_state[0]
        w_s = self.student.lm_head.weight
        kl_sum = h.new_zeros((), dtype=torch.float32)
        ce_sum = h.new_zeros((), dtype=torch.float32)
        kl_tok = []
        for s in range(0, h.shape[0], self.chunk):
            e = s + self.chunk
            k, c, kt = checkpoint(_distill_chunk, h[s:e], teacher_hidden[s:e], w_s, w_t, kl_mask[s:e],
                                  ce_target[s:e], ce_mask[s:e], use_reentrant=False)
            kl_sum = kl_sum + k
            ce_sum = ce_sum + c
            kl_tok.append(kt)
        kl_tok = torch.cat(kl_tok) * kl_mask
        aux_sum = h.new_zeros((), dtype=torch.float32)
        for layer in self.aux_layers:
            logits = layer.linear_attn.last_skip_logits
            layer.linear_attn.last_skip_logits = None
            if use_aux and logits is not None:
                logits = logits[0].index_select(-1, self.memory_heads)
                target = filler[:, None].expand_as(logits)
                aux_sum = aux_sum + F.binary_cross_entropy_with_logits(logits, target, reduction="sum")
        short = kl_tok[: self.window].sum()
        return kl_sum, ce_sum, aux_sum, short, kl_tok.sum() - short


# ---------------------------------------------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------------------------------------------
def build_student(base, window, attn, resume_dir):
    if resume_dir and os.path.exists(os.path.join(resume_dir, "model", "config.json")):
        student = RecurrentQwenForConditionalGeneration.from_pretrained(
            os.path.join(resume_dir, "model"), dtype=torch.float32, attn_implementation=attn)
        return student, False
    from transformers import AutoConfig

    base_cfg = AutoConfig.from_pretrained(base).to_dict()
    text = dict(base_cfg["text_config"])
    text.pop("model_type", None)
    text.update(sliding_window=window, priority_gate=True, priority_gate_bias_init=-5.0, state_rotation_pairs=16)
    vision = dict(base_cfg["vision_config"])
    vision.pop("model_type", None)
    keep = ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id", "tie_word_embeddings")
    cfg = RecurrentQwenConfig(text_config=text, vision_config=vision, **{k: base_cfg[k] for k in keep if k in base_cfg})
    student = RecurrentQwenForConditionalGeneration.from_pretrained(base, config=cfg, dtype=torch.float32,
                                                                    attn_implementation=attn)
    with torch.no_grad():  # make the new-parameter initialisation explicit (near-identity start)
        for layer in student.model.language_model.layers:
            if layer.block_type == "linear_attention":
                layer.linear_attn.in_proj_skip.weight.zero_()
                layer.linear_attn.in_proj_skip.bias.fill_(cfg.text_config.priority_gate_bias_init)
                layer.linear_attn.in_proj_theta.weight.zero_()
    return student, True


def lr_at(step, total, warmup, peak, floor_frac=0.1):
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (floor_frac + (1 - floor_frac) * 0.5 * (1 + math.cos(math.pi * progress)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen3.5-4B")
    p.add_argument("--data", default="/data/rows")
    p.add_argument("--out", default="/ckpt/run1")
    p.add_argument("--window", type=int, default=2048)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--lr", type=float, default=1.5e-5)
    p.add_argument("--new-lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--ce-weight", type=float, default=0.5)
    p.add_argument("--aux-weight", type=float, default=0.1)
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--attn", default="rq_flex")
    p.add_argument("--pad-to", type=int, default=1024)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--workers", type=int, default=3)
    args = p.parse_args()

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group("nccl")
    rank = dist.get_rank() if distributed else 0
    world = dist.get_world_size() if distributed else 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed + rank)
    is_main = rank == 0

    def log(*a):
        if is_main:
            print(*a, flush=True)

    if args.attn == "rq_flex":
        register_flex_attention()

    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.base)
    tokenizer = processor.tokenizer
    image_pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    pad_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    teacher = Qwen3_5ForConditionalGeneration.from_pretrained(args.base, dtype=torch.bfloat16,
                                                              attn_implementation="sdpa").to(device).eval()
    teacher.requires_grad_(False)

    resume_dir = os.path.join(args.out, "latest")
    student, fresh = build_student(args.base, args.window, args.attn, resume_dir)
    student.model.visual.to(torch.bfloat16).requires_grad_(False)  # frozen: identical to the teacher's encoder
    student.to(device)
    student.config.use_cache = False
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.train()

    layers = student.model.language_model.layers
    first_attn = min(i for i, l in enumerate(layers) if l.block_type != "linear_attention")
    aux_layers = [l for i, l in enumerate(layers) if l.block_type == "linear_attention" and i > first_attn]
    for l in aux_layers:
        l.linear_attn.record_skip_logits = True
    n_v = aux_layers[0].linear_attn.num_v_heads
    memory_heads = list(range(1, n_v, 2))

    new_params, base_params = [], []
    for name, prm in student.named_parameters():
        if not prm.requires_grad:
            continue
        (new_params if any(k in name for k in NEW_PARAM_KEYS) else base_params).append(prm)
    log(f"trainable: base={sum(p.numel() for p in base_params) / 1e9:.3f}B new={sum(p.numel() for p in new_params) / 1e6:.2f}M"
        f" | aux layers={len(aux_layers)} memory heads/layer={len(memory_heads)} | fresh={fresh}")
    opt = torch.optim.AdamW([{"params": base_params, "lr": args.lr, "peak": args.lr},
                             {"params": new_params, "lr": args.new_lr, "peak": args.new_lr}],
                            betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0, fused=True)

    start_step = 0
    state_path = os.path.join(resume_dir, "trainer_state.pt")
    if not fresh and os.path.exists(state_path):
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        opt.load_state_dict(state["optimizer"])
        start_step = state["step"]
        log(f"resumed from step {start_step}")

    module = DistillStudent(student, aux_layers, memory_heads, window=args.window).to(device)
    model = nn.parallel.DistributedDataParallel(module, device_ids=[local_rank], gradient_as_bucket_view=True,
                                                static_graph=False) if distributed else module

    files = list_files(args.data)
    buckets, rows = assign_files(files, world, args.seed)
    total_steps = min(rows) // args.accum
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    log(f"files={len(files)} rows/rank={rows} total_tokens={sum(f['tokens'] for f in files) / 1e6:.1f}M "
        f"steps={total_steps} global_batch_rows={args.accum * world}")
    loader = iter(make_loader(buckets[rank], seed=args.seed + 7919 * start_step + rank, num_workers=args.workers))

    w_t = teacher.lm_head.weight
    metrics_path = os.path.join(args.out, "metrics.jsonl")
    os.makedirs(args.out, exist_ok=True)
    agg = {}
    t_last, tok_since = time.time(), 0

    for step in range(start_step, total_steps):
        lr_scale = lr_at(step, total_steps, args.warmup, 1.0)
        for g in opt.param_groups:
            g["lr"] = g["peak"] * lr_scale

        batch = []
        for _ in range(args.accum):
            row = next(loader)
            ids = row["input_ids"]
            T = ids.numel()
            pad = (-T) % args.pad_to
            flags = row["flags"].long()
            ids_p = F.pad(ids, (0, pad), value=pad_id)
            valid = torch.zeros(T + pad, dtype=torch.bool)
            valid[:T] = True
            is_img = ids_p == image_pad
            nxt = torch.roll(ids_p, -1)
            kl_mask = (valid & torch.roll(valid, -1) & ~is_img & ~torch.roll(is_img, -1)).float()
            kl_mask[-1] = 0
            flags_p = F.pad(flags, (0, pad))
            ce_mask = ((torch.roll(flags_p, -1) & FLAG_CE) > 0).float() * kl_mask
            filler = ((flags_p & FLAG_FILLER) > 0).float()
            use_aux = row["source"].startswith("synthetic")
            n_aux = int(valid.sum()) * len(aux_layers) * len(memory_heads) if use_aux else 0
            batch.append(dict(row=row, ids=ids_p, nxt=nxt, kl_mask=kl_mask, ce_mask=ce_mask, filler=filler,
                              use_aux=use_aux, n_aux=n_aux, T=T))
        counts = torch.tensor([sum(b["kl_mask"].sum() for b in batch), sum(b["ce_mask"].sum() for b in batch),
                               sum(b["n_aux"] for b in batch)], device=device, dtype=torch.float64)
        if distributed:
            dist.all_reduce(counts)
        n_kl, n_ce, n_aux = [max(float(c), 1.0) for c in counts]

        for i, b in enumerate(batch):
            row = b["row"]
            ids = b["ids"].to(device)[None]
            position_ids, image_embeds = None, None
            if row["images"]:
                images = [im.to(device, non_blocking=True) for im in row["images"]]
                proc = processor.image_processor(images=images, return_tensors="pt", device=device)
                grid = proc["image_grid_thw"].to(device)
                if not torch.equal(grid.cpu(), row["grids"]):
                    raise RuntimeError(f"grid mismatch {grid.tolist()} vs {row['grids'].tolist()}")
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    feats = teacher.model.get_image_features(proc["pixel_values"].to(device, torch.bfloat16), grid)
                    image_embeds = torch.cat(feats.pooler_output, dim=0).to(torch.bfloat16)
                mm_types = (ids == image_pad).int()
                position_ids, _ = teacher.model.get_rope_index(ids, mm_types, image_grid_thw=grid)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                mm = None
                if image_embeds is not None:
                    from transformers.modeling_outputs import BaseModelOutputWithPooling

                    mm = {"image": BaseModelOutputWithPooling(pooler_output=(image_embeds,))}
                th = teacher.model(input_ids=ids, position_ids=position_ids, mm_encoder_outputs=mm,
                                   use_cache=False).last_hidden_state[0]
            sync = (i == len(batch) - 1) or not distributed
            ctx = contextlib.nullcontext() if sync else model.no_sync()
            with ctx, torch.autocast("cuda", dtype=torch.bfloat16):
                attn_mask = {"sliding_attention": None, "linear_attention": None} if args.attn == "rq_flex" else None
                kl_sum, ce_sum, aux_sum, kl_short, kl_long = model(
                    ids, position_ids, image_embeds, th, w_t, b["kl_mask"].to(device), b["nxt"].to(device),
                    b["ce_mask"].to(device), b["filler"].to(device), b["use_aux"], attn_mask)
                loss = kl_sum / n_kl + args.ce_weight * ce_sum / n_ce + args.aux_weight * aux_sum / n_aux
                (loss * world).backward()
            src = row["source"].split("-")[0]
            for key, val in (("kl", kl_sum), ("ce", ce_sum), ("aux", aux_sum), ("kl_short", kl_short),
                             ("kl_long", kl_long), (f"kl/{src}", kl_sum)):
                agg[key] = agg.get(key, 0.0) + float(val)
            agg["n_kl"] = agg.get("n_kl", 0.0) + float(b["kl_mask"].sum())
            agg["n_ce"] = agg.get("n_ce", 0.0) + float(b["ce_mask"].sum())
            agg["n_aux"] = agg.get("n_aux", 0.0) + b["n_aux"]
            agg["n_short"] = agg.get("n_short", 0.0) + float(b["kl_mask"][: args.window].sum())
            agg[f"n/{src}"] = agg.get(f"n/{src}", 0.0) + float(b["kl_mask"].sum())
            tok_since += b["T"]

        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if (step + 1) % 5 == 0 or step == start_step:
            keys = sorted(agg)
            vec = torch.tensor([agg[k] for k in keys], device=device, dtype=torch.float64)
            tok = torch.tensor([tok_since], device=device, dtype=torch.float64)
            if distributed:
                # every rank may have seen different sources: gather key sets first
                all_keys = [None] * world
                dist.all_gather_object(all_keys, keys)
                union = sorted(set().union(*all_keys))
                vec = torch.tensor([agg.get(k, 0.0) for k in union], device=device, dtype=torch.float64)
                keys = union
                dist.all_reduce(vec)
                dist.all_reduce(tok)
            tot = dict(zip(keys, vec.tolist()))
            dt = time.time() - t_last
            n_long = tot["n_kl"] - tot["n_short"]
            rec = {
                "step": step + 1, "of": total_steps, "lr": opt.param_groups[0]["lr"],
                "kl": tot["kl"] / max(tot["n_kl"], 1), "kl_short": tot["kl_short"] / max(tot["n_short"], 1),
                "kl_long": tot["kl_long"] / max(n_long, 1), "ce": tot["ce"] / max(tot["n_ce"], 1),
                "aux": tot["aux"] / max(tot["n_aux"], 1), "grad_norm": float(grad_norm),
                "tok_per_s": tok.item() / dt, "mem_gb": torch.cuda.max_memory_allocated() / 1e9,
            }
            for k in keys:
                if k.startswith("kl/"):
                    rec[k] = tot[k] / max(tot.get("n/" + k[3:], 1.0), 1.0)
            log(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}))
            if is_main:
                with open(metrics_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            agg, t_last, tok_since = {}, time.time(), 0

        if (step + 1) % args.save_every == 0 or step + 1 == total_steps:
            save(student, opt, step + 1, args.out, is_main, distributed)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


def save(student, opt, step, out, is_main, distributed):
    if is_main:
        t = time.time()
        tmp = os.path.join(out, "latest.tmp")
        os.makedirs(tmp, exist_ok=True)
        student.save_pretrained(os.path.join(tmp, "model"), safe_serialization=True)
        torch.save({"optimizer": opt.state_dict(), "step": step}, os.path.join(tmp, "trainer_state.pt"))
        final = os.path.join(out, "latest")
        if os.path.exists(final):
            os.rename(final, os.path.join(out, "previous.del"))
        os.rename(tmp, final)
        os.system(f"rm -rf {os.path.join(out, 'previous.del')}")
        print(f"saved step {step} in {time.time() - t:.0f}s", flush=True)
    if distributed:
        dist.barrier()


if __name__ == "__main__":
    main()
