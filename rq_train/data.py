"""Streaming loader over the packed parquet rows produced by `modal_prepare_data.py`.

Files are balanced across ranks by token count, then each DataLoader worker interleaves several open files so a
global batch mixes many sources. Images stay JPEG bytes until a worker decodes them to uint8 tensors; pixel
normalisation/patching happens on the GPU in the training loop.
"""

import glob
import io
import os
import random

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info


def list_files(root, exclude=()):
    files = []
    for job_dir in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.exists(os.path.join(job_dir, "_DONE")):
            continue
        if any(x in os.path.basename(job_dir) for x in exclude):
            continue
        for path in sorted(glob.glob(os.path.join(job_dir, "part-*.parquet"))):
            n_tokens = pq.read_table(path, columns=["n_tokens"]).column(0).to_numpy()
            files.append({"path": path, "rows": len(n_tokens), "tokens": int(n_tokens.sum())})
    return files


def assign_files(files, world_size, seed):
    """Greedy balancing of rows (= micro-steps) across ranks; returns per-rank file lists and row counts."""
    rng = random.Random(seed)
    order = sorted(files, key=lambda f: (-f["rows"], f["path"]))
    buckets = [[] for _ in range(world_size)]
    rows = [0] * world_size
    for f in order:
        r = min(range(world_size), key=lambda i: rows[i])
        buckets[r].append(f)
        rows[r] += f["rows"]
    for b in buckets:
        rng.shuffle(b)
    return buckets, rows


def decode_images(blobs):
    from PIL import Image

    out = []
    for b in blobs:
        img = Image.open(io.BytesIO(b)).convert("RGB")
        out.append(torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).contiguous())
    return out


class RowStream(IterableDataset):
    def __init__(self, files, seed, open_files=6):
        self.files = files
        self.seed = seed
        self.open_files = open_files

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info is not None else (0, 1)
        rng = random.Random(self.seed * 1009 + wid)
        pending = list(self.files[wid::nw])
        buffers = []
        while pending or buffers:
            while pending and len(buffers) < self.open_files:
                table = pq.read_table(pending.pop(0)["path"])
                rows = table.to_pylist()
                rng.shuffle(rows)
                buffers.append(rows)
            k = rng.choices(range(len(buffers)), weights=[len(b) for b in buffers])[0]
            row = buffers[k].pop()
            if not buffers[k]:
                buffers.pop(k)
            yield {
                "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
                "flags": torch.from_numpy(np.frombuffer(row["flags"], dtype=np.uint8).copy()),
                "images": decode_images(row["images"]) if row["images"] else [],
                "grids": torch.tensor(row["grids"], dtype=torch.long).view(-1, 3),
                "source": row["source"],
            }


def make_loader(files, seed, num_workers=3):
    return torch.utils.data.DataLoader(
        RowStream(files, seed), batch_size=None, num_workers=num_workers, prefetch_factor=4,
        persistent_workers=False, pin_memory=False,
    )
