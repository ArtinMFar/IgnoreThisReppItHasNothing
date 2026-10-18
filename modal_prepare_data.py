"""Builds the RecurrentQwen training mixture on Modal CPU containers and stores it in a Modal Volume.

    modal run modal_prepare_data.py --scale 0.001     # smoke test: a few rows per source
    modal run --detach modal_prepare_data.py          # full build (~400M tokens)

Output: /rows/<job>/part-*.parquet in the `recurrent-qwen-data` volume, plus /rows/<job>/_DONE (json stats).
"""

import json
import os

import modal

app = modal.App("recurrent-qwen-data")
data_vol = modal.Volume.from_name("recurrent-qwen-data", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.10.0", "torchvision==0.25.0", index_url="https://download.pytorch.org/whl/cpu")
    .uv_pip_install("transformers==5.20.1", "datasets>=4.4", "pillow", "pyarrow", "numpy")
    .env({"HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("rq_data")
)

BASE_MODEL = "Qwen/Qwen3.5-4B"


@app.function(
    image=image, volumes={"/data": data_vol}, secrets=[hf_secret], cpu=2, memory=8192, timeout=4 * 3600,
    retries=1, max_containers=48,
)
def build(job, scale: float = 1.0):
    import shutil

    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    from rq_data.formatting import Packer
    from rq_data.sources import SOURCES

    name, kind, kwargs, budget, max_len = job
    budget = int(budget * scale)
    out_dir = f"/data/rows/{name}"
    done_path = f"{out_dir}/_DONE"
    if os.path.exists(done_path) and scale == 1.0:
        return json.load(open(done_path))
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    schema = pa.schema([
        ("input_ids", pa.list_(pa.int32())),
        ("flags", pa.binary()),
        ("images", pa.list_(pa.binary())),
        ("grids", pa.list_(pa.list_(pa.int32()))),
        ("source", pa.string()),
        ("n_tokens", pa.int32()),
    ])
    rows, part = [], [0]
    stats = {"job": name, "rows": 0, "tokens": 0, "samples": 0, "images": 0, "max_len": max_len}

    def write_part():
        if not rows:
            return
        table = pa.Table.from_pylist(rows, schema=schema)
        pq.write_table(table, f"{out_dir}/part-{part[0]:05d}.parquet", compression="zstd")
        part[0] += 1
        rows.clear()

    def emit(row):
        rows.append(row)
        stats["rows"] += 1
        stats["tokens"] += row["n_tokens"]
        stats["images"] += len(row["images"])
        if len(rows) >= 256:
            write_part()

    packer = Packer(max_len, emit)
    added = 0
    for ids, flags, images, grids in SOURCES[kind](tokenizer, **kwargs):
        if packer.add(ids, flags, name, images, grids):
            added += len(ids)
            stats["samples"] += 1
        if added >= budget:
            break
    packer.flush()
    write_part()
    json.dump(stats, open(done_path, "w"))
    data_vol.commit()
    print(json.dumps(stats))
    return stats


@app.local_entrypoint()
def main(scale: float = 1.0, only: str = ""):
    from rq_data.sources import job_specs

    jobs = [j for j in job_specs() if not only or only in j[0]]
    total = 0
    for result in build.map(jobs, kwargs={"scale": scale}, return_exceptions=True, order_outputs=False):
        if isinstance(result, Exception):
            print("FAILED:", repr(result)[:500])
            continue
        total += result["tokens"]
        print(f"{result['job']:<70} rows={result['rows']:>7} tokens={result['tokens'] / 1e6:8.2f}M "
              f"images={result['images']}")
    print(f"TOTAL tokens: {total / 1e6:.1f}M")
