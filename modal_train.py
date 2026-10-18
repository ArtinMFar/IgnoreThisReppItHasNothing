"""Modal entry points for RecurrentQwen: GPU sanity bench, distillation training (B200) and evaluation.

    modal run modal_train.py::bench                      # sanity checks + attention benchmark on 1x B200
    modal run modal_train.py::train --gpus 1 --max-steps 4 --accum 2 --out /ckpt/smoke   # smoke test
    modal run --detach modal_train.py::train --gpus 4    # full run (resumes from /ckpt/<run>/latest)
"""

import subprocess

import modal

app = modal.App("recurrent-qwen-train")

data_vol = modal.Volume.from_name("recurrent-qwen-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("recurrent-qwen-ckpt", create_if_missing=True)
hf_vol = modal.Volume.from_name("recurrent-qwen-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch==2.13.0", "torchvision==0.28.0", index_url="https://download.pytorch.org/whl/cu130")
    .uv_pip_install(
        "transformers==5.20.1", "flash-linear-attention==0.5.2", "accelerate", "safetensors", "pyarrow", "pillow",
        "numpy", "einops", "datasets",
    )
    .env({
        "HF_HOME": "/hf", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "TRITON_CACHE_DIR": "/tmp/triton",
    })
    .add_local_python_source("recurrent_qwen", "rq_data", "rq_train")
)

VOLUMES = {"/data": data_vol, "/ckpt": ckpt_vol, "/hf": hf_vol}


def _download_base():
    from huggingface_hub import snapshot_download

    snapshot_download("Qwen/Qwen3.5-4B")
    hf_vol.commit()


@app.function(image=image, volumes=VOLUMES, secrets=[hf_secret], gpu="B200", timeout=3600, cpu=8, memory=65536)
def bench(teacher_only: bool = False):
    _download_base()
    from rq_train.bench import main, teacher_memory

    teacher_memory()
    if not teacher_only:
        main()


@app.function(image=image, volumes=VOLUMES, secrets=[hf_secret], gpu="B200", timeout=3600, cpu=8, memory=65536)
def bench_shapes():
    _download_base()
    from rq_train.bench_shapes import main

    main()


def _train(gpus: int, extra: list[str]):
    import os

    _download_base()
    if "--debug-mem" in extra:
        extra = [e for e in extra if e != "--debug-mem"]
        os.environ["RQ_DEBUG_MEM"] = "1"
    cmd = ["torchrun", f"--nproc_per_node={gpus}", "--standalone", "-m", "rq_train.train", *extra]
    print(" ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=False)
    ckpt_vol.commit()
    if proc.returncode != 0:
        raise RuntimeError(f"training exited with {proc.returncode}")


TRAIN_KW = dict(image=image, volumes=VOLUMES, secrets=[hf_secret], timeout=24 * 3600,
                retries=modal.Retries(max_retries=2, initial_delay=30.0))


@app.function(gpu="B200", cpu=12, memory=196608, **TRAIN_KW)
def train_1(extra: list[str]):
    _train(1, extra)


@app.function(gpu="B200:4", cpu=32, memory=262144, **TRAIN_KW)
def train_4(extra: list[str]):
    _train(4, extra)


@app.function(gpu="B200:8", cpu=48, memory=393216, **TRAIN_KW)
def train_8(extra: list[str]):
    _train(8, extra)


@app.local_entrypoint()
def train(gpus: int = 1, out: str = "/ckpt/run1", max_steps: int = 0, accum: int = 8, save_every: int = 50,
          lr: float = 1e-5, gate_lr: float = 1e-4, theta_lr: float = 2e-6, attn: str = "rq_flex",
          warmup: int = 40, debug_mem: bool = False):
    extra = ["--debug-mem"] * debug_mem + ["--out", out, "--max-steps", str(max_steps), "--accum", str(accum), "--save-every", str(save_every),
             "--lr", str(lr), "--gate-lr", str(gate_lr), "--theta-lr", str(theta_lr), "--attn", attn,
             "--warmup", str(warmup)]
    {1: train_1, 4: train_4, 8: train_8}[gpus].remote(extra)


@app.function(image=image, volumes=VOLUMES, secrets=[hf_secret], gpu="B200", cpu=8, memory=131072, timeout=4 * 3600)
def evaluate(model: str, out: str, skip: str = ""):
    _download_base()
    cmd = ["python", "-m", "rq_train.evaluate", "--model", model, "--out", out, "--skip", skip]
    subprocess.run(cmd, check=True)
    ckpt_vol.commit()


@app.function(image=image, volumes=VOLUMES, secrets=[hf_secret], cpu=8, memory=98304, timeout=4 * 3600)
def upload(checkpoint: str, repo: str, readme_path: str = "", private: bool = False):
    import os
    import shutil

    from huggingface_hub import HfApi

    from rq_train.export import export

    _download_base()
    out_dir = export(checkpoint, "/ckpt/export", base="Qwen/Qwen3.5-4B")
    if readme_path:
        shutil.copy(readme_path, os.path.join(out_dir, "README.md"))
    ckpt_vol.commit()
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo, repo_type="model", exist_ok=True, private=private)
    api.upload_folder(folder_path=out_dir, repo_id=repo, repo_type="model",
                      commit_message="Upload RecurrentQwen-4b (Qwen3.5-4B -> sliding-window + recurrent hybrid)")
    print(f"uploaded to https://huggingface.co/{repo}")


@app.function(image=image, volumes=VOLUMES, secrets=[hf_secret], gpu="B200", cpu=8, memory=65536, timeout=3600)
def verify_hub(repo: str):
    """Loads the uploaded repo from the Hub with trust_remote_code and chats with it (text + image)."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(repo)
    model = AutoModelForImageTextToText.from_pretrained(repo, trust_remote_code=True, dtype=torch.bfloat16).cuda()
    messages = [{"role": "user", "content": [{"type": "text", "text": "Give me three tips for writing clean code."}]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                           return_tensors="pt", enable_thinking=False).to("cuda")
    out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    print("TEXT:", processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
    messages = [{"role": "user", "content": [{"type": "image", "url": url},
                                             {"type": "text", "text": "Describe this image in two sentences."}]}]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                           return_tensors="pt", enable_thinking=False).to("cuda")
    out = model.generate(**inputs, max_new_tokens=120, do_sample=False)
    print("VISION:", processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))

    # Multi-turn NPC loop with the cache kept between turns (the usage pattern from the model card), with thousands of
    # tokens of repetitive game ticks between stating facts and asking about them.
    from transformers import DynamicCache

    tok = processor.tokenizer
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    cache = DynamicCache(config=model.config.get_text_config())
    ids = tok("<|im_start|>system\nYou are Garrick, the village blacksmith NPC. Game ticks arrive as user messages; "
              "answer them with a single movement command.<|im_end|>", return_tensors="pt").input_ids.cuda()
    msgs = ["[Player]: Hi Garrick! My name is Lyra and the password to the guild hall is 'silver heron'."]
    msgs += [f"[tick {t}] npc_pos=({10 + t % 7}, {4 + t % 5}) state=walking" for t in range(330)]
    msgs += ["[Player]: Do you remember my name and the guild hall password?"]
    for i, msg in enumerate(msgs):
        turn = f"\n<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        ids = torch.cat([ids, tok(turn, return_tensors="pt").input_ids.cuda()], dim=1)
        last = i in (0, len(msgs) - 1)
        out = model.generate(input_ids=ids, past_key_values=cache, max_new_tokens=60 if last else 12, do_sample=False)
        reply = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        if i in (0, 1, 2) or last:
            print(f"NPC turn {i} (context {ids.shape[1]} tokens): {msg[:60]!r} -> {reply!r}", flush=True)
        ids = out if out[0, -1] == im_end else torch.cat([out, out.new_tensor([[im_end]])], dim=1)
