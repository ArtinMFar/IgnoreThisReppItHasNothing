# RecurrentQwen-4b

Converts [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (vision-language, hybrid Gated-DeltaNet + attention)
into a RecurrentGemma-style model: **2,048-token sliding-window attention + recurrent layers with a fixed-size state**.
After 2k tokens of context the cache stops growing and per-token generation cost stays flat.

Released model: https://huggingface.co/ArtinMFar/RecurrentQwen-4b

## Architecture changes

| | Qwen3.5-4B | RecurrentQwen-4b |
|---|---|---|
| 24 Gated DeltaNet layers | fixed-size recurrent state | same, plus **priority (skip) gate** and **data-dependent state rotation** |
| 8 attention layers | full attention (KV cache grows with context) | **2,048-token sliding window** (KV cache capped) |
| Vision encoder | ✓ | ✓ (unchanged, frozen) |

* **Priority gate** (`in_proj_skip`): `keep = 1 - sigmoid(w·x + b)` per value head scales the step size (decay and
  rotation) and the delta-rule write strength. A token the gate closes on neither decays nor writes the recurrent
  state, so repetitive content (NPC movement commands, game ticks, logs, telemetry, chat spam) can't wash out important
  information. Half the heads of each recurrent layer are explicitly supervised to skip repetitive filler.
* **State rotation** (`in_proj_theta`): Mamba-3's complex-valued state via the RoPE trick. 16 key/query channel pairs
  per head are rotated by `phase_t = Σ_{s≤t} dt_s · θ(x_s)`. The phase is carried in the generation cache.
* New parameters are initialised so that the converted model starts at the original model's behaviour.

## Layout

| Path | What |
|---|---|
| `recurrent_qwen/` | `configuration_recurrent_qwen.py`, `modeling_recurrent_qwen.py` (shipped to the Hub as remote code) |
| `rq_data/` | chat formatting, synthetic long-memory tasks with repetitive filler, dataset readers |
| `rq_train/train.py` | full-parameter distillation: KL to the original model + CE on ground-truth answers + gate aux loss |
| `rq_train/evaluate.py` | cache/latency scaling, recall under repetitive filler, long-text NLL, ChartQA, gate stats |
| `rq_train/export.py`, `rq_train/model_card.py` | Hub export (auto_map + code files) and model card |
| `modal_prepare_data.py` | builds the ~400M-token mixture on Modal CPUs into a Volume |
| `modal_train.py` | Modal GPU entry points: `bench`, `train` (1/4/8× B200), `evaluate`, `upload`, `verify_hub` |
| `tests/test_modeling_cpu.py` | CPU tests on a tiny random config (identity at init, window, cache equivalence, save/load) |

## Reproduce

```bash
pip install "modal[api-proxy-support]"
modal secret create huggingface HF_TOKEN=...          # used to download data/models and to upload
modal run --detach modal_prepare_data.py              # data -> volume `recurrent-qwen-data`
modal run modal_train.py::bench                       # GPU sanity checks
modal run --detach modal_train.py::train --gpus 8     # distillation -> volume `recurrent-qwen-ckpt`
modal run modal_train.py::evaluate --model /ckpt/run1/latest/model --out /ckpt/eval_student.json
```

Local CPU tests: `python -m tests.test_modeling_cpu` (needs `torch` and `transformers>=5.20`).
