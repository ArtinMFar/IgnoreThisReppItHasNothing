"""Per-source sample iterators. Each yields (input_ids, flags, images, grids) for one sample."""

import random

import numpy as np

from .formatting import (
    FLAG_CE,
    IMAGE_SLOT,
    chat_segments,
    encode_segments,
    expand_image_tokens,
    mark_assistant_spans,
    prepare_image,
)
from .synthetic_memory import generate_sample


def _stream(name, config=None, split="train", seed=0, buffer_size=2000):
    from datasets import load_dataset

    ds = load_dataset(name, config, split=split, streaming=True)
    return ds.shuffle(seed=seed, buffer_size=buffer_size)


def chat_source(tokenizer, dataset, config, split, seed=0, **_):
    """smoltalk2-style rows with a `messages` list (think splits carry <think> blocks inside the content)."""
    for row in _stream(dataset, config, split, seed):
        messages = [m for m in row["messages"] if m.get("content") is not None]
        if not messages or messages[-1]["role"] != "assistant" or not any(m["role"] == "user" for m in messages):
            continue
        if messages[0]["role"] == "system" and not messages[0]["content"].strip():
            messages = messages[1:]
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception:
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        yield ids, mark_assistant_spans(ids, tokenizer).tolist(), [], []


def fineweb_source(tokenizer, seed=0, min_tokens=2500, chunk=8192, **_):
    """Long educational web documents as raw text (pretraining format), split into window-sized chunks."""
    eos = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    for row in _stream("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", seed):
        if row.get("token_count", 0) < min_tokens:
            continue
        ids = tokenizer(row["text"], add_special_tokens=False)["input_ids"] + [eos]
        for start in range(0, len(ids), chunk):
            piece = ids[start : start + chunk]
            if len(piece) >= 256:
                yield piece, [0] * len(piece), [], []


def pg19_source(tokenizer, seed=0, window=16384, per_book=4, **_):
    """Contiguous windows from public-domain books: genuinely long-range dependencies (characters, plot)."""
    rng = random.Random(seed)
    for row in _stream("emozilla/pg19", None, "train", seed, buffer_size=16):
        text = row["text"]
        if len(text) < 80_000:
            continue
        for _ in range(per_book):
            start = rng.randint(0, len(text) - 75_000)
            ids = tokenizer(text[start : start + 75_000], add_special_tokens=False)["input_ids"]
            if len(ids) >= window:
                yield ids[:window], [0] * window, [], []


def multidoc_qa_source(tokenizer, seed=0, min_len=4000, max_len=15000, **_):
    """HotpotQA questions whose supporting paragraphs are buried among many distractor paragraphs."""
    rng = random.Random(seed)
    pool = []
    for row in _stream("hotpotqa/hotpot_qa", "distractor", "train", seed):
        paragraphs = [
            (title, "".join(sents)) for title, sents in zip(row["context"]["title"], row["context"]["sentences"])
        ]
        gold_titles = set(row["supporting_facts"]["title"])
        gold = [p for p in paragraphs if p[0] in gold_titles]
        pool.extend(p for p in paragraphs if p[0] not in gold_titles)
        pool = pool[-5000:]
        if len(pool) < 500 or not gold:
            continue
        target = rng.randint(min_len, max_len)
        docs, n_tok = [], 0
        while n_tok < target:
            p = rng.choice(pool)
            docs.append(p)
            n_tok += len(p[1]) // 4 + 12
        for g in gold:  # gold paragraphs early, i.e. far outside the attention window of the question
            docs.insert(rng.randint(0, max(1, int(len(docs) * 0.6))), g)
        body = "\n\n".join(f"Document [{i + 1}] (Title: {t})\n{x}" for i, (t, x) in enumerate(docs))
        messages = [
            {"role": "user", "content": f"Answer the question based on the documents below.\n\n{body}\n\n"
                                        f"Question: {row['question']}\nAnswer with a short phrase."},
            {"role": "assistant", "content": row["answer"], "flag": FLAG_CE},
        ]
        ids, flags = encode_segments(tokenizer, chat_segments(messages))
        yield ids, flags, [], []


def synthetic_source(tokenizer, seed=0, min_len=3000, max_len=15500, **_):
    rng = random.Random(seed)
    while True:
        ids, flags = generate_sample(tokenizer, rng, rng.randint(min_len, max_len))
        yield ids, flags, [], []


def _rating_ok(row, threshold):
    for key in ("relevance_min", "image_correspondence_min", "formatting_min"):
        value = row.get(key)
        if value is not None and value < threshold:
            return False
    return True


def vision_source(tokenizer, config, seed=0, min_rating=4, max_images=4, **_):
    """FineVision rows: `images` (PIL list) + `texts` ([{user, assistant}, ...])."""
    image_pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    for row in _stream("HuggingFaceM4/FineVision", config, "train", seed, buffer_size=64):
        if not _rating_ok(row, min_rating) or not row["images"] or len(row["images"]) > max_images:
            continue
        try:
            prepared = [prepare_image(img) for img in row["images"]]
        except Exception:
            continue
        messages = []
        for turn_idx, turn in enumerate(row["texts"]):
            user = (turn.get("user") or "").strip()
            assistant = (turn.get("assistant") or "").strip()
            if not user or not assistant:
                break
            if turn_idx == 0:
                messages.append({"role": "user", "content": IMAGE_SLOT * len(prepared) + user})
            else:
                messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})
        if not messages:
            continue
        ids, flags = encode_segments(tokenizer, chat_segments(messages))
        try:
            ids, flags = expand_image_tokens(ids, flags, [p[2] for p in prepared], image_pad)
        except ValueError:
            continue
        yield ids, flags, [p[0] for p in prepared], [p[1] for p in prepared]


SOURCES = {
    "chat": chat_source,
    "fineweb": fineweb_source,
    "pg19": pg19_source,
    "multidoc_qa": multidoc_qa_source,
    "synthetic": synthetic_source,
    "vision": vision_source,
}

SMOLTALK = "HuggingFaceTB/smoltalk2"


def job_specs():
    """(name, source kind, kwargs, token budget, max row length)."""
    jobs = []
    no_think = {
        "smoltalk_smollm3_smol_magpie_ultra_no_think": 25e6,
        "smoltalk_smollm3_everyday_conversations_no_think": 3e6,
        "smoltalk_smollm3_systemchats_30k_no_think": 8e6,
        "tulu_3_sft_personas_instruction_following_no_think": 8e6,
        "OpenHermes_2.5_no_think": 15e6,
        "smoltalk_multilingual_8languages_lang_5_no_think": 8e6,
        "table_gpt_no_think": 3e6,
        "hermes_function_calling_v1_no_think": 3e6,
        "Mixture_of_Thoughts_science_no_think": 5e6,
        "OpenThoughts3_1.2M_no_think_no_think": 8e6,
        "smoltalk_smollm3_smol_summarize_no_think": 4e6,
        "smoltalk_smollm3_smol_rewrite_no_think": 2e6,
    }
    think = {
        "OpenThoughts3_1.2M_think": 40e6,
        "multi_turn_reasoning_if_think": 10e6,
        "smoltalk_systemchats_Qwen3_32B_think": 8e6,
        "smoltalk_everyday_convs_reasoning_Qwen3_32B_think": 5e6,
        "aya_dataset_Qwen3_32B_think": 8e6,
        "smolagents_toolcalling_traces_think": 4e6,
        "s1k_1.1_think": 3e6,
        "table_gpt_Qwen3_32B_think": 3e6,
        "smoltalk_multilingual8_Qwen3_32B_think": 6e6,
    }
    for split, budget in {**no_think, **think}.items():
        jobs.append((f"chat-{split}", "chat", {"dataset": SMOLTALK, "config": "SFT", "split": split}, budget, 8192))
    for split, budget in {
        "LongAlign_64k_context_lang_annotated_lang_6_no_think": 15e6,
        "LongAlign_64k_Qwen3_32B_yarn_131k_think": 10e6,
    }.items():
        jobs.append((f"long-{split}", "chat", {"dataset": SMOLTALK, "config": "SFT", "split": split}, budget, 16384))
    jobs.append(("fineweb-edu-long", "fineweb", {}, 50e6, 8192))
    jobs.append(("pg19", "pg19", {}, 40e6, 16384))
    jobs.append(("multidoc-qa", "multidoc_qa", {}, 20e6, 16384))
    for seed in range(3):
        jobs.append((f"synthetic-memory-{seed}", "synthetic", {"seed": 1000 + seed}, 15e6, 16384))
    vision = {
        "LLaVA_Instruct_150K": 5e6, "sharegpt4v(coco)": 3e6, "sharegpt4o": 3e6, "lvis_instruct4v": 3e6,
        "allava_laion": 4e6, "vision_flan(filtered)": 4e6, "densefusion_1m": 3e6, "vqav2": 2e6, "a_okvqa": 1e6,
        "textvqa": 2e6, "docvqa": 3e6, "chartqa": 2e6, "infographic_vqa": 2e6, "ai2d_merged": 2e6,
        "CoSyn_400k_document": 3e6, "CoSyn_400k_chart": 3e6, "textcaps": 1e6, "scienceqa": 1e6,
        "geo170k(qa)": 1e6, "mavis_math_metagen": 2e6, "screen2words": 1e6,
    }
    for config, budget in vision.items():
        safe = config.replace("(", "_").replace(")", "")
        jobs.append((f"vision-{safe}", "vision", {"config": config}, budget, 8192))
    return jobs


def flags_array(flags_bytes):
    return np.frombuffer(flags_bytes, dtype=np.uint8)
