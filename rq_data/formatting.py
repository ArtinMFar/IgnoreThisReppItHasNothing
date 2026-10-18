"""Token-level formatting helpers shared by data preparation and training.

Every training row is a token sequence plus a per-token uint8 flag array:
  FLAG_CE        the token is a supervised target (cross-entropy on ground truth, e.g. synthetic recall answers)
  FLAG_FILLER    the token belongs to repetitive / low-information content (priority-gate auxiliary target)
  FLAG_ASSIST    the token belongs to an assistant turn (stats only)
KL distillation against the teacher is applied to every position except image placeholder tokens.
"""

import io

import numpy as np

FLAG_CE = 1
FLAG_FILLER = 2
FLAG_ASSIST = 4

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
EMPTY_THINK = "<think>\n\n</think>\n\n"
IMAGE_SLOT = "<|vision_start|><|image_pad|><|vision_end|>"

MIN_PIXELS = 256 * 256
MAX_PIXELS = 896 * 896
PATCH = 16
MERGE = 2


class Segment:
    """A piece of text with a flag value, tokenized independently and concatenated."""

    __slots__ = ("text", "flag")

    def __init__(self, text: str, flag: int = 0):
        self.text = text
        self.flag = flag


def chat_segments(messages, think_for_last=True):
    """Renders messages exactly like the Qwen3.5 chat template (text only), as flagged segments.

    `messages` items: {"role", "content", optional "flag" (applied to the content), optional "reasoning"}.
    A system/user message may instead carry "parts": a list of (text, flag) pieces forming its content.
    Assistant turns after the last user turn get a (possibly empty) think block, as in the official template.
    """
    last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
    segs = []
    for i, m in enumerate(messages):
        role, flag = m["role"], m.get("flag", 0)
        if role in ("system", "user"):
            segs.append(Segment(f"{IM_START}{role}\n"))
            if "parts" in m:
                segs.extend(Segment(text, part_flag) for text, part_flag in m["parts"])
            else:
                segs.append(Segment(m["content"].strip(), flag))
            segs.append(Segment(f"{IM_END}\n"))
        elif role == "assistant":
            head = f"{IM_START}assistant\n"
            if i > last_user and think_for_last:
                reasoning = (m.get("reasoning") or "").strip()
                head += f"<think>\n{reasoning}\n</think>\n\n" if reasoning else EMPTY_THINK
            segs.append(Segment(head, FLAG_ASSIST))
            segs.append(Segment(m["content"].strip(), flag | FLAG_ASSIST))
            segs.append(Segment(f"{IM_END}\n", (flag & FLAG_CE) | FLAG_ASSIST))
        else:
            raise ValueError(role)
    return segs


def encode_segments(tokenizer, segments):
    ids, flags = [], []
    texts = [s.text for s in segments]
    encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
    for seg, toks in zip(segments, encoded):
        ids.extend(toks)
        flags.extend([seg.flag] * len(toks))
    return ids, flags


def mark_assistant_spans(ids, tokenizer):
    """Flags tokens of `<|im_start|>assistant ... <|im_end|>` spans in an already tokenized conversation."""
    im_start = tokenizer.convert_tokens_to_ids(IM_START)
    im_end = tokenizer.convert_tokens_to_ids(IM_END)
    assistant = tokenizer("assistant", add_special_tokens=False)["input_ids"]
    flags = np.zeros(len(ids), dtype=np.uint8)
    i, n, k = 0, len(ids), len(assistant)
    while i < n:
        if ids[i] == im_start and ids[i + 1 : i + 1 + k] == assistant:
            j = i
            while j < n and ids[j] != im_end:
                j += 1
            flags[i : min(j + 1, n)] |= FLAG_ASSIST
            i = j
        i += 1
    return flags


def smart_resize(height, width, factor=PATCH * MERGE, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS):
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize as _smart_resize

    return _smart_resize(height, width, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels)


def prepare_image(img):
    """Resizes a PIL image to the processor's fixed point and returns (jpeg_bytes, (t, h, w) grid, n_llm_tokens)."""
    from PIL import Image

    img = img.convert("RGB")
    w, h = img.size
    if min(w, h) < 12 or max(w, h) / min(w, h) > 150:
        raise ValueError("degenerate image")
    new_h, new_w = smart_resize(h, w)
    if (new_w, new_h) != (w, h):
        img = img.resize((new_w, new_h), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    grid = (1, new_h // PATCH, new_w // PATCH)
    return buf.getvalue(), grid, (grid[1] * grid[2]) // (MERGE * MERGE)


def expand_image_tokens(ids, flags, n_tokens_per_image, image_pad_id):
    out_ids, out_flags, k = [], [], 0
    for t, f in zip(ids, flags):
        if t == image_pad_id:
            out_ids.extend([image_pad_id] * n_tokens_per_image[k])
            out_flags.extend([f] * n_tokens_per_image[k])
            k += 1
        else:
            out_ids.append(t)
            out_flags.append(f)
    if k != len(n_tokens_per_image):
        raise ValueError("image placeholder count mismatch")
    return out_ids, out_flags


class Packer:
    """Greedy packer: concatenates samples into rows of at most `max_len` tokens."""

    def __init__(self, max_len, emit):
        self.max_len = max_len
        self.emit = emit
        self._reset()

    def _reset(self):
        self.ids, self.flags, self.images, self.grids, self.sources = [], [], [], [], []

    def add(self, ids, flags, source, images=(), grids=()):
        if len(ids) > self.max_len:
            return False
        if len(self.ids) + len(ids) > self.max_len:
            self.flush()
        self.ids.extend(ids)
        self.flags.extend(flags)
        self.images.extend(images)
        self.grids.extend(grids)
        self.sources.append(source)
        return True

    def flush(self):
        if self.ids:
            self.emit(
                {
                    "input_ids": np.asarray(self.ids, dtype=np.int32),
                    "flags": np.asarray(self.flags, dtype=np.uint8).tobytes(),
                    "images": list(self.images),
                    "grids": [list(g) for g in self.grids],
                    "source": self.sources[0],
                    "n_tokens": len(self.ids),
                }
            )
        self._reset()
