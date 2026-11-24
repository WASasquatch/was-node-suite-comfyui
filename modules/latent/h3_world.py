"""MiniMax H3 world control: keyboard and camera actions as one sentence per video latent.

Sentences follow the H3-World rule table. Times are seconds at 24 fps; latent ``k`` covers the
pixel frames :func:`frame_span` gives it.
"""

from __future__ import annotations

import dataclasses
import math
import re

from .. import log

__all__ = [
    "COND_KEY", "FPS", "KEYS", "MAX_ENTRIES", "MAX_SECONDS", "NULL_SENTENCE", "PRESETS",
    "Routing", "Script", "attention_override", "describe", "frame_span", "latent_keys",
    "parse", "patched", "route", "routed_attention", "script", "sentence",
]

logger = log.get_logger("latent.h3_world")

#: Conditioning key the action script rides under.
COND_KEY = "was_h3_world"

#: Key the sampling wrapper is registered under.
WRAPPER_KEY = "was_h3_world"

#: Frame rate the script's seconds are read at.
FPS = 24.0

#: Pixel frames each video latent covers, repeating every five latents from frame 0.
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)

#: Entries one script holds at most.
MAX_ENTRIES = 1000

#: Video rows routed together in one pass at :data:`HEAD_BUDGET` heads, a whole number of latents.
CHUNK_ROWS = 2048

#: Heads :data:`CHUNK_ROWS` is counted at; a call with fewer heads routes more rows at once.
HEAD_BUDGET = 40

#: Latest time an entry may reach, in seconds.
MAX_SECONDS = 3600.0

#: The nine action bits, movement and camera keys then the fast bit.
KEYS = ("W", "A", "S", "D", "I", "J", "K", "L", "F")

#: Movement clauses, joined in this order.
MOTION = {"W": "walks forward", "S": "walks backward", "A": "strafes left", "D": "strafes right"}
MOTION_ORDER = ("W", "S", "A", "D")
MOTION_IDLE = "stands still"

#: Camera clauses.
PAN_KEY = {"J": "left", "L": "right"}
TILT_KEY = {"I": "tilts down", "K": "tilts up"}
CAMERA_IDLE = "holds steady"
CAMERA_FOLLOW = "follows him"

#: The sentence of a latent with nothing held, and the negative reference for guidance.
NULL_SENTENCE = f"the man {MOTION_IDLE}, camera {CAMERA_IDLE}"

#: Action names a script entry takes, each the keys it holds.
PRESETS = {
    "still": (),
    "forward": ("W",),
    "back": ("S",),
    "strafe-left": ("A",),
    "strafe-right": ("D",),
    "tilt-down": ("I",),
    "tilt-up": ("K",),
    "pan-left": ("J",),
    "pan-right": ("L",),
    "pan-left-fast": ("J", "F"),
    "pan-right-fast": ("L", "F"),
    "fast": ("F",),
}

# One entry: a time span and the words after it.
_ENTRY = re.compile(r"^\s*([0-9]+(?:\.[0-9]*)?|\.[0-9]+)\s*-\s*([0-9]+(?:\.[0-9]*)?|\.[0-9]+)\s*:?\s*(.*?)\s*$")

# What separates two entries, and two words.
_ENTRIES = re.compile(r"[;\n\r]+")
_WORDS = re.compile(r"[\s,+]+")


@dataclasses.dataclass(frozen=True)
class Entry:
    """One span of held keys.

    Attributes:
        start: Seconds the keys go down at.
        end: Seconds they come up at.
        keys: The held keys, a subset of :data:`KEYS`.
    """

    start: float
    end: float
    keys: frozenset


def _purify(on: dict, pairs) -> None:
    """Release both keys of every opposing pair that is held together."""
    for a, b in pairs:
        if on[a] and on[b]:
            on[a] = on[b] = False


def _motion_clause(on: dict) -> str:
    """The movement half of a sentence."""
    _purify(on, (("W", "S"), ("A", "D")))
    words = [MOTION[name] for name in MOTION_ORDER if on[name]]
    return " and ".join(words) if words else MOTION_IDLE


def _camera_clause(on: dict, moving: bool) -> str:
    """The camera half of a sentence."""
    _purify(on, (("J", "L"), ("I", "K")))
    parts = []
    for key, side in PAN_KEY.items():
        if on[key]:
            parts.append(f"pans {side} {'sharply' if on['F'] else 'slowly'}")
    for key, word in TILT_KEY.items():
        if on[key]:
            parts.append(word)
    if parts:
        return " and ".join(parts)
    return CAMERA_FOLLOW if moving else CAMERA_IDLE


def sentence(keys) -> str:
    """The sentence one latent's held keys read as.

    Args:
        keys: Any collection of names from :data:`KEYS`.

    Returns:
        ``the man <movement>, camera <camera>``.
    """
    held = set(keys)
    on = {name: name in held for name in KEYS}
    motion = _motion_clause(dict(on))
    camera = _camera_clause(dict(on), motion != MOTION_IDLE)
    return f"the man {motion}, camera {camera}"


def _keys_of(words: str, where: str) -> frozenset:
    """The keys an entry's words hold.

    Raises:
        ValueError: A word is neither an action name nor a key letter.
    """
    held = set()
    for word in _WORDS.split(words.strip().lower()):
        if not word:
            continue
        if word in PRESETS:
            held.update(PRESETS[word])
        elif len(word) == 1 and word.upper() in KEYS:
            held.add(word.upper())
        else:
            raise ValueError(
                f"H3 World Actions does not know {word!r} in {where}. Use the action names "
                f"{', '.join(PRESETS)} or the keys {' '.join(KEYS)}.")
    return frozenset(held)


def parse(text: str) -> list[Entry]:
    """Read an action script.

    Args:
        text: Entries like ``0-2.5: forward; 1-2: pan-left fast``, one per line or split by
            ``;``. Overlapping entries hold their keys together.

    Returns:
        The entries in the order written.

    Raises:
        ValueError: An entry is malformed, runs backwards or past :data:`MAX_SECONDS`, or there are
            more than :data:`MAX_ENTRIES`.
    """
    entries = []
    for part in _ENTRIES.split(str(text or "")):
        if not part.strip():
            continue
        if len(entries) >= MAX_ENTRIES:
            raise ValueError(f"H3 World Actions takes at most {MAX_ENTRIES} entries.")
        found = _ENTRY.match(part)
        if found is None:
            raise ValueError(
                f"H3 World Actions cannot read {part.strip()!r}. Write each entry as "
                f"`start-end: actions` in seconds, as `0-2.5: forward, pan-left`.")
        start, end = float(found.group(1)), float(found.group(2))
        if not (math.isfinite(start) and math.isfinite(end)) or end <= start or end > MAX_SECONDS:
            raise ValueError(
                f"H3 World Actions needs `{part.strip()}` to end after it starts and by "
                f"{MAX_SECONDS:g} seconds.")
        entries.append(Entry(start, end, _keys_of(found.group(3), repr(part.strip()))))
    return entries


def frame_span(k: int) -> tuple[int, int]:
    """Pixel frames ``[start, stop)`` video latent ``k`` covers."""
    group, place = divmod(int(k), len(FRAME_PER_TOKEN))
    per_group = sum(FRAME_PER_TOKEN)
    start = group * per_group + sum(FRAME_PER_TOKEN[:place])
    return start, start + FRAME_PER_TOKEN[place]


def latent_keys(entries, latents: int) -> list[frozenset]:
    """The keys held in each video latent: every key held on any of its frames.

    Args:
        entries: What :func:`parse` answered.
        latents: How many latents to answer for.

    Returns:
        One set of keys per latent.
    """
    out = []
    for k in range(int(latents)):
        start, stop = frame_span(k)
        held = set()
        for frame in range(start, stop):
            t = frame / FPS
            for entry in entries:
                if entry.start <= t < entry.end:
                    held |= entry.keys
        out.append(frozenset(held))
    return out


def latents_covering(entries) -> int:
    """How many latents it takes to reach the end of the last entry."""
    last = max((entry.end for entry in entries), default=0.0)
    frames = math.ceil(last * FPS - 1e-9)
    k = 0
    while frame_span(k)[1] < frames:
        k += 1
    return k + 1 if frames > 0 else 0


@dataclasses.dataclass(frozen=True)
class Script:
    """An action script with every sentence it can need already encoded.

    Attributes:
        sentences: One sentence per latent the script reaches.
        tail: The sentence of every latent past those.
        encoded: Each distinct sentence's text encoder states, ``[1, tokens, 5120]`` on the CPU.
        node_id: The Actions node's graph id, for its report.
        text: The script as written.
    """

    sentences: tuple
    tail: str
    encoded: dict
    node_id: str | None
    text: str

    def per_latent(self, latents: int) -> list[str]:
        """The sentence of each of ``latents`` video latents."""
        held = list(self.sentences[:latents])
        return held + [self.tail] * (int(latents) - len(held))


def script(clip, text: str, null: bool = False, node_id=None) -> Script:
    """Encode an action script's sentences, each one on its own.

    Args:
        clip: The MiniMax H3 text encoder.
        text: The script, as :func:`parse` reads it.
        null: Every latent reads :data:`NULL_SENTENCE`, for a guidance negative.
        node_id: The Actions node's graph id.

    Returns:
        The script.
    """
    entries = [] if null else parse(text)
    sentences = tuple(sentence(keys) for keys in latent_keys(entries, latents_covering(entries)))
    tail = NULL_SENTENCE
    encoded = {}
    for line in dict.fromkeys(sentences + (tail,)):
        out = clip.encode_from_tokens(clip.tokenize(line), return_dict=True)
        encoded[line] = out["cond"].detach().to("cpu")
    return Script(sentences, tail, encoded, None if node_id is None else str(node_id), str(text or ""))


def describe(sentences) -> list[str]:
    """Each run of latents reading one sentence, as ``latents 0-12: <sentence>``."""
    rows, start = [], 0
    for k in range(1, len(sentences) + 1):
        if k == len(sentences) or sentences[k] != sentences[start]:
            span = f"{start}" if k - 1 == start else f"{start}-{k - 1}"
            rows.append(f"latents {span}: {sentences[start]}")
            start = k
    return rows


@dataclasses.dataclass(frozen=True)
class Routing:
    """Where the action sentences and the video latents sit in one packed sequence.

    Attributes:
        spans: Rows ``[lo, hi)`` of each latent's sentence.
        head_end: First row of the first sentence.
        text_end: Row after the last sentence.
        video_start: First video row.
        frame_rows: Rows in one video latent.
        latents: Video latents.
    """

    spans: tuple
    head_end: int
    text_end: int
    video_start: int
    frame_rows: int
    latents: int


def route(model, cond: dict) -> dict:
    """One processed conditioning with its action sentences appended and bound.

    Args:
        model: The sampling ``BaseModel``, with ``latent_shapes`` set.
        cond: One processed conditioning carrying :data:`COND_KEY`.

    Returns:
        A copy whose text, token tags and packed layout carry the sentences.

    Raises:
        ValueError: The latent is not MiniMax H3 audio and video, the conditioning carries
            references, or the sentences do not fit behind the prompt.
    """
    import comfy.conds
    import comfy.ldm.minimax.model as h3
    import torch

    world = cond[COND_KEY]
    shapes = getattr(model, "latent_shapes", None)
    if not shapes or len(shapes) < 2:
        raise ValueError(
            "H3 World samples MiniMax H3 audio and video latents. Use the latent from MiniMax H3 "
            "Image To Video.")
    video, audio = shapes[0], shapes[1]
    latents, lat_h, lat_w = int(video[2]), (int(video[3]) + 1) // 2 * 2, (int(video[4]) + 1) // 2 * 2
    model_conds = dict(cond["model_conds"])
    payload = dict(model_conds["minimax_payload"].cond)
    if payload.get("refs"):
        raise ValueError(
            "H3 World drives first-frame generations, and this conditioning carries references. "
            "Build it with MiniMax H3 Image To Video and a first frame.")

    head = model_conds["c_crossattn"].cond
    sentences = world.per_latent(latents)
    diffusion = model.diffusion_model
    refined = {}
    for line in dict.fromkeys(sentences):
        states = world.encoded[line].to(device=head.device, dtype=model.get_dtype_inference())
        refined[line] = diffusion.preprocess_text_embeds(states).to(head.dtype)

    spans, cursor = [], int(head.shape[1])
    for line in sentences:
        n = int(refined[line].shape[1])
        spans.append((cursor, cursor + n))
        cursor += n
    text_len = cursor
    text = torch.cat([head] + [refined[line] for line in sentences], dim=1)

    tags = payload.get("text_token_tags")
    head_tags = (torch.ones(int(head.shape[1]), dtype=torch.long) if tags is None
                 else tags.view(-1).to("cpu", torch.long))
    payload["text_token_tags"] = torch.cat(
        [head_tags, torch.ones(text_len - int(head.shape[1]), dtype=torch.long)])

    layout = h3.PackedLayout(text_len, latents, lat_h, lat_w, int(audio[-1]),
                             keyframes=payload.get("keyframes"), refs=None)
    grid = h3._video_t_grid(latents, 0.0)
    origin = float(text_len) - float(grid[-1]) - 1.0
    if origin < int(head.shape[1]):
        raise ValueError(
            f"H3 World's {latents} action sentences take {text_len - int(head.shape[1])} tokens, "
            f"fewer than the {math.ceil(float(grid[-1]) + 1.0)} their positions need. Shorten the "
            f"clip.")
    positions = layout.position_ids.clone()
    for k, (lo, hi) in enumerate(spans):
        positions[lo:hi, 0] = origin + float(grid[k])
    layout.position_ids = positions
    video_start, video_stop, _ = next(s for s in layout.segments if s[2] == "video")
    layout.was_h3_world = Routing(tuple(spans), int(head.shape[1]), text_len, int(video_start),
                                  (int(video_stop) - int(video_start)) // latents, latents)
    payload["layout"] = layout

    model_conds["c_crossattn"] = comfy.conds.CONDRegular(text)
    model_conds["minimax_payload"] = comfy.conds.CONDConstant(payload)
    out = dict(cond)
    out["model_conds"] = model_conds
    out["was_h3_world_sentences"] = sentences
    return out


def _with_lse(q, k, v, scale):
    """Attention over every key, and the log-sum-exp of each query's scores.

    Args:
        q, k, v: ``[B, H, S, D]`` tensors.
        scale: Score scale.

    Returns:
        ``(out, lse)``, ``out`` in ``q``'s dtype and ``lse`` float32 ``[B, H, Sq]``.
    """
    import torch

    if q.is_cuda:
        try:
            out = torch.ops.aten._scaled_dot_product_cudnn_attention(q, k, v, None, True, 0.0, False, False,
                                                                     scale=scale)
            return out[0], out[1].reshape(out[1].shape[:3]).float()
        except Exception:
            pass
        try:
            out = torch.ops.aten._scaled_dot_product_flash_attention(q, k, v, 0.0, False, False, scale=scale)
            return out[0], out[1][..., :q.shape[2]].float()
        except Exception:
            pass
        try:
            out = torch.ops.aten._scaled_dot_product_efficient_attention(q, k, v, None, True, 0.0, False, scale=scale)
            return out[0], out[1][..., :q.shape[2]].float()
        except Exception:
            pass
    outs, lses = [], []
    for lo in range(0, q.shape[2], 256):
        scores = torch.matmul(q[:, :, lo:lo + 256].float(), k.float().transpose(-1, -2)) * scale
        lse = torch.logsumexp(scores, dim=-1)
        outs.append(torch.matmul(torch.exp(scores - lse.unsqueeze(-1)), v.float()).to(q.dtype))
        lses.append(lse)
    return torch.cat(outs, dim=2), torch.cat(lses, dim=2)


def routed_attention(q, k, v, routing: Routing):
    """Attention where each sentence is read only by itself and its own video latent.

    Args:
        q, k, v: ``[B, H, S, D]``, rope applied.
        routing: Where the rows sit.

    Returns:
        ``[B, S, H * D]``.
    """
    import torch
    import torch.nn.functional as F

    b, heads, s, d = q.shape
    scale = d ** -0.5
    a0, a1, v0, fr, latents = (routing.head_end, routing.text_end, routing.video_start,
                               routing.frame_rows, routing.latents)
    # A head group cut from a larger call is made contiguous once, for the attention kernels.
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = q.new_empty(b, heads, s, d)
    # Keys and values outside the sentence block, read as two views rather than copied.
    before = (k[:, :, :a0], v[:, :, :a0])
    after = (k[:, :, a1:], v[:, :, a1:])

    for lo, hi in ((0, a0), (a1, v0)):
        if hi > lo:
            out[:, :, lo:hi] = _merged(q[:, :, lo:hi], (before, after), scale)

    rows_of, pad, readable = _tables(routing, s, q.device)
    longest = rows_of.shape[1]
    step = max(1, min(latents, CHUNK_ROWS * HEAD_BUDGET // max(1, heads) // max(1, fr)))
    for first in range(0, latents, step):
        last = min(latents, first + step)
        n = last - first
        rows = slice(v0 + first * fr, v0 + last * fr)
        main, main_lse = _merged(q[:, :, rows], (before, after), scale, with_lse=True)
        picked = rows_of[first:last].reshape(-1)
        own_k = k.index_select(2, picked).float().view(b, heads, n, longest, d)
        own_v = v.index_select(2, picked).float().view(b, heads, n, longest, d)
        scores = torch.einsum("bhtfd,bhtnd->bhtfn", q[:, :, rows].float().view(b, heads, n, fr, d), own_k)
        del own_k
        scores.mul_(scale).masked_fill_(pad[first:last].view(1, 1, n, 1, longest), float("-inf"))
        own_lse = torch.logsumexp(scores, dim=-1)
        own = torch.einsum("bhtfn,bhtnd->bhtfd", scores.softmax(dim=-1), own_v)
        del scores, own_v
        main_lse = main_lse.view(b, heads, n, fr)
        top = torch.maximum(main_lse, own_lse)
        w_main = torch.exp(main_lse - top).unsqueeze(-1)
        w_own = torch.exp(own_lse - top).unsqueeze(-1)
        own.mul_(w_own).add_(main.view(b, heads, n, fr, d).float().mul_(w_main)).div_(w_main + w_own)
        out[:, :, rows] = own.view(b, heads, n * fr, d).to(q.dtype)

    # Sentence rows: one masked call, each reading the shared rows, itself and its own latent.
    out[:, :, a0:a1] = F.scaled_dot_product_attention(q[:, :, a0:a1], k, v, attn_mask=readable, scale=scale)

    return out.transpose(1, 2).reshape(b, s, heads * d)


def _tables(routing: Routing, tokens: int, device):
    """Index and mask tables for one routing, made once per device and sequence length.

    Args:
        routing: Where the rows sit.
        tokens: Rows in the packed sequence.
        device: The device the tables live on.

    Returns:
        ``(rows, pad, readable)``: ``[latents, longest]`` sentence row indices, padding repeating
        a span's first row; the matching padding mask; and the ``[sentence rows, tokens]`` mask
        of what each sentence row reads.
    """
    import torch

    key = (str(device), int(tokens))
    held = routing.__dict__.get("_was_tables")
    if held is not None and held[0] == key:
        return held[1]
    a0, a1, v0, fr = routing.head_end, routing.text_end, routing.video_start, routing.frame_rows
    longest = max(hi - lo for lo, hi in routing.spans)
    with _outside_graph():
        rows = torch.tensor([[lo + min(i, hi - lo - 1) for i in range(longest)] for lo, hi in routing.spans],
                            dtype=torch.long, device=device)
        pad = torch.tensor([[i >= hi - lo for i in range(longest)] for lo, hi in routing.spans],
                           dtype=torch.bool, device=device)
        readable = torch.zeros(a1 - a0, tokens, dtype=torch.bool, device=device)
        readable[:, :a0] = True
        readable[:, a1:v0] = True
        for i, (lo, hi) in enumerate(routing.spans):
            readable[lo - a0:hi - a0, lo:hi] = True
            readable[lo - a0:hi - a0, v0 + i * fr:v0 + (i + 1) * fr] = True
    tables = (rows, pad, readable)
    object.__setattr__(routing, "_was_tables", (key, tables))
    return tables


def _outside_graph():
    """A context whose allocations stay out of ComfyUI's per-block allocation record."""
    import contextlib

    try:
        import comfy.model_prefetch as prefetch
    except ImportError:
        return contextlib.nullcontext()
    pause = getattr(prefetch, "pause_malloc_graph", None)
    return pause() if pause is not None else contextlib.nullcontext()


def _merged(q, parts, scale, with_lse: bool = False):
    """Attention over several key ranges at once, merged exactly through their log-sum-exps.

    Args:
        q: ``[B, H, Sq, D]`` queries.
        parts: ``(keys, values)`` pairs, each ``[B, H, Sk, D]``; empty ranges are skipped.
        scale: Score scale.
        with_lse: Also return the merged log-sum-exp.

    Returns:
        The output in ``q``'s dtype, or ``(output, lse)`` with ``lse`` float32 ``[B, H, Sq]``.
    """
    import torch

    found = [_with_lse(q, keys, values, scale) for keys, values in parts if keys.shape[2]]
    if len(found) == 1:
        out, lse = found[0]
        return (out, lse) if with_lse else out
    lse = torch.logsumexp(torch.stack([part_lse for _, part_lse in found]), dim=0)
    out = None
    for part, part_lse in found:
        term = part.float().mul_(torch.exp(part_lse - lse).unsqueeze(-1))
        out = term if out is None else out.add_(term)
    out = out.to(q.dtype)
    return (out, lse) if with_lse else out


def attention_override(previous=None):
    """An ``optimized_attention_override`` routing the packed calls that carry sentences.

    Args:
        previous: The override already on the model, called for every other call.

    Returns:
        The override.
    """

    def override(func, *args, **kwargs):
        options = kwargs.get("transformer_options") or {}
        layout = options.get("minimax_h3_layout")
        routing = getattr(layout, "was_h3_world", None)
        q = args[0] if args else None
        if (routing is not None and kwargs.get("skip_reshape") and kwargs.get("mask") is None
                and getattr(q, "dim", lambda: 0)() == 4 and q.shape[2] == layout.seq_len):
            return routed_attention(args[0], args[1], args[2], routing)
        if previous is not None:
            return previous(func, *args, **kwargs)
        return func(*args, **kwargs)

    return override


def _report(node_id, sentences, latents: int) -> None:
    """Publish the sentences each latent read to the Actions node."""
    if node_id is None:
        return
    try:
        from ..interface import run_result

        distinct = len(set(sentences))
        run_result.publish(
            summary=f"{latents} latents read {distinct} sentence{'s' if distinct != 1 else ''}",
            counts={"latents": latents, "sentences": distinct,
                    "seconds": round(frame_span(latents - 1)[1] / FPS, 2)},
            items=describe(sentences),
            node_id=node_id,
        )
    except Exception as error:
        logger.debug("the action report was not published (%s)", error)


def _sampler_wrapper(executor, guider, *args, **kwargs):
    """Bind each conditioning's action sentences before the sampler runs."""
    model = guider.inner_model
    options = getattr(guider, "model_options", {}).get("transformer_options", {})
    replaced = sorted(index for kind, index in options.get("patches_replace", {}).get("dit", {}) if kind == "double_block")
    if replaced:
        logger.warning(
            "H3 World: %d block(s) carry another patch's replacement. One that swaps the attention, such "
            "as H3 Sparse Attention, skips the action routing on those blocks; leave it off this model.",
            len(replaced))
    for name, conds in list(guider.conds.items()):
        if any(c.get(COND_KEY) is not None for c in conds):
            guider.conds[name] = [route(model, c) if c.get(COND_KEY) is not None else c for c in conds]
            for c in guider.conds[name]:
                if "was_h3_world_sentences" in c and name == "positive":
                    _report(c[COND_KEY].node_id, c["was_h3_world_sentences"],
                            len(c["was_h3_world_sentences"]))
    return executor(guider, *args, **kwargs)


def patched(model):
    """A clone of a MiniMax H3 model whose sampling binds action sentences to their latents.

    Args:
        model: A MiniMax H3 ``ModelPatcher``.

    Returns:
        The clone.
    """
    import comfy.patcher_extension

    clone = model.clone()
    options = clone.model_options.setdefault("transformer_options", {})
    previous = options.get("optimized_attention_override")
    options["optimized_attention_override"] = attention_override(previous)
    clone.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE, WRAPPER_KEY,
                               _sampler_wrapper)
    return clone
