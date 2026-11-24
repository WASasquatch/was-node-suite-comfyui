"""Scene-structured attention for MiniMax H3.

Each scene's video reads its own frames, its own shot text, the rows shared by the whole clip and
a few frames from the other scenes. Shots are read as ``[Shot 2] At 00:03.000``.
"""

from __future__ import annotations

import contextlib
import math
import re

from .. import log

__all__ = [
    "FPS", "MAX_SHARED_FRAMES", "POSITIONS", "SHARED_FRAMES", "STASH_KEY", "boundary_frames",
    "build_scopes", "cut_times", "foreign_frames", "latent_index", "scene_attention", "shot_stash",
    "shots_in", "spherical_kmeans", "with_shots",
]

logger = log.get_logger("sampling.h3_scene_attention")

#: Frames per second of every MiniMax H3 clip.
FPS = 24

#: Pixel frames each latent frame of a 5-token H3 clip covers, the first being the intra frame.
FRAME_PER_TOKEN = (1, 4, 4, 4, 4)

#: Conditioning key holding a prompt's timed shots.
STASH_KEY = "was_h3_shots"

#: Frames each scene reads from the other scenes by default.
SHARED_FRAMES = 4

#: Most frames a scene may read from the other scenes.
MAX_SHARED_FRAMES = 16

#: Iterations of the clustering that picks the shared frames.
KMEANS_ITERATIONS = 20

#: How rotary positions run across scenes, the first being the default.
POSITIONS = ("global", "scene")

#: Wrapper key the sampling hook is registered under.
WRAPPER_KEY = "was_h3_scene_attention"

#: A line opening a named prompt section, as ``overall_soundscape:``.
SECTION_LINE = re.compile(r"^[a-z][a-z0-9_]*:(?:[ \t]|$)", re.M)

#: Kinds of packed segment every query and key shares, beside the text outside the shots.
SHARED_KINDS = ("cond", "cond_audio", "ref_img", "ref_audio", "audio")


def _outside_graph():
    """A context whose allocations stay out of ComfyUI's per-block allocation record."""
    try:
        import comfy.model_prefetch as prefetch
    except ImportError:
        return contextlib.nullcontext()
    pause = getattr(prefetch, "pause_malloc_graph", None)
    return pause() if pause is not None else contextlib.nullcontext()


def shots_in(text: str) -> list[tuple[float, int, int]]:
    """The scenes a prompt's shots open, as character spans.

    Args:
        text: An H3 prompt.

    Returns:
        ``[(start_seconds, first_char, end_char)]`` in order. A later shot with no ``At`` time
        joins the scene before it, as does one timed at or before that scene's start. Empty
        where the prompt has no shot heading.
    """
    from ..latent.h3_conditioning import SHOT_HEADING, SHOT_SECTION, SHOT_TIME

    text = str(text or "")
    section = SHOT_SECTION.search(text)
    headings = list(SHOT_HEADING.finditer(text, section.end() if section else 0))
    scenes: list[list] = []
    for number, heading in enumerate(headings):
        stop = headings[number + 1].start() if number + 1 < len(headings) else len(text)
        closing = SECTION_LINE.search(text, heading.end(), stop)
        if closing is not None:
            stop = closing.start()
        timed = SHOT_TIME.search(text, heading.end(), stop)
        seconds = None
        if timed is not None and not text[heading.end():timed.start()].strip():
            minutes, whole, millis = (int(part) for part in timed.groups())
            seconds = minutes * 60 + whole + millis / 1000.0
        if not scenes:
            scenes.append([0.0 if seconds is None else seconds, heading.start(), stop])
        elif seconds is None or seconds <= scenes[-1][0]:
            scenes[-1][2] = stop
        else:
            scenes.append([seconds, heading.start(), stop])
    if scenes:
        scenes[0][0] = 0.0
    return [(float(a), int(b), int(c)) for a, b, c in scenes]


def shot_stash(clip, tokens, text: str) -> dict | None:
    """The timed shots of an encoded prompt, for :data:`STASH_KEY`.

    Args:
        clip: The minimax CLIP the prompt was tokenized with.
        tokens: What ``clip.tokenize`` returned for it.
        text: The prompt text, which the tokenizer placed last.

    Returns:
        ``{"tokens": n, "shots": [(start_seconds, lo, hi)]}`` with ``lo`` and ``hi`` counted back
        from the end of the text segment, or None where the prompt has fewer than two scenes or
        its tokens cannot be matched to its text.
    """
    scenes = shots_in(text)
    if len(scenes) < 2:
        return None
    try:
        holder = getattr(clip, "tokenizer", None)
        name = getattr(holder, "clip", None)
        inner = getattr(holder, name, None) if isinstance(name, str) else None
        reader = getattr(inner, "tokenizer", None)
        if reader is None or not isinstance(tokens, dict) or name not in tokens:
            return None
        encoded = reader(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = list(encoded["input_ids"])
        offsets = list(encoded["offset_mapping"])
        entries = tokens[name][0] if tokens[name] else []
        count = len(ids)
        tail = [entry[0] for entry in entries[-count:]] if count else []
        if not count or len(entries) < count or tail != ids:
            return None
    except Exception as error:
        logger.debug("the prompt's shots could not be matched to its tokens (%s)", error)
        return None

    def first_token(char: int) -> int:
        for index, (_, end) in enumerate(offsets):
            if end > char:
                return index
        return count

    shots = []
    for seconds, start, stop in scenes:
        lo, hi = first_token(start), first_token(stop)
        if hi > lo:
            shots.append((float(seconds), lo - count, hi - count))
    return {"tokens": count, "shots": shots} if len(shots) >= 2 else None


def with_shots(conditioning, clip, text: str):
    """A conditioning carrying its prompt's timed shots under :data:`STASH_KEY`.

    Args:
        conditioning: The prompt as the text encoder encoded it.
        clip: The CLIP it was encoded with.
        text: The prompt text.

    Returns:
        The conditioning with the shots added, or as it came where the prompt has fewer than two
        timed shots or its tokens cannot be matched.
    """
    if len(shots_in(text)) < 2:
        return conditioning
    try:
        stash = shot_stash(clip, clip.tokenize(text), text)
    except Exception as error:
        logger.debug("the prompt's shots were not recorded (%s)", error)
        return conditioning
    if stash is None:
        return conditioning
    import node_helpers

    return node_helpers.conditioning_set_values(conditioning, {STASH_KEY: stash})


def cut_times(text: str) -> list[float]:
    """Cut times typed as seconds, as ``3.5, 7``.

    Args:
        text: Numbers separated by commas, spaces or semicolons.

    Returns:
        The positive times, sorted and without repeats.

    Raises:
        ValueError: A part is not a number.
    """
    found = set()
    for part in re.split(r"[,\s;]+", str(text or "").strip()):
        if not part:
            continue
        try:
            value = float(part)
        except ValueError:
            raise ValueError(
                f"H3 Scene Attention reads scenes as cut times in seconds, as `3.5, 7`, and "
                f"`{part}` is not a number. Leave scenes empty to read the prompt's shots.") from None
        if not math.isfinite(value):
            raise ValueError(f"H3 Scene Attention cannot cut at {part}. Type cut times in seconds.")
        if value > 0:
            found.add(value)
    return sorted(found)


def latent_index(seconds: float, latent_frames: int) -> int:
    """The latent frame a time in the clip falls on.

    Args:
        seconds: From the start of the clip.
        latent_frames: Latent frames in the clip.

    Returns:
        The latent frame whose first pixel frame is nearest the time, earlier on a tie.
    """
    target = round(float(seconds) * FPS)
    best, best_gap, start = 0, None, 0
    for index in range(int(latent_frames)):
        gap = abs(start - target)
        if best_gap is None or gap < best_gap:
            best, best_gap = index, gap
        start += FRAME_PER_TOKEN[index % len(FRAME_PER_TOKEN)]
    return best


class Scopes:
    """Which rows every query group of one packed H3 sequence reads.

    Attributes:
        scenes: ``(first, end)`` latent frames of each scene.
        video: ``(start, stop)`` rows of the target video.
        frame_rows: Rows in one latent frame.
        texts: ``(start, stop)`` rows of each scene's own shot text, or None.
        shared: Rows every query reads, as a long tensor.
        keys: Rows each scene's video reads before its shared frames, one long tensor each.
        text_keys: Rows each scene's text reads, one long tensor each, or None.
        foreign: Frames each scene reads from the others.
        seconds: Start of each scene in seconds.
        tokens: Rows in the sequence.
    """

    def __init__(self, scenes, video, frame_rows, texts, shared, keys, text_keys, foreign, seconds,
                 tokens):
        self.scenes = scenes
        self.video = video
        self.frame_rows = frame_rows
        self.texts = texts
        self.shared = shared
        self.keys = keys
        self.text_keys = text_keys
        self.foreign = foreign
        self.seconds = seconds
        self.tokens = tokens

    def work(self) -> float:
        """Query-key pairs these scopes compute, as a share of full attention."""
        total = float(self.tokens) ** 2
        done = float(self.shared.numel()) * self.tokens
        for index, (first, end) in enumerate(self.scenes):
            done += (end - first) * self.frame_rows * (self.keys[index].numel()
                                                      + self.foreign[index] * self.frame_rows)
            if self.texts[index] is not None:
                start, stop = self.texts[index]
                done += (stop - start) * self.text_keys[index].numel()
        return done / total if total else 1.0


def build_scopes(layout, cuts: list[float], shots, shared_frames: int, device) -> Scopes | None:
    """The query groups of one packed sequence.

    Args:
        layout: The H3 ``PackedLayout`` of the call.
        cuts: Scene starts in seconds after the first, from the prompt or typed.
        shots: ``[(start_seconds, lo, hi)]`` text spans counted back from the end of the text
            segment, or None where the prompt's text is shared by every scene.
        shared_frames: Frames each scene reads from the others.
        device: Where the row tensors are made.

    Returns:
        The scopes, or None where the clip holds fewer than two scenes.
    """
    import torch

    segments = list(getattr(layout, "segments", ()))
    video = next(((a, b) for a, b, kind in segments if kind == "video"), None)
    signature = getattr(layout, "signature", None)
    if video is None or not signature:
        return None
    latent_frames = int(signature[1])
    frame_rows = (video[1] - video[0]) // max(1, latent_frames)
    if latent_frames < 2 or frame_rows * latent_frames != video[1] - video[0]:
        return None
    length = sum(FRAME_PER_TOKEN[k % len(FRAME_PER_TOKEN)] for k in range(latent_frames))
    inside = [seconds for seconds in cuts if round(float(seconds) * FPS) < length]
    starts = sorted({latent_index(seconds, latent_frames) for seconds in inside} - {0})
    starts = [start for start in starts if 0 < start < latent_frames]
    if not starts:
        return None
    edges = [0] + starts + [latent_frames]
    scenes = [(edges[index], edges[index + 1]) for index in range(len(edges) - 1)]
    seconds = [0.0] + [sum(FRAME_PER_TOKEN[k % 5] for k in range(start)) / FPS for start in starts]

    text_len = next((b - a for a, b, kind in segments if kind == "text"), 0)
    texts = [None] * len(scenes)
    if shots:
        for start_seconds, lo, hi in shots:
            frame = latent_index(start_seconds, latent_frames)
            owner = max(index for index, (first, _) in enumerate(scenes) if first <= frame)
            span = (text_len + int(lo), text_len + int(hi))
            if span[0] < 0 or span[1] > text_len or span[1] <= span[0]:
                continue
            held = texts[owner]
            texts[owner] = span if held is None else (min(held[0], span[0]), max(held[1], span[1]))

    owned = torch.zeros(int(layout.seq_len), dtype=torch.bool)
    owned[video[0]:video[1]] = True
    for span in texts:
        if span is not None:
            owned[span[0]:span[1]] = True
    shared = torch.nonzero(~owned).flatten()

    keys, text_keys = [], []
    for index, (first, end) in enumerate(scenes):
        rows = torch.arange(video[0] + first * frame_rows, video[0] + end * frame_rows)
        span = texts[index]
        own_text = torch.arange(span[0], span[1]) if span is not None else torch.zeros(0, dtype=torch.long)
        keys.append(torch.cat([shared, own_text, rows]).to(device))
        text_keys.append(keys[-1] if span is not None else None)
    foreign = [min(int(shared_frames), latent_frames - (end - first)) for first, end in scenes]
    return Scopes(scenes, video, frame_rows, texts, shared.to(device), keys, text_keys, foreign,
                seconds, int(layout.seq_len))


def boundary_frames(scenes, latent_frames: int, count: int) -> list[list[int]]:
    """Each scene's shared frames before any step has been read: the frames nearest its edges.

    Args:
        scenes: ``(first, end)`` latent frames of each scene.
        latent_frames: Latent frames in the clip.
        count: Frames per scene.

    Returns:
        For each scene, ``count`` frames from the other scenes, nearest its first and last frame,
        the earlier frame first on a tie.
    """
    picked = []
    for first, end in scenes:
        outside = [frame for frame in range(latent_frames) if frame < first or frame >= end]
        gap = [(first - frame if frame < first else frame - end + 1, frame) for frame in outside]
        picked.append(sorted(frame for _, frame in sorted(gap)[:count]))
    return picked


def spherical_kmeans(features, clusters: int, iterations: int = KMEANS_ITERATIONS):
    """Cluster unit vectors by cosine similarity.

    Args:
        features: ``(n, d)`` float tensor of unit rows.
        clusters: Clusters wanted, at most ``n``.
        iterations: Most assignment rounds.

    Returns:
        ``(clusters, d)`` unit centroids, seeded by farthest-point picks from row 0.
    """
    import torch

    count = int(features.shape[0])
    clusters = max(1, min(int(clusters), count))
    chosen = [0]
    nearest = features @ features[0]
    for _ in range(1, clusters):
        candidate = int(torch.argmin(nearest))
        chosen.append(candidate)
        nearest = torch.maximum(nearest, features @ features[candidate])
    centroids = features[chosen].clone()
    for _ in range(int(iterations)):
        assigned = torch.argmax(features @ centroids.T, dim=1)
        updated = torch.zeros_like(centroids).index_add_(0, assigned, features)
        empty = updated.norm(dim=1) == 0
        updated[empty] = centroids[empty]
        updated = torch.nn.functional.normalize(updated, dim=1)
        if torch.equal(updated, centroids):
            break
        centroids = updated
    return centroids


def foreign_frames(features, centroids, scenes, counts) -> list[list[int]]:
    """Each scene's shared frames: per cluster, its nearest frame outside the scene.

    Args:
        features: ``(frames, d)`` unit frame features.
        centroids: ``(clusters, d)`` unit centroids.
        scenes: ``(first, end)`` latent frames of each scene.
        counts: Frames each scene takes.

    Returns:
        For each scene, ``counts[i]`` distinct frames from the other scenes, in order.
    """
    import torch

    similarity = features @ centroids.T
    frames = int(features.shape[0])
    picked = []
    for (first, end), count in zip(scenes, counts):
        mask = torch.ones(frames, dtype=torch.bool, device=features.device)
        mask[first:end] = False
        chosen: list[int] = []
        rounds = 0
        while len(chosen) < count and rounds <= count:
            rounds += 1
            for cluster in range(similarity.shape[1]):
                if len(chosen) >= count:
                    break
                scores = similarity[:, cluster].masked_fill(~mask, -math.inf)
                best = int(torch.argmax(scores))
                if not bool(mask[best]):
                    break
                chosen.append(best)
                mask[best] = False
        picked.append(sorted(chosen))
    return picked


class _State:
    """What every patched block of one model shares during a sampling run."""

    def __init__(self, network, cuts, positions, shared_frames, select_block, high, low, node_id):
        self.network = network
        self.cuts = list(cuts)
        self.positions = positions
        self.shared_frames = int(shared_frames)
        self.select_block = int(select_block)
        self.high = high
        self.low = low
        self.node_id = node_id
        self.reset()

    def reset(self):
        """Forget the run's stashes, scopes and shared frames."""
        self.stashes: dict = {}
        self.timing: list[float] | None = None
        self.scopes: dict = {}
        self.rows: dict = {}
        self.ropes: dict = {}
        self.current = None
        self.pending = None
        self.sigma = None

    def begin(self, guider):
        """Read the shots every prompt of the run carries."""
        self.reset()
        conds = getattr(guider, "conds", None) or {}
        for name, items in conds.items():
            for cond in items or []:
                stash = cond.get(STASH_KEY) if isinstance(cond, dict) else None
                if isinstance(stash, dict) and stash.get("shots"):
                    self.stashes[cond.get("uuid")] = stash
                    if self.timing is None and name == "positive":
                        self.timing = [shot[0] for shot in stash["shots"][1:]]
        if self.timing is None and self.stashes:
            first = next(iter(self.stashes.values()))
            self.timing = [shot[0] for shot in first["shots"][1:]]

    def cut_list(self) -> list[float]:
        """Scene starts after the first, typed or read from the prompt."""
        return self.cuts if self.cuts else list(self.timing or [])

    def active(self, options) -> bool:
        """Whether a call's sigma lies between start_percent and end_percent."""
        sigmas = (options or {}).get("sigmas")
        if sigmas is None:
            return True
        sigma = float(sigmas.flatten()[0]) if hasattr(sigmas, "flatten") else float(sigmas[0])
        return self.low <= sigma <= self.high

    def step(self, options):
        """Promote the frames picked during the last step when a new step starts."""
        sigmas = (options or {}).get("sigmas")
        if sigmas is None:
            return
        sigma = float(sigmas.flatten()[0]) if hasattr(sigmas, "flatten") else float(sigmas[0])
        if sigma != self.sigma:
            self.sigma = sigma
            if self.pending is not None:
                self.current, self.pending = self.pending, None

    def scopes_of(self, layout, uuid, device) -> Scopes | None:
        """The scopes of one call, built once per layout and prompt."""
        stash = self.stashes.get(uuid)
        cuts = self.cut_list()
        key = (tuple(getattr(layout, "signature", ())), int(getattr(layout, "seq_len", 0)),
               str(device), id(stash))
        if key not in self.scopes:
            shots = None
            if stash is not None and stash.get("shots"):
                shots = stash["shots"]
            with _outside_graph():
                self.scopes[key] = build_scopes(layout, cuts, shots, self.shared_frames, device)
        return self.scopes[key]

    def frames_of(self, plan: Scopes) -> list[list[int]]:
        """Each scene's shared frames for this step."""
        latent_frames = plan.scenes[-1][1]
        current = self.current
        if current is not None and current[0] == latent_frames and current[1] == len(plan.scenes):
            return current[2]
        return boundary_frames(plan.scenes, latent_frames, max(plan.foreign))

    def pick(self, plan: Scopes, hidden):
        """Cluster this step's frames and keep each scene's shared frames for the next step."""
        import torch

        with _outside_graph(), torch.no_grad():
            start, stop = plan.video
            latent_frames = plan.scenes[-1][1]
            features = hidden[start:stop].float().view(latent_frames, plan.frame_rows, -1).mean(dim=1)
            features = torch.nn.functional.normalize(features, dim=1)
            clusters = max(1, max(plan.foreign))
            centroids = spherical_kmeans(features, clusters)
            chosen = foreign_frames(features, centroids, plan.scenes, plan.foreign)
        self.pending = (latent_frames, len(plan.scenes), chosen)

    def keys_of(self, plan: Scopes, index: int, frames: list[int]):
        """Rows scene ``index`` reads: its own keys and its shared frames."""
        import torch

        key = (id(plan), index, tuple(frames))
        found = self.rows.get(key)
        if found is None:
            with _outside_graph():
                start = plan.video[0]
                parts = [plan.keys[index]]
                for frame in frames:
                    first = start + frame * plan.frame_rows
                    parts.append(torch.arange(first, first + plan.frame_rows, device=plan.keys[index].device))
                found = torch.cat(parts)
            if len(self.rows) > 64:
                self.rows.clear()
            self.rows[key] = found
        return found

    def rope(self, layout, plan: Scopes, rope_freqs):
        """The rotation table with each scene's frames and sound counted from its own start."""
        if self.positions != "scene" or rope_freqs is None:
            return rope_freqs
        key = (id(plan), rope_freqs.shape, rope_freqs.dtype, str(rope_freqs.device))
        found = self.ropes.get(key)
        if found is not None:
            return found
        import torch
        from comfy.ldm.minimax.model import rope_rotation_table

        with _outside_graph():
            positions = layout.position_ids.clone()
            video_start = plan.video[0]
            origin = float(positions[video_start, 0])
            frame_rows = plan.frame_rows
            bounds = []
            for first, end in plan.scenes:
                offset = float(positions[video_start + first * frame_rows, 0]) - origin
                bounds.append(offset)
                positions[video_start + first * frame_rows:video_start + end * frame_rows, 0] -= offset
            audio = next(((a, b) for a, b, kind in layout.segments if kind == "audio"), None)
            if audio is not None:
                times = positions[audio[0]:audio[1], 0] - origin
                shift = torch.zeros_like(times)
                for offset in bounds:
                    shift = torch.where(times >= offset, torch.full_like(times, offset), shift)
                positions[audio[0]:audio[1], 0] -= shift
            table = rope_rotation_table(self.network.rope_freqs(positions, rope_freqs.device), rope_freqs.dtype)
        self.ropes.clear()
        self.ropes[key] = table
        return table


def _attend(attn, q, k, v, options):
    """One query group's attention through the model's own backend.

    Args:
        attn: The block's attention module, for its preferred backend.
        q: ``(queries, heads, head_dim)``.
        k: ``(keys, heads, head_dim)``.
        v: ``(keys, heads, head_dim)``.
        options: The call's transformer options.

    Returns:
        ``(queries, heads * head_dim)``.
    """
    import comfy.ldm.modules.attention as attention

    out = attention.optimized_attention(
        q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0),
        int(attn.heads), preferred_attention=getattr(attn, "comfy_attention", None), mask=None,
        skip_reshape=True, transformer_options=options)
    return out.reshape(q.shape[0], -1)


def scene_attention(attn, x, rope_freqs, options, plan: Scopes, frames, state: _State, layout):
    """One block's attention with each scene's queries reading their own scope.

    Args:
        attn: The block's attention module.
        x: The normalised block input, ``(tokens, hidden)``.
        rope_freqs: The call's rotation table, or None.
        options: The call's transformer options.
        plan: The call's plan.
        frames: Each scene's shared frames.
        state: The run's state, for the rotation table and the key rows.
        layout: The call's packed layout.

    Returns:
        The attention output after ``out_proj``, ``(tokens, hidden)``.
    """
    import comfy.model_management
    import comfy.quant_ops

    count = x.shape[0]
    heads, width = int(attn.heads), int(attn.head_dim)
    q, k, v = attn.qkv_proj(x).split(heads * width, dim=-1)
    v = v.view(count, heads, width)
    rope = state.rope(layout, plan, rope_freqs)
    if rope is not None:
        q = q.view(1, count, heads, width)
        k = k.view(1, count, heads, width)
        qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
        kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
        turned = rope.shape[-3] * 2
        if comfy.model_management.in_training:
            q, k = comfy.quant_ops.ck.rms_rope_split_half(q, k, rope, qw, kw, epsilon=attn.q_norm.eps,
                                                          rot_dim=turned)
        else:
            comfy.quant_ops.ck.rms_rope_split_half_(q, k, rope, qw, kw, epsilon=attn.q_norm.eps,
                                                    rot_dim=turned)
        q, k = q[0], k[0]
    else:
        q = attn.q_norm(q.view(count, heads, width))
        k = attn.k_norm(k.view(count, heads, width))

    out = q.new_empty((count, heads * width))
    if plan.shared.numel():
        out.index_copy_(0, plan.shared, _attend(attn, q.index_select(0, plan.shared), k, v, options))
    start = plan.video[0]
    for index, (first, end) in enumerate(plan.scenes):
        rows = state.keys_of(plan, index, frames[index])
        low, high = start + first * plan.frame_rows, start + end * plan.frame_rows
        out[low:high] = _attend(attn, q[low:high], k.index_select(0, rows), v.index_select(0, rows), options)
        span = plan.texts[index]
        if span is not None:
            keys = plan.text_keys[index]
            out[span[0]:span[1]] = _attend(attn, q[span[0]:span[1]], k.index_select(0, keys),
                                           v.index_select(0, keys), options)
    return attn.out_proj(out)


class _SceneBlock:
    """A block replacement scoping one H3 block's attention by scene."""

    def __init__(self, state: _State, layer: int, attn, previous=None):
        self.state = state
        self.layer = int(layer)
        self.attn = attn
        self.previous = previous

    def __call__(self, args, extra_args):
        state = self.state
        options = args.get("transformer_options") or {}
        layout = args.get("layout")
        if self.layer == 0:
            state.step(options)
        plan = None
        uuids = options.get("uuids") or [None]
        if layout is not None and getattr(layout, "seq_len", None) == int(args["img"].shape[0]):
            plan = state.scopes_of(layout, uuids[0], args["img"].device)
        if plan is not None:
            conditional = 0 in (options.get("cond_or_uncond") or [0])
            if self.layer == state.select_block and conditional:
                state.pick(plan, args["img"])
            if state.active(options):
                frames = state.frames_of(plan)

                def attention(h, rope_freqs=None, transformer_options={}):
                    return scene_attention(self.attn, h, rope_freqs, transformer_options, plan, frames,
                                           state, layout)

                args = {**args, "attention": attention}
        if self.previous is None:
            return extra_args["original_block"](args)
        return self.previous(args, extra_args)

    def to(self, device_or_dtype):
        """Pass a device or dtype move on to the replacement this one wraps."""
        if hasattr(self.previous, "to"):
            self.previous = self.previous.to(device_or_dtype)
        return self

    def models(self) -> list:
        """Whatever the wrapped replacement loads."""
        return self.previous.models() if hasattr(self.previous, "models") else []


def _conflicts(model_options) -> list[str]:
    """Names of the patches that route the attention this one scopes."""
    options = (model_options or {}).get("transformer_options", {})
    replaces = options.get("patches_replace", {}).get("dit", {})
    found = set()
    override = options.get("optimized_attention_override")
    if str(getattr(override, "__module__", "")).endswith(".h3_world"):
        found.add("H3-World action routing")
    for patch in replaces.values():
        seen = 0
        while patch is not None and seen < 64:
            if type(patch).__name__ == "_SparseBlock":
                found.add("H3 Sparse Attention")
            patch = getattr(patch, "previous", None)
            seen += 1
    return sorted(found)


def _publish(state: _State):
    """Report what the run's scenes were, on the node's own panel."""
    try:
        from ..interface import run_result

        if not run_result.watching() or state.node_id is None:
            return
        plan = next((found for found in state.scopes.values() if found is not None), None)
        if plan is None:
            source = "typed cut times" if state.cuts else "the prompt"
            run_result.publish(
                status=run_result.WARNING, node_id=state.node_id,
                summary=(f"one scene found in {source}: every block attended as the model does. "
                         "Write timed shots in the prompt, as `[Shot 2] At 00:03.000, ...`, or "
                         "type cut times in scenes."),
                facts={"scenes from": source})
            return
        tallies = []
        for index, (first, end) in enumerate(plan.scenes):
            tallies.append({"name": f"scene {index + 1} from {plan.seconds[index]:.3f} s",
                            "value": (end - first) * plan.frame_rows})
        scoped = sum(1 for span in plan.texts if span is not None)
        run_result.publish(
            status=run_result.OK, node_id=state.node_id,
            summary=(f"{len(plan.scenes)} scenes, attention work {plan.work() * 100:.0f}% of "
                     f"full attention"),
            counts={"scenes": len(plan.scenes), "tokens": plan.tokens,
                    "shared frames": max(plan.foreign), "frame tokens": plan.frame_rows},
            tallies=tallies,
            facts={"scenes from": "typed cut times" if state.cuts else "the prompt's shots",
                   "shot text": (f"{scoped} of {len(plan.scenes)} scenes read their own shot"
                                 if scoped else "shared by every scene"),
                   "positions": state.positions},
        )
    except Exception as error:
        logger.debug("no scene report was published (%s)", error)


def patched(model, cuts, positions: str = POSITIONS[0], shared_frames: int = SHARED_FRAMES,
            select_block: int | None = None, start_percent: float = 0.0, end_percent: float = 1.0,
            node_id=None):
    """A clone of a MiniMax H3 model whose blocks attend scene by scene.

    Args:
        model: A MiniMax H3 model patcher.
        cuts: Scene starts in seconds after the first, or empty to read the prompt's shots.
        positions: One of :data:`POSITIONS`.
        shared_frames: Frames each scene reads from the others.
        select_block: The block whose input picks the shared frames, the middle one when None.
        start_percent: Point in the schedule scene attention starts.
        end_percent: Point in the schedule it stops.
        node_id: The graph id the run's report is filed under.

    Returns:
        The patched clone.

    Raises:
        ValueError: The model is not MiniMax H3, or a setting is out of range.
    """
    import comfy.patcher_extension

    network = model.get_model_object("diffusion_model")
    blocks = getattr(network, "blocks", None)
    if not any(cls.__name__ == "MiniMaxH3Model" for cls in type(network).__mro__) or blocks is None:
        raise ValueError(
            f"H3 Scene Attention patches MiniMax H3 video models, and this one is "
            f"{type(network).__name__}. Wire the MiniMax H3 model from its loader or LoRA.")
    if positions not in POSITIONS:
        raise ValueError(f"H3 Scene Attention has no positions `{positions}`; pick one of {', '.join(POSITIONS)}.")
    layers = len(blocks)
    select = layers // 2 if select_block is None else int(select_block)
    if not 0 <= select < layers:
        raise ValueError(f"select_block {select} is outside this model's blocks, 0 to {layers - 1}.")
    sampling = model.get_model_object("model_sampling")
    high = math.inf if start_percent <= 0.0 else float(sampling.percent_to_sigma(start_percent))
    low = -math.inf if end_percent >= 1.0 else float(sampling.percent_to_sigma(end_percent))
    state = _State(network, cuts, positions, max(0, min(int(shared_frames), MAX_SHARED_FRAMES)),
                   select, high, low, node_id)
    clone = model.clone()
    for layer, block in enumerate(blocks):
        options = clone.model_options.get("transformer_options", {})
        previous = options.get("patches_replace", {}).get("dit", {}).get(("double_block", layer))
        clone.set_model_patch_replace(_SceneBlock(state, layer, block.attn, previous), "dit",
                                      "double_block", layer)

    def sample(executor, *args, **kwargs):
        guider = args[0] if args else None
        conflicts = _conflicts(getattr(guider, "model_options", None) or {})
        if conflicts:
            raise ValueError(
                f"H3 Scene Attention and {', '.join(conflicts)} both decide what each block's "
                f"attention reads, so only one of them would take effect. Remove one of the two "
                f"from the model chain.")
        state.begin(guider)
        try:
            return executor(*args, **kwargs)
        finally:
            _publish(state)
            plan = next((found for found in state.scopes.values() if found is not None), None)
            if plan is not None:
                logger.info("H3 Scene Attention: %d scenes, attention work %.0f%% of full",
                            len(plan.scenes), plan.work() * 100)

    clone.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE, WRAPPER_KEY, sample)
    return clone
