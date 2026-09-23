"""Encoding a MiniMax H3 task prompt, and one prompt per extension pass.

Row 1 opens the clip and every row after it extends it. Each row carries its own frame
count, and the loop runs once per row.
"""

from __future__ import annotations

import torch

from ..image import resolution
from . import h3_extend

#: The task each mode conditions for.
MODES = ("t2va", "i2va", "fl2va", "fl2va_batched", "ref2va")

#: Prompt rows a node offers, row 1 being the clip.
MAX_ROWS = 24

#: Latent channels in the video and audio halves.
VIDEO_CHANNELS = 24
AUDIO_CHANNELS = 32

#: Pixels per latent cell on each spatial axis.
SPATIAL_STRIDE = 16

#: Audio latent rows, which do not vary with length.
AUDIO_ROWS = 2

#: Canvas side multiple both halves of the model require.
CANVAS_MULTIPLE = 32

#: Width over height used when a megapixel target alone decides both sides.
DEFAULT_ASPECT = 16 / 9

#: Canvas shapes offered: `custom`, the square, the wide shapes from narrowest to widest,
#: then the same shapes turned over. Each is a whole-number pair, the scale coming from
#: megapixels.
ASPECTS: tuple[str, ...] = (
    "custom",
    "1:1",
    "5:4", "4:3", "3:2", "16:10", "16:9", "2:1", "21:9",
    "4:5", "3:4", "2:3", "10:16", "9:16", "1:2", "9:21",
)

#: Images each mode takes, in the order they are drawn.
MODE_IMAGES = {
    "t2va": (),
    "i2va": ("first_frame",),
    "fl2va": ("first_frame", "last_frame"),
    "fl2va_batched": ("images",),
    "ref2va": (),
}

#: Modes that take their keyframes from a batch, one pair per segment.
BATCHED_MODES = ("fl2va_batched",)

#: Modes that build every segment on reference pictures, videos and audio.
REFERENCE_MODES = ("ref2va",)

#: A row's continuity choice that leaves the sampling node's own setting in place.
AS_SET = "as set"

#: What a row may choose for how it continues from the row before it.
ROW_CONTINUITY: tuple[str, ...] = (AS_SET,) + h3_extend.CONTINUITY


def prompt_name(row: int) -> str:
    """The widget name of a row's prompt.

    Args:
        row: Row number, from 1.

    Returns:
        The widget name.
    """
    return f"prompt_{row}"


def overlap_name(row: int) -> str:
    """The widget name of a row's overlap.

    Args:
        row: Row number, from 1.

    Returns:
        The widget name.
    """
    return f"overlap_{row}"


def duration_name(row: int) -> str:
    """The widget name of a row's duration.

    Args:
        row: Row number, from 1.

    Returns:
        The widget name.
    """
    return f"duration_{row}"


def continuity_name(row: int) -> str:
    """The widget name of a row's continuity.

    Args:
        row: Row number, from 1.

    Returns:
        The widget name.
    """
    return f"continuity_{row}"


def frames_of(seconds: float) -> int:
    """Frames a length in seconds covers, before snapping.

    Args:
        seconds: A length in seconds.

    Returns:
        A frame count at the model's frame rate, at least one frame.
    """
    return max(1, int(round(float(seconds) * h3_extend.FPS)))


def duration_of(frames: int) -> float:
    """How long a frame count runs for.

    Args:
        frames: A frame count.

    Returns:
        The length in seconds, to two decimals.
    """
    return round(int(frames) / h3_extend.FPS, 2)


def row_names() -> list[tuple[str, str, str, str]]:
    """Every row's four widget names, in drawn order.

    Returns:
        One ``(prompt, duration, overlap, continuity)`` quad per row.
    """
    return [
        (prompt_name(row), duration_name(row), overlap_name(row), continuity_name(row))
        for row in range(1, MAX_ROWS + 1)
    ]


def filled_rows(widgets: dict) -> list[tuple[str, int, int]]:
    """The rows carrying a prompt, in row order.

    Args:
        widgets: Every row widget's value, keyed by widget name.

    Returns:
        One ``(prompt, frames, overlap, continuity)`` quad per row whose prompt is not
        blank, the frame count taken from that row's duration.
    """
    rows = []
    for prompt_key, duration_key, overlap_key, continuity_key in row_names():
        text = widgets.get(prompt_key)
        if not isinstance(text, str) or not text.strip():
            continue
        choice = widgets.get(continuity_key) or AS_SET
        rows.append((
            text,
            frames_of(widgets.get(duration_key) or 0.0),
            int(widgets.get(overlap_key) or 0),
            choice if choice in ROW_CONTINUITY else AS_SET,
        ))
    return rows


def composed(header: str, prompt: str, footer: str) -> str:
    """One row's prompt with the text that surrounds every row.

    Args:
        header: Text placed before the row's prompt, or blank for none.
        prompt: The row's own prompt.
        footer: Text placed after the row's prompt, or blank for none.

    Returns:
        Whichever of the three carry text, in that order, joined by a blank line.
    """
    parts = [str(part or "").strip() for part in (header, prompt, footer)]
    return "\n\n".join(part for part in parts if part)


def snap_segment(frames: int) -> int:
    """The nearest length a continuing segment may run for.

    Args:
        frames: Frames asked for.

    Returns:
        A positive multiple of the model's clip length.
    """
    return h3_extend.snap_extension(frames)


def snap_overlap_for(frames: int, overlap: int) -> int:
    """The overlap a segment may carry, never more than the segment itself holds.

    Args:
        frames: The segment's snapped frame count.
        overlap: Frames asked to carry, ``0`` for a cut.

    Returns:
        ``0`` for a cut, otherwise a count on the model's guide grid.
    """
    if int(overlap) <= 0:
        return 0
    return min(h3_extend.snap_overlap(overlap), h3_extend.snap_overlap(frames))


def snapped(value: float) -> int:
    """A canvas side rounded to the multiple the model takes.

    Args:
        value: A length in pixels.

    Returns:
        The nearest multiple of :data:`CANVAS_MULTIPLE`, never smaller than one.
    """
    step = CANVAS_MULTIPLE
    return max(step, int(round(float(value) / step)) * step)


def aspect_of(*images) -> float:
    """Width over height of the first picture that arrived.

    Args:
        *images: Image tensors shaped ``[batch, height, width, channels]``, or None.

    Returns:
        The ratio, or :data:`DEFAULT_ASPECT` when no picture arrived.
    """
    for image in images:
        shape = getattr(image, "shape", None)
        if shape is not None and len(shape) >= 3 and shape[-2] and shape[-3]:
            return float(shape[-2]) / float(shape[-3])
    return DEFAULT_ASPECT


def ratio_of(aspect: str, *images) -> float:
    """Width over height for the chosen shape.

    Args:
        aspect: An entry from :data:`ASPECTS`.
        *images: Image tensors the shape falls back to for ``custom``.

    Returns:
        The ratio, taken from the pictures when ``custom`` is chosen.
    """
    chosen = str(aspect or "custom").strip()
    if not chosen or chosen == "custom":
        return aspect_of(*images)
    wide, high = resolution.parse_ratio(chosen)
    return wide / high


def canvas(megapixels: float, width: int, height: int,
           aspect: float = DEFAULT_ASPECT) -> tuple[int, int]:
    """Canvas size from a megapixel target and whichever sides are fixed.

    Args:
        megapixels: Millions of pixels to aim for.
        width: Canvas width in pixels, or 0 to work it out.
        height: Canvas height in pixels, or 0 to work it out.
        aspect: Width over height, used when neither side is given.

    Returns:
        ``(width, height)``, each a multiple of :data:`CANVAS_MULTIPLE`.

    Raises:
        ValueError: A side was left at 0 and megapixels is not above 0.
    """
    fixed_width, fixed_height = max(0, int(width or 0)), max(0, int(height or 0))
    if fixed_width and fixed_height:
        return snapped(fixed_width), snapped(fixed_height)

    area = max(0.0, float(megapixels)) * 1_000_000.0
    if area <= 0.0:
        raise ValueError(
            "MiniMax H3 Conditioning has megapixels at 0 and a canvas side at 0, so "
            "there is no size to work from. Raise megapixels, or set both width and "
            "height"
        )
    if fixed_width:
        side = snapped(fixed_width)
        return side, snapped(area / side)
    if fixed_height:
        side = snapped(fixed_height)
        return snapped(area / side), side

    ratio = float(aspect) if aspect and float(aspect) > 0.0 else DEFAULT_ASPECT
    wide = snapped((area * ratio) ** 0.5)
    # The other side comes off the width by the ratio, then snaps to the multiple.
    return wide, snapped(wide / ratio)


def empty_latent(width: int, height: int, frames: int) -> tuple[dict, int]:
    """A silent, empty H3 joint latent sized for a clip.

    Args:
        width: Canvas width in pixels.
        height: Canvas height in pixels.
        frames: Frames asked for.

    Returns:
        ``(latent, frames)``, the frame count snapped onto the model's grid.

    Raises:
        ValueError: A side is not a multiple of the canvas multiple.
    """
    import comfy.model_management

    if width % CANVAS_MULTIPLE or height % CANVAS_MULTIPLE:
        raise ValueError(
            f"a MiniMax H3 canvas is a multiple of {CANVAS_MULTIPLE} on both sides, and "
            f"{width}x{height} is not"
        )
    count = h3_extend.snap_clip(frames)
    device = comfy.model_management.intermediate_device()
    video = torch.zeros(
        [1, VIDEO_CHANNELS, h3_extend.tokens_for(count),
         height // SPATIAL_STRIDE, width // SPATIAL_STRIDE],
        device=device,
    )
    audio = torch.zeros(
        [1, AUDIO_CHANNELS, AUDIO_ROWS, h3_extend.audio_span(count)], device=device
    )
    return h3_extend.join(video, audio), count


def fitted(image, width: int, height: int):
    """One image stretched onto the canvas, colour channels only.

    Args:
        image: An ``[B, H, W, C]`` tensor.
        width: Canvas width in pixels.
        height: Canvas height in pixels.

    Returns:
        A ``[1, height, width, 3]`` tensor.
    """
    import comfy.utils

    samples = image[:1, ..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled")
    return samples.movedim(1, -1)


def fitted_batch(frames, width: int, height: int):
    """Every frame of a batch stretched onto a canvas, colour channels only.

    Args:
        frames: A ``[T, H, W, C]`` tensor.
        width: Canvas width in pixels.
        height: Canvas height in pixels.

    Returns:
        A ``[T, height, width, 3]`` tensor.
    """
    import comfy.utils

    samples = frames[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled")
    return samples.movedim(1, -1)


def covered(image, width: int, height: int):
    """One image scaled to cover the canvas and cropped to it, colour channels only.

    Args:
        image: An ``[B, H, W, C]`` tensor.
        width: Canvas width in pixels.
        height: Canvas height in pixels.

    Returns:
        A ``[1, height, width, 3]`` tensor.
    """
    import comfy.utils

    samples = image[:1, ..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", "center")
    return samples.movedim(1, -1)


#: How a keyframe is brought to the canvas, the same way for both of its roles.
FITS = ("cover", "stretch")


def prepared(images, width: int, height: int, fit: str) -> list:
    """Every keyframe brought to the canvas once.

    Args:
        images: A ``[N, H, W, C]`` batch in the order the clips run.
        width: Canvas width in pixels.
        height: Canvas height in pixels.
        fit: ``"cover"`` to scale and crop, ``"stretch"`` to fill both sides.

    Returns:
        One ``[1, height, width, 3]`` tensor per keyframe.
    """
    onto = covered if fit == "cover" else fitted
    return [onto(images[index:index + 1], width, height)
            for index in range(images.shape[0])]


#: How the pictures are taken in pairs.
PAIRINGS = ("chain", "bridges")


def pairs(count: int, loop: bool, pairing: str = "chain") -> list[tuple[int, int]]:
    """Which keyframes each clip runs between.

    Args:
        count: How many keyframes there are.
        loop: Whether a closing clip runs from the last keyframe back to the first.
            Read by ``chain`` only.
        pairing: ``"chain"`` runs every neighbouring pair, so N pictures give N-1 clips.
            ``"bridges"`` takes them two at a time, so N pictures give N/2 clips.

    Returns:
        One ``(first, last)`` index pair per clip.

    Raises:
        ValueError: Fewer than two keyframes were given.
    """
    if int(count) < 2:
        raise ValueError(
            f"a keyframe chain needs at least 2 pictures to make a clip and {count} "
            f"arrived. Load Image Sequence reads a folder as one batch, and Image Batch "
            f"Advanced joins single images into one"
        )
    if pairing == "bridges":
        return [(index, index + 1) for index in range(0, int(count) - 1, 2)]
    spans = [(index, index + 1) for index in range(int(count) - 1)]
    if loop and int(count) > 2:
        spans.append((int(count) - 1, 0))
    return spans


def encode(clip, prompt: str, images: list | None = None, references=None):
    """One prompt encoded for the H3 text encoder.

    Args:
        clip: A loaded minimax CLIP.
        prompt: The prompt text.
        images: Keyframe images the encoder reads alongside the text.
        references: A :class:`h3_references.References` presented ahead of the text in
            place of ``images``, its blocks attached as ``minimax_refs``.

    Returns:
        A conditioning.
    """
    if references is None or not references.items:
        tokens = clip.tokenize(prompt, images=list(images or []))
        return clip.encode_from_tokens_scheduled(tokens)

    import node_helpers

    tokens = clip.tokenize(prompt, minimax_ref_items=references.items)
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    return node_helpers.conditioning_set_values(
        conditioning, {"minimax_refs": list(references.blocks)}
    )


def keyframes(vae, first, last, width: int, height: int, frames: int):
    """The guide keyframes a first and last frame contribute.

    Args:
        vae: The video VAE, for encoding each frame.
        first: The opening frame, or ``None``.
        last: The closing frame, or ``None``.
        width: Canvas width in pixels.
        height: Canvas height in pixels.
        frames: The clip's snapped frame count.

    Returns:
        ``(images, blocks)``, the images the encoder reads and the keyframe blocks.
    """
    images, blocks = [], []
    if first is not None:
        picture = fitted(first, width, height)
        images.append(picture)
        blocks.append({"resolved_frame_index": 0, "latent": vae.encode(picture)})
    if last is not None:
        picture = covered(last, width, height)
        images.append(picture)
        blocks.append({"resolved_frame_index": frames - 1, "latent": vae.encode(picture)})
    return images, blocks


def bundle(segments: list[tuple]) -> list[dict]:
    """The per-segment payload a clip loop reads.

    Args:
        segments: One ``(conditioning, frames, overlap)`` triple per segment, in order,
            optionally with that segment's own empty latent as a fourth item and its
            continuity as a fifth.

    Returns:
        One entry per segment.
    """
    entries = []
    for segment in segments:
        conditioning, frames, overlap = segment[:3]
        entry = {
            "conditioning": conditioning,
            "frames": int(frames),
            "overlap": int(overlap),
            "continuity": AS_SET,
        }
        if len(segment) > 3 and segment[3] is not None:
            entry["latent"] = segment[3]
        if len(segment) > 4 and segment[4] in ROW_CONTINUITY:
            entry["continuity"] = segment[4]
        entries.append(entry)
    return entries


def latents_of(prompts) -> list:
    """Every segment's own empty latent, in order.

    Args:
        prompts: A bundle from :func:`bundle`.

    Returns:
        One latent per segment that carries one.
    """
    return [entry["latent"] for entry in (prompts or []) if entry.get("latent") is not None]


def latent_at(prompts, index: int):
    """One segment's own empty latent.

    Args:
        prompts: A bundle from :func:`bundle`.
        index: Segment number, from 0.

    Returns:
        That segment's latent.

    Raises:
        ValueError: The bundle is empty, the index is outside it, or that segment carries
            no latent of its own.
    """
    if not prompts:
        raise ValueError(
            "MiniMax H3 Conditioning passed no segments. Write what the first clip shows "
            "in the first box"
        )
    position = int(index)
    if position < 0 or position >= len(prompts):
        raise ValueError(
            f"clip {position} was asked for and MiniMax H3 Conditioning holds "
            f"{len(prompts)}. Wire its clips output to the loop's count so the two agree"
        )
    latent = prompts[position].get("latent")
    if latent is None:
        raise ValueError(
            f"clip {position} carries no latent of its own. Re-run MiniMax H3 "
            f"Conditioning, which builds one per clip"
        )
    return latent


def pick(prompts, index: int) -> tuple:
    """One segment's conditioning, frame count, overlap and continuity.

    Args:
        prompts: A bundle from :func:`bundle`.
        index: Pass number, from 0, where 0 is the opening clip.

    Returns:
        ``(conditioning, frames, overlap, continuity)``.

    Raises:
        ValueError: The bundle is empty or the index is outside it.
    """
    if not prompts:
        raise ValueError(
            "MiniMax H3 Conditioning passed no prompts. Write what the clip shows in the "
            "first box"
        )
    position = int(index)
    if position < 0 or position >= len(prompts):
        raise ValueError(
            f"pass {position} was asked for and MiniMax H3 Conditioning holds "
            f"{len(prompts)}. Wire its clips output to the loop's count so the two agree"
        )
    entry = prompts[position]
    return (entry["conditioning"], int(entry["frames"]), int(entry.get("overlap", 0)),
            entry.get("continuity", AS_SET))
