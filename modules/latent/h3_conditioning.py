"""Encoding a MiniMax H3 task prompt, and one prompt per extension pass.

Row 1 opens the clip and every row after it extends it. Each row carries its own frame
count, and the loop runs once per row.
"""

from __future__ import annotations

import re

import torch

from ..image import resolution
from . import h3_extend

#: The task each mode conditions for.
MODES = ("t2va", "i2va", "fl2va", "fl2va_batched", "ref2va")

#: Prompt rows a node offers, row 1 being the clip.
MAX_ROWS = 24

#: Continuities that open a fresh scene and carry no frames.
CUT_LIKE = ("cut", "handoff", h3_extend.REFERENCE_VIDEO, h3_extend.REFERENCE_SAMPLE)

#: A row's source that continues from the segment before it.
PREVIOUS_SOURCE = -1

#: Which of `prompt_header` and `prompt_footer` a row takes, the first being the default.
WRAPS = ("both", "header only", "footer only", "neither")

#: A line opening a named prompt section, as ``overall_soundscape:``.
SECTION_LINE = re.compile(r"^([a-z][a-z0-9_]*):(?:\s|$)")

#: The tag opening a spoken line, which the text encoder reads as one token.
DIALOGUE_OPEN = "<d>"

#: A shot heading in a prompt, as ``[Shot 2]``.
SHOT_HEADING = re.compile(r"\[Shot (\d+)\]")

#: A shot's start time in a prompt, as ``At 00:04.000``.
SHOT_TIME = re.compile(r"\bAt (\d+):(\d{2})\.(\d{3})\b")

#: A line opening the section a prompt writes its shots in.
SHOT_SECTION = re.compile(r"^(integrated_multimodal_description|detailed_description):[ \t]*", re.M)

#: The shot a window opens on before a cut, ahead of the segment's own shots.
CUT_LEAD = "The last moments of the previous shot continue unchanged."

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

#: How clean core holds a pinned frame or a reference by default.
KEYFRAME_CLEAN = 0.999

#: The continuity a row takes where it names none.
DEFAULT_CONTINUITY = h3_extend.CONTINUITY[0]

#: The key a keyframe block closing its segment carries.
CLOSING_KEY = "was_closing"

#: Bundle entry key naming the node a segment's prompt came from.
OWNER_KEY = "owner"

#: Bundle entry key marking the last segment of a loop, which closes on the video's first frame.
LOOP_KEY = "closes_loop"

#: Bundle entry key holding what a segment's prompt was encoded from.
ENCODING_KEY = "encoding"

#: Prompt keys a prompt encoded again keeps from the one it replaces.
KEPT_KEYS = ("minimax_keyframes", "minimax_visual_cond_noise_aug", "minimax_audio_cond_noise_aug")

#: What a row may choose for how it continues from the row before it.
ROW_CONTINUITY: tuple[str, ...] = h3_extend.CONTINUITY

#: A row's model choice that picks between the wired models by what the row carries.
AUTO_MODEL = "auto"

#: The two MiniMax H3 models a row may sample with.
FL2VA = "fl2va"
REF2VA = "ref2va"

#: What a row may choose for the model its segment is sampled with.
MODEL_CHOICES: tuple[str, ...] = (AUTO_MODEL, FL2VA, REF2VA)


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


def source_name(row: int) -> str:
    """The widget name of the segment a row continues from.

    Args:
        row: Row number, from 1.

    Returns:
        The widget name.
    """
    return f"source_{row}"


def sources_of(widgets: dict) -> list[int]:
    """The segment each row carrying a prompt continues from, in row order.

    Args:
        widgets: Every row widget's value, keyed by widget name.

    Returns:
        One source per row whose prompt is not blank, as the row holds it.
    """
    sources = []
    for row in range(1, MAX_ROWS + 1):
        text = widgets.get(prompt_name(row))
        if not isinstance(text, str) or not text.strip():
            continue
        value = widgets.get(source_name(row))
        sources.append(PREVIOUS_SOURCE if value is None else int(value))
    return sources


def wrap_name(row: int) -> str:
    """The widget name of which shared text a row takes.

    Args:
        row: Row number, from 1.

    Returns:
        The widget name.
    """
    return f"header_footer_{row}"


def wraps_of(widgets: dict) -> list[str]:
    """The shared text each row carrying a prompt takes, in row order.

    Args:
        widgets: Every row widget's value, keyed by widget name.

    Returns:
        One entry of :data:`WRAPS` per row whose prompt is not blank.
    """
    wraps = []
    for row in range(1, MAX_ROWS + 1):
        text = widgets.get(prompt_name(row))
        if not isinstance(text, str) or not text.strip():
            continue
        value = widgets.get(wrap_name(row))
        wraps.append(value if value in WRAPS else WRAPS[0])
    return wraps


def seed_name(row: int) -> str:
    """The widget name of a row's seed.

    Args:
        row: Row number, from 1.

    Returns:
        The widget name.
    """
    return f"seed_{row}"


def seeds_of(widgets: dict) -> list[int]:
    """The seed each row carrying a prompt names, in row order.

    Args:
        widgets: Every row widget's value, keyed by widget name.

    Returns:
        One seed per row whose prompt is not blank, ``0`` where the row leaves it to the run.
    """
    values = []
    for row in range(1, MAX_ROWS + 1):
        text = widgets.get(prompt_name(row))
        if not isinstance(text, str) or not text.strip():
            continue
        try:
            values.append(max(0, int(widgets.get(seed_name(row)) or 0)))
        except (TypeError, ValueError):
            values.append(0)
    return values


def seed_of(prompts, index: int, base: int) -> int:
    """The seed one segment samples with.

    Args:
        prompts: A bundle from :func:`bundle`, or ``None``.
        index: Segment number, from 0.
        base: The run's seed.

    Returns:
        The row's own seed where it names one, else ``base`` plus the segment number.
    """
    own = 0
    if prompts and 0 <= int(index) < len(prompts):
        own = int(prompts[int(index)].get("seed") or 0)
    return own if own > 0 else int(base) + int(index)


def sound_name(row: int) -> str:
    """The widget name of a row's sound choice.

    Args:
        row: Row number, from 1.

    Returns:
        The widget name.
    """
    return f"sound_{row}"


def sounds_of(widgets: dict) -> list[str]:
    """How each row carrying a prompt takes its sound, in row order.

    Args:
        widgets: Every row widget's value, keyed by widget name.

    Returns:
        One entry of :data:`h3_extend.SOUNDS` per row whose prompt is not blank.
    """
    values = []
    for row in range(1, MAX_ROWS + 1):
        text = widgets.get(prompt_name(row))
        if not isinstance(text, str) or not text.strip():
            continue
        value = widgets.get(sound_name(row))
        values.append(value if value in h3_extend.SOUNDS else h3_extend.SOUNDS[0])
    return values


def sound_of(prompts, index: int) -> str:
    """One segment's sound choice.

    Args:
        prompts: A bundle from :func:`bundle`.
        index: Segment number, from 0.

    Returns:
        An entry of :data:`h3_extend.SOUNDS`, ``auto`` where the segment names none.
    """
    if not prompts or not 0 <= int(index) < len(prompts):
        return h3_extend.SOUNDS[0]
    value = prompts[int(index)].get("sound")
    return value if value in h3_extend.SOUNDS else h3_extend.SOUNDS[0]


def strength_name(row: int) -> str:
    """The widget name of a row's hold on its pinned frames and references.

    Args:
        row: Row number, from 1.

    Returns:
        The widget name.
    """
    return f"strength_{row}"


def strengths_of(widgets: dict) -> list[float]:
    """How firmly each row carrying a prompt holds its pinned frames and references.

    Args:
        widgets: Every row widget's value, keyed by widget name.

    Returns:
        One value in ``[0, 1]`` per row whose prompt is not blank, ``1.0`` where none is set.
    """
    values = []
    for row in range(1, MAX_ROWS + 1):
        text = widgets.get(prompt_name(row))
        if not isinstance(text, str) or not text.strip():
            continue
        try:
            value = float(widgets.get(strength_name(row), 1.0))
        except (TypeError, ValueError):
            value = 1.0
        values.append(max(0.0, min(1.0, value)))
    return values


def held(conditioning, strength: float):
    """A prompt whose pinned frames and references are held at a strength.

    Args:
        conditioning: The segment's prompt.
        strength: ``1.0`` to hold them as given, lower to add noise to them before sampling.

    Returns:
        The prompt, unchanged at ``1.0``.
    """
    if float(strength) >= 1.0:
        return conditioning
    import node_helpers

    return node_helpers.conditioning_set_values(
        conditioning, {"minimax_visual_cond_noise_aug": KEYFRAME_CLEAN * max(0.0, float(strength))})


def model_name(row: int) -> str:
    """The widget name of a row's model choice.

    Args:
        row: Row number, from 1.

    Returns:
        The widget name.
    """
    return f"model_{row}"


def models_of(widgets: dict) -> list[str]:
    """The model each row carrying a prompt chose, in row order.

    Args:
        widgets: Every row widget's value, keyed by widget name.

    Returns:
        One entry of :data:`MODEL_CHOICES` per row whose prompt is not blank.
    """
    choices = []
    for row in range(1, MAX_ROWS + 1):
        text = widgets.get(prompt_name(row))
        if not isinstance(text, str) or not text.strip():
            continue
        value = widgets.get(model_name(row))
        choices.append(value if value in MODEL_CHOICES else AUTO_MODEL)
    return choices


def pick_model(choice: str, fl2va, ref2va, referenced: bool) -> tuple:
    """The model a segment is sampled with.

    Args:
        choice: An entry of :data:`MODEL_CHOICES`.
        fl2va: The wired fl2va model, or ``None``.
        ref2va: The wired ref2va model, or ``None``.
        referenced: Whether the segment's prompt carries references.

    Returns:
        ``(model, kind)``: the model, or ``None`` where the kind asked for is not wired, and
        the kind as :data:`FL2VA`, :data:`REF2VA`, or an empty string where nothing is wired
        to choose from.
    """
    if choice == FL2VA:
        return fl2va, FL2VA
    if choice == REF2VA:
        return ref2va, REF2VA
    if fl2va is not None and ref2va is not None:
        return (ref2va, REF2VA) if referenced else (fl2va, FL2VA)
    if ref2va is not None:
        return ref2va, REF2VA
    if fl2va is not None:
        return fl2va, FL2VA
    return None, ""


def model_or_blocker(prompts, index: int, named: str, referenced: bool | None = None):
    """The model one bundled segment is sampled with, or what stops a node reading it.

    Args:
        prompts: A bundle from :func:`bundle`, or ``None`` where none is wired.
        index: Segment number, from 0.
        named: The node answering, as ``H3 Extend Window``.
        referenced: True where the answering node adds references of its own, which an
            ``auto`` row answers with ref2va. ``None`` keeps the choice the bundle made.

    Returns:
        The model, or an ``ExecutionBlocker`` saying what to wire.
    """
    from comfy_execution.graph_utils import ExecutionBlocker

    guidance = "or feed the guider's model from a loader and leave this output unwired"
    if prompts is None:
        return ExecutionBlocker(
            f"{named} has no prompts wired, so it has no segment model to answer. Wire "
            f"MiniMax H3 Conditioning's prompts into prompts, {guidance}"
        )
    model, kind = model_of(prompts, index, referenced)
    if model is not None:
        return model
    segment = f"segment {int(index) + 1}"
    if kind:
        return ExecutionBlocker(
            f"{segment} asks for the {kind} model and model_{kind} is not wired on MiniMax H3 "
            f"Conditioning. Wire it there, set that row's model to the one that is wired, "
            f"{guidance}"
        )
    return ExecutionBlocker(
        f"no model is wired into MiniMax H3 Conditioning, so it has none to answer for "
        f"{segment}. Wire model_fl2va or model_ref2va there, {guidance}"
    )


def model_of(prompts, index: int, referenced: bool | None = None) -> tuple:
    """The model one bundled segment is sampled with.

    Args:
        prompts: A bundle from :func:`bundle`.
        index: Segment number, from 0.
        referenced: True where the reader adds references of its own, which an ``auto``
            row answers with ref2va. ``None`` keeps the choice the bundle made.

    Returns:
        ``(model, kind)`` as :func:`pick_model` answers them, ``(None, "")`` where the bundle
        carries no model for that segment.
    """
    if not prompts or not 0 <= int(index) < len(prompts):
        return None, ""
    entry = prompts[int(index)]
    models = entry.get("models")
    if referenced and entry.get("model_choice") == AUTO_MODEL and models:
        return pick_model(AUTO_MODEL, models.get(FL2VA), models.get(REF2VA), True)
    return entry.get("model"), str(entry.get("model_kind") or "")


def takes_header(wrap: str) -> bool:
    """Whether a row set to ``wrap`` takes the header.

    Args:
        wrap: An entry of :data:`WRAPS`.

    Returns:
        True for ``both`` and ``header only``.
    """
    return wrap in (WRAPS[0], WRAPS[1])


def takes_footer(wrap: str) -> bool:
    """Whether a row set to ``wrap`` takes the footer.

    Args:
        wrap: An entry of :data:`WRAPS`.

    Returns:
        True for ``both`` and ``footer only``.
    """
    return wrap in (WRAPS[0], WRAPS[2])


def source_of(prompts, index: int) -> int:
    """The source a bundled segment was given.

    Args:
        prompts: A bundle from :func:`bundle`.
        index: Segment number, from 0.

    Returns:
        The source as the row held it, ``-1`` where none was given.
    """
    if not prompts or not 0 <= int(index) < len(prompts):
        return PREVIOUS_SOURCE
    return int(prompts[int(index)].get("source", PREVIOUS_SOURCE))


def resolved_source(source: int, index: int) -> int:
    """The finished segment a segment continues from.

    Args:
        source: Negative counts back from this segment, positive names a segment from 1,
            and ``0`` means the one before.
        index: This segment's number, from 0.

    Returns:
        A segment number from 0, earlier than ``index``.

    Raises:
        ValueError: The source names this segment, a later one, or one before the first.
    """
    source, index = int(source), int(index)
    picked = index - 1 if source == 0 else (index + source if source < 0 else source - 1)
    if not 0 <= picked < index:
        raise ValueError(
            f"segment {index + 1} is set to continue from source {source}, which names "
            f"segment {picked + 1}, and only segments 1 to {index} are finished before it. "
            f"Set its source to -1 for the segment before, or to a number from 1 to {index}"
        )
    return picked


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
        choice = widgets.get(continuity_key) or DEFAULT_CONTINUITY
        choice = h3_extend.CONTINUITY_RENAMED.get(choice, choice)
        rows.append((
            text,
            frames_of(widgets.get(duration_key) or 0.0),
            int(widgets.get(overlap_key) or 0),
            choice if choice in ROW_CONTINUITY else DEFAULT_CONTINUITY,
        ))
    return rows


def sections(text: str) -> list[tuple[str | None, str]]:
    """Text split at every line opening a named section, as ``overall_soundscape:``.

    Args:
        text: A prompt, or a header or footer.

    Returns:
        ``(name, block)`` pairs in order. Text ahead of the first section is named ``None``.
    """
    found: list[tuple[str | None, list[str]]] = [(None, [])]
    for line in str(text or "").splitlines():
        match = SECTION_LINE.match(line)
        if match:
            found.append((match.group(1), [line]))
        else:
            found[-1][1].append(line)
    return [(name, "\n".join(lines)) for name, lines in found if name or "".join(lines).strip()]


def dialogue_blocks(text: str) -> int:
    """How many ``<d>`` dialogue tags a text holds.

    Args:
        text: A prompt, or a header or footer.

    Returns:
        The count of opening ``<d>`` tags.
    """
    return str(text or "").count(DIALOGUE_OPEN)


def without_sections(text: str, names: set[str]) -> str:
    """Text with the named sections taken out.

    Args:
        text: A header or footer.
        names: Section names to take out.

    Returns:
        The rest of the text, trimmed.
    """
    kept = [block for name, block in sections(text) if name not in names]
    return "\n".join(kept).strip()


def composed(header: str, prompt: str, footer: str) -> str:
    """One row's prompt inside the header and footer, less any section the row names itself.

    Args:
        header: Text placed before the row's prompt, or blank for none.
        prompt: The row's own prompt.
        footer: Text placed after the row's prompt, or blank for none.

    Returns:
        Whichever of the three carry text, in that order, joined by a blank line.
    """
    own = {name for name, unused in sections(prompt) if name}
    if own:
        header = without_sections(header, own)
        footer = without_sections(footer, own)
    parts = [str(part or "").strip() for part in (header, prompt, footer)]
    return "\n\n".join(part for part in parts if part)


def snap_segment(frames: int, overlap: int = 0, continuity: str = DEFAULT_CONTINUITY) -> int:
    """New frames a segment adds so its window runs the clip length closest to ``frames``.

    Args:
        frames: Frames the segment's window is asked to run, carried frames included.
        overlap: Frames asked to carry, ``0`` for a cut.
        continuity: The row's continuity; a cut carries nothing.

    Returns:
        A positive multiple of the model's clip length. The window is this plus the
        carried frames, or plus :data:`h3_extend.CLIP_LEAD` for a fresh scene.
    """
    if continuity in CUT_LIKE or int(overlap) <= 0:
        carried = h3_extend.CLIP_LEAD
    else:
        carried = snap_overlap_for(frames, overlap)
    return max(h3_extend.CLIP_FRAMES, h3_extend.snap_clip(frames) - carried)


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
    return min(h3_extend.snap_overlap(overlap), h3_extend.floor_overlap(frames))


#: A timeline line in a row's prompt, as ``0.8–2.2 seconds:``.
TIMELINE_LINE = re.compile(
    r"^([ \t]*)(\d+(?:\.\d+)?)([ \t]*[–-][ \t]*)(\d+(?:\.\d+)?)([ \t]*seconds?\b.*)$",
    re.MULTILINE)

#: The length line of a row's prompt, as ``duration_seconds: 8``.
DURATION_LINE = re.compile(r"^([ \t]*duration_seconds:[ \t]*)(\d+(?:\.\d+)?)[ \t]*$",
                           re.MULTILINE)


def stamp(seconds: float) -> str:
    """A time written as a prompt timeline writes it.

    Args:
        seconds: A time in seconds.

    Returns:
        The time to two decimals at most and one at least, as ``0.8`` or ``1.72``.
    """
    text = f"{float(seconds):.2f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


def shifted(text: str, lead: float, total: float) -> str:
    """A row's prompt timed to the window it is sampled in.

    Args:
        text: The row's own prompt.
        lead: Seconds the carried frames run at the window's start, ``0`` for none.
        total: Seconds the whole window runs.

    Returns:
        The prompt with every ``a–b seconds`` line moved on by ``lead``, the last one
        ending at ``total``, and its ``duration_seconds`` set to ``total``.
    """
    lines = list(TIMELINE_LINE.finditer(text))
    last = lines[-1].start() if lines else -1

    def moved(match):
        start = float(match.group(2)) + lead
        # The last beat runs to the window's end, so no stretch of the window goes unwritten.
        end = total if match.start() == last else float(match.group(4)) + lead
        return f"{match.group(1)}{stamp(start)}{match.group(3)}{stamp(end)}{match.group(5)}"

    text = TIMELINE_LINE.sub(moved, text)
    return DURATION_LINE.sub(lambda m: f"{m.group(1)}{stamp(total)}", text)


def carried_timeline(text: str, frames: int, overlap: int, continuity: str,
                     opening: bool = False) -> str:
    """A row's prompt timed to the window it is sampled in.

    Args:
        text: The row's own prompt.
        frames: Frames the row's window is asked to run.
        overlap: Frames asked to carry, ``0`` for a cut.
        continuity: The row's continuity.
        opening: True for the row that opens the clip.

    Returns:
        The prompt, its last beat and ``duration_seconds`` ending where its window ends.
    """
    _, window = window_of(frames, overlap, continuity, opening)
    return shifted(text, 0.0, window / h3_extend.FPS)


def window_of(frames: int, overlap: int, continuity: str,
              opening: bool = False) -> tuple[int, int]:
    """The window a row is sampled in, and how much of its start is not new.

    Args:
        frames: Frames the row's window is asked to run.
        overlap: Frames asked to carry, ``0`` for a cut.
        continuity: The row's continuity.
        opening: True for the row that opens the clip.

    Returns:
        ``(head, window)``: frames at the start of the window that are carried or bridged
        rather than new, and frames the whole window holds.
    """
    if opening:
        return 0, h3_extend.snap_clip(frames)
    if continuity in h3_extend.BRIDGING:
        # The sound bridge is whole clips of the snapped overlap, one clip at least.
        bridge = h3_extend.bridge_frames(snap_overlap_for(frames, overlap))
        return bridge, h3_extend.snap_clip(bridge + snap_segment(frames, overlap, continuity))
    if continuity in CUT_LIKE or int(overlap) <= 0:
        return 0, snap_segment(frames, overlap, continuity) + h3_extend.CLIP_LEAD
    carried = snap_overlap_for(frames, overlap)
    return carried, carried + snap_segment(frames, overlap, continuity)


def stated_seconds(text: str) -> float | None:
    """The length a row's prompt states on its ``duration_seconds`` line.

    Args:
        text: The row's own prompt.

    Returns:
        The seconds, or None where the prompt states none.
    """
    match = DURATION_LINE.search(str(text or ""))
    return float(match.group(2)) if match else None


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


def covered_batch(frames, width: int, height: int):
    """Every frame of a batch scaled to cover the canvas and cropped to it, colour only.

    Args:
        frames: A ``[T, H, W, C]`` tensor.
        width: Canvas width in pixels.
        height: Canvas height in pixels.

    Returns:
        A ``[T, height, width, 3]`` tensor.
    """
    import comfy.utils

    samples = frames[..., :3].movedim(-1, 1)
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


def encoding_of(clip, text: str, shown, references) -> dict:
    """What a segment's prompt was encoded from, for encoding it again.

    Args:
        clip: The loaded minimax CLIP.
        text: The prompt text as encoded.
        shown: Keyframe images the encoder read.
        references: The segment's references, or ``None``.

    Returns:
        The entry stored under :data:`ENCODING_KEY`.
    """
    return {"clip": clip, "text": text, "shown": list(shown or []), "references": references}


def shot_time(seconds: float) -> str:
    """A time as a prompt writes a shot's start.

    Args:
        seconds: From the start of the clip.

    Returns:
        ``MM:SS.mmm``, as ``00:00.708``.
    """
    millis = max(0, round(float(seconds) * 1000))
    return f"{millis // 60000:02d}:{millis // 1000 % 60:02d}.{millis % 1000:03d}"


def cut_in(text: str, seconds: float) -> str:
    """A prompt whose own shots follow a cut ``seconds`` in, after a shot continuing the last scene.

    Args:
        text: A segment's prompt.
        seconds: Where the cut lands, from the start of the window.

    Returns:
        The prompt with :data:`CUT_LEAD` as ``[Shot 1]`` and its own shots renumbered and retimed
        after it.
    """
    text = str(text or "")
    lead = f"[Shot 1] {CUT_LEAD}\n\n"
    cut = f"At {shot_time(seconds)}, the shot cuts to the next scene."
    first = SHOT_HEADING.search(text)
    if first is None:
        section = SHOT_SECTION.search(text)
        at = section.end() if section else 0
        gap = "\n" if section and not text[at:].startswith("\n") else ""
        return f"{text[:at].rstrip(' ')}{gap}{lead}[Shot 2] {cut} {text[at:].lstrip()}".rstrip()

    def later(match):
        minutes, whole, millis = (int(part) for part in match.groups())
        return f"At {shot_time(minutes * 60 + whole + millis / 1000 + float(seconds))}"

    body = SHOT_TIME.sub(later, text[first.start():])
    body = SHOT_HEADING.sub(lambda match: f"[Shot {int(match.group(1)) + 1}]", body)
    opening = SHOT_HEADING.match(body)
    return f"{text[:first.start()]}{lead}{opening.group(0)} {cut}{body[opening.end():]}"


def encoded_text(entry) -> str:
    """The prompt text a segment was encoded from.

    Args:
        entry: The segment's bundle entry.

    Returns:
        The text, or an empty string where the entry holds none.
    """
    encoding = entry.get(ENCODING_KEY) if isinstance(entry, dict) else None
    return str((encoding or {}).get("text") or "")


def reencoded(entry, positive, items, blocks, pictures=(), text=None):
    """A segment's prompt encoded again with what a pass adds shown to the text encoder.

    Args:
        entry: The segment's bundle entry.
        positive: The prompt as the pass received it, whose keyframes and holds are kept.
        items: Text encoder entries for the references the pass adds.
        blocks: Their ``minimax_refs`` blocks, in the same order.
        pictures: Keyframe pictures the pass adds, read where the prompt has no reference.
        text: The prompt text to encode in place of the segment's own, or ``None``.

    Returns:
        The new prompt, or ``None`` where the entry holds nothing to encode from.
    """
    encoding = entry.get(ENCODING_KEY) if isinstance(entry, dict) else None
    if not encoding or encoding.get("clip") is None:
        return None
    import types

    import node_helpers

    base = encoding.get("references")
    every_item = list(getattr(base, "items", None) or []) + list(items)
    every_block = list(getattr(base, "blocks", None) or []) + list(blocks)
    references = types.SimpleNamespace(items=every_item, blocks=every_block) if every_item else None
    fresh = encode(encoding["clip"], encoding["text"] if text is None else text,
                   list(pictures) + list(encoding.get("shown") or []), references)
    held = positive[0][1] if positive else {}
    kept = {key: held[key] for key in KEPT_KEYS if key in held}
    return node_helpers.conditioning_set_values(fresh, kept) if kept else fresh


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
        blocks.append({"resolved_frame_index": frames - 1, "latent": vae.encode(picture),
                       CLOSING_KEY: True})
    return images, blocks


def bundle(segments: list[tuple]) -> list[dict]:
    """The per-segment payload a clip loop reads.

    Args:
        segments: One ``(conditioning, frames, overlap)`` triple per segment, in order,
            optionally with that segment's own empty latent as a fourth item, its
            continuity as a fifth, its model as a sixth, the model's kind as a seventh, the
            row's model choice as an eighth and ``{kind: model}`` of every wired model as a
            ninth.

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
            "continuity": DEFAULT_CONTINUITY,
            "model": None,
            "model_kind": "",
        }
        if len(segment) > 3 and segment[3] is not None:
            entry["latent"] = segment[3]
        if len(segment) > 4 and segment[4] in ROW_CONTINUITY:
            entry["continuity"] = segment[4]
        if len(segment) > 5:
            entry["model"] = segment[5]
        if len(segment) > 6 and segment[6] in (FL2VA, REF2VA):
            entry["model_kind"] = segment[6]
        if len(segment) > 7 and segment[7] in MODEL_CHOICES:
            entry["model_choice"] = segment[7]
        if len(segment) > 8 and isinstance(segment[8], dict):
            entry["models"] = dict(segment[8])
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
            entry.get("continuity", DEFAULT_CONTINUITY))
