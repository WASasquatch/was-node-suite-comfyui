"""Clips joined end to end with a transition at each join.

Frames are ``(frames, height, width, channels)`` floats. An ``n`` frame transition lays the
incoming clip's first ``n`` frames over the outgoing clip's last ``n``.
"""

from __future__ import annotations

import math
from fractions import Fraction

import torch

from . import mixing

__all__ = [
    "DIRECTIONS",
    "KIND_LETTERS",
    "MAX_CLIPS",
    "MAX_DURATION",
    "RATE_MODES",
    "TRANSITIONS",
    "conformed",
    "joined_audio",
    "join_sheet",
    "overlaps",
    "sample_at",
    "same_rate",
    "share_of",
    "starts_of",
    "timeline_text",
    "transition_frame",
    "wipe_weight",
]

#: The transitions a join can take, in menu order.
TRANSITIONS = ("cut", "dissolve", "dip to black", "dip to white", "wipe")

#: Which way a wipe's edge travels, in menu order.
DIRECTIONS = (
    "left to right",
    "right to left",
    "top to bottom",
    "bottom to top",
    "top left to bottom right",
    "top right to bottom left",
    "bottom left to top right",
    "bottom right to top left",
)

#: How a clip at another frame rate is brought to the first clip's rate.
RATE_MODES = ("hold", "blend", "interpolate")

#: Most clips one join reads.
MAX_CLIPS = 1000

#: Longest transition, in frames.
MAX_DURATION = 10000

#: One letter per transition in the timeline a report carries.
KIND_LETTERS = {"cut": "c", "dissolve": "d", "dip to black": "k", "dip to white": "w", "wipe": "x"}

#: Two frame rates closer than this are one rate.
RATE_TOLERANCE = 1e-3

#: Long side, in pixels, cuts are looked for at before frames are blended or drawn.
CUT_SIDE = 384

#: Longest timeline text a report carries.
TIMELINE_CHARS = 1000

#: Width of one picture on the join sheet, and the most joins it shows.
SHEET_THUMB = 160
SHEET_JOINS = 8
SHEET_GAP = 4


def overlaps(lengths, duration: int, transition: str) -> list[int]:
    """Frames each join overlaps by.

    Args:
        lengths: Frames in each clip, in order.
        duration: Frames asked for at every join.
        transition: A name from :data:`TRANSITIONS`; ``cut`` overlaps nothing.

    Returns:
        One count per join. Each is held to the frames the outgoing clip has left after its
        own incoming transition, and to the incoming clip's frames less the half of them its
        own outgoing transition keeps, so a short clip between two joins splits evenly.
    """
    wanted = 0 if transition == "cut" else max(0, int(duration))
    last = len(lengths) - 1
    shared, taken = [], 0
    for index in range(last):
        incoming = int(lengths[index + 1])
        if index + 1 < last:
            incoming -= min(wanted, incoming // 2)
        overlap = min(wanted, int(lengths[index]) - taken, incoming)
        shared.append(max(0, overlap))
        taken = shared[-1]
    return shared


def starts_of(lengths, shared) -> list[int]:
    """The output frame every clip starts on.

    Args:
        lengths: Frames in each clip.
        shared: Frames each join overlaps by, from :func:`overlaps`.

    Returns:
        One start per clip, the first 0. The joined clip holds ``starts[-1] + lengths[-1]``
        frames, which is every clip's frames less every overlap.
    """
    found = [0]
    for index, overlap in enumerate(shared):
        found.append(found[-1] + int(lengths[index]) - int(overlap))
    return found


def share_of(index: int, overlap: int) -> float:
    """How far through a transition one of its frames sits, at the middle of that frame.

    Args:
        index: The frame within the transition, from 0.
        overlap: Frames the transition lasts.

    Returns:
        ``(index + 0.5) / overlap``, between 0 and 1 and never either.
    """
    return (index + 0.5) / overlap


def _coordinate(height: int, width: int, direction: str, device) -> torch.Tensor:
    """Each pixel's place along a wipe's direction, 0 where the edge starts and 1 where it ends."""
    across = (torch.arange(width, dtype=torch.float32, device=device) + 0.5) / width
    down = (torch.arange(height, dtype=torch.float32, device=device) + 0.5) / height
    u = across[None, :].expand(height, width)
    v = down[:, None].expand(height, width)
    if direction == "right to left":
        return 1.0 - u
    if direction == "top to bottom":
        return v.clone()
    if direction == "bottom to top":
        return 1.0 - v
    if direction == "top left to bottom right":
        return (u + v) * 0.5
    if direction == "top right to bottom left":
        return ((1.0 - u) + v) * 0.5
    if direction == "bottom left to top right":
        return (u + (1.0 - v)) * 0.5
    if direction == "bottom right to top left":
        return ((1.0 - u) + (1.0 - v)) * 0.5
    return u.clone()


def wipe_weight(height: int, width: int, direction: str, softness: float, share: float,
                device=None) -> torch.Tensor:
    """How much of the incoming picture shows at each pixel during a wipe.

    Args:
        height: Frame height.
        width: Frame width.
        direction: A name from :data:`DIRECTIONS`.
        softness: Share of the frame the edge is spread over, 0 for a hard edge.
        share: How far through the wipe, 0 to 1.
        device: Where the weights are made.

    Returns:
        ``(height, width)`` float32, 0 where the outgoing picture shows and 1 where the
        incoming one does. Every pixel is 0 at a share of 0 and 1 at a share of 1.
    """
    where = _coordinate(int(height), int(width), direction, device)
    soft = max(0.0, float(softness))
    if soft <= 0.0:
        return (where < float(share)).to(torch.float32)
    edge = float(share) * (1.0 + soft)
    return ((edge - where) / soft).clamp(0.0, 1.0)


def transition_frame(outgoing: torch.Tensor, incoming: torch.Tensor, transition: str, share: float,
                     space: str, direction: str = DIRECTIONS[0], softness: float = 0.0) -> torch.Tensor:
    """One frame of a transition.

    Args:
        outgoing: ``(height, width, channels)`` of the clip that ends.
        incoming: The same frame of the clip that follows.
        transition: A name from :data:`TRANSITIONS`.
        share: How far through the transition, from :func:`share_of`.
        space: :data:`modules.media.mixing.CODES` or :data:`~modules.media.mixing.LIGHT`.
        direction: Which way a wipe travels.
        softness: A wipe's edge width, as a share of the frame.

    Returns:
        The frame. A dip reaches its colour at the middle, the outgoing picture before it and
        the incoming one after.
    """
    if transition == "dissolve":
        return mixing.mix(outgoing, incoming, share, space)
    if transition in ("dip to black", "dip to white"):
        level = 0.0 if transition == "dip to black" else 1.0
        depth = 1.0 - abs(1.0 - 2.0 * share)
        return mixing.toward(outgoing if share < 0.5 else incoming, level, depth, space)
    if transition == "wipe":
        height, width = int(outgoing.shape[0]), int(outgoing.shape[1])
        weight = wipe_weight(height, width, direction, softness, share, outgoing.device)
        return mixing.mix(outgoing, incoming, weight, space)
    return incoming.clone()


def same_rate(first, second) -> bool:
    """Whether two frame rates are one rate.

    Args:
        first: Frames per second.
        second: Frames per second.

    Returns:
        True where they differ by less than :data:`RATE_TOLERANCE`.
    """
    return abs(float(first) - float(second)) < RATE_TOLERANCE


def conformed(frames, rate, size, target_rate, fit: str, rate_mode: str, net=None, name: str = "",
              progress=None) -> tuple[torch.Tensor, list[str]]:
    """One clip's frames at the joined clip's frame rate and size.

    Args:
        frames: ``(frames, height, width, channels)`` floats on the CPU.
        rate: The clip's frames per second.
        size: ``(width, height)`` of the joined clip.
        target_rate: Its frames per second.
        fit: A name from :data:`modules.image.sizing.MODES`.
        rate_mode: A name from :data:`RATE_MODES`.
        net: The interpolation network, for ``interpolate``.
        name: The node, for messages.
        progress: Optional callable taking a step count.

    Returns:
        ``(frames, notes)``: the frames, the clip's own tensor where nothing changed, and one
        note per change made.

    Raises:
        ValueError: ``interpolate`` has no network, or one that cannot land where the new rate
            needs.
    """
    import comfy.model_management

    from ..image import motion as motion_field
    from ..image import sizing
    from . import retime

    notes = []
    count = int(frames.shape[0])
    if not same_rate(rate, target_rate):
        where = retime.rate_positions(count, Fraction(rate), Fraction(target_rate))
        device = comfy.model_management.get_torch_device()
        cuts = [False] * max(0, count - 1)
        drawn_on = device
        if rate_mode != "hold" and count > 1:
            cuts = motion_field.measure(frames[..., :3], CUT_SIDE, device).cuts()
        if rate_mode == "interpolate":
            if net is None:
                raise ValueError(
                    f"{name} draws frames for a clip at another frame rate with EMA-VFI on "
                    f"'interpolate'. Wire EMA-VFI Video Model Loader into ema_vfi_model, or "
                    f"choose 'blend' or 'hold'."
                )
            retime.check_network(net.name, where)
            net.backend.load()
            model = net.backend.model
            drawn_on = next(model.parameters()).device
        else:
            model = None
        frames = retime.frames_at(
            frames, where, rate_mode, cuts, model, drawn_on, progress, name
        )
        notes.append(f"{float(rate):.6g} to {float(target_rate):.6g} fps ({rate_mode})")
    width, height = int(size[0]), int(size[1])
    if (int(frames.shape[2]), int(frames.shape[1])) != (width, height):
        pad = (0, 0, 0, 0) if int(frames.shape[-1]) > 3 else (0, 0, 0, 255)
        before = f"{int(frames.shape[2])}x{int(frames.shape[1])}"
        frames = sizing.fit_frames(frames, (width, height), fit, pad=pad)
        notes.append(f"{before} to {width}x{height} ({fit})")
    return frames, notes


def sample_at(frame: int, sample_rate: int, frame_rate) -> int:
    """The audio sample a frame starts on.

    Args:
        frame: The frame, from 0.
        sample_rate: Samples per second.
        frame_rate: Frames per second.

    Returns:
        ``frame * (sample_rate / frame_rate)`` in floats, rounded half to even, never below 0.
    """
    return max(0, int(round(int(frame) * (int(sample_rate) / float(frame_rate)))))


def _fades(count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Equal power gains over ``count`` samples, each taken at the middle of its sample."""
    progress = (torch.arange(count, dtype=torch.float64) + 0.5) / count
    turn = 0.5 * math.pi * progress
    return torch.sin(turn), torch.cos(turn)


def joined_audio(audios, lengths, starts, shared, frame_rate) -> dict | None:
    """Every clip's audio laid under its frames, crossfaded under each transition.

    Args:
        audios: One ``AUDIO`` mapping or None per clip.
        lengths: Frames in each clip, after conforming.
        starts: The output frame each clip starts on.
        shared: Frames each join overlaps by.
        frame_rate: The joined clip's frames per second.

    Returns:
        ``AUDIO`` as long as the joined clip, at the first sounding clip's sample rate and the
        widest clip's channel count, silent under a clip with none; None where no clip has any.
        Each clip's audio starts on the sample its first frame starts on, and is cut or padded
        with silence to the samples its frames last. Under a transition the outgoing clip falls
        on a cosine and the incoming rises on a sine, so their squared gains sum to 1.

    Raises:
        ValueError: The clips carry batches that cannot be paired, or the audio would be too
            long.
    """
    from ..audio import clip as sound

    present = [
        (index, audio) for index, audio in enumerate(audios)
        if isinstance(audio, dict) and audio.get("waveform") is not None
    ]
    if not present:
        return None
    waves, rate, channels = sound.matched(
        [audio for _, audio in present], [f"clip {index + 1}'s audio" for index, _ in present]
    )
    batches = {int(wave.shape[0]) for wave in waves}
    batch = max(batches)
    if batches - {1, batch}:
        raise ValueError(
            f"the clips' audio carries batches of {sorted(batches)}; joined clips hold the same "
            f"count, or 1."
        )
    held = {index: wave.expand(batch, -1, -1) for (index, _), wave in zip(present, waves)}
    last_clip = len(lengths) - 1
    total = sample_at(starts[-1] + lengths[-1], rate, frame_rate)
    sound.check_size(batch, channels, total, "the joined clips")
    out = torch.zeros(batch, channels, total, dtype=torch.float64)
    for index, length in enumerate(lengths):
        wave = held.get(index)
        if wave is None:
            continue
        first = sample_at(starts[index], rate, frame_rate)
        last = sample_at(starts[index] + length, rate, frame_rate)
        span = last - first
        if span <= 0:
            continue
        piece = torch.zeros(batch, channels, span, dtype=torch.float64)
        kept = min(span, int(wave.shape[-1]))
        piece[..., :kept] = wave[..., :kept].double()
        gain = torch.ones(span, dtype=torch.float64)
        if index > 0 and shared[index - 1] > 0:
            rising = sample_at(starts[index] + shared[index - 1], rate, frame_rate) - first
            if rising > 0:
                gain[:rising] = _fades(rising)[0]
        if index < last_clip and shared[index] > 0:
            falling = last - sample_at(starts[index + 1], rate, frame_rate)
            if falling > 0:
                gain[span - falling:] *= _fades(falling)[1]
        out[..., first:last] += piece * gain
    return sound.make(out.to(torch.float32), rate)


def timeline_text(lengths, shared, transition: str) -> str:
    """The joined clip's layout as one short line: each clip's frames, and each join's letter and overlap.

    Args:
        lengths: Frames in each clip.
        shared: Frames each join overlaps by.
        transition: The transition at every join.

    Returns:
        Such as ``48 d12 60 d12 30``, cut short with a trailing ``+`` past
        :data:`TIMELINE_CHARS`.
    """
    letter = KIND_LETTERS.get(transition, "c")
    parts = [str(int(lengths[0]))]
    for index, overlap in enumerate(shared):
        parts.append(f"{letter}{int(overlap)}")
        parts.append(str(int(lengths[index + 1])))
    text = ""
    for part in parts:
        if len(text) + len(part) + 1 > TIMELINE_CHARS:
            return text + " +"
        text = f"{text} {part}" if text else part
    return text


def join_sheet(frames, starts, shared, transition: str):
    """Pictures across each join, one row per join: the frame before it, frames inside it and the frame after.

    Args:
        frames: The joined clip, ``(frames, height, width, channels)``.
        starts: The output frame each clip starts on.
        shared: Frames each join overlaps by.
        transition: The transition at every join.

    Returns:
        ``(1, height, width, 3)`` in ``[0, 1]``. A single clip gives one row of its first and
        last frames.
    """
    import torch.nn.functional as F
    from PIL import Image, ImageDraw

    total = int(frames.shape[0])
    height, width = (int(v) for v in frames.shape[1:3])
    thumb_h = max(1, int(round(SHEET_THUMB * height / max(width, 1))))
    rows = []
    for index in range(min(len(shared), SHEET_JOINS)):
        at, overlap = int(starts[index + 1]), int(shared[index])
        if overlap >= 3:
            inside = [at + int((step + 0.5) * overlap / 3) for step in range(3)]
        else:
            inside = list(range(at, at + overlap))
        picked = [at - 1, *inside, at + overlap]
        kind = transition if overlap else "cut"
        label = f"join {index + 1}  {kind}  {overlap} f  at frame {at}"
        rows.append((label, [min(total - 1, max(0, frame)) for frame in picked]))
    if not rows:
        rows.append(("one clip, nothing joined", [0, total - 1]))
    columns = max(len(picked) for _, picked in rows)
    label_h = 16
    sheet_w = columns * SHEET_THUMB + (columns + 1) * SHEET_GAP
    sheet_h = len(rows) * (thumb_h + label_h) + (len(rows) + 1) * SHEET_GAP
    sheet = Image.new("RGB", (sheet_w, sheet_h), (16, 16, 16))
    draw = ImageDraw.Draw(sheet)
    for row, (label, picked) in enumerate(rows):
        y = SHEET_GAP + row * (thumb_h + label_h + SHEET_GAP)
        draw.text((SHEET_GAP + 2, y + 2), label, fill=(230, 230, 230))
        for column, frame in enumerate(picked):
            picture = frames[frame, ..., :3].float().permute(2, 0, 1).unsqueeze(0)
            picture = F.interpolate(picture, size=(thumb_h, SHEET_THUMB), mode="bilinear",
                                    align_corners=False, antialias=True)
            pixels = (picture[0].permute(1, 2, 0).clamp(0, 1) * 255 + 0.5).byte().cpu().numpy()
            x = SHEET_GAP + column * (SHEET_THUMB + SHEET_GAP)
            sheet.paste(Image.fromarray(pixels), (x, y + label_h))
            tag = f"f{frame}"
            draw.rectangle((x, y + label_h, x + 6 + 6 * len(tag), y + label_h + 13), fill=(0, 0, 0))
            draw.text((x + 3, y + label_h + 1), tag, fill=(255, 255, 255))
    data = torch.frombuffer(bytearray(sheet.tobytes()), dtype=torch.uint8)
    return (data.view(sheet_h, sheet_w, 3).float() / 255.0).unsqueeze(0)
