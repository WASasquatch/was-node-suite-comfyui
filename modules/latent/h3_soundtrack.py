"""One long soundtrack laid under every window of a multi-scene MiniMax H3 video.

A track is encoded at 40 audio latent steps a second, then :data:`SILENCE_STEPS` of
silence. Window audio step ``j`` lands on finished clip step ``origin + j``.
"""

from __future__ import annotations

import hashlib
import math
import weakref
from collections import OrderedDict
from typing import NamedTuple

from .. import log
from . import h3_extend

__all__ = [
    "AUDIO_RATE",
    "CACHE_SIZE",
    "Encoded",
    "LEAD_STEPS",
    "LOOP",
    "MODEL_SOUND",
    "PAST_END",
    "PIECE_STEPS",
    "SILENCE",
    "SILENCE_STEPS",
    "TAIL_STEPS",
    "eased",
    "encode_track",
    "lay",
    "origin",
    "plan",
    "track_for",
]

logger = log.get_logger("latent.h3_soundtrack")

MODEL_SOUND = "model sound"
SILENCE = "silence"
LOOP = "loop"

#: What plays once the track runs out, in menu order.
PAST_END = (MODEL_SOUND, SILENCE, LOOP)

#: Sample rate an audio VAE that names none encodes at.
AUDIO_RATE = 32000

#: Steps of silence encoded after the track, one second.
SILENCE_STEPS = h3_extend.AUDIO_LATENT_FPS

#: Steps one encoding call keeps, and the steps encoded before and after them and dropped.
PIECE_STEPS = 30 * h3_extend.AUDIO_LATENT_FPS
LEAD_STEPS = 5 * h3_extend.AUDIO_LATENT_FPS
TAIL_STEPS = h3_extend.AUDIO_LATENT_FPS

#: Encoded tracks kept between calls.
CACHE_SIZE = 4

#: Mask values at or above this count as free for the sampler to generate.
FREE = 1.0 - 1e-6

_ENCODED: OrderedDict = OrderedDict()
_DIGESTS: list = []


class Encoded(NamedTuple):
    """One track encoded by the H3 audio VAE.

    Attributes:
        latent: ``[1, 32, 2, steps + SILENCE_STEPS]`` on the CPU, the track then its silence.
        steps: Steps the track itself covers.
        seconds: How long the track runs.
        pieces: Encoding calls it took.
    """

    latent: object
    steps: int
    seconds: float
    pieces: int


def origin(place: dict) -> int:
    """The finished clip's audio step under a window's first audio step.

    Args:
        place: The window's placement, ``{"start": frame, "head": frames}``, with an
            optional ``"audio"`` step that wins where present.

    Returns:
        An audio latent step, from 0.
    """
    if place.get("audio") is not None:
        return int(place["audio"])
    start, head = int(place.get("start", 0)), int(place.get("head", 0))
    return h3_extend.audio_span(start) - h3_extend.audio_span(head)


def eased(distance: int, release: int) -> float:
    """How far the hold opens on a step a given distance inside an edge of the track.

    Args:
        distance: Steps between this step and the edge, 0 for the step beside it.
        release: Steps the hold eases over.

    Returns:
        ``0.0`` where the hold is whole, rising over a half cosine towards ``1.0`` at the edge.
    """
    release = max(0, int(release))
    if distance >= release:
        return 0.0
    return 0.5 - 0.5 * math.cos(math.pi * (release - int(distance)) / (release + 1))


def plan(first: int, steps: int, offset: int, length: int, past_end: str,
         release: int) -> tuple[list[int], list[float]]:
    """Which encoded step each window audio step plays, and how far its hold eases.

    Args:
        first: Finished clip step under the window's first audio step.
        steps: Audio steps the window holds.
        offset: Finished clip step the track's first step lands on.
        length: Steps the track covers.
        past_end: An entry of :data:`PAST_END`.
        release: Steps the hold eases over at each edge of the track.

    Returns:
        ``(sources, opened)``: per window step, an index into :attr:`Encoded.latent` or ``-1``
        where nothing plays, and how far the hold opens there.
    """
    length = max(1, int(length))
    starts = int(offset) > 0
    sources, opened = [], []
    for step in range(int(steps)):
        position = int(first) + step - int(offset)
        source, distance = -1, 0
        if position >= 0:
            lap, inside = divmod(position, length)
            if lap == 0 or past_end == LOOP:
                source = inside
                # The track's end, or the seam a loop starts again on.
                distance = length - 1 - inside
                if lap > 0 or starts:
                    distance = min(distance, inside)
            elif past_end == SILENCE:
                beyond = position - length
                source = length + min(beyond, SILENCE_STEPS - 1)
                distance = beyond
        sources.append(source)
        opened.append(eased(distance, release) if source >= 0 else 0.0)
    return sources, opened


def _waveform(audio):
    """The waveform and sample rate of a ComfyUI audio, checked."""
    waveform = audio.get("waveform") if isinstance(audio, dict) else None
    rate = int(audio.get("sample_rate") or 0) if isinstance(audio, dict) else 0
    if waveform is None or getattr(waveform, "ndim", 0) != 3 or int(waveform.shape[-1]) == 0:
        raise ValueError(
            "H3 Soundtrack's audio holds no sound. Wire the track in from Load Audio, or "
            "from a node that outputs AUDIO"
        )
    if rate <= 0:
        raise ValueError(
            f"H3 Soundtrack's audio names a sample rate of {rate}. Load the track again with "
            f"Load Audio"
        )
    return waveform, rate


def _resample(waveform, rate: int, wanted: int):
    """A waveform at another sample rate."""
    try:
        import comfy.audio

        return comfy.audio.resample(waveform, rate, wanted)
    except (ImportError, AttributeError):
        import torchaudio

        return torchaudio.functional.resample(waveform, rate, wanted)


def encode_track(audio_vae, audio: dict) -> Encoded:
    """A whole track encoded by the H3 audio VAE, a long one in overlapping pieces.

    Args:
        audio_vae: The H3 audio VAE.
        audio: A ComfyUI audio, ``{"waveform": [B, C, L], "sample_rate": int}``.

    Returns:
        The encoded track, from its first batch entry.

    Raises:
        ValueError: The audio holds no sound, or ``audio_vae`` is not the H3 audio VAE.
    """
    import torch
    import torch.nn.functional as functional

    if (getattr(audio_vae, "latent_dim", 2) != 2
            or getattr(audio_vae, "latent_channels", 32) != 32):
        raise ValueError(
            "H3 Soundtrack's audio_vae is not the MiniMax H3 audio VAE, which encodes sound to "
            "[1, 32, 2, steps]. Wire Load VAE set to the H3 audio VAE, as "
            "`minimax_h3_audio_vae_fp32`"
        )
    waveform, rate = _waveform(audio)
    seconds = int(waveform.shape[-1]) / rate
    wanted = int(getattr(audio_vae, "audio_sample_rate", AUDIO_RATE) or AUDIO_RATE)
    waveform = waveform[:1].float()
    if rate != wanted:
        waveform = _resample(waveform, rate, wanted)
    hop = wanted // h3_extend.AUDIO_LATENT_FPS
    steps = max(1, math.ceil(int(waveform.shape[-1]) / hop))
    total = steps + SILENCE_STEPS
    # Whole steps throughout, so the encoder trims nothing.
    padded = functional.pad(waveform, (0, total * hop - int(waveform.shape[-1])))
    if total <= PIECE_STEPS + LEAD_STEPS + TAIL_STEPS:
        spans = [(0, total)]
    else:
        spans = [(first, min(total, first + PIECE_STEPS)) for first in range(0, total, PIECE_STEPS)]
    pieces = []
    for first, last in spans:
        lead = min(LEAD_STEPS, first) if len(spans) > 1 else 0
        tail = min(TAIL_STEPS, total - last) if len(spans) > 1 else 0
        chunk = padded[..., (first - lead) * hop:(last + tail) * hop]
        latent = audio_vae.encode(chunk.movedim(1, -1))
        if getattr(latent, "ndim", 0) != 4 or int(latent.shape[-1]) < lead + last - first:
            raise ValueError(
                f"H3 Soundtrack's audio_vae encoded {last + tail - first + lead} steps of sound "
                f"to a {tuple(getattr(latent, 'shape', ()))} latent, and the MiniMax H3 audio VAE "
                f"answers [1, 32, 2, steps]. Wire Load VAE set to the H3 audio VAE"
            )
        pieces.append(latent[..., lead:lead + last - first].float().cpu())
    return Encoded(torch.cat(pieces, dim=-1), steps, seconds, len(spans))


def _digest(waveform, rate: int) -> str:
    """A fingerprint of a waveform's first batch entry, remembered by the tensor's identity."""
    for ref, digest in _DIGESTS:
        if ref() is waveform:
            return digest
    import torch

    data = waveform[:1].detach().to("cpu", torch.float32).contiguous()
    hashed = hashlib.blake2b(digest_size=16)
    hashed.update(f"{rate}:{tuple(data.shape)}".encode())
    hashed.update(data.numpy().tobytes())
    digest = hashed.hexdigest()
    try:
        _DIGESTS.append((weakref.ref(waveform), digest))
    except TypeError:
        return digest
    del _DIGESTS[:-CACHE_SIZE]
    return digest


def track_for(audio_vae, audio: dict) -> tuple[Encoded, bool]:
    """A track encoded once and kept for every later window that lays it.

    Args:
        audio_vae: The H3 audio VAE.
        audio: A ComfyUI audio.

    Returns:
        ``(encoded, reused)``, ``reused`` True where the track was already encoded.

    Raises:
        ValueError: The audio holds no sound, or ``audio_vae`` is not the H3 audio VAE.
    """
    waveform, rate = _waveform(audio)
    key = (id(audio_vae), _digest(waveform, rate))
    held = _ENCODED.get(key)
    if held is not None and held[0]() is audio_vae:
        _ENCODED.move_to_end(key)
        return held[1], True
    encoded = encode_track(audio_vae, audio)
    try:
        owner = weakref.ref(audio_vae)
    except TypeError:
        return encoded, False
    _ENCODED[key] = (owner, encoded)
    while len(_ENCODED) > CACHE_SIZE:
        _ENCODED.popitem(last=False)
    logger.info("encoded a %.2f s track to %d audio steps in %d piece(s)", encoded.seconds,
                encoded.steps, encoded.pieces)
    return encoded, False


def _masks(window: dict, video, audio):
    """The window's video mask as it came and its audio mask, ones where none was set."""
    import torch

    mask = window.get("noise_mask")
    tensors = getattr(mask, "tensors", None)
    if tensors is not None and len(tensors) == 2:
        video_mask, audio_mask = tensors[0], tensors[1]
    elif mask is not None:
        video_mask, audio_mask = mask, None
    else:
        video_mask = torch.ones(
            [video.shape[0], 1, video.shape[2], video.shape[3], video.shape[4]],
            device=video.device, dtype=torch.float32,
        )
        audio_mask = None
    if audio_mask is None:
        audio_mask = torch.ones([audio.shape[0], 1, audio.shape[2], audio.shape[-1]],
                                device=audio.device, dtype=torch.float32)
    if int(audio_mask.shape[-1]) != int(audio.shape[-1]):
        raise ValueError(
            f"the window's sound mask covers {audio_mask.shape[-1]} audio steps and its sound "
            f"{audio.shape[-1]}. Wire H3 Extend Window's window straight into H3 Soundtrack"
        )
    return video_mask, audio_mask


def _runs(values: list[int]) -> list[tuple[int, int]]:
    """Consecutive stretches of a list of step numbers, as ``(first, last + 1)``."""
    runs = []
    for value in values:
        if runs and value == runs[-1][1]:
            runs[-1] = (runs[-1][0], value + 1)
        else:
            runs.append((value, value + 1))
    return runs


def _frame(step: int) -> int:
    """The finished clip's frame at an audio step."""
    return int(round(int(step) * h3_extend.FPS / h3_extend.AUDIO_LATENT_FPS))


def lay(window: dict, encoded: Encoded, offset_seconds: float, strength: float,
        release: int = h3_extend.AUDIO_RELEASE, past_end: str = MODEL_SOUND) -> tuple[dict, str]:
    """Write a window's slice of the track into its sound and hold it there with the mask.

    Args:
        window: An H3 joint latent, placed by H3 Extend Window or opening the clip.
        encoded: The track, from :func:`track_for`.
        offset_seconds: Where the track starts on the finished clip.
        strength: ``1.0`` holds the track exactly, ``0.0`` leaves the sound free.
        release: Steps the hold eases over where the track starts, ends or loops.
        past_end: An entry of :data:`PAST_END`.

    Returns:
        ``(latent, report)``. Steps the window already holds keep their sound and mask; the
        video mask is passed on as it came.

    Raises:
        ValueError: The window is not an H3 joint latent, or its sound and the track differ
            in shape.
    """
    import torch

    video, audio = h3_extend.split(window)
    if tuple(encoded.latent.shape[1:3]) != tuple(audio.shape[1:3]):
        raise ValueError(
            f"the track encoded to {tuple(encoded.latent.shape[1:3])} channels and the "
            f"window's sound holds {tuple(audio.shape[1:3])}. Wire the H3 audio VAE into "
            f"audio_vae and an H3 window into window"
        )
    place = window.get(h3_extend.WINDOW_KEY) or {"start": 0, "head": 0}
    first = origin(place)
    steps = int(audio.shape[-1])
    offset = int(round(float(offset_seconds) * h3_extend.AUDIO_LATENT_FPS))
    hold = 1.0 - max(0.0, min(1.0, float(strength)))
    sources, opened = plan(first, steps, offset, encoded.steps, past_end, release)
    video_mask, audio_mask = _masks(window, video, audio)
    free = audio_mask.reshape(-1, steps).amin(dim=0).ge(FREE).tolist()

    covered = [step for step in range(steps) if sources[step] >= 0]
    written = [step for step in covered if free[step]]
    left = len(covered) - len(written)
    if written:
        columns = torch.tensor(written, dtype=torch.long, device=audio.device)
        picked = torch.tensor([sources[step] for step in written], dtype=torch.long)
        audio = audio.clone()
        audio[..., columns] = encoded.latent[..., picked].to(device=audio.device, dtype=audio.dtype)
        audio_mask = audio_mask.clone()
        audio_mask[..., columns] = torch.tensor(
            [hold + (1.0 - hold) * opened[step] for step in written],
            dtype=audio_mask.dtype, device=audio_mask.device,
        )

    out = dict(window)
    out["samples"] = h3_extend.join(video, audio)["samples"]
    out["noise_mask"] = h3_extend.join(video_mask, audio_mask)["samples"]

    opening = int(place.get("start", 0)) - int(place.get("head", 0))
    frames = h3_extend.frames_for(int(video.shape[2]))
    if not written:
        why = ("every step the track covers is already carried and stays as it was" if left
               else "the track does not reach it")
        return out, f"left frames {opening}-{opening + frames} free: {why}"
    heard = [step for step in written if sources[step] < encoded.steps]
    quiet = len(written) - len(heard)
    if heard:
        runs = _runs([sources[step] for step in heard])
        named = " and ".join(f"{a}-{b}" for a, b in runs[:3])
        if len(runs) > 3:
            named += f" and {len(runs) - 3} more"
        report = (f"held {len(heard) / h3_extend.AUDIO_LATENT_FPS:.2f} s of the track "
                  f"(steps {named})")
        if quiet:
            report += f" and {quiet / h3_extend.AUDIO_LATENT_FPS:.2f} s of silence after it"
    else:
        report = f"held {quiet / h3_extend.AUDIO_LATENT_FPS:.2f} s of silence after the track"
    report += (f" at {1.0 - hold:.2f} under frames {_frame(first + written[0])}-"
               f"{_frame(first + written[-1] + 1)}")
    eased_steps = sum(1 for step in written if opened[step] > 0.0)
    if eased_steps:
        report += f"; eased over {eased_steps} steps where the track starts, ends or loops"
    if left:
        report += f"; {left} steps the window carries keep their own sound"
    return out, report
