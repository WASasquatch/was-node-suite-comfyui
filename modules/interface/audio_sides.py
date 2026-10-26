"""The audio a node hands its panel: each side as a kept WAV file, and the figures beside it.

A side is a batch's first clip, as 32-bit float within :data:`PREVIEW_BYTES`, else 16-bit.
"""

from __future__ import annotations

import math
import struct

import torch

from .. import log
from . import channel, compare_files

__all__ = [
    "PREVIEW_BYTES",
    "REPORT_KEY",
    "SIDES_KEY",
    "before_after",
    "encoded",
    "figures",
    "panel_ui",
    "side",
]

logger = log.get_logger("interface.audio_sides")

#: The ``ui`` key naming the written sides, and the one carrying the figures.
SIDES_KEY = "was_audio"
REPORT_KEY = "was_audio_report"

#: Largest WAV written for one side.
PREVIEW_BYTES = 96 * 1024 * 1024

#: Most channels a side keeps; a clip with more is drawn from its first ones.
PREVIEW_CHANNELS = 8

#: Lowest rate a long side is brought down to, to fit.
LOWEST_RATE = 8000

_FLOAT = 3
_PCM = 1


def _wav(frames: bytes, rate: int, channels: int, bits: int, tag: int, count: int) -> bytes:
    """A RIFF WAVE file around interleaved sample bytes."""
    align = channels * bits // 8
    fmt = struct.pack("<HHIIHHH", tag, channels, rate, rate * align, align, bits, 0)
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    if tag == _FLOAT:
        chunks += b"fact" + struct.pack("<II", 4, count)
    chunks += b"data" + struct.pack("<I", len(frames)) + frames
    return b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks


def encoded(wave: torch.Tensor, rate: int) -> tuple[bytes, dict] | None:
    """One clip as WAV bytes small enough for a panel.

    Args:
        wave: ``(channels, samples)`` float tensor.
        rate: Samples per second.

    Returns:
        ``(data, facts)``, where facts carry ``rate`` and ``channels`` as written and ``exact``,
        true only for the 32-bit float form. None when even mono at the lowest rate is too long.
    """
    wave = wave[:PREVIEW_CHANNELS].detach().to(device="cpu", dtype=torch.float32)
    channels, samples = int(wave.shape[0]), int(wave.shape[1])
    if channels * samples * 4 <= PREVIEW_BYTES:
        frames = wave.T.contiguous().numpy().astype("<f4", copy=False).tobytes()
        return _wav(frames, rate, channels, 32, _FLOAT, samples), {
            "rate": rate, "channels": channels, "exact": True,
            "detail": "32-bit float, every sample as the node produced it",
        }
    written_rate = rate
    if channels * samples * 2 > PREVIEW_BYTES:
        from ..audio.resample import resampled

        fitted = int(rate * PREVIEW_BYTES / float(channels * samples * 2))
        if fitted < LOWEST_RATE and channels > 1:
            wave, channels = wave.mean(dim=0, keepdim=True), 1
            fitted = int(rate * PREVIEW_BYTES / float(samples * 2))
        if fitted < LOWEST_RATE:
            return None
        written_rate = max(LOWEST_RATE, fitted)
        wave = resampled(wave[None], rate, written_rate)[0]
        samples = int(wave.shape[-1])
    codes = (wave.clamp(-1.0, 1.0) * 32767.0).round().to(torch.int16)
    frames = codes.T.contiguous().numpy().astype("<i2", copy=False).tobytes()
    lowered = f", at {written_rate} Hz" if written_rate != rate else ""
    return _wav(frames, written_rate, channels, 16, _PCM, samples), {
        "rate": written_rate, "channels": channels, "exact": False,
        "detail": f"16-bit{lowered}, levels past full scale clipped, to fit the panel",
    }


def _db(value: float) -> float | None:
    """A linear amplitude in dBFS, or None for silence."""
    return round(20.0 * math.log10(value), 2) if value > 0 and math.isfinite(value) else None


def figures(wave: torch.Tensor, rate: int) -> dict:
    """The level and shape figures of a waveform, for a panel.

    Args:
        wave: ``(batch, channels, samples)``.
        rate: Samples per second.

    Returns:
        ``duration``, ``rate``, ``channels``, ``batch``, ``samples``, ``peak_db``, ``rms_db``
        (dBFS, None for silence) and ``clipped``, the samples at or past full scale.
    """
    batch, channels, samples = (int(size) for size in wave.shape)
    flat = wave.detach().to(torch.float64)
    peak = float(flat.abs().max()) if flat.numel() else 0.0
    rms = float(flat.square().mean().sqrt()) if flat.numel() else 0.0
    return {
        "duration": round(samples / float(rate), 6),
        "rate": int(rate),
        "channels": channels,
        "batch": batch,
        "samples": samples,
        "peak_db": _db(peak),
        "rms_db": _db(rms),
        "clipped": int((flat.abs() >= 1.0).sum()),
    }


def side(name: str, wave: torch.Tensor, rate: int) -> dict | None:
    """Write one side of a node's panel.

    Args:
        name: The side's name, such as ``"in"`` or ``"out"``.
        wave: ``(batch, channels, samples)``; the first clip is written.
        rate: Samples per second.

    Returns:
        A ``ui`` entry naming the kept file, its ``side``, the run's ``workflow``, the written
        ``rate`` and ``channels``, ``exact`` and its ``detail``; None when nothing was written.
    """
    try:
        made = encoded(wave[0], int(rate))
        if made is None:
            logger.info("the %s side is too long to draw on its panel", name)
            return None
        data, facts = made
        kept = compare_files.keep_bytes(data, ".wav")
        if kept is None:
            return None
        return {
            "filename": kept, "subfolder": "", "type": compare_files.KEPT_TYPE,
            "workflow": channel.executing_workflow_id() or "", "side": name, **facts,
        }
    except Exception as error:  # noqa: BLE001
        logger.warning("the %s side was not written for its panel (%s: %s)", name, type(error).__name__, error)
        logger.debug("a panel side failed", exc_info=True)
        return None


def panel_ui(sides, report: dict | None = None) -> dict:
    """The ``ui`` a node returns for its audio panel.

    Args:
        sides: ``(name, waveform, rate)`` triples. Nothing is written while no browser is
            attached.
        report: Figures for the panel, JSON with no infinite values.

    Returns:
        ``{SIDES_KEY: [...], REPORT_KEY: [report]}``.
    """
    written = []
    if channel.watching():
        for name, wave, rate in sides:
            entry = side(name, wave, rate)
            if entry is not None:
                written.append(entry)
        if written:
            try:
                compare_files.prune({entry["filename"] for entry in written})
            except Exception as error:  # noqa: BLE001
                logger.debug("the kept panel files could not be pruned (%s)", error)
    return {SIDES_KEY: written, REPORT_KEY: [report or {}]}


def before_after(before: torch.Tensor, before_rate: int, after: torch.Tensor, after_rate: int,
                 facts: dict | None = None) -> dict:
    """The ``ui`` of a node that turns one clip into another: both sides and both sets of figures.

    Args:
        before: The waveform the node was handed.
        before_rate: Its samples per second.
        after: The waveform it answers.
        after_rate: Its samples per second.
        facts: Further figures for the panel, such as markers or the gain applied.

    Returns:
        The ``ui`` mapping.
    """
    report = {"in": figures(before, before_rate), "out": figures(after, after_rate), **(facts or {})}
    return panel_ui([("in", before, before_rate), ("out", after, after_rate)], report)
