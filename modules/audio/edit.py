"""Cutting, joining, fading, padding, looping and silence finding on waveforms.

Waveforms are ``(batch, channels, samples)`` float tensors. Spans are ``(first, last)`` sample
indices, ``last`` exclusive.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from . import curves
from .clip import MAX_SECONDS, check_size

__all__ = [
    "MAX_REGIONS",
    "MAX_TEXT",
    "faded",
    "format_regions",
    "joined",
    "looped",
    "loud_spans",
    "padded",
    "parse_regions",
    "parse_times",
    "parse_time",
    "silence_levels",
    "span_of",
    "without",
]

#: Most regions or times one text widget is read for.
MAX_REGIONS = 2000

#: Longest text a region or time widget is read to, in characters.
MAX_TEXT = 65536

#: Frame length the silence finder measures loudness over, in seconds.
LEVEL_FRAME = 0.01


def parse_time(text: str, what: str) -> float:
    """A time written as seconds, ``m:ss`` or ``h:mm:ss``.

    Args:
        text: ``"12.5"``, ``"1:02.5"`` or ``"1:00:02.5"``, with an optional trailing ``s``.
        what: The widget it came from, for the error.

    Returns:
        Seconds, from 0 to :data:`MAX_SECONDS`.

    Raises:
        ValueError: The text is not a time, or lies outside that range.
    """
    refusal = f"{what}: {text!r} is not a time; write 12.5, 1:02.5 or 1:00:02.5."
    raw = str(text).strip().lower().removesuffix("s").strip()
    parts = [part.strip() for part in raw.split(":")]
    if not raw or len(parts) > 3:
        raise ValueError(refusal)
    total = 0.0
    for index, piece in enumerate(parts):
        whole, point, fraction = piece.partition(".")
        last = index == len(parts) - 1
        if not whole.isdigit() or (point and (not last or not fraction.isdigit())):
            raise ValueError(refusal)
        if index > 0 and int(whole) >= 60:
            raise ValueError(f"{what}: {text!r} has {whole} in a minutes or seconds place.")
        total = total * 60.0 + float(piece)
    if not math.isfinite(total) or total > MAX_SECONDS:
        raise ValueError(f"{what}: {text!r} is past the {MAX_SECONDS:.0f} s an audio node reads.")
    return total


def _entries(text: str, what: str) -> list[str]:
    """The comma, semicolon or line separated entries of a widget."""
    body = str(text or "")
    if len(body) > MAX_TEXT:
        raise ValueError(f"{what} holds {len(body):,} characters; at most {MAX_TEXT:,} are read.")
    entries = []
    for line in body.splitlines():
        if line.strip().startswith("#"):
            continue
        for entry in line.replace(";", ",").split(","):
            if entry.strip():
                entries.append(entry.strip())
    if len(entries) > MAX_REGIONS:
        raise ValueError(f"{what} lists {len(entries):,} entries; at most {MAX_REGIONS:,} are read.")
    return entries


def parse_times(text: str, what: str) -> list[float]:
    """Times listed in a widget, rising and without repeats.

    Args:
        text: Entries such as ``"1.5, 3.2, 1:10"``, separated by commas, semicolons or lines.
            A line opening with ``#`` is skipped.
        what: The widget it came from, for errors.

    Returns:
        Seconds, sorted.
    """
    return sorted({parse_time(entry, what) for entry in _entries(text, what)})


def parse_regions(text: str, what: str) -> list[tuple[float, float]]:
    """Regions listed in a widget, merged where they overlap.

    Args:
        text: Entries such as ``"1.2-2.5, 4:00-4:10.5"``, each a start and an end joined by
            ``-`` or ``to``. A line opening with ``#`` is skipped.
        what: The widget it came from, for errors.

    Returns:
        ``(start, end)`` pairs in seconds, sorted, none overlapping, each longer than nothing.

    Raises:
        ValueError: An entry is not two times, or ends before it starts.
    """
    found = []
    for entry in _entries(text, what):
        lowered = entry.lower()
        pieces = lowered.split(" to ") if " to " in lowered else lowered.split("-")
        if len(pieces) != 2:
            raise ValueError(f"{what}: {entry!r} is not a region; write 1.2-2.5 or 1:02-1:04.5.")
        start, end = parse_time(pieces[0], what), parse_time(pieces[1], what)
        if end < start:
            raise ValueError(f"{what}: {entry!r} ends before it starts.")
        if end > start:
            found.append((start, end))
    found.sort()
    merged: list[tuple[float, float]] = []
    for start, end in found:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def format_regions(regions) -> str:
    """Regions as the text :func:`parse_regions` reads.

    Args:
        regions: ``(start, end)`` pairs in seconds.

    Returns:
        ``"1.200-2.500, 4.000-4.100"``.
    """
    return ", ".join(f"{start:.3f}-{end:.3f}" for start, end in regions)


def span_of(rate: int, length: int, start: float, end: float) -> tuple[int, int]:
    """A start and end time as a sample span inside a waveform.

    Args:
        rate: Samples per second.
        length: Samples in the waveform.
        start: Seconds from the start.
        end: Seconds from the start; 0, or anything past the end, reaches the end.

    Returns:
        ``(first, last)``, clamped to the waveform.
    """
    first = min(length, max(0, int(round(float(start) * rate))))
    last = length if float(end) <= 0.0 else min(length, int(round(float(end) * rate)))
    return first, max(first, last)


def faded(wave: torch.Tensor, fade_in: int, fade_out: int, curve_in: str, curve_out: str) -> torch.Tensor:
    """A waveform with a fade at each end.

    Args:
        wave: ``(batch, channels, samples)``.
        fade_in: Samples the fade in lasts.
        fade_out: Samples the fade out lasts.
        curve_in: The fade in's curve, from :data:`curves.CURVES`.
        curve_out: The fade out's curve.

    Returns:
        A new tensor. Fades longer than the clip together are shortened in proportion.
    """
    length = int(wave.shape[-1])
    fade_in, fade_out = max(0, int(fade_in)), max(0, int(fade_out))
    if fade_in + fade_out > length and fade_in + fade_out > 0:
        scale = length / float(fade_in + fade_out)
        fade_in = int(fade_in * scale)
        fade_out = min(length - fade_in, int(fade_out * scale))
    out = wave.clone()
    if fade_in:
        out[..., :fade_in] *= curves.rising(fade_in, curve_in)
    if fade_out:
        out[..., length - fade_out:] *= curves.falling(fade_out, curve_out)
    return out


def joined(pieces, crossfade: int, curve: str, gap: int = 0) -> torch.Tensor:
    """Pieces placed end to end, overlapped by a crossfade or parted by silence.

    Args:
        pieces: ``(batch, channels, samples)`` tensors sharing batch and channel counts.
        crossfade: Samples each join overlaps by. Held to half the shorter neighbour.
        curve: The crossfade's curve, from :data:`curves.CURVES`.
        gap: Samples of silence placed between pieces. Above 0, pieces do not overlap and
            ``crossfade`` becomes a fade out and a fade in at either side of the gap.

    Returns:
        One tensor.
    """
    pieces = [piece for piece in pieces if int(piece.shape[-1]) > 0]
    if not pieces:
        raise ValueError("there is nothing to join: every piece is empty.")
    batch, channels = int(pieces[0].shape[0]), int(pieces[0].shape[1])
    crossfade, gap = max(0, int(crossfade)), max(0, int(gap))
    lengths = [int(piece.shape[-1]) for piece in pieces]
    overlaps = []
    for before, after in zip(lengths, lengths[1:]):
        overlaps.append(0 if gap else min(crossfade, before // 2, after // 2))
    total = sum(lengths) - sum(overlaps) + gap * (len(pieces) - 1)
    check_size(batch, channels, total, "the crossfade or gap")
    out = torch.zeros(batch, channels, total, dtype=torch.float32)
    cursor = 0
    for index, piece in enumerate(pieces):
        piece = piece.to(torch.float32)
        head = overlaps[index - 1] if index > 0 else 0
        tail = overlaps[index] if index < len(overlaps) else 0
        if gap and crossfade:
            edge_in = min(crossfade, lengths[index] // 2) if index > 0 else 0
            edge_out = min(crossfade, lengths[index] // 2) if index < len(pieces) - 1 else 0
            piece = faded(piece, edge_in, edge_out, curve, curve)
        elif head or tail:
            piece = piece.clone()
            if head:
                piece[..., :head] *= curves.rising(head, curve)
            if tail:
                piece[..., lengths[index] - tail:] *= curves.falling(tail, curve)
        start = cursor - head
        out[..., start:start + lengths[index]] += piece
        cursor = start + lengths[index] + (gap if index < len(pieces) - 1 else 0)
    return out


def without(wave: torch.Tensor, spans, crossfade: int, curve: str) -> torch.Tensor:
    """A waveform with sample spans taken out and the pieces either side joined.

    Args:
        wave: ``(batch, channels, samples)``.
        spans: ``(first, last)`` sample spans, sorted and not overlapping.
        crossfade: Samples each join overlaps by.
        curve: The crossfade's curve.

    Returns:
        A new tensor.

    Raises:
        ValueError: The spans take out the whole waveform.
    """
    length = int(wave.shape[-1])
    pieces = []
    cursor = 0
    for first, last in spans:
        first, last = max(cursor, int(first)), min(length, int(last))
        if first > cursor:
            pieces.append(wave[..., cursor:first])
        cursor = max(cursor, last)
    if cursor < length:
        pieces.append(wave[..., cursor:])
    if not pieces:
        raise ValueError("the cuts remove the whole clip; nothing is left to keep.")
    if len(pieces) == 1:
        return pieces[0].clone()
    return joined(pieces, crossfade, curve)


def padded(wave: torch.Tensor, before: int, after: int) -> torch.Tensor:
    """A waveform with silence placed before and after it.

    Args:
        wave: ``(batch, channels, samples)``.
        before: Samples of silence at the start.
        after: Samples of silence at the end.

    Returns:
        A new tensor.
    """
    before, after = max(0, int(before)), max(0, int(after))
    batch, channels, length = (int(size) for size in wave.shape)
    check_size(batch, channels, before + length + after, "the padding")
    out = torch.zeros(batch, channels, before + length + after, dtype=torch.float32)
    out[..., before:before + length] = wave
    return out


def looped(wave: torch.Tensor, times: int, total: int, crossfade: int, curve: str) -> torch.Tensor:
    """A waveform played several times over, each seam crossfaded.

    Args:
        wave: ``(batch, channels, samples)``.
        times: Plays in all, at least 1.
        total: Samples to fill instead, repeating as often as needed and cutting the last play
            short. 0 uses ``times``.
        crossfade: Samples each seam overlaps by.
        curve: The seam's curve.

    Returns:
        A new tensor.
    """
    length = int(wave.shape[-1])
    overlap = min(max(0, int(crossfade)), length // 2)
    pitch = max(1, length - overlap)
    if int(total) > 0:
        times = max(1, math.ceil((int(total) - overlap) / pitch))
    times = max(1, int(times))
    batch, channels = int(wave.shape[0]), int(wave.shape[1])
    check_size(batch, channels, pitch * times + overlap, "the repeat count or duration")
    out = joined([wave] * times, overlap, curve)
    if int(total) > 0:
        out = out[..., : int(total)].contiguous()
    return out


def silence_levels(wave: torch.Tensor, rate: int) -> tuple[np.ndarray, int]:
    """The loudness of each short frame, the loudest channel and clip counted.

    Args:
        wave: ``(batch, channels, samples)``.
        rate: Samples per second.

    Returns:
        ``(levels, hop)``: one RMS level in dBFS per frame of ``hop`` samples, ``-inf`` where
        the frame is digital silence.
    """
    hop = max(1, int(round(rate * LEVEL_FRAME)))
    length = int(wave.shape[-1])
    frames = max(1, math.ceil(length / hop))
    padded_wave = torch.nn.functional.pad(wave.to(torch.float64), (0, frames * hop - length))
    power = padded_wave.reshape(*wave.shape[:2], frames, hop).square().mean(dim=-1)
    loudest = power.amax(dim=(0, 1)).numpy()
    with np.errstate(divide="ignore"):
        levels = 10.0 * np.log10(loudest)
    return levels, hop


def loud_spans(levels: np.ndarray, hop: int, length: int, threshold: float,
               min_silence: int, padding: int, min_length: int) -> list[tuple[int, int]]:
    """The stretches of a clip that are not silent.

    Args:
        levels: Per-frame levels from :func:`silence_levels`.
        hop: Samples per frame.
        length: Samples in the clip.
        threshold: Level in dBFS at or below which a frame is silent.
        min_silence: Samples a quiet stretch lasts before it parts two sounds.
        padding: Samples of the quiet kept either side of each sound, held to half the quiet.
        min_length: Samples a sound lasts to be kept, its padding not counted.

    Returns:
        ``(first, last)`` sample spans in order.
    """
    loud = levels > float(threshold)
    spans: list[list[int]] = []
    frame = 0
    count = int(loud.shape[0])
    while frame < count:
        if not loud[frame]:
            frame += 1
            continue
        start = frame
        while frame < count and loud[frame]:
            frame += 1
        first, last = start * hop, min(length, frame * hop)
        if spans and first - spans[-1][1] < min_silence:
            spans[-1][1] = last
        else:
            spans.append([first, last])
    spans = [span for span in spans if span[1] - span[0] >= max(1, int(min_length))]
    out = []
    for index, (first, last) in enumerate(spans):
        room_before = first - (spans[index - 1][1] if index else 0)
        room_after = (spans[index + 1][0] if index + 1 < len(spans) else length) - last
        before = min(padding, room_before if index == 0 else room_before // 2)
        after = min(padding, room_after if index + 1 == len(spans) else room_after // 2)
        out.append((max(0, first - before), min(length, last + after)))
    return out
