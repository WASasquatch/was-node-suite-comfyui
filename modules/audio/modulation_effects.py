"""Chorus, flanger, phaser and tremolo: effects whose delay, filter or level an LFO moves.

Signals are float64 numpy arrays shaped ``(channels, samples)``. Delays are in samples unless
named in milliseconds; rates and frequencies are hertz.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from . import delays, filters, modulation

__all__ = [
    "BLOCK_MIN",
    "CHORUS_MODES",
    "CHUNK",
    "CYCLE_POINTS",
    "DECLICK",
    "STEP_POINTS",
    "FLANGER_MODES",
    "TREMOLO_MODES",
    "applied",
    "blended",
    "channel_phases",
    "chorus",
    "chorus_figures",
    "cycle",
    "flanger",
    "flanger_figures",
    "lfo_rate",
    "phaser",
    "phaser_figures",
    "phaser_notches",
    "random_steps",
    "read",
    "recirculated",
    "swept_lfo",
    "tremolo",
    "tremolo_figures",
]

#: The voicings :func:`chorus` offers, in menu order.
CHORUS_MODES = ("chorus", "ensemble", "vibrato")

#: The paths :func:`flanger` offers, in menu order.
FLANGER_MODES = ("classic", "through-zero")

#: The movements :func:`tremolo` offers, in menu order.
TREMOLO_MODES = ("tremolo", "auto-pan")

#: Samples a chunked read or one pass of a per-sample loop handles at a time.
CHUNK = 1 << 16

#: Shortest feedback delay, in samples, run as numpy blocks rather than sample by sample.
BLOCK_MIN = 48

#: Seconds a jump in a stepped or sawtooth LFO is smoothed over.
DECLICK = 0.002

#: Points of one LFO cycle handed to a panel.
CYCLE_POINTS = 48

#: Most cycles of a random LFO whose values are handed to a panel.
STEP_POINTS = 4096

#: LFO shapes that jump, smoothed over :data:`DECLICK`.
_JUMPS = ("square", "saw up", "saw down", "random steps")

#: Ensemble: the fast wobble as a multiple of the rate and as a share of the depth.
_ENSEMBLE_FAST = 8.0
_ENSEMBLE_FAST_DEPTH = 0.06

#: Ensemble: hertz above which the copies are softened.
_ENSEMBLE_CUT = 8000.0

#: Chorus: spread of the voices' delays and rates either side of the set value.
_VOICE_DELAY_SPREAD = 0.5
_VOICE_RATE_SPREAD = 0.2


def lfo_rate(rate: float, bpm: float, division: str) -> float:
    """The LFO rate, from a tempo and a note length when ``bpm`` is above 0.

    Args:
        rate: Hertz used when ``bpm`` is 0.
        bpm: Beats per minute, or 0.
        division: A name from ``modulation.DIVISIONS``.

    Returns:
        Hertz.
    """
    if float(bpm) > 0.0:
        return modulation.rate_of_division(division, float(bpm))
    return float(rate)


def channel_phases(channels: int, spread: float) -> list[float]:
    """The LFO phase of each channel, the first at 0 and the last ``spread`` of a cycle ahead.

    Args:
        channels: Channels in the clip.
        spread: Share of a cycle, 0 to 1.

    Returns:
        One phase per channel.
    """
    if channels < 2:
        return [0.0]
    return [float(spread) * index / (channels - 1) for index in range(channels)]


def swept_lfo(count: int, sample_rate: int, rate: float, shape: str, phase: float, seed: int = 0) -> np.ndarray:
    """An LFO, the jumps of a stepped or sawtooth shape smoothed over :data:`DECLICK`.

    Args:
        count: Samples to make.
        sample_rate: Samples per second.
        rate: Cycles per second.
        shape: A name from ``modulation.LFO_SHAPES``.
        phase: Where the first cycle starts, as a share of a cycle.
        seed: Seed of the random shapes.

    Returns:
        A float64 array of ``count`` values from ``-1`` to ``1``.
    """
    values = modulation.lfo(count, sample_rate, rate, shape, float(phase) % 1.0, seed)
    if shape not in _JUMPS or count < 2:
        return values
    from scipy.signal import lfilter

    pole = math.exp(-1.0 / max(1.0, DECLICK * sample_rate))
    smoothed, _ = lfilter([1.0 - pole], [1.0, -pole], values, zi=[pole * values[0]])
    return smoothed


def cycle(shape: str) -> list[float]:
    """One cycle of an LFO shape at :data:`CYCLE_POINTS` points, for a panel.

    Args:
        shape: A name from ``modulation.LFO_SHAPES``.

    Returns:
        Values from ``-1`` to ``1``, rounded to 4 places.
    """
    values = modulation.lfo(CYCLE_POINTS, CYCLE_POINTS, 1.0, shape, 0.0, 0)
    return [round(float(value), 4) for value in values]


def random_steps(shape: str, rate: float, seconds: float) -> list[float] | None:
    """The value each cycle of a random LFO starts from, as :func:`swept_lfo` draws them, for a panel.

    Args:
        shape: A name from ``modulation.LFO_SHAPES``.
        rate: Cycles per second.
        seconds: Length of the clip.

    Returns:
        One value per cycle from the first, rounded to 4 places; None for a repeating shape or
        past :data:`STEP_POINTS` cycles.
    """
    if shape not in ("random steps", "random smooth"):
        return None
    count = int(math.floor(max(0.0, float(seconds)) * max(0.0, float(rate)))) + 2
    if count > STEP_POINTS:
        return None
    return [round(float(value), 4) for value in np.random.default_rng(0).uniform(-1.0, 1.0, count)]


def blended(dry: np.ndarray, wet: np.ndarray, mix: float, output_db: float) -> np.ndarray:
    """A dry and a wet signal crossfaded, then a gain applied.

    Args:
        dry: The signal as it arrived.
        wet: The effect, same shape.
        mix: 0 = dry only; 1 = wet only.
        output_db: Gain applied to the result.

    Returns:
        ``(dry * (1 - mix) + wet * mix) * gain``.
    """
    gain = 10.0 ** (float(output_db) / 20.0)
    result = wet * float(mix)
    result += dry * (1.0 - float(mix))
    if gain != 1.0:
        result *= gain
    return result


def applied(wave: torch.Tensor, mix: float, output_db: float, effect) -> torch.Tensor:
    """Every clip of a batch through an effect, crossfaded with its dry signal.

    Args:
        wave: ``(batch, channels, samples)`` float32.
        mix: 0 = dry only, the effect never run; 1 = effect only.
        output_db: Gain applied to the result.
        effect: Called with one clip as ``(channels, samples)`` float64; answers its wet signal
            in the same shape.

    Returns:
        ``(batch, channels, samples)`` float32; the input itself, copied, at ``mix`` 0 and
        ``output_db`` 0.
    """
    gain = 10.0 ** (float(output_db) / 20.0)
    if float(mix) <= 0.0:
        if gain == 1.0:
            return wave.clone()
        return (wave.double() * gain).to(torch.float32)
    results = []
    for index in range(int(wave.shape[0])):
        dry = wave[index].double().numpy()
        wet = effect(dry)
        results.append(torch.from_numpy(blended(dry, wet, mix, output_db).astype(np.float32)))
    return torch.stack(results, dim=0)


def read(signal: np.ndarray, lag: np.ndarray) -> np.ndarray:
    """A signal read back at a moving fractional delay, a negative delay reading ahead.

    Args:
        signal: ``(channels, samples)``.
        lag: Samples of delay, one per sample.

    Returns:
        ``(channels, samples)`` on a Catmull-Rom cubic, silent where the read falls outside
        the signal.
    """
    channels, length = signal.shape
    out = np.empty((channels, length))
    for start in range(0, length, CHUNK):
        stop = min(length, start + CHUNK)
        part = lag[start:stop]
        low = math.floor(float(part.min()))
        high = math.ceil(float(part.max()))
        first = start - high - 2
        last = max(stop, stop - low + 3)
        window = np.zeros((channels, last - first))
        begin, end = max(0, first), min(length, last)
        if end > begin:
            window[:, begin - first:end - first] = signal[:, begin:end]
        shift = max(0, -low)
        offset = start - first + shift
        lags = np.zeros(last - first)
        lags[offset:offset + stop - start] = part + shift
        out[:, start:stop] = delays.read_delayed(window, lags, cubic=True)[:, offset:offset + stop - start]
    return out


def recirculated(source: np.ndarray, taps, shortest: float, longest: float, feedback: float) -> np.ndarray:
    """The signal running through a delay line that feeds its own averaged taps back in.

    Args:
        source: ``(samples,)`` entering the line.
        taps: Called as ``taps(start, stop)``; answers ``(count, stop - start)`` tap delays in
            samples.
        shortest: Least delay any tap reaches, in samples.
        longest: Most delay any tap reaches, in samples.
        feedback: Share of the averaged taps fed back.

    Returns:
        ``(samples,)`` with ``line[n] = source[n] + feedback * mean(taps of line)``, each tap read
        at least one sample back on a straight line between samples.
    """
    length = int(source.shape[0])
    keep = float(feedback)
    floor = max(1.0, float(shortest))
    ceiling = max(floor, float(longest))
    line = np.zeros(length)
    span = int(math.floor(floor))
    if span >= BLOCK_MIN:
        for first in range(0, length, span):
            last = min(length, first + span)
            lags = np.clip(taps(first, last), floor, ceiling)
            where = np.arange(first, last, dtype=np.float64) - lags
            base = np.ceil(where).astype(np.int64) - 1
            frac = where - base
            early = np.where(base >= 0, line[np.maximum(base, 0)], 0.0)
            late = np.where(base >= -1, line[np.maximum(base + 1, 0)], 0.0)
            echo = (early + frac * (late - early)).mean(axis=0)
            line[first:last] = source[first:last] + keep * echo
        return line
    history = int(math.ceil(ceiling)) + 2
    buffer = [0.0] * history
    for start in range(0, length, CHUNK):
        stop = min(length, start + CHUNK)
        lags = np.clip(taps(start, stop), floor, ceiling)
        where = np.arange(start, stop, dtype=np.float64) - lags
        base = np.ceil(where).astype(np.int64) - 1
        index = base - (start - history)
        frac = where - base
        fed = source[start:stop].tolist()
        append = buffer.append
        if lags.shape[0] == 1:
            for value, at, part in zip(fed, index[0].tolist(), frac[0].tolist()):
                early = buffer[at]
                append(value + keep * (early + part * (buffer[at + 1] - early)))
        else:
            share = keep / lags.shape[0]
            for value, ats, parts in zip(fed, index.T.tolist(), frac.T.tolist()):
                total = 0.0
                for at, part in zip(ats, parts):
                    early = buffer[at]
                    total += early + part * (buffer[at + 1] - early)
                append(value + share * total)
        line[start:stop] = buffer[history:]
        buffer = buffer[-history:]
    return line


def _voices(mode: str, rate: float, depth: float, delay: float, voices: int, phase: float) -> list[tuple]:
    """The taps of one channel as ``(centre, depth, rate, phase, fast depth, fast rate, fast phase)``."""
    if mode == "vibrato":
        return [(0.0, depth, rate, phase, 0.0, 0.0, 0.0)]
    if mode == "ensemble":
        fast = rate * _ENSEMBLE_FAST
        return [
            (delay, depth, rate, phase + k / 3.0, depth * _ENSEMBLE_FAST_DEPTH, fast, phase + k / 3.0)
            for k in range(3)
        ]
    count = max(2, int(voices))
    table = []
    for k in range(count):
        offset = k / (count - 1) - 0.5
        table.append((
            delay * (1.0 + _VOICE_DELAY_SPREAD * offset), depth, rate * (1.0 + _VOICE_RATE_SPREAD * offset),
            phase + k / count, 0.0, 0.0, 0.0,
        ))
    return table


def _voice_lag(voice: tuple, start: int, count: int, sample_rate: int, floor: float | None) -> np.ndarray:
    """One tap's delay in samples over ``count`` samples from ``start``."""
    centre, depth, rate, phase, fast_depth, fast_rate, fast_phase = voice
    lag = centre + depth * modulation.lfo(count, sample_rate, rate, "sine", (phase + start * rate / sample_rate) % 1.0)
    if fast_depth:
        lag = lag + fast_depth * modulation.lfo(
            count, sample_rate, fast_rate, "sine", (fast_phase + start * fast_rate / sample_rate) % 1.0,
        )
    return lag if floor is None else np.maximum(lag, floor)


def chorus(signal: np.ndarray, sample_rate: int, mode: str, rate: float, depth_ms: float, delay_ms: float,
           voices: int, spread: float, feedback: float) -> np.ndarray:
    """The wet signal of a chorus, a three-phase ensemble or a vibrato.

    Args:
        signal: ``(channels, samples)``.
        sample_rate: Samples per second.
        mode: A name from :data:`CHORUS_MODES`.
        rate: LFO hertz.
        depth_ms: Milliseconds the delay swings either side of its centre.
        delay_ms: Centre delay in milliseconds; ``vibrato`` is read centred on the present.
        voices: Copies in ``chorus`` mode.
        spread: Share of a cycle the last channel's LFO runs ahead of the first.
        feedback: Share of the copies fed back, ``chorus`` mode only.

    Returns:
        ``(channels, samples)``, the copies averaged, without the dry signal.
    """
    channels, length = signal.shape
    per_ms = sample_rate / 1000.0
    depth, delay = max(0.0, float(depth_ms)) * per_ms, max(0.0, float(delay_ms)) * per_ms
    floor = None if mode == "vibrato" else 0.0
    keep = float(feedback) if mode == "chorus" else 0.0
    wet = np.empty((channels, length))
    for channel, phase in enumerate(channel_phases(channels, spread)):
        table = _voices(mode, float(rate), depth, delay, voices, phase)
        line = signal[channel]
        if keep:
            reach = [(v[0] - v[1] - v[4], v[0] + v[1] + v[4]) for v in table]

            def taps(start, stop, table=table):
                return np.stack([_voice_lag(v, start, stop - start, sample_rate, 0.0) for v in table])

            line = recirculated(
                line * math.sqrt(1.0 - keep * keep), taps,
                min(r[0] for r in reach), max(r[1] for r in reach), keep,
            )
        total = np.zeros(length)
        for voice in table:
            total += read(line[None], _voice_lag(voice, 0, length, sample_rate, floor))[0]
        wet[channel] = total / len(table)
    if mode == "ensemble":
        from scipy.signal import sosfilt

        wet = sosfilt(filters.cut("high cut", _ENSEMBLE_CUT, 12, sample_rate), wet, axis=-1)
    return wet


def chorus_figures(sample_rate: int, mode: str, rate: float, depth_ms: float, delay_ms: float, voices: int) -> dict:
    """The figures of a chorus for a panel.

    Args:
        sample_rate: Samples per second.
        mode: A name from :data:`CHORUS_MODES`.
        rate: LFO hertz.
        depth_ms: Swing either side of the centre, in milliseconds.
        delay_ms: Centre delay in milliseconds.
        voices: Copies in ``chorus`` mode.

    Returns:
        ``voices`` used, the ``min_ms`` and ``max_ms`` the copies reach behind the dry signal,
        and ``pitch_cents``, the widest pitch swing.
    """
    table = _voices(mode, float(rate), float(depth_ms), float(delay_ms), voices, 0.0)
    floor = -math.inf if mode == "vibrato" else 0.0
    low = min(max(floor, v[0] - v[1] - v[4]) for v in table)
    high = max(v[0] + v[1] + v[4] for v in table)
    slope = max(2.0 * math.pi * (v[1] * v[2] + v[4] * v[5]) / 1000.0 for v in table)
    return {
        "voices": len(table),
        "min_ms": round(low, 4),
        "max_ms": round(high, 4),
        "pitch_cents": round(1200.0 * math.log2(1.0 + min(0.99, slope)), 3),
    }


def flanger(signal: np.ndarray, sample_rate: int, mode: str, rate: float, shape: str, depth: float,
            manual_ms: float, feedback: float, spread: float) -> np.ndarray:
    """The wet signal of a flanger, its delay swept around ``manual_ms``.

    Args:
        signal: ``(channels, samples)``.
        sample_rate: Samples per second.
        mode: A name from :data:`FLANGER_MODES`.
        rate: LFO hertz.
        shape: A name from ``modulation.LFO_SHAPES``.
        depth: Share of the centre delay swept either side, 0 to 1.
        manual_ms: Centre delay in milliseconds.
        feedback: Share of the delayed copy fed back, -0.95 to 0.95.
        spread: Share of a cycle the last channel's LFO runs ahead of the first.

    Returns:
        ``(channels, samples)``. ``through-zero`` answers the copy inverted and read against the
        dry signal delayed by ``manual_ms``, then moved back into time with the input.
    """
    channels, length = signal.shape
    centre = max(0.0, float(manual_ms)) * sample_rate / 1000.0
    keep = float(feedback)
    wet = np.empty((channels, length))
    for channel, phase in enumerate(channel_phases(channels, spread)):
        sweep = centre * (1.0 + float(depth) * swept_lfo(length, sample_rate, rate, shape, phase))
        line = signal[channel]
        if keep:
            line = recirculated(
                line * math.sqrt(1.0 - keep * keep), lambda start, stop: sweep[None, start:stop],
                float(sweep.min()), float(sweep.max()), keep,
            )
        if mode == "through-zero":
            wet[channel] = -read(line[None], sweep - centre)[0]
        else:
            wet[channel] = read(line[None], sweep)[0]
    return wet


def flanger_figures(mode: str, depth: float, manual_ms: float) -> dict:
    """The figures of a flanger for a panel.

    Args:
        mode: A name from :data:`FLANGER_MODES`.
        depth: Share of the centre delay swept either side.
        manual_ms: Centre delay in milliseconds.

    Returns:
        ``min_ms`` and ``max_ms`` the copy reaches against the dry signal, and ``notch_low_hz``
        and ``notch_high_hz``, the range of the first notch, None where it leaves the spectrum.
    """
    centre = max(0.0, float(manual_ms))
    swing = centre * float(depth)
    low, high = centre - swing, centre + swing
    if mode == "through-zero":
        low, high = -swing, swing
        nearest, farthest = 0.0, swing
        first = 1.0
    else:
        nearest, farthest = low, high
        first = 0.5
    return {
        "min_ms": round(low, 4),
        "max_ms": round(high, 4),
        "notch_low_hz": round(first * 1000.0 / farthest, 2) if farthest > 0 else None,
        "notch_high_hz": round(first * 1000.0 / nearest, 2) if nearest > 0 else None,
    }


def phaser_notches(stages: int, corner: float, sample_rate: int) -> list[float]:
    """Where a chain of matched first order allpass stages cancels against the dry signal.

    Args:
        stages: Allpass stages.
        corner: Hertz at which each stage shifts the phase a quarter turn.
        sample_rate: Samples per second.

    Returns:
        Notch frequencies in hertz, lowest first.
    """
    warped = math.tan(math.pi * corner / sample_rate)
    found = []
    for k in range(int(stages)):
        if 2 * k + 1 >= int(stages):
            break
        angle = (2 * k + 1) * math.pi / (2 * int(stages))
        found.append(round(sample_rate / math.pi * math.atan(warped * math.tan(angle)), 2))
    return found


def _sweep_bounds(low_hz: float, high_hz: float, sample_rate: int) -> tuple[float, float]:
    """The phaser sweep's ends in hertz, in order and below the Nyquist limit."""
    top = 0.45 * sample_rate
    low = min(max(1.0, float(low_hz)), top)
    high = min(max(1.0, float(high_hz)), top)
    return (low, high) if low <= high else (high, low)


def phaser(signal: np.ndarray, sample_rate: int, stages: int, rate: float, low_hz: float, high_hz: float,
           feedback: float, spread: float, shape: str) -> np.ndarray:
    """The wet signal of a phaser: matched allpass stages swept on a log scale.

    Args:
        signal: ``(channels, samples)``.
        sample_rate: Samples per second.
        stages: First order allpass stages, 1 or more.
        rate: LFO hertz.
        low_hz: One end of the sweep.
        high_hz: The other end.
        feedback: Share of the last stage fed into the first, -0.95 to 0.95.
        spread: Share of a cycle the last channel's LFO runs ahead of the first.
        shape: A name from ``modulation.LFO_SHAPES``.

    Returns:
        ``(channels, samples)``, the phase shifted signal without the dry.
    """
    channels, length = signal.shape
    low, high = _sweep_bounds(low_hz, high_hz, sample_rate)
    count = max(1, int(stages))
    keep = float(feedback)
    scale = math.sqrt(1.0 - keep * keep)
    order = range(1, count + 1)
    wet = np.empty((channels, length))
    for channel, phase in enumerate(channel_phases(channels, spread)):
        position = 0.5 * (swept_lfo(length, sample_rate, rate, shape, phase) + 1.0)
        state = [0.0] * (count + 1)
        last = 0.0
        for start in range(0, length, CHUNK):
            stop = min(length, start + CHUNK)
            warped = np.tan(np.pi * low * (high / low) ** position[start:stop] / sample_rate)
            coefficients = ((warped - 1.0) / (warped + 1.0)).tolist()
            fed = (signal[channel, start:stop] * scale).tolist()
            result = []
            append = result.append
            for value, a in zip(fed, coefficients):
                v = value + keep * last
                prev = state[0]
                state[0] = v
                for i in order:
                    held = state[i]
                    v = a * (v - held) + prev
                    prev = held
                    state[i] = v
                last = v
                append(v)
            wet[channel, start:stop] = result
    return wet


def phaser_figures(stages: int, low_hz: float, high_hz: float, sample_rate: int) -> dict:
    """The figures of a phaser for a panel.

    Args:
        stages: Allpass stages.
        low_hz: One end of the sweep.
        high_hz: The other end.
        sample_rate: Samples per second.

    Returns:
        ``low_hz`` and ``high_hz`` as swept, ``notches``, and ``notches_low`` and
        ``notches_high``, the notch frequencies at either end of the sweep.
    """
    low, high = _sweep_bounds(low_hz, high_hz, sample_rate)
    at_low = phaser_notches(stages, low, sample_rate)
    return {
        "low_hz": round(low, 2),
        "high_hz": round(high, 2),
        "notches": len(at_low),
        "notches_low": at_low,
        "notches_high": phaser_notches(stages, high, sample_rate),
    }


def tremolo(signal: np.ndarray, sample_rate: int, mode: str, rate: float, depth: float, shape: str,
            stereo_phase: float) -> np.ndarray:
    """The wet signal of a tremolo or an auto-pan.

    Args:
        signal: ``(channels, samples)``; ``auto-pan`` needs 2 or more channels.
        sample_rate: Samples per second.
        mode: A name from :data:`TREMOLO_MODES`.
        rate: LFO hertz.
        depth: 0 = no change; 1 = down to silence, or from one side fully to the other.
        shape: A name from ``modulation.LFO_SHAPES``.
        stereo_phase: Share of a cycle the last channel's LFO runs ahead, ``tremolo`` only.

    Returns:
        ``(channels, samples)``. ``auto-pan`` moves the first two channels at constant power,
        unity at the centre; further channels pass through.
    """
    channels, length = signal.shape
    amount = min(1.0, max(0.0, float(depth)))
    if mode == "auto-pan":
        position = amount * swept_lfo(length, sample_rate, rate, shape, 0.0)
        angle = (position + 1.0) * (math.pi / 4.0)
        wet = signal.copy()
        wet[0] = signal[0] * (math.sqrt(2.0) * np.cos(angle))
        wet[1] = signal[1] * (math.sqrt(2.0) * np.sin(angle))
        return wet
    wet = np.empty((channels, length))
    for channel, phase in enumerate(channel_phases(channels, stereo_phase)):
        level = swept_lfo(length, sample_rate, rate, shape, phase)
        wet[channel] = signal[channel] * (1.0 - amount * 0.5 * (1.0 - level))
    return wet


def tremolo_figures(mode: str, depth: float, mix: float) -> dict:
    """The figures of a tremolo for a panel.

    Args:
        mode: A name from :data:`TREMOLO_MODES`.
        depth: Depth as set.
        mix: Mix as set.

    Returns:
        ``floor_db``, the quietest a ``tremolo`` dips to, None for silence or an ``auto-pan``.
    """
    lowest = 1.0 - min(1.0, max(0.0, float(depth))) * min(1.0, max(0.0, float(mix)))
    floor = round(20.0 * math.log10(lowest), 2) if mode == "tremolo" and lowest > 0.0 else None
    return {"floor_db": floor}
