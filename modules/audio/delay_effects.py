"""Feedback delays in four voicings, and the wet and dry handling every space effect shares.

Signals are numpy arrays shaped ``(channels, samples)``, float32 outside a loop and float64 inside
one. Delays are in samples unless named in seconds.
"""

from __future__ import annotations

import math

import numpy as np

from . import filters, modulation

__all__ = [
    "MAX_AUTO_TAIL",
    "MAX_DELAY_SECONDS",
    "MIN_DELAY_SECONDS",
    "MODES",
    "REPEATS",
    "auto_tail",
    "blended",
    "cut_sections",
    "delay_seconds",
    "delayed",
    "ducked",
    "filtered",
    "repeat_figures",
    "ring_read",
    "saturated",
    "widened",
]

#: The voicings :func:`delayed` offers, in menu order.
MODES = ("digital", "tape", "BBD", "ping-pong")

#: Shortest and longest time between repeats, in seconds.
MIN_DELAY_SECONDS = 0.001
MAX_DELAY_SECONDS = 10.0

#: Longest automatic tail, in seconds.
MAX_AUTO_TAIL = 30.0

#: Most repeats a delay reports.
REPEATS = 16

#: Level a reported repeat stays above, in dB.
REPEAT_FLOOR_DB = -60.0

#: Slope of the cut filters, in dB per octave.
CUT_SLOPE = 12

#: Samples a whole signal is filtered in per pass.
CHUNK = 1 << 18

#: Saturation reached at full drive, as the gain of the curve's knee.
DRIVE_SCALE = 4.0

#: Tape: saturation at zero drive, speed wander shares for wow and flutter, and their rates in hertz.
TAPE_DRIVE = 0.8
TAPE_WOW = 0.003
TAPE_FLUTTER = 0.0008
TAPE_WOW_HZ = 0.55
TAPE_DRIFT_HZ = 1.3
TAPE_FLUTTER_HZ = 6.5

#: Tape: the head loss lowpass and the low cut every pass goes through, in hertz.
TAPE_TOP = 5500.0
TAPE_BOTTOM = 45.0

#: Bucket brigade: corner times delay, and the corner's limits, in hertz.
BBD_CLOCK = 1200.0
BBD_LOWEST = 2500.0
BBD_HIGHEST = 9000.0

#: Bucket brigade: compander time constant in seconds, level reference, floor, grit and hiss.
BBD_ENVELOPE = 0.01
BBD_REFERENCE = 0.25
BBD_FLOOR = 1e-4
BBD_GRIT = 1.6
BBD_NOISE = 3e-4
BBD_SEED = 7

#: Ducking: attack and release in seconds, and the share of the loudest level that ducks in full.
DUCK_ATTACK = 0.01
DUCK_RELEASE = 0.3
DUCK_FULL = 0.3


def delay_seconds(time_ms: float, bpm: float, division: str) -> float:
    """The time between repeats, from milliseconds or from a tempo and a note length.

    Args:
        time_ms: Milliseconds, read when ``bpm`` is 0.
        bpm: Beats per minute; above 0 the delay is one ``division`` long.
        division: A name from :data:`modulation.DIVISIONS`.

    Returns:
        Seconds.

    Raises:
        ValueError: The time falls outside :data:`MIN_DELAY_SECONDS` to :data:`MAX_DELAY_SECONDS`.
    """
    if float(bpm) > 0.0:
        seconds = modulation.seconds_of_division(division, float(bpm))
        asked = f"{division} at {float(bpm):g} bpm"
    else:
        seconds = float(time_ms) / 1000.0
        asked = f"time_ms {float(time_ms):g}"
    if not MIN_DELAY_SECONDS <= seconds <= MAX_DELAY_SECONDS:
        raise ValueError(
            f"{asked} is a delay of {seconds:.3f} s; a delay runs {MIN_DELAY_SECONDS * 1000:g} ms to "
            f"{MAX_DELAY_SECONDS:g} s. Raise bpm, pick a shorter division, or set bpm to 0 and use time_ms."
        )
    return seconds


def auto_tail(seconds: float, feedback: float) -> float:
    """Seconds the repeats take to fall 60 dB, held to :data:`MAX_AUTO_TAIL`.

    Args:
        seconds: Time between repeats.
        feedback: Share of each repeat fed into the next.

    Returns:
        Seconds.
    """
    keep = float(feedback)
    count = 1 if keep <= 0.0 else max(1, math.ceil(-3.0 / math.log10(min(keep, 0.999))))
    return min(MAX_AUTO_TAIL, seconds * count)


def repeat_figures(seconds: float, feedback: float) -> tuple[list[float], list[float]]:
    """When the first repeats land and their level, those above :data:`REPEAT_FLOOR_DB`.

    Args:
        seconds: Time between repeats.
        feedback: Share of each repeat fed into the next.

    Returns:
        ``(times, levels)``: seconds after the dry, and dB against the first repeat.
    """
    times, levels = [], []
    keep = max(0.0, float(feedback))
    for index in range(REPEATS):
        level = 0.0 if index == 0 else (20.0 * math.log10(keep) * index if keep > 0.0 else -math.inf)
        if level < REPEAT_FLOOR_DB:
            break
        times.append(round(seconds * (index + 1), 6))
        levels.append(round(level, 2))
    return times, levels


def saturated(signal: np.ndarray, amount: float) -> np.ndarray:
    """A soft clip whose small signal gain is 1.

    Args:
        signal: Samples.
        amount: Gain of the knee; 0 leaves the signal as it is.

    Returns:
        ``tanh(amount * x) / amount``.
    """
    if amount <= 1e-6:
        return signal
    return np.tanh(signal * amount) / amount


def cut_sections(rate: int, low_cut: float, high_cut: float) -> np.ndarray | None:
    """A low cut and a high cut as one chain of second order sections.

    Args:
        rate: Samples per second.
        low_cut: Hertz below which the signal is cut; 0 for none.
        high_cut: Hertz above which the signal is cut; 0 for none.

    Returns:
        ``(n, 6)`` sections, or None when neither cut is set.
    """
    chain = []
    if float(low_cut) > 0.0:
        chain.append(filters.cut("low cut", low_cut, CUT_SLOPE, rate))
    if 0.0 < float(high_cut) < 0.49 * rate:
        chain.append(filters.cut("high cut", high_cut, CUT_SLOPE, rate))
    return np.vstack(chain) if chain else None


def filtered(signal: np.ndarray, sections: np.ndarray | None) -> np.ndarray:
    """A signal through second order sections, in place, a chunk at a time.

    Args:
        signal: ``(channels, samples)`` float array, overwritten.
        sections: ``(n, 6)`` sections, or None to leave the signal as it is.

    Returns:
        ``signal``.
    """
    if sections is None:
        return signal
    from scipy.signal import sosfilt

    state = np.zeros((len(sections), signal.shape[0], 2))
    for start in range(0, signal.shape[-1], CHUNK):
        piece, state = sosfilt(sections, signal[:, start:start + CHUNK], axis=-1, zi=state)
        signal[:, start:start + CHUNK] = piece
    return signal


def widened(wet: np.ndarray, width: float) -> np.ndarray:
    """Channels pulled toward their mean, in place.

    Args:
        wet: ``(channels, samples)``, overwritten.
        width: 0 for every channel the same, 1 to leave them as they are.

    Returns:
        ``wet``.
    """
    if wet.shape[0] < 2 or float(width) >= 1.0:
        return wet
    mid = wet.mean(axis=0, keepdims=True)
    wet -= mid
    wet *= np.float32(max(0.0, float(width)))
    wet += mid
    return wet


def ducked(wet: np.ndarray, dry: np.ndarray, rate: int, duck: float) -> np.ndarray:
    """A wet signal turned down while the dry one is loud, in place.

    Args:
        wet: ``(channels, samples)``, overwritten.
        dry: ``(channels, samples)`` whose level drives the dip; may be shorter than ``wet``.
        rate: Samples per second.
        duck: 0 for none, 1 for silence while the dry runs at :data:`DUCK_FULL` of its loudest.

    Returns:
        ``wet``.
    """
    amount = min(1.0, max(0.0, float(duck)))
    if amount <= 0.0:
        return wet
    span = min(dry.shape[-1], wet.shape[-1])
    level = np.zeros((1, wet.shape[-1]))
    level[0, :span] = np.abs(dry[:, :span]).max(axis=0)
    follow = modulation.envelope(level, rate, DUCK_ATTACK, DUCK_RELEASE)
    loudest = float(follow.max()) if follow.size else 0.0
    if loudest <= 0.0:
        return wet
    gain = 1.0 - amount * np.clip(follow / (DUCK_FULL * loudest), 0.0, 1.0)
    wet *= gain.astype(np.float32)[None, :]
    return wet


def blended(dry: np.ndarray, wet: np.ndarray | None, mix: float, output_db: float, total: int,
            channels: int | None = None) -> np.ndarray:
    """The dry clip and the wet signal blended, padded with silence to ``total`` samples.

    Args:
        dry: ``(channels, samples)`` float32; a mono clip is copied to every output channel.
        wet: ``(channels, total)`` float32, or None for the dry clip alone.
        mix: 0 for the dry clip alone, 1 for the wet signal alone.
        output_db: Gain applied to the blend.
        total: Samples in the result.
        channels: Channels in the result; the wet signal's, or the dry clip's, when None.

    Returns:
        ``(channels, total)`` float32.
    """
    if channels is None:
        channels = dry.shape[0] if wet is None else wet.shape[0]
    if dry.shape[0] != channels:
        dry = np.broadcast_to(dry[:1], (channels, dry.shape[-1]))
    out = np.zeros((channels, total), dtype=np.float32)
    span = min(total, dry.shape[-1])
    share = min(1.0, max(0.0, float(mix)))
    if wet is None or share <= 0.0:
        out[:, :span] = dry[:, :span]
    else:
        np.multiply(wet[:channels], np.float32(share), out=out)
        out[:, :span] += dry[:, :span] * np.float32(1.0 - share)
    gain = 10.0 ** (float(output_db) / 20.0)
    if gain != 1.0:
        out *= np.float32(gain)
    return out


def ring_read(ring: np.ndarray, where: np.ndarray, size: int) -> np.ndarray:
    """A ring buffer read at fractional sample times through a Catmull-Rom cubic.

    Args:
        ring: ``(rows, size)``, sample time ``t`` held at ``t % size``.
        where: Times to read, ``(samples,)`` for every row alike or ``(rows, samples)``.
        size: Length of the ring.

    Returns:
        ``(rows, samples)`` float64.
    """
    floor = np.floor(where)
    frac = where - floor
    base = floor.astype(np.int64)
    rows = np.arange(ring.shape[0])[:, None]

    def tap(offset: int) -> np.ndarray:
        index = (base + offset) % size
        return ring[:, index] if where.ndim == 1 else ring[rows, index]

    before, here, after, beyond = tap(-1), tap(0), tap(1), tap(2)
    f2 = frac * frac
    f3 = f2 * frac
    return (
        here
        + 0.5 * frac * (after - before)
        + f2 * (before - 2.5 * here + 2.0 * after - 0.5 * beyond)
        + f3 * (1.5 * (here - after) + 0.5 * (beyond - before))
    )


def _tape_lag(total: int, rate: int, delay: float) -> np.ndarray:
    """Samples of delay at each sample of a tape loop whose speed wanders with wow and flutter."""
    speed = (
        1.0
        + TAPE_WOW * (0.7 * modulation.lfo(total, rate, TAPE_WOW_HZ)
                      + 0.3 * modulation.lfo(total, rate, TAPE_DRIFT_HZ, "random smooth", seed=11))
        + TAPE_FLUTTER * modulation.lfo(total, rate, TAPE_FLUTTER_HZ, phase=0.25)
    )
    travelled = np.concatenate([[0.0], np.cumsum(speed[:-1])])
    times = np.arange(total, dtype=np.float64)
    target = travelled - delay
    # The sample recorded where the playback head now reads, the tape before the start running at speed 1.
    recorded = np.where(target < 0.0, target, np.interp(target, travelled, times))
    return times - recorded


def _shaper(mode: str, rate: int, channels: int, seconds: float, low_cut: float, high_cut: float,
            drive: float):
    """The filters, compander and saturation every pass of a delay goes through, keeping their state."""
    from scipy.signal import butter, lfilter, sosfilt

    chain = []
    if mode == "tape":
        chain.append(filters.biquad("high cut", TAPE_TOP, 0.6, 0.0, rate)[None])
        chain.append(filters.biquad("low cut", TAPE_BOTTOM, 0.707, 0.0, rate)[None])
    elif mode == "BBD":
        corner = min(max(BBD_CLOCK / seconds, BBD_LOWEST), BBD_HIGHEST, 0.45 * rate)
        chain.append(butter(4, corner, btype="lowpass", fs=rate, output="sos"))
    user = cut_sections(rate, low_cut, high_cut)
    if user is not None:
        chain.append(user)
    sections = np.vstack(chain) if chain else None
    state = {
        "filter": np.zeros((len(sections), channels, 2)) if sections is not None else None,
        "level": np.zeros((channels, 1)),
    }
    amount = DRIVE_SCALE * float(drive) + (TAPE_DRIVE if mode == "tape" else 0.0)
    hiss = np.random.default_rng(BBD_SEED)
    rise = math.exp(-1.0 / (BBD_ENVELOPE * rate))

    def shape(block: np.ndarray) -> np.ndarray:
        out = block
        if sections is not None:
            out, state["filter"] = sosfilt(sections, out, axis=-1, zi=state["filter"])
        if mode == "BBD":
            level, state["level"] = lfilter([1.0 - rise], [1.0, -rise], np.abs(out), axis=-1, zi=state["level"])
            spread = np.sqrt(np.maximum(level, BBD_FLOOR) / BBD_REFERENCE)
            # Compressed into the bucket chain, clipped and hissed there, then expanded back out.
            squeezed = saturated(out / spread, BBD_GRIT) + BBD_NOISE * hiss.standard_normal(out.shape)
            out = squeezed * spread
        return saturated(out, amount)

    return shape


def _repeats(source: np.ndarray, delay: float, lag: np.ndarray | None, feedback: float, shape,
             swap: bool) -> np.ndarray:
    """The repeats of a feedback delay line, run in blocks that read only finished history."""
    channels, total = source.shape
    if lag is None:
        gap = max(1, int(round(delay)))
        block, reach = gap, gap
    else:
        block = max(1, int(math.floor(float(lag.min()))) - 2)
        reach = int(math.ceil(float(lag.max()))) + 3
    size = reach + block + 8
    ring = np.zeros((channels, size))
    wet = np.zeros((channels, total), dtype=np.float32)
    keep = min(0.99, max(0.0, float(feedback)))
    for start in range(0, total, block):
        stop = min(total, start + block)
        times = np.arange(start, stop)
        if lag is None:
            heard = ring[:, (times - gap) % size]
        else:
            heard = ring_read(ring, times - lag[start:stop], size)
        wet[:, start:stop] = heard
        entering = shape(source[:, start:stop] + keep * heard)
        if swap:
            entering = entering[::-1]
        ring[:, times % size] = entering
    return wet


def delayed(dry: np.ndarray, rate: int, mode: str, seconds: float, feedback: float, low_cut: float,
            high_cut: float, drive: float, width: float, total: int) -> np.ndarray:
    """The repeats a feedback delay makes of one clip, without the clip itself.

    Args:
        dry: ``(channels, samples)`` float32.
        rate: Samples per second.
        mode: A name from :data:`MODES`.
        seconds: Time between repeats.
        feedback: Share of each repeat fed into the next, 0 to 0.98.
        low_cut: Hertz cut below on every pass; 0 for none.
        high_cut: Hertz cut above on every pass; 0 for none.
        drive: Saturation on every pass, 0 to 1.
        width: Spread of the repeats across channels, 0 to 1.
        total: Samples the repeats run to.

    Returns:
        ``(channels, total)`` float32; ``ping-pong`` answers at least two channels.

    Raises:
        ValueError: ``mode`` is not one of :data:`MODES`.
    """
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} is not one of {', '.join(MODES)}.")
    channels, length = dry.shape
    span = min(length, total)
    if mode == "ping-pong":
        source = np.zeros((2, total), dtype=np.float32)
        # The right channel takes the input, and the swap lands the first repeat on the left.
        source[1, :span] = dry[:, :span].mean(axis=0)
    else:
        source = np.zeros((channels, total), dtype=np.float32)
        source[:, :span] = dry[:, :span]
    delay = float(seconds) * rate
    lag = _tape_lag(total, rate, delay) if mode == "tape" else None
    shape = _shaper(mode, rate, source.shape[0], float(seconds), low_cut, high_cut, drive)
    wet = _repeats(source, delay, lag, feedback, shape, swap=mode == "ping-pong")
    widened(wet, width)
    if mode == "ping-pong" and channels > 2:
        wet = np.concatenate([wet, np.zeros((channels - 2, total), dtype=np.float32)], axis=0)
    return wet
