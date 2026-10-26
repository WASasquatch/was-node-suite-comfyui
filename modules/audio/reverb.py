"""Algorithmic rooms, convolution with an impulse response, and the decay curve a reverb is measured by.

Signals are numpy arrays shaped ``(channels, samples)``. Reverberation time is set and measured in the
:data:`BAND` octave.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .delay_effects import CHUNK, filtered, ring_read

__all__ = [
    "BAND",
    "CURVE_POINTS",
    "MAX_AUTO_TAIL",
    "MAX_IMPULSE_SECONDS",
    "ROOMS",
    "Reverb",
    "auto_tail",
    "convolved",
    "decay_figures",
    "prepared_impulse",
]

#: The rooms :class:`Reverb` builds, in menu order.
ROOMS = ("room", "hall", "plate", "chamber", "spring")

#: The octave around 1 kHz a decay is set and measured in, in hertz.
BAND = (707.1, 1414.2)

#: Frequency the loop gains are set at, in hertz.
REFERENCE_HZ = 1000.0

#: Most points a decay curve holds, and the lowest level it reaches in dB.
CURVE_POINTS = 400
CURVE_FLOOR = -120.0

#: Longest impulse response convolved, in seconds.
MAX_IMPULSE_SECONDS = 30.0

#: Longest impulse response a room is measured over, in seconds.
MEASURE_SECONDS = 16.0

#: Longest automatic tail, in seconds.
MAX_AUTO_TAIL = 30.0

#: Delay lines in a room's feedback network.
LINES = 16

#: Fade at the end of a trimmed impulse response: most seconds, and most share of what is kept.
TRIM_FADE = 0.05
TRIM_FADE_SHARE = 0.25

#: Spring: loop time at size 1 in ms, the spread between springs, allpass stages and coefficient.
SPRING_LOOP_MS = 37.0
SPRING_SPREAD = 0.135
SPRING_STAGES = 100
SPRING_COEFFICIENT = 0.65

#: Spring: top of the chirp, the damping corner and the low cut, in hertz.
SPRING_TOP = 4800.0
SPRING_CORNER = 3500.0
SPRING_LOW = 60.0


@dataclass(frozen=True)
class Room:
    """The layout of one room at size 1.

    Attributes:
        lines: Shortest and longest delay line, in milliseconds.
        diffusers: Delays of the input allpasses, in milliseconds.
        diffusion: Gain of every input allpass.
        early: First and last early reflection in milliseconds and their count, or None.
        early_share: Share of the wet energy the early reflections carry.
        corner: Hertz above which damping shortens the decay.
        depth: Most milliseconds the modulation moves a line.
    """

    lines: tuple[float, float]
    diffusers: tuple[float, ...]
    diffusion: float
    early: tuple[float, float, int] | None
    early_share: float
    corner: float
    depth: float


_ROOMS = {
    "room": Room((11.0, 43.0), (1.3, 2.1, 3.4, 5.1), 0.7, (2.5, 48.0, 18), 0.45, 4500.0, 0.25),
    "hall": Room((31.0, 101.0), (2.3, 3.7, 5.9, 8.9), 0.75, (8.0, 105.0, 24), 0.3, 3500.0, 0.5),
    "plate": Room((7.0, 31.0), (1.1, 1.7, 3.9, 5.9), 0.78, None, 0.0, 7000.0, 0.35),
    "chamber": Room((17.0, 61.0), (1.7, 2.9, 4.3, 6.7), 0.72, (4.0, 70.0, 22), 0.4, 4000.0, 0.3),
}


def auto_tail(decay: float, pre_delay: float) -> float:
    """Seconds a room rings past its input, held to :data:`MAX_AUTO_TAIL`.

    Args:
        decay: Seconds to fall 60 dB.
        pre_delay: Seconds before the room answers.

    Returns:
        Seconds.
    """
    return min(MAX_AUTO_TAIL, max(0.0, float(decay)) + max(0.0, float(pre_delay)))


def _is_prime(value: int) -> bool:
    """Whether ``value`` is prime."""
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _prime_at_least(value: int, used: set) -> int:
    """The smallest prime at or above ``value`` not in ``used``."""
    candidate = max(2, int(value))
    while candidate in used or not _is_prime(candidate):
        candidate += 1
    used.add(candidate)
    return candidate


def _reference(rate: int) -> float:
    """The radian frequency loop gains are set at."""
    return 2.0 * math.pi * min(REFERENCE_HZ, 0.25 * rate) / rate


def _shelf_constant(corner: float, rate: int) -> float:
    """The bilinear constant of a first order shelf with its corner at ``corner`` hertz."""
    return math.tan(math.pi * min(float(corner), 0.45 * rate) / rate)


def _shelf(k: float, ratio: float) -> tuple[np.ndarray, np.ndarray]:
    """A first order shelf passing 1 below its corner and ``ratio`` above it, as ``(b, a)``."""
    return (np.array([(ratio + k) / (1.0 + k), (k - ratio) / (1.0 + k)]),
            np.array([1.0, (k - 1.0) / (1.0 + k)]))


def _shelf_gain(k: float, ratio, omega: float):
    """The magnitude of :func:`_shelf` at ``omega`` radians per sample."""
    z = np.exp(-1j * omega)
    low = k * (1.0 + z) / ((1.0 + k) + (k - 1.0) * z)
    return np.abs(low + np.asarray(ratio) * (1.0 - low))


def _loop_gains(periods, decay: float, high_ratio: float, k: float, rate: int):
    """Gain and shelf ratio per loop for ``decay`` at the reference and ``decay * high_ratio`` at the top.

    Args:
        periods: Samples each loop takes to come round.
        decay: Seconds to fall 60 dB at :data:`REFERENCE_HZ`.
        high_ratio: Decay at the top as a share of ``decay``.
        k: Bilinear constant of the damping shelf.
        rate: Samples per second.

    Returns:
        ``(gain, ratio)`` arrays: the loop gain below the shelf, and the shelf's ratio above it.
    """
    periods = np.asarray(periods, dtype=np.float64)
    omega = _reference(rate)
    target = 10.0 ** (-3.0 * periods / (decay * rate))
    top = 10.0 ** (-3.0 * periods / (decay * high_ratio * rate))
    ratio = np.minimum(1.0, top / target)
    for _ in range(8):
        gain = target / _shelf_gain(k, ratio, omega)
        ratio = np.minimum(1.0, top / gain)
    return np.minimum(target / _shelf_gain(k, ratio, omega), 0.99999), ratio


def _allpass(signal: np.ndarray, delay: int, gain: float) -> np.ndarray:
    """A signal through a Schroeder allpass, in place, run as ``delay`` interleaved first order filters."""
    from scipy.signal import lfilter

    channels, length = signal.shape
    span = max(1, CHUNK // delay) * delay
    state = np.zeros((channels, 1, delay))
    for start in range(0, length, span):
        piece = signal[:, start:start + span]
        count = piece.shape[-1]
        rows = -(-count // delay)
        grid = np.zeros((channels, rows * delay))
        grid[:, :count] = piece
        out, state = lfilter([-gain, 1.0], [1.0, -gain], grid.reshape(channels, rows, delay), axis=1, zi=state)
        signal[:, start:start + count] = out.reshape(channels, -1)[:, :count]
    return signal


class Reverb:
    """One room's reverberation, built for a sample rate, a channel count and its settings.

    Attributes:
        impulse_response: ``(channels, samples)`` float32 wet answer to an impulse on every channel.
    """

    def __init__(self, room: str, rate: int, channels: int, decay: float, size: float = 1.0,
                 damping: float = 0.5, modulation: float = 0.3):
        """Build the room and measure its impulse response.

        Args:
            room: A name from :data:`ROOMS`.
            rate: Samples per second.
            channels: Channels in and out.
            decay: Seconds to fall 60 dB in :data:`BAND`.
            size: Scale on every delay in the room.
            damping: 0 for highs lasting as long as the band, 1 for a tenth as long.
            modulation: 0 to 1 of the room's most line movement; the spring takes none.

        Raises:
            ValueError: ``room`` is not one of :data:`ROOMS`.
        """
        if room not in ROOMS:
            raise ValueError(f"room {room!r} is not one of {', '.join(ROOMS)}.")
        self.room = room
        self.rate = int(rate)
        self.channels = int(channels)
        self.decay = max(0.05, float(decay))
        high_ratio = max(0.05, 1.0 - 0.9 * min(1.0, max(0.0, float(damping))))
        self.early = None
        self.early_share = 0.0
        self.diffusers = []
        if room == "spring":
            self._build_spring(max(0.05, float(size)), high_ratio)
        else:
            self._build_network(_ROOMS[room], max(0.05, float(size)), high_ratio, float(modulation))
        self._calibrate()

    def _build_network(self, spec: Room, size: float, high_ratio: float, modulation: float) -> None:
        """Lay out the delay lines, mixing, damping, diffusion and early reflections of a room."""
        from scipy.linalg import hadamard

        rate, channels = self.rate, self.channels
        rng = np.random.default_rng(ROOMS.index(self.room) + 1)
        self.depth = max(0.0, min(1.0, modulation)) * spec.depth * rate / 1000.0
        low, high = spec.lines
        used: set = set()
        least = int(math.ceil(self.depth)) + 8
        self.lengths = np.array([
            _prime_at_least(max(least, int(round(low * (high / low) ** (i / (LINES - 1)) * size * rate / 1000.0))), used)
            for i in range(LINES)
        ], dtype=np.int64)
        self.block = max(1, int(math.floor(float(self.lengths.min()) - self.depth)) - 2)
        self.ring_size = int(self.lengths.max()) + int(math.ceil(self.depth)) + self.block + 8
        self.omegas = rng.uniform(0.2, 1.2, LINES) * 2.0 * math.pi / rate
        self.phases = rng.uniform(0.0, 2.0 * math.pi, LINES)
        k = _shelf_constant(spec.corner, rate)
        self.lowpass = (np.array([k / (1.0 + k), k / (1.0 + k)]), np.array([1.0, (k - 1.0) / (1.0 + k)]))
        self.gain, self.ratio = _loop_gains(self.lengths, self.decay, high_ratio, k, rate)
        walsh = hadamard(LINES).astype(np.float64)
        self.matrix = walsh / math.sqrt(LINES)
        signs_in = rng.choice([-1.0, 1.0], LINES)
        signs_out = rng.choice([-1.0, 1.0], LINES)
        self.inputs = np.zeros((LINES, channels))
        for line in range(LINES):
            if channels <= LINES:
                self.inputs[line, line % channels] = signs_in[line]
            else:
                self.inputs[line, line::LINES] = signs_in[line]
        self.outputs = np.stack([signs_out * walsh[(3 + 5 * c) % LINES] for c in range(channels)]) / math.sqrt(LINES)
        self.diffusers = [(max(1, int(round(ms * size * rate / 1000.0))), spec.diffusion) for ms in spec.diffusers]
        if spec.early is not None:
            first, last, count = spec.early
            self.early = []
            for channel in range(channels):
                taps = np.random.default_rng(100 * (ROOMS.index(self.room) + 1) + channel)
                when = np.sort(taps.uniform(first, last, count))
                # The first reflection lands at the room's first time on every channel.
                when[0] = first
                level = taps.choice([-1.0, 1.0], count) * (first / when) ** 0.8 * taps.uniform(0.6, 1.0, count)
                level /= math.sqrt(float(np.sum(level * level)))
                self.early.append((np.maximum(1, np.round(when * size * rate / 1000.0)).astype(np.int64), level))
            self.early_share = spec.early_share
        self._late = self._late_network

    def _build_spring(self, size: float, high_ratio: float) -> None:
        """Lay out one dispersive spring loop per channel."""
        from scipy.signal import butter, lfilter, sosfilt

        rate = self.rate
        stretch = max(1, int(round(rate / (2.0 * SPRING_TOP))))
        coefficient = SPRING_COEFFICIENT
        longest = SPRING_STAGES * stretch * (1.0 + coefficient) / (1.0 - coefficient)
        length = 1 << int(math.ceil(math.log2(1.5 * longest + 2048)))
        response = np.zeros(length)
        response[0] = 1.0
        forward = np.zeros(stretch + 1)
        forward[0], forward[-1] = coefficient, 1.0
        back = np.zeros(stretch + 1)
        back[0], back[-1] = 1.0, coefficient
        for _ in range(SPRING_STAGES):
            response = lfilter(forward, back, response)
        top = min(0.9 * rate / (2.0 * stretch), 0.45 * rate)
        bands = np.vstack([
            butter(4, top, btype="lowpass", fs=rate, output="sos"),
            butter(2, min(SPRING_LOW, 0.2 * rate), btype="highpass", fs=rate, output="sos"),
        ])
        response = sosfilt(bands, response)
        fade = length // 10
        response[-fade:] *= 0.5 + 0.5 * np.cos(np.linspace(0.0, math.pi, fade))
        omega = _reference(rate)
        ticks = np.arange(length, dtype=np.float64)
        phasor = np.exp(-1j * omega * ticks)
        spectrum = np.sum(response * phasor)
        lag = float(np.real(np.sum(ticks * response * phasor) / spectrum))
        used: set = set()
        self.loops = [
            _prime_at_least(int(round(SPRING_LOOP_MS * size * (1.0 + SPRING_SPREAD * c) * rate / 1000.0)), used)
            for c in range(self.channels)
        ]
        k = _shelf_constant(SPRING_CORNER, rate)
        periods = np.array(self.loops, dtype=np.float64) + lag
        _, ratios = _loop_gains(periods, self.decay, high_ratio, k, rate)
        targets = 10.0 ** (-3.0 * periods / (self.decay * rate))
        self.springs = []
        for loop, ratio, target in zip(self.loops, ratios, targets):
            b, a = _shelf(k, float(ratio))
            shaped = lfilter(b, a, response)
            size_fft = 1 << int(math.ceil(math.log2(loop + length - 1)))
            answer = np.fft.rfft(shaped, size_fft)
            gain = target / abs(np.sum(shaped * phasor))
            gain = min(gain, 0.995 / float(np.abs(answer).max()))
            self.springs.append((loop, gain, answer, size_fft, length))
        self._late = self._late_spring

    def _diffused(self, source: np.ndarray) -> np.ndarray:
        """A copy of ``source`` through the room's input allpasses."""
        out = np.array(source, dtype=np.float32, copy=True)
        for delay, gain in self.diffusers:
            _allpass(out, delay, gain)
        return out

    def _late_network(self, source: np.ndarray) -> np.ndarray:
        """The feedback network's answer to a diffused input, run in blocks shorter than every line."""
        from scipy.signal import lfilter

        length = source.shape[-1]
        lines, size, depth = self.lengths, self.ring_size, self.depth
        ring = np.zeros((LINES, size))
        state = np.zeros((LINES, 1))
        out = np.zeros((self.channels, length), dtype=np.float32)
        rows = np.arange(LINES)[:, None]
        low_b, low_a = self.lowpass
        gain = self.gain[:, None]
        ratio = self.ratio[:, None]
        for start in range(0, length, self.block):
            stop = min(length, start + self.block)
            times = np.arange(start, stop)
            if depth > 0.0:
                swing = depth * np.sin(self.omegas[:, None] * times[None, :] + self.phases[:, None])
                heard = ring_read(ring, times[None, :] - (lines[:, None] + swing), size)
            else:
                heard = ring[rows, (times[None, :] - lines[:, None]) % size]
            low, state = lfilter(low_b, low_a, heard, axis=-1, zi=state)
            damped = gain * (low + ratio * (heard - low))
            out[:, start:stop] = self.outputs @ damped
            ring[:, times % size] = self.matrix @ damped + self.inputs @ source[:, start:stop]
        return out

    def _late_spring(self, source: np.ndarray) -> np.ndarray:
        """Each channel's spring answering its input, the chirp convolved inside the loop a block at a time."""
        length = source.shape[-1]
        out = np.zeros((self.channels, length), dtype=np.float32)
        for channel, (loop, gain, answer, size_fft, taps) in enumerate(self.springs):
            line = np.zeros(length + loop + taps)
            feed = source[channel]
            for start in range(0, length, loop):
                stop = min(length, start + loop)
                entering = feed[start:stop] + gain * line[start:stop]
                passed = np.fft.irfft(np.fft.rfft(entering, size_fft) * answer, size_fft)[:stop - start + taps - 1]
                line[start + loop:start + loop + passed.shape[-1]] += passed
            out[channel] = line[:length]
        return out

    def _early_into(self, wet: np.ndarray, source: np.ndarray, gain: float) -> None:
        """Add each channel's early reflections of ``source`` to ``wet``."""
        length = source.shape[-1]
        for channel, (when, level) in enumerate(self.early):
            for offset, tap in zip(when.tolist(), level.tolist()):
                if offset < length:
                    wet[channel, offset:] += np.float32(tap * gain) * source[channel, :length - offset]

    def _calibrate(self) -> None:
        """Set the early and late levels for a unit energy answer, and keep the impulse response."""
        length = max(1, int(round(min(MEASURE_SECONDS, 0.8 * self.decay + 0.15) * self.rate)))
        impulse = np.zeros((self.channels, length), dtype=np.float32)
        impulse[:, 0] = 1.0
        late = self._late(self._diffused(impulse))
        energy = float(np.mean(np.sum(late.astype(np.float64) ** 2, axis=-1)))
        share = self.early_share if self.early is not None else 0.0
        self.early_gain = math.sqrt(share)
        self.late_gain = math.sqrt((1.0 - share) / energy) if energy > 0.0 else 0.0
        late *= np.float32(self.late_gain)
        if self.early is not None:
            self._early_into(late, impulse, self.early_gain)
        self.impulse_response = late

    def render(self, source: np.ndarray) -> np.ndarray:
        """The room's wet answer to a signal.

        Args:
            source: ``(channels, samples)`` float32, already delayed and padded to the length wanted.

        Returns:
            ``(channels, samples)`` float32.
        """
        wet = self._late(self._diffused(source))
        wet *= np.float32(self.late_gain)
        if self.early is not None:
            self._early_into(wet, source, self.early_gain)
        return wet


def _floor_cut(energy: np.ndarray, rate: int) -> tuple[np.ndarray, float]:
    """Energy cut where a decay meets its noise floor, and the range above that floor in dB."""
    window = max(1, int(round(0.01 * rate)))
    count = energy.shape[-1] // window
    if count < 10:
        return energy, math.inf
    envelope = energy[:count * window].reshape(count, window).mean(axis=1)
    loudest = int(np.argmax(envelope))
    peak = float(envelope[loudest])
    floor = float(envelope[int(count * 0.9):].mean())
    if peak <= 0.0 or floor <= peak * 1e-9:
        return energy, math.inf
    near = envelope[loudest:] <= floor * 10.0 ** 0.5
    end = loudest + int(np.argmax(near)) if near.any() else count
    return energy[:max(1, end) * window], 10.0 * math.log10(peak / floor)


def decay_figures(response: np.ndarray, rate: int, noisy: bool = False) -> dict:
    """The energy decay curve of an impulse response in :data:`BAND`, and the reverberation time read off it.

    Args:
        response: ``(channels, samples)`` impulse response.
        rate: Samples per second.
        noisy: True to cut the response where it meets its noise floor, as a recorded one does.

    Returns:
        ``rt60`` in seconds or None, the ``band`` in hertz, and ``decay`` as ``step`` seconds between
        ``values`` in dB, at most :data:`CURVE_POINTS` of them.
    """
    from scipy.signal import butter, sosfilt

    data = np.asarray(response, dtype=np.float64)
    band = list(BAND)
    if BAND[1] < 0.45 * rate:
        data = sosfilt(butter(3, BAND, btype="bandpass", fs=rate, output="sos"), data, axis=-1)
    else:
        band = [0.0, rate / 2.0]
    energy = np.mean(data * data, axis=0)
    span = math.inf
    if noisy:
        energy, span = _floor_cut(energy, rate)
    remaining = np.cumsum(energy[::-1])[::-1]
    empty = {"rt60": None, "band": band, "decay": {"step": 0.0, "values": []}}
    if remaining.size == 0 or remaining[0] <= 0.0:
        return empty
    with np.errstate(divide="ignore"):
        curve = 10.0 * np.log10(np.maximum(remaining / remaining[0], 1e-300))
    lowest = -35.0 if span >= 45.0 else max(-35.0, -(span - 10.0))
    rt60 = None
    if lowest <= -10.0:
        start = int(np.argmax(curve <= -5.0))
        stop = int(np.argmax(curve <= lowest))
        if curve[start] <= -5.0 and curve[stop] <= lowest and stop > start + 1:
            pick = np.linspace(start, stop, min(2000, stop - start + 1)).astype(np.int64)
            slope = float(np.polyfit(pick / float(rate), curve[pick], 1)[0])
            if slope < 0.0:
                rt60 = round(-60.0 / slope, 4)
    stride = max(1, -(-curve.size // CURVE_POINTS))
    values = np.round(np.maximum(curve[::stride], CURVE_FLOOR), 2)
    return {"rt60": rt60, "band": band, "decay": {"step": round(stride / float(rate), 6), "values": values.tolist()}}


def prepared_impulse(impulse, impulse_rate: int, rate: int, channels: int, length: float, stretch: float,
                     normalise: bool, sections: np.ndarray | None) -> np.ndarray:
    """An impulse response trimmed, brought to a rate and channel count, stretched, levelled and cut.

    Args:
        impulse: ``(batch, channels, samples)`` float tensor.
        impulse_rate: Its samples per second.
        rate: Samples per second wanted.
        channels: Channels wanted.
        length: Seconds kept from its start, faded out at the end; 0 keeps all of it.
        stretch: Factor on its duration, read slower or faster.
        normalise: True to bring every response to unit energy, averaged over its channels.
        sections: Cut filters applied to it, or None.

    Returns:
        ``(batch, channels, samples)`` float64 array at ``rate``.

    Raises:
        ValueError: The response would run past :data:`MAX_IMPULSE_SECONDS`.
    """
    import torch

    from .clip import conformed, samples_at
    from .resample import resampled

    wave = impulse
    have = int(wave.shape[-1])
    if float(length) > 0.0:
        keep = max(1, min(have, samples_at(length, impulse_rate)))
        if keep < have:
            wave = wave[..., :keep].clone()
            fade = max(1, min(int(TRIM_FADE * impulse_rate), int(TRIM_FADE_SHARE * keep)))
            ramp = 0.5 + 0.5 * torch.cos(torch.linspace(0.0, math.pi, fade, dtype=torch.float32))
            wave[..., -fade:] *= ramp
    seconds = int(wave.shape[-1]) / float(impulse_rate) * float(stretch)
    if seconds > MAX_IMPULSE_SECONDS + 1e-6:
        raise ValueError(
            f"impulse runs {seconds:.1f} s after length and stretch; at most {MAX_IMPULSE_SECONDS:g} s is "
            f"convolved. Set length to {MAX_IMPULSE_SECONDS / float(stretch):.1f} or less, or lower stretch."
        )
    wave = conformed(wave, int(impulse_rate), int(rate), int(channels))
    # Every resampling scales the response by the rate ratio, keeping a unit impulse at unit gain.
    scale = float(impulse_rate) / float(rate)
    if abs(float(stretch) - 1.0) > 1e-6:
        stretched = max(1, int(round(rate * float(stretch))))
        wave = resampled(wave, int(rate), stretched)
        scale *= float(rate) / float(stretched)
    data = wave.detach().to(torch.float64).cpu().numpy().copy()
    if scale != 1.0:
        data *= scale
    if normalise:
        energy = np.mean(np.sum(data * data, axis=-1), axis=-1)
        scale = np.where(energy > 0.0, 1.0 / np.sqrt(np.maximum(energy, 1e-300)), 1.0)
        data *= scale[:, None, None]
    if sections is not None:
        for index in range(data.shape[0]):
            filtered(data[index], sections)
    return data


def convolved(dry: np.ndarray, response: np.ndarray, offset: int, total: int) -> np.ndarray:
    """A clip convolved with an impulse response, each channel with its own, a chunk at a time.

    Args:
        dry: ``(channels, samples)`` float32.
        response: ``(channels, taps)`` float64.
        offset: Samples the answer starts late.
        total: Samples in the result.

    Returns:
        ``(channels, total)`` float32.
    """
    from scipy.signal import oaconvolve

    channels, length = dry.shape
    out = np.zeros((channels, total), dtype=np.float32)
    span = max(1 << 19, 4 * response.shape[-1])
    for start in range(0, length, span):
        first = start + offset
        if first >= total:
            break
        piece = dry[:, start:start + span].astype(np.float64)
        answer = oaconvolve(piece, response, mode="full", axes=-1)
        stop = min(total, first + answer.shape[-1])
        out[:, first:stop] += answer[:, :stop - first].astype(np.float32)
    return out
