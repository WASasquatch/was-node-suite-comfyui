"""Procedural fields the affine mask is built from.

Every generator answers a ``(height, width)`` tensor scaled to 0.0 to 1.0, built on the CPU
for the caller to move onto the latent.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .filters import gaussian_blur_depthwise, sobel_grad_mag

__all__ = [
    "CONTENT_PATTERNS",
    "MIN_CONTENT_EXTENT",
    "PATTERNS",
    "bayer_matrix",
    "black_noise",
    "checker",
    "content_field",
    "cross_hatch",
    "dot_screen",
    "field",
    "highpass_white",
    "laplacian_magnitude",
    "local_variance",
    "perlin",
    "poisson_blue",
    "ring_noise",
    "sobel_magnitude",
    "solid",
    "spectral_noise",
    "tile_lines",
    "unsharp",
    "velvet_noise",
    "worley_edges",
]

#: Patterns read off the latent itself rather than generated.
CONTENT_PATTERNS = ("detail_region", "smooth_region", "edges_sobel", "edges_laplacian")

#: Smallest height or width a content-aware pattern can pool over.
MIN_CONTENT_EXTENT = 8

#: Every pattern name, in the order the widget offers them.
PATTERNS = [
    "white_noise",
    "pink_noise",
    "brown_noise",
    "red_noise",
    "blue_noise",
    "violet_noise",
    "purple_noise",
    "green_noise",
    "black_noise",
    "cross_hatch",
    "highpass_white",
    "ring_noise",
    "poisson_blue_mask",
    "worley_edges",
    "tile_oriented_lines",
    "dot_screen_jitter",
    "velvet_noise",
    "perlin",
    "checker",
    "bayer",
    "solid",
    "detail_region",
    "smooth_region",
    "edges_sobel",
    "edges_laplacian",
    "external_mask",
]

#: Where fields are built. CPU keeps a seed answering the same field everywhere.
_BUILD_DEVICE = torch.device("cpu")

#: Field dtype. Half precision loses the FFT shaping the spectral patterns rely on.
_BUILD_DTYPE = torch.float32


def _generator(seed: int) -> torch.Generator:
    """Seeded random source for a field.

    Args:
        seed: Any whole number.

    Returns:
        A CPU generator set to that seed.
    """
    rng = torch.Generator(device=_BUILD_DEVICE)
    rng.manual_seed(int(seed) & 0xFFFFFFFF)
    return rng


def _unit(x: torch.Tensor) -> torch.Tensor:
    """Stretch a field so its lowest value is 0.0 and its highest 1.0.

    Args:
        x: Any tensor.

    Returns:
        The rescaled tensor. A flat field comes back as zeros.
    """
    lo = x.min()
    hi = x.max()
    return (x - lo) / (hi - lo + 1e-12)


def _radial_frequency(height: int, width: int) -> torch.Tensor:
    """Spatial frequency of every bin of a 2D FFT, in cycles per sample.

    Args:
        height: Field height.
        width: Field width.

    Returns:
        A ``(height, width)`` tensor holding ``sqrt(fx^2 + fy^2)``.
    """
    fy = torch.fft.fftfreq(height, d=1.0, device=_BUILD_DEVICE, dtype=torch.float64)
    fx = torch.fft.fftfreq(width, d=1.0, device=_BUILD_DEVICE, dtype=torch.float64)
    r = torch.sqrt(fx.reshape(1, width) ** 2 + fy.reshape(height, 1) ** 2)
    return r.to(_BUILD_DTYPE)


def spectral_noise(height: int, width: int, beta: float, seed: int) -> torch.Tensor:
    """White noise reshaped so its power follows ``f ** beta``.

    Args:
        height: Field height.
        width: Field width.
        beta: Slope of the power spectrum. 0 is white, -1 pink, -2 brown, 1 blue, 2 violet.
        seed: Seeds the noise.

    Returns:
        A ``(height, width)`` field scaled to 0.0 to 1.0.
    """
    base = torch.randn(
        (height, width), generator=_generator(seed), device=_BUILD_DEVICE, dtype=_BUILD_DTYPE
    )
    spectrum = torch.fft.fft2(base)
    radius = torch.clamp(_radial_frequency(height, width), min=1e-6)
    shaped = spectrum * torch.pow(radius, beta * 0.5)
    return _unit(torch.fft.ifft2(shaped).real)


def _band_noise(height: int, width: int, centre: float, bandwidth: float, seed: int) -> torch.Tensor:
    """White noise kept only inside one ring of the frequency plane.

    Args:
        height: Field height.
        width: Field width.
        centre: Ring radius as a fraction of the highest frequency present, 0.0 to 1.0.
        bandwidth: Ring width on the same scale.
        seed: Seeds the noise.

    Returns:
        A ``(height, width)`` field scaled to 0.0 to 1.0.
    """
    base = torch.randn(
        (height, width), generator=_generator(seed), device=_BUILD_DEVICE, dtype=_BUILD_DTYPE
    )
    spectrum = torch.fft.fft2(base)
    radius = _radial_frequency(height, width)
    top = radius.max().clamp(min=1e-6)
    c = min(max(float(centre), 0.0), 1.0) * top
    sigma = min(max(float(bandwidth), 1e-6), 1.0) * top
    ring = torch.exp(-0.5 * ((radius - c) / sigma) ** 2)
    return _unit(torch.fft.ifft2(spectrum * ring).real)


def ring_noise(height: int, width: int, centre: float, bandwidth: float, seed: int) -> torch.Tensor:
    """White noise held to a narrow high-frequency ring.

    Args:
        height: Field height.
        width: Field width.
        centre: Ring radius as a fraction of the highest frequency present.
        bandwidth: Ring width on the same scale.
        seed: Seeds the noise.

    Returns:
        A ``(height, width)`` field scaled to 0.0 to 1.0.
    """
    return _band_noise(height, width, centre, bandwidth, seed)


def highpass_white(height: int, width: int, cutoff: float, order: int, seed: int) -> torch.Tensor:
    """White noise with its low frequencies rolled off by a Butterworth response.

    Args:
        height: Field height.
        width: Field width.
        cutoff: Corner frequency as a fraction of the highest frequency present.
        order: Steepness of the roll-off, 1 and up.
        seed: Seeds the noise.

    Returns:
        A ``(height, width)`` field scaled to 0.0 to 1.0.
    """
    base = torch.randn(
        (height, width), generator=_generator(seed), device=_BUILD_DEVICE, dtype=_BUILD_DTYPE
    )
    spectrum = torch.fft.fft2(base)
    radius = torch.clamp(_radial_frequency(height, width), min=1e-6)
    corner = min(max(float(cutoff), 1e-6), 1.0) * radius.max().clamp(min=1e-6)
    n = max(1, int(order))
    response = 1.0 / (1.0 + torch.pow(corner / radius, 2 * n))
    return _unit(torch.fft.ifft2(spectrum * response.to(spectrum.dtype)).real)


def black_noise(height: int, width: int, bins: int, seed: int) -> torch.Tensor:
    """A handful of live frequency bins and silence everywhere else.

    Args:
        height: Field height.
        width: Field width.
        bins: How many bins to fill.
        seed: Seeds the bin choice and their values.

    Returns:
        A ``(height, width)`` field scaled to 0.0 to 1.0.
    """
    rng = _generator(seed)
    shape = (height, width // 2 + 1)
    total = shape[0] * shape[1]
    k = int(max(1, min(int(bins), total - 1)))

    order = torch.randperm(total, generator=rng, device=_BUILD_DEVICE)
    chosen = order[order != 0][:k]
    values = torch.complex(
        torch.randn((chosen.numel(),), generator=rng, device=_BUILD_DEVICE),
        torch.randn((chosen.numel(),), generator=rng, device=_BUILD_DEVICE),
    )

    spectrum = torch.zeros(shape, device=_BUILD_DEVICE, dtype=torch.complex64)
    spectrum.view(-1)[chosen] = values
    return _unit(torch.fft.irfft2(spectrum, s=(height, width)).to(_BUILD_DTYPE))


def velvet_noise(height: int, width: int, taps: int, seed: int) -> torch.Tensor:
    """Sparse impulses on an otherwise flat field.

    Args:
        height: Field height.
        width: Field width.
        taps: How many impulses to place, held inside 1 to ``height * width``.
        seed: Seeds where the impulses land and which way they point.

    Returns:
        A ``(height, width)`` field where an impulse reads 0.0 or 1.0 and the rest 0.5.
    """
    count = int(max(1, min(int(taps), height * width)))
    rng = _generator(seed)
    out = torch.zeros((height, width), device=_BUILD_DEVICE, dtype=_BUILD_DTYPE)
    where = torch.randperm(height * width, generator=rng, device=_BUILD_DEVICE)[:count]
    signs = (
        torch.randint(0, 2, (count,), generator=rng, device=_BUILD_DEVICE) * 2 - 1
    ).to(_BUILD_DTYPE)
    out.view(-1)[where] = signs
    return (out + 1.0) * 0.5


def perlin(
    height: int,
    width: int,
    scale: float,
    octaves: int,
    persistence: float,
    lacunarity: float,
    seed: int,
) -> torch.Tensor:
    """Smooth gradient noise summed over several octaves.

    Args:
        height: Field height.
        width: Field width.
        scale: Samples per feature. Larger values make larger blobs.
        octaves: How many passes are summed.
        persistence: How much of its predecessor's strength each octave keeps.
        lacunarity: How much finer each octave is than the one before it.
        seed: Seeds the gradients.

    Returns:
        A ``(height, width)`` field scaled to 0.0 to 1.0.
    """
    if scale <= 0:
        scale = 1.0
    rng = _generator(seed)

    def gradients(rows: int, columns: int) -> torch.Tensor:
        theta = (
            torch.rand(
                (rows + 1, columns + 1),
                generator=rng,
                device=_BUILD_DEVICE,
                dtype=_BUILD_DTYPE,
            )
            * 2
            * math.pi
        )
        return torch.stack((torch.cos(theta), torch.sin(theta)), dim=-1)

    def octave(freq_y: float, freq_x: float) -> torch.Tensor:
        rows = int(math.floor(freq_y)) + 1
        columns = int(math.floor(freq_x)) + 1
        g = gradients(rows, columns)
        y = torch.linspace(0, freq_y, steps=height, device=_BUILD_DEVICE, dtype=_BUILD_DTYPE)
        x = torch.linspace(0, freq_x, steps=width, device=_BUILD_DEVICE, dtype=_BUILD_DTYPE)
        yi = y.floor().long()
        xi = x.floor().long()
        yf = (y - yi).unsqueeze(1).expand(height, width)
        xf = (x - xi).unsqueeze(0).expand(height, width)

        def corner(ix: torch.Tensor, iy: torch.Tensor) -> torch.Tensor:
            return g[iy.clamp_max(rows), ix.clamp_max(columns)]

        x0 = xi.unsqueeze(0).expand(height, width)
        y0 = yi.unsqueeze(1).expand(height, width)
        offsets = (
            (corner(x0, y0), xf, yf),
            (corner(x0 + 1, y0), xf - 1.0, yf),
            (corner(x0, y0 + 1), xf, yf - 1.0),
            (corner(x0 + 1, y0 + 1), xf - 1.0, yf - 1.0),
        )
        n00, n10, n01, n11 = (
            (grad * torch.stack((dx, dy), dim=-1)).sum(dim=-1) for grad, dx, dy in offsets
        )

        def fade(t: torch.Tensor) -> torch.Tensor:
            return t * t * t * (t * (t * 6 - 15) + 10)

        u = fade(xf)
        v = fade(yf)
        return (n00 * (1 - u) + n10 * u) * (1 - v) + (n01 * (1 - u) + n11 * u) * v

    freq_y = max(height / scale, 1.0)
    freq_x = max(width / scale, 1.0)
    total = torch.zeros((height, width), device=_BUILD_DEVICE, dtype=_BUILD_DTYPE)
    amplitude = 1.0
    multiplier = 1.0
    peak = 0.0
    for _ in range(max(1, int(octaves))):
        total += amplitude * octave(freq_y * multiplier, freq_x * multiplier)
        peak += amplitude
        amplitude *= persistence
        multiplier *= lacunarity
    if peak > 0:
        total = total / peak
    return _unit(total)


def bayer_matrix(size: int) -> torch.Tensor:
    """One tile of an ordered dither matrix.

    Args:
        size: Tile side in samples, 2 and up.

    Returns:
        A ``(size, size)`` tensor scaled to 0.0 to 1.0.
    """
    base = torch.tensor([[0, 2], [3, 1]], device=_BUILD_DEVICE, dtype=_BUILD_DTYPE)
    if size <= 2:
        tile = base
    else:
        four = torch.cat(
            [
                torch.cat([4 * base + 0, 4 * base + 2], dim=1),
                torch.cat([4 * base + 3, 4 * base + 1], dim=1),
            ],
            dim=0,
        )
        if size <= 4:
            tile = four
        else:
            rows = -(-size // four.shape[0])
            columns = -(-size // four.shape[1])
            tile = four.repeat(rows, columns)[:size, :size]
    return _unit(tile)


def checker(height: int, width: int, cell: int) -> torch.Tensor:
    """A checkerboard.

    Args:
        height: Field height.
        width: Field width.
        cell: Square side in samples, 2 and up.

    Returns:
        A ``(height, width)`` field of 0.0 and 1.0 squares.
    """
    size = max(2, int(cell))
    rows = torch.arange(height, device=_BUILD_DEVICE).view(height, 1)
    columns = torch.arange(width, device=_BUILD_DEVICE).view(1, width)
    return (((rows // size) + (columns // size)) % 2).to(_BUILD_DTYPE)


def solid(height: int, width: int, alpha: float) -> torch.Tensor:
    """One value everywhere.

    Args:
        height: Field height.
        width: Field width.
        alpha: The value, held inside 0.0 to 1.0.

    Returns:
        A ``(height, width)`` field of that value.
    """
    value = min(max(float(alpha), 0.0), 1.0)
    return torch.full((height, width), value, device=_BUILD_DEVICE, dtype=_BUILD_DTYPE)


def cross_hatch(
    height: int,
    width: int,
    frequency: float,
    angles: tuple[float, float],
    square: bool,
    phase_jitter: float,
    supersample: int,
    seed: int,
) -> torch.Tensor:
    """Two gratings crossed over each other.

    Args:
        height: Field height.
        width: Field width.
        frequency: Cycles per sample of each grating.
        angles: The two grating angles in degrees.
        square: Whether to square off the waves rather than leave them sinusoidal.
        phase_jitter: How far each grating's phase is allowed to shift, 0.0 to 1.0.
        supersample: How many times over to build the field before averaging it down.
        seed: Seeds the phase shift.

    Returns:
        A ``(height, width)`` field scaled to 0.0 to 1.0.
    """
    factor = max(1, int(supersample))
    tall, wide = height * factor, width * factor
    rng = _generator(seed)
    rows = torch.arange(tall, device=_BUILD_DEVICE, dtype=_BUILD_DTYPE).view(tall, 1)
    columns = torch.arange(wide, device=_BUILD_DEVICE, dtype=_BUILD_DTYPE).view(1, wide)

    two_pi = 2.0 * math.pi
    listed = list(angles) if isinstance(angles, (list, tuple)) else [0.0, 90.0]
    total = torch.zeros((tall, wide), device=_BUILD_DEVICE, dtype=_BUILD_DTYPE)
    for angle in listed:
        theta = math.radians(float(angle))
        phase = 0.0
        if phase_jitter > 0.0:
            spin = torch.rand((), generator=rng, device=_BUILD_DEVICE, dtype=_BUILD_DTYPE)
            phase = float(spin) * two_pi * float(phase_jitter)
        projection = columns * math.cos(theta) + rows * math.sin(theta)
        wave = torch.sin(two_pi * float(frequency) * projection + phase)
        total = total + (torch.sign(wave) if square else wave)
    if listed:
        total = total / float(len(listed))
    total = _unit(total)

    if factor > 1:
        total = gaussian_blur_depthwise(total.view(1, 1, tall, wide), 0.6)
        total = F.interpolate(total, size=(height, width), mode="area").view(height, width)
        total = _unit(total)
    return total


def poisson_blue(height: int, width: int, radius: float, softness: float, seed: int) -> torch.Tensor:
    """Distance to the nearest of a set of points no closer than a fixed spacing.

    Args:
        height: Field height.
        width: Field width.
        radius: Closest two points may sit, in samples.
        softness: How fast the field rises with distance. Larger is smoother.
        seed: Seeds where the points land.

    Returns:
        A ``(height, width)`` field scaled to 0.0 to 1.0, dark at every point.
    """
    rng = _generator(seed)
    spacing = max(1.0, float(radius))
    cell = spacing / math.sqrt(2.0)
    columns, rows = int(math.ceil(width / cell)), int(math.ceil(height / cell))
    grid = -torch.ones((rows, columns), device=_BUILD_DEVICE, dtype=torch.int32)
    points: list[tuple[float, float]] = []

    def clear(x: float, y: float) -> bool:
        gi, gj = int(y // cell), int(x // cell)
        for ii in range(max(0, gi - 2), min(rows - 1, gi + 2) + 1):
            for jj in range(max(0, gj - 2), min(columns - 1, gj + 2) + 1):
                index = int(grid[ii, jj].item())
                if index >= 0:
                    px, py = points[index]
                    if (px - x) ** 2 + (py - y) ** 2 < spacing * spacing:
                        return False
        return True

    start_x = float(torch.rand((), generator=rng, device=_BUILD_DEVICE)) * width
    start_y = float(torch.rand((), generator=rng, device=_BUILD_DEVICE)) * height
    points.append((start_x, start_y))
    grid[int(start_y // cell), int(start_x // cell)] = 0
    active = [0]

    tries = 30
    ceiling = int(4.0 * (height * width) / (spacing * spacing + 1e-6))
    while active and len(points) < ceiling:
        pick = int(torch.randint(0, len(active), (1,), generator=rng, device=_BUILD_DEVICE))
        seat = active[pick]
        cx, cy = points[seat]
        placed = False
        for _ in range(tries):
            reach = spacing * (1.0 + float(torch.rand((), generator=rng, device=_BUILD_DEVICE)))
            angle = 2 * math.pi * float(torch.rand((), generator=rng, device=_BUILD_DEVICE))
            nx = cx + reach * math.cos(angle)
            ny = cy + reach * math.sin(angle)
            if 0 <= nx < width and 0 <= ny < height and clear(nx, ny):
                points.append((nx, ny))
                grid[int(ny // cell), int(nx // cell)] = len(points) - 1
                active.append(len(points) - 1)
                placed = True
                break
        if not placed:
            active.remove(seat)

    placed_points = torch.tensor(points, device=_BUILD_DEVICE, dtype=_BUILD_DTYPE)
    nearest, _ = _nearest_distances(height, width, placed_points, keep=1)
    field_2d = 1.0 - torch.exp(-nearest.reshape(height, width) / max(1e-6, float(softness)))
    return _unit(field_2d)


def _nearest_distances(
    height: int, width: int, points: torch.Tensor, keep: int, metric: str = "L2"
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Distance from every sample to its nearest points.

    Args:
        height: Field height.
        width: Field width.
        points: An ``(n, 2)`` tensor of x, y positions.
        keep: How many of the closest to answer, 1 or 2.
        metric: ``"L2"`` for straight-line distance, ``"L1"`` for city blocks.

    Returns:
        ``(first, second)``, each a flat tensor of ``height * width`` distances. ``second``
        is None where only one was asked for.
    """
    rows, columns = torch.meshgrid(
        torch.arange(height, device=_BUILD_DEVICE, dtype=_BUILD_DTYPE),
        torch.arange(width, device=_BUILD_DEVICE, dtype=_BUILD_DTYPE),
        indexing="ij",
    )
    samples = torch.stack([columns.reshape(-1), rows.reshape(-1)], dim=1)
    total = samples.shape[0]
    best = [
        torch.full((total,), float("inf"), device=_BUILD_DEVICE, dtype=_BUILD_DTYPE)
        for _ in range(keep)
    ]

    sample_chunk = 65536
    point_chunk = 256
    for start in range(0, total, sample_chunk):
        block = samples[start : start + sample_chunk]
        running = [
            torch.full((block.shape[0],), float("inf"), device=_BUILD_DEVICE, dtype=_BUILD_DTYPE)
            for _ in range(keep)
        ]
        for offset in range(0, points.shape[0], point_chunk):
            batch = points[offset : offset + point_chunk]
            gap = block[:, None, :] - batch[None, :, :]
            if metric.upper() == "L1":
                distance = gap.abs().sum(dim=2)
            else:
                distance = torch.sqrt((gap * gap).sum(dim=2) + 1e-12)
            merged = torch.cat([distance] + [r.view(-1, 1) for r in running], dim=1)
            values, _ = torch.topk(merged, k=keep, dim=1, largest=False)
            running = [values[:, j] for j in range(keep)]
        for j in range(keep):
            best[j][start : start + block.shape[0]] = running[j]
    return best[0], (best[1] if keep >= 2 else None)


def worley_edges(
    height: int, width: int, density: float, metric: str, sharpness: float, seed: int
) -> torch.Tensor:
    """The boundaries between cells grown around scattered points.

    Args:
        height: Field height.
        width: Field width.
        density: Cell seeds per thousand samples.
        metric: ``"L2"`` for round cells, ``"L1"`` for diamond ones.
        sharpness: How tightly the field hugs a boundary. Larger is thinner.
        seed: Seeds where the cell centres land.

    Returns:
        A ``(height, width)`` field scaled to 0.0 to 1.0, bright along every boundary.
    """
    rng = _generator(seed)
    # A boundary is the gap between the two nearest centres, so one centre has none.
    count = max(2, int(density * (height * width) / 1000.0))
    points = torch.stack(
        [
            torch.rand((count,), generator=rng, device=_BUILD_DEVICE, dtype=_BUILD_DTYPE) * width,
            torch.rand((count,), generator=rng, device=_BUILD_DEVICE, dtype=_BUILD_DTYPE) * height,
        ],
        dim=1,
    )
    first, second = _nearest_distances(height, width, points, keep=2, metric=metric)
    gap = (second - first).reshape(height, width)
    gap = torch.nan_to_num(gap, nan=0.0, posinf=0.0, neginf=0.0)
    gap = gap / (gap.max() + 1e-12)
    edges = torch.clamp(1.0 - gap, 0.0, 1.0) ** max(0.1, float(sharpness))
    return _unit(edges)


def tile_lines(
    height: int, width: int, tile: int, frequency: float, jitter: float, seed: int
) -> torch.Tensor:
    """A grating in every tile, each turned a different way.

    Args:
        height: Field height.
        width: Field width.
        tile: Tile side in samples, 2 and up.
        frequency: Cycles per sample within a tile.
        jitter: How far a tile's phase is allowed to shift, 0.0 to 1.0.
        seed: Seeds the angle and phase of each tile.

    Returns:
        A ``(height, width)`` field scaled to 0.0 to 1.0.
    """
    side = max(2, int(tile))
    rng = _generator(seed)
    rows, columns = torch.meshgrid(
        torch.arange(height, device=_BUILD_DEVICE, dtype=_BUILD_DTYPE),
        torch.arange(width, device=_BUILD_DEVICE, dtype=_BUILD_DTYPE),
        indexing="ij",
    )
    out = torch.zeros((height, width), device=_BUILD_DEVICE, dtype=_BUILD_DTYPE)
    two_pi = 2.0 * math.pi
    for y0 in range(0, height, side):
        for x0 in range(0, width, side):
            y1 = min(height, y0 + side)
            x1 = min(width, x0 + side)
            theta = float(torch.rand((), generator=rng, device=_BUILD_DEVICE)) * two_pi
            phase = (
                (float(torch.rand((), generator=rng, device=_BUILD_DEVICE)) - 0.5)
                * two_pi
                * float(jitter)
            )
            projection = (columns[y0:y1, x0:x1] - x0) * math.cos(theta) + (
                rows[y0:y1, x0:x1] - y0
            ) * math.sin(theta)
            out[y0:y1, x0:x1] = torch.sin(two_pi * float(frequency) * projection + phase)
    return _unit(out)


def dot_screen(
    height: int, width: int, cell: int, jitter: float, fill: float, seed: int
) -> torch.Tensor:
    """A halftone lattice of dots, each nudged off its cell centre.

    Args:
        height: Field height.
        width: Field width.
        cell: Cell side in samples, 2 and up.
        jitter: How far a dot may move from its centre, in samples.
        fill: Roughly what share of a cell a dot covers, 0.0 to 1.0.
        seed: Seeds the nudge and the size of each dot.

    Returns:
        A ``(height, width)`` field scaled to 0.0 to 1.0.
    """
    side = max(2, int(cell))
    rng = _generator(seed)
    rows, columns = torch.meshgrid(
        torch.arange(height, device=_BUILD_DEVICE, dtype=_BUILD_DTYPE),
        torch.arange(width, device=_BUILD_DEVICE, dtype=_BUILD_DTYPE),
        indexing="ij",
    )
    out = torch.zeros((height, width), device=_BUILD_DEVICE, dtype=_BUILD_DTYPE)
    base = 0.5 * math.sqrt(max(0.0, float(fill))) * side
    for y0 in range(0, height, side):
        for x0 in range(0, width, side):
            y1 = min(height, y0 + side)
            x1 = min(width, x0 + side)
            dx = (float(torch.rand((), generator=rng, device=_BUILD_DEVICE)) - 0.5) * 2.0 * float(jitter)
            dy = (float(torch.rand((), generator=rng, device=_BUILD_DEVICE)) - 0.5) * 2.0 * float(jitter)
            cx = x0 + side * 0.5 + dx
            cy = y0 + side * 0.5 + dy
            spread = base * (0.8 + 0.4 * float(torch.rand((), generator=rng, device=_BUILD_DEVICE)))
            gap = (columns[y0:y1, x0:x1] - cx) ** 2 + (rows[y0:y1, x0:x1] - cy) ** 2
            out[y0:y1, x0:x1] = (gap <= spread * spread).to(_BUILD_DTYPE)
    out = gaussian_blur_depthwise(out.view(1, 1, height, width), 0.6).view(height, width)
    return _unit(out)


def unsharp(x: torch.Tensor, sigma: float, amount: float, threshold: float) -> torch.Tensor:
    """Raise the contrast of whatever a blur of the field does not hold.

    Args:
        x: A ``[B, C, H, W]`` tensor.
        sigma: Blur radius the detail is measured against. 0.0 and below does nothing.
        amount: How much of the detail to add back. 0.0 does nothing, negative softens.
        threshold: Detail below this is left alone, on a 0.0 to 1.0 scale.

    Returns:
        The sharpened tensor, held inside 0.0 to 1.0.
    """
    if sigma <= 0.0 or amount == 0.0:
        return x
    blurred = gaussian_blur_depthwise(x, sigma)
    detail = x - blurred
    if threshold > 0.0:
        detail = torch.where(detail.abs() >= threshold, detail, torch.zeros_like(detail))
    return torch.clamp(x + amount * detail, 0.0, 1.0)


def sobel_magnitude(x: torch.Tensor) -> torch.Tensor:
    """How fast a latent changes at each position, measured with a Sobel pair.

    Args:
        x: A ``[B, C, H, W]`` tensor.

    Returns:
        A ``[B, 1, H, W]`` map scaled to 0.0 to 1.0 per plane.
    """
    b, c, h, w = x.shape
    magnitude = sobel_grad_mag(x.reshape(b * c, 1, h, w)).reshape(b, c, h, w)
    return _plane_unit(magnitude.mean(dim=1, keepdim=True))


def laplacian_magnitude(x: torch.Tensor) -> torch.Tensor:
    """How sharply a latent turns at each position, measured with a Laplacian.

    Args:
        x: A ``[B, C, H, W]`` tensor.

    Returns:
        A ``[B, 1, H, W]`` map scaled to 0.0 to 1.0 per plane.
    """
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=x.device, dtype=x.dtype
    ).view(1, 1, 3, 3)
    c = x.shape[1]
    response = F.conv2d(x, kernel.expand(c, 1, 3, 3), padding=1, groups=c)
    return _plane_unit(response.abs().mean(dim=1, keepdim=True))


def local_variance(x: torch.Tensor, window: int) -> torch.Tensor:
    """How much a latent varies inside a sliding window.

    Args:
        x: A ``[B, C, H, W]`` tensor.
        window: Window side in samples. Rounded up to the next odd number, 3 and up.

    Returns:
        A ``[B, 1, H, W]`` map scaled to 0.0 to 1.0 per plane.
    """
    k = max(3, int(window) | 1)
    c = x.shape[1]
    weights = torch.ones((1, 1, k, k), device=x.device, dtype=x.dtype) / float(k * k)
    weights = weights.expand(c, 1, k, k)
    pad = (k // 2, k // 2)
    mean = F.conv2d(x, weights, padding=pad, groups=c)
    mean_square = F.conv2d(x * x, weights, padding=pad, groups=c)
    variance = (mean_square - mean * mean).clamp_min(0.0)
    return _plane_unit(variance.mean(dim=1, keepdim=True))


def _plane_unit(x: torch.Tensor) -> torch.Tensor:
    """Stretch every plane of a map to 0.0 to 1.0 on its own.

    Args:
        x: A ``[B, 1, H, W]`` map.

    Returns:
        The rescaled map.
    """
    lo = x.amin(dim=(-2, -1), keepdim=True)
    hi = x.amax(dim=(-2, -1), keepdim=True)
    return (x - lo) / (hi - lo + 1e-12)


def content_field(x: torch.Tensor, pattern: str, window: int) -> torch.Tensor:
    """A mask read off the latent rather than generated.

    Args:
        x: A ``[B, C, H, W]`` tensor.
        pattern: One of :data:`CONTENT_PATTERNS`.
        window: Window side for the variance the region patterns use.

    Returns:
        A ``[B, 1, H, W]`` map scaled to 0.0 to 1.0.

    Raises:
        ValueError: The pattern is not one this reads off the latent.
    """
    if pattern == "edges_sobel":
        return sobel_magnitude(x)
    if pattern == "edges_laplacian":
        return laplacian_magnitude(x)
    if pattern in ("detail_region", "smooth_region"):
        busy = (sobel_magnitude(x) + local_variance(x, window)) * 0.5
        return busy if pattern == "detail_region" else (1.0 - busy).clamp(0.0, 1.0)
    raise ValueError(f"'{pattern}' is not read off the latent")


def field(pattern: str, height: int, width: int, seed: int, params: dict) -> torch.Tensor:
    """Build one procedural field.

    Args:
        pattern: A name from :data:`PATTERNS` that is not content-aware or external.
        height: Field height.
        width: Field width.
        seed: Seeds whatever the pattern draws at random.
        params: Resolved pattern parameters, as :data:`modules.latent.affine.DEFAULTS` keys.

    Returns:
        A ``(height, width)`` field on the CPU, scaled to 0.0 to 1.0.

    Raises:
        ValueError: The pattern is not generated here.
    """
    if pattern == "white_noise":
        return torch.rand(
            (height, width), generator=_generator(seed), device=_BUILD_DEVICE, dtype=_BUILD_DTYPE
        )
    if pattern == "pink_noise":
        return spectral_noise(height, width, -1.0, seed)
    if pattern in ("brown_noise", "red_noise"):
        return spectral_noise(height, width, -2.0, seed)
    if pattern == "blue_noise":
        return spectral_noise(height, width, 1.0, seed)
    if pattern in ("violet_noise", "purple_noise"):
        return spectral_noise(height, width, 2.0, seed)
    if pattern == "green_noise":
        return _band_noise(
            height, width, params["green_center_frac"], params["green_bandwidth_frac"], seed
        )
    if pattern == "black_noise":
        density = max(1, int(params["black_bins_per_kpx"]))
        bins = max(1, int(round(density * (height * width) / 1000.0)))
        return black_noise(height, width, bins, seed)
    if pattern == "cross_hatch":
        return cross_hatch(
            height,
            width,
            params["hatch_freq_cyc_px"],
            (params["hatch_angle1_deg"], params["hatch_angle2_deg"]),
            params["hatch_square"],
            params["hatch_phase_jitter"],
            params["hatch_supersample"],
            seed,
        )
    if pattern == "highpass_white":
        return highpass_white(
            height, width, params["highpass_cutoff_frac"], params["highpass_order"], seed
        )
    if pattern == "ring_noise":
        return ring_noise(
            height, width, params["ring_center_frac"], params["ring_bandwidth_frac"], seed
        )
    if pattern == "poisson_blue_mask":
        return poisson_blue(
            height, width, params["poisson_radius_px"], params["poisson_softness"], seed
        )
    if pattern == "worley_edges":
        return worley_edges(
            height,
            width,
            params["worley_points_per_kpx"],
            params["worley_metric"],
            params["worley_edge_sharpness"],
            seed,
        )
    if pattern == "tile_oriented_lines":
        return tile_lines(
            height,
            width,
            params["tile_line_tile_size"],
            params["tile_line_freq_cyc_px"],
            params["tile_line_jitter"],
            seed,
        )
    if pattern == "dot_screen_jitter":
        return dot_screen(
            height,
            width,
            params["dot_cell_size"],
            params["dot_jitter_px"],
            params["dot_fill_ratio"],
            seed,
        )
    if pattern == "velvet_noise":
        density = max(1, int(params["velvet_taps_per_kpx"]))
        taps = max(1, int(round(density * (height * width) / 1000.0)))
        return velvet_noise(height, width, taps, seed)
    if pattern == "perlin":
        return perlin(
            height,
            width,
            params["perlin_scale"],
            params["perlin_octaves"],
            params["perlin_persistence"],
            params["perlin_lacunarity"],
            seed,
        )
    if pattern == "checker":
        return checker(height, width, params["checker_size"])
    if pattern == "bayer":
        size = max(2, int(params["bayer_size"]))
        tile = bayer_matrix(size)
        rows = -(-height // size)
        columns = -(-width // size)
        return tile.repeat(rows, columns)[:height, :width]
    if pattern == "solid":
        return solid(height, width, params["solid_alpha"])
    raise ValueError(f"'{pattern}' is not a generated pattern")
