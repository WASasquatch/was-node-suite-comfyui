"""The scopes dashboard drawn frame by frame from a measured clip.

Pictures are ``(H, W, 3)`` uint8: the source, waveform, vectorscope, histogram and RGB parade,
each scope computed at its panel's own pixel size, over a timeline with a playhead.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from . import clip_scopes, scopes

#: Page, panel, edge, title and muted colours.
PAGE = (17, 18, 17)
PANEL = (25, 27, 25)
EDGE = (74, 96, 72)
TITLE = (222, 228, 220)
MUTED = (132, 142, 130)
ACCENT = (163, 210, 155)
GRID = (52, 56, 52)

#: Trace colours of the red, green, blue and luma planes.
TRACES = ((255, 92, 92), (110, 226, 120), (96, 150, 255), (226, 232, 224))

#: Colours of the timeline's curves, by name.
CURVE_COLOURS = {
    "luma": (232, 236, 230), "highlights": (255, 110, 110), "shadows": (150, 120, 255), "saturation": ACCENT,
}

#: Outer margin and gap between panels, as shares of the frame height.
MARGIN, GAP = 0.018, 0.014

#: Bins of the histogram.
HIST_BINS = 256

#: Longest side a frame is read at for the scopes.
SCOPE_SIDE = 1280


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _spectrum_colours() -> np.ndarray:
    """256 colours from black through deep green and the accent to white."""
    stops = [(0.0, (10, 11, 10)), (0.35, (28, 70, 40)), (0.7, ACCENT), (1.0, (250, 252, 248))]
    out = np.zeros((256, 3), np.float32)
    for index in range(256):
        t = index / 255.0
        for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
            if t <= t1:
                share = (t - t0) / max(t1 - t0, 1e-6)
                out[index] = np.array(c0) + (np.array(c1) - np.array(c0)) * share
                break
    return out.astype(np.uint8)


def _level(counts: torch.Tensor) -> torch.Tensor:
    """Counts as 0 to 1 by logarithm, the busiest cell at 1."""
    peak = float(counts.max())
    if peak <= 0:
        return torch.zeros_like(counts, dtype=torch.float32)
    return (torch.log1p(counts.float()) / np.log1p(peak)).clamp(0.0, 1.0)


def waveform(plane: torch.Tensor, columns: int, rows: int) -> torch.Tensor:
    """A plane's waveform: ``(rows, columns)`` levels 0 to 1, white at the top."""
    height, width = int(plane.shape[0]), int(plane.shape[1])
    column = (torch.arange(width, device=plane.device) * columns // width).expand(height, width)
    row = (rows - 1) - (plane * (rows - 1)).round().long()
    counts = torch.bincount((row * columns + column).reshape(-1), minlength=rows * columns)
    return _level(counts.view(rows, columns))


class Dashboard:
    """Draws one dashboard frame per source frame of a measured clip.

    Args:
        measured: The clip from :func:`clip_scopes.measure`.
        width: Frame width in pixels, even.
        height: Frame height in pixels, even.
        device: Where the scopes are computed; ComfyUI's compute device when None.
    """

    def __init__(self, measured: clip_scopes.Measured, width: int = 1920, height: int = 1080, device=None):
        if device is None:
            import comfy.model_management as mm

            device = mm.get_torch_device()
        self.device = device
        self.measured = measured
        self.width, self.height = int(width), int(height)
        m, g = round(self.height * MARGIN), round(self.height * GAP)
        top_h = round(self.height * 0.52)
        mid_h = round(self.height * 0.22)
        y1 = m + top_h + g
        y2 = y1 + mid_h + g
        source_w = round((self.width - 2 * m - 2 * g) * 0.5)
        side = top_h
        wave_w = self.width - 2 * m - 2 * g - source_w - side
        hist_w = round((self.width - 2 * m - g) * 0.32)
        self.boxes = {
            "source": (m, m, source_w, top_h),
            "waveform": (m + source_w + g, m, wave_w, top_h),
            "vectorscope": (self.width - m - side, m, side, top_h),
            "histogram": (m, y1, hist_w, mid_h),
            "parade": (m + hist_w + g, y1, self.width - 2 * m - hist_w - g, mid_h),
            "timeline": (m, y2, self.width - 2 * m, self.height - m - y2),
        }
        self.title_h = max(14, round(self.height * 0.026))
        self.base = self._base()
        self.panel = torch.tensor(PANEL, dtype=torch.float32, device=device)

    def _inner(self, name: str) -> tuple[int, int, int, int]:
        x, y, w, h = self.boxes[name]
        pad = max(4, round(self.height * 0.008))
        top = y + self.title_h + pad
        return x + pad, top, w - 2 * pad, y + h - pad - top

    def _base(self) -> np.ndarray:
        canvas = Image.new("RGB", (self.width, self.height), PAGE)
        draw = ImageDraw.Draw(canvas)
        font = _font(max(10, round(self.title_h * 0.78)))
        titles = {"source": "Source", "waveform": "Luma waveform", "vectorscope": "Vectorscope",
                  "histogram": "Histogram", "parade": "RGB parade", "timeline": "Timeline"}
        for name, (x, y, w, h) in self.boxes.items():
            draw.rounded_rectangle((x, y, x + w - 1, y + h - 1), radius=max(4, self.height // 120), fill=PANEL, outline=EDGE)
            draw.text((x + 8, y + 3), titles[name], font=font, fill=TITLE)
        base = np.asarray(canvas).copy()
        vx, vy, vw, vh = self._inner("vectorscope")
        side = min(vw, vh)
        grid, labels = scopes._graticule(side)
        self.vector_place = (vx + (vw - side) // 2, vy + (vh - side) // 2, side)
        self.vector_grid = (grid, labels)
        tx, ty, tw, th = self._inner("timeline")
        base[ty:ty + th, tx:tx + tw] = self._timeline_picture()
        legend = Image.fromarray(base)
        pen = ImageDraw.Draw(legend)
        small = _font(max(9, round(self.title_h * 0.62)))
        x = self.boxes["timeline"][0] + round(self.title_h * 5.5)
        y = self.boxes["timeline"][1] + 4
        words = [("luma", CURVE_COLOURS["luma"]), ("highlights clipped", CURVE_COLOURS["highlights"]),
                 ("shadows crushed", CURVE_COLOURS["shadows"]), ("saturation", CURVE_COLOURS["saturation"])]
        if self.measured.spectrum is not None:
            words.append(("sound", MUTED))
        for word, colour in words:
            pen.text((x, y), word, font=small, fill=colour)
            x += round(pen.textlength(word, font=small)) + round(self.title_h * 1.2)
        return np.asarray(legend).copy()

    def _timeline_picture(self) -> np.ndarray:
        tx, ty, tw, th = self._inner("timeline")
        sound = self.measured.spectrum is not None
        curves_h = th if not sound else round(th * 0.55)
        canvas = Image.new("RGB", (tw, th), PANEL)
        draw = ImageDraw.Draw(canvas)
        frames = max(1, self.measured.frames - 1)
        for name, colour in CURVE_COLOURS.items():
            values = self.measured.curves.get(name) or []
            if not values:
                continue
            scale = 1.0 if name in ("luma", "saturation") else max(0.02, max(values))
            points = [(round(i * (tw - 1) / frames), round((curves_h - 2) * (1.0 - min(1.0, v / scale))) + 1)
                      for i, v in enumerate(values)]
            if len(points) == 1:
                points.append((tw - 1, points[0][1]))
            draw.line(points, fill=colour, width=max(1, round(self.height / 720)))
        picture = np.asarray(canvas).copy()
        if sound:
            strip = _spectrum_colours()[self.measured.spectrum]
            image = Image.fromarray(strip).resize((tw, th - curves_h - 2), Image.BILINEAR)
            picture[curves_h + 2:, :] = np.asarray(image)
        return picture

    def _tinted(self, level: torch.Tensor, colour) -> torch.Tensor:
        trace = torch.tensor(colour, dtype=torch.float32, device=level.device)
        return self.panel * (1.0 - level[..., None]) + trace * level[..., None]

    def _grid(self, picture: torch.Tensor) -> torch.Tensor:
        rows = int(picture.shape[0])
        lines = torch.tensor(GRID, dtype=torch.float32, device=picture.device)
        for level in (0.0, 0.25, 0.5, 0.75, 1.0):
            row = round((1.0 - level) * (rows - 1))
            picture[row] = torch.maximum(picture[row], lines)
        return picture

    def _histogram(self, frame: torch.Tensor, luma: torch.Tensor, iw: int, ih: int) -> torch.Tensor:
        planes = (frame[..., 0], frame[..., 1], frame[..., 2], luma)
        heights = []
        for plane in planes:
            counts = torch.bincount((plane.reshape(-1) * (HIST_BINS - 1)).round().long(), minlength=HIST_BINS).float()
            counts = torch.sqrt(counts / counts.max().clamp_min(1.0))
            heights.append(F.interpolate(counts[None, None], size=iw, mode="linear", align_corners=True)[0, 0])
        rows = torch.arange(ih, device=frame.device, dtype=torch.float32)[:, None]
        picture = self.panel.expand(ih, iw, 3).clone()
        for channel in range(3):
            filled = (rows >= (ih - 1) - heights[channel][None, :] * (ih - 2)).float()
            picture = picture + filled[..., None] * torch.tensor(TRACES[channel], device=frame.device, dtype=torch.float32) * 0.42
        top = ((ih - 1) - heights[3] * (ih - 2)).round().long().clamp(0, ih - 1)
        picture[top, torch.arange(iw, device=frame.device)] = torch.tensor(TRACES[3], device=frame.device, dtype=torch.float32)
        return picture

    def frame(self, number: int, source: torch.Tensor) -> np.ndarray:
        """The dashboard for one frame.

        Args:
            number: The frame's number in the clip.
            source: The frame, ``(h, w, 3)`` uint8 or float codes.

        Returns:
            ``(height, width, 3)`` uint8.
        """
        out = torch.from_numpy(self.base).to(self.device).float()
        full = clip_scopes.codes(source[None], self.device, None)[0]
        picture = clip_scopes.codes(source[None], self.device, SCOPE_SIDE)[0]
        luma = clip_scopes.luma_of(picture)

        ix, iy, iw, ih = self._inner("source")
        height, width = int(full.shape[0]), int(full.shape[1])
        scale = min(iw / width, ih / height)
        w, h = max(1, round(width * scale)), max(1, round(height * scale))
        shown = F.interpolate(full.movedim(-1, 0)[None], size=(h, w), mode="area" if scale < 1 else "bilinear")[0].movedim(0, -1)
        out[iy:iy + ih, ix:ix + iw] = 0.0
        x0, y0 = ix + (iw - w) // 2, iy + (ih - h) // 2
        out[y0:y0 + h, x0:x0 + w] = shown * 255.0

        ix, iy, iw, ih = self._inner("waveform")
        out[iy:iy + ih, ix:ix + iw] = self._grid(self._tinted(waveform(luma, iw, ih), ACCENT))
        ix, iy, iw, ih = self._inner("parade")
        third = iw // 3
        for channel in range(3):
            left = ix + channel * third
            span = third if channel < 2 else iw - 2 * third
            out[iy:iy + ih, left:left + span] = self._grid(
                self._tinted(waveform(picture[..., channel], span, ih), TRACES[channel]))
        ix, iy, iw, ih = self._inner("histogram")
        out[iy:iy + ih, ix:ix + iw] = self._histogram(picture, luma, iw, ih).clamp(0, 255)

        vx, vy, side = self.vector_place
        trace = _level(scopes.vectorscope_density(picture, side)).cpu().numpy()[..., None]
        grid, labels = self.vector_grid
        drawn = grid * (1.0 - trace) + np.array(scopes.TRACE_COLOUR, np.float32) * trace
        alpha = labels[..., 3:4]
        drawn = drawn * (1.0 - alpha) + labels[..., :3] * alpha
        out[vy:vy + side, vx:vx + side] = torch.from_numpy(np.clip(drawn, 0.0, 1.0) * 255.0).to(self.device)

        tx, ty, tw, th = self._inner("timeline")
        column = tx + round(number * (tw - 1) / max(1, self.measured.frames - 1))
        out[ty:ty + th, max(tx, column - 1):column + 1] = torch.tensor(ACCENT, dtype=torch.float32, device=self.device)
        result = out.clamp(0, 255).round().to(torch.uint8).cpu().numpy()

        sx, sy, sw, _sh = self.boxes["source"]
        stamp = Image.fromarray(result[sy:sy + self.title_h + 4, sx:sx + sw])
        seconds = number / max(self.measured.rate, 1e-6)
        ImageDraw.Draw(stamp).text(
            (sw - 8, 3), f"frame {number + 1} / {self.measured.frames}   {seconds:6.2f} s",
            font=_font(max(10, round(self.title_h * 0.72))), fill=MUTED, anchor="ra",
        )
        result[sy:sy + stamp.height, sx:sx + stamp.width] = np.asarray(stamp)
        return result
