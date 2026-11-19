"""The film stocks a picture can be developed as.

Each :class:`Stock` is a look authored after a stock's published character: its tone curve,
colour, latitude, grain and halation. They are not measured reproductions of any product.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["BANDS_INDEX", "FAMILIES", "NAMES", "NEUTRAL", "STOCKS", "Stock", "find"]

#: Position of each colour band in a mixer row.
BANDS_INDEX = {
    name: index
    for index, name in enumerate(("red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta"))
}

#: Families in menu order.
FAMILIES = (
    "Color Negative", "Slide", "Cinema", "Instant", "Black and White", "Process", "Camera",
)


@dataclass(frozen=True)
class Stock:
    """One film look.

    Attributes:
        name: The stock's name.
        family: One of :data:`FAMILIES`.
        iso: Film speed, which sets the grain. 0 = no grain.
        curve: Output levels at inputs 0, 0.25, 0.5, 0.75 and 1, in picture codes.
        per_channel: 0 = the curve bends brightness only, 1 = each channel on its own.
        latitude: Stops above white the stock folds back under it. 0 clips.
        exposure: Stops the stock adds before its curve.
        temperature: -100 to 100, its colour balance from blue to amber.
        tint: -100 to 100, from green to magenta.
        saturation: Overall colour, 1 = unchanged.
        bands: ``(band, hue, saturation, luminance)`` mixer rows, each -100 to 100.
        shadows: ``(hue degrees, saturation 0 to 100)`` the shadows are tinted.
        midtones: The same for the midtones.
        highlights: The same for the highlights.
        grain: Grain relative to the speed's own.
        grain_colour: 0 = one grain on every channel, 1 = a separate grain per channel.
        halation: Glow around highlights, 1 = strong.
        mono: Red, green and blue sensitivity of a black and white stock, or None.
        matrix: A 3 by 3 channel mix applied before the curve, or None.
    """

    name: str
    family: str
    iso: int = 0
    curve: tuple = (0.0, 0.25, 0.5, 0.75, 1.0)
    per_channel: float = 0.35
    latitude: float = 2.5
    exposure: float = 0.0
    temperature: float = 0.0
    tint: float = 0.0
    saturation: float = 1.0
    bands: tuple = ()
    shadows: tuple = (0.0, 0.0)
    midtones: tuple = (0.0, 0.0)
    highlights: tuple = (0.0, 0.0)
    grain: float = 1.0
    grain_colour: float = 0.25
    halation: float = 0.08
    mono: tuple | None = None
    matrix: tuple | None = None

    @property
    def label(self) -> str:
        """The menu entry, family first."""
        return f"{self.family}: {self.name}"


#: The untouched picture.
NEUTRAL = Stock("None", "None", per_channel=0.0, latitude=0.0, grain=0.0, halation=0.0)


def _neg(name, iso, curve, **kw) -> Stock:
    """A colour negative stock."""
    return Stock(name, "Color Negative", iso, curve, **{"latitude": 2.5, **kw})


def _slide(name, iso, curve, **kw) -> Stock:
    """A slide stock, which clips above white."""
    return Stock(name, "Slide", iso, curve, **{"latitude": 0.0, "per_channel": 0.5,
                                              "grain_colour": 0.2, "halation": 0.04, **kw})


def _cine(name, iso, curve, **kw) -> Stock:
    """A motion picture stock or print."""
    return Stock(name, "Cinema", iso, curve, **{"latitude": 3.0, "grain_colour": 0.2, **kw})


def _instant(name, iso, curve, **kw) -> Stock:
    """An instant film."""
    return Stock(name, "Instant", iso, curve, **{"latitude": 1.0, "per_channel": 0.25,
                                                "grain": 0.4, "grain_colour": 0.15,
                                                "halation": 0.04, **kw})


def _mono(name, iso, weights, curve, **kw) -> Stock:
    """A black and white stock."""
    return Stock(name, "Black and White", iso, curve, **{"latitude": 2.0, "mono": weights,
                                                        "grain_colour": 0.0, "halation": 0.03, **kw})


def _process(name, curve, **kw) -> Stock:
    """A lab process or treatment rather than a stock."""
    return Stock(name, "Process", kw.pop("iso", 200), curve, **kw)


def _camera(name, curve, **kw) -> Stock:
    """A digital camera's film simulation, without grain."""
    return Stock(name, "Camera", kw.pop("iso", 0), curve, **{"latitude": 1.5, "halation": 0.0, **kw})


_ALL = (
    # Colour negative
    _neg("Kodak Portra 160", 160, (0.025, 0.255, 0.50, 0.745, 0.985), saturation=0.92, temperature=4,
         bands=(("red", 0, -10, 0), ("orange", 3, -8, 4), ("yellow", -4, -10, 0), ("green", 8, -22, -4),
                ("aqua", 0, -10, 0), ("blue", -4, -12, 0)),
         shadows=(200, 6), highlights=(45, 8), grain=0.8),
    _neg("Kodak Portra 400", 400, (0.03, 0.25, 0.505, 0.755, 0.985), saturation=0.96, temperature=6,
         bands=(("red", 0, -6, 0), ("orange", 3, -5, 4), ("yellow", -5, -8, 2), ("green", 10, -18, -3),
                ("blue", -5, -8, 0)),
         shadows=(190, 7), highlights=(40, 9)),
    _neg("Kodak Portra 800", 800, (0.035, 0.245, 0.51, 0.765, 0.985), saturation=1.0, temperature=8,
         tint=3, bands=(("orange", 2, -4, 3), ("green", 8, -12, -3)), shadows=(185, 9), highlights=(35, 10)),
    _neg("Kodak Ektar 100", 100, (0.015, 0.22, 0.50, 0.78, 0.99), saturation=1.16, temperature=-2,
         per_channel=0.45,
         bands=(("red", 2, 14, -4), ("orange", 0, 5, 0), ("yellow", 0, 5, 0), ("green", -4, 10, -5),
                ("aqua", 0, 10, 0), ("blue", -6, 20, -8)),
         shadows=(230, 6), highlights=(50, 4), grain=0.7),
    _neg("Kodak Gold 200", 200, (0.03, 0.245, 0.51, 0.765, 0.985), saturation=1.08, temperature=14,
         tint=-2, bands=(("red", 2, 6, 0), ("orange", 2, 10, 3), ("yellow", -3, 14, 4), ("green", 12, -10, 0),
                         ("blue", 0, -6, 0)),
         shadows=(30, 6), highlights=(55, 12)),
    _neg("Kodak ColorPlus 200", 200, (0.04, 0.25, 0.50, 0.75, 0.975), saturation=0.98, temperature=12,
         tint=2, bands=(("green", 12, -15, 0), ("blue", 0, -12, 0)), shadows=(30, 8), highlights=(50, 10),
         grain=1.1),
    _neg("Kodak UltraMax 400", 400, (0.03, 0.235, 0.51, 0.775, 0.99), saturation=1.12, temperature=8,
         bands=(("red", 0, 10, 0), ("orange", 2, 6, 2), ("yellow", -2, 10, 0), ("green", 6, 0, 0),
                ("blue", -4, 6, -4)),
         shadows=(210, 6), highlights=(45, 8), grain=1.15),
    _neg("Kodak Pro Image 100", 100, (0.025, 0.245, 0.505, 0.765, 0.985), saturation=1.05, temperature=8,
         bands=(("orange", 2, 4, 4), ("green", 8, -8, 0)), shadows=(200, 5), highlights=(45, 8), grain=0.8),
    _neg("Fujifilm Pro 400H", 400, (0.03, 0.26, 0.52, 0.765, 0.985), saturation=0.9, temperature=-6,
         tint=4, bands=(("red", -4, -6, 2), ("orange", -3, -8, 6), ("yellow", 6, -15, 4), ("green", 18, -10, 4),
                        ("aqua", -6, 6, 0), ("blue", -6, -4, 4)),
         shadows=(170, 8), highlights=(330, 4)),
    _neg("Fujicolor Superia X-TRA 400", 400, (0.025, 0.235, 0.505, 0.77, 0.99), saturation=1.1,
         temperature=-4, tint=-4, bands=(("red", 4, 6, 0), ("green", -6, 14, -2), ("blue", -6, 10, -3)),
         shadows=(150, 9), highlights=(60, 5), grain=1.15),
    _neg("Fujicolor C200", 200, (0.035, 0.25, 0.505, 0.76, 0.98), saturation=1.0, temperature=-6, tint=-6,
         bands=(("red", 0, -4, 0), ("green", -4, 6, 0)), shadows=(160, 8), highlights=(80, 5)),
    _neg("Fujicolor Reala 100", 100, (0.02, 0.24, 0.50, 0.765, 0.99), saturation=1.04,
         bands=(("green", -3, 6, 0),), shadows=(190, 4), highlights=(60, 3), grain=0.7),
    _neg("Fujicolor Natura 1600", 1600, (0.05, 0.27, 0.51, 0.75, 0.97), saturation=0.88, temperature=2,
         tint=-3, shadows=(140, 8), highlights=(50, 6)),
    _neg("Agfa Vista 200", 200, (0.025, 0.23, 0.505, 0.775, 0.99), saturation=1.15, temperature=-2,
         bands=(("red", 2, 14, -2), ("green", 4, 4, 0), ("blue", -4, 10, -4)),
         shadows=(220, 6), highlights=(30, 6)),
    _neg("Lomography Color Negative 400", 400, (0.03, 0.225, 0.51, 0.785, 0.99), saturation=1.2,
         temperature=10, tint=2, bands=(("red", 0, 12, 0), ("yellow", 0, 10, 0), ("blue", -5, 10, -5)),
         shadows=(220, 10), highlights=(40, 10), grain=1.2),
    _neg("Lomography Color Negative 800", 800, (0.04, 0.22, 0.51, 0.79, 0.985), saturation=1.15,
         temperature=12, shadows=(210, 12), highlights=(35, 12), grain=1.25),

    # Slide
    _slide("Fujichrome Velvia 50", 50, (0.0, 0.18, 0.49, 0.80, 1.0), saturation=1.28, temperature=-2,
           tint=4, bands=(("red", 2, 16, -6), ("yellow", -4, 10, 0), ("green", -6, 22, -10),
                          ("aqua", -4, 16, -4), ("blue", -6, 22, -12), ("purple", 0, 12, 0),
                          ("magenta", 0, 10, 0)),
           shadows=(250, 8), highlights=(60, 4), grain=0.55),
    _slide("Fujichrome Velvia 100", 100, (0.0, 0.175, 0.49, 0.81, 1.0), saturation=1.26, tint=2,
           bands=(("red", 2, 14, -6), ("green", -6, 20, -10), ("aqua", -4, 14, -4), ("blue", -6, 20, -12)),
           shadows=(245, 6), highlights=(60, 3), grain=0.6),
    _slide("Fujichrome Provia 100F", 100, (0.0, 0.20, 0.50, 0.785, 1.0), saturation=1.1, temperature=-2,
           bands=(("green", 0, 6, 0), ("blue", -3, 8, -3)), shadows=(220, 4), grain=0.55),
    _slide("Fujichrome Astia 100F", 100, (0.01, 0.23, 0.505, 0.765, 0.995), saturation=1.02, temperature=2,
           bands=(("red", 0, -4, 0), ("orange", 2, -6, 4), ("blue", -3, 6, 0)), highlights=(40, 4),
           grain=0.55),
    _slide("Kodak Ektachrome E100", 100, (0.0, 0.20, 0.50, 0.785, 1.0), saturation=1.12, temperature=-6,
           bands=(("red", 0, 6, 0), ("aqua", -4, 10, 0), ("blue", -4, 12, -4)), shadows=(230, 6)),
    _slide("Kodak Elite Chrome 100", 100, (0.0, 0.19, 0.495, 0.795, 1.0), saturation=1.2, temperature=2,
           bands=(("red", 0, 12, -2), ("green", 0, 10, -2), ("blue", -2, 12, -4))),
    _slide("Kodachrome 25", 25, (0.005, 0.195, 0.49, 0.79, 0.99), saturation=1.15, temperature=6, tint=-2,
           per_channel=0.55, bands=(("red", 3, 14, -8), ("orange", 0, 8, -2), ("yellow", -6, 10, -4),
                                    ("green", -8, -4, -6), ("aqua", -10, 8, -4), ("blue", -10, 6, -8)),
           shadows=(20, 5), highlights=(50, 7), grain=0.45),
    _slide("Kodachrome 64", 64, (0.005, 0.19, 0.49, 0.795, 0.99), saturation=1.18, temperature=6, tint=-2,
           per_channel=0.55, bands=(("red", 3, 16, -8), ("orange", 0, 10, -2), ("yellow", -6, 12, -4),
                                    ("green", -8, -4, -6), ("aqua", -10, 8, -4), ("blue", -10, 6, -8)),
           shadows=(20, 6), highlights=(50, 8), grain=0.6),
    _slide("Agfa CT Precisa 100", 100, (0.0, 0.195, 0.495, 0.79, 1.0), saturation=1.18, temperature=-10,
           tint=-3, bands=(("green", -8, 10, 0), ("aqua", -6, 14, 0), ("blue", -4, 18, -4)),
           shadows=(200, 10)),

    # Cinema
    _cine("Kodak Vision3 50D", 50, (0.03, 0.255, 0.50, 0.745, 0.975), saturation=0.95, temperature=2,
          shadows=(200, 4), highlights=(40, 4), grain=0.5),
    _cine("Kodak Vision3 250D", 250, (0.035, 0.255, 0.505, 0.75, 0.975), saturation=0.98, temperature=3,
          shadows=(195, 5), highlights=(42, 6), grain=0.7),
    _cine("Kodak Vision3 200T", 200, (0.035, 0.25, 0.505, 0.755, 0.975), saturation=0.96, temperature=-6,
          shadows=(190, 8), highlights=(35, 5), grain=0.7),
    _cine("Kodak Vision3 500T", 500, (0.04, 0.25, 0.51, 0.76, 0.97), saturation=1.0, temperature=-8, tint=2,
          shadows=(195, 10), highlights=(30, 8), grain=0.85, halation=0.12),
    _cine("Fujifilm Eterna 250D", 250, (0.04, 0.265, 0.505, 0.74, 0.965), saturation=0.82, temperature=-2,
          tint=-3, bands=(("red", 0, -12, 0), ("orange", 0, -8, 0), ("green", 10, -20, 0), ("blue", -4, -10, 0)),
          shadows=(175, 8), highlights=(55, 4), grain=0.7),
    _cine("Kodak 2383 Print", 250, (0.0, 0.17, 0.48, 0.80, 0.985), saturation=1.08, per_channel=0.6,
          latitude=2.0, bands=(("red", 0, 4, -4), ("orange", 0, 4, 0), ("yellow", -6, 0, 0),
                               ("green", 10, -15, -6), ("aqua", 0, 6, -4), ("blue", -8, -4, -8)),
          shadows=(185, 14), highlights=(40, 10), grain=0.5, halation=0.05),
    _cine("Fujifilm 3513 Print", 250, (0.0, 0.185, 0.485, 0.79, 0.985), saturation=1.02, temperature=-4,
          per_channel=0.55, latitude=2.0, shadows=(175, 12), highlights=(60, 6), grain=0.5, halation=0.05),
    _cine("CineStill 50D", 50, (0.025, 0.24, 0.505, 0.77, 0.985), saturation=1.08, temperature=2,
          shadows=(200, 6), highlights=(40, 6), grain=0.5, halation=0.5),
    _cine("CineStill 400D", 400, (0.03, 0.24, 0.51, 0.77, 0.985), saturation=1.1, temperature=6,
          shadows=(200, 8), highlights=(35, 10), grain=0.9, halation=0.5),
    _cine("CineStill 800T", 800, (0.04, 0.245, 0.51, 0.765, 0.98), saturation=1.06, temperature=-10, tint=3,
          bands=(("red", 0, 8, 0), ("aqua", -4, 8, 0)), shadows=(190, 12), highlights=(25, 8),
          halation=1.0),

    # Instant
    _instant("Polaroid SX-70", 150, (0.08, 0.30, 0.53, 0.74, 0.93), saturation=0.8, temperature=14, tint=-2,
             bands=(("green", 10, -15, 0), ("blue", -6, -14, 0)), shadows=(220, 10), highlights=(55, 14)),
    _instant("Polaroid 600", 640, (0.06, 0.275, 0.52, 0.755, 0.95), saturation=0.9, temperature=6, tint=2,
             shadows=(210, 10), highlights=(45, 8)),
    _instant("Polaroid i-Type", 640, (0.05, 0.265, 0.515, 0.76, 0.96), saturation=0.95, temperature=4,
             shadows=(200, 8), highlights=(40, 6), grain=0.35),
    _instant("Polaroid 669 Peel-Apart", 80, (0.04, 0.24, 0.50, 0.77, 0.96), saturation=0.95,
             temperature=-8, tint=-6, bands=(("red", 4, -8, 0), ("green", -6, 8, 0)),
             shadows=(165, 12), highlights=(70, 6), grain=0.5),
    _instant("Polaroid Time-Zero", 150, (0.10, 0.32, 0.54, 0.73, 0.91), saturation=0.72, temperature=4,
             tint=8, shadows=(240, 14), highlights=(30, 14), grain=0.5),
    _instant("Fujifilm Instax Mini", 800, (0.05, 0.26, 0.525, 0.78, 0.975), saturation=1.05,
             temperature=-4, tint=2, bands=(("blue", -4, 8, 4),), shadows=(215, 8), highlights=(60, 4),
             grain=0.35),
    _instant("Fujifilm Instax Wide", 800, (0.05, 0.265, 0.52, 0.77, 0.97), saturation=1.0, temperature=-2,
             shadows=(210, 7), highlights=(55, 4), grain=0.35),
    _instant("Fujifilm FP-100C", 100, (0.03, 0.235, 0.50, 0.775, 0.98), saturation=1.1, temperature=-10,
             tint=-2, bands=(("red", 4, 0, 0), ("aqua", -4, 12, 0), ("blue", -6, 12, 0)),
             shadows=(190, 10), highlights=(80, 4)),

    # Black and white
    _mono("Kodak Tri-X 400", 400, (0.26, 0.58, 0.16), (0.02, 0.215, 0.50, 0.79, 0.99), grain=1.35),
    _mono("Kodak T-Max 100", 100, (0.24, 0.62, 0.14), (0.01, 0.225, 0.50, 0.775, 0.995), grain=0.55),
    _mono("Kodak T-Max 400", 400, (0.25, 0.60, 0.15), (0.015, 0.22, 0.50, 0.78, 0.995), grain=0.85),
    _mono("Kodak T-Max P3200", 3200, (0.28, 0.56, 0.16), (0.03, 0.205, 0.50, 0.80, 0.985), grain=1.4),
    _mono("Kodak Double-X 5222", 250, (0.27, 0.57, 0.16), (0.02, 0.20, 0.49, 0.80, 0.985), grain=1.1,
          halation=0.05),
    _mono("Ilford HP5 Plus", 400, (0.27, 0.57, 0.16), (0.025, 0.235, 0.50, 0.765, 0.98), grain=1.2),
    _mono("Ilford FP4 Plus", 125, (0.26, 0.58, 0.16), (0.015, 0.225, 0.50, 0.775, 0.99), grain=0.7),
    _mono("Ilford Delta 100", 100, (0.24, 0.60, 0.16), (0.01, 0.225, 0.50, 0.78, 0.995), grain=0.5),
    _mono("Ilford Delta 3200", 3200, (0.28, 0.55, 0.17), (0.035, 0.215, 0.50, 0.785, 0.98), grain=1.3),
    _mono("Ilford Pan F Plus 50", 50, (0.25, 0.60, 0.15), (0.005, 0.205, 0.50, 0.80, 1.0), grain=0.4),
    _mono("Ilford SFX 200", 200, (0.55, 0.38, 0.07), (0.02, 0.22, 0.50, 0.785, 0.99), grain=0.9),
    _mono("Ilford XP2 Super", 400, (0.25, 0.60, 0.15), (0.03, 0.245, 0.505, 0.76, 0.975), grain=0.45),
    _mono("Ilford Ortho Plus 80", 80, (0.0, 0.45, 0.55), (0.01, 0.21, 0.50, 0.795, 1.0), grain=0.55),
    _mono("Fujifilm Neopan Acros 100", 100, (0.30, 0.56, 0.14), (0.01, 0.215, 0.50, 0.785, 0.995),
          grain=0.45),
    _mono("Fomapan 100", 100, (0.28, 0.56, 0.16), (0.03, 0.24, 0.50, 0.765, 0.975), grain=0.8),
    _mono("Kodak High Speed Infrared", 400, (0.286, 1.571, -0.857), (0.02, 0.21, 0.50, 0.80, 1.0),
          grain=1.3, halation=1.0),

    # Process
    _process("Cross Processed", (0.0, 0.16, 0.50, 0.84, 1.0), per_channel=0.8, latitude=0.5,
             saturation=1.25, temperature=-2, tint=-10,
             bands=(("red", 0, 10, 0), ("yellow", 0, 20, 4), ("blue", 0, 10, -6)),
             shadows=(200, 25), highlights=(75, 25)),
    _process("Bleach Bypass", (0.0, 0.17, 0.48, 0.81, 1.0), per_channel=0.2, saturation=0.5,
             shadows=(200, 4), highlights=(40, 3), grain=0.8),
    _process("Redscale", (0.03, 0.22, 0.50, 0.78, 0.97), exposure=0.3, temperature=60, tint=10,
             saturation=1.2, bands=(("aqua", 0, -80, 0), ("blue", 0, -80, 0), ("red", 0, 20, 0)),
             shadows=(10, 30), highlights=(45, 30), iso=100),
    _process("Expired Film", (0.09, 0.30, 0.52, 0.735, 0.93), saturation=0.8, temperature=6, tint=10,
             shadows=(300, 10), highlights=(70, 10), grain=1.2, iso=400),
    _process("Technicolor Three-Strip", (0.01, 0.20, 0.495, 0.79, 0.995), per_channel=0.6, saturation=1.25,
             bands=(("red", 0, 20, -6), ("yellow", 0, 10, 0), ("green", -4, 15, -4), ("blue", -4, 20, -6)),
             shadows=(200, 6), highlights=(40, 6), grain=0.6, iso=50),
    _process("Two-Strip", (0.01, 0.21, 0.50, 0.78, 0.99), per_channel=0.5, saturation=1.1,
             matrix=((1.0, 0.0, 0.0), (0.0, 0.5, 0.5), (0.0, 0.5, 0.5)), highlights=(35, 6), grain=0.6, iso=50),

    # Camera
    _camera("Fujifilm Classic Chrome", (0.01, 0.21, 0.50, 0.775, 0.99), saturation=0.85, per_channel=0.4,
            bands=(("red", 0, -15, 0), ("orange", -2, -10, 0), ("yellow", -6, -20, 0), ("green", 10, -25, -4),
                   ("aqua", 0, -5, -4), ("blue", -8, -10, -6)),
            shadows=(200, 6), highlights=(50, 5)),
    _camera("Fujifilm Classic Negative", (0.015, 0.205, 0.495, 0.79, 0.98), saturation=0.95,
            temperature=-2, tint=-2, per_channel=0.45,
            bands=(("red", -4, 8, -4), ("orange", -2, -6, 0), ("yellow", -8, -15, 0), ("green", 15, -20, -6),
                   ("aqua", 0, 10, -4), ("blue", -8, 0, -8)),
            shadows=(180, 12), highlights=(35, 8)),
    _camera("Fujifilm Nostalgic Neg", (0.02, 0.235, 0.505, 0.765, 0.98), temperature=6,
            bands=(("orange", 0, 4, 2), ("yellow", -4, 8, 0)), shadows=(200, 6), highlights=(40, 14)),
    _camera("Fujifilm Eterna", (0.04, 0.265, 0.505, 0.745, 0.965), saturation=0.75,
            shadows=(180, 8), highlights=(50, 4)),
    _camera("Fujifilm Acros", (0.01, 0.215, 0.50, 0.785, 0.995), mono=(0.30, 0.58, 0.12), iso=100,
            grain=0.4, grain_colour=0.0),
)

#: Every stock by its menu entry, with the untouched picture first.
STOCKS = {"None": NEUTRAL, **{stock.label: stock for stock in _ALL}}

#: Menu entries in order.
NAMES = tuple(STOCKS)


def find(label: str) -> Stock:
    """The stock a menu entry names.

    Args:
        label: A key of :data:`STOCKS`.

    Returns:
        The stock.

    Raises:
        ValueError: No stock has that entry.
    """
    stock = STOCKS.get(label)
    if stock is None:
        raise ValueError(
            f"unknown film stock '{label}'. Pick one from the stock menu, such as "
            f"'Color Negative: Kodak Portra 400'."
        )
    return stock
