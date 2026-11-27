"""Blockout scenes read from JSON: ready-made props, props built from shapes, posed people, camera rigs and shots.

Metres, +Y up, +Z toward the default camera. A prop's own front is its local +Z.
"""

from __future__ import annotations

import bisect
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from .spec import create_spec

#: Longest JSON text read, in bytes.
MAX_BYTES = 64 * 1024
#: Deepest nesting of objects and lists read.
MAX_DEPTH = 12
#: Most props after counts are expanded.
MAX_PROPS = 150
#: Most shapes in the whole scene.
MAX_PARTS = 2000
#: Most shapes in one built prop or kind.
MAX_KIND_PARTS = 64
MAX_KINDS = 24
MAX_PEOPLE = 12
MAX_SHOTS = 24
#: Longest film, in seconds.
MAX_SECONDS = 60.0
MAX_COUNT = 12
#: Most steps in one prop's or person's motion.
MAX_STEPS = 8
#: Problems listed in one refusal.
MAX_PROBLEMS = 20
#: Longest reference chain, as in a cup on a plate on a table.
MAX_CHAIN = 12

TEXT_LIMITS = {"look": 1500, "setting": 1500, "sound": 400, "description": 300, "action": 600, "name": 40,
               "music": 400, "voice": 160, "line": 300}

#: Most spoken lines in one shot.
MAX_LINES = 6

#: Timed sounds one shot holds at most.
MAX_SOUNDS = 8

#: The look a scene takes when it names none.
LIVE_ACTION = ("The target video is a live-action film shot on a cinema camera, with real actors, real places, "
               "natural light and true-to-life detail.")

ID = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
COLOUR = re.compile(r"^#?([0-9a-fA-F]{6})$")
CONTROL = re.compile(r"[\x00-\x1f\x7f]+")

#: Plain shapes a prop or a part may be.
SHAPES = ("box", "cylinder", "sphere", "cone", "capsule", "ramp", "stairs", "arch", "ground")

POSES = ("stand", "sit", "walk", "run", "point", "wave")

RIGS = ("wide", "establishing", "two shot", "over the shoulder", "medium", "close up", "insert", "top down",
        "pov", "fly by", "fly through", "fly over", "reveal", "chase", "orbit", "custom")

#: Shot names read as one of :data:`RIGS`, and whether the camera follows the subject.
RIG_ALIASES = {"follow": ("medium", True), "tracking": ("medium", True), "tracking shot": ("medium", True),
               "follow shot": ("medium", True), "wide shot": ("wide", False), "long shot": ("wide", False),
               "full shot": ("wide", False), "establishing shot": ("establishing", False),
               "closeup": ("close up", False), "close up shot": ("close up", False),
               "extreme close up": ("close up", False), "medium shot": ("medium", False),
               "mid shot": ("medium", False), "medium close up": ("medium", False), "ots": ("over the shoulder", False),
               "over shoulder": ("over the shoulder", False), "insert shot": ("insert", False),
               "aerial": ("top down", False), "overhead": ("top down", False), "birds eye": ("top down", False),
               "bird's eye": ("top down", False), "point of view": ("pov", False), "pov shot": ("pov", False),
               "flyby": ("fly by", False), "fly past": ("fly by", False), "flythrough": ("fly through", False),
               "flyover": ("fly over", False), "drone": ("fly over", False), "crane reveal": ("reveal", False),
               "chase cam": ("chase", False), "chase shot": ("chase", False), "orbit shot": ("orbit", False),
               "360": ("orbit", False)}

#: Camera moves a shot may carry: the Three Camera Move each one drives and its default amount.
MOVES = {
    "static": ("static", 0.0),
    "push in": ("push in", None),
    "pull out": ("push in", None),
    "orbit left": ("orbit", -20.0),
    "orbit right": ("orbit", 20.0),
    "truck left": ("truck", -0.6),
    "truck right": ("truck", 0.6),
    "crane up": ("crane", 0.6),
    "crane down": ("crane", -0.6),
    "pedestal up": ("pedestal", 0.4),
    "pedestal down": ("pedestal", -0.4),
    "pan left": ("pan", -15.0),
    "pan right": ("pan", 15.0),
    "tilt up": ("tilt", 10.0),
    "tilt down": ("tilt", -10.0),
    "zoom in": ("zoom", None),
    "zoom out": ("zoom", None),
    "dolly zoom": ("dolly zoom", None),
    "dutch": ("dutch", 10.0),
}

MOVE_WORDS = {
    "static": "the camera holds still",
    "push in": "the camera pushes in slowly",
    "pull out": "the camera pulls back slowly",
    "orbit left": "the camera arcs slowly to the left",
    "orbit right": "the camera arcs slowly to the right",
    "truck left": "the camera tracks sideways to the left",
    "truck right": "the camera tracks sideways to the right",
    "crane up": "the camera cranes up",
    "crane down": "the camera cranes down",
    "pedestal up": "the camera rises",
    "pedestal down": "the camera lowers",
    "pan left": "the camera pans left",
    "pan right": "the camera pans right",
    "tilt up": "the camera tilts up",
    "tilt down": "the camera tilts down",
    "zoom in": "the lens zooms in",
    "zoom out": "the lens zooms out",
    "dolly zoom": "the camera dolly zooms, the background stretching",
    "dutch": "the camera rolls into a tilted angle",
}

GREY = "#a0a0a0"
DARK = "#8a8a8a"
LIGHT = "#b4b4b4"
GREEN = "#7d8a7d"

#: Anchor names, with the compass words also accepted.
#: Types that belong at the back of a set, standing in its walls.
FIXTURES = ("wall", "wall_with_window", "wall_with_door", "window", "door")

#: Camera move words read as one of :data:`MOVES`; ``None`` holds the camera and follows the subject.
MOVE_ALIASES = {"dolly in": "push in", "dolly out": "pull out", "push": "push in", "pull back": "pull out",
                "track left": "truck left", "track right": "truck right", "tracking": None, "track": None,
                "follow": None, "handheld": "static", "still": "static", "none": "static", "crane": "crane up",
                "pan": "pan right", "tilt": "tilt up", "orbit": "orbit right", "arc left": "orbit left",
                "arc right": "orbit right", "zoom": "zoom in", "rise": "pedestal up", "boom up": "crane up",
                "boom down": "crane down"}

ANCHOR_ALIASES = {"north": "back", "south": "front", "west": "left", "east": "right", "centre": "center"}


def _p(shape, size, at, tone=GREY, turn=(0.0, 0.0, 0.0)):
    return {"shape": shape, "size": size, "at": at, "tone": tone, "rotate": turn}


def _box_legs(w, d, h, leg=0.06, inset=0.05):
    x, z = w / 2 - inset, d / 2 - inset
    return [_p("box", (leg, h, leg), (sx * x, h / 2, sz * z), DARK) for sx in (-1, 1) for sz in (-1, 1)]


def _sides(w, d, gap=0.35, y=0.0):
    """Anchors on the four sides of a footprint, each facing the middle."""
    return {
        "front": (0.0, y, d / 2 + gap, 180.0),
        "back": (0.0, y, -(d / 2 + gap), 0.0),
        "left": (-(w / 2 + gap), y, 0.0, 90.0),
        "right": (w / 2 + gap, y, 0.0, -90.0),
    }


def _frame(w, h, d, hole_w, hole_h, sill, tone=GREY):
    """Four boxes around a rectangular hole."""
    side = (w - hole_w) / 2
    top = h - sill - hole_h
    parts = [
        _p("box", (side, h, d), (-(hole_w + side) / 2, h / 2, 0.0), tone),
        _p("box", (side, h, d), ((hole_w + side) / 2, h / 2, 0.0), tone),
        _p("box", (hole_w, top, d), (0.0, h - top / 2, 0.0), tone),
    ]
    if sill > 0:
        parts.append(_p("box", (hole_w, sill, d), (0.0, sill / 2, 0.0), tone))
    return parts


def _room(w: float, h: float, d: float) -> dict:
    """A room of four walls, a window in the back one and a door in the left one, its anchors facing in."""
    t = 0.15
    back = [{**q, "at": (q["at"][0], q["at"][1], -d / 2)}
            for q in _frame(w, h, t, min(1.6, w * 0.5), min(1.3, h * 0.45), 0.9)]
    left = [{**q, "at": (-w / 2, q["at"][1], q["at"][0]), "rotate": (0, 90, 0)}
            for q in _frame(d, h, t, min(1.0, d * 0.4), min(2.2, h - 0.2), 0.0)]
    right = [_p("box", (t, h, d), (w / 2, h / 2, 0))]
    front = [_p("box", (w, h, t), (0, h / 2, d / 2))]
    inside = 0.7
    anchors = {"center": (0, 0, 0, 0.0), "back": (0, 0, -d / 2 + inside, 0.0), "front": (0, 0, d / 2 - inside, 180.0),
               "left": (-w / 2 + inside, 0, 0, 90.0), "right": (w / 2 - inside, 0, 0, -90.0),
               "window": (0, 0, -d / 2 + inside, 180.0), "door": (-w / 2 + inside, 0, 0, -90.0), "top": (0, h, 0, 0.0)}
    return {"size": (w, h, d), "parts": back + left + right + front, "anchors": anchors, "top": h, "room": True}


def _prefabs() -> dict[str, dict]:
    """The ready-made props: size, parts at their centres, anchors and top height."""
    table = {"size": (1.2, 0.75, 0.8), "parts": [_p("box", (1.2, 0.05, 0.8), (0, 0.725, 0), LIGHT)]
             + _box_legs(1.2, 0.8, 0.7), "top": 0.75}
    table["anchors"] = {**_sides(1.2, 0.8), "center": (0, 0.75, 0, 0.0), "top": (0, 0.75, 0, 0.0)}
    desk = {"size": (1.4, 0.75, 0.7), "parts": [_p("box", (1.4, 0.05, 0.7), (0, 0.725, 0), LIGHT),
                                                _p("box", (0.04, 0.7, 0.66), (-0.66, 0.35, 0), DARK),
                                                _p("box", (0.04, 0.7, 0.66), (0.66, 0.35, 0), DARK)], "top": 0.75}
    desk["anchors"] = {**_sides(1.4, 0.7), "center": (0, 0.75, 0, 0.0), "top": (0, 0.75, 0, 0.0)}
    chair = {"size": (0.46, 0.9, 0.46),
             "parts": [_p("box", (0.46, 0.05, 0.46), (0, 0.445, 0), LIGHT),
                       _p("box", (0.46, 0.45, 0.05), (0, 0.695, -0.205))] + _box_legs(0.46, 0.46, 0.42, 0.04, 0.03),
             "top": 0.47}
    chair["anchors"] = {"seat": (0, 0.47, -0.04, 0.0), "front": (0, 0, 0.55, 180.0)}
    stool = {"size": (0.36, 0.66, 0.36), "parts": [_p("cylinder", (0.36, 0.05, 0.36), (0, 0.635, 0), LIGHT),
                                                   _p("cylinder", (0.05, 0.61, 0.05), (0, 0.305, 0), DARK)],
             "top": 0.66, "anchors": {"seat": (0, 0.66, 0, 0.0)}}
    sofa = {"size": (2.0, 0.85, 0.9),
            "parts": [_p("box", (1.6, 0.42, 0.8), (0, 0.21, 0.05)), _p("box", (2.0, 0.85, 0.2), (0, 0.425, -0.35)),
                      _p("box", (0.2, 0.62, 0.9), (-0.9, 0.31, 0)), _p("box", (0.2, 0.62, 0.9), (0.9, 0.31, 0))],
            "top": 0.45}
    sofa["anchors"] = {"seat_left": (-0.5, 0.44, 0.05, 0.0), "seat_middle": (0, 0.44, 0.05, 0.0),
                       "seat_right": (0.5, 0.44, 0.05, 0.0), "front": (0, 0, 1.0, 180.0)}
    bench = {"size": (1.5, 0.45, 0.4), "parts": [_p("box", (1.5, 0.06, 0.4), (0, 0.42, 0), LIGHT),
                                                 _p("box", (0.06, 0.39, 0.36), (-0.65, 0.195, 0), DARK),
                                                 _p("box", (0.06, 0.39, 0.36), (0.65, 0.195, 0), DARK)], "top": 0.45}
    bench["anchors"] = {"seat_left": (-0.4, 0.45, 0, 0.0), "seat_right": (0.4, 0.45, 0, 0.0),
                        "front": (0, 0, 0.6, 180.0)}
    bed = {"size": (1.6, 0.6, 2.1),
           "parts": [_p("box", (1.6, 0.42, 2.1), (0, 0.21, 0), DARK), _p("box", (1.55, 0.14, 2.0), (0, 0.49, 0.03), LIGHT),
                     _p("box", (1.6, 1.0, 0.08), (0, 0.5, -1.05)), _p("box", (0.6, 0.12, 0.35), (-0.38, 0.62, -0.78), LIGHT),
                     _p("box", (0.6, 0.12, 0.35), (0.38, 0.62, -0.78), LIGHT)], "top": 0.56}
    bed["anchors"] = {**_sides(1.6, 2.1), "foot": (0, 0, 1.4, 180.0), "top": (0, 0.56, 0.2, 0.0),
                      "pillow": (0, 0.6, -0.78, 0.0)}
    door = {"size": (1.0, 2.2, 0.12),
            "parts": [_p("box", (0.08, 2.2, 0.12), (-0.46, 1.1, 0)), _p("box", (0.08, 2.2, 0.12), (0.46, 1.1, 0)),
                      _p("box", (1.0, 0.08, 0.12), (0, 2.16, 0)), _p("box", (0.84, 2.08, 0.05), (0, 1.04, 0), LIGHT)],
            "anchors": {"front": (0, 0, 0.7, 180.0), "back": (0, 0, -0.7, 0.0)}, "top": 2.2}
    window = {"size": (2.0, 2.7, 0.15), "parts": _frame(2.0, 2.7, 0.15, 1.2, 1.2, 0.9)
              + [_p("box", (1.2, 0.04, 0.05), (0, 1.5, 0), LIGHT), _p("box", (0.04, 1.2, 0.05), (0, 1.5, 0), LIGHT)],
              "anchors": {"front": (0, 0, 0.8, 180.0), "back": (0, 0, -0.8, 0.0)}, "top": 2.7}
    wall = {"size": (4.0, 2.7, 0.15), "parts": [_p("box", (4.0, 2.7, 0.15), (0, 1.35, 0))],
            "anchors": {"front": (0, 0, 0.8, 180.0), "back": (0, 0, -0.8, 0.0)}, "top": 2.7}
    wall_window = {"size": (4.0, 2.7, 0.15), "parts": _frame(4.0, 2.7, 0.15, 1.4, 1.3, 0.9),
                   "anchors": dict(wall["anchors"]), "top": 2.7}
    wall_door = {"size": (4.0, 2.7, 0.15), "parts": _frame(4.0, 2.7, 0.15, 1.0, 2.2, 0.0),
                 "anchors": dict(wall["anchors"]), "top": 2.7}
    car = {"size": (1.8, 1.45, 4.4),
           "parts": [_p("box", (1.8, 0.65, 4.4), (0, 0.6, 0)), _p("box", (1.6, 0.55, 2.2), (0, 1.2, -0.2), LIGHT)]
           + [_p("cylinder", (0.66, 0.25, 0.66), (sx * 0.82, 0.33, sz * 1.38), DARK, (0, 0, 90))
              for sx in (-1, 1) for sz in (-1, 1)], "top": 1.45}
    car["anchors"] = {**_sides(1.8, 4.4, 0.5), "driver": (1.3, 0, 0.3, -90.0), "passenger": (-1.3, 0, 0.3, 90.0)}
    def vehicle(w, h, d, cabin, wheel):
        parts = [_p("box", (w, h * 0.45, d), (0, wheel * 0.55 + h * 0.225, 0)),
                 _p("box", (w * 0.94, h * 0.5, d * cabin), (0, wheel * 0.55 + h * 0.45 + h * 0.25, -d * (1 - cabin) / 2.2),
                    LIGHT)]
        parts += [_p("cylinder", (wheel, 0.28, wheel), (sx * (w / 2 - 0.1), wheel / 2, sz * (d / 2 - wheel * 0.9)), DARK,
                     (0, 0, 90)) for sx in (-1, 1) for sz in (-1, 1)]
        anchors = {**_sides(w, d, 0.5), "driver": (w / 2 + 0.4, 0, d * 0.12, -90.0),
                   "passenger": (-(w / 2 + 0.4), 0, d * 0.12, 90.0)}
        return {"size": (w, h, d), "parts": parts, "anchors": anchors, "top": h}

    tree = {"size": (2.4, 5.0, 2.4), "parts": [_p("cylinder", (0.3, 2.0, 0.3), (0, 1.0, 0), DARK),
                                               _p("sphere", (2.4, 3.2, 2.4), (0, 3.4, 0), GREEN)], "top": 5.0}
    tree["anchors"] = _sides(0.6, 0.6, 0.5)
    bush = {"size": (1.2, 1.0, 1.2), "parts": [_p("sphere", (1.2, 1.0, 1.2), (0, 0.5, 0), GREEN)], "top": 1.0}
    lamp = {"size": (0.4, 1.7, 0.4), "parts": [_p("cylinder", (0.32, 0.04, 0.32), (0, 0.02, 0), DARK),
                                               _p("cylinder", (0.04, 1.5, 0.04), (0, 0.79, 0), DARK),
                                               _p("cone", (0.42, 0.32, 0.42), (0, 1.54, 0), LIGHT)], "top": 1.7}
    small = {
        "plate": ((0.26, 0.02, 0.26), "cylinder"), "bowl": ((0.18, 0.08, 0.18), "cylinder"),
        "cup": ((0.08, 0.1, 0.08), "cylinder"), "glass": ((0.07, 0.13, 0.07), "cylinder"),
        "food": ((0.14, 0.06, 0.14), "sphere"), "book": ((0.15, 0.03, 0.22), "box"),
        "crate": ((0.6, 0.5, 0.4), "box"), "box": ((0.5, 0.5, 0.5), "box"),
    }
    out = {"table": table, "desk": desk, "chair": chair, "stool": stool, "sofa": sofa, "bench": bench, "bed": bed,
           "door": door, "window": window, "wall": wall, "wall_with_window": wall_window,
           "wall_with_door": wall_door, "car": car, "tree": tree, "bush": bush, "lamp": lamp,
           "van": vehicle(2.0, 2.3, 5.2, 0.85, 0.7), "truck": vehicle(2.4, 3.3, 7.5, 0.3, 0.9),
           "bus": vehicle(2.5, 3.1, 11.0, 0.95, 0.95)}
    def block(w, h, d, roof):
        parts = [_p("box", (w, h, d), (0, h / 2, 0))]
        door = _p("box", (1.1, 2.2, 0.06), (0, 1.1, d / 2 + 0.03), DARK)
        windows = [_p("box", (1.0, 1.2, 0.06), (x, y, d / 2 + 0.03), DARK)
                   for y in (h * 0.62,) for x in (-w * 0.3, w * 0.3)]
        if roof:
            parts += [_p("ramp", (w, roof, d / 2), (0, h + roof / 2, d / 4), LIGHT),
                      _p("ramp", (w, roof, d / 2), (0, h + roof / 2, -d / 4), LIGHT, (0, 180, 0))]
        anchors = {**_sides(w, d, 0.8), "door": (0, 0, d / 2 + 0.8, 180.0)}
        return {"size": (w, h + roof, d), "parts": parts + [door] + windows, "anchors": anchors, "top": h + roof}

    out["room"] = _room(6.0, 2.8, 5.0)
    out["building"] = block(8.0, 8.0, 7.0, 0.0)
    out["house"] = block(7.0, 4.5, 6.0, 2.2)
    bicycle = {"size": (0.6, 1.05, 1.75),
               "parts": [_p("cylinder", (0.68, 0.05, 0.68), (0, 0.34, sz * 0.52), DARK, (0, 0, 90)) for sz in (-1, 1)]
               + [_p("box", (0.05, 0.05, 1.05), (0, 0.62, 0), GREY, (-12, 0, 0)),
                  _p("box", (0.05, 0.55, 0.05), (0, 0.62, -0.3), GREY),
                  _p("box", (0.22, 0.05, 0.1), (0, 0.9, -0.3), DARK),
                  _p("box", (0.55, 0.04, 0.04), (0, 1.0, 0.42), GREY)],
               "top": 1.05, "anchors": {"left": (-0.6, 0, 0, 90.0), "right": (0.6, 0, 0, -90.0),
                                        "seat": (0, 0.9, -0.3, 0.0)}}
    out["bicycle"] = bicycle
    for name, (size, shape) in small.items():
        out[name] = {"size": size, "parts": [_p(shape, size, (0, size[1] / 2, 0), LIGHT)], "top": size[1]}
    out["bottle"] = {"size": (0.08, 0.32, 0.08), "parts": [_p("cylinder", (0.08, 0.24, 0.08), (0, 0.12, 0), LIGHT),
                                                          _p("cylinder", (0.03, 0.08, 0.03), (0, 0.28, 0), LIGHT)],
                     "top": 0.32}
    out["laptop"] = {"size": (0.33, 0.24, 0.23), "parts": [_p("box", (0.33, 0.02, 0.23), (0, 0.01, 0), DARK),
                                                          _p("box", (0.33, 0.22, 0.01), (0, 0.12, -0.11), DARK, (-15, 0, 0))],
                     "top": 0.02}
    for prefab in out.values():
        prefab.setdefault("anchors", {})
        prefab["anchors"].setdefault("top", (0.0, prefab["top"], 0.0, 0.0))
        prefab["anchors"].setdefault("center", (0.0, 0.0, 0.0, 0.0))
    return out


PREFABS = _prefabs()


class BlockoutError(ValueError):
    """A scene that cannot be built, with every problem found."""


@dataclass
class Placed:
    """A prop or person resolved into the world.

    Attributes:
        id: Its id.
        kind: Its type, or ``person``.
        x, y, z: Base centre in the world.
        turn: Radians about +Y; 0 faces +Z.
        size: Width, height and depth.
        anchors: Local anchors, ``(x, y, z, facing degrees)``.
        top: Height of its top surface above its base.
    """

    id: str
    kind: str
    x: float
    y: float
    z: float
    turn: float
    size: tuple
    anchors: dict
    top: float
    parts: list = field(default_factory=list)
    colour: str | None = None
    person: dict | None = None


class _Problems:
    def __init__(self):
        self.items: list[str] = []

    def add(self, where: str, message: str) -> None:
        if len(self.items) < MAX_PROBLEMS:
            self.items.append(f"{where}: {message}")

    def raise_if_any(self) -> None:
        if self.items:
            raise BlockoutError("The blockout scene has problems:\n- " + "\n- ".join(self.items))


def _depth(text: str) -> int:
    """Deepest nesting of brackets outside strings."""
    deepest = level = 0
    inside = escaped = False
    for ch in text:
        if inside:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                inside = False
        elif ch == '"':
            inside = True
        elif ch in "[{":
            level += 1
            deepest = max(deepest, level)
        elif ch in "]}":
            level -= 1
    return deepest


def _refuse_constant(name):
    raise ValueError(f"{name} is not a number this scene accepts")


def _pairs(pairs):
    seen = {}
    for key, value in pairs:
        key = key.strip().lower() if isinstance(key, str) else key
        if value is None:
            continue
        if key in seen:
            raise ValueError(f'the key "{key}" appears twice in one object')
        seen[key] = value
    return seen


def parse(text: str) -> dict:
    """The scene JSON read into a dict, bounded before it is parsed.

    Raises:
        BlockoutError: The text is too long, too deep or not a JSON object.
    """
    if not isinstance(text, str) or not text.strip():
        raise BlockoutError("The blockout scene is empty. Write a JSON object such as {\"props\": [...], \"shots\": [...]}.")
    size = len(text.encode("utf-8"))
    if size > MAX_BYTES:
        raise BlockoutError(f"The blockout scene is {size} bytes; the most read is {MAX_BYTES}.")
    if _depth(text) > MAX_DEPTH:
        raise BlockoutError(f"The blockout scene nests deeper than {MAX_DEPTH} levels.")
    try:
        data = json.loads(text, parse_constant=_refuse_constant, object_pairs_hook=_pairs)
    except json.JSONDecodeError as error:
        raise BlockoutError(f"The blockout scene is not valid JSON at line {error.lineno}, column {error.colno}: "
                            f"{error.msg}. Wire it through JSON Validate with repair on.") from None
    except ValueError as error:
        raise BlockoutError(f"The blockout scene is not valid JSON: {error}.") from None
    if not isinstance(data, dict):
        raise BlockoutError("The blockout scene has to be one JSON object, not a list or a value.")
    return data


# ---------------------------------------------------------------- field readers
def _keys(obj: dict, allowed: set, where: str, problems: _Problems) -> None:
    for key in obj:
        if key not in allowed:
            problems.add(where, f'unknown key "{key}"; allowed: {", ".join(sorted(allowed))}')


def _number(obj, key, where, problems, low, high, default=None):
    if key not in obj:
        return default
    value = obj[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        problems.add(f"{where}.{key}", f"needs a number from {low} to {high}")
        return default
    if not low <= value <= high:
        problems.add(f"{where}.{key}", f"{value} is outside {low} to {high}")
        return default
    return float(value)


def _vector(obj, key, where, problems, lengths=(2, 3), low=-500.0, high=500.0):
    if key not in obj:
        return None
    value = obj[key]
    if (not isinstance(value, list) or len(value) not in lengths
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value)):
        problems.add(f"{where}.{key}", f"needs a list of {' or '.join(map(str, lengths))} numbers")
        return None
    if any(not low <= v <= high for v in value):
        problems.add(f"{where}.{key}", f"every number has to be from {low} to {high}")
        return None
    return [float(v) for v in value]


def _word(obj, key, where, problems, choices, default=None):
    if key not in obj:
        return default
    value = obj[key]
    if isinstance(value, str):
        value = value.strip().lower().replace("-", " ").replace("_", " ")
        value = re.sub(r"\s+", " ", value)
        if value in choices:
            return value
    problems.add(f"{where}.{key}", f"needs one of: {', '.join(choices)}")
    return default


def _text(obj, key, where, problems, default=""):
    if key not in obj:
        return default
    value = obj[key]
    if not isinstance(value, str):
        problems.add(f"{where}.{key}", "needs a string")
        return default
    limit = TEXT_LIMITS.get(key, 300)
    value = CONTROL.sub(" ", value).replace("[", "(").replace("]", ")")
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) > limit:
        problems.add(f"{where}.{key}", f"is {len(value)} characters; the most is {limit}")
        return value[:limit]
    return value


def _colour(obj, where, problems):
    if "colour" not in obj and "color" not in obj:
        return None
    value = obj.get("colour", obj.get("color"))
    found = COLOUR.match(value.strip()) if isinstance(value, str) else None
    if not found:
        problems.add(f"{where}.colour", "needs #rrggbb, such as #a0a0a0")
        return None
    return "#" + found.group(1).lower()


def _h3_length(start: float, seconds: float, fps: float) -> float:
    """The last shot's seconds, lengthened so the scene ends on MiniMax H3's 17k+5 frame grid.

    Args:
        start: Seconds the shot starts at.
        seconds: Seconds the shot was asked for.
        fps: Frames a second.

    Returns:
        Seconds the shot lasts, never fewer than ``seconds``.
    """
    frames = max(5, int(round((start + seconds) * fps)))
    frames += (5 - frames) % 17
    return max(seconds, frames / fps - start)


def _blueprint_shots(text: str | None) -> list[dict]:
    """Each measured shot's ``seconds``, ``keys`` and ``handheld`` from Video Blueprint's JSON, or none."""
    if not text or not str(text).strip():
        return []
    try:
        data = json.loads(text)
    except ValueError:
        raise BlockoutError("The blueprint is not Video Blueprint's JSON. Wire its blueprint output in.") from None
    shots = data.get("shots") if isinstance(data, dict) else None
    out = []
    for shot in shots if isinstance(shots, list) else []:
        if isinstance(shot, dict) and isinstance(shot.get("keys"), list) and shot["keys"]:
            out.append({"seconds": float(shot.get("seconds", 3.0)), "keys": shot["keys"],
                        "handheld": bool(shot.get("handheld"))})
    return out


def _blueprint_people(text: str | None, key: str = "people") -> list[dict]:
    """The people Video Blueprint followed, or with ``key`` ``objects`` the objects it placed; each has a ``track``."""
    if not text or not str(text).strip():
        return []
    try:
        data = json.loads(text)
    except ValueError:
        return []
    people = data.get(key) if isinstance(data, dict) else None
    return [p for p in people if isinstance(p, dict) and p.get("track")] if isinstance(people, list) else []


def _still(keys: list[dict]) -> bool:
    """Whether a camera path stands still: under 10 cm of travel, 2 degrees of turn and 3% of lens."""
    first = keys[0]
    ahead = [first["target"][i] - first["p"][i] for i in range(3)]
    for key in keys[1:]:
        if math.dist(key["p"], first["p"]) > 0.1:
            return False
        now = [key["target"][i] - key["p"][i] for i in range(3)]
        dot = sum(a * b for a, b in zip(ahead, now))
        size = math.hypot(*ahead) * math.hypot(*now)
        if size > 0 and math.degrees(math.acos(max(-1.0, min(1.0, dot / size)))) > 2.0:
            return False
        if abs(key.get("fov", 1.0) / max(first.get("fov", 1.0), 1e-6) - 1.0) > 0.03:
            return False
    return True


def _picture(item, where, problems, key="picture") -> int | None:
    """A reference picture's number, `<Picture N>` on the reference node, 1 to 9."""
    value = _number(item, key, where, problems, 1, 9)
    if value is None:
        return None
    if value != int(value):
        problems.add(f"{where}.{key}", "needs a whole number from 1 to 9")
        return None
    return int(value)


def _gender(item, where, problems, name, description) -> str:
    """The kind of person a figure is drawn as: man, woman, boy, girl, child or person."""
    if "gender" in item:
        value = item["gender"]
        word = value.strip().lower() if isinstance(value, str) else ""
        for kind, words in GENDER_WORDS.items():
            if word in words:
                return kind
        problems.add(f"{where}.gender", "needs one of man, woman, boy, girl, child or person")
        return "person"
    words = {w for w in re.findall(r"[a-z]+", f"{name or ''} {description or ''}".lower()) if len(w) > 1}
    pronouns = {"he", "his", "him", "she", "her", "hers"}
    for kind in ("girl", "boy", "woman", "man", "child"):
        if words & (set(GENDER_WORDS[kind]) - pronouns):
            return kind
    for kind in ("woman", "man"):
        if words & set(GENDER_WORDS[kind]) & pronouns:
            return kind
    return "person"


def _grey(colour: str) -> str:
    """A colour as the light-to-mid grey of the same brightness, kept between 42% and 85%."""
    value = colour.lstrip("#")
    r, g, b = (int(value[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    level = min(0.85, max(0.42, 0.2126 * r + 0.7152 * g + 0.0722 * b))
    return "#" + f"{round(level * 255):02x}" * 3


def _ref(obj, key, where, problems):
    if key not in obj:
        return None
    value = obj[key]
    if not isinstance(value, str) or not value.strip():
        problems.add(f"{where}.{key}", "needs an id, or id.anchor such as table.left")
        return None
    name, _, anchor = value.strip().lower().partition(".")
    anchor = ANCHOR_ALIASES.get(anchor, anchor)
    if not ID.match(name) or (anchor and not ID.match(anchor)):
        problems.add(f"{where}.{key}", f'"{value}" is not an id or id.anchor')
        return None
    return name, anchor


# ---------------------------------------------------------------- props and parts
def _ramped(f: float, seconds: float) -> float:
    """How far along a walk is, speeding up and slowing down over its first and last third of a second."""
    r = min(0.35 / max(seconds, 1e-6), 1.0 / 3.0)
    top = 1.0 / (1.0 - r)
    if f < r:
        return top * f * f / (2 * r)
    if f > 1 - r:
        return 1 - top * (1 - f) * (1 - f) / (2 * r)
    return top * (f - r / 2)


def _rotate(x: float, z: float, turn: float) -> tuple[float, float]:
    c, s = math.cos(turn), math.sin(turn)
    return x * c + z * s, -x * s + z * c


def _read_parts(raw, where, problems) -> list[dict] | None:
    if not isinstance(raw, list) or not raw:
        problems.add(where, "needs a list of parts, each {\"shape\": ..., \"size\": [w, h, d], \"position\": [x, y, z]}")
        return None
    if len(raw) > MAX_KIND_PARTS:
        problems.add(where, f"has {len(raw)} parts; the most is {MAX_KIND_PARTS}")
        return None
    parts = []
    for i, part in enumerate(raw):
        at = f"{where}[{i}]"
        if not isinstance(part, dict):
            problems.add(at, "needs an object")
            continue
        _keys(part, {"shape", "size", "position", "rotate", "colour", "color"}, at, problems)
        shape = _word(part, "shape", at, problems, SHAPES, "box")
        size = _vector(part, "size", at, problems, (3,), 0.005, 200.0) or [0.5, 0.5, 0.5]
        position = _vector(part, "position", at, problems, (3,), -200.0, 200.0)
        if position is None:
            position = [0.0, size[1] / 2, 0.0]
        rotate = _vector(part, "rotate", at, problems, (3,), -360.0, 360.0) or [0.0, 0.0, 0.0]
        parts.append({"shape": shape, "size": tuple(size), "at": tuple(position),
                      "tone": _colour(part, at, problems) or GREY, "rotate": tuple(rotate)})
    return parts


def _built(parts: list[dict]) -> dict:
    """A prefab made of parts: its size from the parts' extent, its top their highest point."""
    xs, ys, zs = [], [], []
    for part in parts:
        (w, h, d), (x, y, z) = part["size"], part["at"]
        xs += [x - w / 2, x + w / 2]
        ys += [y - h / 2, y + h / 2]
        zs += [z - d / 2, z + d / 2]
    width, height, depth = max(xs) - min(xs), max(ys), max(zs) - min(zs)
    anchors = {**_sides(width, depth), "center": (0.0, 0.0, 0.0, 0.0), "top": (0.0, height, 0.0, 0.0)}
    return {"size": (width, height, depth), "parts": parts, "anchors": anchors, "top": height}


def _scaled(prefab: dict, size: list | None) -> dict:
    if not size:
        return prefab
    if prefab.get("room"):
        return _room(*(float(v) for v in size))
    sx, sy, sz = (size[i] / max(1e-6, prefab["size"][i]) for i in range(3))
    parts = [{**p, "size": (p["size"][0] * sx, p["size"][1] * sy, p["size"][2] * sz),
              "at": (p["at"][0] * sx, p["at"][1] * sy, p["at"][2] * sz)} for p in prefab["parts"]]
    anchors = {k: (a[0] * sx, a[1] * sy, a[2] * sz, a[3]) for k, a in prefab["anchors"].items()}
    return {"size": tuple(size), "parts": parts, "anchors": anchors, "top": prefab["top"] * sy}


def _person_size(height: float) -> tuple:
    return (0.5 * height / 1.75, height, 0.3 * height / 1.75)


def _person_anchors(height: float) -> dict:
    return {**_sides(0.5, 0.3, 0.45), "center": (0.0, 0.0, 0.0, 0.0), "top": (0.0, height, 0.0, 0.0)}


# ---------------------------------------------------------------- the scene
@dataclass
class Scene:
    """A resolved scene.

    Attributes:
        objects: The object descriptor holding everything placed.
        camera: The camera descriptor carrying every shot.
        summary, description, soundscape: The H3 prompt fields.
        seconds: Length of all shots together.
        shots: One row per shot for the report.
        counts: Props, people, shapes and shots built.
    """

    objects: dict
    camera: dict
    summary: str
    description: str
    soundscape: str
    prompt: str
    seconds: float
    shots: list
    counts: dict


TOP_KEYS = {"look", "setting", "sound", "floor", "kinds", "props", "people", "shots", "lighting", "tempo",
            "figures", "music", "set_picture"}

#: What the camera does in a moving rig's shot, for the H3 prompt.
RIG_WORDS = {
    "fly by": "the camera sweeps fast past its subject in {s} seconds, turning to keep it in frame",
    "fly through": "the camera rushes forward through the opening in {s} seconds without slowing",
    "fly over": "the camera passes high overhead in {s} seconds and looks on ahead",
    "reveal": "the camera starts low and close, then rises and pulls back over {s} seconds",
    "chase": "the camera rides behind, keeping pace for the whole {s} seconds",
    "orbit": "the camera circles its subject steadily for {s} seconds",
}

#: What each ready-made type is in the film when a prop gives no description of its own.
REAL = {"table": "a real wooden table", "desk": "a real desk", "chair": "a real chair", "stool": "a real bar stool",
        "sofa": "a real upholstered sofa", "bench": "a real bench", "bed": "a real made bed", "car": "a real car",
        "van": "a real panel van", "truck": "a real truck", "bus": "a real bus", "bicycle": "a real bicycle",
        "tree": "a real full-grown tree", "bush": "a real leafy bush", "lamp": "a real lamp",
        "plate": "a real plate", "bowl": "a real bowl", "cup": "a real cup", "glass": "a real glass",
        "bottle": "a real bottle", "food": "real food", "book": "a real book", "laptop": "a real laptop",
        "crate": "a real wooden crate", "box": "a real box", "building": "a real building",
        "house": "a real house", "room": "a real, fully furnished room"}

#: How each kind of prop looks in the block-out, for saying what replaces it.
BLOCK_LOOK = {"car": "the grey box on round wheels", "van": "the tall grey box on round wheels",
              "truck": "the long grey box on round wheels", "bus": "the long grey box on round wheels",
              "tree": "the trunk and ball shape", "bush": "the round shape", "building": "the plain block",
              "house": "the plain block with a sloped top", "room": "the plain grey walls"}

#: The colours each kind of person is drawn in, first, second and third of that kind, with the prompt's word.
PEOPLE_COLOURS = {"man": (("#4f7fd6", "blue"), ("#24418c", "navy"), ("#7fd0e6", "sky blue")),
                  "woman": (("#d65050", "red"), ("#8c2430", "dark red"), ("#ee8a5a", "coral")),
                  "boy": (("#8bbcef", "light blue"), ("#5aa0a0", "teal"), ("#b6c8f2", "pale blue")),
                  "girl": (("#f0a0b4", "pink"), ("#c0609a", "magenta"), ("#f2c2e0", "pale pink")),
                  "child": (("#e6c552", "yellow"), ("#c89a30", "amber"), ("#f2e08a", "pale yellow")),
                  "person": (("#c8ae86", "tan"), ("#8c6e4a", "brown"), ("#e0d0b0", "beige"))}

#: Words read as each kind of person, in a gender value or, when none is given, in the name and description.
GENDER_WORDS = {"man": ("man", "male", "m", "guy", "gentleman", "father", "dad", "husband", "boyfriend", "groom",
                        "he", "his", "him", "cowboy", "king", "brother", "son", "uncle", "sir"),
                "woman": ("woman", "female", "f", "lady", "mother", "mom", "wife", "girlfriend", "bride", "she",
                          "her", "hers", "queen", "sister", "daughter", "aunt", "actress", "waitress"),
                "boy": ("boy",), "girl": ("girl",),
                "child": ("child", "kid", "toddler", "baby", "infant"),
                "person": ("person", "any", "neutral", "unknown", "other")}

#: How a scene's lighting value reads in the H3 prompt; the block-out itself is always lit flat and bright.
LIGHTING_WORDS = {"day": "", "studio": "", "evening": "It is evening, in warm, low golden sunlight.",
                  "night": "It is night, lit by streetlights, windows and other practical lights."}

#: What the H3 prompt says must never appear from the block-out.
NOT_SHOWN = ("Nothing of the block-out appears: no grey models, clay shading, blocks, boxes, cubes, capsules, "
             "coloured stand-in figures, drawn faces, checkerboard floor, plain backdrop, flat studio light or "
             "untextured geometry.")
PLACE_KEYS = {"id", "at", "on", "beside", "side", "position", "facing", "turn"}
PROP_KEYS = PLACE_KEYS | {"type", "size", "colour", "color", "count", "parts", "steps", "motion", "held_by",
                          "against", "overlap", "description", "picture", "track"}
PERSON_KEYS = PLACE_KEYS | {"name", "description", "pose", "sit_on", "look_at", "height", "colour", "color",
                            "motion", "holding", "held_by", "against", "voice", "gender", "picture",
                            "keep_outfit", "track"}
SHOT_KEYS = {"seconds", "rig", "subjects", "from", "onto", "side", "angle", "lens", "move", "amount", "action",
             "position", "target", "look_at", "follow", "keys", "shake", "beats", "ease", "through", "dialogue",
             "sounds"}

#: How a camera speeds up and slows down between keys or through a move.
EASES = ("linear", "ease in", "ease out", "ease in and out", "slow in", "slow out", "snap", "anticipate",
         "overshoot", "hold")

#: Camera shake: position wobble in metres, then aim wobble in degrees.
SHAKES = {"none": (0.0, 0.0), "light": (0.008, 0.15), "handheld": (0.02, 0.35), "heavy": (0.05, 0.9),
          "impact": (0.09, 1.6)}

#: Most keys in one shot's camera path.
MAX_KEYS = 16

#: Points on a person the camera can lock on to.
PERSON_POINTS = ("face", "eyes", "head", "chest", "hands", "feet", "center")

#: A person given a height under this, in metres, stands on the floor; higher is a raised floor such as a stage.
PERSON_FLOOR = 2.2

#: Seconds between two moves short enough that they are told as one, set off to stop.
RUN_JOIN = 0.2

#: Words for a direction rather than a thing to face, read as no facing given.
FACING_WORDS = {"forward", "forwards", "ahead", "front", "straight", "camera", "viewer", "the camera"}

#: Metres a measured person moves before a new motion step starts, and the speed past which they run, m/s.
TRACK_STEP = 0.3
RUN_SPEED = 2.2

#: Other words for a point on a person, read as the point named.
PERSON_POINT_WORDS = {"top": "head", "bottom": "feet", "hand": "hands", "foot": "feet", "eye": "eyes",
                      "centre": "center", "middle": "center", "body": "center", "torso": "chest"}


class _Builder:
    def __init__(self, data: dict):
        self.data = data
        self.problems = _Problems()
        self.kinds: dict[str, dict] = {}
        self.raw: dict[str, tuple[str, dict, str]] = {}
        self.placed: dict[str, Placed] = {}
        self.visiting: list[str] = []
        self.order: list[str] = []
        self.parts_total = 0
        self.paths: dict[str, list[tuple]] = {}
        self.held: dict[str, str] = {}
        self.voices: dict[str, str] = {}
        self.facing_later: list[tuple] = []
        self.prop_text: dict[str, str] = {}
        self.prop_picture: dict[str, int] = {}
        self.grid: list[float] | None = None
        self.downbeats: list[float] = []
        self.song_tempo = 0.0
        self.fps = 24.0
        self.labels: dict[str, int] = {}
        self.labelled = False
        self.mentioned: set = set()
        self.moving: set[str] = set()
        self.moved: set[str] = set()
        self.seats_taken: dict[str, set] = {}
        self.now = 0.0
        self.span = 0.0

    # -- reading
    def read(self) -> None:
        p = self.problems
        _keys(self.data, TOP_KEYS, "scene", p)
        kinds = self.data.get("kinds", {})
        if not isinstance(kinds, dict):
            p.add("kinds", "needs an object of name: {\"parts\": [...]}")
            kinds = {}
        if len(kinds) > MAX_KINDS:
            p.add("kinds", f"has {len(kinds)} kinds; the most is {MAX_KINDS}")
        for name, spec in list(kinds.items())[:MAX_KINDS]:
            where = f"kinds.{name}"
            if not ID.match(str(name)):
                p.add(where, "a kind's name is lower case letters, digits and _")
                continue
            if name in PREFABS or name in SHAPES or name == "person":
                p.add(where, f'"{name}" is already a ready-made type; choose another name')
                continue
            if not isinstance(spec, dict):
                p.add(where, "needs {\"parts\": [...]}")
                continue
            _keys(spec, {"parts"}, where, p)
            parts = _read_parts(spec.get("parts"), f"{where}.parts", p)
            if parts:
                self.kinds[name] = _built(parts)
        for section, kind in (("props", "prop"), ("people", "person")):
            items = self.data.get(section, [])
            if not isinstance(items, list):
                p.add(section, "needs a list")
                continue
            limit = MAX_PROPS if section == "props" else MAX_PEOPLE
            if len(items) > limit:
                p.add(section, f"has {len(items)} entries; the most is {limit}")
                items = items[:limit]
            for i, item in enumerate(items):
                where = f"{section}[{i}]"
                if not isinstance(item, dict):
                    p.add(where, "needs an object")
                    continue
                _keys(item, PROP_KEYS if kind == "prop" else PERSON_KEYS, where, p)
                ident = item.get("id", f"{kind}_{i + 1}")
                if not isinstance(ident, str) or not ID.match(ident.strip().lower()):
                    p.add(f"{where}.id", "is lower case letters, digits and _, starting with a letter")
                    continue
                ident = ident.strip().lower()
                count = int(_number(item, "count", where, p, 1, MAX_COUNT, 1) or 1) if kind == "prop" else 1
                names = [ident] if count == 1 else [f"{ident}_{k + 1}" for k in range(count)]
                for k, name in enumerate(names):
                    if name in self.raw:
                        p.add(f"{where}.id", f'"{name}" is used twice')
                        continue
                    self.raw[name] = (kind, item, where if count == 1 else f"{where} (copy {k + 1})")
                    self.order.append(name)
                if len(self.raw) > MAX_PROPS + MAX_PEOPLE:
                    p.add(section, "too many props and people in all")
                    return

    # -- resolving
    def resolve(self, name: str, where: str) -> Placed | None:
        if name in self.placed:
            return self.placed[name]
        if name not in self.raw:
            known = ", ".join(self.order[:30])
            self.problems.add(where, f'no prop or person "{name}"; there is {known or "nothing yet"}')
            return None
        if name in self.visiting:
            self.problems.add(where, "places things relative to each other in a loop: " + " -> ".join(self.visiting + [name]))
            return None
        if len(self.visiting) >= MAX_CHAIN:
            self.problems.add(where, f"chains more than {MAX_CHAIN} placements")
            return None
        self.visiting.append(name)
        try:
            kind, item, at = self.raw[name]
            placed = self._prop(name, item, at) if kind == "prop" else self._person(name, item, at)
        finally:
            self.visiting.pop()
        if placed is not None:
            self.placed[name] = placed
        return placed

    def _anchor(self, ref, where) -> tuple[Placed | None, tuple]:
        """A referenced thing and the world point and facing of its anchor."""
        target = self.resolve(ref[0], where)
        if target is None:
            return None, (0.0, 0.0, 0.0, 0.0)
        anchor = ref[1] or "center"
        if anchor not in target.anchors:
            self.problems.add(where, f'"{target.id}" has no anchor "{anchor}"; it has {", ".join(sorted(target.anchors))}')
            return target, (target.x, target.y, target.z, target.turn)
        ax, ay, az, facing = target.anchors[anchor]
        dx, dz = _rotate(ax, az, target.turn)
        return target, (target.x + dx, target.y + ay, target.z + dz, target.turn + math.radians(facing))

    def _place(self, name, item, where, size) -> tuple[float, float, float, float]:
        """Where a prop or person stands and the way it faces."""
        p = self.problems
        x = y = z = 0.0
        turn = None
        position = _vector(item, "position", where, p)
        if position is not None:
            if len(position) == 2:
                x, z = position
            else:
                x, y, z = position
                if self.raw.get(name, ("",))[0] == "person" and y < PERSON_FLOOR:
                    y = 0.0
        for key in ("at", "on", "beside", "against"):
            ref = _ref(item, key, where, p)
            if ref is None:
                continue
            target, (ax, ay, az, facing) = self._anchor(ref, f"{where}.{key}")
            if target is None:
                continue
            if key == "at":
                x, y, z, turn = ax, ay, az, facing
            elif key == "on":
                x, z = (ax, az) if ref[1] else (target.x, target.z)
                y = target.y + (target.anchors.get(ref[1], (0, target.top))[1] if ref[1] else target.top)
                turn = target.turn
            elif key == "against":
                reach = target.size[2] / 2 + size[2] / 2 + 0.03
                dx, dz = _rotate(0.0, reach, target.turn)
                x, y, z, turn = target.x + dx, target.y, target.z + dz, target.turn
            else:
                side = _word(item, "side", where, p, ("left", "right", "front", "behind"), "right")
                tw, _, td = target.size
                gap = 0.2
                lx, lz = {"left": (-(tw + size[0]) / 2 - gap, 0.0), "right": ((tw + size[0]) / 2 + gap, 0.0),
                          "front": (0.0, (td + size[2]) / 2 + gap), "behind": (0.0, -(td + size[2]) / 2 - gap)}[side]
                dx, dz = _rotate(lx, lz, target.turn)
                x, y, z, turn = target.x + dx, target.y, target.z + dz, target.turn
            break
        if not any(k in item for k in ("position", "at", "on", "beside", "sit_on", "held_by", "against")):
            fixture = isinstance(item.get("type"), str) and item["type"].strip().lower() in FIXTURES
            x, z = self._free_spot(size, -3.0 if fixture else 0.0)
        face = _ref(item, "facing", where, p)
        if face is not None and face[0] in self.visiting:
            # Two things facing each other: the turn waits until both stand somewhere.
            self.facing_later.append((name, face[0], _number(item, "turn", where, p, -360.0, 360.0)))
        elif face is not None:
            target = self.resolve(face[0], f"{where}.facing")
            if target is not None and (abs(target.x - x) + abs(target.z - z)) > 1e-6:
                turn = math.atan2(target.x - x, target.z - z)
        extra = _number(item, "turn", where, p, -360.0, 360.0)
        if extra is not None:
            turn = (turn or 0.0) + math.radians(extra) if face is not None or turn is not None else math.radians(extra)
        return x, y, z, turn or 0.0

    def _prop(self, name, item, where) -> Placed | None:
        p = self.problems
        kind = item.get("type", "box")
        kind = kind.strip().lower().replace(" ", "_").replace("-", "_") if isinstance(kind, str) else ""
        size = _vector(item, "size", where, p, (3,), 0.005, 200.0)
        if "parts" in item:
            parts = _read_parts(item["parts"], f"{where}.parts", p)
            if not parts:
                return None
            prefab = _scaled(_built(parts), size)
            kind = "built"
        elif kind in self.kinds:
            prefab = _scaled(self.kinds[kind], size)
        elif kind in PREFABS:
            prefab = _scaled(PREFABS[kind], size)
        elif kind in SHAPES:
            dims = tuple(size) if size else ((20.0, 0.02, 20.0) if kind == "ground" else (1.0, 1.0, 1.0))
            steps = int(_number(item, "steps", where, p, 1, 60, 6) or 6)
            part = {"shape": kind, "size": dims, "at": (0.0, dims[1] / 2, 0.0), "tone": GREY, "rotate": (0, 0, 0),
                    "steps": steps}
            prefab = {"size": dims, "parts": [part], "top": dims[1],
                      "anchors": {**_sides(dims[0], dims[2]), "center": (0, 0, 0, 0.0), "top": (0, dims[1], 0, 0.0)}}
        else:
            choices = sorted(set(PREFABS) | set(self.kinds) | set(SHAPES))
            p.add(f"{where}.type", f'unknown type "{item.get("type")}"; use one of: {", ".join(choices)}, '
                  "or give \"parts\"")
            return None
        self.parts_total += len(prefab["parts"])
        if self.parts_total > MAX_PARTS:
            p.add(where, f"the scene holds more than {MAX_PARTS} shapes")
            return None
        holder = _ref(item, "held_by", where, p)
        if holder is not None:
            person = self.resolve(holder[0], f"{where}.held_by")
            if person is None or person.person is None:
                if person is not None:
                    p.add(f"{where}.held_by", f'"{holder[0]}" is not a person')
                return None
            self.held[name] = holder[0]
            local = self._local_point(holder[0], "hands", f"{where}.held_by")
            dx, dz = _rotate(local[0], local[2], person.turn)
            x, y, z, turn = person.x + dx, person.y + local[1], person.z + dz, person.turn
        else:
            x, y, z, turn = self._place(name, item, where, prefab["size"])
        copies = [n for n in self.order if n.startswith(f"{item.get('id', '')}_")] if "count" in item else []
        if copies and name in copies:
            index, total = copies.index(name), len(copies)
            spread = (index - (total - 1) / 2) * (prefab["size"][0] + 0.08)
            dx, dz = _rotate(spread, 0.0, turn)
            x, z = x + dx, z + dz
        text = _text(item, "description", where, p)
        if text:
            self.prop_text[name] = text
        picture = _picture(item, where, p)
        if picture is not None:
            self.prop_picture[name] = picture
        return Placed(name, kind, x, y, z, turn, prefab["size"], prefab["anchors"], prefab["top"], prefab["parts"],
                      _colour(item, where, p))

    def _person(self, name, item, where) -> Placed | None:
        p = self.problems
        height = _number(item, "height", where, p, 0.5, 2.5, 1.75) or 1.75
        pose = _word(item, "pose", where, p, POSES, "stand")
        seat = None
        sit = _ref(item, "sit_on", where, p)
        if sit is not None:
            seat_ref = self._seat(sit, f"{where}.sit_on", name)
            if seat_ref is None:
                return None
            target, (ax, ay, az, facing) = self._anchor(seat_ref, f"{where}.sit_on")
            if target is None:
                return None
            x, y, z, turn = ax, target.y, az, facing
            seat = max(0.2, ay - target.y)
            pose = "sit"
        else:
            x, y, z, turn = self._place(name, item, where, _person_size(height))
            if pose == "sit":
                seat = 0.46
        voice = _text(item, "voice", where, p)
        if voice:
            self.voices[name] = voice
        info = {"height": height, "pose": pose, "seat": seat, "head": 0.0,
                "name": _text(item, "name", where, p) or name.replace("_", " ").title(),
                "description": _text(item, "description", where, p)}
        info["gender"] = _gender(item, where, p, info["name"], info["description"])
        info["picture"] = _picture(item, where, p)
        keep = item.get("keep_outfit", False)
        if not isinstance(keep, bool):
            p.add(f"{where}.keep_outfit", "needs true or false")
            keep = False
        info["keep_outfit"] = keep
        return Placed(name, "person", x, y, z, turn, _person_size(height), _person_anchors(height), height, [],
                      _colour(item, where, p), info)

    def _footprint(self, placed) -> tuple[float, float, float, float]:
        """A thing's floor extent as x and z half widths around its centre, turned."""
        w, _, d = placed.size
        c, s = abs(math.cos(placed.turn)), abs(math.sin(placed.turn))
        return placed.x, placed.z, (w * c + d * s) / 2, (w * s + d * c) / 2

    def _separate(self) -> None:
        """Push apart props and standing people that intersect, unless allowed to overlap or set on, in or at something."""
        movable = []
        for name in self.order:
            placed = self.placed.get(name)
            kind, item, _ = self.raw[name]
            if placed is None or item.get("overlap") is True or name in self.held:
                continue
            if placed.kind in ("room",) + FIXTURES:
                continue
            if any(k in item for k in ("on", "sit_on", "at", "held_by")) or name.endswith("_chair"):
                continue
            if placed.person is not None and placed.person["pose"] == "sit":
                continue
            movable.append(name)
        settled: list[str] = []
        for name in movable:
            placed = self.placed[name]
            for _ in range(24):
                hit = None
                for other in settled:
                    ox, oz, ow, od = self._footprint(self.placed[other])
                    x, z, w, d = self._footprint(placed)
                    gap_x = ow + w + 0.1 - abs(x - ox)
                    gap_z = od + d + 0.1 - abs(z - oz)
                    if gap_x > 0 and gap_z > 0:
                        hit = (ox, oz, gap_x, gap_z)
                        break
                if hit is None:
                    break
                ox, oz, gap_x, gap_z = hit
                if gap_x <= gap_z:
                    placed.x += gap_x if placed.x >= ox else -gap_x
                else:
                    placed.z += gap_z if placed.z >= oz else -gap_z
            settled.append(name)

    def _free_spot(self, size, line: float = 0.0) -> tuple[float, float]:
        """The nearest clear floor along x on a line, then further back, for something given no place."""
        w, _, d = size
        step = w + 0.6
        for row in range(6):
            for k in range(13):
                x = (k + 1) // 2 * step * (1 if k % 2 else -1)
                z = line - row * (d + 1.0)
                clear = True
                for other in self.placed.values():
                    if other.kind in ("room",) + FIXTURES:
                        continue
                    reach = max(other.size[0], other.size[2]) / 2
                    if abs(other.x - x) < reach + w / 2 + 0.3 and abs(other.z - z) < reach + d / 2 + 0.3:
                        clear = False
                        break
                if clear:
                    return x, z
        return 0.0, 0.0

    def _seat(self, ref, where, person) -> tuple[str, str] | None:
        """The seat a person takes: the one named, the first one free, or a chair set at a side of something."""
        target = self.resolve(ref[0], where)
        if target is None:
            return None
        taken = self.seats_taken.setdefault(ref[0], set())
        seats = [a for a in target.anchors if a == "seat" or a.startswith("seat_")]
        if seats:
            if ref[1]:
                taken.add(ref[1])
                return ref[0], ref[1]
            for seat in seats:
                if seat not in taken:
                    taken.add(seat)
                    return ref[0], seat
            self.problems.add(where, f'every seat on "{ref[0]}" is taken: {", ".join(seats)}')
            return None
        sides = [ref[1]] if ref[1] else [a for a in ("left", "right", "front", "back") if a in target.anchors]
        free = [a for a in sides if a in target.anchors and a not in taken]
        if not free:
            self.problems.add(where, f'"{ref[0]}" has no free side to set a chair at; use a chair, stool, sofa, '
                              "bench or bed")
            return None
        side = free[0]
        taken.add(side)
        _, (ax, ay, az, facing) = self._anchor((ref[0], side), where)
        chair = f"{person}_chair"
        prefab = PREFABS["chair"]
        self.placed[chair] = Placed(chair, "chair", ax, target.y, az, facing, prefab["size"], prefab["anchors"],
                                    prefab["top"], prefab["parts"])
        self.order.append(chair)
        self.raw[chair] = ("prop", {}, where)
        self.parts_total += len(prefab["parts"])
        return chair, "seat"

    # -- motion
    def _destination(self, step, where, when: float, total: float) -> tuple | None:
        value = step.get("to")
        if isinstance(value, list):
            point = _vector(step, "to", where, self.problems)
            if point is None:
                return None
            return (point[0], 0.0, point[1]) if len(point) == 2 else tuple(point)
        ref = _ref(step, "to", where, self.problems)
        if ref is None:
            self.problems.add(f"{where}.to", "needs an id, id.anchor or [x, z]")
            return None
        target = self.resolve(ref[0], f"{where}.to")
        if target is None:
            return None
        anchor = ref[1] or ("front" if "front" in target.anchors else "center")
        if anchor not in target.anchors:
            self.problems.add(f"{where}.to", f'"{ref[0]}" has no anchor "{anchor}"; it has '
                              f'{", ".join(sorted(target.anchors))}')
            return None
        # A moving target is met where it is when the step ends.
        if ref[0] not in self.moved and ref[0] not in self.moving:
            self._motion(ref[0], total)
        x, y, z, turn = self._at(ref[0], when)
        ax, ay, az, _ = target.anchors[anchor]
        dx, dz = _rotate(ax, az, turn)
        return x + dx, (y + ay if ref[1] else y), z + dz

    def _motion(self, name: str, total: float) -> None:
        """Key the path a prop or person travels: hold, turn toward the next place, go there."""
        kind, item, where = self.raw[name]
        placed = self.placed.get(name)
        if placed is None or name in self.moved or name in self.moving:
            return
        if "motion" not in item:
            if placed.person is None or placed.person["pose"] not in ("walk", "run"):
                return
            # Someone posed walking or running, given nowhere to go, heads for what they face at a natural pace.
            speed = 3.5 if placed.person["pose"] == "run" else 1.3
            face = item.get("facing")
            goal = self.placed.get(face.strip().lower().split(".")[0]) if isinstance(face, str) else None
            if goal is not None:
                to, distance = face, math.hypot(goal.x - placed.x, goal.z - placed.z)
            else:
                distance = 3.0
                to = [placed.x + math.sin(placed.turn) * distance, placed.z + math.cos(placed.turn) * distance]
            item = {**item, "motion": [{"to": to, "start": 0.0, "end": min(total, max(1.0, distance / speed))}]}
        self.moving.add(name)
        try:
            self._steps(name, item, where, placed, total)
        finally:
            self.moving.discard(name)
            self.moved.add(name)

    def _steps(self, name, item, where, placed, total) -> None:
        p = self.problems
        steps = item["motion"]
        if not isinstance(steps, list) or not steps or len(steps) > MAX_STEPS:
            p.add(f"{where}.motion", f'needs a list of 1 to {MAX_STEPS} steps, each {{"to": ..., "start": s, "end": s}}')
            return
        x, y, z, turn = placed.x, placed.y, placed.z, placed.turn
        keys = [(0.0, x, y, z, turn)]
        last = 0.0
        for i, step in enumerate(steps):
            at = f"{where}.motion[{i}]"
            if not isinstance(step, dict):
                p.add(at, 'needs {"to": ..., "start": seconds, "end": seconds}')
                continue
            _keys(step, {"to", "start", "end"}, at, p)
            start = _number(step, "start", at, p, 0.0, MAX_SECONDS, last)
            end = _number(step, "end", at, p, 0.0, MAX_SECONDS)
            if end is None:
                p.add(f"{at}.end", "needs the second the move ends")
                continue
            if start < last - 1e-6:
                p.add(at, f"starts at {start:g} s, before the step before it ends at {last:g} s")
                continue
            if end <= start:
                p.add(at, "end has to come after start")
                continue
            if start >= total - 1e-6:
                continue
            end = min(end, total)
            dest = self._destination(step, at, end, total)
            if dest is not None and placed.person is not None and dest[1] < PERSON_FLOOR:
                dest = (dest[0], 0.0, *dest[2:])
            if dest is None:
                continue
            dx, dz = dest[0] - x, dest[2] - z
            heading = math.atan2(dx, dz) if math.hypot(dx, dz) > 1e-3 else turn
            lead = max(last, start - 0.4)
            if lead > keys[-1][0] + 1e-6:
                keys.append((lead, x, y, z, turn))
            keys.append((start, x, y, z, heading))
            x, y, z, turn = dest[0], dest[1], dest[2], heading
            keys.append((end, x, y, z, turn))
            last = end
        if len(keys) > 1:
            self.paths[name] = keys

    def _at(self, name: str, seconds: float) -> tuple:
        """Where a prop or person is, and its turn, at a moment."""
        placed = self.placed[name]
        keys = self.paths.get(name)
        if not keys:
            return placed.x, placed.y, placed.z, placed.turn
        if seconds <= keys[0][0]:
            return keys[0][1:]
        for a, b in zip(keys, keys[1:]):
            if seconds <= b[0]:
                f = (seconds - a[0]) / max(1e-9, b[0] - a[0])
                if placed.person is None:
                    f = f * f * (3 - 2 * f)
                else:
                    f = _ramped(f, b[0] - a[0])
                delta = (b[4] - a[4] + math.pi) % (2 * math.pi) - math.pi
                return (a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f, a[3] + (b[3] - a[3]) * f,
                        a[4] + delta * f)
        return keys[-1][1:]

    def _local_point(self, name: str, point: str, where: str) -> tuple | None:
        """A named point on a prop or person, in its own coordinates."""
        placed = self.placed[name]
        if placed.person is not None:
            info = placed.person
            h, seat, yaw = info["height"], info["seat"] or 0.46, info["head"]
            sitting = info["pose"] == "sit"
            head_y = seat + 0.406 * h if sitting else 0.906 * h
            points = {
                "face": (math.sin(yaw) * 0.1, head_y, math.cos(yaw) * 0.1),
                "head": (0.0, head_y, 0.0),
                "chest": (0.0, head_y - 0.2 * h, 0.05),
                "hands": (0.0, seat + 0.12 * h, 0.25) if sitting else (0.0, 0.5 * h, 0.12),
                "feet": (0.0, 0.05, 0.4) if sitting else (0.0, 0.05, 0.12),
                "center": (0.0, seat if sitting else h / 2, 0.0),
            }
            points["eyes"] = points["face"]
            point = PERSON_POINT_WORDS.get(point, point)
            if point not in points:
                self.problems.add(where, f'a person has no point "{point}"; use {", ".join(PERSON_POINTS)}')
                return None
            return points[point]
        if point == "center":
            return 0.0, placed.top / 2, 0.0
        if point not in placed.anchors:
            self.problems.add(where, f'"{name}" has no point "{point}"; it has {", ".join(sorted(placed.anchors))}')
            return None
        ax, ay, az, _ = placed.anchors[point]
        return ax, ay, az

    def _world_point(self, name: str, local: tuple) -> list[float]:
        x, y, z, turn = self._at(name, self.now)
        dx, dz = _rotate(local[0], local[2], turn)
        return [x + dx, y + local[1], z + dz]

    def _moves_during(self, name: str) -> bool:
        """Whether a prop or person travels during the shot being built."""
        keys = self.paths.get(name)
        if not keys:
            return False
        start, end = self.now, self.now + self.span
        return any(a[0] < end and b[0] > start and (a[1], a[3]) != (b[1], b[3]) for a, b in zip(keys, keys[1:]))

    def _inside(self, point, skip) -> bool:
        """Whether a camera point lies within any prop or person at any moment of the shot."""
        for k in range(5):
            when = self.now + self.span * k / 4
            for name, placed in self.placed.items():
                if name in skip or placed.kind in ("ground", "room") + FIXTURES:
                    continue
                x, y, z, turn = self._at(name, when)
                lx, lz = _rotate(point[0] - x, point[2] - z, -turn)
                w, _, d = placed.size
                if abs(lx) < w / 2 + 0.25 and abs(lz) < d / 2 + 0.25 and y - 0.1 < point[1] < y + placed.top + 0.4:
                    return True
        return False

    def _sight(self, position, target, skip, aimed, rider) -> bool:
        """Whether the camera sees its target past every other prop and person through the shot."""
        base = self._at(rider, self.now) if rider else None
        for k in range(5):
            when = self.now + self.span * k / 4
            cam, aim_at = list(position), list(target)
            if rider:
                x, y, z, _ = self._at(rider, when)
                shift = (x - base[0], y - base[1], z - base[2])
                cam = [cam[i] + shift[i] for i in range(3)]
                aim_at = [aim_at[i] + shift[i] for i in range(3)]
            if aimed:
                saved, self.now = self.now, when
                aim_at = self._world_point(aimed[0], aimed[1])
                self.now = saved
            for name, placed in self.placed.items():
                if name in skip or placed.kind in ("ground", "room") + FIXTURES:
                    continue
                x, y, z, turn = self._at(name, when)
                w, _, d = placed.size
                for j in range(1, 12):
                    f = 0.08 + 0.84 * j / 12
                    q = [cam[i] + (aim_at[i] - cam[i]) * f for i in range(3)]
                    lx, lz = _rotate(q[0] - x, q[2] - z, -turn)
                    if abs(lx) < w / 2 and abs(lz) < d / 2 and y <= q[1] <= y + placed.top:
                        return False
        return True

    def _unblock(self, position, target, skip, aimed, rider) -> list[float]:
        """The camera swung around its target, least first, until nothing stands in its way."""
        dx, dz = position[0] - target[0], position[2] - target[2]
        for swing in (0.0, 0.5, -0.5, 1.0, -1.0, 1.6, -1.6, 2.3, -2.3, math.pi):
            ox, oz = _rotate(dx, dz, swing)
            moved = self._contain([target[0] + ox, position[1], target[2] + oz])
            if self._sight(moved, target, skip, aimed, rider) and not self._inside(moved, {rider} if rider else set()):
                return moved
        return position

    def _walls(self) -> list:
        """Every wall in the set: its centre, facing, half length, top and turn."""
        out = []
        for name, placed in self.placed.items():
            if placed.kind != "room":
                continue
            w, h, d = placed.size
            for lx, lz, face, half in ((0.0, -d / 2, 0.0, w / 2), (0.0, d / 2, math.pi, w / 2),
                                       (-w / 2, 0.0, math.pi / 2, d / 2), (w / 2, 0.0, -math.pi / 2, d / 2)):
                dx, dz = _rotate(lx, lz, placed.turn)
                out.append((placed.x + dx, placed.z + dz, placed.turn + face, half, placed.y + h))
        return out

    def _contain(self, point) -> list[float]:
        """A camera point kept in front of every wall, below the lowest wall top, and inside the walls' span."""
        walls = self._walls()
        if not walls:
            return point
        x, y, z = point
        margin = 0.35
        for cx, cz, turn, half, _ in walls:
            fx, fz = math.sin(turn), math.cos(turn)
            along = (x - cx) * fz - (z - cz) * fx
            ahead = (x - cx) * fx + (z - cz) * fz
            if abs(along) <= half + 0.5 and ahead < margin:
                x, z = x + fx * (margin - ahead), z + fz * (margin - ahead)
        y = min(y, min(top for *_, top in walls) - 0.3)
        if len(walls) >= 2:
            ends = []
            for cx, cz, turn, half, _ in walls:
                fx, fz = math.sin(turn), math.cos(turn)
                ends += [(cx + fz * half, cz - fx * half), (cx - fz * half, cz + fx * half)]
            xs, zs = [e[0] for e in ends], [e[1] for e in ends]
            if max(xs) - min(xs) > 2.0:
                x = min(max(x, min(xs) + margin), max(xs) - margin)
            if max(zs) - min(zs) > 2.0:
                z = min(max(z, min(zs) + margin), max(zs) - margin)
        return [x, max(0.15, y), z]

    def _clear(self, position, target, skip) -> list[float]:
        """The camera backed away from what it looks at until no prop or person is around it, kept above the floor."""
        point = self._contain([position[0], max(0.15, position[1]), position[2]])
        back = [point[i] - target[i] for i in range(3)]
        length = math.sqrt(sum(v * v for v in back)) or 1.0
        back = [v / length for v in back]
        for _ in range(30):
            if not self._inside(point, skip):
                return point
            point = self._contain([point[0] + back[0] * 0.4, max(0.15, point[1] + back[1] * 0.4),
                                   point[2] + back[2] * 0.4])
        return [point[0], point[1] + 3.0, point[2]]

    def _gaze(self, name: str) -> None:
        """Turn a person, or only their head when seated or set facing, toward what they look at."""
        kind, item, where = self.raw[name]
        placed = self.placed.get(name)
        if kind != "person" or placed is None:
            return
        look = _ref(item, "look_at", where, self.problems)
        if look is None:
            return
        target = self.placed.get(look[0]) or self.resolve(look[0], f"{where}.look_at")
        if target is None or (abs(target.x - placed.x) + abs(target.z - placed.z)) <= 1e-6:
            return
        want = math.atan2(target.x - placed.x, target.z - placed.z)
        if placed.person["pose"] == "sit" or any(k in item for k in ("facing", "turn", "at", "sit_on")):
            delta = (want - placed.turn + math.pi) % (2 * math.pi) - math.pi
            placed.person["head"] = max(-1.3, min(1.3, delta))
        else:
            placed.turn = want

    # -- building
    def build(self, fps: float) -> Scene:
        self._tracked()
        self.read()
        for name in list(self.order):
            kind, item, where = self.raw[name]
            # A person "held_by" a prop is read the other way round: the person holds it.
            key = "holding" if "holding" in item else "held_by" if "held_by" in item else None
            if kind == "person" and key:
                for prop in (item[key] if isinstance(item[key], list) else [item[key]])[:4]:
                    ident = prop.strip().lower() if isinstance(prop, str) else ""
                    if ident in self.raw and self.raw[ident][0] == "prop":
                        self.raw[ident][1].setdefault("held_by", name)
                    else:
                        self.problems.add(f"{where}.{key}", f'no prop "{prop}" to hold')
        for name in self.order:
            self.resolve(name, self.raw[name][2])
        self._separate()
        for name, other, extra in self.facing_later:
            a, b = self.placed.get(name), self.placed.get(other)
            if a is not None and b is not None and (abs(b.x - a.x) + abs(b.z - a.z)) > 1e-6:
                a.turn = math.atan2(b.x - a.x, b.z - a.z) + math.radians(extra or 0.0)
        for name in self.order:
            self._gaze(name)
        floor = self.data.get("floor", True)
        shots_raw = self.data.get("shots", [])
        if not isinstance(shots_raw, list):
            self.problems.add("shots", "needs a list")
            shots_raw = []
        if len(shots_raw) > MAX_SHOTS:
            self.problems.add("shots", f"has {len(shots_raw)} shots; the most is {MAX_SHOTS}")
            shots_raw = shots_raw[:MAX_SHOTS]
        shots_raw = self._measured(shots_raw)
        self.tempo = _number(self.data, "tempo", "scene", self.problems, 40.0, 240.0)
        self.fps = fps
        if self.grid:
            self.tempo = self.song_tempo
        planned = 0.0
        for shot in shots_raw or [{"seconds": 5}]:
            planned += self._length(shot, None, planned) if isinstance(shot, dict) else 3.0
        for name in self.order:
            self._motion(name, min(planned, MAX_SECONDS))
        shots = self._shots(shots_raw)
        look = _text(self.data, "look", "scene", self.problems)
        setting = _text(self.data, "setting", "scene", self.problems)
        self.setting = setting
        self.set_picture = _picture(self.data, "scene", self.problems, "set_picture")
        sound = _text(self.data, "sound", "scene", self.problems)
        lighting = _word(self.data, "lighting", "scene", self.problems, ("day", "evening", "night", "studio"), "day")
        self.figures = _word(self.data, "figures", "scene", self.problems, ("block", "jointed"), "block")
        self.lighting = lighting
        seen_kinds: dict[str, int] = {}
        for name in self.order:
            placed = self.placed[name]
            if placed.person is not None:
                kind = placed.person["gender"]
                shades = PEOPLE_COLOURS[kind]
                placed.person["tint"] = shades[seen_kinds.get(kind, 0) % len(shades)]
                seen_kinds[kind] = seen_kinds.get(kind, 0) + 1
                if placed.colour:
                    placed.person["tint"] = (placed.colour, None)
        self.problems.raise_if_any()

        children = self._lights()
        if floor is not False:
            children.append(create_spec("object", "BlockoutProp", params={
                "shape": "ground", "size": [60.0, 1.0, 60.0], "position": [0.0, 0.0, 0.0], "turn": 0.0,
                "colour": "#a6a6a6", "steps": 1, "checker": 1.0}))
        total = sum(s["seconds"] for s in shots)
        bodies: dict[str, list] = {}
        for name in self.order:
            placed = self.placed[name]
            body = []
            if placed.person is not None:
                info = placed.person
                body.append(create_spec("object", "BlockoutProp", params={
                    "shape": "person", "size": [0.5, info["height"], 0.3], "position": [0.0, 0.0, 0.0],
                    "turn": 0.0, "colour": placed.colour or info["tint"][0], "steps": 1, "pose": info["pose"],
                    "seat": info["seat"] or 0.0, "head": info["head"], "figure": self.figures}))
            for part in [] if placed.kind in FIXTURES else placed.parts:
                rx, ry, rz = (math.radians(v) for v in part["rotate"])
                body.append(create_spec("object", "BlockoutProp", params={
                    "shape": part["shape"], "size": [float(v) for v in part["size"]],
                    "position": [float(v) for v in part["at"]], "turn": 0.0, "rotate": [rx, ry, rz],
                    "pivot": "center", "colour": _grey(placed.colour or part["tone"]),
                    "steps": int(part.get("steps", 6))}))
            bodies[name] = body
        for name, holder in self.held.items():
            local = self._local_point(holder, "hands", "held_by")
            prop = self.placed[name]
            bodies[holder].append(create_spec("object", "BlockoutBody", params={
                "name": f"blockout:{name}", "position": [local[0], local[1] - prop.top / 2, local[2] + 0.05],
                "turn": 0.0}, children=bodies[name]))
        for name in self.order:
            if name in self.held:
                continue
            placed = self.placed[name]
            params = {"name": f"blockout:{name}", "position": [placed.x, placed.y, placed.z], "turn": placed.turn}
            keys = self.paths.get(name)
            if keys:
                params["path"] = [{"t": k[0] / total, "p": [k[1], k[2], k[3]], "turn": k[4]} for k in keys]
                params["walk"] = placed.person is not None and placed.person["pose"] != "sit"
                params["ease"] = placed.person is None
                params["seconds"] = total
            children.append(create_spec("object", "BlockoutBody", params=params, children=bodies[name]))
        objects = create_spec("object", "Group", params={"name": "Blockout scene", "visible": True,
                                                         "position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0],
                                                         "scale": [1.0, 1.0, 1.0]},
                              children=children, meta={"blockout": True})
        cuts, rows, start = [], [], 0.0
        for index, s in enumerate(shots):
            cut = {"start": start / total, "end": (start + s["seconds"]) / total, "position": s["position"],
                   "target": s["target"], "fov": s["fov"], "moves": s["moves"]}
            for key in ("aim", "ride", "path", "shake"):
                if s.get(key):
                    cut[key] = s[key]
            cut["seconds"] = s["seconds"]
            cuts.append(cut)
            rows.append(f"{index + 1}. {start:.1f}-{start + s['seconds']:.1f} s  {s['label']}"
                        + (f", {s['move']}" if s["move"] != "static" else ""))
            start += s["seconds"]
        first = shots[0]
        camera = create_spec("camera", "PerspectiveCamera", params={
            "fov": first["fov"], "near": 0.05, "far": 1000.0, "position": first["position"],
            "target": first["target"], "shots": cuts})
        description = self._prompt(look, setting, shots, total)
        music = _text(self.data, "music", "scene", self.problems)
        people = sum(1 for n in self.order if self.placed[n].person is not None)
        summary = self._summary(setting, shots, total)
        self._label(shots)
        self.labelled = True
        labelled_description = self._prompt(look, setting, shots, total)
        labelled_summary = self._summary(setting, shots, total)
        self.labelled = False
        heard = [f"{words} at {when:.2f}s" for shot in shots for when, words in shot.get("sounds", [])]
        if heard:
            sound = (f"{sound.rstrip('.')}. " if sound else "") + "Timed sounds: " + "; ".join(heard) + "."
        prompt = self._ref2va(labelled_summary, labelled_description, sound, music)
        return Scene(objects, camera, summary, description, sound, prompt, total, rows,
                     {"props": len(self.order) - people, "people": people, "shapes": self.parts_total + people, "shots": len(shots)})

    def _lights(self) -> list[dict]:
        """Soft, even white light: a high key light with soft shadows and a strong sky fill."""

        def light(kind, power, pos, shadow):
            return create_spec("object", "Light", params={
                "lightType": kind, "color": "#ffffff", "groundColor": "#9a9a9a", "intensity": power, "position": pos,
                "distance": 0.0, "decay": 2.0, "angle": math.radians(45.0), "penumbra": 0.0, "castShadow": shadow})

        return [light("directional", 1.8, [-6.0, 14.0, 8.0], True),
                light("hemisphere", 1.4, [0.0, 10.0, 0.0], False)]

    # -- camera
    def _points(self, name: str) -> tuple[list[float], list[float]] | None:
        """A thing's head (or top centre) and chest (or middle) in the world."""
        placed = self.placed.get(name)
        if placed is None:
            return None
        if placed.person is not None:
            info = placed.person
            h = info["height"]
            head_y = placed.y + ((info["seat"] or 0.46) + 0.406 * h if info["pose"] == "sit" else 0.906 * h)
            return [placed.x, head_y, placed.z], [placed.x, head_y - 0.2 * h, placed.z]
        top = placed.y + placed.top
        return [placed.x, top, placed.z], [placed.x, placed.y + placed.top / 2, placed.z]

    def _shots(self, raw: list) -> list[dict]:
        p = self.problems
        if not raw:
            raw = [{"rig": "wide", "seconds": 5}]
        shots, total = [], 0.0
        for i, shot in enumerate(raw):
            where = f"shots[{i}]"
            if not isinstance(shot, dict):
                p.add(where, "needs an object")
                continue
            _keys(shot, SHOT_KEYS, where, p)
            seconds = self._length(shot, where, total)
            if i == len(raw) - 1 and self.fps:
                seconds = _h3_length(total, seconds, self.fps)
            self.now = total
            self.span = seconds
            self.written = self._written(shot)
            total += seconds
            if total > MAX_SECONDS:
                p.add(where, f"the shots run past {MAX_SECONDS:.0f} seconds")
                break
            saved = {}
            for name in self.paths:
                placed = self.placed[name]
                saved[name] = (placed.x, placed.y, placed.z, placed.turn)
                placed.x, placed.y, placed.z, placed.turn = self._at(name, self.now)
            try:
                built = self._rig(shot, where)
            finally:
                for name, pose in saved.items():
                    placed = self.placed[name]
                    placed.x, placed.y, placed.z, placed.turn = pose
            if built is not None:
                built["seconds"] = seconds
                built["action"] = _text(shot, "action", where, p)
                built["dialogue"] = self._dialogue(shot, where)
                built["sounds"] = self._sounds(shot, where, total - seconds, seconds)
                shots.append(built)
        if not shots and not p.items:
            p.add("shots", "no shot could be built")
        return shots

    def _length(self, shot: dict, where: str | None = None, start: float = 0.0) -> float:
        """A shot's seconds: its beats at the tempo when both are given, else its seconds, else 3."""
        problems = self.problems if where else _Problems()
        grid = getattr(self, "grid", None)
        if grid:
            if "beats" in shot:
                beats = _number(shot, "beats", where or "shot", problems, 0.25, 64.0)
                if beats is not None:
                    base = 0.0 if start <= self.grid[0] else round(self._beat_index(start) * 4) / 4
                    end = self._beat_time(base + beats)
                    return max(0.5, min(20.0, end - start))
            seconds = _number(shot, "seconds", where or "shot", problems, 0.5, 20.0, 3.0) or 3.0
            index = round(self._beat_index(start + seconds))
            end = self._beat_time(index)
            while end - start < 0.5:
                index += 1
                end = self._beat_time(index)
            return min(20.0, end - start)
        if "beats" in shot and getattr(self, "tempo", None):
            beats = _number(shot, "beats", where or "shot", problems, 0.25, 64.0)
            if beats is not None:
                return max(0.5, min(20.0, beats * 60.0 / self.tempo))
        elif "beats" in shot and where:
            problems.add(f"{where}.beats", 'needs "tempo" at the top of the scene, in beats per minute')
        return _number(shot, "seconds", where or "shot", problems, 0.5, 20.0, 3.0) or 3.0

    def _written(self, shot: dict) -> float:
        """A shot's length as its JSON reads with no song: beats at `tempo`, or at 120 when none is given."""
        grid, self.grid = self.grid, None
        tempo = self.tempo
        if grid and "beats" in shot:
            self.tempo = _number(self.data, "tempo", "scene", _Problems(), 40.0, 240.0) or 120.0
        try:
            return self._length(shot)
        finally:
            self.grid, self.tempo = grid, tempo

    def _beat_index(self, seconds: float) -> float:
        """Where a time falls on the song's beats, as a fractional beat number."""
        grid, period = self.grid, 60.0 / self.tempo
        if seconds <= grid[0]:
            return (seconds - grid[0]) / period
        if seconds >= grid[-1]:
            return len(grid) - 1 + (seconds - grid[-1]) / period
        k = bisect.bisect_right(grid, seconds) - 1
        return k + (seconds - grid[k]) / max(1e-6, grid[k + 1] - grid[k])

    def _beat_time(self, index: float) -> float:
        """The time of a fractional beat number, carried on at the tempo past either end of the song."""
        grid, period = self.grid, 60.0 / self.tempo
        if index <= 0:
            return grid[0] + index * period
        if index >= len(grid) - 1:
            return grid[-1] + (index - len(grid) + 1) * period
        k = int(index)
        return grid[k] + (index - k) * (grid[k + 1] - grid[k])

    def _on_beat(self, seconds: float) -> str:
        """`on the downbeat`, `on the beat` or nothing, for a cut time."""
        if not getattr(self, "grid", None):
            return ""
        near = 0.6 / self.fps
        if any(abs(seconds - t) <= near for t in self.downbeats):
            return "on the downbeat"
        if any(abs(seconds - t) <= near for t in self.grid):
            return "on the beat"
        return ""

    def _dialogue(self, shot, where) -> list[tuple[str, str]]:
        """A shot's spoken lines as (person, words), in order."""
        raw = shot.get("dialogue")
        if raw is None:
            return []
        p = self.problems
        if isinstance(raw, dict):
            raw = [{"who": k, "line": v} for k, v in raw.items()]
        if not isinstance(raw, list) or len(raw) > MAX_LINES:
            p.add(f"{where}.dialogue", f'needs a list of up to {MAX_LINES} lines, each {{"who": id, "line": "..."}}')
            return []
        lines = []
        for i, item in enumerate(raw):
            at = f"{where}.dialogue[{i}]"
            if not isinstance(item, dict):
                p.add(at, 'needs {"who": id, "line": "..."}')
                continue
            _keys(item, {"who", "line"}, at, p)
            who = item.get("who")
            who = who.strip().lower() if isinstance(who, str) else ""
            if who not in self.placed or self.placed[who].person is None:
                p.add(f"{at}.who", f'no person "{item.get("who")}"')
                continue
            words = _text(item, "line", at, p).replace("<", "").replace(">", "").strip()
            if not words:
                p.add(f"{at}.line", "needs the words spoken")
                continue
            if words[-1] not in ".?!":
                words += "."
            lines.append((who, words))
        return lines

    def _tracked(self) -> None:
        """Every prop and person written with a `"track"` placed, and people turned and moved, as measured."""
        props = self.data.get("props")
        things = {str(t.get("track", "")).strip().lower(): t for t in getattr(self, "object_tracks", [])}
        for i, item in enumerate(props if isinstance(props, list) else []):
            if not isinstance(item, dict):
                continue
            if "track" not in item and str(item.get("id", "")).strip().lower() in things:
                item = dict(item, track=item["id"])
            if "track" not in item:
                continue
            item = dict(item)
            asked = str(item.pop("track") or "").strip()
            props[i] = item
            thing = things.get(asked.lower())
            if thing is None:
                held = ", ".join(sorted(t.upper() for t in things)) or "nothing, so wire Video Blueprint's blueprint in"
                self.problems.add(f"props[{i}].track", f'no measured object "{asked}"; the blueprint holds {held}')
                continue
            for key in ("at", "on", "beside"):
                item.pop(key, None)
            item["position"] = [thing["position"][0], thing["position"][2]]
        people = self.data.get("people")
        if not isinstance(people, list):
            return
        known = {str(t.get("track", "")).strip().lower(): t for t in getattr(self, "people_tracks", [])}
        self.tracks_used: set = set()
        for i, item in enumerate(people):
            if not isinstance(item, dict):
                continue
            face = item.get("facing")
            if isinstance(face, str) and face.strip().lower() in FACING_WORDS:
                item = people[i] = {k: v for k, v in item.items() if k != "facing"}
            elif isinstance(face, (int, float)) and not isinstance(face, bool):
                item = people[i] = {k: v for k, v in item.items() if k != "facing"}
                item.setdefault("turn", face)
            elif (isinstance(face, list) and len(face) in (2, 3)
                  and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in face)):
                item = people[i] = {k: v for k, v in item.items() if k != "facing"}
                dx, dz = face[0], face[-1]
                if abs(dx) + abs(dz) > 1e-9:
                    item.setdefault("turn", round(math.degrees(math.atan2(dx, dz))))
            if "track" not in item and str(item.get("id", "")).strip().lower() in known:
                item = people[i] = dict(item, track=item["id"])
            if "track" not in item:
                continue
            item = dict(item)
            asked = str(item.pop("track") or "").strip()
            people[i] = item
            track = known.get(asked.lower())
            if track is None:
                # A name the measurement does not hold takes the next measured person nobody claimed.
                claimed = {str(p.get("track", p.get("id", ""))).strip().lower() for p in people if isinstance(p, dict)}
                spare = [t for name, t in known.items() if name not in claimed and name not in self.tracks_used]
                track = spare[0] if spare else None
            if track is not None:
                self.tracks_used.add(str(track.get("track", "")).strip().lower())
            if track is None:
                held = ", ".join(sorted(t.upper() for t in known)) or "nobody, so wire Video Blueprint's blueprint in"
                self.problems.add(f"people[{i}].track", f'no measured person "{asked}"; the blueprint holds {held}')
                continue
            path = [pt for pt in track.get("path", []) if isinstance(pt, dict) and "position" in pt and "at" in pt]
            if not path:
                continue
            first = path[0]
            named = {str(x.get("id", "")).strip().lower() for key in ("props", "people")
                     for x in (self.data.get(key) or []) if isinstance(x, dict)}
            face = item.get("facing")
            if isinstance(face, str) and face.strip().lower().split(".")[0] not in named:
                item.pop("facing")
            item["position"] = [first["position"][0], first["position"][2]]
            turn = next((pt["turn"] for pt in path if pt.get("turn") is not None), None)
            if turn is not None and "facing" not in item and "turn" not in item:
                item["turn"] = turn
            steps, anchor = [], first
            for pt in path[1:]:
                if math.dist(pt["position"], anchor["position"]) >= TRACK_STEP:
                    steps.append({"to": [pt["position"][0], pt["position"][2]], "start": anchor["at"], "end": pt["at"]})
                    anchor = pt
            if len(steps) > MAX_STEPS:
                kept = [steps[round(k * (len(steps) - 1) / (MAX_STEPS - 1))] for k in range(MAX_STEPS)]
                steps = [dict(step, start=(kept[k - 1]["end"] if k else steps[0]["start"])) for k, step in enumerate(kept)]
            if steps:
                item["motion"] = steps
                if "pose" not in item:
                    item["pose"] = "run" if float(track.get("speed", 0.0)) > RUN_SPEED else "walk"

    def _measured(self, shots_raw: list) -> list:
        """The shots with every `"rig": "measured"` given the camera Video Blueprint measured for it."""
        measured = getattr(self, "blueprint", None) or []
        out = []
        for i, shot in enumerate(shots_raw):
            if not isinstance(shot, dict) or str(shot.get("rig", "")).strip().lower() != "measured":
                out.append(shot)
                continue
            if i >= len(measured):
                self.problems.add(f"shots[{i}].rig", f'"measured" needs shot {i + 1} in a blueprint from Video '
                                                     f'Blueprint, and it holds {len(measured)}')
                out.append(shot)
                continue
            taken = measured[i]
            shot = {k: v for k, v in shot.items() if k not in ("rig", "keys", "position", "target", "move")}
            shot.update(rig="custom", keys=taken["keys"], seconds=taken["seconds"])
            shot.pop("beats", None)
            if taken.get("handheld"):
                shot.setdefault("shake", "handheld")
            out.append(shot)
        return out

    def _sounds(self, shot, where, start: float, seconds: float) -> list[tuple[float, str]]:
        """A shot's timed sounds as (seconds into the scene, what is heard), in time order."""
        raw = shot.get("sounds")
        if raw is None:
            return []
        p = self.problems
        if isinstance(raw, str):
            raw = [{"at": 0, "sound": raw}]
        if not isinstance(raw, list) or len(raw) > MAX_SOUNDS:
            p.add(f"{where}.sounds", f'needs a list of up to {MAX_SOUNDS} sounds, each {{"at": seconds, "sound": "..."}}')
            return []
        written = getattr(self, "written", seconds) or seconds
        heard = []
        for i, item in enumerate(raw):
            at = f"{where}.sounds[{i}]"
            if isinstance(item, str):
                item = {"at": 0, "sound": item}
            if not isinstance(item, dict):
                p.add(at, 'needs {"at": seconds, "sound": "..."}')
                continue
            _keys(item, {"at", "sound"}, at, p)
            given = item.get("at")
            if (isinstance(given, (int, float)) and not isinstance(given, bool) and given > written
                    and start - 1e-6 <= given <= start + seconds + 1e-6):
                item = dict(item, at=min(written, given - start))
            when = (_number(item, "at", at, p, 0.0, written, 0.0) or 0.0) * seconds / written
            words = _text(item, "sound", at, p).replace("<", "").replace(">", "").strip().rstrip(".")
            if not words:
                p.add(f"{at}.sound", "needs what is heard")
                continue
            heard.append((start + when, words))
        return sorted(heard)

    def _ids(self, shot, key, where) -> list[str]:
        value = shot.get(key)
        if value is None:
            return []
        values = value if isinstance(value, list) else [value]
        values = [v.split(".")[0] if isinstance(v, str) else v for v in values]
        out = []
        for v in values[:8]:
            if not isinstance(v, str) or v.strip().lower() not in self.placed:
                self.problems.add(f"{where}.{key}", f'no prop or person "{v}"; there is {", ".join(self.order[:30])}')
                continue
            out.append(v.strip().lower())
        return out

    def _rig(self, shot, where) -> dict | None:
        p = self.problems
        said = shot.get("rig")
        said = re.sub(r"\s+", " ", said.strip().lower().replace("-", " ").replace("_", " ")) if isinstance(said, str) else ""
        rig, chase = RIG_ALIASES.get(said, (None, False))
        if rig is None:
            rig = _word(shot, "rig", where, p, RIGS, "wide")
        lens = _number(shot, "lens", where, p, 12.0, 300.0)
        angle = math.radians(_number(shot, "angle", where, p, -180.0, 180.0, 0.0) or 0.0)
        side = _word(shot, "side", where, p, ("front", "back"), "front")
        subjects = self._ids(shot, "subjects", where)
        onto = self._ids(shot, "onto", where)
        source = self._ids(shot, "from", where)
        everything = [n for n in self.order if n in self.placed]
        label = rig
        sign = 1.0 if side == "front" else -1.0

        def fit(points, fov_deg, margin=1.2):
            cx = sum(q[0] for q in points) / len(points)
            cy = sum(q[1] for q in points) / len(points)
            cz = sum(q[2] for q in points) / len(points)
            radius = max(0.3, max(math.dist((cx, cy, cz), q) for q in points))
            half = math.radians(min(fov_deg, fov_deg * 16 / 9) / 2)
            return [cx, cy, cz], radius * margin / math.tan(half)

        def heads(names):
            return [self._points(n)[0] for n in names if self._points(n)]

        def bounds(names):
            pts = []
            for n in names:
                placed = self.placed[n]
                hw, hd = placed.size[0] / 2, placed.size[2] / 2
                for sx in (-1, 1):
                    for sz in (-1, 1):
                        dx, dz = _rotate(sx * hw, sz * hd, placed.turn)
                        pts.append([placed.x + dx, placed.y + placed.top, placed.z + dz])
                        pts.append([placed.x + dx, placed.y, placed.z + dz])
            return pts

        def mm(fov):
            return 12.0 / math.tan(math.radians(fov) / 2)

        def lens_fov(default_mm):
            return math.degrees(2 * math.atan(12.0 / (lens or default_mm)))

        def perpendicular(a, b):
            ax, az = b[0] - a[0], b[2] - a[2]
            n = math.hypot(ax, az) or 1.0
            px_, pz_ = -az / n, ax / n
            if px_ * 0.0 + pz_ * 1.0 < 0 or (abs(pz_) < 1e-6 and px_ < 0):
                px_, pz_ = -px_, -pz_
            return px_ * sign, pz_ * sign

        def facing_of(name):
            placed = self.placed[name]
            turn = placed.turn + (placed.person["head"] if placed.person else 0.0)
            return math.sin(turn), math.cos(turn)

        position = target = None
        path_keys = None
        if rig in ("wide", "establishing", "top down"):
            names = subjects or [n for n in everything if self.placed[n].kind not in ("wall", "wall_with_window",
                                                                                    "wall_with_door")] or everything
            if not names:
                p.add(where, "a wide shot needs something in the scene")
                return None
            fov = lens_fov(18.0 if rig == "establishing" else 24.0 if rig == "wide" else 28.0)
            centre, distance = fit(bounds(names), fov, 1.05)
            if rig == "top down":
                position, target = [centre[0], centre[1] + distance, centre[2] + 0.01], centre
            else:
                lift = 0.45 if rig == "establishing" else 0.22
                dx, dz = math.sin(angle) * sign, math.cos(angle) * sign
                position = [centre[0] + dx * distance, max(1.6, centre[1] + distance * lift), centre[2] + dz * distance]
                target = centre
            label = f"{rig} shot"
        elif rig == "two shot":
            names = subjects or [n for n in everything if self.placed[n].person][:2]
            if len(names) < 2:
                p.add(where, 'a two shot needs two "subjects"')
                return None
            fov = lens_fov(35.0)
            pts = heads(names)
            centre, distance = fit(pts, fov, 1.6)
            px_, pz_ = perpendicular(pts[0], pts[1])
            c, s = math.cos(angle), math.sin(angle)
            px_, pz_ = px_ * c + pz_ * s, -px_ * s + pz_ * c
            position = [centre[0] + px_ * distance, centre[1] + 0.05, centre[2] + pz_ * distance]
            target = [centre[0], centre[1] - 0.12, centre[2]]
            label = "two shot of " + " and ".join(self._name(n) for n in names)
        elif rig == "over the shoulder":
            if not source or not onto:
                p.add(where, 'over the shoulder needs "from" (whose shoulder) and "onto" (who is seen)')
                return None
            fov = lens_fov(40.0)
            a, b = self._points(source[0])[0], self._points(onto[0])[0]
            ax, az = b[0] - a[0], b[2] - a[2]
            n = math.hypot(ax, az) or 1.0
            ux, uz = ax / n, az / n
            first = sorted([source[0], onto[0]])
            px_, pz_ = perpendicular(self._points(first[0])[0], self._points(first[1])[0])
            position = [a[0] - ux * 0.9 + px_ * 0.55, a[1] + 0.06, a[2] - uz * 0.9 + pz_ * 0.55]
            target = [b[0], b[1] - 0.08, b[2]]
            label = f"over {self._name(source[0])}'s shoulder onto {self._name(onto[0])}"
        elif rig in ("medium", "close up", "insert"):
            names = onto or subjects or [n for n in everything if self.placed[n].person][:1]
            if not names:
                p.add(where, f'a {rig} needs "onto"')
                return None
            name = names[0]
            head, chest = self._points(name)
            placed = self.placed[name]
            if rig == "insert" or placed.person is None:
                fov = lens_fov(50.0)
                radius = max(0.12, max(placed.size) / 2)
                distance = max(0.45, 2.6 * radius)
                dx, dz = math.sin(angle) * sign, math.cos(angle) * sign
                position = [head[0] + dx * distance * 0.8, head[1] + distance * 0.6, head[2] + dz * distance * 0.8]
                target = head
                label = (f"insert of {self._name(name)}" if placed.person is not None
                         else f"insert of the {self._name(name).lower()}")
            else:
                fov = lens_fov(55.0 if rig == "close up" else 40.0)
                fx, fz = facing_of(name)
                c, s = math.cos(angle + 0.35 * sign), math.sin(angle + 0.35 * sign)
                fx, fz = fx * c + fz * s, -fx * s + fz * c
                distance = 0.95 if rig == "close up" else 1.7
                point = head if rig == "close up" else chest
                position = [point[0] + fx * distance, point[1] + 0.05, point[2] + fz * distance]
                target = [point[0], point[1] + (0.0 if rig == "close up" else 0.1), point[2]]
                label = f"{rig} of {self._name(name)}"
        elif rig in ("fly by", "fly through", "fly over", "reveal", "chase", "orbit"):
            built = self._dynamic(rig, shot, where, onto or subjects, angle, sign, lens_fov)
            if built is None:
                return None
            position, target, fov, label, path_keys, rig_follow = built
            if rig_follow and not self._ids(shot, "follow", where):
                shot = {**shot, "follow": rig_follow}
        elif rig == "pov":
            names = source or subjects or [n for n in everything if self.placed[n].person][:1]
            if not names or self.placed[names[0]].person is None:
                p.add(where, 'a pov needs "from": the person whose eyes it sees through')
                return None
            fov = lens_fov(28.0)
            eye = self._points(names[0])[0]
            fx, fz = facing_of(names[0])
            position = [eye[0] + fx * 0.12, eye[1], eye[2] + fz * 0.12]
            target = self._points(onto[0])[0] if onto else [eye[0] + fx * 5.0, eye[1] - 0.3, eye[2] + fz * 5.0]
            label = f"{self._name(names[0])}'s point of view" + (f" onto {self._name(onto[0])}" if onto else "")
        else:
            fov = lens_fov(35.0)
            position = _vector(shot, "position", where, p, (3,))
            target = _vector(shot, "target", where, p, (3,))
            if target is None and onto:
                target = self._points(onto[0])[0]
            if (position is None or target is None) and "keys" in shot:
                position, target = [0.0, 1.6, 5.0], [0.0, 1.0, 0.0]
            if position is None or target is None:
                p.add(where, 'a custom shot needs "position" [x, y, z] and "target" [x, y, z] or "onto"')
                return None
            label = "moving shot" if "keys" in shot else "shot"
        if rig != "custom" and "position" in shot and "target" in shot:
            given_position = _vector(shot, "position", where, p, (3,))
            given_target = _vector(shot, "target", where, p, (3,))
            if given_position is not None and given_target is not None:
                position, target = given_position, given_target
        aim = None
        defaults = {"close up": (onto or subjects, "face"), "medium": (onto or subjects, "chest"),
                    "over the shoulder": (onto, "face"), "insert": (onto or subjects, "top")}
        if rig in defaults and defaults[rig][0]:
            name = defaults[rig][0][0]
            point = defaults[rig][1] if self.placed[name].person is not None else "top"
            aim = (name, point)
        look = _ref(shot, "look_at", where, p)
        if look is not None:
            if look[0] not in self.placed:
                p.add(f"{where}.look_at", f'no prop or person "{look[0]}"; there is {", ".join(self.order[:30])}')
            else:
                aim = (look[0], look[1] or ("face" if self.placed[look[0]].person is not None else "center"))
        aim_spec = None
        local = None
        if aim is not None:
            local = self._local_point(aim[0], aim[1], f"{where}.look_at")
            if local is not None:
                if look is not None:
                    target = self._world_point(aim[0], local)
                aim_spec = {"name": f"blockout:{aim[0]}", "offset": [float(v) for v in local]}
                if look is not None:
                    label += f", looking at {self._name(aim[0])}'s {aim[1]}" if self.placed[aim[0]].person else ""
        ride = None
        follow = self._ids(shot, "follow", where)
        said = shot.get("move", "")
        said = said.strip().lower() if isinstance(said, str) else ""
        if not follow and MOVE_ALIASES.get(said, "x") is None and (onto or subjects):
            follow = (onto or subjects)[:1]
        if not follow and chase:
            follow = (onto or subjects or [n for n in everything if self.placed[n].person])[:1]
        if rig == "pov" and not follow:
            names = source or subjects or [n for n in everything if self.placed[n].person][:1]
            follow = names[:1] if names and self._moves_during(names[0]) else []
        if not follow and rig in ("medium", "close up") and aim is not None and self._moves_during(aim[0]):
            follow = [aim[0]]
        if follow:
            x, y, z, _ = self._at(follow[0], self.now)
            ride = {"name": f"blockout:{follow[0]}", "origin": [x, y, z]}
            if rig not in ("chase", "orbit", "fly by", "fly over", "reveal"):
                label += f", following {self._name(follow[0])}"
        rider = follow[0] if follow else None
        aimed = (aim[0], local) if aim is not None and local is not None else None
        given = self._keys_path(shot, where, fov)
        if given and rig == "custom" and _still(given):
            position, target, fov = given[0]["p"], given[0]["target"], given[0].get("fov", fov)
            label = "locked-off shot"
            given = None
        if given:
            path_keys = given
        if path_keys:
            position, target = path_keys[0]["p"], path_keys[0]["target"]
        elif rig not in ("over the shoulder", "pov"):
            skip = set(onto) | set(source) | set(subjects) | ({aim[0]} if aim else set()) | ({rider} if rider else set())
            position = self._unblock(position, target, skip, aimed, rider)
        if not path_keys:
            own = set(source or subjects or []) if rig == "pov" else set()
            position = self._clear(position, target, ({rider} if rider else set()) | own)
        move = "static"
        raw_move = shot.get("move")
        if isinstance(raw_move, str):
            said = re.sub(r"\s+", " ", raw_move.strip().lower().replace("-", " ").replace("_", " "))
            if said in MOVE_ALIASES:
                move = MOVE_ALIASES[said] or "static"
            else:
                move = _word(shot, "move", where, p, tuple(MOVES), "static")
        elif raw_move is not None:
            move = _word(shot, "move", where, p, tuple(MOVES), "static")
        moves = self._move(shot, where, move, position, target, fov)
        ease = _word(shot, "ease", where, p, EASES, "ease in and out")
        for m in moves:
            m["ease"] = ease
        shake = shot.get("shake")
        shake_spec = None
        if isinstance(shake, (int, float)) and not isinstance(shake, bool):
            amount = _number(shot, "shake", where, p, 0.0, 1.0, 0.0)
            shake_spec = {"style": "handheld", "move": 0.06 * amount, "turn": 1.2 * amount} if amount else None
        elif shake is not None:
            style = _word(shot, "shake", where, p, tuple(SHAKES), "none")
            if style != "none":
                shake_spec = {"style": style, "move": SHAKES[style][0], "turn": SHAKES[style][1]}
        if path_keys and aim_spec is not None and look is None and rig not in ("orbit",):
            aim_spec = aim_spec if rig in ("fly by", "fly over", "reveal") else None
        return {"position": [float(v) for v in position], "target": [float(v) for v in target], "fov": float(fov),
                "moves": moves, "move": move, "label": label, "rig": rig, "aim": aim_spec, "ride": ride,
                "path": path_keys, "shake": shake_spec}

    def _ref_point(self, value, where, height=None) -> list[float] | None:
        """A world point from [x, y, z], [x, z] or an id or id.point, at the shot's start."""
        if isinstance(value, list):
            point = _vector({"v": value}, "v", where, self.problems)
            if point is None:
                return None
            return [point[0], 1.6 if height is None else height, point[1]] if len(point) == 2 else point
        ref = _ref({"v": value}, "v", where, self.problems)
        if ref is None:
            return None
        if ref[0] not in self.placed:
            self.problems.add(where, f'no prop or person "{ref[0]}"; there is {", ".join(self.order[:30])}')
            return None
        placed = self.placed[ref[0]]
        point = ref[1] or ("face" if placed.person is not None else "center")
        local = self._local_point(ref[0], point, where)
        if local is None:
            return None
        world = self._world_point(ref[0], local)
        if height is not None and not ref[1]:
            world[1] = height
        return world

    def _keys_path(self, shot, where, fov) -> list[dict] | None:
        """A shot's own camera keys: where the camera is and looks at moments through the shot."""
        raw = shot.get("keys")
        if raw is None:
            return None
        p = self.problems
        if not isinstance(raw, list) or len(raw) < 2 or len(raw) > MAX_KEYS:
            p.add(f"{where}.keys", f'needs 2 to {MAX_KEYS} keys, each {{"at": seconds, "position": ..., "look_at": ...}}')
            return None
        seconds = self.span
        written = getattr(self, "written", seconds) or seconds
        keys, last = [], -1.0
        for i, key in enumerate(raw):
            at = f"{where}.keys[{i}]"
            if not isinstance(key, dict):
                p.add(at, "needs an object")
                return None
            _keys(key, {"at", "position", "look_at", "target", "lens", "roll", "ease", "height"}, at, p)
            when = (_number(key, "at", at, p, 0.0, written, written * i / (len(raw) - 1)) or 0.0) * seconds / written
            if when < last:
                p.add(f"{at}.at", "keys run in time order")
                return None
            last = when
            height = _number(key, "height", at, p, 0.05, 200.0)
            position = self._ref_point(key.get("position"), f"{at}.position", height) if "position" in key else None
            look = key.get("look_at", key.get("target"))
            target = self._ref_point(look, f"{at}.look_at") if look is not None else None
            if position is None or target is None:
                p.add(at, 'needs "position" and "look_at"')
                return None
            lens = _number(key, "lens", at, p, 12.0, 300.0)
            keys.append({"t": when / seconds, "p": [position[0], max(0.1, position[1]), position[2]], "target": target,
                         "fov": math.degrees(2 * math.atan(12.0 / lens)) if lens else fov,
                         "roll": math.radians(_number(key, "roll", at, p, -90.0, 90.0, 0.0) or 0.0),
                         "ease": _word(key, "ease", at, p, EASES, "ease in and out")})
        return keys

    def _dynamic(self, rig, shot, where, names, angle, sign, lens_fov):
        """A moving rig's path, tried at other angles and then smaller until nothing stands on it."""
        p = self.problems
        through = self._ids(shot, "through", where) if rig == "fly through" else []
        subject = (through or names or [n for n in self.order if n in self.placed and self.placed[n].person]
                   or [n for n in self.order if n in self.placed])[:1]
        if not subject:
            p.add(where, f'a {rig} needs "onto" or "through"')
            return None
        name = subject[0]
        rider = name if self._moves_during(name) or rig == "chase" else None
        first = roomy = None
        for swing in (0.0, 90.0, -90.0, 180.0, 45.0, -45.0, 135.0, -135.0):
            for scale in (1.0, 0.7, 0.45):
                built = self._dynamic_try(rig, shot, where, name, angle + math.radians(swing), sign, lens_fov, scale)
                if built is None:
                    return None
                squeezed = False
                if built[4]:
                    for key in built[4]:
                        held = self._contain(key["p"])
                        squeezed = squeezed or math.dist(held, key["p"]) > 0.3
                        key["p"] = held
                    built = (built[4][0]["p"],) + tuple(built[1:])
                first = first or built
                if rig == "fly through":
                    return built
                if squeezed:
                    continue
                roomy = roomy or built
                if self._path_clear(built[4] or [], {name}, rider):
                    return built
        # Nothing both clear and unsqueezed: an unsqueezed path, else the first tried.
        return roomy or first

    def _path_clear(self, keys, skip, rider) -> bool:
        """Whether a camera path stays out of every prop and person, ridden along with a moving subject."""
        base = self._at(rider, self.now) if rider else None
        for a, b in zip(keys, keys[1:]):
            for k in range(6):
                f = k / 5
                point = [a["p"][i] + (b["p"][i] - a["p"][i]) * f for i in range(3)]
                if rider:
                    x, y, z, _ = self._at(rider, self.now + self.span * (a["t"] + (b["t"] - a["t"]) * f))
                    point = [point[0] + x - base[0], point[1] + y - base[1], point[2] + z - base[2]]
                if self._inside(point, {rider} if rider else set()):
                    return False
                aim_at = [a["target"][i] + (b["target"][i] - a["target"][i]) * f for i in range(3)]
                if rider:
                    aim_at = [aim_at[0] + x - base[0], aim_at[1] + y - base[1], aim_at[2] + z - base[2]]
                if not self._line_free(point, aim_at, skip, self.now + self.span * (a["t"] + (b["t"] - a["t"]) * f)):
                    return False
        return True

    def _line_free(self, cam, aim_at, skip, when) -> bool:
        """Whether nothing but what is skipped stands between a camera and the point it looks at."""
        for name, placed in self.placed.items():
            if name in skip or placed.kind in ("ground", "room") + FIXTURES:
                continue
            x, y, z, turn = self._at(name, when)
            w, _, d = placed.size
            for j in range(1, 12):
                f = 0.08 + 0.84 * j / 12
                q = [cam[i] + (aim_at[i] - cam[i]) * f for i in range(3)]
                lx, lz = _rotate(q[0] - x, q[2] - z, -turn)
                if abs(lx) < w / 2 and abs(lz) < d / 2 and y <= q[1] <= y + placed.top:
                    return False
        return True

    def _dynamic_try(self, rig, shot, where, name, angle, sign, lens_fov, scale):
        p = self.problems
        placed = self.placed[name]
        x, y, z, turn = self._at(name, self.now)
        if placed.person is not None:
            centre = self._world_point(name, self._local_point(name, "chest", where))
        else:
            centre = [x, y + placed.top / 2, z]
        reach = max(placed.size[0], placed.size[2], 1.0) * scale
        ux, uz = math.sin(angle) * sign, math.cos(angle) * sign
        sx, sz = uz, -ux
        fov = lens_fov(28.0)
        ease = _word(shot, "ease", where, p, EASES, "linear" if rig in ("fly by", "chase") else "ease in and out")

        def key(t, pos, tgt, f=fov, roll=0.0, e=ease):
            return {"t": t, "p": [pos[0], max(0.15, pos[1]), pos[2]], "target": list(tgt), "fov": f, "roll": roll,
                    "ease": e}

        follow = None
        if rig == "fly by":
            side, near = (3.0 + reach) * scale, (1.2 + reach * 0.6) * scale
            h = centre[1] + 0.2
            keys = [key(0.0, [centre[0] - sx * side + ux * near * 1.6, h, centre[2] - sz * side + uz * near * 1.6], centre),
                    key(0.5, [centre[0] + ux * near, h, centre[2] + uz * near], centre),
                    key(1.0, [centre[0] + sx * side + ux * near * 1.6, h, centre[2] + sz * side + uz * near * 1.6], centre)]
            label = f"a fast fly-by past {self._called(name)}"
        elif rig == "fly through":
            gap = {"door": 1.1, "wall_with_door": 1.1, "window": 1.5, "wall_with_window": 1.55,
                   "arch": placed.size[1] * 0.4}.get(placed.kind, placed.top * 0.5)
            fx, fz = math.sin(turn), math.cos(turn)
            opening = [x, y + gap, z]
            far = (5.0 + reach) * max(scale, 0.7)
            start = [x + fx * far, y + gap + 0.2, z + fz * far]
            end = [x - fx * far * 0.8, y + gap, z - fz * far * 0.8]
            ahead = [end[0] - fx * 4, end[1], end[2] - fz * 4]
            keys = [key(0.0, start, opening), key(0.55, [x + fx * 0.2, y + gap, z + fz * 0.2], ahead),
                    key(1.0, end, ahead)]
            label = f"a fly-through of {self._called(name)}"
        elif rig == "fly over":
            far, high = (6.0 + reach) * scale, max(3.5, placed.top + 2.5)
            keys = [key(0.0, [centre[0] + ux * far, high * 0.6, centre[2] + uz * far], centre),
                    key(0.5, [centre[0] + ux * 0.8, high, centre[2] + uz * 0.8], centre),
                    key(1.0, [centre[0] - ux * far, high * 0.8, centre[2] - uz * far],
                        [centre[0] - ux * (far + 6), 0.5, centre[2] - uz * (far + 6)])]
            label = f"a high fly-over of {self._called(name)}"
        elif rig == "reveal":
            close, far = (1.2 + reach * 0.4) * scale, (6.0 + reach * 1.5) * scale
            keys = [key(0.0, [centre[0] + ux * close, 0.35, centre[2] + uz * close], [centre[0], centre[1] * 0.6, centre[2]]),
                    key(1.0, [centre[0] + ux * far, 3.5 + reach, centre[2] + uz * far], centre, lens_fov(24.0))]
            label = f"a rising reveal of {self._called(name)}"
        elif rig == "chase":
            fx, fz = math.sin(turn), math.cos(turn)
            behind = [x - fx * 3.0 + sx * 0.4, y + 1.7, z - fz * 3.0 + sz * 0.4]
            ahead = [x + fx * 4.0, y + 1.0, z + fz * 4.0]
            keys = None
            follow = name
            label = f"a chase shot behind {self._called(name)}"
            return behind, ahead, lens_fov(32.0), label, None, follow
        else:
            turns = _number(shot, "amount", where, p, -1080.0, 1080.0, 180.0)
            far = (2.2 + reach) * scale
            keys = []
            for k in range(9):
                a = angle + math.radians(turns) * k / 8
                keys.append(key(k / 8, [centre[0] + math.sin(a) * far * sign, centre[1] + 0.4,
                                        centre[2] + math.cos(a) * far * sign], centre, e="linear"))
            keys[0]["ease"] = ease
            label = f"an orbit around {self._called(name)}"
        if follow is None and self._moves_during(name):
            follow = name
        return keys[0]["p"], keys[0]["target"], fov, label, keys, follow

    def _move(self, shot, where, move, position, target, fov) -> list[dict]:
        if move == "static":
            return []
        kind, default = MOVES[move]
        distance = math.dist(position, target)
        amount = _number(shot, "amount", where, self.problems, -1000.0, 1000.0)
        if amount is None:
            if move in ("push in", "pull out"):
                amount = 0.3 * distance
            elif move == "dolly zoom":
                amount = 0.4 * distance
            elif move in ("zoom in", "zoom out"):
                lens = 12.0 / math.tan(math.radians(fov) / 2)
                amount = lens * (1.6 if move == "zoom in" else 0.65)
            else:
                amount = default
        if move == "pull out":
            amount = -abs(amount)
        return [{"move": kind, "amount": float(amount), "start": 0.0, "end": 1.0, "ease": "ease in and out",
                 "rise": 0.0, "units": "per capture"}]

    def _called(self, name: str) -> str:
        """A person's name, or "the" and a prop's id."""
        placed = self.placed[name]
        return placed.person["name"] if placed.person is not None else "the " + name.replace("_", " ")

    def _name(self, name: str) -> str:
        plain = self._plain(name)
        if getattr(self, "labelled", False) and name in getattr(self, "labels", {}):
            if name in self.mentioned:
                return f"<Subject {self.labels[name]}>"
            self.mentioned.add(name)
            return f"<Subject {self.labels[name]}> ({plain})"
        return plain

    def _plain(self, name: str) -> str:
        placed = self.placed[name]
        if placed.person is not None:
            return placed.person["name"]
        return name.replace("_", " ")

    def _seen(self, shot, when) -> tuple[list[tuple[str, str]], list[str]]:
        """The people in a shot's opening frame with the side of the frame each is on, and the props in view."""
        position, target = shot["position"], shot["target"]
        if shot.get("path"):
            position, target = shot["path"][0]["p"], shot["path"][0]["target"]
        fx, fy, fz = (target[i] - position[i] for i in range(3))
        length = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
        fx, fy, fz = fx / length, fy / length, fz / length
        rx, rz = -fz, fx
        norm = math.hypot(rx, rz) or 1.0
        rx, rz = rx / norm, rz / norm
        half_h = math.tan(math.radians(shot["fov"]) / 2)
        half_w = half_h * 16 / 9
        saved, self.now = self.now, when
        people, props = [], []
        for name in self.order:
            placed = self.placed[name]
            if placed.kind == "ground":
                continue
            x, y, z, _ = self._at(name, when)
            mid = [x, y + placed.top / 2, z]
            if placed.person is not None:
                mid = self._world_point(name, self._local_point(name, "chest", "shot"))
            dx, dy, dz = mid[0] - position[0], mid[1] - position[1], mid[2] - position[2]
            ahead = dx * fx + dy * fy + dz * fz
            if ahead <= 0.2:
                continue
            across = (dx * rx + dz * rz) / (ahead * half_w)
            up = (dy - ahead * fy) / (ahead * half_h)
            reach = max(placed.size) / 2 / (ahead * half_w)
            if abs(across) > 1.0 + reach or abs(up) > 1.0 + reach:
                continue
            if placed.person is not None:
                side = "on the left" if across < -0.3 else "on the right" if across > 0.3 else "in the centre"
                people.append((name, side))
            elif placed.kind not in FIXTURES or placed.kind == "door":
                props.append(name)
        self.now = saved
        return people, props

    def _framing(self, label: str) -> str:
        """A rig's label as a framing with its article: a two shot, a close-up, a medium shot."""
        text = label[0].lower() + label[1:]
        for said, written in (("two shot", "a two shot"), ("close up", "a close-up"), ("medium of", "a medium shot of"),
                              ("insert", "an insert"), ("wide shot", "a wide shot"), ("establishing shot",
                              "an establishing shot"), ("top down shot", "a top-down shot"), ("over ", "a shot over "),
                              ("fast fly by", "a fast fly-by"), ("moving shot", "a moving shot"),
                              ("locked-off shot", "a locked-off shot"), ("shot", "a shot")):
            if text.startswith(said):
                return written + text[len(said):]
        return text

    def _listed(self, names: list[str]) -> str:
        """Props in view, alike ones counted: the table, two plates and a bowl."""
        counts: dict[str, list[str]] = {}
        for n in names:
            placed = self.placed[n]
            word = n.replace("_", " ") if placed.kind in ("built",) + SHAPES else placed.kind.replace("_", " ")
            counts.setdefault(word, []).append(n)
        numbers = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
        parts = [self._thing(group[0]) if len(group) == 1 else f"{numbers.get(len(group), str(len(group)))} {w}s"
                 for w, group in list(counts.items())[:5]]
        return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]

    def _thing(self, name: str) -> str:
        placed = self.placed[name]
        if getattr(self, "labelled", False) and name in getattr(self, "labels", {}):
            word = name.replace("_", " ") if placed.kind in ("built",) + SHAPES else placed.kind.replace("_", " ")
            if name in self.mentioned:
                return f"<Subject {self.labels[name]}>"
            self.mentioned.add(name)
            return f"<Subject {self.labels[name]}> (the {word})"
        word = name.replace("_", " ") if placed.kind in ("built",) + SHAPES else placed.kind.replace("_", " ")
        return f"the {word}"

    def _label(self, shots) -> None:
        """Number the people, then the props any shot sees, as the prompt's subjects."""
        self.appears: dict[str, list[int]] = {}
        start = 0.0
        for index, shot in enumerate(shots):
            seen, props = self._seen(shot, start)
            for name in [n for n, _ in seen] + props:
                self.appears.setdefault(name, []).append(index + 1)
            start += shot["seconds"]
        order = [n for n in self.order if self.placed[n].person is not None]
        order += [n for n in self.order if self.placed[n].person is None and n in self.prop_picture]
        order += [n for n in self.order if self.placed[n].person is None and n in self.appears
                  and self.placed[n].kind not in FIXTURES and n not in self.prop_picture][:10]
        rooms = [n for n in self.order if self.placed[n].kind == "room" and n not in order]
        named = order + rooms
        if getattr(self, "set_picture", None):
            named = [n for n in named if self.placed[n].kind != "room"]
        self.labels = {name: k + 1 for k, name in enumerate(named)}

    def _ref2va(self, summary, description, sound, music) -> str:
        """The six-part MiniMax H3 reference prompt: each block replaced by the real thing it stands for."""
        labels = self.labels
        defined, kept = [], []
        for name, k in labels.items():
            placed = self.placed[name]
            shots = self.appears.get(name, [])
            where = (" (appears in " + ", ".join(f"[Shot {i}]" for i in shots) + ")") if shots else ""
            if placed.person is not None:
                info = placed.person
                colour = info.get("tint", (None, None))[1] or "coloured"
                picture = info.get("picture")
                if picture:
                    wears = f", {info['description']}" if info["description"] else ""
                    defined.append(f"<Subject {k}> is {info['name']}, the person in <Picture {picture}>{wears}, who "
                                   f"stands where the {colour} figure with a drawn face stands in <Video 1>.")
                    if info.get("keep_outfit"):
                        kept.append(f"<Subject {k}>{where}: fully_preserved - face, hairstyle, build and clothing are "
                                    f"retained; the place, facing and movement follow the {colour} figure in "
                                    "<Video 1>.")
                    else:
                        clothes = ("the clothing is not, and the outfit follows the description above"
                                   if info["description"] else "the clothing follows the scene")
                        kept.append(f"<Subject {k}>{where}: partially_preserved - facial identity, hairstyle and "
                                    f"build are retained; {clothes}; the place, facing and movement follow the "
                                    f"{colour} figure in <Video 1>.")
                    continue
                defined.append(f"<Subject {k}> is {info['name']}, {info['description'] or 'a real person'}, who stands "
                               f"where the {colour} figure with a drawn face stands in <Video 1>.")
                kept.append(f"<Subject {k}>{where}: attribute_transfer - only the place, facing and movement of the "
                            f"{colour} figure in <Video 1> are transferred to a real, living person; the figure's "
                            "shape, colour and drawn face are not kept.")
            elif name in self.prop_picture:
                look = BLOCK_LOOK.get(placed.kind, "the plain grey shape")
                text = self.prop_text.get(name)
                defined.append(f"<Subject {k}> is the {name.replace('_', ' ')} in <Picture {self.prop_picture[name]}>"
                               + (f", {text}" if text else "") + f", standing where {look} for {self._thing(name)} "
                               "stands in <Video 1>.")
                kept.append(f"<Subject {k}>{where}: fully_preserved - its shape, material, colour and detail are "
                            f"retained; its place, size and movement follow {look} in <Video 1>.")
            else:
                real = (self.prop_text.get(name) or (None if placed.kind in SHAPES else REAL.get(placed.kind))
                        or f"a real {name.replace('_', ' ')}")
                look = BLOCK_LOOK.get(placed.kind, "the plain grey shape")
                defined.append(f"<Subject {k}> is {real}, standing where {look} for {self._thing(name)} stands in "
                               "<Video 1>.")
                kept.append(f"<Subject {k}>{where}: attribute_transfer - only the place, size and movement of "
                            f"{look} in <Video 1> are transferred to {real}, with its full real shape, material and "
                            "detail; the plain block shape and flat grey of <Video 1> are not kept.")
        set_picture = getattr(self, "set_picture", None)
        if set_picture:
            k = max(labels.values(), default=0) + 1
            place = (self.setting or "the location").rstrip(".")
            place = place[0].lower() + place[1:]
            defined.append(f"<Subject {k}> is the environment in <Picture {set_picture}>: {place}.")
            kept.append(f"<Subject {k}>: fully_preserved - the location, its architecture, materials and light are "
                        "retained, extended to every angle the camera sees.")
        return (
            "subject_definitions:\n"
            "<Video 1> is a rough grey block-out of the shots on a checkerboard floor: grey shapes stand in for the "
            "set and props and coloured figures with drawn faces for the people. It defines camera movement, "
            "blocking and timing only.\n" + "\n".join(defined) + "\n\n"
            "summary:\n"
            "[reference generation] The target video replaces every block in <Video 1> with the real person, object "
            "or place it stands for, filmed as live action, and follows <Video 1> only for where things stand, the "
            f"camera's path and pace and the moments of the cuts. {summary}\n\n"
            "retention_analysis:\n"
            "<Video 1> (rough placement, camera path and cut timing): weak_reference - only where the people and the "
            "objects stand, the direction and pace of the camera and the moments of the cuts are taken; its grey "
            "models, coloured figures, checkerboard floor, plain background and flat light are not used.\n"
            + "\n".join(kept) + "\n\n"
            f"detailed_description:\n{description} {NOT_SHOWN}\n\n"
            f"overall_soundscape:\n{sound or 'N/A'}\n\n"
            f"non_diegetic_music:\n{music or 'N/A'}"
        )

    def _summary(self, setting, shots, total) -> str:
        cuts = [sum(s["seconds"] for s in shots[:i]) for i in range(1, len(shots))]
        people = [self._name(n) for n in self.order if self.placed[n].person is not None]
        cast = (", ".join(people[:-1]) + " and " + people[-1]) if len(people) > 1 else (people[0] if people else "")
        count = f"{len(shots)} shot{'s' if len(shots) > 1 else ''}"
        cut_text = (" with hard cuts at " + ", ".join(f"{t:.2f}s" for t in cuts)) if cuts else ", one continuous take"
        text = f"A {total:.1f}-second sequence in {count}{cut_text}."
        if setting:
            text += f" {setting.rstrip('.')}."
        if cast:
            text += f" It follows {cast}."
        actions = [s["action"].rstrip(".") for s in shots if s["action"]]
        if actions:
            text += " " + ". ".join(actions[:3]) + "."
        return text

    def _prompt(self, look, setting, shots, total) -> str:
        self._end = total
        lines = [look or LIVE_ACTION]
        time_of_day = LIGHTING_WORDS.get(getattr(self, "lighting", "day"), "")
        if time_of_day:
            lines.append(time_of_day)
        if setting:
            lines.append(setting.rstrip(".") + ".")
        people = [n for n in self.order if self.placed[n].person is not None]
        cast = []
        for n in people:
            info = self.placed[n].person
            colour = info.get("tint", (None, None))[1]
            text = f"The {colour} block in <Video 1> marks {info['name']}" if colour else info["name"]
            if info["description"]:
                text += f", {info['description'].rstrip('.')}"
            cast.append(text[0].upper() + text[1:])
        if cast and not self.labelled:
            lines.append(". ".join(cast) + ".")
        start = 0.0
        introduced: set = set()
        speakers: dict[str, int] = {}
        for index, shot in enumerate(shots):
            self.mentioned = set()
            seen, props = self._seen(shot, start)
            framing = self._framing(shot["label"])
            if index == 0:
                sentence = f"[Shot 1] {framing[0].upper() + framing[1:]}"
            else:
                minutes, seconds = divmod(start, 60.0)
                beat = self._on_beat(start)
                sentence = (f"[Shot {index + 1}] At {int(minutes):02d}:{seconds:06.3f}, "
                            f"{beat + ', ' if beat else ''}the shot cuts to {framing}")
            if seen:
                sides: dict[str, list[str]] = {}
                for n, side in seen[:4]:
                    sides.setdefault(side, []).append(self._name(n))
                sentence += ", with " + ", ".join(
                    (" and ".join(names) if len(names) < 3 else ", ".join(names[:-1]) + " and " + names[-1]) + f" {side}"
                    for side, names in sides.items())
            if props:
                sentence += "; " + self._listed(props) + " in view"
            sentence += "."
            for n, _ in seen:
                if n not in introduced and self.placed[n].person["description"]:
                    introduced.add(n)
            events = []
            for name, keys in self.paths.items():
                runs: list[list] = []
                for a, b in zip(keys, keys[1:]):
                    if (a[1], a[3]) == (b[1], b[3]):
                        continue
                    if runs and a[0] - runs[-1][1][0] <= RUN_JOIN:
                        runs[-1][1] = b
                    else:
                        runs.append([a, b])
                for a, b in runs:
                    who = self._name(name) if self.placed[name].person else self._thing(name)
                    if a[0] >= start - 1e-6 and a[0] < start + shot["seconds"] - 1e-6:
                        verb = "starts to move" if self.placed[name].person is None else "sets off"
                        events.append(f"{who} {verb} at {a[0]:.2f}s")
                        who = self._name(name) if self.placed[name].person else self._thing(name)
                    if start - 1e-6 <= b[0] <= start + shot["seconds"] + 1e-6 and b[0] < self._end - 1e-6:
                        verb = "comes to a stop" if self.placed[name].person is None else "reaches the spot"
                        events.append(f"{who} {verb} at {b[0]:.2f}s")
            if shot["action"]:
                sentence += " " + shot["action"].rstrip(".") + "."
            for who, words in shot.get("dialogue", []):
                if who not in speakers:
                    speakers[who] = len(speakers) + 1
                    voice = self.voices.get(who)
                    manner = f" in {voice}" if voice else ""
                else:
                    manner = ""
                sentence += (f" {self._name(who)[0].upper() + self._name(who)[1:]} (S{speakers[who]}) says{manner}, "
                             f"<d>[English] {words}</d>")
            if events:
                sentence += " " + " ".join(e[0].upper() + e[1:] + "." for e in events[:3])
            for when, words in shot.get("sounds", []):
                sentence += f" {words[0].upper() + words[1:]} at {when:.2f}s."
            move = MOVE_WORDS[shot["move"]]
            if shot["move"] != "static":
                move += f" over {shot['seconds']:.1f} seconds"
            if shot["rig"] in RIG_WORDS:
                move = RIG_WORDS[shot["rig"]].format(s=f"{shot['seconds']:.1f}") + (
                    f", and {MOVE_WORDS[shot['move']]}" if shot["move"] != "static" else "")
            elif shot.get("path"):
                move = f"the camera travels its path for the whole {shot['seconds']:.1f} seconds without stopping"
            sentence += " " + move[0].upper() + move[1:] + "."
            if shot.get("shake"):
                sentence += {"light": " The camera has a light handheld drift.",
                             "handheld": " The camera is handheld and moving.",
                             "heavy": " The camera shakes hard.",
                             "impact": " The camera jolts on the cut and settles."}.get(shot["shake"]["style"], "")
            lines.append(sentence)
            start += shot["seconds"]
        cuts = len(shots) - 1
        lines.append(f"The people, their clothes and the set stay the same in every shot, with exactly {len(shots)} "
                     f"shot{'s' if len(shots) > 1 else ''} and {cuts} hard cut{'s' if cuts != 1 else ''}, "
                     "no other cuts, no fades and no on-screen text.")
        return " ".join(lines)

def build(text: str, fps: float = 24.0, beats: Any = None, blueprint: str | None = None) -> Scene:
    """Read and resolve a blockout scene.

    Args:
        text: The scene JSON.
        fps: Frames per second the shots are timed at.
        beats: A song's beats, with ``times``, ``downbeats`` and ``tempo``; shots are counted and cut on them.
        blueprint: Video Blueprint's measurements as JSON, read for every shot whose rig is ``measured``.

    Returns:
        The resolved :class:`Scene`.

    Raises:
        BlockoutError: The text is not a scene this builds, with every problem listed.
    """
    builder = _Builder(parse(text))
    builder.blueprint = _blueprint_shots(blueprint)
    builder.people_tracks = _blueprint_people(blueprint)
    builder.object_tracks = _blueprint_people(blueprint, "objects")
    times = sorted(float(t) for t in getattr(beats, "times", None) or [])
    if len(times) >= 2 and float(getattr(beats, "tempo", 0) or 0) > 0:
        builder.grid = times
        builder.downbeats = [float(t) for t in getattr(beats, "downbeats", None) or []]
        builder.song_tempo = min(240.0, max(40.0, float(beats.tempo)))
    return builder.build(float(fps))
