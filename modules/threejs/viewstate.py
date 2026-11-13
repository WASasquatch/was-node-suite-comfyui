"""A view saved by Three Viewer: its camera and the objects moved in it.

JSON: ``{"camera": {position, target, fov | height}, "objects": [{path, pivot, position,
quaternion, scale}]}``; a path is child indices from the scene down.
"""

from __future__ import annotations

import json
import math

from .spec import create_spec, require_spec

__all__ = ["MAX_OBJECTS", "MAX_TEXT", "apply", "parse"]

#: Longest view text read, in characters.
MAX_TEXT = 262144

#: Most moved objects one view holds.
MAX_OBJECTS = 256

#: Deepest path an object is addressed by.
MAX_DEPTH = 64

#: Largest coordinate a view may carry, so a typo cannot place a camera at infinity.
MAX_COORD = 1.0e6


def _vector(value, size: int, label: str) -> list[float]:
    """A list of ``size`` finite numbers within :data:`MAX_COORD`.

    Raises:
        ValueError: The value is not such a list.
    """
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{label} should be a list of {size} numbers")
    out = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{label} should hold numbers only")
        number = float(item)
        if not math.isfinite(number) or abs(number) > MAX_COORD:
            raise ValueError(f"{label} holds {item}, which is out of range")
        out.append(number)
    return out


def parse(text) -> dict:
    """The view a viewer saved, checked.

    Args:
        text: The view's JSON, or an empty string for none.

    Returns:
        ``{"camera": dict | None, "objects": list}``; both empty for an empty text.

    Raises:
        ValueError: The text is too long, is not JSON, or holds a value out of shape.
    """
    text = str(text or "").strip()
    if not text:
        return {"camera": None, "objects": []}
    if len(text) > MAX_TEXT:
        raise ValueError(f"the view is {len(text)} characters and the limit is {MAX_TEXT}")
    try:
        raw = json.loads(text)
    except ValueError as error:
        raise ValueError(f"the view is not JSON: {error}") from None
    if not isinstance(raw, dict):
        raise ValueError("the view should be a JSON object")

    camera = None
    if raw.get("camera") is not None:
        given = raw["camera"]
        if not isinstance(given, dict):
            raise ValueError("camera should be an object")
        camera = {
            "position": _vector(given.get("position"), 3, "camera position"),
            "target": _vector(given.get("target"), 3, "camera target"),
        }
        if given.get("fov") is not None:
            fov = _vector([given["fov"]], 1, "camera fov")[0]
            camera["fov"] = min(max(fov, 1.0), 179.0)
        if given.get("height") is not None:
            height = _vector([given["height"]], 1, "camera height")[0]
            camera["height"] = max(height, 1.0e-4)

    objects = []
    for entry in raw.get("objects") or []:
        if len(objects) >= MAX_OBJECTS:
            raise ValueError(f"the view moves more than {MAX_OBJECTS} objects")
        if not isinstance(entry, dict):
            raise ValueError("each moved object should be an object")
        path = entry.get("path")
        if (
            not isinstance(path, list) or not path or len(path) > MAX_DEPTH
            or not all(isinstance(step, int) and not isinstance(step, bool) and 0 <= step < 100000 for step in path)
        ):
            raise ValueError("a moved object's path should be a list of child indices")
        objects.append({
            "path": list(path),
            "pivot": _vector(entry.get("pivot"), 3, "pivot"),
            "position": _vector(entry.get("position"), 3, "position"),
            "quaternion": _vector(entry.get("quaternion"), 4, "quaternion"),
            "scale": _vector(entry.get("scale"), 3, "scale"),
        })
    return {"camera": camera, "objects": objects}


def _rebuilt(spec: dict, params: dict | None = None, deps: dict | None = None) -> dict:
    """A descriptor made again with new params or deps, so its id follows its contents."""
    return create_spec(
        spec["kind"],
        spec["type"],
        params=params if params is not None else spec.get("params"),
        deps=deps if deps is not None else spec.get("deps"),
        children=spec.get("children"),
        meta=spec.get("meta"),
    )


def apply(app: dict, view: dict) -> dict:
    """The app with a viewer's view in place of its own camera and object placement.

    Args:
        app: An app descriptor.
        view: :func:`parse`'s answer.

    Returns:
        A new app descriptor; ``app`` is left as it was. A camera set here no longer follows
        a Three Track.
    """
    require_spec(app, "app")
    deps = dict(app.get("deps") or {})

    camera = deps.get("camera")
    chosen = view.get("camera")
    if isinstance(camera, dict) and chosen:
        params = dict(camera.get("params") or {})
        params["position"] = list(chosen["position"])
        params["target"] = list(chosen["target"])
        if "fov" in chosen and "fov" in params:
            params["fov"] = chosen["fov"]
        if "height" in chosen and "viewHeight" in params:
            params["viewHeight"] = chosen["height"]
        camera_deps = {key: value for key, value in (camera.get("deps") or {}).items() if key != "track"}
        deps["camera"] = _rebuilt(camera, params=params, deps=camera_deps)

    scene = deps.get("scene")
    if isinstance(scene, dict) and view.get("objects"):
        params = dict(scene.get("params") or {})
        params["overrides"] = [dict(entry) for entry in view["objects"]]
        deps["scene"] = _rebuilt(scene, params=params)

    return _rebuilt(app, deps=deps)
