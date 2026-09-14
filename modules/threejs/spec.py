"""Descriptors a Three.js graph is built from, before the browser resolves them.

A descriptor is a plain dict carrying ``kind``, ``type``, ``params``, ``deps`` and an ``id``
taken from its contents, so two identical descriptors share one browser-side resource.
"""

from __future__ import annotations

__all__ = [
    "ALLOW_SCRIPTS",
    "SCHEMA_VERSION",
    "WRAPPER_KEY",
    "carries_script",
    "compact_deps",
    "create_spec",
    "identifier",
    "parse_json_array",
    "parse_json_object",
    "refuse_script",
    "require_spec",
]

import hashlib
import json
from typing import Any

#: Version stamped into every descriptor and checked when one is read back.
SCHEMA_VERSION = 1

#: Key holding :data:`SCHEMA_VERSION`, which marks a dict as a descriptor.
WRAPPER_KEY = "__was_threejs__"

#: Decimal places a float is rounded to before it reaches an identifier.
PLACES = 10

#: Characters of the digest kept as an identifier.
DIGEST_CHARS = 24


def _settled(value: Any) -> Any:
    """One value with its keys ordered and its floats rounded.

    Args:
        value: Any JSON-compatible value.

    Returns:
        The value, with dicts key-ordered and floats rounded to :data:`PLACES`.
    """
    if isinstance(value, dict):
        return {str(key): _settled(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_settled(item) for item in value]
    if isinstance(value, float):
        return 0.0 if value == 0 else round(value, PLACES)
    return value


def identifier(payload: dict[str, Any]) -> str:
    """The identifier one descriptor's contents hash to.

    Args:
        payload: The descriptor fields the identifier is taken from.

    Returns:
        The first :data:`DIGEST_CHARS` characters of the SHA-256 digest.
    """
    encoded = json.dumps(
        _settled(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:DIGEST_CHARS]


SCRIPT_PARAM = "javascript"

#: Config key that lets a scene carry code the browser runs.
ALLOW_SCRIPTS = "threejs.allow_scripts"


def carries_script(value) -> bool:
    """Whether a descriptor, or anything under it, carries browser code.

    Args:
        value: A descriptor, or any part of one.

    Returns:
        True when a ``params.javascript`` anywhere in the tree holds a non-empty string.
    """
    if isinstance(value, dict):
        params = value.get("params")
        if isinstance(params, dict) and str(params.get(SCRIPT_PARAM) or "").strip():
            return True
        return any(carries_script(entry) for entry in value.values())
    if isinstance(value, (list, tuple)):
        return any(carries_script(entry) for entry in value)
    return False


def refuse_script(value, where: str) -> None:
    """Stop a scene carrying browser code unless the config permits it.

    Args:
        value: The descriptor about to reach the browser.
        where: Name of the node, for the message.

    Raises:
        ValueError: The scene carries code and ``threejs.allow_scripts`` is off.
    """
    from ..config import group_enabled

    if not carries_script(value) or group_enabled(ALLOW_SCRIPTS):
        return
    raise ValueError(
        f"{where} was given a scene carrying javascript, which runs in this browser with "
        f"the same reach as ComfyUI itself. It is refused while {ALLOW_SCRIPTS} is false. "
        f"Set it to true in config.yaml to run scenes that script themselves, and only for "
        f"workflows you wrote or have read."
    )


def create_spec(
    kind: str,
    spec_type: str,
    *,
    params: dict[str, Any] | None = None,
    deps: dict[str, Any] | None = None,
    children: list[Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one descriptor.

    Args:
        kind: Family the descriptor belongs to, such as ``material`` or ``object``.
        spec_type: Three.js class the browser builds, such as ``MeshStandardMaterial``.
        params: Values passed to that class.
        deps: Descriptors this one is built from, keyed by the name it uses them under.
        children: Descriptors parented to this one, for a scene-graph entry.
        meta: Anything carried alongside that does not reach the class.

    Returns:
        The descriptor, with ``id`` taken from every other field.
    """
    payload: dict[str, Any] = {
        WRAPPER_KEY: SCHEMA_VERSION,
        "kind": kind,
        "type": spec_type,
        "params": params or {},
        "deps": deps or {},
    }
    if children is not None:
        payload["children"] = children
    if meta:
        payload["meta"] = meta

    payload["id"] = identifier(
        {
            "kind": payload["kind"],
            "type": payload["type"],
            "params": payload["params"],
            "deps": payload["deps"],
            "children": payload.get("children"),
            "meta": payload.get("meta"),
        }
    )
    return payload


def require_spec(value: Any, expected_kind: str | tuple[str, ...] | None = None) -> dict[str, Any]:
    """One input read back as a descriptor.

    Args:
        value: The socket value to read.
        expected_kind: Kind, or kinds, the descriptor is allowed to be. None accepts any.

    Returns:
        The descriptor.

    Raises:
        ValueError: The value is not a descriptor, or is not one of ``expected_kind``.
    """
    if not isinstance(value, dict) or value.get(WRAPPER_KEY) != SCHEMA_VERSION:
        raise ValueError(
            "This input did not arrive as a Three.js descriptor. Wire it from a Three node "
            "rather than from another socket that happens to carry a dictionary."
        )
    if expected_kind is None:
        return value

    expected = (expected_kind,) if isinstance(expected_kind, str) else expected_kind
    if value.get("kind") not in expected:
        raise ValueError(
            f"This input wanted a {' or '.join(expected)} descriptor and was handed a "
            f"{value.get('kind')!r} one. Check which output the wire came from."
        )
    return value


def compact_deps(**deps: Any) -> dict[str, Any]:
    """The dependencies that were given, with the unwired ones dropped.

    Args:
        **deps: Descriptors keyed by the name the Three.js class uses.

    Returns:
        The same mapping without its None entries.
    """
    return {name: value for name, value in deps.items() if value is not None}


def parse_json_object(text: str, label: str) -> dict[str, Any]:
    """One JSON object typed into a widget.

    Args:
        text: The widget's text. Empty reads as an empty object.
        label: The widget's name, for the message when it does not parse.

    Returns:
        The decoded object.

    Raises:
        ValueError: The text is not JSON, or is JSON that is not an object.
    """
    text = (text or "").strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} has to be a JSON object, written between {{ and }}.")
    return value


def parse_json_array(text: str, label: str) -> list[Any]:
    """One JSON array typed into a widget.

    Args:
        text: The widget's text. Empty reads as an empty array.
        label: The widget's name, for the message when it does not parse.

    Returns:
        The decoded array.

    Raises:
        ValueError: The text is not JSON, or is JSON that is not an array.
    """
    text = (text or "").strip()
    if not text:
        return []
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {error}") from error
    if not isinstance(value, list):
        raise ValueError(f"{label} has to be a JSON array, written between [ and ].")
    return value
