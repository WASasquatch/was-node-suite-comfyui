"""WAS Node Suite configuration.

``load_config()`` returns a mapping holding the full schema. It resolves ``$WAS_CONFIG``,
then ``<user_dir>/was-node-suite/config.yaml`` (``config.json`` also accepted), then
``<repo>/config.yaml``, then the built-in defaults.
"""

from __future__ import annotations

import copy
import json
import traceback
from collections.abc import Mapping
from pathlib import Path

from .. import log
from . import migrate, reconcile
from .defaults import DEFAULTS, FEATURE_GROUPS, LEGACY_GROUPS, VERSION
from .paths import (
    comfyui_user_directory,
    config_directory,
    find_config_file,
    repo_root,
    state_file,
    user_directory,
)

__all__ = [
    "DEFAULTS",
    "FEATURE_GROUPS",
    "LEGACY_GROUPS",
    "WINDOWS_PATH_FIX",
    "config_directory",
    "group_enabled",
    "load_config",
    "node_enabled",
    "state_file",
    "styles_file",
    "user_directory",
    "wildcards_directory",
]

logger = log.get_logger("config")

_config: dict | None = None


def load_config(refresh: bool = False) -> Mapping:
    """The merged configuration. Read once per process; ``refresh`` re-reads from disk."""
    global _config
    if _config is None or refresh:
        _config = _resolve()
    return _config


def _resolve() -> dict:
    try:
        path = find_config_file()
        if comfyui_user_directory() is None:
            # No ComfyUI, so no install to set up and nowhere a config file belongs.
            # Tooling reads whatever is already there and writes nothing, so nothing that
            # only reads the pack leaves state behind.
            return _merge((_read(path) if path is not None else None) or {}, path)
        if path is None:
            path = migrate.run(repo_root(), config_directory(), derive_config=True)
        else:
            migrate.run(repo_root(), config_directory(), derive_config=False)
        if path is None:
            return _merge({}, None)
        raw = _read(path)
        if raw is None:
            # A file that will not read is left exactly as it is.
            return _merge({}, path)
        if reconcile.reconcile(path, raw, repo_root()):
            # Re-read rather than merge what was just parsed: the rewrite is the file the
            # next start will read, so a difference between the two is worth finding now.
            raw = _read(path) or {}
        return _merge(raw, path)
    except Exception as error:
        logger.warning("the configuration could not be read (%s), using built-in defaults", error)
        logger.debug("%s", traceback.format_exc())
        return copy.deepcopy(DEFAULTS)


def _read(path: Path) -> Mapping | None:
    """One config file's settings.

    Args:
        path: The config file, YAML or JSON by its suffix.

    Returns:
        The settings, ``{}`` for an empty file, or ``None`` when the file cannot be opened,
        will not parse, or holds something other than a block of settings. Each ``None``
        is logged as a warning.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, ValueError) as error:
        logger.warning(
            "%s could not be opened (%s). It was left as it is, and this session runs on "
            "the built-in defaults",
            path, error,
        )
        return None
    try:
        if path.suffix == ".json":
            data = json.loads(text)
        else:
            import yaml

            data = yaml.safe_load(text)
    except Exception as error:
        logger.warning("%s", unreadable(path, text, error))
        return None
    if data is None:
        logger.info("%s is empty, using built-in defaults", path)
        return {}
    if not isinstance(data, Mapping):
        logger.warning(
            "%s should hold a block of settings but holds a %s. It was left as it is, and "
            "this session runs on the built-in defaults",
            path, type(data).__name__,
        )
        return None
    return data


#: The fix for a Windows path a double-quoted YAML string has read as escapes.
WINDOWS_PATH_FIX = "Use forward slashes or single quotes for Windows paths, as 'D:/prompts'."


def unreadable(path: Path, text: str, error: Exception) -> str:
    """The warning for a config file that will not parse.

    Args:
        path: The config file.
        text: Its contents.
        error: What the parser raised.

    Returns:
        One message naming the file, the line and column, the line itself and the fix.
    """
    mark = getattr(error, "problem_mark", None)
    if mark is not None:
        line, column = mark.line + 1, mark.column + 1
    else:
        line, column = getattr(error, "lineno", None), getattr(error, "colno", None)
    problem = getattr(error, "problem", None) or getattr(error, "msg", None) or str(error)
    rows = text.splitlines()
    shown = rows[line - 1].strip() if line and 0 < line <= len(rows) else ""

    where = f"line {line}, column {column}" if line and column else "an unknown line"
    parts = [f"{path} could not be read: {problem} at {where}"]
    if shown:
        parts[0] += f", in `{shown}`"
    parts[0] += "."
    if "\\" in shown or "escape" in str(problem):
        parts.append("Inside double quotes a backslash starts an escape sequence.")
        parts.append(WINDOWS_PATH_FIX)
    else:
        parts.append("Correct that line.")
    parts.append(
        "The file was left as it is, and this session runs on the built-in defaults until "
        "it is corrected and ComfyUI restarted."
    )
    return " ".join(parts)


def _merge(raw: Mapping, path: Path | None) -> dict:
    """User settings over the defaults, one block deep, reporting anything unrecognised."""
    config = copy.deepcopy(DEFAULTS)
    unknown = []
    for key, value in raw.items():
        if key not in config:
            unknown.append(str(key))
            continue
        block = config[key]
        if not isinstance(block, dict):
            config[key] = value
            continue
        if not isinstance(value, Mapping):
            logger.warning(
                "%s should be a block of settings, not a %s, so its defaults were kept",
                key, type(value).__name__,
            )
            continue
        for name, setting in value.items():
            if name in block:
                block[name] = setting
            else:
                unknown.append(f"{key}.{name}")
    if unknown:
        logger.warning("ignored unknown setting(s) in %s: %s", path, ", ".join(unknown))
    if config["version"] != VERSION:
        logger.warning(
            "%s declares version %s; this build reads version %s and may ignore parts of it",
            path, config["version"], VERSION,
        )
    for key in ("enable", "disable"):
        # `disable: Image Blend` instead of `disable: [Image Blend]` is an easy mistake to
        # make and an impossible one to notice: the node just stays where it was.
        if not isinstance(config["nodes"][key], list):
            logger.warning(
                "nodes.%s should be a list of node names, not a %s, so it was ignored",
                key, type(config["nodes"][key]).__name__,
            )
            config["nodes"][key] = []
    return config


def group_enabled(group: str | None, config: Mapping | None = None) -> bool:
    """Is ``"features.blip"`` / ``"legacy.cache"`` on? No group means the default tier."""
    if not group:
        return True
    config = load_config() if config is None else config
    section, _, name = group.partition(".")
    block = config.get(section)
    return bool(block.get(name, False)) if isinstance(block, Mapping) else False


def node_enabled(node_id: str, group: str | None = None, config: Mapping | None = None) -> bool:
    """Whether ``node_id`` is enabled, ``nodes.disable`` and ``nodes.enable`` first."""


    config = load_config() if config is None else config
    overrides = config.get("nodes")
    if isinstance(overrides, Mapping):
        if node_id in _ids(overrides.get("disable")):
            return False
        if node_id in _ids(overrides.get("enable")):
            return True
    return group_enabled(group, config)


def _ids(value) -> frozenset[str]:
    return frozenset(value) if isinstance(value, (list, tuple, set)) else frozenset()


def wildcards_directory() -> Path:
    """The wildcards directory: ``paths.wildcards``, or one in the config directory."""
    configured = load_config()["paths"]["wildcards"]
    return Path(configured).expanduser() if configured else config_directory() / "wildcards"


def luts_directory() -> Path:
    """The LUT directory: ``paths.luts``, or one in the config directory."""
    configured = load_config()["paths"]["luts"]
    return Path(configured).expanduser() if configured else config_directory() / "luts"


def styles_file() -> Path:
    """The style library: ``paths.styles``, which may be a .json or an A1111 .csv."""
    configured = load_config()["paths"]["styles"]
    return Path(configured).expanduser() if configured else state_file("styles.json")
