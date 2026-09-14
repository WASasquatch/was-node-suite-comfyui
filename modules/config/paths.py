"""Where the config file and the pack's writable state live.

Both sit under ComfyUI's user directory. ``$WAS_CONFIG_DIR`` replaces the configuration
directory and ``$WAS_CONFIG`` names the config file outright. Outside ComfyUI the
platform's per-user data directory stands in.
"""

from __future__ import annotations

import os
from pathlib import Path

from .. import log

logger = log.get_logger("config")

CONFIG_DIR_NAME = "was-node-suite"
CONFIG_NAMES = ("config.yaml", "config.json")
ENV_CONFIG_FILE = "WAS_CONFIG"
ENV_CONFIG_DIR = "WAS_CONFIG_DIR"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def comfyui_user_directory() -> Path | None:
    """ComfyUI's own user directory, or ``None`` when this is not a ComfyUI process."""
    try:
        import folder_paths
    except ImportError:
        return None
    try:
        return Path(folder_paths.get_user_directory())
    except Exception as error:
        logger.debug("ComfyUI's user directory could not be read (%s)", error)
        return None


def user_directory() -> Path:
    """Where the pack's writable state lives. Never the install directory.

    Returns:
        ComfyUI's user directory inside ComfyUI, and the platform's per-user data
        directory outside it.
    """
    # State in the install directory is lost on every update, cannot be written at all on
    # a read-only install, and makes a re-cloneable directory the only copy of a user's
    # settings.
    resolved = comfyui_user_directory()
    if resolved is not None:
        return resolved
    fallback = _data_home()
    logger.debug("ComfyUI is not present, so %s stands in for its user directory", fallback)
    return fallback


def env_value(name: str) -> str:
    """One environment variable's value.

    Args:
        name: The variable's name.

    Returns:
        Its value with surrounding space removed, or ``""`` where it is unset or empty.
    """
    return (os.environ.get(name) or "").strip()


def _data_home() -> Path:
    """The platform's per-user data directory."""
    for name in ("LOCALAPPDATA", "XDG_DATA_HOME"):
        value = env_value(name)
        if value:
            return Path(value)
    return Path.home() / ".local" / "share"


def config_directory() -> Path:
    """``<user_dir>/was-node-suite``, or ``$WAS_CONFIG_DIR`` when that is set."""
    override = env_value(ENV_CONFIG_DIR)
    return Path(override) if override else user_directory() / CONFIG_DIR_NAME


def state_file(name: str) -> Path:
    """A writable state file: settings db, history, styles, NSP pantry."""
    return config_directory() / name


def find_config_file() -> Path | None:
    """``$WAS_CONFIG``, then the config directory, then the repo root. YAML before JSON."""
    explicit = env_value(ENV_CONFIG_FILE)
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return path
        logger.warning("$%s points at %s, which is not a file", ENV_CONFIG_FILE, path)
    for directory in (config_directory(), repo_root()):
        for name in CONFIG_NAMES:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None
