"""Prepares the pack before ComfyUI imports any custom node.

The views manifest is rewritten on every start, the drop directory for ``.zip`` view
extensions is created whether or not the installer is switched on, and a feature group that
is on with nothing installed for it has its requirements file installed.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

CONFIG_DIR_NAME = "was-node-suite"
CONFIG_NAMES = ("config.yaml", "config.json")
ENV_CONFIG_FILE = "WAS_CONFIG"
ENV_CONFIG_DIR = "WAS_CONFIG_DIR"

#: Whether the viewer loads at all, read from ``features.viewer``. Repeated from
#: ``modules/config/defaults.py`` for the same reason.
VIEWER_DEFAULT = True

#: Default of ``viewer.install_extensions``. Repeated from ``modules/config/defaults.py``
#: because that module is not importable this early.
INSTALL_EXTENSIONS_DEFAULT = False

#: Where ``.zip`` view extensions are dropped, under the pack's config directory. In the
#: user directory rather than the install directory so the packages survive an update.
DROP_DIR_NAME = "viewer-extensions"

#: The pack's logger name and the prefix its records carry. Repeated from ``modules/log.py``
#: because that module is not importable this early.
LOGGER_NAME = "was_node_suite"
PREFIX = "[WAS Node Suite] "

logger = logging.getLogger(f"{LOGGER_NAME}.viewer")


def configure_logging(level: str = "info") -> None:
    """Give the pack's logger a prefixed handler for the run of this script.

    Args:
        level: A level name. Anything unrecognised reads as ``"info"``.
    """
    root = logging.getLogger(LOGGER_NAME)
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    # Without this the records reach the stdlib's last-resort handler, which writes the
    # message to stderr with nothing naming the pack it came from.
    root.propagate = False
    for existing in list(root.handlers):
        root.removeHandler(existing)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(PREFIX + "%(message)s"))
    root.addHandler(handler)


configure_logging()


def config_directory() -> Path | None:
    """``<user_dir>/was-node-suite``, or ``$WAS_CONFIG_DIR`` when it is set."""
    override = os.environ.get(ENV_CONFIG_DIR)
    if override:
        return Path(override)
    try:
        import folder_paths

        return Path(folder_paths.get_user_directory()) / CONFIG_DIR_NAME
    except Exception as error:
        logger.debug("the config directory could not be located (%s)", error)
        return None


def find_config_file() -> Path | None:
    """``$WAS_CONFIG``, then the config directory, then the repository. YAML before JSON.

    The same order ``modules.config`` resolves, so both read one file and agree on which.
    """
    explicit = os.environ.get(ENV_CONFIG_FILE)
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    for directory in (config_directory(), Path(__file__).resolve().parent):
        if directory is None:
            continue
        for name in CONFIG_NAMES:
            candidate = directory / name
            try:
                if candidate.is_file():
                    return candidate
            except OSError as error:
                logger.debug("%s could not be read (%s)", candidate, error)
    return None


def read_config(path: Path) -> dict:
    """One config file as a mapping. A file that will not parse is a warning and ``{}``."""
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text) if path.suffix == ".json" else _yaml(text)
    except Exception as error:
        logger.warning(
            "could not read %s (%s), so every viewer key was left at its default",
            path, error,
        )
        return {}
    return data if isinstance(data, dict) else {}


def _yaml(text: str) -> object:
    import yaml

    return yaml.safe_load(text)


def block_of(path: Path | None, name: str) -> dict:
    """One top-level block of the config file, or ``{}``."""
    if path is None:
        return {}
    block = read_config(path).get(name)
    return block if isinstance(block, dict) else {}


def viewer_installer():
    """``modules/viewer/install.py``, loaded by path.

    Returns:
        The loaded module, or ``None`` when it cannot be read, which leaves the viewer's
        built-in views working and only the extension machinery absent.
    """
    import importlib.util

    path = Path(__file__).resolve().parent / "modules" / "viewer" / "install.py"
    try:
        spec = importlib.util.spec_from_file_location("was_node_suite_viewer_install", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception as error:
        logger.debug("%s could not be loaded (%s)", path, error)
        return None
    return module


def prepare_viewer(features: dict, viewer: dict) -> None:
    """Unpack view extensions if asked to, and list the ones that are present.

    Args:
        features: The config's ``features`` block, which decides whether the viewer loads.
        viewer: The config's ``viewer`` block, holding ``install_extensions``.
    """
    if not bool(features.get("viewer", VIEWER_DEFAULT)):
        return
    module = viewer_installer()
    if module is None:
        return

    pack_root = Path(__file__).resolve().parent
    directory = config_directory()
    drop = (directory / DROP_DIR_NAME) if directory is not None else None
    try:
        if drop is not None and module.ensure_drop_dir(drop):
            if bool(viewer.get("install_extensions", INSTALL_EXTENSIONS_DEFAULT)):
                module.install_all(pack_root, drop)
                module.sync_siblings(pack_root, pack_root.parent)
        module.write_manifest(pack_root)
    except Exception as error:
        logger.warning(
            "the content viewer's extensions could not be prepared (%s: %s). Its built-in "
            "views are unaffected.",
            type(error).__name__, error,
        )
        logger.debug("the viewer extension pass failed", exc_info=True)


def main() -> None:
    """Prepare the viewer."""
    try:
        path = find_config_file()
        configure_logging(str(block_of(path, "logging").get("level", "info")))
        prepare_viewer(block_of(path, "features"), block_of(path, "viewer"))
    except Exception as error:
        logger.warning(
            "the content viewer's extensions could not be prepared (%s: %s). Its built-in "
            "views are unaffected.",
            type(error).__name__, error,
        )
        logger.debug("the viewer extension pass failed", exc_info=True)


main()
