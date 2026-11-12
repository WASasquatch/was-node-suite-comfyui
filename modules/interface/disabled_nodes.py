"""The node ids this pack held back at startup, and what brings each one back.

``GET /was/interface/api/disabled_nodes`` answers :func:`payload`. Paths in it are relative to
ComfyUI's own folder, or name an environment variable.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path

from .. import log
from ..config import paths
from ..util.sandbox import contains
from .channel import NO_STORE

__all__ = ["ROUTE", "config_label", "install_command", "payload", "record", "register_routes"]

logger = log.get_logger("interface.disabled_nodes")

ROUTE = "/was/interface/api/disabled_nodes"

#: Folder name a path inside ComfyUI's own folder is shown under.
COMFY_DIRECTORY = "ComfyUI"

#: Folder name the portable Windows build keeps its python in, beside ComfyUI's own folder.
EMBEDDED_PYTHON = "python_embeded"

#: A requirements file comment: ``#`` at the start of a line or after whitespace, to its end.
COMMENT = re.compile(r"(?:^|\s)#.*$")

_table: dict = {"config": "config.yaml", "groups": [], "disabled": [], "failed": []}

_registered = False


def _comfy_root() -> Path | None:
    """ComfyUI's own folder, absolute, or None outside ComfyUI."""
    try:
        import folder_paths
    except ImportError:
        return None
    value = getattr(folder_paths, "base_path", None)
    return Path(os.path.abspath(value)) if value else None


def _inside_comfy(path: Path, root: Path | None) -> str | None:
    """A path shown below ComfyUI's folder, as ``ComfyUI/user/...``.

    Args:
        path: An absolute path.
        root: ComfyUI's folder, or None.

    Returns:
        The path from ComfyUI's folder down with ``/`` separators, or None when it lies
        elsewhere.
    """
    if root is None or not contains(root, path):
        return None
    below = os.path.relpath(path, root).replace("\\", "/")
    return COMFY_DIRECTORY if below == "." else f"{COMFY_DIRECTORY}/{below}"


def config_label() -> str:
    """The config file this pack reads, named the way the docs name it.

    Returns:
        Such as ``ComfyUI/user/was-node-suite/config.yaml``, ``$WAS_CONFIG_DIR/config.yaml``
        or ``the file $WAS_CONFIG names``. Never an absolute path.
    """
    found = paths.find_config_file()
    explicit = paths.env_value(paths.ENV_CONFIG_FILE)
    if found is not None and explicit and found == Path(explicit):
        return f"the file ${paths.ENV_CONFIG_FILE} names"
    name = found.name if found is not None else paths.CONFIG_NAMES[0]
    if found is not None and found.parent == paths.repo_root():
        return f"{name} in the WAS Node Suite folder"
    if paths.env_value(paths.ENV_CONFIG_DIR):
        return f"${paths.ENV_CONFIG_DIR}/{name}"
    directory = Path(os.path.abspath(paths.config_directory()))
    shown = _inside_comfy(directory, _comfy_root())
    if shown is not None:
        return f"{shown}/{name}"
    return f"{paths.CONFIG_DIR_NAME}/{name} in ComfyUI's user folder"


def _python(root: Path | None) -> str:
    """The python that runs ComfyUI, as a command typed in ComfyUI's folder.

    Args:
        root: ComfyUI's folder, or None.

    Returns:
        A path relative to ComfyUI's folder for a python inside it or the portable build's,
        and ``python`` for any other.
    """
    if root is None or not sys.executable:
        return "python"
    executable = Path(os.path.abspath(sys.executable))
    embedded = (
        executable.parent.name.lower() == EMBEDDED_PYTHON
        and executable.parent.parent == root.parent
    )
    if contains(root, executable) or embedded:
        return os.path.relpath(executable, root)
    return "python"


def _satisfied(file: Path) -> bool:
    """Whether every package a requirements file names is installed at a version it accepts.

    Args:
        file: A requirements file.

    Returns:
        False when a package is missing or at a version the file refuses, and when a line is
        an option, an include or unreadable.
    """
    try:
        from importlib import metadata

        from packaging.requirements import InvalidRequirement, Requirement
    except ImportError:
        return False
    try:
        lines = file.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    for line in lines:
        text = COMMENT.sub("", line).strip()
        if not text:
            continue
        if text.startswith("-"):
            return False
        try:
            requirement = Requirement(text)
        except InvalidRequirement:
            return False
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        try:
            version = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            return False
        if not requirement.specifier.contains(version, prereleases=True):
            return False
    return True


def install_command(key: str) -> dict | None:
    """The command that installs one disabled group's packages.

    Args:
        key: Config key such as ``"features.llm"``.

    Returns:
        ``{"command": ..., "folder": ...}``, ``folder`` naming where the command is run:
        ``"ComfyUI"`` or ``"WAS Node Suite"``. None when the group needs no package, or when
        every package it needs is installed already.
    """
    try:
        from .. import deps
    except Exception as error:
        logger.debug("modules.deps is unavailable (%s), so no install command is given", error)
        return None
    requirements = deps.group_requirements(key)
    if requirements is None:
        return None
    file = Path(os.path.abspath(requirements))
    if _satisfied(file):
        return None
    root = _comfy_root()
    python = [_python(root)]
    if sys.flags.no_user_site:
        python.append("-s")
    if root is not None and contains(root, file):
        listed, folder = os.path.relpath(file, root), "ComfyUI"
    else:
        try:
            listed = os.path.relpath(file, paths.repo_root())
        except ValueError:
            listed = os.path.join(file.parent.name, file.name)
        folder = "WAS Node Suite"
    parts = (*python, "-m", "pip", "install", "-r", listed)
    command = " ".join(f'"{part}"' if " " in part else part for part in parts)
    return {"command": command, "folder": folder}


def record(groups: Mapping[str, Iterable[str]], disabled: Iterable[str], failed: Iterable[str]) -> None:
    """Keep what the loader held back, for :func:`payload` to answer.

    Args:
        groups: Config key of each group turned off, such as ``"features.llm"``, to the node
            ids it holds back.
        disabled: Node ids ``nodes.disable`` lists.
        failed: Node ids in modules that failed to import.
    """
    global _table
    held = []
    for key in sorted(groups):
        section, _, group = key.partition(".")
        held.append({
            "key": key,
            "section": section,
            "group": group,
            "nodes": sorted(set(groups[key])),
            "install": install_command(key),
        })
    _table = {
        "config": config_label(),
        "groups": held,
        "disabled": sorted(set(disabled)),
        "failed": sorted(set(failed)),
    }


def payload() -> dict:
    """The held-back node ids by cause, and the config file that brings them back.

    Returns:
        ``config``: the config file's label from :func:`config_label`. ``groups``: one entry
        per group turned off, with ``key``, ``section``, ``group``, ``nodes`` and ``install``,
        which is null where nothing is left to install.
        ``disabled``: ids ``nodes.disable`` lists. ``failed``: ids whose module failed to
        import.
    """
    return _table


def register_routes() -> bool:
    """Register the route answering :func:`payload`.

    Returns:
        True when the route was registered. False when it was registered already, or when the
        server could not be reached.
    """
    global _registered
    if _registered:
        return False
    try:
        from aiohttp import web
        from server import PromptServer

        @PromptServer.instance.routes.get(ROUTE)
        async def get_disabled_nodes(request):
            return web.json_response(payload(), headers=NO_STORE)

    except Exception as error:
        logger.warning(
            "%s was not registered (%s: %s), so opening a workflow that uses a disabled node "
            "shows only ComfyUI's missing node warning",
            ROUTE, type(error).__name__, error,
        )
        logger.debug("%s could not be registered", ROUTE, exc_info=True)
        return False
    _registered = True
    logger.debug("%s is serving the disabled node table", ROUTE)
    return True
