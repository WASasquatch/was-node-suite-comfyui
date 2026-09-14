"""Containment for every filesystem path a node accepts from a user.

:func:`resolve_read` and :func:`resolve_write` map an untrusted input onto an absolute
path inside a permitted root, or raise :class:`PathNotAllowed`. Writes exclude ComfyUI's
``input`` directory.
"""

from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath
from typing import Iterable

from .. import log
from ..config import load_config, paths

__all__ = [
    "PathNotAllowed",
    "contains",
    "names_another_host",
    "configured_read_roots",
    "configured_write_roots",
    "read_roots",
    "resolve_read",
    "resolve_write",
    "resolve_write_file",
    "write_roots",
]

logger = log.get_logger("util.sandbox")

#: Directory name the frozen ``./ComfyUI/...`` widget defaults use for ComfyUI's own tree.
COMFY_DIRECTORY = "ComfyUI"

#: Verbs the resolution errors are phrased with, and the only two values ``purpose`` takes.
READ = "read"
WRITE = "write"


class PathNotAllowed(ValueError):
    """A user-supplied path resolved outside every permitted root."""


def _comfy_directory(name: str) -> Path | None:
    """Return one of ComfyUI's directories, or ``None`` outside ComfyUI.

    Args:
        name: Attribute on ``folder_paths``, such as ``"get_input_directory"``.

    Returns:
        The resolved directory, or ``None`` when folder_paths is unavailable or the
        directory does not exist.
    """
    try:
        import folder_paths
    except ImportError:
        return None
    getter = getattr(folder_paths, name, None)
    if getter is None:
        return None
    try:
        value = getter()
    except Exception:
        return None
    return Path(value).expanduser().resolve() if value else None


def _comfy_root() -> Path | None:
    """Return ComfyUI's own directory, the one holding ``input/``, ``output/`` and ``temp/``.

    Returns:
        The resolved directory, or ``None`` outside ComfyUI or where folder_paths does not
        carry ``base_path``.
    """
    try:
        import folder_paths
    except ImportError:
        return None
    value = getattr(folder_paths, "base_path", None)
    if not value:
        return None
    try:
        return Path(value).expanduser().resolve()
    except OSError:
        return None


def _configured(key: str) -> list[Path]:
    """Return the extra roots listed under a config key.

    Args:
        key: Either ``"allow_read"`` or ``"allow_write"``.

    Returns:
        Resolved directories. Entries that are not directories are dropped with a warning.
    """
    configured = (load_config().get("paths") or {}).get(key) or []
    if isinstance(configured, (str, os.PathLike)):
        configured = [configured]
    roots = []
    for entry in configured:
        root = Path(entry).expanduser()
        try:
            root = root.resolve()
        except OSError:
            logger.warning("paths.%s entry %s cannot be resolved and is ignored", key, entry)
            continue
        if not root.is_dir():
            logger.warning("paths.%s entry %s is not a directory and is ignored", key, root)
            continue
        roots.append(root)
    return roots


def _pack_roots() -> list[Path]:
    """Return the pack's own state directory, where wildcards and styles live."""
    # config_directory(), not user_directory(): the latter is ComfyUI's whole user tree and
    # holds default/comfy.settings.json and default/workflows/, which a contained writer
    # must not reach. Every state file this pack owns resolves through paths.state_file(),
    # already inside the narrower directory.
    try:
        return [paths.config_directory().resolve()]
    except Exception:
        return []


def configured_read_roots() -> list[Path]:
    """Directories ``paths.allow_read`` names, without ComfyUI's own or this pack's."""
    return _configured("allow_read")


def configured_write_roots() -> list[Path]:
    """Directories ``paths.allow_write`` names, without ComfyUI's own or this pack's."""
    return _configured("allow_write")


def read_roots() -> list[Path]:
    """Directories a node may read from, most specific first."""
    roots = [
        _comfy_directory("get_input_directory"),
        _comfy_directory("get_output_directory"),
        _comfy_directory("get_temp_directory"),
    ]
    return [root for root in roots if root is not None] + _pack_roots() + _configured("allow_read")


def write_roots() -> list[Path]:
    """Directories a node may write to, most specific first."""
    roots = [
        _comfy_directory("get_output_directory"),
        _comfy_directory("get_temp_directory"),
    ]
    return [root for root in roots if root is not None] + _pack_roots() + _configured("allow_write")


def contains(root: Path, target: Path) -> bool:
    """Report whether ``target`` is ``root`` or lies beneath it.

    Args:
        root: A permitted root, already resolved.
        target: The candidate path, already resolved.

    Returns:
        True when target is inside root. Comparison is case-insensitive on Windows.
    """
    if os.name == "nt":
        root = Path(str(root).casefold())
        target = Path(str(target).casefold())
    return root == target or root in target.parents


def names_another_host(value: str | os.PathLike) -> bool:
    r"""Whether a path names a host rather than this machine.

    Args:
        value: The raw path, as written.

    Returns:
        True for a UNC path such as ``\\server\share\file``. Reading one reaches that host
        over the network, which resolving the path is enough to do.
    """
    drive = PureWindowsPath(str(value).strip()).drive
    return drive.startswith("\\\\") or drive.startswith("//")


def _host_permitted(text: str, roots: list[Path]) -> bool:
    r"""Whether a permitted root sits on the same host and share as ``text``.

    Args:
        text: The raw path, as written.
        roots: Permitted roots.

    Returns:
        True when a root names the same ``\\server\share``. Compared as written, so no path
        is resolved to answer this.
    """
    drive = PureWindowsPath(text).drive.replace("/", "\\").casefold()
    return any(
        PureWindowsPath(str(root)).drive.replace("/", "\\").casefold() == drive
        for root in roots
    )


def _rebased(text: str) -> Path | None:
    """Read ``./ComfyUI/output/x`` against ComfyUI's own root.

    Args:
        text: The raw widget value, stripped.

    Returns:
        The value with its leading ``ComfyUI`` component replaced by ComfyUI's root, so the
        frozen defaults name ComfyUI's own tree whatever directory the process was started
        in. ``None`` when the value is absolute or carries a drive, when its leading
        component is not ComfyUI's directory, or when that directory is unknown.
    """
    relative = PureWindowsPath(text)
    if relative.drive or relative.root:
        return None
    parts = relative.parts
    if not parts:
        return None
    root = _comfy_root()
    if root is None:
        return None
    if parts[0].casefold() not in {COMFY_DIRECTORY.casefold(), root.name.casefold()}:
        return None
    # '..' segments in the remainder survive the join and are collapsed by resolve, so a
    # rebased value can still land above ComfyUI's root, where the root check refuses it.
    return root.joinpath(*parts[1:]).resolve(strict=False)


def _candidates(text: str) -> list[Path]:
    """Return the absolute paths ``text`` may name, in the order they are tried.

    Args:
        text: The raw widget value, stripped.

    Returns:
        The value read absolute as written, or relative against the process working
        directory, followed, for a ``./ComfyUI/...`` value only, by the same value read
        against ComfyUI's root.
    """
    # strict=False so a write target that does not exist yet still resolves; parents and
    # symlinks along the existing prefix are resolved either way.
    found = [Path(text).expanduser().resolve(strict=False)]
    rebased = _rebased(text)
    if rebased is not None and rebased != found[0]:
        found.append(rebased)
    return found


def _relocated(target: Path) -> Path | None:
    """Return where a write into ComfyUI's input directory goes instead.

    Args:
        target: A resolved path that landed outside every write root.

    Returns:
        The matching path under ComfyUI's temp directory, or ``None`` when ``target`` is
        not inside the input directory or either directory is unknown.
    """
    source = _comfy_directory("get_input_directory")
    # The temp directory is a write root, is readable by a loader afterwards, and is
    # cleaned up with the rest of a run's scratch data.
    temp = _comfy_directory("get_temp_directory")
    if source is None or temp is None or not contains(source, target):
        return None
    return temp / os.path.relpath(target, source)


def _refusal(candidates: list[Path], roots: list[Path], key: str, purpose: str) -> PathNotAllowed:
    """Build the error naming the path, every permitted root and the key that permits it."""
    listed = "\n".join(f"    {root}" for root in roots) or "    (none)"
    message = (
        f"refusing to {purpose} {candidates[0]}\n"
        f"  It is outside every directory this pack may {purpose}:\n{listed}\n"
        f"  Add the directory to {key} in config.yaml to permit it."
    )
    if len(candidates) > 1:
        message += (
            f"\n  It was read against ComfyUI's own directory as well, as {candidates[1]}, "
            f"and refused there too."
        )
    return PathNotAllowed(message)


def _resolve(value: str | os.PathLike, roots: Iterable[Path], key: str, purpose: str) -> Path:
    """Resolve a user-supplied path and confirm it lies within one of ``roots``.

    Args:
        value: The raw widget value.
        roots: Permitted roots.
        key: Config key naming the list that would permit an outside path.
        purpose: Verb used in the error message, :data:`READ` or :data:`WRITE`.

    Returns:
        The resolved absolute path. It is not required to exist; a write target normally
        does not.

    Raises:
        PathNotAllowed: The path is empty, or resolved outside every permitted root.
    """
    text = str(value).strip()
    if not text:
        raise PathNotAllowed(f"no path given to {purpose}")
    roots = list(roots)
    if names_another_host(text) and not _host_permitted(text, roots):
        raise PathNotAllowed(
            f"`{text}` names another machine. A path is {purpose} from this computer only. "
            f"Add the share to {key} in config.yaml to reach it."
        )
    candidates = _candidates(text)
    for target in candidates:
        for root in roots:
            if contains(root, target):
                return target
    if purpose == WRITE:
        for target in candidates:
            moved = _relocated(target)
            if moved is not None:
                logger.warning(
                    "%s is in ComfyUI's input directory, which holds uploads and is not "
                    "written to; writing to %s instead. Add the input directory to "
                    "%s in config.yaml to write there.",
                    target, moved, key,
                )
                return moved
    raise _refusal(candidates, roots, key, purpose)


def _join(parent: Path, name: str | os.PathLike, purpose: str) -> Path:
    """Place a file name inside an already-resolved directory.

    Args:
        parent: Resolved directory, already confirmed to be inside a permitted root.
        name: File name to place in it. Sub-directories are allowed.
        purpose: Verb used in the error message, :data:`READ` or :data:`WRITE`.

    Returns:
        The resolved absolute path of the file, inside ``parent``.

    Raises:
        PathNotAllowed: The name is empty, names somewhere other than inside ``parent``, or
            resolves out of it through a symlink.
    """
    text = str(name).strip()
    if not text:
        raise PathNotAllowed(f"no file name given to {purpose} in {parent}")
    relative = PureWindowsPath(text)
    if relative.drive or relative.root or ".." in relative.parts:
        raise PathNotAllowed(
            f"refusing to {purpose} `{text}` in {parent}\n"
            f"  A file name is a name inside that directory, and this one carries a drive, "
            f"starts at a root, or steps out of it with '..'.\n"
            f"  Joining it onto the directory would discard the directory and "
            f"{purpose} somewhere else entirely."
        )
    target = parent.joinpath(*relative.parts).resolve(strict=False)
    if not contains(parent, target):
        raise PathNotAllowed(
            f"refusing to {purpose} {target}\n"
            f"  `{text}` leaves {parent}, which it was to be placed inside, through a "
            f"symlink that points out of that directory."
        )
    return target


def resolve_read(value: str | os.PathLike) -> Path:
    """Resolve a path a node intends to read.

    Args:
        value: The raw widget value.

    Returns:
        The resolved absolute path, inside a permitted read root.

    Raises:
        PathNotAllowed: The path resolved outside every permitted root.
    """
    return _resolve(value, read_roots(), "paths.allow_read", READ)


def resolve_write(value: str | os.PathLike) -> Path:
    """Resolve a path a node intends to write.

    Args:
        value: The raw widget value.

    Returns:
        The resolved absolute path, inside a permitted write root. The file need not exist.

    Raises:
        PathNotAllowed: The path resolved outside every permitted root.
    """
    return _resolve(value, write_roots(), "paths.allow_write", WRITE)


def resolve_write_file(directory: str | os.PathLike, name: str | os.PathLike) -> Path:
    """Resolve a directory a node writes into and the file name it writes there, together.

    Args:
        directory: The raw directory widget value, or an already-resolved directory.
        name: The file name to write there. Sub-directories are allowed; a drive, a leading
            root and a ``..`` segment are not.

    Returns:
        The resolved absolute file path, inside the resolved directory and so inside a
        permitted write root. It need not exist.

    Raises:
        PathNotAllowed: The directory resolved outside every permitted write root, or the
            name names somewhere other than inside it.
    """
    # os.path.join and Path both drop the left side for an absolute name and for a
    # drive-relative one such as `C:Windows\x`, and a '..' segment steps back out of a
    # directory checked a moment earlier; _join refuses all three.
    return _join(resolve_write(directory), name, WRITE)
