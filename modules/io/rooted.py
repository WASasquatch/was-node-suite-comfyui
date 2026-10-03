"""Where a node reads and writes: a permitted folder, and a path relative to it.

:func:`options` and :func:`read_options` are what a ``root`` widget offers.
:func:`destination` and :func:`source` settle a choice and a relative path.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from .. import log
from ..util import sandbox

__all__ = [
    "DEFAULT", "INPUT", "OUTPUT", "READ_DEFAULT", "TEMP", "destination", "options",
    "read_options", "read_roots", "roots", "source",
]

logger = log.get_logger("io.rooted")

#: The folders ComfyUI always has, and the one each kind of widget starts on.
INPUT = "input"
OUTPUT = "output"
TEMP = "temp"
DEFAULT = OUTPUT
READ_DEFAULT = INPUT

#: What a root is called when its folder shares a name with one already offered.
SPARE = "elsewhere"


def roots(extra: Sequence[tuple[str, object]] = ()) -> list[tuple[str, Path]]:
    """Every folder a node may write to, each with the name its widget offers.

    Args:
        extra: ``(name, path)`` pairs a node adds ahead of the rest, such as a directory
            of its own. A pair whose folder cannot be reached is dropped.

    Returns:
        ``[(name, path)]``, ``extra`` first, then ComfyUI's output and temp folders, then
        whatever ``paths.allow_write`` adds, each under its own folder's name. Empty
        outside ComfyUI and where none can be reached.
    """
    return _gathered(
        extra,
        ((OUTPUT, "get_output_directory"), (TEMP, "get_temp_directory")),
        _permitted("configured_write_roots"),
    )


def options(extra: Sequence[tuple[str, object]] = ()) -> list[str]:
    """The names a ``root`` widget offers, ``extra`` first and the output folder after."""
    return [name for name, _path in roots(extra)] or [OUTPUT]


def destination(root: str, relative: str, extra: Sequence[tuple[str, object]] = ()) -> Path:
    """The folder one root and one relative path name.

    Args:
        root: A name :func:`options` offered.
        relative: A path below it, with ``/`` or ``\\`` between the parts. Empty is the root
            itself.
        extra: The same pairs :func:`roots` was offered, so a name from them resolves.

    Returns:
        The absolute folder, which is always inside the root.

    Raises:
        PathNotAllowed: The root is not a folder this pack may write to.
        ValueError: The root is not one that was offered, or ``relative`` is absolute,
            names a drive, or climbs out of the root.
    """
    return _inside(roots(extra), root or DEFAULT, relative, options(extra), sandbox.resolve_write)


def read_roots(extra: Sequence[tuple[str, object]] = ()) -> list[tuple[str, Path]]:
    """Every folder a node may read from, each with the name its widget offers.

    Args:
        extra: ``(name, path)`` pairs a node adds ahead of the rest.

    Returns:
        ``[(name, path)]``, ``extra`` first, then ComfyUI's input, output and temp folders,
        then whatever ``paths.allow_read`` adds, each under its own folder's name.
    """
    return _gathered(
        extra,
        ((INPUT, "get_input_directory"), (OUTPUT, "get_output_directory"),
         (TEMP, "get_temp_directory")),
        _permitted("configured_read_roots"),
    )


def read_options(extra: Sequence[tuple[str, object]] = ()) -> list[str]:
    """The names a reading ``root`` widget offers, the input folder first."""
    return [name for name, _path in read_roots(extra)] or [INPUT]


def source(root: str, relative: str, extra: Sequence[tuple[str, object]] = ()) -> Path:
    r"""The folder one read root and one relative path name.

    Args:
        root: A name :func:`read_options` offered.
        relative: A path below it, with ``/`` or ``\`` between the parts. Empty is the
            root itself.
        extra: The same pairs :func:`read_roots` was offered.

    Returns:
        The absolute folder, which is always inside the root. It need not exist.

    Raises:
        PathNotAllowed: The root is not a folder this pack may read from.
        ValueError: The root is not one that was offered, or ``relative`` is absolute,
            names a drive, or climbs out of the root.
    """
    return _inside(read_roots(extra), root or READ_DEFAULT, relative, read_options(extra),
                   sandbox.resolve_read)


def _gathered(extra, comfy, configured) -> list[tuple[str, Path]]:
    """One root list: the caller's own folders, ComfyUI's, then the configured ones."""
    found, taken = [], set()
    for name, directory in extra:
        try:
            resolved = Path(directory).expanduser().resolve()
        except (OSError, TypeError):
            continue
        found.append((name, resolved))
        taken.add(name)

    for name, getter in comfy:
        directory = _comfy(getter)
        if directory is not None:
            found.append((name, directory))
            taken.add(name)

    known = {path for _name, path in found}
    for directory in configured:
        if directory in known:
            continue
        known.add(directory)
        base = "".join(ch if ch.isalnum() else "-" for ch in directory.name.lower()).strip("-")
        name = base or SPARE
        while name in taken:
            name = f"{name}-{len(taken)}"
        taken.add(name)
        found.append((name, directory))
    return found


def _inside(available, root: str, relative: str, offered, resolve) -> Path:
    """The one folder a root name and a relative path settle on, contained by ``resolve``."""
    name = str(root or "").strip()
    base = dict(available).get(name)
    if base is None:
        # Landing somewhere other than the folder that was asked for is the one thing worth
        # stopping over, so an unknown root is named rather than quietly swapped.
        raise ValueError(
            f"there is no folder called {name!r}. The choices are {', '.join(offered)}"
        )

    text = str(relative or "").strip().replace("\\", "/")
    if text.startswith("/"):
        raise ValueError(
            f"`{relative}` starts at a filesystem root, and this is a path inside {name}. "
            f"Write the part below the folder, such as `plates/shot`, and pick the folder "
            f"above"
        )
    text = text.strip("/")
    if not text:
        return resolve(base)

    candidate = Path(text)
    if candidate.is_absolute() or candidate.drive or candidate.anchor:
        raise ValueError(
            f"`{relative}` is a full path, and this is a path inside {name}. Write the part "
            f"below the folder, such as `plates/shot`, and pick the folder above"
        )
    reason = sandbox.leaves_folder(text)
    if reason is not None:
        raise ValueError(
            f"`{relative}` {reason}, so it leaves {name}, which is not reached. Write a "
            f"path that stays inside the folder"
        )

    settled = Path(os.path.normpath(base / candidate))
    if not sandbox.contains(base, settled):
        raise ValueError(
            f"`{relative}` lands outside {name}, which is not reached. Write a path that "
            f"stays inside the folder"
        )
    return resolve(settled)


def _comfy(getter: str):
    """One of ComfyUI's own directories, or None outside ComfyUI."""
    try:
        import folder_paths

        value = getattr(folder_paths, getter, None)
        if value is None:
            return None
        directory = Path(value()).expanduser().resolve()
    except Exception as error:
        logger.debug("the %s directory could not be read: %s", getter, error)
        return None
    return directory if directory.is_dir() else None


def _permitted(getter: str) -> list[Path]:
    """The folders one ``paths.allow_*`` key adds, resolved and reachable."""
    try:
        configured = getattr(sandbox, getter)()
    except Exception as error:
        logger.debug("the %s could not be read: %s", getter, error)
        return []
    found = []
    for directory in configured:
        try:
            resolved = Path(directory).resolve()
        except OSError:
            continue
        if resolved.is_dir():
            found.append(resolved)
    return found
