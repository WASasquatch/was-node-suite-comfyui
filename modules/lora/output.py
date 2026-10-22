"""Where a merged LoRA lands.

:func:`resolve_output` places a widget's file name in the ``folder_paths`` LoRA directory,
refusing a drive, a leading separator or a ``..`` segment. :func:`foreign` tells a merge's own
file from any other.
"""

from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath

from ..util.sandbox import PathNotAllowed, contains, leaves_folder

__all__ = [
    "DEFAULT_FILENAME",
    "SUFFIX",
    "TOOL_KEY",
    "TOOL_NAME",
    "built_from",
    "claimed",
    "foreign",
    "header_metadata",
    "lora_directory",
    "resolve_output",
]

#: Used when the filename widget is left empty, so the node still produces a file.
DEFAULT_FILENAME = "merged_lora.safetensors"

#: Appended to a name that does not already end in it. Nothing else can be loaded.
SUFFIX = ".safetensors"

#: Header metadata key, and its value, stamped into every merged LoRA.
TOOL_KEY = "tool"
TOOL_NAME = "WAS Merge Loras"

#: Largest safetensors header read, in bytes.
HEADER_LIMIT = 100_000_000


def lora_directory() -> Path:
    """The directory a merged LoRA is written to, the first ``folder_paths`` lists for ``loras``.

    Returns:
        The resolved directory. It is not required to exist yet.

    Raises:
        RuntimeError: ``folder_paths`` lists no LoRA directory at all, which means there
            is nowhere a merged LoRA could be loaded from.
    """
    import folder_paths

    directories = folder_paths.folder_names_and_paths.get("loras", [[], []])[0]
    if not directories:
        raise RuntimeError(
            "This ComfyUI install has no LoRA directory, so there is nowhere to save a "
            "merged LoRA to. Add one under models/loras or in extra_model_paths.yaml."
        )
    return Path(directories[0]).expanduser().resolve()


def resolve_output(directory: Path, filename: str) -> tuple[Path, str]:
    """Place a user-supplied file name inside the LoRA directory.

    Args:
        directory: The LoRA directory, already resolved, from :func:`lora_directory`.
        filename: The raw filename widget value. Sub-directories are allowed and are
            created by the caller; a drive, a leading separator and a ``..`` segment are
            not. An empty value becomes :data:`DEFAULT_FILENAME`.

    Returns:
        ``(path, relative)``, the absolute path to write, and the name relative to the
        LoRA directory with ``/`` separators, which is what a loader takes and what the
        node reports.

    Raises:
        PathNotAllowed: The name names somewhere other than inside ``directory``.
    """
    text = (filename or "").strip().replace("\\", "/").lstrip("/")
    if not text:
        text = DEFAULT_FILENAME
    if not text.lower().endswith(SUFFIX):
        text += SUFFIX

    relative = PureWindowsPath(text)
    reason = leaves_folder(text)
    if reason is not None:
        raise PathNotAllowed(
            f"refusing to write `{text}` in {directory}\n"
            f"  A LoRA file name is a name inside the LoRA directory, and this one "
            f"{reason}.\n"
            f"  Joining it onto the directory would discard the directory and write "
            f"somewhere else entirely."
        )

    target = directory.joinpath(*relative.parts).resolve(strict=False)
    if not contains(directory, target):
        raise PathNotAllowed(
            f"refusing to write `{text}` in the LoRA folder\n"
            f"  It leaves that folder through a link that points out of it."
        )
    return target, "/".join(relative.parts)


def claimed(target: Path) -> str | None:
    """Whether something else holds the file open against writing.

    Args:
        target: The file a merge is about to be written to.

    Returns:
        What refused the write, or ``None`` where it can be written.
    """
    if not target.exists():
        return None
    try:
        with target.open("r+b"):
            return None
    except PermissionError as refused:
        return str(refused)
    except OSError as refused:
        return str(refused)


def header_metadata(target: Path) -> dict | None:
    """The ``__metadata__`` block of a safetensors file, read from its header alone.

    Args:
        target: The file to read.

    Returns:
        The metadata, empty where the header holds none, or ``None`` where the file is not
        a readable safetensors file.
    """
    try:
        with open(target, "rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                return None
            length = int.from_bytes(prefix, "little")
            if length < 2 or length > HEADER_LIMIT:
                return None
            header = handle.read(length)
        if len(header) != length:
            return None
        parsed = json.loads(header)
    except (OSError, ValueError, RecursionError):
        return None
    if not isinstance(parsed, dict):
        return None
    metadata = parsed.get("__metadata__") or {}
    return metadata if isinstance(metadata, dict) else {}


def foreign(target: Path) -> bool:
    """Whether a file already at the target was written by something other than a merge.

    Args:
        target: The file a merge is about to be written to.

    Returns:
        True where anything is there whose header does not carry :data:`TOOL_KEY` set to
        :data:`TOOL_NAME`; False where nothing is there or a merge wrote it.
    """
    if not target.exists():
        return False
    held = header_metadata(target)
    return not held or held.get(TOOL_KEY) != TOOL_NAME


def built_from(target: Path, wanted: dict) -> bool:
    """Whether the file already there carries the metadata a merge would write.

    Args:
        target: The file a merge is about to be written to.
        wanted: The metadata this merge would save inside it.

    Returns:
        True where every key matches, so the merge would only rebuild the same file.
    """
    if not target.exists() or not wanted:
        return False
    held = header_metadata(target)
    if not held:
        return False
    return all(held.get(key) == value for key, value in wanted.items())
