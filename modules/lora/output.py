"""Where a merged LoRA is allowed to land.

The directory comes from ``folder_paths``; the file name is a widget value.
:func:`resolve_output` joins them, refusing a drive letter, a leading separator or a ``..``
segment.
"""

from __future__ import annotations

from pathlib import Path, PureWindowsPath

from ..util.sandbox import PathNotAllowed, contains

__all__ = [
    "DEFAULT_FILENAME",
    "SUFFIX",
    "built_from",
    "claimed",
    "lora_directory",
    "resolve_output",
]

#: Used when the filename widget is left empty, so the node still produces a file.
DEFAULT_FILENAME = "merged_lora.safetensors"

#: Appended to a name that does not already end in it. Nothing else can be loaded.
SUFFIX = ".safetensors"


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
    if relative.drive or relative.root or ".." in relative.parts:
        raise PathNotAllowed(
            f"refusing to write `{text}` in {directory}\n"
            f"  A LoRA file name is a name inside the LoRA directory, and this one carries "
            f"a drive, starts at a root, or steps out of it with '..'.\n"
            f"  Joining it onto the directory would discard the directory and write "
            f"somewhere else entirely."
        )

    target = directory.joinpath(*relative.parts).resolve(strict=False)
    if not contains(directory, target):
        raise PathNotAllowed(
            f"refusing to write {target}\n"
            f"  `{text}` leaves {directory}, which it was to be written inside, through a "
            f"symlink that points out of that directory."
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
    try:
        from safetensors import safe_open

        with safe_open(str(target), framework="pt") as handle:
            held = handle.metadata() or {}
    except Exception:
        return False
    return all(held.get(key) == value for key, value in wanted.items())
