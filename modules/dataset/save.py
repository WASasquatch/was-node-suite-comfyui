"""Writing a dataset's captions, and its pictures beside them, under each item's name.

An unreplaced picture is copied byte for byte, replaced pixels are encoded as PNG, and captions
are UTF-8 with ``\\n`` line endings.
"""

from __future__ import annotations

import filecmp
import os
import shutil
from pathlib import Path
from typing import Callable, NamedTuple

from .. import log
from ..io import text_batch
from ..util import sandbox, text_files
from .dataset import Dataset, Item

__all__ = ["CAPTIONS", "EXISTING", "EXPORT", "WRITES", "Saved", "encode_png", "save"]

logger = log.get_logger("dataset.save")

#: Write only the caption files.
CAPTIONS = "captions"

#: Write each picture and its caption.
EXPORT = "pictures and captions"

#: What a save writes.
WRITES = (CAPTIONS, EXPORT)

#: What a save does with a file already at a name: write over it where its content differs,
#: leave it alone, or write under the next free numbered name.
EXISTING = ("replace", "skip", "number")

#: Between a taken name and its number.
DELIMITER = "_"

#: Digits a number is padded to.
PADDING = 4


class Saved(NamedTuple):
    """What one save did, counted in files except where noted.

    Attributes:
        paths: Every file written, new or replaced, in order.
        written: Files that did not exist before.
        replaced: Files written over.
        unchanged: Files already holding what would be written, left as they were.
        skipped: Files already there and left alone.
        uncaptioned: Items with no caption, which wrote no caption file.
        no_picture: Items with neither pixels nor a picture file, left out of an export.
        failed: Files that could not be written; the log names each.
        renamed: ``(name, numbered name)`` per item written under a number.
    """

    paths: list
    written: int
    replaced: int
    unchanged: int
    skipped: int
    uncaptioned: int
    no_picture: int
    failed: int
    renamed: list


class _File(NamedTuple):
    """One file a save plans: its suffix and what it holds."""

    suffix: str
    payload: object


def encode_png(pixels) -> bytes:
    """A picture tensor as PNG bytes, without metadata.

    Args:
        pixels: A ``(1, height, width, channels)`` or ``(height, width, channels)`` tensor in
            ``[0, 1]``; 1, 2, 3 or 4 channels.

    Returns:
        The encoded file.
    """
    import io

    import numpy as np
    from PIL import Image

    frame = pixels[0] if pixels.ndim == 4 else pixels
    array = np.clip(frame.detach().float().cpu().numpy() * 255.0, 0, 255).round().astype(np.uint8)
    channels = array.shape[-1]
    if channels == 1:
        picture = Image.fromarray(array[..., 0], "L")
    elif channels == 2:
        picture = Image.fromarray(array, "LA")
    elif channels == 4:
        picture = Image.fromarray(array, "RGBA")
    else:
        picture = Image.fromarray(np.ascontiguousarray(array[..., :3]), "RGB")
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    return buffer.getvalue()


def _caption(item: Item) -> str | None:
    """An item's caption as it is written, or None where it has none."""
    if not item.captioned:
        return None
    return text_files.normalize_newlines(item.caption).strip()


def _picture(item: Item) -> _File | None:
    """The picture file an export writes for an item, or None where it holds no picture."""
    if item.pixels is not None:
        return _File(".png", item.pixels)
    if item.image:
        return _File(os.path.splitext(item.image)[1] or ".png", Path(item.image))
    return None


def _key(path: Path) -> str:
    """A path as a run compares it, case folded."""
    return os.path.normcase(str(path)).casefold()


def _numbered(folder: Path, name: str, suffixes: list[str], claimed: set) -> str:
    """The next ``<name>_0001`` under which every suffix is free on disk and in this run.

    Args:
        folder: The resolved save folder.
        name: The item's name; it may hold folders below ``folder``.
        suffixes: Every suffix the item writes.
        claimed: Paths this run has named, as :func:`_key` spells them.

    Returns:
        The numbered name, folders kept.

    Raises:
        PathNotAllowed: The name leaves the folder.
    """
    below, slash, leaf = name.rpartition("/")
    home = sandbox.resolve_save_file(folder, below) if below else folder
    head = f"{leaf}{DELIMITER}".casefold()
    endings = [suffix.casefold() for suffix in suffixes]
    seen = [0]
    try:
        entries = os.listdir(home)
    except OSError:
        entries = []
    for entry in entries:
        folded = entry.casefold()
        if not folded.startswith(head):
            continue
        for ending in endings:
            number = folded[len(head): len(folded) - len(ending)]
            if folded.endswith(ending) and len(number) >= PADDING and number.isdigit():
                seen.append(int(number))
    counter = max(seen) + 1
    while True:
        numbered = f"{below}{slash}{leaf}{DELIMITER}{counter:0{PADDING}}"
        paths = [sandbox.resolve_save_file(folder, f"{numbered}{suffix}") for suffix in suffixes]
        if not any(path.exists() or _key(path) in claimed for path in paths):
            return numbered
        counter += 1


def _same(path: Path, payload) -> bool:
    """Whether a file already holds what would be written there.

    Args:
        path: The resolved target.
        payload: Caption text, encoded bytes, or the picture file to copy.

    Returns:
        False where the file is missing or cannot be read.
    """
    try:
        if not path.is_file():
            return False
        if isinstance(payload, str):
            return text_batch.same_text(path, payload, "utf-8")
        if isinstance(payload, bytes):
            if os.path.getsize(path) != len(payload):
                return False
            with open(path, "rb") as handle:
                return handle.read(len(payload) + 1) == payload
        source = sandbox.resolve_read(payload)
        return os.path.samefile(source, path) or filecmp.cmp(source, path, shallow=False)
    except OSError:
        return False


def _write(folder: Path, file: str, payload) -> Path:
    """Write one file placed in a folder, making the folders between.

    Args:
        folder: The resolved save folder.
        file: The file name; it may hold folders below ``folder``.
        payload: Caption text, written with ``\\n`` line endings; encoded bytes; or the
            picture file to copy byte for byte.

    Returns:
        The written path.

    Raises:
        PathNotAllowed: The name leaves the folder, or the picture is outside every read root.
        OSError: The folder could not be made, or the file written.
    """
    path = sandbox.resolve_save_file(folder, file)
    os.makedirs(path.parent, exist_ok=True)
    if isinstance(payload, str):
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
    elif isinstance(payload, bytes):
        with open(path, "wb") as handle:
            handle.write(payload)
    else:
        shutil.copyfile(sandbox.resolve_read(payload), path)
    return path


def save(
    dataset: Dataset,
    folder: Path,
    write: str = CAPTIONS,
    suffix: str = ".txt",
    existing: str = "skip",
    step: Callable[[], None] | None = None,
) -> Saved:
    """Write a dataset's captions, or its pictures and captions, under each item's name.

    Args:
        dataset: The dataset to write.
        folder: The save folder, already resolved through :func:`rooted.save_destination`.
        write: One of :data:`WRITES`.
        suffix: The caption extension, with its leading dot.
        existing: One of :data:`EXISTING`.
        step: Called once per item written or passed over.

    Returns:
        What was written and what was not.

    Raises:
        PathNotAllowed: An item's name leaves the folder; nothing is written then.
        ValueError: ``write`` or ``existing`` is not one offered.
    """
    if write not in WRITES:
        raise ValueError(f"write `{write}` is not one of {', '.join(WRITES)}. Pick one from the menu")
    if existing not in EXISTING:
        raise ValueError(
            f"existing `{existing}` is not one of {', '.join(EXISTING)}. Pick one from the menu"
        )

    planned, uncaptioned, no_picture = [], 0, 0
    for item in dataset.items:
        files = []
        caption = _caption(item)
        if caption is None:
            uncaptioned += 1
        if write == EXPORT:
            picture = _picture(item)
            if picture is None:
                no_picture += 1
                logger.warning(
                    "Save Dataset left out `%s`, which holds no picture to export", item.name,
                )
            else:
                files.append(picture)
        if caption is not None and (write == CAPTIONS or files):
            files.append(_File(suffix, caption))
        if files:
            targets = [sandbox.resolve_save_file(folder, f"{item.name}{entry.suffix}") for entry in files]
            planned.append((item.name, files, targets))
        elif step is not None:
            step()

    paths, renamed, claimed = [], [], set()
    written = replaced = unchanged = skipped = failed = 0
    for name, files, targets in planned:
        stem = name
        if existing == "number" and any(_key(path) in claimed or path.exists() for path in targets):
            stem = _numbered(folder, name, [entry.suffix for entry in files], claimed)
            renamed.append((name, stem))
            targets = [
                sandbox.resolve_save_file(folder, f"{stem}{entry.suffix}") for entry in files
            ]
        for entry, path in zip(files, targets):
            existed = path.exists()
            again = _key(path) in claimed
            claimed.add(_key(path))
            if existed and existing == "skip" and not again:
                skipped += 1
                continue
            payload = entry.payload
            if not isinstance(payload, (str, bytes, Path)):
                payload = encode_png(payload)
            if existed and _same(path, payload):
                unchanged += 1
                continue
            if again:
                skipped += 1
                logger.warning(
                    "Save Dataset was handed `%s%s` twice in one run; the first is kept",
                    stem, entry.suffix,
                )
                continue
            try:
                done = _write(folder, f"{stem}{entry.suffix}", payload)
            except OSError as error:
                failed += 1
                logger.warning(
                    "Save Dataset could not write `%s%s` (%s)", stem, entry.suffix,
                    error.strerror or type(error).__name__,
                )
                continue
            paths.append(done)
            replaced += existed
            written += not existed
        if step is not None:
            step()
    return Saved(paths, written, replaced, unchanged, skipped, uncaptioned, no_picture, failed, renamed)
