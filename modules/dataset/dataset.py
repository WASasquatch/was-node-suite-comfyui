"""A dataset of pictures and captions as one immutable value, and the operations that make new ones.

Pictures are decoded on demand into one cache shared by every dataset, keyed by file and stamp.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path

from .. import log
from ..util import sandbox
from . import health

__all__ = [
    "MAX_CACHED_BYTES",
    "MAX_ITEMS",
    "Dataset",
    "Item",
    "captions",
    "from_value",
    "load",
    "merge",
    "names",
    "pixels",
    "select",
    "with_captions",
    "with_images",
]

logger = log.get_logger("dataset")

#: Pictures one dataset holds, at most.
MAX_ITEMS = 20000

#: Bytes of decoded pictures the shared cache holds before the least recently read is dropped.
MAX_CACHED_BYTES = 2 * 1024 * 1024 * 1024

_lock = threading.Lock()

#: ``(path, mtime_ns, size)`` -> a ``(1, height, width, 3)`` float tensor, least recently read first.
_decoded: OrderedDict[tuple, object] = OrderedDict()
_cached_bytes = 0


@dataclass(frozen=True)
class Item:
    """One picture of a dataset with its caption.

    Attributes:
        name: Its path below the dataset's folder without extension, ``/`` separated.
        image: The picture file, already contained, or ``""`` for a picture held only as pixels.
        caption: The caption, or None where the picture has none.
        pixels: A ``(1, height, width, channels)`` tensor a node put in place of the file's.
        width: Width in pixels, 0 when not yet known.
        height: Height in pixels, 0 when not yet known.
    """

    name: str
    image: str = ""
    caption: str | None = None
    pixels: object = field(default=None, compare=False, repr=False)
    width: int = 0
    height: int = 0

    @property
    def captioned(self) -> bool:
        """Whether the item carries a caption holding more than whitespace."""
        return bool(self.caption and self.caption.strip())


@dataclass(frozen=True)
class Dataset:
    """Pictures and captions in a fixed order, shared freely between the nodes reading them.

    Attributes:
        items: The items, in order.
        source: The folder label it was loaded from, or a short description of how it was made.
        caption_extension: The extension its captions are saved with, such as ``.txt``.
    """

    items: tuple[Item, ...] = ()
    source: str = ""
    caption_extension: str = ".txt"

    def __len__(self) -> int:
        """How many items the dataset holds."""
        return len(self.items)


def from_value(value, where: str) -> Dataset:
    """The dataset a socket carried, refusing anything else.

    Args:
        value: What arrived, or a one-entry list of it from a list-input node.
        where: The node's name, for the message.

    Returns:
        The dataset.

    Raises:
        ValueError: Nothing, or something other than a dataset, arrived.
    """
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, Dataset):
        return value
    raise ValueError(
        f"{where} needs a dataset on its dataset input. Wire Load Dataset, or another dataset "
        f"node's dataset output, into it"
    )


def load(directory: Path, pattern: str, extension: str, source: str) -> Dataset:
    """Read a folder's pictures and the captions saved beside them.

    Args:
        directory: The folder, already resolved inside a permitted read root.
        pattern: Glob pattern for the pictures; ``**`` crosses folders.
        extension: The caption extension, such as ``.txt``.
        source: The folder's menu label.

    Returns:
        The dataset, sorted by name, at most :data:`MAX_ITEMS` pictures.

    Raises:
        PathNotAllowed: The pattern leaves the folder.
        ValueError: The caption extension is unusable.
    """
    suffix = health.caption_suffix(extension)
    base = Path(os.path.realpath(directory))
    paths = health.scan(base, pattern)
    if len(paths) > MAX_ITEMS:
        logger.warning(
            "%s holds %d matching pictures; the first %d were loaded. Narrow the pattern to load the rest",
            source, len(paths), MAX_ITEMS,
        )
        paths = paths[:MAX_ITEMS]
    items = []
    for path in paths:
        relative = os.path.relpath(path, base).replace("\\", "/")
        name = os.path.splitext(relative)[0]
        items.append(Item(name=name, image=str(path), caption=health.read_caption(path, suffix)))
    return Dataset(tuple(items), source=source, caption_extension=suffix)


def fingerprint(directory: Path, pattern: str, extension: str) -> str:
    """A digest that moves when a picture or a caption under the folder is added, removed or edited.

    Args:
        directory: The folder, already resolved inside a permitted read root.
        pattern: Glob pattern for the pictures.
        extension: The caption extension.

    Returns:
        A hexadecimal digest.
    """
    suffix = health.caption_suffix(extension)
    return health.fingerprint(health.scan(Path(os.path.realpath(directory)), pattern)[:MAX_ITEMS], suffix)


def names(dataset: Dataset) -> list[str]:
    """Every item's name, in order."""
    return [item.name for item in dataset.items]


def captions(dataset: Dataset) -> list[str]:
    """Every item's caption, in order, an empty string where an item has none."""
    return [item.caption or "" for item in dataset.items]


def pixels(item: Item):
    """An item's picture as a ``(1, height, width, 3)`` float tensor in ``[0, 1]``.

    Args:
        item: An item of a dataset.

    Returns:
        The tensor a node put in place, or the file decoded once and held in the shared cache.

    Raises:
        ValueError: The item has neither pixels nor a picture file that can be read.
    """
    if item.pixels is not None:
        return item.pixels
    if not item.image:
        raise ValueError(f"the dataset item `{item.name}` holds no picture")
    try:
        info = os.stat(item.image)
    except OSError as error:
        raise ValueError(
            f"the picture of `{item.name}` could not be read ({error.strerror}). It may have been "
            f"moved or deleted since the dataset was loaded; run Load Dataset again"
        ) from error
    key = (item.image, info.st_mtime_ns, info.st_size)
    with _lock:
        held = _decoded.get(key)
        if held is not None:
            _decoded.move_to_end(key)
            return held
    tensor = _decode(item.image, item.name)
    _hold(key, tensor)
    return tensor


def with_captions(dataset: Dataset, texts: list[str], by_name: list[str] | None = None) -> tuple[Dataset, int]:
    """A copy of a dataset carrying new captions; a blank text leaves an item uncaptioned.

    Args:
        dataset: The dataset to copy.
        texts: One caption per item in order, or a single caption for every item, or one per
            name of ``by_name``.
        by_name: Item names the texts belong to, matched exactly. Left out, texts go by position.

    Returns:
        ``(dataset, changed)``: the copy, sharing every item it did not change, and how many
        items' captions changed.

    Raises:
        ValueError: The counts do not line up, or a name is not in the dataset.
    """
    wanted = _aligned(dataset, list(texts), by_name, "captions")
    items, changed = list(dataset.items), 0
    for index, text in wanted.items():
        caption = text if text and text.strip() else None
        if caption != items[index].caption:
            items[index] = replace(items[index], caption=caption)
            changed += 1
    return (replace(dataset, items=tuple(items)) if changed else dataset), changed


def with_images(dataset: Dataset, images: list, by_name: list[str] | None = None) -> tuple[Dataset, int]:
    """A copy of a dataset holding new pictures in place of its files'.

    Args:
        dataset: The dataset to copy.
        images: ``IMAGE`` tensors, one per item in order or one per name of ``by_name``. A batch
            of several frames counts as that many pictures.
        by_name: Item names the pictures belong to. Left out, pictures go by position.

    Returns:
        ``(dataset, changed)``: the copy and how many items took a new picture.

    Raises:
        ValueError: The counts do not line up, or a name is not in the dataset.
    """
    frames = []
    for image in images:
        if image is None:
            continue
        frames.extend(image[index:index + 1] for index in range(int(image.shape[0])))
    wanted = _aligned(dataset, frames, by_name, "images")
    items = list(dataset.items)
    for index, frame in wanted.items():
        items[index] = replace(
            items[index], pixels=frame, width=int(frame.shape[2]), height=int(frame.shape[1]),
        )
    return replace(dataset, items=tuple(items)), len(wanted)


def select(dataset: Dataset, keep) -> tuple[Dataset, Dataset]:
    """Split a dataset in two by a test on each item.

    Args:
        dataset: The dataset to split.
        keep: Called with each :class:`Item`; true keeps it.

    Returns:
        ``(kept, removed)``, both in the original order and sharing the original items.
    """
    kept, removed = [], []
    for item in dataset.items:
        (kept if keep(item) else removed).append(item)
    return replace(dataset, items=tuple(kept)), replace(dataset, items=tuple(removed))


def merge(first: Dataset, second: Dataset) -> tuple[Dataset, int]:
    """Join two datasets, the second after the first, numbering a name the first already holds.

    Args:
        first: Its items keep their names.
        second: Its items follow; a name already taken gains ``_2``, ``_3`` and so on.

    Returns:
        ``(dataset, renamed)``: the joined dataset and how many of the second's items were renamed.

    Raises:
        ValueError: The joined dataset would hold more than :data:`MAX_ITEMS` items.
    """
    if len(first) + len(second) > MAX_ITEMS:
        raise ValueError(
            f"the two datasets hold {len(first) + len(second)} pictures together, past the "
            f"{MAX_ITEMS} one dataset holds. Filter one of them first"
        )
    taken = {item.name.casefold() for item in first.items}
    joined, renamed = list(first.items), 0
    for item in second.items:
        name = item.name
        if name.casefold() in taken:
            number = 2
            while f"{name}_{number}".casefold() in taken:
                number += 1
            name = f"{name}_{number}"
            renamed += 1
        taken.add(name.casefold())
        joined.append(replace(item, name=name) if name != item.name else item)
    source = first.source if first.source == second.source else f"{first.source} + {second.source}"
    return Dataset(tuple(joined), source=source, caption_extension=first.caption_extension), renamed


def _aligned(dataset: Dataset, values: list, by_name: list[str] | None, what: str) -> dict[int, object]:
    """Item index -> value, by position or by name.

    Raises:
        ValueError: The counts do not line up, or a name is not in the dataset.
    """
    count = len(dataset.items)
    if by_name:
        if len(values) == 1 and len(by_name) > 1:
            values = values * len(by_name)
        if len(values) != len(by_name):
            raise ValueError(
                f"{len(values)} {what} arrived for {len(by_name)} names, and they pair by "
                f"position. Wire one per name, in the same order"
            )
        where = {item.name: index for index, item in enumerate(dataset.items)}
        missing = [name for name in by_name if name not in where]
        if missing:
            shown = ", ".join(f"`{name}`" for name in missing[:3])
            raise ValueError(
                f"{len(missing)} of the names are not in the dataset, such as {shown}. Wire the "
                f"names that came out of the same dataset, such as Dataset to Lists' names"
            )
        return {where[name]: value for name, value in zip(by_name, values)}
    if len(values) == 1 and count > 1:
        values = values * count
    if len(values) != count:
        raise ValueError(
            f"{len(values)} {what} arrived for a dataset of {count}, and they pair by position. "
            f"Wire one per picture in the dataset's order, such as from Dataset to Lists, or "
            f"wire names beside them"
        )
    return dict(enumerate(values))


def _decode(path: str, name: str):
    """A picture file as a ``(1, height, width, 3)`` float tensor, upright and in RGB.

    Raises:
        ValueError: The file is not a picture that can be read.
    """
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    try:
        with Image.open(sandbox.resolve_read(path)) as opened:
            picture = ImageOps.exif_transpose(opened).convert("RGB")
    except (OSError, ValueError) as error:
        raise ValueError(f"the picture of `{name}` could not be decoded ({type(error).__name__})") from error
    array = np.asarray(picture, dtype=np.float32) / 255.0
    return torch.from_numpy(array)[None]


def _hold(key: tuple, tensor) -> None:
    """Keep one decoded picture, dropping the least recently read past :data:`MAX_CACHED_BYTES`."""
    global _cached_bytes
    size = int(tensor.numel()) * int(tensor.element_size())
    if size > MAX_CACHED_BYTES:
        return
    with _lock:
        if key in _decoded:
            return
        _decoded[key] = tensor
        _cached_bytes += size
        while _cached_bytes > MAX_CACHED_BYTES and _decoded:
            _, dropped = _decoded.popitem(last=False)
            _cached_bytes -= int(dropped.numel()) * int(dropped.element_size())
