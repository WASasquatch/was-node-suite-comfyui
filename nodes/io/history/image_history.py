"""Reload an image from the pack's recently loaded image history."""

from __future__ import annotations

import os
import time

from comfy_api.latest import io

from ....modules import log
from ....modules.convert.tensors import pil2tensor
from ....modules.state import history
from ....modules.util import sandbox

logger = log.get_logger("nodes.io.history")

#: History key holding the paths every image-loading node appends to.
HISTORY_KEY = "Images"

#: Combo entry shown when the history holds nothing.
EMPTY = "No History"

#: Seconds a combo option list is reused for before the history is read again.
OPTIONS_TTL = 1.0

_options_cache: tuple[float, list[str]] = (0.0, [])


def label(path: str) -> str:
    """The menu entry for a history path: ``...<sep><parent dir><sep><file name>``."""
    parent = os.path.basename(os.path.dirname(path))
    return os.path.join("..." + os.sep + parent, os.path.basename(path))


def labelled_history(limit: int | None = None) -> dict[str, str]:
    """``{menu entry: absolute path}`` for images in the history, oldest first.

    Args:
        limit: How many of the newest to include, or None for every one recorded.

    Returns:
        One entry per label. Two paths sharing a label leave the newer one.
    """
    database = history.open_history_db()
    if limit is None:
        paths = database.get("History", HISTORY_KEY)
    else:
        paths = database.newest(HISTORY_KEY, limit)
        paths.reverse()
    return {label(path): path for path in paths}


def contained_history(entry: str) -> dict[str, str]:
    """``{menu entry: contained absolute path}`` for one selection from the menu.

    Args:
        entry: The combo selection, one of the entries :func:`options` offers.

    Returns:
        A one-item mapping for a selection the history lists, or an empty mapping for one
        it does not, which is what the :data:`EMPTY` placeholder is.

    Raises:
        PathNotAllowed: The recorded path lies outside every permitted read root.
    """
    # The newest entries first, the whole history only when the selection is older.
    recorded = labelled_history(history.display_limit()).get(entry)
    if recorded is None:
        recorded = labelled_history().get(entry)
    if recorded is None:
        return {}
    # A recorded path is checked against the read roots permitted now.
    return {entry: str(sandbox.resolve_read(recorded))}


def options() -> list[str]:
    """The combo's entries: the newest ``history.display_limit()`` images, or ``[EMPTY]``."""
    global _options_cache
    stamp, cached = _options_cache
    now = time.monotonic()
    if cached and now - stamp < OPTIONS_TTL:
        return cached
    limit = history.display_limit()
    entries = [label(path) for path in history.recent(HISTORY_KEY, limit)]
    entries = entries or [EMPTY]
    _options_cache = (now, entries)
    return entries


class ImageHistoryLoader(io.ComfyNode):
    """Load one of the images this pack has read or written before."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image History Loader",
            display_name="Image History Loader",
            search_aliases=["Image History Loader", "recent images", "history"],
            category="WAS Suite/History",
            description=(
                "Reload one of the images the suite's loading and saving nodes have recently "
                "touched. A file deleted since gives a black 512x512 image, and one in a "
                "folder the pack may no longer read stops the prompt with that folder named."
            ),
            inputs=[
                io.Combo.Input(
                    "image",
                    options=options(),
                    tooltip=(
                        "Which recently used image to reload. Entries are listed newest "
                        "last as '.../<folder>/<file>', and read 'No History' until a load "
                        "or save node has run."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="image",
                    tooltip=(
                        "The reloaded image, as a batch of one, with any transparency "
                        "discarded."
                    ),
                ),
                io.String.Output(
                    display_name="filename_text",
                    tooltip=(
                        "The file's own name, without the folders leading to it, for reuse as "
                        "a caption or a save prefix. A missing file reports 'null'."
                    ),
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, image) -> float:
        """Always stale: the file behind a history entry can be overwritten in place."""
        return float("NaN")

    @classmethod
    def execute(cls, image) -> io.NodeOutput:
        """Reload the selected image, or a black one when it is no longer there.

        Raises:
            PathNotAllowed: The recorded path lies outside every permitted read root.
        """
        from PIL import Image

        import node_helpers

        paths = contained_history(image)
        if image in paths and os.path.exists(paths[image]):
            loaded = node_helpers.pillow(Image.open, paths[image])
            return io.NodeOutput(
                pil2tensor(loaded.convert("RGB")), os.path.basename(paths[image])
            )
        logger.error("the image `%s` does not exist!", image)
        return io.NodeOutput(pil2tensor(Image.new("RGB", (512, 512), (0, 0, 0, 0))), "null")
