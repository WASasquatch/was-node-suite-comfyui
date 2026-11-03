"""Split a dataset in two by item name, by whether a caption is there, or by what the caption says."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import DATASET, LIST
from ...modules.dataset import dataset as datasets
from ...modules.interface import run_result
from ...modules.io import text_batch
from ...modules.prompt import text_lists

logger = log.get_logger("nodes.dataset")

#: Display name, written into the messages this node raises.
NAME = "Dataset Filter"

#: Which items a filter keeps.
KEEPS = (
    "named", "not named", "captioned", "uncaptioned", "caption matches", "caption does not match",
)

#: The keeps that read names.
BY_NAME = ("named", "not named")

#: The keeps that read find.
BY_CAPTION = ("caption matches", "caption does not match")

#: The keeps whose test is met by the items kept rather than by the items removed.
POSITIVE = ("named", "captioned", "caption matches")

#: What a sample row shows for an item with no caption.
NO_CAPTION = "(no caption)"


class DatasetFilter(io.ComfyNode):
    """Keep the items of a dataset that pass a test, and hand the rest on separately."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASDatasetFilter",
            display_name="Dataset Filter",
            search_aliases=[
                "WASDatasetFilter", "Dataset Filter",
                "filter dataset",
                "remove flagged",
                "drop pictures",
                "uncaptioned",
                "missing trigger",
                "training set",
            ],
            category="WAS Suite/Dataset",
            description=(
                "Split a dataset into the items to keep and the rest: by a list of names, such "
                "as the pictures Dataset Health flagged, by whether a caption is there, or by "
                "words the caption holds. Both halves are datasets, so each can be edited, "
                "saved or checked on its own."
            ),
            is_input_list=True,
            inputs=[
                DATASET.Input(
                    "dataset",
                    tooltip=(
                        "The dataset to split, from Load Dataset or any node that hands one on. "
                        "It is left as it is for every other node reading it."
                    ),
                ),
                io.Combo.Input(
                    "keep",
                    options=list(KEEPS),
                    default="captioned",
                    tooltip=(
                        "`named` / `not named` = the items names lists, or the rest; "
                        "`captioned` / `uncaptioned` = items with a caption, or without; "
                        "`caption matches` / `caption does not match` = captions holding a "
                        "find term, or the rest, uncaptioned items among them."
                    ),
                ),
                io.String.Input(
                    "find",
                    default="[trigger]",
                    tooltip=(
                        "For the caption keeps: words or phrases separated by commas, any of "
                        "which matches, as `[trigger], red hair`. Whole words, case ignored; "
                        "`[trigger]` is matched as written."
                    ),
                ),
                io.MultiType.Input(
                    "names",
                    [LIST, io.String],
                    optional=True,
                    tooltip=(
                        "For named and not named: the item names to pick, as Dataset Health's "
                        "flagged or Text List Filter's names. `cat`, `cat.png` and `set_b/cat` "
                        "all work; several lines are several names."
                    ),
                ),
            ],
            outputs=[
                DATASET.Output(
                    display_name="dataset",
                    tooltip=(
                        "The items kept, in their order, for Save Dataset, Dataset to Lists or "
                        "another filter. May hold no item."
                    ),
                ),
                DATASET.Output(
                    display_name="removed",
                    tooltip=(
                        "The items not kept, in their order: the flagged pictures, or the "
                        "captions still missing the trigger, to fix on their own."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip="How many items were kept.",
                ),
                io.Int.Output(
                    display_name="removed_count",
                    tooltip="How many items were removed. 0 = every item was kept.",
                ),
            ],
        )

    @classmethod
    def execute(cls, dataset, keep=None, find=None, names=None) -> io.NodeOutput:
        """Split the dataset and publish the counts and the first removed items.

        Raises:
            ValueError: No dataset arrived, keep is not one offered, a named keep has no
                names wired, or a caption keep has nothing in find.
        """
        data = datasets.from_value(dataset, NAME)
        keep = str(text_lists.single(keep, "captioned"))
        find = str(text_lists.single(find, "") or "")
        if keep not in KEEPS:
            raise ValueError(f"keep `{keep}` is not one of {', '.join(KEEPS)}. Pick one from the menu")

        match, unmatched = None, []
        if keep in BY_NAME:
            wanted = _wanted(names, keep, data.caption_extension)
            held = {item.name.casefold() for item in data.items}
            unmatched = [given for name, given in wanted.items() if name not in held]
            if unmatched:
                logger.info(
                    "%s: %d of the %d names match no item in %s, such as `%s`",
                    NAME, len(unmatched), len(wanted), data.source, unmatched[0],
                )

            def hit(item):
                return item.name.casefold() in wanted
        elif keep in BY_CAPTION:
            if not find.strip():
                raise ValueError(
                    f"keep `{keep}` tests each caption against find, and find is empty. Type "
                    f"the words to look for, as `[trigger], woman`, or pick another keep"
                )
            match = text_lists.matcher(find, "whole words", True, f"{NAME}'s find")

            def hit(item):
                return match(item.caption or "") is not None
        else:

            def hit(item):
                return item.captioned

        positive = keep in POSITIVE
        kept, removed = datasets.select(data, lambda item: hit(item) == positive)
        logger.debug("%s kept %d of %d items", NAME, len(kept), len(data))
        _publish(data, kept, removed, keep, find, match, len(unmatched))
        return io.NodeOutput(kept, removed, len(kept), len(removed))


def _wanted(names, keep: str, extension: str) -> dict[str, str]:
    """The names wired as item names, folded for comparison, each mapped to its spelling.

    Raises:
        ValueError: Nothing is wired to names.
    """
    given = text_batch.name_list(names)
    if given is None:
        raise ValueError(
            f"keep `{keep}` picks items by name, and names is not wired. Wire a list of names, "
            f"such as Dataset Health's flagged or Text List Filter's names, or pick another keep"
        )
    found = {}
    for name in given:
        stem = text_batch.stem(name, extension)
        if stem:
            found.setdefault(stem.casefold(), stem)
    return found


def _row(item, keep: str, match) -> dict:
    """One removed item as a sample row: its caption around a match, or its name."""
    if keep not in BY_CAPTION:
        return {"text": item.name, "note": "captioned" if item.captioned else "no caption"}
    caption = item.caption or ""
    span = match(caption) if match is not None and caption else None
    if span is not None and span[1] > span[0]:
        return run_result.excerpt(caption, span[0], span[1], item.name)
    return {"text": text_batch.snippet(caption) or NO_CAPTION, "note": item.name}


def _publish(data, kept, removed, keep: str, find: str, match, unmatched: int) -> None:
    """Report the counts and the first removed items to the node's panel.

    Never raises, and never changes what the node returns.

    Args:
        data: The dataset that arrived.
        kept: The items kept.
        removed: The items removed.
        keep: The keep chosen.
        find: The find text.
        match: The caption test, or None for the other keeps.
        unmatched: Wired names that match no item.
    """
    try:
        if not run_result.watching():
            return
        facts = {"keep": keep, "source": data.source or "unnamed"}
        if keep in BY_CAPTION:
            facts["find"] = find
        if unmatched:
            facts["unmatched names"] = unmatched
        if not len(data):
            status, summary = run_result.WARNING, "the dataset is empty"
        elif not len(kept):
            status, summary = run_result.WARNING, f"kept none of {len(data)} items with keep `{keep}`"
        else:
            status = run_result.OK
            summary = f"kept {len(kept)} of {len(data)} items, removed {len(removed)}"
        run_result.publish(
            status=status,
            summary=summary,
            counts={"kept": len(kept), "removed": len(removed)},
            facts=facts,
            items=[_row(item, keep, match) for item in removed.items[: run_result.MAX_ITEMS]],
            items_total=len(removed),
        )
    except Exception as error:
        logger.debug("%s did not report its split (%s)", NAME, error)
