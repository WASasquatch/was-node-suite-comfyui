"""Put the entries of a text list in order, dropping repeats and blanks on request."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.compat.types import LIST
from ....modules.interface import run_result
from ....modules.prompt import text_lists

logger = log.get_logger("nodes.text.list")

#: The node's name in messages.
NAME = "Text List Sort"

#: What the report shows for an entry holding nothing.
EMPTY_ROW = "(empty)"


class TextListSort(io.ComfyNode):
    """Order a text list, keeping names in step with the texts."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASTextListSort",
            display_name="Text List Sort",
            search_aliases=[
                "WASTextListSort", "Text List Sort",
                "sort list",
                "natural sort",
                "shuffle list",
                "dedupe list",
                "unique lines",
                "order captions",
            ],
            category="WAS Suite/Text/List",
            description=(
                "Put the entries of a text list in order: alphabetical, natural so img2 comes "
                "before img10, by length, or shuffled by a seed, optionally dropping repeats "
                "and blank entries. Names wired beside the texts, such as file names, move "
                "with their texts."
            ),
            is_input_list=True,
            inputs=[
                io.MultiType.Input(
                    "texts",
                    [LIST, io.String],
                    tooltip=(
                        "The entries to order: a LIST, such as Text Split to List's, or a "
                        "STRING list, such as the captions Load Text Batch reads."
                    ),
                ),
                io.Combo.Input(
                    "order",
                    options=list(text_lists.ORDERS),
                    tooltip=(
                        "`as given` = unchanged; `a-z` / `z-a` = alphabetical; `natural` = "
                        "digits read as numbers, img2 before img10; `shortest first` / "
                        "`longest first` = by length; `shuffle` = random, set by seed."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    tooltip=(
                        "The shuffle's seed: the same seed and list always give the same "
                        "order; another seed gives another. Read only by order `shuffle`."
                    ),
                ),
                io.Boolean.Input(
                    "dedupe",
                    default=False,
                    tooltip=(
                        "On = an entry repeating an earlier one is dropped, the first kept, so "
                        "`a cat` and ` a cat ` leave one; off = repeats stay."
                    ),
                ),
                io.Boolean.Input(
                    "ignore_case",
                    default=True,
                    tooltip=(
                        "On = `Cat` and `cat` sort together and count as repeats for dedupe; "
                        "off = capitals sort before lower case and differ."
                    ),
                ),
                io.Boolean.Input(
                    "drop_empty",
                    default=False,
                    tooltip=(
                        "On = an entry holding nothing or only spaces is dropped, so 3 "
                        "captions with 1 blank give 2; off = blanks stay in place."
                    ),
                ),
                io.MultiType.Input(
                    "names",
                    [LIST, io.String],
                    optional=True,
                    tooltip=(
                        "Optional names running beside texts, one per text, as the "
                        "filename_list from Load Image Batch. Each moves and drops with its "
                        "text. Another count than texts stops the run."
                    ),
                ),
            ],
            outputs=[
                LIST.Output(
                    display_name="texts",
                    tooltip=(
                        "The entries in their new order, on one wire for Text List Get, Text "
                        "List to Text and the other list nodes."
                    ),
                ),
                LIST.Output(
                    display_name="names",
                    tooltip=(
                        "The names in the same order as texts. Empty when names is not wired."
                    ),
                ),
                io.String.Output(
                    display_name="text",
                    is_output_list=True,
                    tooltip=(
                        "The entries in their new order as a STRING list, so a node reading "
                        "it runs once per entry. An empty list stops the nodes reading it."
                    ),
                ),
                io.String.Output(
                    display_name="name",
                    is_output_list=True,
                    tooltip=(
                        "The names as a STRING list, in step with text. Stops the nodes "
                        "reading it when names is not wired or the list is empty."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip="How many entries came out.",
                ),
                io.Int.Output(
                    display_name="removed",
                    tooltip=(
                        "How many entries dedupe and drop_empty took out. 0 = every entry "
                        "came through."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        texts,
        order=None,
        seed=None,
        dedupe=None,
        ignore_case=None,
        drop_empty=None,
        names=None,
    ) -> io.NodeOutput:
        from ....modules.compat.lists import block_if_empty

        order = text_lists.single(order, "as given")
        seed = int(text_lists.single(seed, 0))
        dedupe = bool(text_lists.single(dedupe, False))
        ignore_case = bool(text_lists.single(ignore_case, True))
        drop_empty = bool(text_lists.single(drop_empty, False))

        entries = text_lists.entries(texts)
        labels = text_lists.paired_names(names, len(entries), NAME)
        positions, empties, duplicates = text_lists.sort_entries(
            entries, order, seed, dedupe, ignore_case, drop_empty
        )
        sorted_texts = [entries[index] for index in positions]
        sorted_names = [labels[index] for index in positions] if labels is not None else []
        removed = empties + duplicates
        logger.debug("%s ordered %d of %d entries", NAME, len(positions), len(entries))
        _publish(entries, labels, positions, order, seed, empties, duplicates)

        nothing = (
            f"{NAME} has no entries to hand on: it was given {len(entries)} and drop_empty "
            f"and dedupe left {len(positions)}. Check what feeds texts."
        )
        no_names = (
            f"{NAME} has no names to hand on: wire the names input beside texts, such as the "
            f"filename_list from Load Image Batch."
        )
        return io.NodeOutput(
            sorted_texts,
            sorted_names,
            block_if_empty(list(sorted_texts), nothing),
            block_if_empty(list(sorted_names), no_names if labels is None else nothing),
            len(positions),
            removed,
        )


def _publish(entries, labels, positions, order, seed, empties, duplicates) -> None:
    """Report the new order to the node's own interface.

    Args:
        entries: Every entry the node was given.
        labels: Their names, or None.
        positions: The positions kept, in their new order.
        order: The order applied.
        seed: The shuffle's seed.
        empties: Blank entries dropped.
        duplicates: Repeated entries dropped.
    """
    try:
        if not run_result.watching():
            return
        items = []
        for place, index in enumerate(positions[: run_result.MAX_ITEMS], start=1):
            note = f"{place}, was {index + 1}"
            if labels is not None:
                note = f"{note}: {labels[index]}"
            text = entries[index]
            items.append({"text": text if text.strip() else EMPTY_ROW, "note": note})
        facts = {"order": order}
        if order == "shuffle":
            facts["seed"] = str(seed)
        facts["names"] = "wired" if labels is not None else "not wired"
        run_result.publish(
            status=run_result.OK if positions else run_result.WARNING,
            summary=f"Kept {len(positions)} of {len(entries)} entries, order: {order}.",
            counts={
                "out": len(positions),
                "in": len(entries),
                "repeats": duplicates,
                "empty": empties,
            },
            facts=facts,
            items=items,
            items_total=len(positions),
        )
    except Exception as error:
        logger.debug("%s did not report its run (%s)", NAME, error)
