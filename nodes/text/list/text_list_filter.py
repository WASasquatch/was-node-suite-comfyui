"""Keep the entries of a text list that match a search, or the ones that do not."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.compat.types import LIST
from ....modules.interface import run_result
from ....modules.prompt import text_lists

logger = log.get_logger("nodes.text.list")

#: The node's name in messages.
NAME = "Text List Filter"

#: Sample rows the report gives the kept side when both sides have entries.
KEPT_ROWS = run_result.MAX_ITEMS // 2

#: What the report shows for an entry holding nothing.
EMPTY_ROW = "(empty)"


class TextListFilter(io.ComfyNode):
    """Filter a text list by a search, keeping names in step with the texts."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASTextListFilter",
            display_name="Text List Filter",
            search_aliases=[
                "WASTextListFilter", "Text List Filter",
                "filter list",
                "filter captions",
                "keep matching",
                "drop matching",
                "grep",
                "select lines",
            ],
            category="WAS Suite/Text/List",
            description=(
                "Keep the entries of a text list that match a search, or the ones that do "
                "not, such as every caption that mentions a woman. Names wired beside the "
                "texts, such as file names, are kept in step, so each caption stays paired "
                "with its file."
            ),
            is_input_list=True,
            inputs=[
                io.MultiType.Input(
                    "texts",
                    [LIST, io.String],
                    tooltip=(
                        "The entries to filter: a LIST, such as Text Split to List's, or a "
                        "STRING list, such as the captions Load Text Batch reads. Each "
                        "entry is tested whole."
                    ),
                ),
                io.String.Input(
                    "find",
                    default="",
                    multiline=False,
                    tooltip=(
                        "What to look for. `woman, girl` = either term; with match `regex` "
                        "the whole box is one pattern, as `\\d+ steps`. Empty matches every "
                        "entry; match `is empty` ignores it."
                    ),
                ),
                io.Combo.Input(
                    "match",
                    options=list(text_lists.MATCHES),
                    tooltip=(
                        "`contains` = a term anywhere; `whole words` = `cat` but not "
                        "`category`; `starts with` / `ends with` = at either end, spaces "
                        "aside; `regex` = a regular expression; `is empty` = blank or only "
                        "spaces."
                    ),
                ),
                io.Boolean.Input(
                    "ignore_case",
                    default=True,
                    tooltip="On = `Woman` matches `woman`; off = case must match.",
                ),
                io.Combo.Input(
                    "keep",
                    options=list(text_lists.KEEPS),
                    tooltip=(
                        "`matching` = keep the entries that match and drop the rest; "
                        "`not matching` = drop the entries that match, as captions still "
                        "missing `[trigger]`."
                    ),
                ),
                io.MultiType.Input(
                    "names",
                    [LIST, io.String],
                    optional=True,
                    tooltip=(
                        "Optional names running beside texts, one per text, as the "
                        "filename_list from Load Image Batch. Kept and dropped with their "
                        "texts. Another count than texts stops the run."
                    ),
                ),
            ],
            outputs=[
                LIST.Output(
                    display_name="texts",
                    tooltip=(
                        "The entries kept, in their order, on one wire for Text List Get, "
                        "Text List to Text and the other list nodes. Empty when nothing "
                        "was kept."
                    ),
                ),
                LIST.Output(
                    display_name="names",
                    tooltip=(
                        "The names of the entries kept, in step with texts. Empty when "
                        "names is not wired."
                    ),
                ),
                io.String.Output(
                    display_name="text",
                    is_output_list=True,
                    tooltip=(
                        "The entries kept as a STRING list, so a node reading it runs once "
                        "per entry, such as Save Text File. Nothing kept stops the nodes "
                        "reading it."
                    ),
                ),
                io.String.Output(
                    display_name="name",
                    is_output_list=True,
                    tooltip=(
                        "The names kept as a STRING list, in step with text, for Save Text "
                        "File's filenames. Stops the nodes reading it when names is not "
                        "wired or nothing was kept."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip="How many entries were kept.",
                ),
                io.Int.Output(
                    display_name="removed",
                    tooltip="How many entries were dropped. 0 = every entry was kept.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        texts,
        find=None,
        match=None,
        ignore_case=None,
        keep=None,
        names=None,
    ) -> io.NodeOutput:
        from ....modules.compat.lists import block_if_empty

        find = text_lists.single(find, "")
        match = text_lists.single(match, "contains")
        ignore_case = bool(text_lists.single(ignore_case, True))
        keep = text_lists.single(keep, "matching")

        entries = text_lists.entries(texts)
        labels = text_lists.paired_names(names, len(entries), NAME)
        kept, spans = text_lists.filter_entries(
            entries, find, match, ignore_case, keep, f"{NAME}'s find"
        )
        kept_texts = [entries[index] for index in kept]
        kept_names = [labels[index] for index in kept] if labels is not None else []
        removed = len(entries) - len(kept)
        logger.debug("%s kept %d of %d entries", NAME, len(kept), len(entries))
        _publish(entries, labels, kept, spans, find, match, keep)

        nothing = (
            f"{NAME} kept none of the {len(entries)} entries it was given, with match "
            f"'{match}' and keep '{keep}', so the nodes reading this output have nothing to "
            f"run on. Check find, or switch keep."
        )
        no_names = (
            f"{NAME} has no names to hand on: wire the names input beside texts, such as the "
            f"filename_list from Load Image Batch."
        )
        return io.NodeOutput(
            kept_texts,
            kept_names,
            block_if_empty(list(kept_texts), nothing),
            block_if_empty(list(kept_names), no_names if labels is None else nothing),
            len(kept),
            removed,
        )


def _row(text: str, span, note: str) -> dict:
    """One sample row for the report, the matched span marked where there is one."""
    if span is not None and span[1] > span[0]:
        return run_result.excerpt(text, span[0], span[1], note)
    return {"text": text if text.strip() else EMPTY_ROW, "note": note}


def _publish(entries, labels, kept, spans, find, match, keep) -> None:
    """Report what the filter kept and dropped to the node's own interface.

    Args:
        entries: Every entry the node was given.
        labels: Their names, or None.
        kept: The positions kept.
        spans: Each entry's matched span, or None.
        find: The search.
        match: How entries were tested.
        keep: Which side was kept.
    """
    try:
        if not run_result.watching():
            return
        kept_set = set(kept)
        dropped = [index for index in range(len(entries)) if index not in kept_set]

        def note(word: str, index: int) -> str:
            return f"{word}: {labels[index]}" if labels is not None else word

        def rows(word: str, indices: list[int]) -> list[dict]:
            return [
                _row(entries[index], spans[index], note(word, index))
                for index in indices[: run_result.MAX_ITEMS]
            ]

        kept_rows = rows("kept", kept)
        dropped_rows = rows("removed", dropped)
        if kept_rows and dropped_rows:
            kept_rows = kept_rows[: max(KEPT_ROWS, run_result.MAX_ITEMS - len(dropped_rows))]
        items = (kept_rows + dropped_rows)[: run_result.MAX_ITEMS]
        if match == "is empty":
            test = "empty"
        elif find:
            test = f"{match} '{find}'"
        else:
            test = "anything, find being empty"
        run_result.publish(
            status=run_result.OK if kept else run_result.WARNING,
            summary=f"Kept {len(kept)} of {len(entries)} entries, those {keep}: {test}.",
            counts={"kept": len(kept), "removed": len(dropped), "entries": len(entries)},
            facts={
                "match": match,
                "keep": keep,
                "names": "wired" if labels is not None else "not wired",
            },
            items=items,
            items_total=len(entries),
        )
    except Exception as error:
        logger.debug("%s did not report its run (%s)", NAME, error)
