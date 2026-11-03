"""Replace words and phrases across a batch of texts, and make sure each holds a phrase."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.compat.lists import require_values
from ....modules.compat.types import LIST
from ....modules.interface import run_result
from ....modules.prompt import replace_words

logger = log.get_logger("nodes.text.search")

#: The node's name in messages.
NODE = replace_words.NODE

#: Characters of a text a report line quotes.
LINE_CHARS = 160

#: Characters of a file name or a matched phrase an item's note quotes.
NOTE_CHARS = 28


class TextReplaceWords(io.ComfyNode):
    """Replace whole words and phrases across a batch of texts and report what changed."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASTextReplaceWords",
            display_name="Text Replace Words",
            search_aliases=[
                "WASTextReplaceWords",
                "Text Replace Words",
                "replace words",
                "trigger word",
                "caption replace",
                "batch replace",
                "dataset captions",
            ],
            category="WAS Suite/Text/Search",
            description=(
                "Swap words and phrases across a whole batch of captions or prompts, such as "
                "every `a woman` or `woman` for `[trigger]`, and make sure each text holds a "
                "phrase. It matches whole words, keeps capitals, tidies the commas a deleted "
                "word leaves, and names the texts no rule matched."
            ),
            is_input_list=True,
            is_output_node=True,
            inputs=[
                io.MultiType.Input(
                    io.String.Input(
                        "text",
                        default="",
                        multiline=True,
                        placeholder="A woman stands on a beach, blurry",
                    ),
                    [io.String, LIST],
                    tooltip=(
                        "The captions or prompts to change, as `a woman on a beach`: a STRING "
                        "list such as Load Text Batch's texts, a LIST, or one text typed here. "
                        "Order is kept, so each text still pairs with its file name."
                    ),
                ),
                io.String.Input(
                    "rules",
                    default="",
                    multiline=True,
                    placeholder="a woman, the woman, woman -> [trigger]\nblurry ->",
                    tooltip=(
                        "One rule per line, as `a woman, the woman, woman -> [trigger]`: "
                        "commas separate the finds, and nothing after the arrow deletes them, "
                        "as `blurry ->`. Longer finds win. Lines starting with # and blank "
                        "lines are skipped. Empty = no change."
                    ),
                ),
                io.Boolean.Input(
                    "whole_words",
                    default=True,
                    tooltip=(
                        "true = `man` never matches inside `woman` or `manly`; false = finds "
                        "match inside other words too. A find starting or ending in a symbol, "
                        "such as `[trigger]` or `(red)`, matches wherever it stands."
                    ),
                ),
                io.Boolean.Input(
                    "ignore_case",
                    default=True,
                    tooltip=(
                        "true = `woman` also finds `Woman` and `WOMAN`; false = only the "
                        "spelling typed. The ensured phrase is looked for the same way."
                    ),
                ),
                io.Boolean.Input(
                    "keep_capitals",
                    default=True,
                    tooltip=(
                        "true = `Woman` becomes `Person` and `WOMAN` becomes `PERSON`; false = "
                        "every replacement as typed. A replacement not starting with a "
                        "letter, such as `[trigger]`, is always written as typed."
                    ),
                ),
                io.Boolean.Input(
                    "tidy",
                    default=True,
                    tooltip=(
                        "true = in each text a rule changed, doubled spaces, spaces before "
                        "commas and the stray commas a deleted word leaves are cleared, so "
                        "`a woman, , outdoors` becomes `a woman, outdoors`; false = only the "
                        "replacements are made."
                    ),
                ),
                io.String.Input(
                    "ensure",
                    default="",
                    tooltip=(
                        "A phrase every text has to hold, such as `[trigger]`, added to each "
                        "text that still lacks it after the rules. It is found the way the "
                        "finds are. A blank text stays blank. Empty = nothing added."
                    ),
                ),
                io.Combo.Input(
                    "ensure_at",
                    options=list(replace_words.PLACES),
                    default="start",
                    tooltip=(
                        "start = `[trigger], a woman on a beach`; end = `a woman on a beach, "
                        "[trigger]`, placed before a closing full stop when the separator is "
                        "a comma."
                    ),
                ),
                io.String.Input(
                    "separator",
                    default=", ",
                    tooltip=(
                        "What joins the ensured phrase to the text: `, ` for tag captions, a "
                        "space for prose, `\\n` for a line of its own."
                    ),
                ),
                io.MultiType.Input(
                    "names",
                    [LIST, io.String],
                    optional=True,
                    tooltip=(
                        "The file names that go with the texts, in the same order, such as "
                        "from Load Text Batch. Used only to name texts in the report; "
                        "unwired, texts are named by place, counting from 0."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="text",
                    is_output_list=True,
                    tooltip=(
                        "The texts after the rules, one per text that came in and in the same "
                        "order, so a node reading this runs once per text, such as a save "
                        "writing each caption back under its own name."
                    ),
                ),
                LIST.Output(
                    display_name="texts",
                    tooltip="The same texts as one LIST, for the list nodes.",
                ),
                io.Int.Output(
                    display_name="changed",
                    tooltip=(
                        "How many texts a rule or the ensured phrase changed. 0 = the batch "
                        "came through as it went in."
                    ),
                ),
                io.Int.Output(
                    display_name="unmatched",
                    tooltip=(
                        "How many texts no rule matched, such as captions that never mention "
                        "a woman for `woman -> [trigger]`. The report names each one; the "
                        "ensured phrase may still have been added to them."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        text,
        rules,
        whole_words,
        ignore_case,
        keep_capitals,
        tidy,
        ensure,
        ensure_at,
        separator,
        names=None,
    ) -> io.NodeOutput:
        texts = require_values(
            _strings(text),
            f"{NODE} received an empty list of texts, so there is nothing to change. Check "
            f"that the node feeding its text input found any files.",
        )
        rule_list = replace_words.parse_rules("\n".join(_strings(rules)))
        phrase = _first(ensure, "")
        place = "end" if _first(ensure_at, "start") == "end" else "start"
        outcomes = replace_words.apply(
            texts,
            rule_list,
            whole_words=bool(_first(whole_words, True)),
            ignore_case=bool(_first(ignore_case, True)),
            keep_capitals=bool(_first(keep_capitals, True)),
            tidied=bool(_first(tidy, True)),
            ensure=phrase,
            ensure_at=place,
            separator=_unescape(_first(separator, ", ")),
        )
        results = [outcome.text for outcome in outcomes]
        changed = sum(result != source for result, source in zip(results, texts))
        unmatched = sum(not outcome.matched for outcome in outcomes)
        logger.debug(
            "%s: %d texts, %d rules, %d changed, %d unmatched",
            NODE, len(texts), len(rule_list), changed, unmatched,
        )
        _publish(texts, outcomes, rule_list, _labels(names, len(texts)), phrase.strip(), place)
        return io.NodeOutput(list(results), list(results), changed, unmatched)


def _first(value, default):
    """The one value a list input carries, or ``default`` where it carries none."""
    if isinstance(value, list):
        return value[0] if value else default
    return default if value is None else value


def _strings(value) -> list[str]:
    """Every string a list input carries, wired LISTs opened however deep they nest.

    Args:
        value: The input as the node received it.

    Returns:
        The strings, in order, None read as an empty string.
    """
    found = []
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, (list, tuple)):
            pending.extend(reversed(item))
        else:
            found.append("" if item is None else str(item))
    return found


def _unescape(value: str) -> str:
    """A single-line widget's ``\\n`` and ``\\t`` read as the characters they name."""
    return str(value).replace("\\n", "\n").replace("\\t", "\t")


def _labels(names, total: int) -> list[str]:
    """What the report calls each text: its file name where one was wired, else its place.

    Args:
        names: The names input as the node received it, or None.
        total: How many texts arrived.

    Returns:
        ``total`` labels. A name is cut to its last path segment.
    """
    given = []
    for name in _strings(names) if names is not None else []:
        lines = [line.strip() for line in name.splitlines() if line.strip()]
        given.extend(lines if len(lines) > 1 else [name.strip()])
    labels = []
    for index in range(total):
        name = given[index] if index < len(given) else ""
        name = name.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
        labels.append(name or f"#{index}")
    return labels


def _texts(count: int) -> str:
    """``1 text`` or ``12 texts``."""
    return f"{count:,} text" if count == 1 else f"{count:,} texts"


def _cut(value: str, limit: int) -> str:
    """A string on one line, cut to ``limit`` characters with an ellipsis."""
    flat = " ".join(value.split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def _publish(texts, outcomes, rules, labels, phrase, place) -> None:
    """Report the run to the node's own panel.

    Args:
        texts: The texts as they arrived.
        outcomes: What :func:`modules.prompt.replace_words.apply` answered for them.
        rules: The rules the run applied.
        labels: What to call each text.
        phrase: The ensured phrase, or empty.
        place: Where the ensured phrase goes.
    """
    try:
        if not run_result.watching():
            return
        total = len(texts)
        changed = [index for index, outcome in enumerate(outcomes) if outcome.text != texts[index]]
        unmatched = [index for index, outcome in enumerate(outcomes) if not outcome.matched]
        ensured = sum(outcome.ensured for outcome in outcomes)
        hits = [sum(outcome.hits[index] for outcome in outcomes) for index in range(len(rules))]
        texts_hit = [
            sum(1 for outcome in outcomes if outcome.hits[index]) for index in range(len(rules))
        ]

        if not rules and not phrase:
            summary = f"No rules, so {_texts(total)} came through unchanged."
        else:
            summary = f"{len(changed):,} of {_texts(total)} changed"
            if rules:
                summary += f", {len(unmatched):,} matched no rule"
            if phrase:
                summary += f", {ensured:,} given {phrase}"
            summary += "."

        facts = {}
        for rule, count, reached in zip(rules[: run_result.MAX_FACTS - 1], hits, texts_hit):
            facts[rule.name] = f"{count:,} replaced in {_texts(reached)}"
        if phrase:
            facts["ensured"] = f"{phrase} at the {place}"

        items = []
        for index in changed[: run_result.MAX_ITEMS]:
            outcome = outcomes[index]
            label = _cut(labels[index], NOTE_CHARS)
            if outcome.found is not None:
                note = f"{label}, was '{_cut(outcome.found, NOTE_CHARS)}'"
            else:
                note = f"{label}, {phrase} added"
            start, end = outcome.after or (0, 0)
            items.append(run_result.excerpt(outcome.text, start, end, note=note))

        bodies = []
        if rules and unmatched:
            listing = "\n".join(
                f"{labels[index]}: {_cut(texts[index], LINE_CHARS)}" for index in unmatched
            )
            bodies.append(run_result.body("matched no rule", listing))
        if changed:
            listing = "\n".join(
                f"{labels[index]}: {_cut(texts[index], LINE_CHARS)}\n"
                f"  now: {_cut(outcomes[index].text, LINE_CHARS)}"
                for index in changed
            )
            bodies.append(run_result.body("changes", listing))

        run_result.publish(
            status=run_result.WARNING if unmatched or not changed else run_result.OK,
            summary=summary,
            counts={
                "texts": total,
                "changed": len(changed),
                "unchanged": total - len(changed),
                "unmatched": len(unmatched),
                "ensured": ensured,
            },
            tallies=[
                {"name": rule.name, "value": count}
                for rule, count in zip(rules[: run_result.MAX_TALLIES], hits)
            ],
            tallies_total=len(rules),
            facts=facts,
            items=items,
            items_total=len(changed),
            bodies=bodies,
        )
    except Exception as error:
        logger.debug("%s did not report its run (%s)", NODE, error)
