"""Measure a set of captions: lengths, repeats, subject coverage and the phrases they share."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import DATASET, LIST
from ...modules.dataset import dataset as datasets
from ...modules.interface import run_result
from ...modules.prompt import caption_stats

logger = log.get_logger("nodes.dataset")

#: Display name, written into the messages this node raises.
NAME = "Caption Statistics"

#: Raised when neither the dataset input nor the text input holds a caption.
NOTHING = (
    f"{NAME} was handed no captions. Wire a dataset into dataset, such as Load Dataset's, or a "
    f"list of captions into text, such as the text output of Load Text Batch, one caption per "
    f"picture."
)

#: Raised when the wired dataset holds no picture.
EMPTY_DATASET = (
    f"the dataset wired into {NAME} holds no picture, so there is no caption to measure. Wire a "
    f"dataset with pictures in it, or loosen whatever filtered it."
)

#: The report's phrase table when no phrase is in two captions.
NO_PHRASES = "no phrase appears in two or more captions"

#: The report's flagged section when every caption passed.
NONE_FLAGGED = "no caption was flagged"

#: Frequent phrases named in the summary line and the facts.
MAX_NAMED = 3


class CaptionStatistics(io.ComfyNode):
    """Report how consistent a caption set is and which phrases dominate it."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASCaptionStatistics",
            display_name="Caption Statistics",
            search_aliases=[
                "WASCaptionStatistics", "Caption Statistics",
                "caption stats",
                "caption report",
                "caption consistency",
                "phrase frequency",
                "tag frequency",
                "trigger coverage",
                "dataset captions",
                "training captions",
                "lora dataset",
            ],
            category="WAS Suite/Dataset",
            description=(
                "Measure a set of captions, from a text list or a dataset, before training: "
                "word counts, repeats, short or overlong captions, how many name the subject, "
                "and the phrases most of them share, which a LoRA learns to lean on. Hands on "
                "a report, the phrase list, the flagged captions and two 0 to 1 scores."
            ),
            inputs=[
                io.MultiType.Input(
                    "text",
                    [io.String, LIST],
                    optional=True,
                    tooltip=(
                        "The captions, one per picture, such as Load Text Batch's text. A LIST "
                        "of captions works too. A blank entry counts as an empty caption. "
                        "Ignored while dataset is wired."
                    ),
                ),
                io.String.Input(
                    "subject",
                    default="[trigger]",
                    tooltip=(
                        "What every caption should name: `[trigger]`, `woman`, `ohwx man`. "
                        "Case ignored, whole words. Phrases holding it leave the phrase table; "
                        "coverage counts captions holding it. Empty = no subject check."
                    ),
                ),
                io.Int.Input(
                    "min_words",
                    default=5,
                    min=0,
                    max=1000,
                    tooltip=(
                        "0 = off; 5 = a caption under 5 words is flagged short and lowers "
                        "consistency."
                    ),
                ),
                io.Int.Input(
                    "max_words",
                    default=0,
                    min=0,
                    max=10000,
                    tooltip=(
                        "0 = off; 75 = a caption over 75 words is flagged long, which is about "
                        "where a CLIP text encoder cuts a caption off."
                    ),
                ),
                io.Int.Input(
                    "top",
                    default=40,
                    min=1,
                    max=1000,
                    tooltip=(
                        "Most phrases in the table and the phrases output, most frequent "
                        "first. 40 = the forty phrases in the most captions."
                    ),
                ),
                io.MultiType.Input(
                    "names",
                    [LIST, io.String],
                    optional=True,
                    tooltip=(
                        "One name per caption, in the captions' order, such as Load Image "
                        "Batch's filename_list, so flagged captions are named. A caption with "
                        "no name is shown by its position, counted from 0. Ignored while "
                        "dataset is wired."
                    ),
                ),
                DATASET.Input(
                    "dataset",
                    optional=True,
                    tooltip=(
                        "Unwired = captions come from text; wired = the captions and names of a "
                        "dataset, such as Load Dataset's, as edited upstream. A picture with no "
                        "caption counts as an empty one. text and names are ignored while it "
                        "is wired."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="report",
                    tooltip=(
                        "The figures as plain text: counts, word lengths, consistency, subject "
                        "coverage, the phrase table and every flagged caption with its reasons."
                    ),
                ),
                LIST.Output(
                    display_name="phrases",
                    tooltip=(
                        "The phrases or tags in two or more captions, most frequent first, up "
                        "to top. Empty when no phrase repeats."
                    ),
                ),
                LIST.Output(
                    display_name="flagged",
                    tooltip=(
                        "Names of the captions flagged empty, short, long, identical or "
                        "missing the subject, in caption order. Empty when none is."
                    ),
                ),
                io.Float.Output(
                    display_name="consistency",
                    tooltip=(
                        "0 to 1; 1 = even lengths, no repeats, nothing short. Length spread "
                        "costs up to half; identical and short captions cost their share."
                    ),
                ),
                io.Float.Output(
                    display_name="coverage",
                    tooltip=(
                        "0 to 1: share of captions holding the subject; 1 = every caption. "
                        "0 when subject is empty."
                    ),
                ),
            ],
            is_input_list=True,
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls, subject, min_words, max_words, top, text=None, names=None, dataset=None,
    ) -> io.NodeOutput:
        """Measure the dataset's captions, or else the text's, and publish the figures.

        Returns:
            The report, the phrases, the flagged names, consistency and coverage.

        Raises:
            ValueError: The dataset holds no picture, or neither input carried a caption.
        """
        source = None
        if dataset is not None:
            data = datasets.from_value(dataset, NAME)
            if not data.items:
                raise ValueError(EMPTY_DATASET)
            texts = datasets.captions(data)
            names = datasets.names(data)
            source = data.source or "unnamed"
        else:
            texts = _entries(text) if text is not None else []
        if not texts:
            raise ValueError(NOTHING)
        subject = str(_first(subject, "") or "").strip()
        min_words = max(0, int(_first(min_words, 5)))
        max_words = max(0, int(_first(max_words, 0)))
        top = max(1, int(_first(top, 40)))
        labels = _names(names, len(texts))

        stats = caption_stats.consistency(texts, min_words, max_words)
        table = caption_stats.phrases(texts, subject, top)
        hits = caption_stats.coverage(texts, subject) if subject else 0
        flagged = caption_stats.flags(texts, labels, subject, min_words, max_words)
        share = round(hits / len(texts), 3) if subject else 0.0

        figures = {
            "entries": len(texts),
            "subject": subject,
            "min_words": min_words,
            "max_words": max_words,
            "hits": hits,
            "named": len(labels) if names is not None else None,
            "dataset": source,
        }
        report = _report(stats, table, flagged, figures)
        _publish(stats, table, flagged, figures)
        return io.NodeOutput(
            report,
            [row["text"] for row in table["rows"]],
            [label for label, _ in flagged],
            float(stats["score"]),
            float(share),
        )


def _first(value, default):
    """The single value of an input that arrived as a list, or ``default`` when it is empty."""
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return default if value is None else value


def _entries(value) -> list:
    """Every entry of a list input, a LIST nested one deep read flat.

    Args:
        value: What the input received: one entry per item, or one entry holding a list.

    Returns:
        The entries in order, each a string or ``None``.
    """
    entries = []
    for item in value if isinstance(value, (list, tuple)) else [value]:
        for entry in item if isinstance(item, (list, tuple)) else [item]:
            entries.append(None if entry is None else str(entry))
    return entries


def _names(value, total: int) -> list:
    """The names wired beside the captions, one text of several lines read as one name per line.

    Args:
        value: What the names input received, or ``None`` when it is unwired.
        total: How many captions arrived.

    Returns:
        The names in order, which may be fewer or more than ``total``.
    """
    if value is None:
        return []
    found = _entries(value)
    if len(found) == 1 and total > 1 and found[0] and "\n" in found[0]:
        found = [line.strip() for line in found[0].splitlines() if line.strip()]
    return found


def _percent(share: float) -> str:
    """A 0 to 1 share written as a whole percentage."""
    return f"{round(share * 100)}%"


def _plural(count: int, word: str) -> str:
    """A count and a noun, the noun taking an ``s`` unless the count is one."""
    return f"{count} {word}{'' if count == 1 else 's'}"


def _phrase_table(table: dict) -> str:
    """The phrase rows as aligned text: share, caption count and phrase on each line."""
    rows = table["rows"]
    if not rows:
        return NO_PHRASES
    width = max(len("captions"), len(str(rows[0]["count"])))
    lines = [f"share  {'captions':>{width}}  phrase"]
    for row in rows:
        lines.append(f"{_percent(row['share']):>5}  {row['count']:>{width}}  {row['text']}")
    return "\n".join(lines)


def _flagged_text(flagged: list) -> str:
    """Each flagged caption on its own line, its name then its reasons."""
    if not flagged:
        return NONE_FLAGGED
    return "\n".join(f"{label}: {', '.join(reasons)}" for label, reasons in flagged)


def _frequent_named(table: dict) -> str:
    """The first frequent phrases with their shares, as one line of text."""
    named = [f"{row['text']} ({_percent(row['share'])})" for row in table["frequent"][:MAX_NAMED]]
    more = len(table["frequent"]) - len(named)
    return ", ".join(named) + (f" and {more} more" if more > 0 else "")


def _report(stats: dict, table: dict, flagged: list, figures: dict) -> str:
    """Every figure of a run as readable plain text.

    Args:
        stats: :func:`caption_stats.consistency`'s answer.
        table: :func:`caption_stats.phrases`'s answer.
        flagged: :func:`caption_stats.flags`'s answer.
        figures: The run's entry count, subject, word limits, subject hits, name count and
            dataset source, None for text.

    Returns:
        The report, sections separated by blank lines.
    """
    entries = figures["entries"]
    subject = figures["subject"]
    lines = [
        f"{_plural(entries, 'caption')}, {stats['empty']} empty, written as {table['style']}",
        f"Words: median {stats['words_median']:g}, spread {stats['words_spread']:g}",
    ]
    if figures["min_words"]:
        lines.append(f"Short, under {figures['min_words']} words: {stats['short']}")
    if figures["max_words"]:
        lines.append(f"Long, over {figures['max_words']} words: {stats['long']}")
    lines.append(f"Identical to an earlier caption: {stats['identical']}")
    if subject:
        lines.append(
            f"With {subject}: {figures['hits']} of {entries} "
            f"({_percent(figures['hits'] / entries)})"
        )
    lines.append(f"Consistency: {stats['score']:.3f}")
    named = figures["named"]
    if named is not None and named != entries:
        lines.append(
            f"Names: {named} for {_plural(entries, 'caption')}, paired by position; a caption "
            f"with no name is shown by its position from 0"
        )
    sections = ["\n".join(lines)]
    if table["frequent"]:
        sections.append(f"In more than half the captions: {_frequent_named(table)}")
    sections.append(
        f"Phrases, as a share of {_plural(table['captions'], 'caption')}:\n{_phrase_table(table)}"
    )
    sections.append(f"Flagged ({len(flagged)}):\n{_flagged_text(flagged)}")
    return "\n\n".join(sections)


def _summary(stats: dict, table: dict, flagged: list, entries: int) -> str:
    """The one line the report panel draws above the figures."""
    head = _plural(entries, "caption")
    if flagged:
        head += f", {len(flagged)} flagged"
    if table["frequent"]:
        row = table["frequent"][0]
        return f"{head}; '{row['text']}' is in {_percent(row['share'])} of them"
    if flagged:
        return head
    return f"{head}, consistency {stats['score']:.2f}"


def _publish(stats: dict, table: dict, flagged: list, figures: dict) -> None:
    """Report the run's figures, top phrases and flagged captions to the node's panel.

    Args:
        stats: :func:`caption_stats.consistency`'s answer.
        table: :func:`caption_stats.phrases`'s answer.
        flagged: :func:`caption_stats.flags`'s answer.
        figures: The run's entry count, subject, word limits, subject hits, name count and
            dataset source, None for text.
    """
    try:
        if not run_result.watching():
            return
        entries = figures["entries"]
        counts = {
            "captions": stats["captions"],
            "empty": stats["empty"],
            "median words": stats["words_median"],
            "short": stats["short"],
        }
        if figures["max_words"]:
            counts["long"] = stats["long"]
        counts["identical"] = stats["identical"]
        if figures["subject"]:
            counts["with subject"] = figures["hits"]
        counts["consistency"] = round(stats["score"] * 100)
        facts = {
            "style": table["style"],
            "subject": figures["subject"] or "none",
        }
        if figures["dataset"] is not None:
            facts["dataset"] = figures["dataset"]
        if table["frequent"]:
            facts["in over half"] = _frequent_named(table)
        named = figures["named"]
        if named is not None and named != entries:
            facts["names"] = f"{named} for {_plural(entries, 'caption')}"
        rows = table["rows"]
        tallies = [
            {"name": row["text"], "value": round(row["share"] * 100, 1)}
            for row in rows[: run_result.MAX_TALLIES]
        ]
        run_result.publish(
            status=run_result.WARNING if flagged or table["frequent"] else run_result.OK,
            summary=_summary(stats, table, flagged, entries),
            counts=counts,
            tallies=tallies,
            tallies_total=len(rows),
            facts=facts,
            bodies=[
                run_result.body("phrases", _phrase_table(table)),
                run_result.body("flagged", _flagged_text(flagged)),
            ],
        )
    except Exception as error:
        logger.debug("%s did not report its figures (%s)", NAME, error)
