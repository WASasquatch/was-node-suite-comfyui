"""Choose words out of the stored Noodle Soup Prompts terminology."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.compat.lists import require_values
from ....modules.compat.types import LIST
from ....modules.interface import library_report, run_result
from ....modules.prompt import nsp, nsp_picks

logger = log.get_logger("nodes.text.terminology")

#: Where a user is pointed when the pantry holds nothing to pick from.
NO_PANTRY = (
    "the pantry holds no terminology yet, so nothing could be picked. The bundled pantry "
    "seeds the database on first start; restart ComfyUI to read it in"
)


class NoodleSoupPick(io.ComfyNode):
    """Answer the chosen words of the stored terminology pantry."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASNoodleSoupPick",
            display_name="Noodle Soup Pick",
            search_aliases=[
                "WASNoodleSoupPick",
                "Noodle Soup Pick",
                "Noodle Soup Term List",
                "nsp",
                "noodle soup",
                "pantry",
                "terminology",
                "pick entries",
                "choose words",
            ],
            category="WAS Suite/Text/Terminology",
            description=(
                "Choose words out of the Noodle Soup Prompts pantry and answer them as a "
                "list, as text and as a count. The browser panel ticks them; the picked box "
                "holds one pick per line either way. With nothing picked the terminology "
                "names go out instead, which is how the pantry is read. The yours figure "
                "counts what was added from a node or brought in from a file, apart from "
                "what the published pantry supplied."
            ),
            inputs=[
                io.String.Input(
                    "picked",
                    default="",
                    multiline=True,
                    tooltip=(
                        "One pick per line, as `term: word`. `artist: Greg Rutkowski` takes "
                        "one word, spelled as the pantry holds it; `artist: *` takes the "
                        "whole terminology, and so does a line naming one on its own. Blank "
                        "and `#` lines are ignored, a word named twice goes out once, and "
                        "line order is output order."
                    ),
                ),
                io.Int.Input(
                    "limit",
                    default=0,
                    min=0,
                    max=100000,
                    step=1,
                    tooltip=(
                        "How many words go out at most, counting from the first. 0 = every "
                        "one; 50 = the first 50. `artist` alone holds over 2000."
                    ),
                ),
                io.String.Input(
                    "term",
                    default="",
                    multiline=False,
                    optional=True,
                    placeholder="Eg: my-animals",
                    tooltip=(
                        "One terminology taken whole, added to what is picked above, for "
                        "wiring the term output of Noodle Soup Term Edit straight in. Eg: "
                        "`my-animals`. Empty takes only what is picked."
                    ),
                ),
            ],
            outputs=[
                LIST.Output(
                    display_name="entries",
                    tooltip=(
                        "The picked words, in the order the picked box lists them, a `*` "
                        "line expanded in pantry order. With nothing picked, the "
                        "terminology names. Text List to Text joins them with a comma."
                    ),
                ),
                io.String.Output(
                    display_name="text",
                    tooltip=(
                        "The same words, one per line. Feeds the entries box of Noodle Soup "
                        "Term Edit, and Text Random Line for one word a run."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip="How many words went out, after limit cut the list.",
                ),
                io.Int.Output(
                    display_name="yours",
                    tooltip=(
                        "How many of them were added from a node or brought in from a file "
                        "rather than supplied by the published pantry."
                    ),
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def fingerprint_inputs(cls, picked="", limit=0, term=""):
        """The pantry's own version stamp.

        Returns:
            The stamp, or NaN when the store could not be reached, which never compares
            equal and so runs the node again.
        """
        try:
            return nsp.generation()
        except Exception as error:
            logger.debug("the pantry version stamp could not be read (%s)", error)
            return float("nan")

    @classmethod
    def execute(cls, picked="", limit=0, term="") -> io.NodeOutput:
        counts = nsp.terms()
        wanted = str(picked or "")
        named = str(term or "").strip()
        if named:
            wanted = f"{wanted}\n{named}" if wanted.strip() else named

        picks, overflow = nsp_picks.parse(wanted, counts)
        if picks:
            reading = nsp_picks.resolve(picks, counts)
            entries, mine = reading.entries, reading.own
            listing = "words"
        else:
            reading = None
            entries, mine = nsp_picks.term_names(counts, nsp.local_counts())
            listing = "terminologies"

        total = len(entries)
        cut = int(limit or 0)
        shown = entries[:cut] if cut > 0 else list(entries)
        if reading is None:
            lines = [f"{name} {counts[name]}" for name in shown]
        else:
            lines = list(shown)

        status, summary, facts = cls.state(counts, reading, picks, total, len(shown), overflow)
        tiles = {"picked": len(shown), "total": total, "yours": mine}
        if reading is not None and reading.missing:
            tiles["missing"] = len(reading.missing)
        cls.report(summary, tiles, total, facts, lines, listing, status)
        require_values(shown, cls.refusal(counts, reading))
        return io.NodeOutput(shown, "\n".join(lines), len(shown), mine)

    @classmethod
    def state(cls, counts, reading, picks, total, shown, overflow):
        """What the run came to, as a status, one line about it and the rows beside it.

        Args:
            counts: ``{term: entries}`` as the pantry holds it.
            reading: The ``nsp_picks.Resolved`` for the picks, or None when none were made.
            picks: The picks that were read.
            total: How many words the picks came to, before ``limit``.
            shown: How many went out.
            overflow: Lines of the picked box past the read bound.

        Returns:
            ``(status, summary, facts)``.
        """
        facts = {"pantry": f"{len(counts)} terminology(s), {sum(counts.values())} word(s)"}
        if reading is None:
            return (
                run_result.WARNING,
                f"nothing is picked, so the {len(counts)} terminology name(s) went out. Open "
                f"a terminology below and tick its words",
                facts,
            )

        facts["picked"] = f"{len(picks)} line(s) naming {total} word(s)"
        went = f"{shown} word(s)" if shown == total else f"{shown} of {total} word(s)"
        if shown != total:
            facts["limit"] = f"the first {shown} of {total} word(s) went out"
        notes = []
        if reading.missing:
            notes.append(f"{len(reading.missing)} no longer in the pantry")
        if reading.unknown:
            notes.append(f"{', '.join(reading.unknown)} names no terminology")
        if reading.empty:
            notes.append(f"{', '.join(reading.empty)} holds no word")
        if overflow:
            notes.append(f"{overflow} line(s) past the {nsp_picks.MAX_LINES} read were ignored")
        if reading.repeats:
            facts["repeats"] = f"{reading.repeats} line(s) named a word already taken"
        if notes:
            facts["warning"] = "; ".join(notes)
            return run_result.WARNING, f"{went} went out, {'; '.join(notes)}", facts
        return run_result.OK, f"{went} went out, picked on {len(picks)} line(s)", facts

    @classmethod
    def refusal(cls, counts, reading) -> str:
        """The message for a run that produced no word at all.

        Args:
            counts: ``{term: entries}`` as the pantry holds it.
            reading: The ``nsp_picks.Resolved`` for the picks, or None.

        Returns:
            One sentence naming the cause and the fix.
        """
        if not counts:
            return NO_PANTRY
        if reading is not None and (reading.unknown or reading.empty):
            named = ", ".join([*reading.unknown, *reading.empty])
            return (
                f"nothing was picked: {named} holds no word this pantry knows. Tick a word "
                f"below, or take those lines out of the picked box"
            )
        return (
            "nothing was picked, so there is no word to answer with. Tick a word below, or "
            "write one pick per line in the picked box as `term: word`"
        )

    @classmethod
    def report(cls, summary, tiles, total, facts, lines, listing, status) -> None:
        """Draw what went out on the node."""
        library_report.publish(
            summary=summary,
            counts=tiles,
            facts=facts,
            lines=lines,
            listing=listing,
            total=total,
            status=status,
        )
