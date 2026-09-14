"""Add, remove and replace the entries of one Noodle Soup Prompts terminology."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.compat.types import LIST
from ....modules.interface import library_report, run_result
from ....modules.prompt import nsp

logger = log.get_logger("nodes.text.terminology")

#: The delimiter the noodle output wraps a term in, which is what the parsers expect.
NOODLE_KEY = "__"

#: What each action does, in the order the widget offers them.
ADD = "add entries"
REMOVE = "remove entries"
REPLACE = "replace entries"
CREATE = "create the term"
DELETE = "delete the term"

ACTIONS = (ADD, REMOVE, REPLACE, CREATE, DELETE)


def entry_lines(text) -> list[str]:
    """The entries typed into a box, one per line, blank lines dropped."""
    return [line.strip() for line in str(text or "").splitlines() if line.strip()]


class NoodleSoupTermEdit(io.ComfyNode):
    """Change one terminology's entries in the stored pantry."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASNoodleSoupTermEdit",
            display_name="Noodle Soup Term Edit",
            search_aliases=[
                "WASNoodleSoupTermEdit",
                "Noodle Soup Term Edit",
                "nsp",
                "noodle soup",
                "pantry",
                "terminology",
                "add term",
            ],
            category="WAS Suite/Text/Terminology",
            description=(
                "Add words to a Noodle Soup Prompts terminology, take words out of one, or "
                "make a terminology of your own, so __your_term__ draws from your own list "
                "in Text Parse Noodle Soup Prompts and Prompt Parse. Entries you add are "
                "kept apart from the ones the bundled pantry supplied, and a later "
                "release of that pantry never removes them."
            ),
            inputs=[
                io.String.Input(
                    "term",
                    default="",
                    multiline=False,
                    placeholder="Eg: my-animals",
                    tooltip=(
                        "The terminology name, written without the __ markers. Eg: "
                        "my-animals, which a prompt then draws from as __my-animals__. A "
                        "name the pantry does not have yet is created."
                    ),
                ),
                io.Combo.Input(
                    "action",
                    options=list(ACTIONS),
                    tooltip=(
                        "`add entries` stores the words below that the term does not "
                        "already hold; `remove entries` takes them out and no refresh puts "
                        "them back; `replace entries` leaves the term holding exactly what "
                        "is typed; `create the term` makes an empty one; `delete the term` "
                        "removes it and everything in it."
                    ),
                ),
                io.String.Input(
                    "entries",
                    default="",
                    multiline=True,
                    placeholder="one word or phrase per line",
                    tooltip=(
                        "The words to add, remove or store, one per line. Blank lines are "
                        "skipped. Eg: a line reading 'red fox asleep in long grass'. "
                        "Ignored by `create the term` and `delete the term`."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="noodle",
                    tooltip=(
                        "The term wrapped in the default markers, __my-animals__, ready to "
                        "paste into a prompt for Text Parse Noodle Soup Prompts. Empty when "
                        "the term was deleted."
                    ),
                ),
                io.String.Output(
                    display_name="term",
                    tooltip=(
                        "The terminology name as it was stored, for wiring on to Noodle "
                        "Soup Pantry Export or another edit. Empty when the term was "
                        "deleted."
                    ),
                ),
                LIST.Output(
                    display_name="entries",
                    tooltip=(
                        "Every entry the term holds after the edit, in draw order. Text "
                        "List to Text turns it into one line per entry."
                    ),
                ),
                io.Int.Output(
                    display_name="entry_count",
                    tooltip="How many entries the term holds after the edit.",
                ),
            ],
            is_output_node=True,
            not_idempotent=True,
        )

    @classmethod
    def execute(cls, term="", action=ADD, entries="") -> io.NodeOutput:
        """Apply one edit and answer the term as it stands afterwards.

        Raises:
            ValueError: ``term`` is empty, or an action needing entries was given none.
        """
        name = str(term or "").strip()
        if not name:
            raise ValueError(
                "no terminology name was given, so there is nothing to change. Type the "
                "name a prompt draws it by, without the __ markers, such as my-animals"
            )
        wanted = entry_lines(entries)
        if action in (ADD, REMOVE, REPLACE) and not wanted:
            raise ValueError(
                f"'{action}' needs at least one entry, and the entries box is empty. Type "
                f"one word or phrase per line, or choose 'create the term' to make an "
                f"empty terminology"
            )

        if action == ADD:
            report = nsp.add_entries(name, wanted)
        elif action == REMOVE:
            report = nsp.remove_entries(name, wanted)
        elif action == REPLACE:
            report = nsp.set_term_entries(name, wanted)
        elif action == CREATE:
            report = nsp.add_term(name)
        elif action == DELETE:
            report = nsp.delete_term(name)
        else:
            raise ValueError(
                f"'{action}' is not something this node does, so nothing was changed. Pick "
                f"one of: {', '.join(ACTIONS)}"
            )

        held = [] if action == DELETE else nsp.term_entries(name)
        logger.info(
            "%s: %s, %s added, %s removed, %s left alone, %s entry(s) now",
            action,
            name,
            report["added"],
            report["removed"],
            report["skipped"],
            report["total"],
        )
        cls.report(action, name, report, held)
        noodle = "" if action == DELETE else f"{NOODLE_KEY}{name}{NOODLE_KEY}"
        return io.NodeOutput(
            noodle, "" if action == DELETE else name, held, len(held)
        )

    @classmethod
    def report(cls, action, name, report, held) -> None:
        """Draw what the edit changed on the node."""
        status = run_result.OK
        if not report["saved"]:
            status = run_result.ERROR
            summary = f"{name} could not be changed, so nothing was stored"
        elif action == DELETE:
            summary = f"deleted {name} and its {report['removed']} entry(s)"
        elif action == CREATE:
            summary = (
                f"{name} was already there, holding {report['total']} entry(s)"
                if report["skipped"]
                else f"created {name}, which holds nothing yet"
            )
        elif report["added"] or report["removed"]:
            summary = (
                f"{name}: {report['added']} entry(s) added, {report['removed']} removed"
            )
        else:
            status = run_result.WARNING
            summary = f"{name} was left as it was, {report['skipped']} entry(s) already there"
        library_report.publish(
            summary=summary,
            counts={
                "added": report["added"],
                "removed": report["removed"],
                "entries": report["total"],
                "yours": report["local"],
            },
            facts={"term": name, "action": action},
            lines=held,
            total=len(held),
            status=status,
        )
