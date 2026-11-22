"""Apply a language model's code reply to the code it was asked to change."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.interface import run_result
from ....modules.util import code_edits

logger = log.get_logger("nodes.text.parse")

#: The node's name in messages.
NODE = "Apply Code Edits"


class ApplyCodeEdits(io.ComfyNode):
    """Whole files and SEARCH/REPLACE blocks from a reply, applied to a code bundle."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASApplyCodeEdits",
            display_name="Apply Code Edits",
            search_aliases=[
                "WASApplyCodeEdits",
                "Apply Code Edits",
                "SEARCH/REPLACE",
                "search replace blocks",
                "apply patch",
                "code edit",
                "LLM code",
                "diff",
            ],
            category="WAS Suite/Text/Parse",
            description=(
                "Apply a language model's answer to the code it was asked to change: whole files "
                "under their paths, or SEARCH/REPLACE blocks that change only the lines they name. "
                "Hands on the new code, a diff and a report of any edit that did not fit, for the "
                "model to try again."
            ),
            inputs=[
                io.String.Input(
                    "reply",
                    default="",
                    multiline=True,
                    placeholder="app.py\n```python\n<<<<<<< SEARCH\nold lines\n=======\nnew lines\n>>>>>>> REPLACE\n```",
                    tooltip=(
                        "The model's answer. `app.py` on a line, then a code block = that whole file; "
                        "a `<<<<<<< SEARCH` ... `=======` ... `>>>>>>> REPLACE` block = those lines "
                        "changed. Words around them are ignored."
                    ),
                ),
                io.String.Input(
                    "code",
                    default="",
                    multiline=True,
                    optional=True,
                    tooltip=(
                        "The code before the edits: files as `path` lines each followed by a code "
                        "block, as this node hands them on, or one file as bare text. Empty = no "
                        "code yet, so the reply's whole files start it."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="code",
                    tooltip=(
                        "The code after the edits, each file as its path then a code block; one "
                        "unnamed file stays bare text. Unchanged where nothing applied."
                    ),
                ),
                io.String.Output(
                    display_name="diff",
                    tooltip=(
                        "A unified diff of every changed file, `--- a/app.py` then `+++ b/app.py`. "
                        "Empty when nothing changed."
                    ),
                ),
                io.String.Output(
                    display_name="report",
                    tooltip=(
                        "`Applied 3 of 4 edits to 2 files.`, then each edit that did not fit with "
                        "the closest lines found, to send back to the model."
                    ),
                ),
                io.Boolean.Output(
                    display_name="all_applied",
                    tooltip=(
                        "true when no edit failed, a reply with no edits included; false when at "
                        "least one block found nothing to replace."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, reply, code="") -> io.NodeOutput:
        """Apply the reply and report what landed.

        Args:
            reply: The model's answer.
            code: The code before the edits.

        Returns:
            The edited code, the diff, the report and whether every edit applied.

        Raises:
            ValueError: The code or reply is over the size limit.
        """
        try:
            result = code_edits.apply_reply(code or "", reply or "")
        except ValueError as error:
            raise ValueError(f"{NODE}: {error}") from error
        report = code_edits.report_text(result)
        out = code_edits.format_bundle(result.files)
        logger.info("%s: %s", NODE, report.split("\n", 1)[0])
        _publish(result, report, out)
        return io.NodeOutput(out, result.diff, report, result.failed == 0)


def _publish(result, report: str, code: str) -> None:
    """Publish the run for the panel: tallies by how each edit matched, the report, diff and code."""
    try:
        if not run_result.watching():
            return
        total = len(result.outcomes)
        if total and not result.failed:
            status = run_result.OK
        elif result.applied:
            status = run_result.WARNING
        else:
            status = run_result.ERROR
        tallies: dict[str, int] = {}
        for outcome in result.outcomes:
            name = outcome.how if outcome.applied else "not applied"
            name = name.split(",", 1)[0]
            tallies[name] = tallies.get(name, 0) + 1
        added = sum(1 for line in result.diff.split("\n") if line.startswith("+") and not line.startswith("+++"))
        removed = sum(1 for line in result.diff.split("\n") if line.startswith("-") and not line.startswith("---"))
        bodies = [run_result.body("report", report)]
        if result.diff:
            bodies.append(run_result.body("diff", result.diff))
        bodies.append(run_result.body("code", code))
        run_result.publish(
            status=status,
            summary=report.split("\n", 1)[0],
            counts={
                "edits": total, "applied": result.applied, "failed": result.failed,
                "files": len(result.files), "lines added": added, "lines removed": removed,
            },
            tallies=[
                {"name": name, "value": count}
                for name, count in sorted(tallies.items(), key=lambda pair: -pair[1])[: run_result.MAX_TALLIES]
            ],
            tallies_total=len(tallies),
            bodies=bodies,
        )
    except Exception as error:
        logger.debug("%s did not report its run (%s)", NODE, error)
