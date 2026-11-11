"""Check that text is JSON, and repair it in place where it can be."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.interface import run_result
from ....modules.util import json_repair

logger = log.get_logger("nodes.text.parse")

#: The node's name in messages.
NODE = "JSON Validate"

#: Characters of removed text an item's note quotes.
NOTE_CHARS = 24


class JSONValidate(io.ComfyNode):
    """Strict JSON check, with a repair that changes only the spots that break it."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASJSONValidate",
            display_name="JSON Validate",
            search_aliases=[
                "WASJSONValidate",
                "JSON Validate",
                "JSON repair",
                "repair JSON",
                "fix JSON",
                "validate JSON",
                "JSON lint",
                "LLM JSON",
            ],
            category="WAS Suite/Text/Parse",
            is_output_node=True,
            description=(
                "Check that text is valid JSON, such as a language model's answer, before another "
                "node reads it. Repair fixes common faults in place: unescaped quotes inside a "
                "string, missing or extra commas, comments, single quotes, unclosed brackets, and "
                "words or a code fence around the JSON. The panel marks every change."
            ),
            inputs=[
                io.String.Input(
                    "text",
                    default="",
                    multiline=True,
                    placeholder='{"caption": "a cat", "aspect_ratio": "16:9"}',
                    tooltip=(
                        'The JSON to check, such as an LLM Query answer: `{"caption": "a cat"}`. '
                        "Up to 16 million characters."
                    ),
                ),
                io.Boolean.Input(
                    "repair",
                    default=True,
                    tooltip=(
                        'true = fix only what breaks the JSON, as `"Closed"` inside a string '
                        'becoming `\\"Closed\\"`, plus extra or missing commas, comments, single '
                        "quotes, unclosed brackets and words around it; false = check only."
                    ),
                ),
                io.Boolean.Input(
                    "stop_if_invalid",
                    default=True,
                    tooltip=(
                        "true = JSON that is still invalid stops the run, naming the line and "
                        "column; false = the text is passed on as it came, with valid `false` "
                        "and error saying why."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="json",
                    tooltip=(
                        "The JSON: as it came when valid, or with only the repaired spots "
                        "changed. For Extract Text from JSON or any node that reads JSON."
                    ),
                ),
                io.Boolean.Output(
                    display_name="valid",
                    tooltip=(
                        "true when json is valid JSON; false only when stop_if_invalid is off "
                        "and the text could not be repaired."
                    ),
                ),
                io.Boolean.Output(
                    display_name="repaired",
                    tooltip="true when a repair changed the text; the panel marks each change.",
                ),
                io.String.Output(
                    display_name="error",
                    tooltip=(
                        "Why the text is not JSON, with its line and column, such as `Expecting "
                        "',' delimiter at line 2, column 341`. Empty when json is valid."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, text, repair, stop_if_invalid) -> io.NodeOutput:
        """Check the text, repair it when asked, and report what was found.

        Args:
            text: The text to check.
            repair: Whether JSON that fails is repaired.
            stop_if_invalid: Whether JSON that still fails stops the run.

        Returns:
            The JSON, whether it is valid, whether it was repaired, and the fault or ``""``.

        Raises:
            ValueError: The text is not JSON and ``stop_if_invalid`` is on.
        """
        text = "" if text is None else str(text)
        value, fault = json_repair.check(text)
        if fault is None:
            _report_valid(text, value)
            return io.NodeOutput(text, True, False, "")
        reason = ""
        if repair:
            try:
                fixed = json_repair.repair(text)
            except json_repair.Unrepairable as error:
                reason = str(error)
            else:
                logger.info(
                    "%s: repaired with %d change%s (%s)",
                    NODE, len(fixed.changes), "" if len(fixed.changes) == 1 else "s",
                    ", ".join(f"{count} {kind}" for kind, count in _tally(fixed.changes)),
                )
                _report_repaired(text, fixed, fault)
                return io.NodeOutput(fixed.text, True, True, "")
        error = f"Not valid JSON: {fault.where()}"
        if reason:
            error += f"; repair could not fix it: {reason}"
        _report_invalid(text, fault, reason, repair)
        if stop_if_invalid:
            hint = (
                "Fix the text at that spot, or turn stop_if_invalid off to pass it on with valid false."
                if repair else "Turn repair on, or fix the text at that spot."
            )
            raise ValueError(f"{NODE}: {error}. {hint}")
        return io.NodeOutput(text, False, False, error)


def _shape(value) -> str:
    """What a JSON value is, in a few words."""
    if isinstance(value, dict):
        return f"an object of {len(value):,} key{'' if len(value) == 1 else 's'}"
    if isinstance(value, list):
        return f"an array of {len(value):,} item{'' if len(value) == 1 else 's'}"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, bool) or value is None:
        return "null" if value is None else ("true" if value else "false")
    return "a number"


def _tally(changes) -> list[tuple[str, int]]:
    """How many changes of each kind, most first."""
    counts = {}
    for change in changes:
        counts[change.kind] = counts.get(change.kind, 0) + 1
    order = {kind: index for index, kind in enumerate(json_repair.KINDS)}
    return sorted(counts.items(), key=lambda pair: (-pair[1], order.get(pair[0], len(order))))


def _cut(text: str, limit: int) -> str:
    """The text on one line, shortened to ``limit`` characters."""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _lines(text: str) -> int:
    return text.count("\n") + 1 if text else 0


def _report_valid(text: str, value) -> None:
    """Publish a run that found valid JSON."""
    try:
        if not run_result.watching():
            return
        run_result.publish(
            status=run_result.OK,
            summary=f"Valid JSON, {_shape(value)}.",
            counts={"characters": len(text), "lines": _lines(text)},
            bodies=[run_result.body("json", text)],
        )
    except Exception as error:
        logger.debug("%s did not report its run (%s)", NODE, error)


def _report_repaired(text: str, fixed, fault) -> None:
    """Publish a repair: what was wrong, every change marked, and both texts."""
    try:
        if not run_result.watching():
            return
        changes = fixed.changes
        tallies = _tally(changes)
        items = []
        for change, (start, end) in list(zip(changes, fixed.spans))[: run_result.MAX_ITEMS]:
            note = f"line {fixed.text.count(chr(10), 0, start) + 1}: {change.kind}"
            room = min(NOTE_CHARS, run_result.MAX_LABEL_CHARS - len(note) - len(", was ''"))
            if change.end > change.start and not change.text and room >= 8:
                note += f", was '{_cut(text[change.start:change.end], room)}'"
            items.append(run_result.excerpt(fixed.text, start, end, note=note))
        first = fixed.spans[0][0] if fixed.spans else 0
        bodies = [
            run_result.body(
                "repaired", fixed.text, marks=fixed.spans, marks_total=len(fixed.spans),
                start=run_result.window(len(fixed.text), first)[0],
            ),
            run_result.body(
                "as it came", text, marks=[(change.start, change.end) for change in changes],
                marks_total=len(changes),
                start=run_result.window(len(text), fault.position or 0)[0],
            ),
        ]
        run_result.publish(
            status=run_result.WARNING,
            summary=(
                f"Repaired with {len(changes):,} change{'' if len(changes) == 1 else 's'}, "
                f"now valid JSON, {_shape(fixed.value)}."
            ),
            counts={"changes": len(changes), "characters": len(fixed.text), "lines": _lines(fixed.text)},
            tallies=[{"name": kind, "value": count} for kind, count in tallies[: run_result.MAX_TALLIES]],
            tallies_total=len(tallies),
            facts={"was": fault.where()},
            items=items,
            items_total=len(changes),
            bodies=bodies,
        )
    except Exception as error:
        logger.debug("%s did not report its run (%s)", NODE, error)


def _report_invalid(text: str, fault, reason: str, repair: bool) -> None:
    """Publish a run that ends with text that is not JSON."""
    try:
        if not run_result.watching():
            return
        facts = {"fault": fault.where(), "repair": reason if repair else "off"}
        items = []
        marks = []
        if fault.position is not None and text:
            end = min(fault.position + 1, len(text))
            items.append(run_result.excerpt(
                text, fault.position, end, note=f"line {fault.line}, column {fault.column}",
            ))
            marks.append((fault.position, end))
        bodies = []
        if text:
            bodies.append(run_result.body(
                "text", text, marks=marks, start=run_result.window(len(text), fault.position or 0)[0],
            ))
        run_result.publish(
            status=run_result.ERROR,
            summary=f"Not valid JSON: {fault.where()}.",
            counts={"characters": len(text), "lines": _lines(text)},
            facts={name: value for name, value in facts.items() if value},
            items=items,
            bodies=bodies,
        )
    except Exception as error:
        logger.debug("%s did not report its run (%s)", NODE, error)
