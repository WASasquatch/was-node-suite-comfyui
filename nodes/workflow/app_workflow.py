"""Running a saved app workflow as one node inside another graph."""

from __future__ import annotations

import json

from comfy_api.latest import io, ui

from ...modules.compat.types import DICT

NODE_NAME = "App Workflow"

#: Shown in the menu when the workflows directory holds no app workflow.
NONE_FOUND = "(no app workflow saved)"

#: Exposed inputs a wire can be fed into, and results that can leave on a wire.
SLOTS = 4

IN_TIP = (
    "Value for the {ordinal} input the workflow exposes, replacing what it saved; any type. "
    "Choosing a workflow renames this socket to that input, such as red_offset, and narrows "
    "its type to match. A file input takes an IMAGE here instead of reading the file."
)
OUT_TIP = (
    "The {ordinal} result the workflow presents; any type. It carries whatever feeds that "
    "node, so a workflow ending in a Preview Image answers the IMAGE itself, and choosing a "
    "workflow renames this socket to what it carries."
)

def app_choices() -> list[str]:
    """Every app workflow that can be run, or a single placeholder when there are none."""
    from ...modules.workflow import apps

    return apps.discover() or [NONE_FOUND]


class AppWorkflow(io.ComfyNode):
    """Run a saved app workflow, feeding its exposed inputs and taking its results."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAppWorkflow",
            display_name=NODE_NAME,
            search_aliases=[
                "WASAppWorkflow",
                "App Workflow", "run workflow", "workflow player", "subworkflow",
                "nested workflow", "call workflow", "app", "linear",
            ],
            category="WAS Suite/Workflow",
            description=(
                "Run a workflow saved in app mode as a single node. Its exposed inputs "
                "become values this node sets, and each result it presents leaves on a "
                "wire, so a whole saved graph can be reused inside a larger one."
            ),
            inputs=[
                io.Combo.Input(
                    "app",
                    options=app_choices(),
                    tooltip=(
                        "Which saved app workflow to run; a name ending .app.json from the "
                        "workflows directory, such as upscale.app.json."
                    ),
                ),
                io.String.Input(
                    "overrides",
                    multiline=True,
                    default="{}",
                    tooltip=(
                        "Values for the workflow's exposed inputs, as JSON keyed on the "
                        'input name: {"steps": 30, "text": "a red car"}. Anything left out '
                        "keeps the value the workflow was saved with, and the widgets below "
                        "are sent through here."
                    ),
                ),
                io.AnyType.Input(
                    "input_1", optional=True, tooltip=IN_TIP.format(n=1, ordinal="first"),
                ),
                io.AnyType.Input(
                    "input_2", optional=True, tooltip=IN_TIP.format(n=2, ordinal="second"),
                ),
                io.AnyType.Input(
                    "input_3", optional=True, tooltip=IN_TIP.format(n=3, ordinal="third"),
                ),
                io.AnyType.Input(
                    "input_4", optional=True, tooltip=IN_TIP.format(n=4, ordinal="fourth"),
                ),
            ],
            outputs=[
                DICT.Output(
                    display_name="exposed",
                    tooltip=(
                        "What the workflow offers, as one value; DICT. Holds its exposed "
                        "input names, its result count and the node count it ran."
                    ),
                ),
                io.AnyType.Output(
                    display_name="output_1", tooltip=OUT_TIP.format(n=1, ordinal="first"),
                ),
                io.AnyType.Output(
                    display_name="output_2", tooltip=OUT_TIP.format(n=2, ordinal="second"),
                ),
                io.AnyType.Output(
                    display_name="output_3", tooltip=OUT_TIP.format(n=3, ordinal="third"),
                ),
                io.AnyType.Output(
                    display_name="output_4", tooltip=OUT_TIP.format(n=4, ordinal="fourth"),
                ),
            ],
            enable_expand=True,
        )

    @classmethod
    def execute(cls, app, overrides="{}", **wired) -> io.NodeOutput:
        """Convert the chosen workflow, apply the given values and expand into it.

        Args:
            app: Name of a saved app workflow.
            overrides: JSON object of values for its exposed inputs.
            **wired: ``input_1`` to ``input_4``, feeding exposed inputs in order.

        Returns:
            What the workflow exposes, then one value per result it presents.
        """
        from ...modules.workflow import apps, convert, expand

        if app == NONE_FOUND:
            raise ValueError(
                "no app workflow was found. Save a workflow with app mode on, under a name "
                "ending .app.json, then refresh the node definitions"
            )

        workflow = apps.load(app)
        exposure = apps.exposure(workflow)
        prompt, origins, missing = convert.prompt_with_origins(workflow)
        if missing:
            raise ValueError(
                f"{app} uses {len(missing)} node type(s) this ComfyUI does not have: "
                f"{', '.join(sorted(missing))}. Install them, or open the workflow and "
                f"replace them"
            )
        if not exposure.outputs:
            raise ValueError(
                f"{app} presents no results, so it has nothing to answer with. Open it, "
                f"turn on app mode and mark at least one node's result as shown"
            )

        assignments = cls.assignments(exposure, origins, cls.decoded(overrides, app), prompt)
        prompt, _ = expand.overridden(prompt, assignments)
        feeds, swaps = cls.feeds(exposure, origins, wired, prompt)
        graph, links = expand.build(prompt, exposure.outputs[:SLOTS], feeds, swaps)

        exposed = {
            "app": app,
            "inputs": [entry.label for entry in exposure.inputs],
            "results": len(exposure.outputs),
            "nodes": len(graph),
        }
        answers = list(links) + [None] * (SLOTS - len(links))
        return io.NodeOutput(
            exposed, *answers[:SLOTS], expand=graph,
            ui=ui.PreviewText(cls.readout(app, exposure)),
        )

    @classmethod
    def decoded(cls, overrides, app):
        """Read the overrides widget.

        Args:
            overrides: The widget's text.
            app: Name of the workflow, for the message when the text will not read.

        Returns:
            The decoded object, empty when the text is blank.

        Raises:
            ValueError: The text is not a JSON object.
        """
        text = (overrides or "").strip()
        if not text:
            return {}
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"overrides for {app} is not readable JSON: {error}. Write an object such "
                f'as {{"steps": 30}}, or leave it as {{}}'
            ) from error
        if not isinstance(decoded, dict):
            raise ValueError(
                f"overrides for {app} is {type(decoded).__name__}, not an object. Write "
                f'{{"steps": 30}} rather than a bare value'
            )
        return decoded

    @classmethod
    def assignments(cls, exposure, origins, values, prompt):
        """Turn named overrides into per-node widget assignments.

        Args:
            exposure: What the workflow exposes.
            origins: Where each converted node came from.
            values: ``{name_or_widget_id: value}``.
            prompt: The converted workflow, for the type each target declares.

        Returns:
            ``{(api_id, widget_name): value}``, each value brought to the declared type.

        Raises:
            ValueError: A name matches nothing the workflow exposes.
        """
        from ...modules.workflow import apps, convert

        by_name = {}
        for entry in exposure.inputs:
            by_name.setdefault(entry.label, entry)
            by_name.setdefault(entry.widget_id, entry)
            by_name.setdefault(entry.widget, entry)

        assignments = {}
        for name, value in values.items():
            entry = by_name.get(name)
            if entry is None:
                offered = ", ".join(sorted({e.label for e in exposure.inputs})) or "nothing"
                raise ValueError(
                    f"{name!r} is not an input the workflow exposes. It offers: {offered}"
                )
            for api_id in apps.targets(entry, origins):
                config = convert.declared_input(prompt[api_id]["class_type"], entry.widget)
                assignments[(api_id, entry.widget)] = convert.coerced(
                    value, config, f"overrides[{name!r}]"
                )
        return assignments

    @classmethod
    def feeds(cls, exposure, origins, wired, prompt):
        """Bind wired values to the exposed inputs they stand in for.

        Args:
            exposure: What the workflow exposes.
            origins: Where each converted node came from.
            wired: ``{"input_N": value}``, absent or ``None`` where nothing is wired.
            prompt: The converted workflow, for the type each target declares.

        Returns:
            ``({(api_id, input_name): value}, {(api_id, output_slot): value})``: values a
            widget takes, and values standing in for a node's whole output.
        """
        from ...modules.workflow import apps, convert, expand

        bound, swaps = {}, {}
        sockets = apps.socketed(exposure, prompt, origins, SLOTS)
        arriving = {}
        for place, index in enumerate(sockets):
            arriving[index] = (f"input_{place + 1}", wired.get(f"input_{place + 1}"))
        # A widget turned into a socket arrives under the name the workflow exposes it as.
        for index, entry in enumerate(exposure.inputs):
            if index not in arriving and entry.label in wired:
                arriving[index] = (entry.label, wired[entry.label])

        for index, (where, value) in arriving.items():
            if value is None:
                continue
            entry = exposure.inputs[index]
            # The socket says what it carries, so a value is not read to find out what it is.
            kinds = [one for one in (sockets.get(index) or "").split(",") if one]
            carried = expand.matching(value, kinds)
            for api_id in apps.targets(entry, origins):
                class_type = prompt[api_id]["class_type"]
                # A wire onto a menu of filenames stands in for what that node reads.
                slot = convert.output_slot(class_type, carried) if carried else None
                if slot is not None:
                    swaps[(api_id, slot)] = value
                    continue
                config = convert.declared_input(class_type, entry.widget)
                bound[(api_id, entry.widget)] = convert.coerced(
                    value, config, f"{where}, feeding {entry.label}"
                )
        return bound, swaps

    @classmethod
    def readout(cls, app, exposure):
        """Lines naming the workflow, what it takes and what it answers."""
        lines = [app, ""]
        if exposure.inputs:
            lines.append(f"inputs ({len(exposure.inputs)})")
            for index, entry in enumerate(exposure.inputs):
                slot = f"input_{index + 1}" if index < SLOTS else "overrides only"
                lines.append(f"  {entry.label}  [{slot}]")
        else:
            lines.append("inputs: none exposed")
        lines.append("")
        lines.append(f"results ({len(exposure.outputs)})")
        for index in range(len(exposure.outputs)):
            reach = f"output_{index + 1}" if index < SLOTS else "not reachable"
            lines.append(f"  result {index + 1}  [{reach}]")
        if exposure.panels:
            lines.append("")
            lines.append(f"panels skipped: {len(exposure.panels)}")
        return "\n".join(lines)
