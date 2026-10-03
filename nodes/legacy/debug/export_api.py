"""Write the prompt API JSON of the running workflow to disk."""

from __future__ import annotations

import json
import os
import re

from comfy_api.latest import io, ui

from ....modules.io import rooted
from ....modules.log import get_logger
from ....modules.util import sandbox

REQUIRES = "debug"

logger = get_logger("nodes.debug")


def parse_tokens(value, tokens):
    """Expand ``[token]`` patterns in every string of a nested prompt structure.

    Args:
        value: A dict, a list, a string, or any other JSON value.
        tokens: The :class:`~modules.state.tokens.TextTokens` doing the substitution.

    Returns:
        The same structure with every string expanded.
    """
    if isinstance(value, dict):
        return {key: parse_tokens(entry, tokens) for key, entry in value.items()}
    if isinstance(value, list):
        return [parse_tokens(entry, tokens) for entry in value]
    if isinstance(value, str):
        return tokens.parseTokens(value)
    return value


class ExportAPI(io.ComfyNode):
    """Serialise the running prompt to a numbered JSON file under ComfyUI's output tree."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Export API",
            display_name="Export API",
            search_aliases=["Export API", "prompt json", "export prompt"],
            category="WAS Suite/Debug",
            description=(
                "Deprecated: use ComfyUI's own Workflow > Export (API) menu item instead, "
                "which saves the same document without a node in the graph. Writes the "
                "prompt API JSON of the running workflow to a numbered file. The folder has "
                "to be one this pack may write to: ComfyUI's output and temp folders, the "
                "pack's own folder, or a folder listed under paths.allow_write in "
                "config.yaml."
            ),
            inputs=[
                io.Combo.Input(
                    "save_prompt_api",
                    options=["true", "true"],
                    tooltip=(
                        "Whether to write the file. Both entries of this menu read `true`, so "
                        "there is no way to turn writing off; the JSON is always written and "
                        "always printed to the console."
                    ),
                ),
                io.Combo.Input(
                    "root",
                    options=rooted.options(),
                    tooltip=(
                        "Which folder the JSON lands in: ComfyUI's own 'output' or 'temp', "
                        "or any folder added under paths.allow_write in config.yaml, listed "
                        "by its own name. filename_prefix names the part below it, so "
                        "'[time(%Y-%m-%d)]/prompt' files each day's under a dated folder."
                    ),
                ),
                io.String.Input(
                    "filename_prefix",
                    default="ComfyUI_Prompt",
                    tooltip="The name part of the file, before the number.",
                ),
                io.String.Input(
                    "filename_delimiter",
                    default="_",
                    tooltip=(
                        "What sits between the name and the number: "
                        "'ComfyUI_Prompt_0001.json' with the default."
                    ),
                ),
                io.Int.Input(
                    "filename_number_padding",
                    default=4,
                    min=2,
                    max=9,
                    step=1,
                    tooltip=(
                        "How many digits the number is padded to with leading zeros: 4 gives "
                        "'_0001', 2 gives '_01'."
                    ),
                ),
                io.Boolean.Input(
                    "parse_text_tokens",
                    default=False,
                    tooltip=(
                        "Whether to expand '[token]' patterns in every string of the saved "
                        "document, so a prompt containing '[time(%Y-%m-%d)]' is recorded as "
                        "the date it ran. Off, the widget values are saved exactly as typed, "
                        "which is what a reloadable workflow needs."
                    ),
                ),
            ],
            outputs=[],
            hidden=[io.Hidden.prompt],
            is_output_node=True,
            is_deprecated=True,
        )

    @classmethod
    def execute(
        cls,
        save_prompt_api,
        filename_prefix,
        filename_delimiter,
        filename_number_padding,
        parse_text_tokens,
        root=rooted.DEFAULT,
    ) -> io.NodeOutput:
        from ....modules.state.tokens import TextTokens

        number_padding = filename_number_padding if filename_number_padding > 1 else 4
        tokens = TextTokens()
        below, _, filename_prefix = filename_prefix.replace("\\", "/").rpartition("/")
        directory = rooted.destination(root, below)
        directory.mkdir(parents=True, exist_ok=True)

        pattern = f"{re.escape(filename_prefix)}{re.escape(filename_delimiter)}(\\d{{{number_padding}}})"
        existing_counters = [
            int(re.search(pattern, filename).group(1))
            for filename in os.listdir(directory)
            if re.match(pattern, filename)
        ]
        counter = max(existing_counters) + 1 if existing_counters else 1

        file = f"{filename_prefix}{filename_delimiter}{counter:0{number_padding}}.json"
        output_file = sandbox.resolve_write_file(directory, file)

        prompt = cls.hidden.prompt
        prompt_json = ""
        if prompt:
            if parse_text_tokens:
                prompt = parse_tokens(prompt, tokens)

            prompt_json = json.dumps(prompt, indent=4)
            logger.info("Prompt API JSON:\n%s", prompt_json)

            if save_prompt_api == "true":
                with open(output_file, "w", encoding="utf-8") as handle:
                    handle.write(prompt_json)
                logger.info("Output file path: %s", output_file)

        return io.NodeOutput(ui=ui.PreviewText(prompt_json))
