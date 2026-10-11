"""Run every prompt markup the pack understands, without encoding the result."""

from __future__ import annotations

from comfy_api.latest import io, ui

from ....modules import log
from ....modules.prompt.dynamic import parse_dynamic_prompt
from ....modules.prompt.nsp import nsp_parse
from ....modules.prompt.variables import parse_prompt_vars
from ....modules.prompt.wildcards import replace_wildcards

logger = log.get_logger("nodes.text.parse")


class PromptParse(io.ComfyNode):
    """Expand ``__terms__``, ``<a|b|c>`` groups and ``$|phrase|$`` variables into text.

    The three parsers run in that order.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASPromptParse",
            display_name="Prompt Parse",
            search_aliases=[
                "WASPromptParse", "Prompt Parse",
                "dynamic prompts",
                "prompt variables",
                "wildcards",
                "nsp",
                "expand prompt",
            ],
            category="WAS Suite/Text/Parse",
            description=(
                "Expand __terms__, <a|b|c> groups and $|phrase|$ variables and return the "
                "finished text. Every markup CLIPTextEncode (NSP) understands, stopping at "
                "the string so it can be saved, split or encoded elsewhere."
            ),
            inputs=[
                io.String.Input(
                    "text",
                    multiline=True,
                    tooltip=(
                        "The prompt, in any of three markups: __term__ = a phrase from the "
                        "terminology pantry or a line from a wildcard file; <a|b|c> = one of "
                        "the alternatives; $|a stormy sky|$ = a phrase captured as $1 to "
                        "repeat later."
                    ),
                ),
                io.Combo.Input(
                    "mode",
                    options=["Noodle Soup Prompts", "Wildcards", "none"],
                    tooltip=(
                        "What __terms__ are replaced with. `none` = left alone; `Wildcards` = "
                        "a random line from the matching file in the wildcards directory, so "
                        "__animals/birds__ reads animals/birds.txt; `Noodle Soup Prompts` = a "
                        "phrase from the shared terminology pantry, downloaded once."
                    ),
                ),
                io.String.Input(
                    "noodle_key",
                    default="__",
                    multiline=False,
                    tooltip=(
                        "The marker put either side of a term to flag it for replacement. "
                        "With the default '__', __animals__ is replaced and plain animals is "
                        "not."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    tooltip=(
                        "Which words get drawn. 0 = __terms__ differ every run while <a|b|c> "
                        "groups stay fixed; any other value = the same seed and prompt always "
                        "give the same text."
                    ),
                ),
                io.Boolean.Input(
                    "dynamic_prompts",
                    default=True,
                    tooltip=(
                        "Whether <a|b|c> groups are resolved to one of their options. Turn "
                        "it off to keep the brackets in the text, which is what a prompt "
                        "being passed on to another parser needs."
                    ),
                ),
                io.Boolean.Input(
                    "prompt_variables",
                    default=True,
                    tooltip=(
                        "Whether $|phrase|$ captures are expanded. Each capture is numbered "
                        "$1, $2 and so on in order, and every reference to that number "
                        "becomes the phrase, so one long description is written once and "
                        "repeated."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="text",
                    tooltip=(
                        "The finished prompt, with every enabled markup expanded. Worth "
                        "saving beside the image with Text Save, since a new seed produces "
                        "different words."
                    ),
                ),
                io.String.Output(
                    display_name="raw_text",
                    tooltip="The prompt exactly as it arrived, markup and all.",
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        text,
        mode="Noodle Soup Prompts",
        noodle_key="__",
        seed=0,
        dynamic_prompts=True,
        prompt_variables=True,
    ) -> io.NodeOutput:
        parsed = text
        if mode == "Noodle Soup Prompts":
            parsed = nsp_parse(parsed, seed, noodle_key)
        elif mode == "Wildcards":
            # replace_wildcards reads a falsy seed as "leave the shared RNG alone", and 0 is
            # the widget's default, so it is mapped to None rather than seeding on it.
            parsed = replace_wildcards(parsed, (None if seed == 0 else seed), noodle_key)

        if dynamic_prompts:
            parsed = parse_dynamic_prompt(parsed, seed)
        if prompt_variables:
            parsed, _variables = parse_prompt_vars(parsed)

        logger.info("Prompt Parse:\n%s", parsed)
        return io.NodeOutput(parsed, text, ui=ui.PreviewText(parsed))
