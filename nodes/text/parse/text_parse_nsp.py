"""Expand Noodle Soup Prompts terminology or wildcard files inside a prompt."""

from __future__ import annotations

from comfy_api.latest import io, ui

from ....modules import log
from ....modules.prompt.nsp import nsp_parse
from ....modules.prompt.wildcards import replace_wildcards

logger = log.get_logger("nodes.text.parse")


class TextParseNoodleSoupPrompts(io.ComfyNode):
    """Replace each ``__term__`` in ``text`` with a random entry drawn for that term."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Text Parse Noodle Soup Prompts",
            display_name="Text Parse Noodle Soup Prompts",
            search_aliases=[
                "Text Parse Noodle Soup Prompts",
                "nsp",
                "noodle soup",
                "wildcards",
            ],
            category="WAS Suite/Text/Parse",
            description=(
                "Replace __terms__ in a prompt with random Noodle Soup Prompts terminology, "
                "or with a random line from the matching wildcard file."
            ),
            inputs=[
                io.Combo.Input(
                    "mode",
                    options=["Noodle Soup Prompts", "Wildcards"],
                    tooltip=(
                        "Where the replacements come from. `Wildcards` = one random line from "
                        "the matching file in the wildcards directory, so __animals/birds__ "
                        "reads animals/birds.txt; `Noodle Soup Prompts` = the shared pantry "
                        "of subjects, styles and materials, downloaded once and cached."
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
                    max=18446744073709551615,
                    tooltip=(
                        "Which terms get drawn. Any value other than 0 makes the draw "
                        "repeatable, so the same seed and the same prompt always give the "
                        "same words. 0 draws differently every run and cannot be reproduced."
                    ),
                ),
                io.String.Input(
                    "text",
                    multiline=True,
                    placeholder="Eg: a __animals__ in a __location__",
                    tooltip=(
                        "Prompt with __terms__ to expand; STRING. Each occurrence is "
                        "drawn separately. Eg: a __animals__ in a __location__"
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    tooltip=(
                        "The prompt with every marked term replaced. A term with no matching "
                        "pantry entry or wildcard file is left as written."
                    ),
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, text, mode="Noodle Soup Prompts", noodle_key="__", seed=0) -> io.NodeOutput:
        if mode == "Noodle Soup Prompts":
            new_text = nsp_parse(text, seed, noodle_key)
            logger.info("Text Parse NSP:\n%s", new_text)
        else:
            new_text = replace_wildcards(text, (None if seed == 0 else seed), noodle_key)
            logger.info("CLIPTextEncode Wildcards:\n%s", new_text)

        return io.NodeOutput(new_text, ui=ui.PreviewText(new_text))
