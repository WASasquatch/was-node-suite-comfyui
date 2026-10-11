"""CLIP text encoding with Noodle Soup Prompts and wildcard substitution."""

from __future__ import annotations

from comfy_api.latest import io, ui

from ...modules import log
from ...modules.prompt.dynamic import parse_dynamic_prompt
from ...modules.prompt.nsp import nsp_parse
from ...modules.prompt.variables import parse_prompt_vars
from ...modules.prompt.wildcards import replace_wildcards

logger = log.get_logger("conditioning.nsp_cliptextencoder")


class NSPCLIPTextEncoder(io.ComfyNode):
    """Parse a prompt, then encode it with CLIP.

    ``mode`` selects Noodle Soup Prompts or Wildcards. Dynamic prompts and prompt variables
    apply in both.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CLIPTextEncode (NSP)",
            display_name="CLIPTextEncode (NSP)",
            search_aliases=[
                "CLIPTextEncode (NSP)",
                "noodle soup prompts",
                "wildcards",
                "prompt",
                "text encode",
            ],
            category="WAS Suite/Conditioning",
            description=(
                "Encode a prompt with CLIP after substituting Noodle Soup Prompts "
                "terminology or wildcards, dynamic prompt groups and prompt variables. "
                "The parsed prompt is shown on the node and returned alongside the "
                "conditioning."
            ),
            inputs=[
                io.Combo.Input(
                    "mode",
                    options=["Noodle Soup Prompts", "Wildcards"],
                    tooltip=(
                        "Which substitution to run first. `Noodle Soup Prompts` swaps each "
                        "__term__ for a phrase from the pack's built-in pantry of "
                        "terminology; `Wildcards` swaps it for a random line from the "
                        "matching file in the pack's wildcards directory, so __colors__ "
                        "reads a line from colors.txt."
                    ),
                ),
                io.String.Input(
                    "noodle_key",
                    default="__",
                    multiline=False,
                    tooltip=(
                        "The marker that wraps a term to be substituted. With the default "
                        "'__', the prompt writes __subject__; changing it to '$$' would make "
                        "that $$subject$$ instead."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    tooltip=(
                        "Seed for the random choices, so the same seed always picks the same "
                        "phrases and lines. 0 is the exception: it leaves the __term__ draw "
                        "unseeded, so those come out different on every run while the "
                        "<a|b|c> groups stay fixed."
                    ),
                ),
                io.String.Input(
                    "text",
                    multiline=True,
                    tooltip=(
                        "The prompt, with three markups: __term__ = a pantry phrase or "
                        "wildcard line; <a|b|c> = one alternative at random; $|a stormy sky|$ "
                        "= a phrase captured as $1, $2 and so on, to repeat by number."
                    ),
                ),
                io.Clip.Input(
                    "clip",
                    tooltip=(
                        "The text encoder that turns the finished prompt into conditioning, "
                        "normally the CLIP output of a checkpoint loader."
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output(
                    display_name="conditioning",
                    tooltip="The encoded prompt, for a sampler's positive or negative input.",
                ),
                io.String.Output(
                    display_name="parsed_text",
                    tooltip=(
                        "The prompt after substitution, the words that were actually "
                        "encoded. Worth saving alongside the image, since a new seed "
                        "produces different words."
                    ),
                ),
                io.String.Output(
                    display_name="raw_text",
                    tooltip=(
                        "The prompt exactly as typed, markup and all, before any "
                        "substitution."
                    ),
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, mode, noodle_key, seed, text, clip) -> io.NodeOutput:
        if mode == "Noodle Soup Prompts":
            new_text = nsp_parse(text, seed, noodle_key)
        else:
            # replace_wildcards treats a falsy seed as "leave the RNG alone", and 0 is the
            # widget's default, so it is mapped to None rather than seeding on it.
            new_text = replace_wildcards(text, (None if seed == 0 else seed), noodle_key)

        new_text = parse_dynamic_prompt(new_text, seed)
        new_text, _variables = parse_prompt_vars(new_text)
        logger.info("CLIPTextEncode parsed prompt:\n %s", new_text)

        if clip is None:
            raise ValueError(
                "The clip input is None. If the CLIP came from a checkpoint loader, that "
                "checkpoint holds no usable text encoder."
            )
        conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(new_text))

        return io.NodeOutput(conditioning, new_text, text, ui=ui.PreviewText(new_text))
