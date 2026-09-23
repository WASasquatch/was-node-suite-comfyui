"""Encode a frame-indexed list of prompts into a conditioning schedule."""

from __future__ import annotations

import re

from comfy_api.latest import io

from ....modules.compat.types import CONDITIONING_SEQ
from ....modules.sampling.conditioning import (
    TOKEN_NORMALIZATIONS,
    WEIGHT_INTERPRETATIONS,
    encode_prompt,
)

REQUIRES = "extras"

#: Prompt lines are ``<frame index>:<prompt>``; a line without the prefix is skipped.
FRAME_PREFIX = re.compile(r"(\d+):")

#: The example schedule the text box starts with.
DEFAULT_SCHEDULE = """0:A portrait of a rosebud
5:A portrait of a blooming rosebud
10:A portrait of a blooming rose
15:A portrait of a rose"""


class WASCLIPTextEncodeList(io.ComfyNode):
    """Turn numbered prompt lines into a conditioning schedule."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASCLIPTextEncodeList",
            display_name="CLIP Text Encode Sequence (Advanced)",
            search_aliases=[
                "WASCLIPTextEncodeList",
                "CLIP Text Encode Sequence (Advanced)",
                "prompt schedule",
                "prompt travel",
                "conditioning sequence",
            ],
            category="WAS Suite/Conditioning",
            description=(
                "Encode one prompt per line, each tagged with the frame it takes effect "
                "on, into a schedule for KSampler Sequence. Write '0:a rosebud' and "
                "'10:a rose' and the run opens on the first prompt and switches to the "
                "second at frame 10."
            ),
            inputs=[
                io.Clip.Input(
                    "clip",
                    tooltip=(
                        "The CLIP model the prompts are encoded with. Use the one belonging "
                        "to the checkpoint that will sample them."
                    ),
                ),
                io.Combo.Input(
                    "token_normalization",
                    options=TOKEN_NORMALIZATIONS,
                    tooltip=(
                        "How token weights are evened out before encoding. Read only when a "
                        "pack registering BNK_CLIPTextEncodeAdvanced is installed; without "
                        "one this setting has no effect at all. 'none' leaves the weights "
                        "alone, 'mean' recentres them, 'length' scales by prompt length, "
                        "'length+mean' does both."
                    ),
                ),
                io.Combo.Input(
                    "weight_interpretation",
                    options=WEIGHT_INTERPRETATIONS,
                    tooltip=(
                        "Which prompt weighting dialect the '(word:1.2)' syntax is read in. "
                        "Read only when a pack registering BNK_CLIPTextEncodeAdvanced is "
                        "installed; without one this setting has no effect and the prompt is "
                        "read the way ComfyUI's own CLIP Text Encode reads it."
                    ),
                ),
                io.String.Input(
                    "text",
                    multiline=True,
                    default=DEFAULT_SCHEDULE,
                    tooltip=(
                        "One prompt per line, each written as 'frame:prompt', for example "
                        "'0:a rosebud'. The number is the loop the prompt takes over on, "
                        "counting from zero, and it stays in force until the next numbered "
                        "line. A line with no number in front of a colon is ignored."
                    ),
                ),
            ],
            outputs=[
                CONDITIONING_SEQ.Output(
                    display_name="conditioning_sequence",
                    tooltip=(
                        "The frame-tagged prompts, for the positive_seq or negative_seq "
                        "input of KSampler Sequence. It is not an ordinary conditioning and "
                        "does not fit a plain sampler."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, clip, token_normalization, weight_interpretation, text) -> io.NodeOutput:
        conditionings = []
        for line in text.strip().splitlines():
            match = FRAME_PREFIX.match(line)
            if not match:
                continue
            _, prompt = line.split(":", 1)
            encoded = encode_prompt(
                clip, prompt.strip(), token_normalization, weight_interpretation
            )
            conditionings.append((int(match.group(1)), encoded))
        return io.NodeOutput(conditionings)
