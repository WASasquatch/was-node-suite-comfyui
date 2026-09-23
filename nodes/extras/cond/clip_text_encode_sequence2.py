"""Encode a list of prompts and work out when each one takes over."""

from __future__ import annotations

import numpy as np
from comfy_api.latest import io

from ....modules.sampling.conditioning import (
    TOKEN_NORMALIZATIONS,
    WEIGHT_INTERPRETATIONS,
    encode_prompt,
)

REQUIRES = "extras"

#: How the changeover frames are spread across the run.
KEYFRAME_TYPES = ["linear", "sinus", "sinus_inverted", "half_sinus", "half_sinus_inverted"]

#: The example prompt list the text box starts with.
DEFAULT_PROMPTS = """A portrait of a rosebud
A portrait of a blooming rosebud
A portrait of a blooming rose
A portrait of a rose"""


def keyframes_for(kind: str, frame_count: int, conditioning_count: int) -> list:
    """The frames at which the run steps from one prompt to the next.

    Args:
        kind: One of :data:`KEYFRAME_TYPES`.
        frame_count: Frames in the whole run.
        conditioning_count: How many prompts there are to get through.

    Returns:
        The changeover frames, one per prompt.

    Raises:
        ValueError: ``kind`` is not one of :data:`KEYFRAME_TYPES`.
    """
    if kind == "linear":
        return np.linspace(
            frame_count // conditioning_count, frame_count, conditioning_count, dtype=int
        ).tolist()

    if kind == "sinus":
        curve = np.sin(np.linspace(0, np.pi, conditioning_count))
        span = curve.max() - curve.min()
        normalized = (curve - curve.min()) / span
        scaled = normalized * (frame_count - 1) + 1
        rounded = np.round(scaled).astype(int)
        return sorted(np.unique(rounded, return_index=True)[1].tolist())

    if kind == "sinus_inverted":
        curve = np.cos(np.linspace(0, np.pi, conditioning_count))
        return (curve * (frame_count - 1) + 1).astype(int).tolist()

    if kind == "half_sinus":
        curve = np.sin(np.linspace(0, np.pi / 2, conditioning_count))
        return (curve * (frame_count - 1) + 1).astype(int).tolist()

    if kind == "half_sinus_inverted":
        curve = np.cos(np.linspace(0, np.pi / 2, conditioning_count))
        return (curve * (frame_count - 1) + 1).astype(int).tolist()

    raise ValueError("Unsupported cond_keyframes_type: " + kind)


class WASCLIPTextEncodeSequence2(io.ComfyNode):
    """Encode a list of prompts and schedule when each one takes over."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASCLIPTextEncodeSequence2",
            display_name="CLIP Text Encode Sequence (v2)",
            search_aliases=[
                "WASCLIPTextEncodeSequence2",
                "CLIP Text Encode Sequence (v2)",
                "prompt schedule",
                "prompt travel",
                "keyframes",
            ],
            category="WAS Suite/Conditioning",
            description=(
                "Encode one prompt per line and work out the frame each one takes over on, "
                "spread across the length of the run. The three outputs plug straight into "
                "KSampler Sequence (v2)'s positive_seq or negative_seq, cond_keyframes "
                "and frame_count, so a prompt list becomes an animation schedule with no "
                "numbers typed by hand."
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
                io.Combo.Input(
                    "cond_keyframes_type",
                    options=KEYFRAME_TYPES,
                    tooltip=(
                        "How the changeovers are spaced. `linear` gives every prompt an equal "
                        "share of the run. The sinus shapes bunch them up at one end or the "
                        "other, so the sequence lingers on the opening prompts and races "
                        "through the rest, or the reverse, useful when the first shot needs "
                        "to be held and the last few are only a flourish."
                    ),
                ),
                io.Int.Input(
                    "frame_count",
                    default=100,
                    min=1,
                    max=1024,
                    step=1,
                    tooltip=(
                        "How long the whole run is, in frames. The changeovers are spread "
                        "across this many, so at 100 frames and four prompts each one holds "
                        "for about 25."
                    ),
                ),
                io.String.Input(
                    "text",
                    multiline=True,
                    default=DEFAULT_PROMPTS,
                    tooltip=(
                        "One prompt per line, in the order the run works through them. No "
                        "frame numbers: cond_keyframes_type and frame_count decide when each "
                        "one takes over. A blank line is encoded as an empty prompt and takes "
                        "its turn like any other."
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output(
                    display_name="conditioning_sequence",
                    tooltip=(
                        "Every prompt, encoded, in the order they were written. Wire it into "
                        "KSampler Sequence (v2)'s positive_seq or negative_seq."
                    ),
                ),
                io.Int.Output(
                    display_name="cond_keyframes",
                    tooltip=(
                        "The frames at which the run steps to the next prompt. Wire it into "
                        "KSampler Sequence (v2)'s cond_keyframes."
                    ),
                ),
                io.Int.Output(
                    display_name="frame_count",
                    tooltip=(
                        "The frame count as it was given, passed straight through so one wire "
                        "carries it to KSampler Sequence (v2) rather than the number being "
                        "typed twice."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip,
        token_normalization,
        weight_interpretation,
        cond_keyframes_type,
        frame_count,
        text,
    ) -> io.NodeOutput:
        conditionings = [
            encode_prompt(clip, line, token_normalization, weight_interpretation)
            for line in text.strip().splitlines()
        ]
        keyframes = keyframes_for(cond_keyframes_type, frame_count, len(conditionings))
        return io.NodeOutput(conditionings, keyframes, frame_count)
