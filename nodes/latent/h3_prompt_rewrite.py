"""A MiniMax H3 scene prompt rewritten by a language model under a system prompt of rules."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.interface import run_result
from ...modules.latent import h3_writer

logger = log.get_logger("nodes.h3_prompt_rewrite")

NODE_NAME = "MiniMax H3 Prompt Rewrite"

#: The largest seed the sampler takes.
MAX_SEED = 0xFFFFFFFFFFFFFFFF


class H3PromptRewrite(io.ComfyNode):
    """One MiniMax H3 scene prompt rewritten, or written from directions, by a language model."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASH3PromptRewrite",
            display_name=NODE_NAME,
            search_aliases=[
                "WASH3PromptRewrite",
                NODE_NAME,
                "h3 rewrite",
                "rewrite prompt",
                "prompt enhance",
                "minimax h3",
                "prompt timeline",
                "llm",
                "vlm",
            ],
            category="WAS Suite/Latent/Video",
            description=(
                "Rewrites one MiniMax H3 scene prompt with a language model so it follows a "
                "system prompt of rules, keeping its story, tags and dialogue unless the "
                "directions change them. An empty prompt is written from the directions. The "
                "Prompt Timeline's Rewrite runs it on the chosen scene."
            ),
            inputs=[
                io.Clip.Input(
                    "vlm_clip",
                    tooltip=(
                        "The language model that writes, from Load CLIP, as "
                        "`qwen3vl_8b_fp8_scaled.safetensors` or `qwen_3_8b_fp8mixed.safetensors`."
                    ),
                ),
                io.String.Input(
                    "prompt", multiline=True, default="",
                    tooltip=(
                        "The scene prompt to rewrite, as written for MiniMax H3 Conditioning. "
                        "Blank writes a new one from directions."
                    ),
                ),
                io.String.Input(
                    "directions", multiline=True, default="",
                    tooltip=(
                        "What to change, as `make it night and add rain` or `give Jo a line "
                        "asking where everyone went`. Blank brings the prompt in line with the rules."
                    ),
                ),
                io.String.Input(
                    "context", multiline=True, default="", optional=True,
                    tooltip=(
                        "What the model should know around the scene, as `The scene before: the "
                        "gang reach the factory.` or `<Picture 1> is purz.png`."
                    ),
                ),
                io.String.Input(
                    "system_prompt", multiline=True, default=h3_writer.DEFAULT_SYSTEM_PROMPT, optional=True,
                    tooltip=(
                        "The rules the prompt is rewritten under: the prompt layout, shots, "
                        "dialogue and sound. A line such as `Keep every line of dialogue under 12 "
                        "words.` adds a rule; blank rewrites under none."
                    ),
                ),
                io.Int.Input(
                    "seed", default=0, min=0, max=MAX_SEED, control_after_generate=True,
                    tooltip="Which draw of the model's choices to take, as `0` or `42`.",
                ),
                io.Float.Input(
                    "temperature", default=0.7, min=0.01, max=2.0, step=0.01, optional=True, advanced=True,
                    tooltip="How adventurous each pick is. `0.7` is balanced, `0.3` stays close to the likeliest words, `1.1` wanders.",
                ),
                io.Int.Input(
                    "top_k", default=64, min=0, max=1000, optional=True, advanced=True,
                    tooltip="Pick only among this many likeliest tokens. `0` turns the limit off.",
                ),
                io.Float.Input(
                    "top_p", default=0.95, min=0.0, max=1.0, step=0.01, optional=True, advanced=True,
                    tooltip="Pick only among the likeliest tokens whose chances add up to this. `1.0` turns it off.",
                ),
                io.Float.Input(
                    "min_p", default=0.05, min=0.0, max=1.0, step=0.01, optional=True, advanced=True,
                    tooltip="Drop any token less likely than this fraction of the likeliest one. `0` turns it off.",
                ),
                io.Float.Input(
                    "repetition_penalty", default=1.05, min=0.0, max=5.0, step=0.01, optional=True, advanced=True,
                    tooltip="Above `1.0` makes a token already used less likely again. `1.0` turns it off.",
                ),
                io.Boolean.Input(
                    "thinking", default=False, optional=True, advanced=True,
                    tooltip="`true` lets a model that reasons, such as Qwen3, think before answering.",
                ),
                io.Boolean.Input(
                    "fast_decode", default=True, optional=True, advanced=True,
                    tooltip="`true` decodes on the fixed cache graph path where the model allows it; `false` runs as core Generate Text.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="prompt", tooltip="The rewritten prompt."),
                io.String.Output(display_name="report", tooltip="How long the prompt was before and after."),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, vlm_clip, prompt, directions, seed, context="",
                system_prompt=h3_writer.DEFAULT_SYSTEM_PROMPT, temperature=0.7, top_k=64, top_p=0.95,
                min_p=0.05, repetition_penalty=1.05, thinking=False, fast_decode=True) -> io.NodeOutput:
        """Rewrite the prompt, or write one from the directions.

        Raises:
            ValueError: No model arrived, there is nothing to work from, or the model wrote no
                usable prompt.
        """
        if vlm_clip is None:
            raise ValueError(
                f"{NODE_NAME} needs a language model. Wire Load CLIP with a model such as "
                f"qwen3vl_8b_fp8_scaled.safetensors into vlm_clip"
            )
        if not str(prompt or "").strip() and not str(directions or "").strip():
            raise ValueError(f"{NODE_NAME} has nothing to work from. Type a prompt, or directions to write one")
        sampling = h3_writer.Sampling(float(temperature), int(top_k), float(top_p), float(min_p),
                                      float(repetition_penalty), bool(thinking), bool(fast_decode))
        write = h3_writer.make_writer(vlm_clip, int(seed), sampling, str(system_prompt or ""))
        try:
            text = h3_writer.rewrite(write, prompt, directions, context)
        except ValueError as error:
            raise ValueError(f"{NODE_NAME}: {error}") from error
        before = len(str(prompt or "").split())
        after = len(text.split())
        report = f"rewrote {before} words as {after}" if before else f"wrote {after} words from the directions"
        logger.info("%s: %s", NODE_NAME, report)
        run_result.publish(summary=report, counts={"words before": before, "words after": after})
        return io.NodeOutput(text, report, ui={"was_h3_rewrite": [text]})
