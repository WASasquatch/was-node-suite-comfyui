"""Revising a text-to-image prompt from a judge's notes on the last attempt, with a CLIP's language model."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.convert.tensors import tensor2pil
from ...modules.interface import run_result
from ...modules.model import text_decode, vision_agent

NODE_NAME = "Revise Prompt"

#: The largest seed the sampler takes.
MAX_SEED = 0xFFFFFFFFFFFFFFFF


class RevisePrompt(io.ComfyNode):
    """Rewrite a prompt so the next picture fixes what a judge found wrong with the last."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASRevisePrompt",
            display_name=NODE_NAME,
            search_aliases=[
                "WASRevisePrompt", NODE_NAME,
                "refine prompt", "prompt revision", "retry prompt", "prompt adherence", "fix prompt",
            ],
            category="WAS Suite/Text",
            description=(
                "Rewrite a text-to-image prompt from a judge's notes so the next attempt shows what the last "
                "one missed, keeping the prompt's subject and style. Without notes the prompt passes through "
                "as written, so a retry loop's first pass uses it unchanged."
            ),
            inputs=[
                io.Clip.Input(
                    "clip",
                    tooltip="qwen3.5_9b_int8_convrot.safetensors from Load CLIP with type krea2, or any language model CLIP.",
                ),
                io.String.Input(
                    "prompt", multiline=True, default="",
                    tooltip="The prompt as written, as 'three red apples on a blue plate'.",
                ),
                io.String.Input(
                    "feedback", optional=True, force_input=True,
                    tooltip=(
                        "Judge Images' notes on the last attempt, as 'only two apples, the plate is white'. "
                        "Unwired or empty = the prompt passes through as written."
                    ),
                ),
                io.String.Input(
                    "last_prompt", optional=True, force_input=True,
                    tooltip="The prompt the last attempt was made from, as 'three red apples on a blue plate'.",
                ),
                io.Image.Input(
                    "last_image", optional=True,
                    tooltip="The last attempt's picture, as Judge Images' best. Read only by a model that sees pictures.",
                ),
                io.Float.Input(
                    "temperature", default=0.0, min=0.0, max=2.0, step=0.01, advanced=True,
                    tooltip="0 = the likeliest words, the same revision every time; 0.7 = varied wording per seed.",
                ),
                io.Int.Input(
                    "seed", default=0, min=0, max=MAX_SEED, advanced=True,
                    tooltip="The seed sampling draws from. Ignored at temperature 0.",
                ),
                io.Int.Input(
                    "max_length", default=384, min=32, max=2048, advanced=True,
                    tooltip="The most tokens the revised prompt may run to: 384 covers about 280 words.",
                ),
            ],
            outputs=[
                io.String.Output("prompt", tooltip="The prompt for the next attempt, for CLIP Text Encode."),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, feedback=None, last_prompt=None, last_image=None, temperature=0.0, seed=0,
                max_length=384) -> io.NodeOutput:
        """Pass the prompt on, or revise it from the judge's notes.

        Raises:
            ValueError: The prompt is empty, or notes are wired and no CLIP is connected.
        """
        feedback = str(feedback or "").strip()
        if not prompt.strip():
            raise ValueError(f"{NODE_NAME} needs the prompt the pictures are meant to show.")
        if not feedback:
            run_result.publish(
                summary="the prompt as written, no notes yet",
                bodies=[run_result.body("prompt", prompt.strip())],
            )
            return io.NodeOutput(prompt.strip())
        if clip is None:
            raise ValueError(f"{NODE_NAME} needs a CLIP to revise with. Connect Load CLIP with qwen3.5_9b_int8_convrot.safetensors.")
        picture = tensor2pil(last_image[0]) if last_image is not None else None
        text, report = vision_agent.revise(
            clip, prompt, feedback, str(last_prompt or ""), picture, max_length, temperature, seed,
        )
        _, counts, _ = text_decode.figures(report)
        run_result.publish(
            summary="revised from the judge's notes",
            counts=counts,
            bodies=[
                run_result.body("asked", prompt.strip()),
                run_result.body("judge's notes", feedback),
                run_result.body("revised", text),
            ],
        )
        return io.NodeOutput(text)
