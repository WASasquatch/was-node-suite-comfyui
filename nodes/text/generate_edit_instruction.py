"""Writing an explicit image edit instruction from a short one, with a CLIP's language model."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.convert.tensors import tensor2pil
from ...modules.interface import run_result
from ...modules.model import text_decode, vision_agent

NODE_NAME = "Generate Edit Instruction"

#: The largest seed the sampler takes.
MAX_SEED = 0xFFFFFFFFFFFFFFFF


class GenerateEditInstruction(io.ComfyNode):
    """Turn a short edit request into an explicit instruction for an image editing model, reading the picture."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASGenerateEditInstruction",
            display_name=NODE_NAME,
            search_aliases=[
                "WASGenerateEditInstruction", NODE_NAME,
                "rewrite instruction", "prompt rewrite", "edit instruction", "refine instruction",
                "bluepixel", "kontext", "qwen image edit",
            ],
            category="WAS Suite/Text",
            description=(
                "Turn a short edit request into an explicit instruction an image editing model follows "
                "fully, written by a language model that reads the picture: what changes, implied effects "
                "included, and what stays. With a judge's notes wired in, it corrects the last attempt."
            ),
            inputs=[
                io.Clip.Input(
                    "clip",
                    tooltip="qwen3.5_9b_int8_convrot.safetensors from Load CLIP with type krea2, or another Qwen-VL model.",
                ),
                io.Image.Input("image", tooltip="The picture to edit. The first of a batch is read."),
                io.String.Input(
                    "instruction", multiline=True, default="",
                    tooltip="The edit as written, as 'Make it a snowy winter night'.",
                ),
                io.String.Input(
                    "feedback", optional=True, force_input=True,
                    tooltip=(
                        "Judge Images' notes on the last attempt, as 'the sky is blown out and shows no snow'. "
                        "Unwired or empty = a first attempt."
                    ),
                ),
                io.String.Input(
                    "last_instruction", optional=True, force_input=True,
                    tooltip="The instruction the last attempt was made from, as 'Add falling snow and a dark blue sky'.",
                ),
                io.Float.Input(
                    "temperature", default=0.0, min=0.0, max=2.0, step=0.01, advanced=True,
                    tooltip="0 = the likeliest words, the same instruction every time; 0.7 = varied wording per seed.",
                ),
                io.Int.Input(
                    "seed", default=0, min=0, max=MAX_SEED, advanced=True,
                    tooltip="The seed sampling draws from. Ignored at temperature 0.",
                ),
                io.Int.Input(
                    "max_length", default=256, min=32, max=2048, advanced=True,
                    tooltip="The most tokens the instruction may run to: 256 covers about 180 words.",
                ),
            ],
            outputs=[
                io.String.Output(
                    "instruction", tooltip="The explicit instruction, for the edit model's prompt and Judge Images.",
                ),
            ],
        )

    @classmethod
    def execute(cls, clip, image, instruction, feedback=None, last_instruction=None, temperature=0.0, seed=0,
                max_length=256) -> io.NodeOutput:
        """Write the instruction.

        Raises:
            ValueError: No CLIP is connected, the instruction is empty, or the model cannot read pictures.
        """
        if clip is None:
            raise ValueError(f"{NODE_NAME} needs a CLIP. Connect Load CLIP with qwen3.5_9b_int8_convrot.safetensors.")
        feedback = str(feedback or "").strip()
        previous = str(last_instruction or "").strip()
        text, report = vision_agent.rewrite(
            clip, tensor2pil(image[0]), instruction, feedback, previous, max_length, temperature, seed,
        )
        _, counts, _ = text_decode.figures(report)
        bodies = [run_result.body("asked", instruction.strip())]
        if feedback:
            bodies.append(run_result.body("judge's notes", feedback))
        bodies.append(run_result.body("written", text))
        run_result.publish(
            summary="corrected from the judge's notes" if feedback else "written from the request",
            counts=counts, bodies=bodies,
        )
        return io.NodeOutput(text)
