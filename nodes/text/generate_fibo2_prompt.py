"""Writing a Fibo 2.0 caption with a CLIP's language model."""

from __future__ import annotations

import json

from comfy_api.latest import io

from ...modules.interface import run_result
from ...modules.model import text_decode
from ...modules.model.fibo2 import conditioning, writer

NODE_NAME = "Generate Fibo 2 Prompt"

#: The largest seed the sampler takes.
MAX_SEED = 0xFFFFFFFFFFFFFFFF


class GenerateFibo2Prompt(io.ComfyNode):
    """Write the JSON caption Fibo 2.0 reads, from an idea, pictures or an earlier caption."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        sampling = [
            io.DynamicCombo.Option(
                key="on",
                inputs=[
                    io.Float.Input(
                        "temperature", default=0.2, min=0.01, max=2.0, step=0.01,
                        tooltip="How adventurous each pick is. 0.2 stays close to the likeliest words, 0.7 varies more.",
                    ),
                    io.Int.Input(
                        "top_k", default=0, min=0, max=1000,
                        tooltip="Pick only among this many likeliest tokens. 0 = no limit.",
                    ),
                    io.Float.Input(
                        "top_p", default=1.0, min=0.0, max=1.0, step=0.01,
                        tooltip="Pick only among the likeliest tokens whose chances add up to this. 1.0 = off.",
                    ),
                    io.Float.Input(
                        "min_p", default=0.0, min=0.0, max=1.0, step=0.01,
                        tooltip="Drop any token less likely than this share of the likeliest. 0 = off.",
                    ),
                    io.Float.Input(
                        "repetition_penalty", default=1.0, min=0.0, max=5.0, step=0.01,
                        tooltip="Above 1.0 makes a used token less likely again. 1.0 = off.",
                    ),
                    io.Int.Input(
                        "seed", default=0, min=0, max=MAX_SEED,
                        tooltip="The same seed with the same settings writes the same caption.",
                    ),
                ],
            ),
            io.DynamicCombo.Option(key="off", inputs=[]),
        ]
        return io.Schema(
            node_id="WASGenerateFibo2Prompt",
            display_name=NODE_NAME,
            search_aliases=[
                "WASGenerateFibo2Prompt", NODE_NAME,
                "prompt enhance", "fibo", "fibo 2", "bria", "vgl", "json caption", "LLM", "qwen",
            ],
            category="WAS Suite/Text",
            description=(
                "Write the JSON caption Fibo 2.0 reads, with the language model inside a CLIP: from a "
                "short idea, from pictures to edit or caption, or by revising a caption. Hands it to Text "
                "Encode Fibo 2."
            ),
            inputs=[
                io.Clip.Input(
                    "clip",
                    tooltip=(
                        "A Qwen-VL model from Load CLIP, as qwen3vl_4b_bf16.safetensors. A text-only model "
                        "writes from ideas and captions, not pictures."
                    ),
                ),
                io.String.Input(
                    "prompt", multiline=True, dynamic_prompts=True, default="",
                    tooltip=(
                        "The idea, as 'a lighthouse on a stormy night', or the edit, as 'make it winter'. "
                        "Empty with an image = describe the image."
                    ),
                ),
                io.Combo.Input(
                    "task", options=list(writer.TASKS), default="auto",
                    tooltip=(
                        "auto = edit with images, caption with an image alone, refine with a caption alone, "
                        "else generate; inspire = a new scene borrowing from image_1; refine = revise caption."
                    ),
                ),
                io.Image.Input("image_1", optional=True, tooltip="A picture to edit, caption or borrow from."),
                io.Image.Input("image_2", optional=True, tooltip="Another picture an edit reads, as image 2."),
                io.Image.Input("image_3", optional=True, tooltip="Another picture an edit reads, as image 3."),
                io.Image.Input("image_4", optional=True, tooltip="Another picture an edit reads, as image 4."),
                io.Mask.Input(
                    "mask", optional=True,
                    tooltip="White where image_1 is filled, for an edit of one picture.",
                ),
                io.String.Input(
                    "caption", optional=True, force_input=True,
                    tooltip="A caption to revise, or image_1's own caption for an edit to start from.",
                ),
                io.String.Input(
                    "style", multiline=True, default="", advanced=True,
                    tooltip="Style guidelines for generate and inspire, as 'muted film colours, soft grain'.",
                ),
                io.Int.Input(
                    "max_length", default=4096, min=256, max=16384,
                    tooltip="The most tokens the caption may run to. A full caption takes 800 to 2000.",
                ),
                io.DynamicCombo.Input(
                    "sampling_mode", options=sampling, display_name="Sampling Mode",
                    tooltip="on = draws each token within the limits below; off = always the likeliest token.",
                ),
            ],
            outputs=[
                io.String.Output(display_name="prompt", tooltip="One line of JSON, for Text Encode Fibo 2's prompt."),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, task, max_length, sampling_mode, image_1=None, image_2=None, image_3=None,
                image_4=None, mask=None, caption=None, style="") -> io.NodeOutput:
        if clip is None:
            raise ValueError(f"{NODE_NAME} needs a CLIP. Connect Load CLIP with qwen3vl_4b_bf16.safetensors.")
        pictures = [conditioning.to_pil(frame) for image in (image_1, image_2, image_3, image_4)
                    if image is not None for frame in image]
        layer = None
        if mask is not None:
            from PIL import Image

            values = (mask[0].float().clamp(0, 1) * 255).round().byte().cpu().numpy()
            layer = Image.fromarray(values, "L")
        sampling_mode = sampling_mode or {}
        sampling = {
            "do_sample": sampling_mode.get("sampling_mode") == "on",
            "temperature": sampling_mode.get("temperature", writer.TEMPERATURE),
            "top_k": sampling_mode.get("top_k", 0),
            "top_p": sampling_mode.get("top_p", 1.0),
            "min_p": sampling_mode.get("min_p", 0.0),
            "repetition_penalty": sampling_mode.get("repetition_penalty", 1.0),
            "seed": sampling_mode.get("seed", 0),
        }
        text, ran, report = writer.write(
            clip, task, prompt, pictures, mask=layer, caption=caption or "", style=style or "",
            max_length=max_length, sampling=sampling,
        )
        summary, counts, facts = text_decode.figures(report)
        run_result.publish(
            summary=f"{ran.capitalize()}: {summary}",
            counts=counts,
            facts={"task": ran, **facts},
            bodies=run_result.body("caption", json.dumps(json.loads(text), indent=2, ensure_ascii=False)),
        )
        return io.NodeOutput(text)
