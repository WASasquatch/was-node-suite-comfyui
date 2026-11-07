"""Scoring a batch of pictures against their prompt or source and keeping the best, with a CLIP's language model."""

from __future__ import annotations

import json

from comfy_api.latest import io

from ....modules.convert.tensors import tensor2pil
from ....modules.interface import run_result
from ....modules.model import vision_agent

NODE_NAME = "Judge Images"

#: Characters of each picture's notes the panel carries.
NOTES_SHOWN = 1000


class JudgeImages(io.ComfyNode):
    """Score each picture of a batch side by side and keep the best."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASJudgeImages",
            display_name=NODE_NAME,
            search_aliases=[
                "WASJudgeImages", NODE_NAME,
                "best of", "pick best", "verify edit", "score images", "image reward", "judge edits",
            ],
            category="WAS Suite/Image/Analyze",
            description=(
                "Score each picture in a batch, as several seeds of one prompt make, from 0 to 10 with a "
                "language model that reads them. Edits are scored on following, consistency and quality "
                "against their source; generations on following and quality. Hands on the best and its notes."
            ),
            inputs=[
                io.Clip.Input(
                    "clip",
                    tooltip="qwen3.5_9b_int8_convrot.safetensors from Load CLIP with type krea2, or another Qwen-VL model.",
                ),
                io.Image.Input(
                    "images", tooltip="The pictures to score, 1 to 6, as VAE Decode gives a batch from Repeat Latent Batch.",
                ),
                io.String.Input(
                    "prompt", multiline=True, default="",
                    tooltip="The instruction the edits followed or the prompt the pictures came from, as 'Make it snow'.",
                ),
                io.Image.Input(
                    "source", optional=True,
                    tooltip="The picture before the edit. Unwired = the pictures are judged as generations.",
                ),
                io.Int.Input(
                    "least_following", default=9, min=0, max=10, advanced=True,
                    tooltip="The following score a picture needs to pass, 0 to 10.",
                ),
                io.Int.Input(
                    "least_consistency", default=8, min=0, max=10, advanced=True,
                    tooltip="The consistency score an edit needs to pass, 0 to 10. Unused without a source.",
                ),
                io.Int.Input(
                    "least_quality", default=6, min=0, max=10, advanced=True,
                    tooltip="The quality score a picture needs to pass, 0 to 10.",
                ),
            ],
            outputs=[
                io.Image.Output("best", tooltip="The picture kept: the best that passes, or the best of all when none does."),
                io.Int.Output("index", tooltip="The kept picture's place in the batch, from 0."),
                io.Boolean.Output("passed", tooltip="True when the kept picture reaches every least score."),
                io.String.Output("scores", tooltip="Every picture's scores and notes as a JSON list, in batch order."),
                io.String.Output(
                    "notes", tooltip="The flaws found in the kept picture, for Generate Edit Instruction's feedback.",
                ),
            ],
        )

    @classmethod
    def execute(cls, clip, images, prompt, source=None, least_following=9, least_consistency=8,
                least_quality=6) -> io.NodeOutput:
        """Score every picture and keep the best.

        Raises:
            ValueError: No CLIP is connected, the prompt is empty, there are too many pictures, or the
                model cannot read pictures.
        """
        if clip is None:
            raise ValueError(f"{NODE_NAME} needs a CLIP. Connect Load CLIP with qwen3.5_9b_int8_convrot.safetensors.")
        if not prompt.strip():
            raise ValueError(f"{NODE_NAME} needs the instruction or prompt the pictures were made from.")
        if len(images) > vision_agent.MOST_PICTURES:
            raise ValueError(
                f"{NODE_NAME} compares up to {vision_agent.MOST_PICTURES} pictures at once and was given "
                f"{len(images)}. Set Repeat Latent Batch's amount or the batch size to {vision_agent.MOST_PICTURES} or fewer."
            )
        least = {"following": least_following, "consistency": least_consistency, "quality": least_quality}
        scored, pick, _ = vision_agent.judge(
            clip, [tensor2pil(frame) for frame in images], prompt,
            tensor2pil(source[0]) if source is not None else None, least,
        )
        keep = vision_agent.best(scored, pick)
        names = [name for name in vision_agent.PASS if name in scored[0]]
        chosen = scored[keep]
        word = "edit" if source is not None else "image"
        report = {
            "least": {name: least[name] for name in names},
            "pictures": [
                {"index": i, "kept": i == keep, "passed": s["passed"], "scores": {n: s[n] for n in names},
                 "notes": s["notes"][:NOTES_SHOWN]}
                for i, s in enumerate(scored)
            ],
        }
        run_result.publish(
            status=run_result.OK if chosen["passed"] else run_result.WARNING,
            summary=f"{word} {keep} of {len(scored)} kept, {'passing' if chosen['passed'] else 'none passing'}",
            bodies=run_result.body("pictures", json.dumps(report, ensure_ascii=False)),
        )
        return io.NodeOutput(
            images[keep:keep + 1], keep, bool(chosen["passed"]), json.dumps(scored, ensure_ascii=False),
            chosen["notes"],
        )
