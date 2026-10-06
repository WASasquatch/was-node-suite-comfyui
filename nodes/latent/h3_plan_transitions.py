"""How each MiniMax H3 scene follows the one before, chosen by a language model reading the scenes."""

from __future__ import annotations

import json

from comfy_api.latest import io

from ...modules import log
from ...modules.interface import run_result
from ...modules.latent import h3_writer

logger = log.get_logger("nodes.h3_plan_transitions")

NODE_NAME = "MiniMax H3 Plan Transitions"

#: The largest seed the sampler takes.
MAX_SEED = 0xFFFFFFFFFFFFFFFF


class H3PlanTransitions(io.ComfyNode):
    """Each scene's transition into the next, read from the scenes by a language model."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASH3PlanTransitions",
            display_name=NODE_NAME,
            search_aliases=[
                "WASH3PlanTransitions",
                NODE_NAME,
                "h3 transitions",
                "plan cuts",
                "cut or carry",
                "minimax h3",
                "prompt timeline",
                "llm",
            ],
            category="WAS Suite/Latent/Video",
            description=(
                "Reads the scenes of a MiniMax H3 video with a language model and chooses how "
                "each follows the one before: a cut, a carry of the same shot, a cut that keeps "
                "the sound, or a cut that keeps the cast with or without the sound. The Prompt "
                "Timeline's Plan transitions runs it and sets each row's transition and overlap."
            ),
            inputs=[
                io.Clip.Input(
                    "vlm_clip",
                    tooltip=(
                        "The language model that reads the scenes, from Load CLIP, as "
                        "`qwen3vl_8b_fp8_scaled.safetensors` or `qwen_3_8b_fp8mixed.safetensors`."
                    ),
                ),
                io.String.Input(
                    "scenes", multiline=True, default="",
                    tooltip=(
                        "The scene prompts in order, divided by a line of dashes as MiniMax H3 "
                        "Scene Writer's prompts output, or its scenes JSON."
                    ),
                ),
                io.Int.Input(
                    "seed", default=0, min=0, max=MAX_SEED, control_after_generate=True,
                    tooltip="Which draw of the model's choices to take, as `0` or `42`.",
                ),
                io.Float.Input(
                    "temperature", default=0.3, min=0.01, max=2.0, step=0.01, optional=True, advanced=True,
                    tooltip="How adventurous each pick is. `0.3` keeps to the likeliest reading, `0.8` varies it.",
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
                io.String.Output(
                    display_name="transitions",
                    tooltip=(
                        "One entry per scene from the second, as JSON: scene number, transition, "
                        "the continuity and overlap it sets, and why."
                    ),
                ),
                io.String.Output(display_name="report", tooltip="How many of each transition were chosen."),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, vlm_clip, scenes, seed, temperature=0.3, top_k=64, top_p=0.95, min_p=0.05,
                repetition_penalty=1.05, thinking=False, fast_decode=True) -> io.NodeOutput:
        """Read the scenes and choose each transition.

        Raises:
            ValueError: No model arrived, or fewer than two scenes did.
        """
        if vlm_clip is None:
            raise ValueError(
                f"{NODE_NAME} needs a language model. Wire Load CLIP with a model such as "
                f"qwen3vl_8b_fp8_scaled.safetensors into vlm_clip"
            )
        found = h3_writer.split_scenes(scenes)
        if len(found) < 2:
            raise ValueError(
                f"{NODE_NAME} needs at least two scenes and read {len(found)}. Divide the prompts "
                f"with a line of dashes, or wire MiniMax H3 Scene Writer's prompts output in"
            )
        sampling = h3_writer.Sampling(float(temperature), int(top_k), float(top_p), float(min_p),
                                      float(repetition_penalty), bool(thinking), bool(fast_decode))
        write = h3_writer.make_writer(vlm_clip, int(seed), sampling)
        planned = h3_writer.plan(write, found)
        joins = {}
        for choice in planned:
            joins[choice["transition"]] = joins.get(choice["transition"], 0) + 1
        report = f"{len(found)} scenes: " + ", ".join(f"{count} {name}" for name, count in joins.items())
        text = json.dumps(planned, ensure_ascii=False)
        logger.info("%s: %s", NODE_NAME, report)
        run_result.publish(
            summary=report,
            counts={"scenes": len(found), **{name: count for name, count in joins.items()}},
            bodies=run_result.body("transitions", "\n".join(
                f"{choice['scene']}. {choice['transition']}" + (f": {choice['why']}" if choice["why"] else "")
                for choice in planned
            )),
        )
        return io.NodeOutput(text, report, ui={"was_h3_transitions": [text]})
