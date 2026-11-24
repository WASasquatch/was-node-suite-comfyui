"""Scene-structured attention for MiniMax H3 clips cut into several shots."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.interface import channel, run_result
from ...modules.sampling import h3_scene_attention as h3_scenes

logger = log.get_logger("nodes.h3_scene_attention")

NODE_NAME = "H3 Scene Attention"


class H3SceneAttention(io.ComfyNode):
    """Make each scene of a multi-shot MiniMax H3 clip attend to its own shot."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASH3SceneAttention",
            display_name=NODE_NAME,
            search_aliases=[
                "WASH3SceneAttention", NODE_NAME,
                "scene attention",
                "multi shot",
                "multi scene",
                "shot cuts",
                "onereel",
                "minimax h3",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Make each shot of a multi-shot MiniMax H3 clip follow its own part of the prompt. "
                "Each scene attends to its own frames and shot text, plus a few frames shared from "
                "the other scenes for consistent characters, which also makes long clips faster. "
                "Scenes come from the prompt's timed shots, as [Shot 2] At 00:03.000."
            ),
            inputs=[
                io.Model.Input("model", tooltip="The MiniMax H3 video model, after any LoRA."),
                io.String.Input(
                    "scenes",
                    default="",
                    tooltip=(
                        "Where scenes start: empty = default, read from the prompt's timed shots; "
                        "`3.5, 7` = cuts at 3.5 s and 7 s, for prompts written without shot times."
                    ),
                ),
                io.Int.Input(
                    "shared_frames",
                    default=h3_scenes.SHARED_FRAMES,
                    min=0,
                    max=h3_scenes.MAX_SHARED_FRAMES,
                    tooltip=(
                        "Latent frames each scene reads from the other scenes, picked afresh each "
                        "step: `4` = default; `0` = scenes fully apart, fastest, faces may drift; "
                        "`8` = closer identities, slower."
                    ),
                ),
                io.Combo.Input(
                    "positions",
                    options=list(h3_scenes.POSITIONS),
                    default=h3_scenes.POSITIONS[0],
                    tooltip=(
                        "`global` = default, frames and sound keep their place in the clip; "
                        "`scene` = each scene's frames and sound counted from its own start."
                    ),
                ),
                io.Int.Input(
                    "select_block",
                    default=25,
                    min=0,
                    max=49,
                    advanced=True,
                    tooltip=(
                        "Block whose input picks the shared frames: `25` = default, the middle of "
                        "the 50."
                    ),
                ),
                io.Float.Input(
                    "start_percent",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                    tooltip="Point in the schedule scene attention starts: `0` = default, the first step.",
                ),
                io.Float.Input(
                    "end_percent",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    advanced=True,
                    tooltip="Point in the schedule scene attention stops: `1.0` = default, the last step.",
                ),
            ],
            outputs=[
                io.Model.Output(
                    display_name="model",
                    tooltip="The model attending scene by scene, for a guider or sampler.",
                ),
            ],
        )

    @classmethod
    def execute(cls, model, scenes="", shared_frames=h3_scenes.SHARED_FRAMES,
                positions=h3_scenes.POSITIONS[0], select_block=25, start_percent=0.0,
                end_percent=1.0) -> io.NodeOutput:
        """Patch every block.

        Raises:
            ValueError: The model is not MiniMax H3, or scenes cannot be read.
        """
        cuts = h3_scenes.cut_times(scenes)
        node_id = channel.executing_node_id()
        patched = h3_scenes.patched(model, cuts, positions, shared_frames, select_block,
                                  start_percent, end_percent, node_id)
        if run_result.watching():
            run_result.publish(
                status=run_result.OK,
                summary=(f"cuts at {', '.join(f'{cut:g} s' for cut in cuts)}" if cuts
                         else "scenes are read from the prompt's timed shots when sampling starts"),
                facts={"shared frames": int(shared_frames), "positions": positions},
            )
        return io.NodeOutput(patched)

