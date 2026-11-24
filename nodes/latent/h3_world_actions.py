"""Write keyboard and camera actions over time into a MiniMax H3 first-frame conditioning."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.interface import run_result
from ...modules.latent import h3_world

logger = log.get_logger("nodes.h3_world_actions")

NODE_NAME = "H3 World Actions"

#: The script a new node opens with: walking forward for the 124 frame clip.
DEFAULT_ACTIONS = "0-5.2: forward"


def _with_script(conditioning, value) -> list:
    """A conditioning list with the action script on every entry."""
    return [[tensor, {**options, h3_world.COND_KEY: value}] for tensor, options in conditioning]


class H3WorldActions(io.ComfyNode):
    """Drive a first-frame H3 video with keyboard and camera actions."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASH3WorldActions",
            display_name=NODE_NAME,
            search_aliases=[
                "WASH3WorldActions", NODE_NAME, "h3-world", "keyboard", "wasd", "camera pan",
                "action timeline", "world model",
            ],
            category="WAS Suite/Latent/Video",
            description=(
                "Say what the character and the camera do over time, as held keys on a timeline: "
                "walk, strafe, pan, tilt. Each video latent reads one sentence of what is held. "
                "Sample with the model from H3 World."
            ),
            inputs=[
                io.Clip.Input(
                    "clip",
                    tooltip="The MiniMax H3 text encoder, qwen3vl_32b_minimax_h3 from Load CLIP.",
                ),
                io.Conditioning.Input(
                    "positive",
                    tooltip="MiniMax H3 Image To Video's positive, with the first frame and a prompt describing the scene.",
                ),
                io.String.Input(
                    "actions",
                    multiline=True,
                    default=DEFAULT_ACTIONS,
                    tooltip=(
                        "Held keys over time, in seconds: `0-2: forward; 1-3: pan-left fast`. "
                        "Overlaps hold together; gaps stand still. Keys W A S D move, J L pan, I K tilt, F fast."
                    ),
                ),
                io.Conditioning.Input(
                    "negative",
                    optional=True,
                    tooltip="Only for CFG above 1: a negative that gets the standing-still sentence on every latent.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(
                    display_name="positive",
                    tooltip="The positive with the action sentences, bound to their latents by H3 World's model.",
                ),
                io.Conditioning.Output(
                    display_name="negative",
                    tooltip="The negative with the standing-still sentence on every latent; blocked when none is wired.",
                ),
            ],
            hidden=[io.Hidden.unique_id],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, clip, positive, actions=DEFAULT_ACTIONS, negative=None) -> io.NodeOutput:
        """Encode the sentences and put them on the conditioning.

        Raises:
            ValueError: The script cannot be read.
        """
        from comfy_execution.graph_utils import ExecutionBlocker

        node_id = cls.hidden.unique_id
        script = h3_world.script(clip, actions, node_id=node_id)
        positive = _with_script(positive, script)
        if negative is not None:
            negative = _with_script(negative, h3_world.script(clip, "", null=True))
        else:
            negative = ExecutionBlocker(
                f"{NODE_NAME} has no negative wired. Wire MiniMax H3's negative into it for CFG above 1.")
        entries = h3_world.parse(actions)
        distinct = len(set(script.sentences)) or 1
        run_result.publish(
            summary=(f"{len(entries)} entr{'ies' if len(entries) != 1 else 'y'}, {distinct} "
                     f"sentence{'s' if distinct != 1 else ''}; bound to latents when sampled"),
            counts={"entries": len(entries), "sentences": distinct,
                    "latents": len(script.sentences)},
            items=h3_world.describe(list(script.sentences)) or [f"every latent: {script.tail}"],
        )
        logger.info("%s: %d entries, %d sentences encoded", NODE_NAME, len(entries), len(script.encoded))
        return io.NodeOutput(positive, negative)
