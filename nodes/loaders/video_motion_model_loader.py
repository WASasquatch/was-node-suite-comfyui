"""Loading a learned optical flow network for Video Motion."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import MOTION_MODEL
from ...modules.model import flow_networks

NODE_NAME = "Video Motion Model Loader"

#: Shown in the checkpoint list when the model folder holds none, so the widget has
#: something to draw.
NO_CHECKPOINT = "put a checkpoint in models/optical_flow"

#: Refinement passes the checkpoints were published with, fast and accurate.
FAST = 4
ACCURATE = 12


class VideoMotionModelLoader(io.ComfyNode):
    """Build a SEA-RAFT or FlowSeek optical flow network for Video Motion."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        found = flow_networks.offered()
        return io.Schema(
            node_id="WASVideoMotionModelLoader",
            display_name=NODE_NAME,
            search_aliases=[
                "WASVideoMotionModelLoader", NODE_NAME,
                "SEA-RAFT",
                "FlowSeek",
                "RAFT",
                "optical flow model",
                "flow network",
            ],
            category="WAS Suite/Loaders",
            description=(
                "Build a SEA-RAFT or FlowSeek optical flow network for the motion_model input "
                "of Video Motion, which then measures the clip's motion with it. Checkpoints "
                "are read from ComfyUI/models/optical_flow; with features.network on, a listed "
                "checkpoint not yet there is fetched on first use."
            ),
            inputs=[
                io.Combo.Input(
                    "checkpoint",
                    options=found or [NO_CHECKPOINT],
                    tooltip=(
                        "Which weights to build: 'sea_raft_m_spring' = default; "
                        "'sea_raft_s_spring' = about 1.3x faster; 'sea_raft_m_ct' = general "
                        "purpose; 'flowseek_t_ct' = reads depth too, about 1.5x slower. Files "
                        "already on disk are listed first."
                    ),
                ),
                io.Int.Input(
                    "iterations",
                    default=FAST,
                    min=1,
                    max=32,
                    tooltip=(
                        f"Refinement passes per frame pair: {FAST} = default, the published fast "
                        f"setting; {ACCURATE} = the published accurate setting, 1.5 to 2x slower."
                    ),
                ),
            ],
            outputs=[
                MOTION_MODEL.Output(
                    display_name="motion_model",
                    tooltip="The built network, for the motion_model input of Video Motion.",
                ),
            ],
        )

    @classmethod
    def execute(cls, checkpoint, iterations=FAST) -> io.NodeOutput:
        """Build the network for ``checkpoint``.

        Raises:
            ValueError: No checkpoint is on disk and none can be fetched, or the file holds
                neither SEA-RAFT nor FlowSeek weights.
            ModelUnavailable: The chosen checkpoint is not on disk.
        """
        if checkpoint == NO_CHECKPOINT or not checkpoint:
            raise ValueError(
                f"{NODE_NAME} has no weights to build. Either set "
                f"{flow_networks.NETWORK_FEATURE}: true in config.yaml and let the first run "
                f"fetch one, or put a SEA-RAFT or FlowSeek checkpoint in "
                f"ComfyUI/models/{flow_networks.FOLDER}. Either way, restart ComfyUI so the "
                f"list is rebuilt."
            )
        built = flow_networks.backend(checkpoint)
        return io.NodeOutput(
            flow_networks.FlowNetwork(
                backend=built, name=checkpoint, family=built.name, iterations=int(iterations)
            )
        )
