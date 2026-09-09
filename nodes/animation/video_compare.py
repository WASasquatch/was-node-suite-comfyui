"""Two videos under a divider, drawn on the node that received them."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.media.temp_video import to_temp

#: What each side is written under in the temp folder.
PREFIX_A = "was.compare.video.a"
PREFIX_B = "was.compare.video.b"


class VideoCompare(io.ComfyNode):
    """Play two videos on one node, split by a divider that drags across."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASVideoCompare",
            display_name="Compare Video",
            search_aliases=[
                "WASVideoCompare",
                "Compare Video",
                "video compare",
                "video difference",
                "before after video",
                "wipe",
            ],
            category="WAS Suite/Animation",
            description=(
                "Play two videos on the node under a divider that drags left and right, so "
                "one run can be judged against another at the same frame. Both play from "
                "one clock. Wire a render into each socket; nothing is passed on."
            ),
            is_output_node=True,
            inputs=[
                io.Video.Input(
                    "video_a",
                    optional=True,
                    tooltip="The video drawn left of the divider.",
                ),
                io.Video.Input(
                    "video_b",
                    optional=True,
                    tooltip="The video drawn right of the divider.",
                ),
            ],
            outputs=[],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, video_a=None, video_b=None) -> io.NodeOutput:
        """Write both sides to the temp folder for the node's own player.

        Args:
            video_a: The video drawn left of the divider, or None.
            video_b: The video drawn right of the divider, or None.

        Returns:
            An output carrying one entry per side that was written.
        """
        sides = {"a_video": [], "b_video": []}
        for key, video, prefix in (
            ("a_video", video_a, PREFIX_A),
            ("b_video", video_b, PREFIX_B),
        ):
            if video is None:
                continue
            written = to_temp(video, prefix)
            if written is not None:
                sides[key].append(written)
        return io.NodeOutput(ui=sides)
