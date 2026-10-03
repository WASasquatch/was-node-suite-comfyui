"""Splitting a clip into its scenes at every cut found in its motion."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import MOTION

logger = log.get_logger("nodes.video_split_scenes")

NODE_NAME = "Video Split Scenes"

#: Scene outputs the node declares; scenes past the last are reached through ``scenes``.
SCENE_SLOTS = 24

#: Default agreement below which a pair of frames counts as a cut.
DEFAULT_THRESHOLD = 0.4

class VideoSplitScenes(io.ComfyNode):
    """Find the cuts in a clip and answer each scene as a clip of its own."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASVideoSplitScenes",
            display_name=NODE_NAME,
            search_aliases=[
                "WASVideoSplitScenes", NODE_NAME,
                "scene detection",
                "scene cuts",
                "shot boundary",
                "split video",
                "cut detection",
            ],
            category="WAS Suite/Animation",
            is_output_node=True,
            description=(
                "Find every cut in a clip, where one shot ends and the next begins, and answer "
                "each scene as a clip of its own with its stretch of the audio: one per "
                "output, all of them as a list, and where each starts. The panel shows the "
                "first frame of every scene."
            ),
            inputs=[
                io.Video.Input("video", tooltip="The clip to split, at least two frames."),
                io.Float.Input(
                    "cut_threshold",
                    default=DEFAULT_THRESHOLD,
                    min=0.05,
                    max=0.95,
                    step=0.01,
                    tooltip=(
                        "Share of the picture that must still follow from the frame before for "
                        "the shot to continue: 0.4 = default; 0.6 = finds softer cuts, and can "
                        "split fast action; 0.2 = only hard cuts."
                    ),
                ),
                io.Int.Input(
                    "min_scene_frames",
                    default=6,
                    min=1,
                    max=10000,
                    tooltip=(
                        "Fewest frames a scene holds: 6 = default; 1 = every cut, flashes "
                        "included; 24 = one second at 24 fps."
                    ),
                ),
                MOTION.Input(
                    "motion",
                    optional=True,
                    tooltip=(
                        "Motion from Video Motion, measured from this same clip at any size. Left empty, it "
                        "is measured here at 768 px on the long side."
                    ),
                ),
            ],
            outputs=[
                io.Int.Output(display_name="scene_count", tooltip="How many scenes were found, 1 or more."),
                io.Int.Output(
                    display_name="starts",
                    is_output_list=True,
                    tooltip="The first frame of every scene, counting from 0: [0, 48, 96].",
                ),
                io.Video.Output(
                    display_name="scenes",
                    is_output_list=True,
                    tooltip="Every scene as a list, so the node after runs once per scene.",
                ),
                io.Video.Output(
                    display_name="scene_1",
                    tooltip="Scene 1, with its audio; blocked when the clip has fewer scenes.",
                ),
                io.Video.Output(
                    display_name="scene_2",
                    tooltip="Scene 2, with its audio; blocked when the clip has fewer scenes.",
                ),
                io.Video.Output(
                    display_name="scene_3",
                    tooltip="Scene 3, with its audio; blocked when the clip has fewer scenes.",
                ),
                io.Video.Output(
                    display_name="scene_4",
                    tooltip="Scene 4, with its audio; blocked when the clip has fewer scenes.",
                ),
                io.Video.Output(
                    display_name="scene_5",
                    tooltip="Scene 5, with its audio; blocked when the clip has fewer scenes.",
                ),
                io.Video.Output(
                    display_name="scene_6",
                    tooltip="Scene 6, with its audio; blocked when the clip has fewer scenes.",
                ),
                io.Video.Output(
                    display_name="scene_7",
                    tooltip="Scene 7, with its audio; blocked when the clip has fewer scenes.",
                ),
                io.Video.Output(
                    display_name="scene_8",
                    tooltip="Scene 8, with its audio; blocked when the clip has fewer scenes.",
                ),
                io.Video.Output(
                    display_name="scene_9",
                    tooltip="Scene 9, with its audio; blocked when the clip has fewer scenes.",
                ),
                io.Video.Output(
                    display_name="scene_10",
                    tooltip="Scene 10, with its audio; blocked when the clip has fewer scenes.",
                ),
                io.Video.Output(
                    display_name="scene_11",
                    tooltip="Scene 11, with its audio; blocked when the clip has fewer scenes.",
                ),
                io.Video.Output(
                    display_name="scene_12",
                    tooltip="Scene 12, with its audio; blocked when the clip has fewer scenes.",
                ),
                io.Video.Output(
                    display_name="scene_13",
                    tooltip="Scene 13, with its audio; blocked when the clip has fewer scenes.",
                ),
                io.Video.Output(
                    display_name="scene_14",
                    tooltip="Scene 14, with its audio; blocked when the clip has fewer scenes.",
                ),
                io.Video.Output(
                    display_name="scene_15",
                    tooltip="Scene 15, with its audio; blocked when the clip has fewer scenes.",
                ),
                io.Video.Output(
                    display_name="scene_16",
                    tooltip="Scene 16, with its audio; blocked when the clip has fewer scenes.",
                ),
                io.Video.Output(
                    display_name="scene_17",
                    tooltip="Scene 17, with its audio; blocked when the clip has fewer scenes.",
                ),
                io.Video.Output(
                    display_name="scene_18",
                    tooltip="Scene 18, with its audio; blocked when the clip has fewer scenes.",
                ),
                io.Video.Output(
                    display_name="scene_19",
                    tooltip="Scene 19, with its audio; blocked when the clip has fewer scenes.",
                ),
                io.Video.Output(
                    display_name="scene_20",
                    tooltip="Scene 20, with its audio; blocked when the clip has fewer scenes.",
                ),
                io.Video.Output(
                    display_name="scene_21",
                    tooltip="Scene 21, with its audio; blocked when the clip has fewer scenes.",
                ),
                io.Video.Output(
                    display_name="scene_22",
                    tooltip="Scene 22, with its audio; blocked when the clip has fewer scenes.",
                ),
                io.Video.Output(
                    display_name="scene_23",
                    tooltip="Scene 23, with its audio; blocked when the clip has fewer scenes.",
                ),
                io.Video.Output(
                    display_name="scene_24",
                    tooltip="Scene 24, with its audio; blocked when the clip has fewer scenes.",
                ),
            ],
        )

    @classmethod
    def execute(cls, video, cut_threshold=DEFAULT_THRESHOLD, min_scene_frames=6, motion=None) -> io.NodeOutput:
        """Split the clip at its cuts.

        Raises:
            ValueError: The clip holds fewer than two frames, or the motion was measured from
                another clip.
        """
        import comfy.model_management
        from comfy_execution.graph_utils import ExecutionBlocker

        from ...modules.interface import preview as published
        from ...modules.interface import run_result
        from ...modules.media import clip as clips
        from ...modules.media import scenes

        source = clips.open_clip(video, NODE_NAME)
        count = int(source.frames.shape[0])
        if count < 2:
            raise ValueError(
                f"{NODE_NAME} looks for cuts between frames and the clip holds {count}. "
                f"Connect a clip of two frames or more."
            )
        device = comfy.model_management.get_torch_device()
        step = clips.progress(count - 1)
        motion = clips.motion_for(source, motion, NODE_NAME, device, step)

        first_frames = scenes.starts(motion.cuts(float(cut_threshold)), int(min_scene_frames))
        pieces = []
        for start, stop in scenes.ranges(first_frames, count):
            alpha = source.alpha[start:stop] if source.alpha is not None else None
            audio = clips.slice_audio(source.audio, start, stop, source.rate)
            pieces.append(clips.rebuild(source, source.frames[start:stop], alpha=alpha, audio=audio))

        held = sum(motion.holds())
        if len(pieces) > SCENE_SLOTS:
            logger.info(
                "%d scenes found; scenes past %d are on the scenes list only",
                len(pieces), SCENE_SLOTS,
            )
        published.publish_output(scenes.contact_sheet(source.frames, first_frames))
        cut_frames = ", ".join(str(frame) for frame in first_frames[1:]) or "none"
        run_result.publish(
            summary=f"{len(pieces)} scene(s) in {count} frames",
            counts={"scenes": len(pieces), "cuts": len(pieces) - 1, "held frames": held},
            facts={
                "cuts at": cut_frames,
                "frames": count,
                "rate": f"{float(source.rate):.6g} fps",
            },
        )

        slots = [
            pieces[number] if number < len(pieces) else ExecutionBlocker(
                f"{NODE_NAME} found {len(pieces)} scene(s), so scene_{number + 1} is empty."
            )
            for number in range(SCENE_SLOTS)
        ]
        return io.NodeOutput(len(pieces), first_frames, pieces, *slots)
