"""Removing dust and dirt that sits on single frames of a clip, repaired from the frames around it."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import MOTION_MODEL
from ...modules.image import motion as motion_field
from ...modules.image import temporal_restore, video_dust

logger = log.get_logger("nodes.video_remove_dust")

NODE_NAME = "Video Remove Dust"

#: What both sides of the comparison on the node are written under in the temp folder.
PREFIX = "was.remove_dust"


class VideoRemoveDust(io.ComfyNode):
    """Find one-frame specks and repair them from the neighbouring frames."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASVideoRemoveDust",
            display_name=NODE_NAME,
            search_aliases=[
                "WASVideoRemoveDust", NODE_NAME,
                "dust busting",
                "dirt removal",
                "film restoration",
                "despeckle",
                "blotch removal",
            ],
            category="WAS Suite/Animation",
            is_output_node=True,
            description=(
                "Remove dust and dirt that lands on single frames of a scanned film. A speck is "
                "found where a frame differs from both neighbours the same way while they agree, "
                "and is repaired with their own pixels lined up along the motion. Fast-moving "
                "edges are left alone."
            ),
            inputs=[
                io.MultiType.Input(
                    "video",
                    [io.Video, io.Image],
                    tooltip=(
                        "The clip, as a VIDEO or IMAGE frames in picture codes, three frames or "
                        "more. A VIDEO gives the video output its frame rate and audio."
                    ),
                ),
                io.Float.Input(
                    "sensitivity",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How faint a speck may be and still be found: 0.2 = bold specks only; "
                        "0.5 = default; 0.9 = faint specks too, and more risk on busy motion."
                    ),
                ),
                io.Int.Input(
                    "max_size",
                    default=24,
                    min=1,
                    max=256,
                    tooltip=(
                        "Widest speck repaired, in pixels: 8 = pinpricks only; 24 = default; "
                        "96 = large blotches on a 4K scan."
                    ),
                ),
                io.Combo.Input(
                    "polarity",
                    options=list(video_dust.POLARITIES),
                    default="light and dark",
                    tooltip=(
                        "Which specks to look for: 'light' = dust printed from a negative; "
                        "'dark' = dust on a slide or print; 'light and dark' = both."
                    ),
                ),
                io.Int.Input(
                    "motion_resolution",
                    default=motion_field.MOTION_SIDE,
                    min=0,
                    max=4096,
                    tooltip=(
                        "Long side, in pixels, the motion is measured at: 0 = the clip's own "
                        "size; 512 = fastest; 768 = default; 1280 = small, fine motion."
                    ),
                ),
                MOTION_MODEL.Input(
                    "motion_model",
                    optional=True,
                    tooltip=(
                        "A SEA-RAFT or FlowSeek network from Video Motion Model Loader to line "
                        "the frames up with. Empty = the built-in texture flow, slower and "
                        "finding fewer specks on moving areas."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip="The repaired frames, the same size and count as the clip, in picture codes.",
                ),
                io.Video.Output(
                    display_name="video",
                    tooltip=(
                        "The repaired frames as a clip, at the source's frame rate and with its "
                        "audio when it is a VIDEO, otherwise at 24 fps with none."
                    ),
                ),
                io.Mask.Output(
                    display_name="mask",
                    tooltip="White where a speck was repaired, soft at its edge, one mask per frame.",
                ),
            ],
        )

    @classmethod
    def execute(cls, video, sensitivity=0.5, max_size=24, polarity="light and dark",
                motion_resolution=motion_field.MOTION_SIDE, motion_model=None) -> io.NodeOutput:
        """Repair the clip's specks and write both sides for the comparison on the node.

        Raises:
            ValueError: The clip holds fewer than three frames or fewer than three channels, or
                ``polarity`` is not on the menu.
            MemoryError: Neither free memory nor a scratch drive can hold the result.
        """
        import comfy.model_management

        from ...modules.media import clip as clips

        if polarity not in video_dust.POLARITIES:
            raise ValueError(
                f"{NODE_NAME} was given polarity '{polarity}'. Pick one of: "
                f"{', '.join(video_dust.POLARITIES)}."
            )
        source = clips.open_clip(video, NODE_NAME)
        frames = source.frames
        count = int(frames.shape[0])
        if count < 3:
            raise ValueError(
                f"{NODE_NAME} compares each frame with two others and the clip holds {count}. "
                f"Connect a clip of three frames or more."
            )
        if int(frames.shape[-1]) < 3:
            raise ValueError(
                f"{NODE_NAME} needs colour frames and these hold {int(frames.shape[-1])} channel(s). "
                f"Convert them to RGB first."
            )
        device = comfy.model_management.get_torch_device()
        flows = temporal_restore.Flows(frames, int(motion_resolution), device, motion_model)
        repaired, mask, report = video_dust.remove_dust(
            frames, flows, sensitivity=float(sensitivity), max_size=int(max_size),
            polarity=polarity, device=device, progress=clips.progress(count), node=NODE_NAME,
        )
        logger.info(
            "repaired %d speck(s) over %d frame(s) with %s, %d left alone on fast motion or "
            "found again in a neighbour; %.4f%% of the clip changed",
            sum(report.specks), count, flows.engine, sum(report.refused),
            100.0 * sum(report.area) / max(len(report.area), 1),
        )
        shown, alpha = repaired, source.alpha
        if int(repaired.shape[-1]) > 3:
            shown, alpha = repaired[..., :3], repaired[..., 3]
        answer = clips.rebuild(source, shown, alpha=alpha)
        ui = clips.compare(clips.rebuild(source, frames[..., :3], audio=None), answer, PREFIX)
        return io.NodeOutput(repaired, answer, mask, ui=ui)
