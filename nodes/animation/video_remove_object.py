"""Removing a masked object from a clip with the background its other frames show."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat import limits
from ...modules.compat.types import MOTION_MODEL

logger = log.get_logger("nodes.video_remove_object")

NODE_NAME = "Video Remove Object"

#: What both sides of the comparison on the node are written under in the temp folder.
PREFIX = "was.remove_object"


class VideoRemoveObject(io.ComfyNode):
    """Fill a masked object in every frame with background other frames show, carried along the motion."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASVideoRemoveObject",
            display_name=NODE_NAME,
            search_aliases=[
                "WASVideoRemoveObject", NODE_NAME,
                "object removal",
                "video inpaint",
                "remove person",
                "clean plate",
                "flow guided inpainting",
            ],
            category="WAS Suite/Animation",
            is_output_node=True,
            description=(
                "Remove a masked object from a clip with background other frames really show, "
                "carried along its motion. Suits locked-off shots, pans and subjects moving off "
                "their background; parallax, changing light, motion behind the object and shadows "
                "outside the mask degrade it. What no frame shows comes out as remaining: edit it on "
                "edit_frame and wire the result back as fill."
            ),
            inputs=[
                io.MultiType.Input(
                    "video",
                    [io.Image, io.Video],
                    tooltip=(
                        "The clip, as IMAGE frames or a VIDEO, at least two frames. Picture codes "
                        "or linear light, kept as given."
                    ),
                ),
                io.Mask.Input(
                    "mask",
                    tooltip=(
                        "The object, white: one mask per frame, such as Image Track Subject's, or one "
                        "for every frame. Include its shadow and reflection, or they stay."
                    ),
                ),
                io.Float.Input(
                    "grow",
                    default=1.0,
                    min=0.0,
                    max=10.0,
                    step=0.05,
                    tooltip=(
                        "Grows the mask, in hundredths of the short side: 0 = as drawn; 1 = 7 px at "
                        "720p, past a soft edge; 3 = motion blur and halos."
                    ),
                ),
                io.Int.Input(
                    "reach",
                    default=0,
                    min=0,
                    max=limits.max_resolution(),
                    tooltip=(
                        "Frames searched before and after each frame: 0 = the whole clip; 24 = one "
                        "second, when the light changes; 6 = near frames only, when the background "
                        "itself moves (crowds, water, machines), leaving the rest as remaining."
                    ),
                ),
                MOTION_MODEL.Input(
                    "motion_model",
                    optional=True,
                    tooltip=(
                        "A SEA-RAFT or FlowSeek network from Video Motion Model Loader to measure "
                        "the motion with. Empty = the built-in texture flow."
                    ),
                ),
                io.Image.Input(
                    "fill",
                    optional=True,
                    tooltip=(
                        "Second pass: frame edit_frame of this node's images after an image edit "
                        "filled its remaining area. Carried along the motion into every frame's "
                        "remaining area."
                    ),
                ),
                io.MultiType.Input(
                    io.String.Input("fill_frames", default="0", optional=True),
                    [io.String, io.Int],
                    optional=True,
                    tooltip=(
                        "Clip frame each fill picture shows, counting from 0: 12 for one; "
                        "0, 40, 80 for three. Wire edit_frame here."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip="The clip with the object filled: same size and count as video.",
                ),
                io.Mask.Output(
                    display_name="remaining",
                    tooltip=(
                        "1 where no frame showed the background and a smooth placeholder stands. "
                        "Hand it to an image edit or inpaint on edit_frame."
                    ),
                ),
                io.Int.Output(
                    display_name="edit_frame",
                    tooltip=(
                        "The frame to fill with an image edit and wire back as fill: central among "
                        "the frames with the most remaining area, so the fill travels least. 0 when "
                        "none is left."
                    ),
                ),
                io.Video.Output(
                    display_name="video",
                    tooltip=(
                        "images as a clip, at video's frame rate and with its audio when it is a "
                        "VIDEO, otherwise at 24 fps with none."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, video, mask, grow=1.0, reach=0, motion_model=None, fill=None,
                fill_frames="0") -> io.NodeOutput:
        """Fill the object and write both sides for the comparison on the node.

        Raises:
            ValueError: The clip holds fewer than two frames, the masks do not fit it, the mask is
                empty or covers a whole frame, or the fill pictures and fill_frames disagree.
            MemoryError: The result fits neither in memory nor on a scratch drive.
        """
        import comfy.model_management

        from ...modules.image import remove_object
        from ...modules.media import clip as clips

        clip = clips.open_clip(video, NODE_NAME)
        frames = clip.frames
        count, height, width = (int(side) for side in frames.shape[:3])
        if count < 2:
            raise ValueError(
                f"{NODE_NAME} fills each frame from the others, and the clip holds {count}. "
                f"Connect two frames or more; for a single picture use an image edit."
            )
        masks = remove_object.check_masks(mask, count, height, width, NODE_NAME)
        fills = None
        if fill is not None:
            if fill.ndim != 4 or int(fill.shape[0]) == 0:
                raise ValueError(f"{NODE_NAME} needs at least one picture on fill, or nothing wired there.")
            shape = (int(fill.shape[2]) / max(int(fill.shape[1]), 1)) / (width / max(height, 1))
            if abs(shape - 1.0) > 0.02:
                raise ValueError(
                    f"{NODE_NAME} was given a fill of {int(fill.shape[2])}x{int(fill.shape[1])} for "
                    f"frames of {width}x{height}. Use the edited frame at the clip's shape, such as "
                    f"Image Edit Restore's output."
                )
            indices = remove_object.fill_frames(fill, fill_frames, count, NODE_NAME)
            fills = {index: fill[position] for position, index in enumerate(indices)}
        grow_px = round(float(grow) / 100.0 * min(height, width))
        device = comfy.model_management.get_torch_device()
        step = clips.progress(3 * count - 2)
        try:
            removal = remove_object.remove(
                frames, masks, grow_px, int(reach), network=motion_model, fills=fills, device=device,
                progress=step, node=NODE_NAME,
            )
        finally:
            comfy.model_management.soft_empty_cache()
        masked = sum(1 for value in removal.holes if value)
        left = sum(removal.left)
        logger.info(
            "filled %d masked frame(s) of %d; %d pixel(s) left without a source, most on frame %d",
            masked, count, left, removal.edit_frame,
        )
        answer = clips.rebuild(clip, removal.images)
        ui = clips.compare(clips.rebuild(clip, frames, audio=None), answer, PREFIX)
        return io.NodeOutput(removal.images, removal.remaining, removal.edit_frame, answer, ui=ui)
