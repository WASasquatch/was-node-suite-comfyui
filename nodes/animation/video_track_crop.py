"""A fixed-size crop that follows a masked subject through a clip, for restoring it apart."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import VIDEO_TRACK

logger = log.get_logger("nodes.video_track_crop")

NODE_NAME = "Video Track Crop"

#: What both sides of the comparison on the node are written under in the temp folder.
PREFIX = "was.track_crop"

#: Share of the frame's brightness kept outside the window on the preview.
OUTSIDE_SHADE = 0.35

#: Share of a window that must lie inside the frame for its crop to be the reference.
WHOLE_SHARE = 0.98

#: Longest side of the reference crop, in pixels.
REFERENCE_LIMIT = 2048


class VideoTrackCrop(io.ComfyNode):
    """Cut a fixed-size window that follows the masked subject out of every frame."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASVideoTrackCrop",
            display_name=NODE_NAME,
            search_aliases=[
                "WASVideoTrackCrop", NODE_NAME,
                "face crop video",
                "follow crop",
                "face restore",
                "track and crop",
                "stabilised crop",
            ],
            category="WAS Suite/Animation",
            is_output_node=True,
            description=(
                "Cut the subject a mask follows, such as a face, out of every frame at one fixed "
                "size, centred and steadied, however far away it is. Restore or edit the crops as "
                "their own clip, then lay them back with Video Track Paste."
            ),
            inputs=[
                io.Video.Input("video", tooltip="The clip to crop from."),
                io.Mask.Input(
                    "mask",
                    tooltip=(
                        "The subject on every frame: white = subject. Wire Image Track Subject's mask; "
                        "frames where it is empty take the window from the frames around them."
                    ),
                ),
                io.Int.Input(
                    "width",
                    default=512,
                    min=64,
                    max=4096,
                    step=32,
                    tooltip="Width of every crop in pixels: 512 = MiniMax H3 at a fast size; 768 = more detail.",
                ),
                io.Int.Input(
                    "height",
                    default=512,
                    min=64,
                    max=4096,
                    step=32,
                    tooltip="Height of every crop in pixels: 512 with width 512 = square.",
                ),
                io.Float.Input(
                    "margin",
                    default=2.0,
                    min=1.0,
                    max=8.0,
                    step=0.05,
                    tooltip=(
                        "Window size against the subject's: 1.3 = the face alone; 2 = face, hair and "
                        "neck; 4 = head and shoulders."
                    ),
                ),
                io.Float.Input(
                    "smoothing",
                    default=0.25,
                    min=0.0,
                    max=5.0,
                    step=0.05,
                    tooltip="Seconds of movement averaged in the window's path and size: 0 = locked to the mask; 0.25 = steady; 1 = slow drift.",
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="crops",
                    tooltip="Every frame's window at width x height, the subject centred at one size.",
                ),
                io.Mask.Output(
                    display_name="masks",
                    tooltip="The mask cut out alongside, at the crop's size: wire it to Video Track Paste's masks.",
                ),
                io.Image.Output(
                    display_name="reference",
                    tooltip=(
                        "The subject from the frame that shows it in most detail, whole and in frame, "
                        "cut at that frame's own resolution up to 2048 px: the identity reference for a restore."
                    ),
                ),
                VIDEO_TRACK.Output(
                    display_name="track",
                    tooltip="Where the window sat on every frame, for Video Track Paste.",
                ),
                io.Video.Output(
                    display_name="preview",
                    tooltip="The clip with the window outlined and the rest dimmed.",
                ),
            ],
        )

    @classmethod
    def execute(cls, video, mask, width=512, height=512, margin=2.0, smoothing=0.25) -> io.NodeOutput:
        """Plan the window, cut every frame and write both sides for the comparison on the node.

        Raises:
            ValueError: The mask holds a different number of frames than the clip, or no frame's
                mask holds the subject.
            MemoryError: Neither free memory nor a scratch drive can hold the result.
        """
        import torch

        import comfy.model_management

        from ...modules.image import scratch
        from ...modules.media import clip as clips
        from ...modules.media import track_crop

        source = clips.open_clip(video, NODE_NAME)
        frames = source.frames
        count, frame_h, frame_w = (int(v) for v in frames.shape[:3])
        if int(mask.shape[0]) not in (1, count):
            raise ValueError(
                f"The mask batch holds {int(mask.shape[0])} frame(s) and the clip {count}. "
                f"Connect one mask per frame of the clip, as Image Track Subject answers."
            )
        size = (int(height), int(width))
        try:
            track = track_crop.plan(
                mask, count, (frame_h, frame_w), size, float(margin), float(smoothing) * float(source.rate)
            )
        except ValueError:
            raise ValueError(
                f"{NODE_NAME} found no subject: the mask is empty on every frame. Check Image Track "
                f"Subject found it, or wire a mask that covers it."
            ) from None
        device = comfy.model_management.get_torch_device()
        advice = "A shorter clip or a smaller crop also fits it."
        crops = scratch.allocate((count, size[0], size[1], 3), torch.float32, NODE_NAME, advice)
        masks = scratch.allocate((count, size[0], size[1]), torch.float32, NODE_NAME, advice)
        shown = scratch.allocate(tuple(frames[..., :3].shape), frames.dtype, NODE_NAME, advice)
        step = clips.progress(count)
        scores = []
        for index in range(count):
            cut, inside = track_crop.crop_one(frames[index, ..., :3], track, index, device)
            crops[index] = cut.cpu()
            plane = mask[min(index, int(mask.shape[0]) - 1)].unsqueeze(-1)
            cut_mask, _inside = track_crop.crop_one(plane, track, index, device)
            masks[index] = cut_mask[..., 0].cpu()
            whole = track.found[index] and float(inside.mean()) >= WHOLE_SHARE
            scores.append(track_crop.detail(cut, cut_mask[..., 0]) if whole else -1.0)
            shown[index] = _marked(frames[index, ..., :3], track.box(index)).to(shown.dtype)
            step(1)
        if max(scores) > 0.0:
            track.reference = max(range(count), key=lambda index: scores[index])

        magnify = [1.0 / track.scale(index) for index in range(count)]
        missing = count - sum(track.found)
        logger.info(
            "track crop: %d frame(s) at %dx%d, magnified %.2fx to %.2fx, %d frame(s) filled from "
            "their neighbours, reference frame %d",
            count, size[1], size[0], min(magnify), max(magnify), missing, track.reference,
        )
        preview = clips.rebuild(source, shown, audio=None)
        best = track.reference
        reference, _inside = track_crop.crop_one(
            frames[best, ..., :3], track_crop.native(track, best, REFERENCE_LIMIT), 0, device
        )
        reference = reference.unsqueeze(0).cpu()
        return io.NodeOutput(
            crops, masks, reference, track, preview,
            ui=clips.compare(clips.rebuild(source, frames[..., :3], audio=None), preview, PREFIX),
        )


def _marked(frame, box):
    """A frame with everything outside the window dimmed and the window outlined."""
    import torch

    height, width = int(frame.shape[0]), int(frame.shape[1])
    left, top, right, bottom = box
    x0, y0 = max(0, int(round(left))), max(0, int(round(top)))
    x1, y1 = min(width, int(round(right))), min(height, int(round(bottom)))
    out = frame.float() * OUTSIDE_SHADE
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = frame[y0:y1, x0:x1].float()
        line = max(1, round(min(height, width) / 360))
        colour = torch.tensor([1.0, 0.8, 0.2])
        out[y0:min(y1, y0 + line), x0:x1] = colour
        out[max(y0, y1 - line):y1, x0:x1] = colour
        out[y0:y1, x0:min(x1, x0 + line)] = colour
        out[y0:y1, max(x0, x1 - line):x1] = colour
    return out
