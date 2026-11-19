"""Lay crops cut by Video Track Crop back into the clip they came from."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import VIDEO_TRACK

logger = log.get_logger("nodes.video_track_paste")

NODE_NAME = "Video Track Paste"

#: What both sides of the comparison on the node are written under in the temp folder.
PREFIX = "was.track_paste"

#: Magnification, as a multiple of keep_original, at which a frame takes the crop in full.
FULL_AT = 1.5

#: Blur the colour match reads light and colour through, as a share of the crop's short side.
LIGHT_SCALE = 0.125

#: Largest shift the alignment makes, as a share of the crop's short side.
ALIGN_REACH = 0.1

#: Gaussian sigma, in frames, the alignment shifts are smoothed over.
ALIGN_SMOOTHING = 1.0


class VideoTrackPaste(io.ComfyNode):
    """Lay restored crops back over the subject they were cut from, frame by frame."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASVideoTrackPaste",
            display_name=NODE_NAME,
            search_aliases=[
                "WASVideoTrackPaste", NODE_NAME,
                "face paste video",
                "uncrop",
                "face restore",
                "composite crop",
            ],
            category="WAS Suite/Animation",
            is_output_node=True,
            description=(
                "Lay crops cut by Video Track Crop back into the clip they came from, such as a "
                "restored face, each through its soft mask at the size and place it was cut. "
                "Frames where the original subject was already as sharp as the crop keep their own."
            ),
            inputs=[
                io.Video.Input(
                    "video",
                    tooltip="The clip the crops were cut from, or an upscaled copy of it, such as 2x, where the crops land with more pixels.",
                ),
                io.Image.Input(
                    "crops",
                    tooltip="The crops to lay back, one per frame from the first, at any size of the same shape: the restored or edited crops.",
                ),
                VIDEO_TRACK.Input("track", tooltip="Video Track Crop's track."),
                io.Mask.Input(
                    "masks",
                    optional=True,
                    tooltip="What of each crop is laid back: Video Track Crop's masks. Empty = an ellipse filling the crop.",
                ),
                io.Float.Input(
                    "grow",
                    default=12.0,
                    min=-128.0,
                    max=128.0,
                    step=1.0,
                    tooltip="Pixels of the crop the mask is grown by before its edge is softened: 12 = past the hairline; 0 = as cut; -8 = inside it.",
                ),
                io.Float.Input(
                    "feather",
                    default=12.0,
                    min=0.0,
                    max=128.0,
                    step=0.5,
                    tooltip="Softness of the edge in pixels of the crop: 12 = a soft join; 0 = hard.",
                ),
                io.Boolean.Input(
                    "color_match",
                    default=True,
                    tooltip="`true` = each crop takes the original's colour and light, blurred over an eighth of the crop, and keeps all its own features; `false` = laid back as it is.",
                ),
                io.Boolean.Input(
                    "align",
                    default=True,
                    tooltip=(
                        "`true` = each crop is shifted onto the original crop before it is laid back, "
                        "so a restored face that drifted lands on the face; `false` = laid back where it is."
                    ),
                ),
                io.Float.Input(
                    "keep_original",
                    default=1.0,
                    min=0.0,
                    max=8.0,
                    step=0.05,
                    tooltip=(
                        "Magnification at or below which a frame keeps its own subject, fading to the "
                        "crop in full by 1.5 times it: 1 = wherever the frame held as many pixels as "
                        "the crop; 0 = every frame takes the crop."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(display_name="images", tooltip="The clip's frames with the crops laid back."),
                io.Video.Output(display_name="video", tooltip="The same as a clip, with the source's frame rate and sound."),
            ],
        )

    @classmethod
    def execute(cls, video, crops, track, masks=None, grow=12.0, feather=12.0, color_match=True,
                keep_original=1.0, align=True) -> io.NodeOutput:
        """Lay every crop back and write both sides for the comparison on the node.

        Raises:
            ValueError: The track was made from a clip of another shape or length, or more
                crops arrive than the track has frames.
            MemoryError: Neither free memory nor a scratch drive can hold the result.
        """
        import torch

        import comfy.model_management

        from ...modules.image import scratch
        from ...modules.image.develop import smoothstep, soft_blur
        from ...modules.media import clip as clips
        from ...modules.media import track_crop

        source = clips.open_clip(video, NODE_NAME)
        frames = source.frames
        count, frame_h, frame_w = (int(v) for v in frames.shape[:3])
        aspect = track.frame[1] / track.frame[0]
        same_shape = abs(frame_w / frame_h - aspect) <= 0.01 * aspect
        if not same_shape or track.count != count:
            raise ValueError(
                f"The track was made from a {track.frame[1]}x{track.frame[0]} clip of {track.count} "
                f"frame(s), and this clip is {frame_w}x{frame_h} with {count}. Wire the same clip, "
                f"or an enlarged copy of it, into Video Track Crop and Video Track Paste."
            )
        if tuple(track.frame) != (frame_h, frame_w):
            track = track_crop.scaled(track, (frame_h, frame_w))
        laid = int(crops.shape[0])
        if laid > count:
            raise ValueError(
                f"{laid} crops arrived for a clip of {count} frame(s). Pass one crop per frame."
            )
        device = comfy.model_management.get_torch_device()
        crop_h, crop_w = track.size
        alpha = track_crop.soft_alpha(masks, laid, track.size, float(grow), float(feather), device)
        advice = "A shorter clip also fits it."
        out = scratch.allocate(tuple(frames[..., :3].shape), frames.dtype, NODE_NAME, advice)
        out.copy_(frames[..., :3])
        floor = float(keep_original)
        step = clips.progress(2 * laid)
        shares, shifts = [], []
        for index in range(laid):
            magnify = 1.0 / track.scale(index)
            share = 1.0 if floor <= 0.0 else float(smoothstep(floor, floor * FULL_AT, torch.tensor(magnify)))
            shares.append(share)
            dy = dx = 0.0
            if align and share > 0.0:
                crop = _fitted(crops[index, ..., :3], track.size, device)
                original, _inside = track_crop.crop_one(frames[index, ..., :3], track, index, device)
                dy, dx = track_crop.register(crop, original, alpha[index], ALIGN_REACH * min(crop_h, crop_w))
            shifts.append((dy, dx))
            step(1)
        if align:
            from ...modules.image import camera_motion

            rows = camera_motion.smooth([shift[0] for shift in shifts], ALIGN_SMOOTHING)
            cols = camera_motion.smooth([shift[1] for shift in shifts], ALIGN_SMOOTHING)
            shifts = list(zip(rows, cols))
        kept = 0
        for index in range(laid):
            share = shares[index]
            if share <= 0.0:
                kept += 1
                step(1)
                continue
            crop = _fitted(crops[index, ..., :3], track.size, device)
            if align:
                crop = track_crop.shifted(crop, *shifts[index])
            if color_match:
                original, _inside = track_crop.crop_one(frames[index, ..., :3], track, index, device)
                sigma = LIGHT_SCALE * min(crop_h, crop_w)
                made = crop.permute(2, 0, 1)[None]
                light = soft_blur(original.permute(2, 0, 1)[None], sigma) - soft_blur(made, sigma)
                crop = (made + light)[0].permute(1, 2, 0).clamp(0.0, 1.0)
            track_crop.paste_frames(out, crop[None], alpha[index:index + 1] * share, track, [index], device)
            step(1)

        if laid < count:
            logger.warning(
                "%s: %d crop(s) for a clip of %d frame(s); frames %d to %d keep their own subject",
                NODE_NAME, laid, count, laid, count - 1,
            )
        logger.info("track paste: %d of %d frame(s) laid back, %d kept their own", laid - kept, count, kept)
        answer = clips.rebuild(source, out)
        return io.NodeOutput(
            out, answer, ui=clips.compare(clips.rebuild(source, frames[..., :3], audio=None), answer, PREFIX)
        )


def _fitted(crop, size, device):
    """A ``(height, width, 3)`` crop on ``device`` at ``size``, resampled when it differs."""
    import torch
    import torch.nn.functional as F

    picture = crop.to(device=device, dtype=torch.float32)
    if tuple(picture.shape[:2]) == tuple(size):
        return picture
    planes = F.interpolate(picture.permute(2, 0, 1)[None], size=tuple(size), mode="bicubic", align_corners=False)
    return planes[0].permute(1, 2, 0).clamp(0.0, 1.0)
