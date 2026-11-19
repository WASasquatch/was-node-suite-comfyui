"""Moving a clip from one look to another along keyframes."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LIST
from ...modules.media import keyframes as keyframing
from ...modules.media import mixing

logger = log.get_logger("nodes.video_look_transition")

NODE_NAME = "Video Look Transition"

#: Keyframes a fresh node carries: from the first look at frame 0 to the second at the last frame.
DEFAULT_KEYFRAMES = "0:0:ease_in_out_sine; -1:1"

#: Most pictures the strip on the node shows.
STRIP_FRAMES = 16

#: Height of one picture on that strip, in pixels.
STRIP_HEIGHT = 96


def _strip(images):
    """Evenly spaced frames of a clip side by side, one row, for the panel on the node."""
    import torch
    import torch.nn.functional as F

    count, height, width = (int(v) for v in images.shape[:3])
    shown = min(STRIP_FRAMES, count)
    thumb_w = max(1, int(round(STRIP_HEIGHT * width / max(height, 1))))
    tiles = []
    for slot in range(shown):
        frame = min(count - 1, int((slot + 0.5) * count / shown))
        picture = images[frame, ..., :3].float().permute(2, 0, 1).unsqueeze(0)
        picture = F.interpolate(picture, size=(STRIP_HEIGHT, thumb_w), mode="bilinear",
                                align_corners=False, antialias=True)
        tiles.append(picture[0].permute(1, 2, 0).clamp(0.0, 1.0))
    return torch.cat(tiles, dim=1).unsqueeze(0)


def _mask_for(mask, count: int, height: int, width: int):
    """A mask batch as one plane per frame at the frames' size, or None.

    Raises:
        ValueError: The mask holds neither one plane nor one per frame.
    """
    import torch
    import torch.nn.functional as F

    if mask is None:
        return None
    planes = mask if mask.ndim == 3 else mask.reshape((-1,) + tuple(mask.shape[-2:]))
    held = int(planes.shape[0])
    if held not in (1, count):
        raise ValueError(
            f"{NODE_NAME}'s mask holds {held} planes and the clip {count} frames. Wire one "
            f"plane for every frame, or one plane for the whole clip."
        )
    planes = planes.to(dtype=torch.float32, device="cpu")
    if tuple(planes.shape[-2:]) != (height, width):
        planes = F.interpolate(planes.unsqueeze(1), size=(height, width), mode="bilinear",
                               align_corners=False).squeeze(1)
    return planes.clamp(0.0, 1.0)


class VideoLookTransition(io.ComfyNode):
    """Move a clip from one look to another along keyframes."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASVideoLookTransition",
            display_name=NODE_NAME,
            search_aliases=[
                "WASVideoLookTransition", NODE_NAME,
                "grade transition",
                "day to night",
                "fade in grade",
                "keyframe mix",
                "look blend",
            ],
            category="WAS Suite/Animation",
            is_output_node=True,
            description=(
                "Move a clip from one look to another over time, such as a day grade into a "
                "night grade, along keyframes set on a curve drawn over the clip's frames. A "
                "mask limits the change to part of the frame, such as a tracked subject's matte."
            ),
            inputs=[
                io.Image.Input(
                    "image_a",
                    tooltip="The clip graded the first way, shown where the amount is 0.",
                ),
                io.Image.Input(
                    "image_b",
                    tooltip=(
                        "The same clip graded the second way, shown where the amount is 1. "
                        "Same frame count, size and channels as image_a."
                    ),
                ),
                io.String.Input(
                    "keyframes",
                    default=DEFAULT_KEYFRAMES,
                    tooltip=(
                        "frame:amount:easing entries split by ';'. 0:0; -1:1 = the whole clip "
                        "from image_a to image_b; -1 = the last frame. An easing shapes the "
                        "move to the next keyframe; hold = jumps there. Drag the curve on the "
                        "node."
                    ),
                ),
                io.Combo.Input(
                    "mix",
                    options=list(mixing.SPACES),
                    default=mixing.CODES,
                    tooltip=(
                        "`codes` = mixes the values as they arrive, an editor's default and "
                        "right for linear-light frames; `light` = decodes sRGB codes to light, "
                        "mixes, encodes again, brighter through the middle."
                    ),
                ),
                io.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip=(
                        "Where the change happens: 1 = the full amount, 0 = image_a kept. One "
                        "plane per frame, or one for the whole clip, such as Image Track "
                        "Subject's matte. Resized to the frames."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip=(
                        "The clip moving between the two looks: image_a's frames exactly where "
                        "the amount is 0, image_b's where it is 1."
                    ),
                ),
                LIST.Output(
                    display_name="amounts",
                    tooltip="The amount at every frame, 0 to 1, such as [0.0, 0.25, 0.75, 1.0].",
                ),
            ],
        )

    @classmethod
    def execute(cls, image_a, image_b, keyframes=DEFAULT_KEYFRAMES, mix=mixing.CODES,
                mask=None) -> io.NodeOutput:
        """Mix the two clips frame by frame by the keyframed amount.

        Raises:
            ValueError: The two clips differ in frame count, size or channels, a keyframe does
                not read, the mask's count matches neither one nor the frames, or ``mix`` names
                no space.
            MemoryError: Neither free memory nor a scratch drive can hold the result.
        """
        import torch

        from ...modules.image import scratch

        if tuple(image_a.shape) != tuple(image_b.shape):
            raise ValueError(
                f"{NODE_NAME} mixes two gradings of one clip, and image_a is "
                f"{_described(image_a)} while image_b is {_described(image_b)}. Grade the same "
                f"frames both ways, at the same size."
            )
        if mix not in mixing.SPACES:
            raise ValueError(f"{NODE_NAME}'s mix is '{mix}'; it offers {', '.join(mixing.SPACES)}.")
        count, height, width = (int(v) for v in image_a.shape[:3])
        keys = keyframing.read(keyframes, NODE_NAME)
        shares = keyframing.amounts(keys, count)
        planes = _mask_for(mask, count, height, width)

        out = scratch.allocate(tuple(image_a.shape), torch.float32, node=NODE_NAME,
                               advice="A shorter clip also fits it.")
        for frame, share in enumerate(shares):
            a = image_a[frame].to(device="cpu", dtype=torch.float32)
            b = image_b[frame].to(device="cpu", dtype=torch.float32)
            if planes is None:
                out[frame] = mixing.mix(a, b, share, mix)
            else:
                plane = planes[frame if int(planes.shape[0]) > 1 else 0]
                out[frame] = mixing.mix(a, b, plane * share, mix)

        _report(out, shares, keys, mix, planes is not None)
        logger.info("moved %d frame(s) between two looks along %d keyframe(s)", count, len(keys))
        return io.NodeOutput(out, shares)


def _described(images) -> str:
    """A batch's frame count, size and channels, written for a message."""
    count, height, width, channels = (int(v) for v in images.shape)
    return f"{count} frame(s) of {width}x{height} with {channels} channel(s)"


def _report(images, shares, keys, space: str, masked: bool) -> None:
    """Publish the strip and the figures the panel on the node draws.

    Args:
        images: The mixed clip.
        shares: The amount at every frame.
        keys: The keyframes read.
        space: The space the mix ran in.
        masked: Whether a mask limited it.
    """
    from ...modules.interface import preview as published
    from ...modules.interface import run_result

    published.publish_output(_strip(images))
    first_change = next((frame for frame, share in enumerate(shares) if share != shares[0]), None)
    run_result.publish(
        summary=(
            f"{len(shares)} frames from amount {shares[0]:.2f} to {shares[-1]:.2f}"
            if first_change is not None else f"{len(shares)} frames held at amount {shares[0]:.2f}"
        ),
        counts={
            "frames": len(shares),
            "keyframes": len(keys),
            "at image_a": sum(1 for share in shares if share <= 0.0),
            "at image_b": sum(1 for share in shares if share >= 1.0),
        },
        facts={
            "mix": f"in {space}",
            "mask": "limits the change" if masked else "none, the whole frame changes",
            "moves from": f"frame {first_change}" if first_change is not None else "never",
        },
    )
