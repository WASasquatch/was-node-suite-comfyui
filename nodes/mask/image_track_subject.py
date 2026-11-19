"""Pick a subject on one frame and follow it through a batch with SAM 3, as a soft matte."""

from __future__ import annotations

import json

from comfy_api.latest import io

from ...modules import log
from ...modules.compat import limits
from ...modules.image import develop, track as tracking
from ...modules.interface import preview, run_result

logger = log.get_logger("nodes.mask")

NODE = "Image Track Subject"

#: Frames the panel is sent, evenly spaced, and the long edge they are sent at.
SHOWN_FRAMES = 48
SHOWN_EDGE = 384

#: Text finds listed on the panel.
MAX_CANDIDATES = 12


class ImageTrackSubject(io.ComfyNode):
    """Follow one subject through a video or batch from a frame it was picked on."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageTrackSubject",
            display_name="Image Track Subject",
            search_aliases=[
                "WASImageTrackSubject", "Image Track Subject", "track", "rotoscope", "subject matte",
                "video mask", "sam3 track", "follow", "isolate subject",
            ],
            category="WAS Suite/Image/Masking",
            description=(
                "Pick a subject on any frame, by clicking it on the node or naming it, and follow it "
                "forward and backward through a video or batch as a soft matte. Feed the matte to Image "
                "Spotlight, or to a grade's mask, to change only the subject."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="The frames to follow the subject through: a video's frames or a single picture.",
                ),
                io.Model.Input(
                    "model",
                    tooltip="SAM 3, from Load Checkpoint with sam3.1_multiplex_fp16.safetensors.",
                ),
                io.String.Input(
                    "subject", default="person",
                    tooltip="What to look for when nothing is clicked: `person`, `the robot`, `red car`. Needs `clip`. Every find is outlined on the panel.",
                ),
                io.Int.Input(
                    "key_frame", default=0, min=0, max=limits.max_resolution(),
                    tooltip="The frame the subject is picked on, counting from 0. Pick a frame where it is clear and whole; 0 = the first.",
                ),
                io.Int.Input(
                    "pick", default=0, min=0, max=254,
                    tooltip="Which find of `subject` to follow. 0 = the largest, 1 = the next. Click an outline on the panel to set it.",
                ),
                io.String.Input(
                    "points", default="", advanced=True,
                    tooltip='Points clicked on the key frame, such as [{"x": 0.5, "y": 0.4, "keep": true}]. Set from the panel; shift-click excludes. Empty = use `subject`.',
                ),
                io.Combo.Input(
                    "track",
                    options=list(tracking.TRACK_MODES),
                    tooltip="`both ways` = from the key frame to both ends; `forward` = on to the last frame; `backward` = back to the first. Frames not reached get no matte.",
                ),
                io.Float.Input(
                    "grow", default=0.0, min=-5.0, max=5.0, step=0.05,
                    tooltip="Grows the matte, in hundredths of the short side. 0.3 = a little past the edge, to keep hair; -0.3 = pulled in.",
                ),
                io.Float.Input(
                    "feather", default=0.3, min=0.0, max=5.0, step=0.05,
                    tooltip="Softens the matte's edge, in hundredths of the short side. 0 = crisp; 0.3 = natural; 2 = a soft glow.",
                ),
                io.Clip.Input(
                    "clip", optional=True,
                    tooltip="SAM 3's text encoder, the CLIP from the same Load Checkpoint. Needed for `subject` and the outlines on the panel.",
                ),
                io.Mask.Input(
                    "mask", optional=True,
                    tooltip="A mask of the subject on the key frame, from a brush or another node. Used in place of points and `subject`.",
                ),
            ],
            outputs=[
                io.Mask.Output(
                    display_name="mask",
                    tooltip="The subject's matte on every frame: 1 = the subject, 0 = everything else.",
                ),
            ],
        )

    @classmethod
    def execute(cls, images, model, subject="person", key_frame=0, pick=0, points="", track="both ways",
                grow=0.0, feather=0.3, clip=None, mask=None) -> io.NodeOutput:
        if images is None or images.ndim != 4 or int(images.shape[0]) == 0:
            raise ValueError(f"{NODE} needs at least one frame on `images`.")
        count = int(images.shape[0])
        key = min(max(int(key_frame), 0), count - 1)
        clicked = tracking.parse_points(points, NODE)
        given = develop.matte_for(images, mask)[key] if mask is not None else None
        chosen = tracking.pick_subject(model, clip, images[key], subject, clicked, pick, given, NODE)
        mattes = tracking.track(model, images, key, chosen.mask, track, NODE)
        mattes = tracking.finish(mattes, images, grow, feather)
        tracked = int((mattes.flatten(1).amax(dim=1) > 0.5).sum())
        shown = cls.shown_frames(count, key)
        preview.publish_frames(cls.thumbnails(images[shown]), slot="images")
        preview.publish_mask_output_frames(cls.thumbnails(mattes[shown][..., None])[..., 0], slot="mask")
        candidates = [
            {"index": found["index"], "box": [round(v, 3) for v in found["box"]], "score": round(found["score"], 2)}
            for found in chosen.candidates[:MAX_CANDIDATES]
        ]
        run_result.publish(
            status=run_result.OK if tracked else run_result.WARNING,
            summary=f"followed the subject on {tracked} of {count} frame(s) from frame {key}",
            counts={"frames": count, "key_frame": key, "tracked": tracked},
            facts={
                "source": chosen.source,
                "key_frame": key,
                "picked": chosen.picked,
                "candidates": json.dumps(candidates, separators=(",", ":")),
                "points": json.dumps([{"x": round(x, 4), "y": round(y, 4), "keep": keep} for x, y, keep in clicked]),
                "shown": json.dumps(shown, separators=(",", ":")),
            },
        )
        logger.info("%s followed the subject on %d of %d frame(s)", NODE, tracked, count)
        return io.NodeOutput(mattes)

    @staticmethod
    def shown_frames(count: int, key: int) -> list[int]:
        """Evenly spaced frame indices for the panel, the key frame always among them."""
        if count <= SHOWN_FRAMES:
            return list(range(count))
        spaced = {round(i * (count - 1) / (SHOWN_FRAMES - 2)) for i in range(SHOWN_FRAMES - 1)}
        return sorted(spaced | {key})

    @staticmethod
    def thumbnails(pictures):
        """Pictures brought down to :data:`SHOWN_EDGE` on their long side."""
        import torch.nn.functional as F

        height, width = int(pictures.shape[1]), int(pictures.shape[2])
        scale = SHOWN_EDGE / max(height, width)
        if scale >= 1.0:
            return pictures
        size = (max(1, round(height * scale)), max(1, round(width * scale)))
        return F.interpolate(pictures.movedim(-1, 1).float(), size=size, mode="area").movedim(1, -1)

