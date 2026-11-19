"""An elliptical gradient mask placed by dragging its centre, edges and turn on the picture."""

from __future__ import annotations

import json

from comfy_api.latest import io

from ...modules.interface import preview, run_result
from ...modules.mask import local

NODE = "Image Radial Mask"

#: Furthest the centre may sit outside the picture, as a share of its width or height.
REACH = (-1.0, 2.0)


class ImageRadialMask(io.ComfyNode):
    """Elliptical gradient mask: full inside, fading to none at the edge."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageRadialMask",
            display_name="Image Radial Mask",
            search_aliases=[
                "WASImageRadialMask", "Image Radial Mask", "radial gradient mask", "radial filter",
                "ellipse mask", "vignette mask", "spot mask", "local adjustment",
            ],
            category="WAS Suite/Image/Masking",
            description=(
                "An elliptical mask, like a photo editor's radial gradient: full effect inside, fading to "
                "none at the ellipse's edge. Drag its centre, edges, turn and feather on the node, then feed "
                "the mask to Image Develop or another adjustment's mask."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="The pictures the mask is for. One mask is made per frame, at the pictures' size.",
                ),
                io.Float.Input(
                    "feather", default=50.0, min=0.0, max=100.0, step=1.0,
                    tooltip="How much of the radius the fade spans, inward from the edge. 0 = hard edge; 50 = full inside half the radius; 100 = fades from the centre.",
                ),
                io.Combo.Input(
                    "profile",
                    options=list(local.PROFILES),
                    tooltip="How the fade runs across the feather. `smooth` = eases in and out; `linear` = an even ramp.",
                ),
                io.Boolean.Input(
                    "invert", default=False,
                    tooltip="`true` = full effect outside the ellipse and none inside, as for a vignette; `false` = full inside.",
                ),
                io.Combo.Input(
                    "combine",
                    options=list(local.COMBINE_MODES),
                    tooltip="How this ellipse joins `mask`. `replace` = the ellipse alone; `add` = the larger of the two; `subtract` = mask x (1 - ellipse); `intersect` = mask x ellipse.",
                ),
                io.Float.Input(
                    "centre_x", default=0.5, min=REACH[0], max=REACH[1], step=0.001, advanced=True,
                    tooltip="The ellipse's centre across, as a share of the width: 0 = left edge, 1 = right edge. Drag the centre on the node.",
                ),
                io.Float.Input(
                    "centre_y", default=0.5, min=REACH[0], max=REACH[1], step=0.001, advanced=True,
                    tooltip="The ellipse's centre down, as a share of the height: 0 = top, 1 = bottom.",
                ),
                io.Float.Input(
                    "radius_x", default=0.3, min=0.001, max=4.0, step=0.001, advanced=True,
                    tooltip="Half the ellipse's width before turning, as a share of the picture's width. 0.5 = reaches both side edges.",
                ),
                io.Float.Input(
                    "radius_y", default=0.3, min=0.001, max=4.0, step=0.001, advanced=True,
                    tooltip="Half the ellipse's height before turning, as a share of the picture's height. 0.5 = reaches the top and bottom.",
                ),
                io.Float.Input(
                    "angle", default=0.0, min=-180.0, max=180.0, step=0.1, advanced=True,
                    tooltip="Degrees the ellipse is turned clockwise. 0 = radius_x lies across; 90 = radius_x runs down.",
                ),
                io.Mask.Input(
                    "mask", optional=True,
                    tooltip="A mask to join this ellipse to, as `combine` says, such as another Radial, Graduated or Color Range mask.",
                ),
            ],
            outputs=[
                io.Mask.Output(
                    display_name="mask",
                    tooltip="One mask per frame at the pictures' size: 1 = full effect, 0 = none.",
                ),
            ],
        )

    @classmethod
    def execute(cls, images, feather=50.0, profile="smooth", invert=False, combine="replace", centre_x=0.5,
                centre_y=0.5, radius_x=0.3, radius_y=0.3, angle=0.0, mask=None) -> io.NodeOutput:
        if images is None or images.ndim != 4 or int(images.shape[0]) == 0:
            raise ValueError(f"{NODE} needs at least one picture on `images`.")
        if profile not in local.PROFILES:
            raise ValueError(f"{NODE}: profile must be one of {', '.join(local.PROFILES)}, not {profile!r}.")
        if combine not in local.COMBINE_MODES:
            raise ValueError(f"{NODE}: combine must be one of {', '.join(local.COMBINE_MODES)}, not {combine!r}.")
        height, width = int(images.shape[1]), int(images.shape[2])
        made = local.radial_gradient(height, width, (centre_x, centre_y), (radius_x, radius_y), angle, feather, profile)
        result, given = local.finish(images, made, mask, combine, bool(invert))
        settings = {
            "feather": float(feather), "profile": profile, "invert": bool(invert), "combine": combine,
            "centre_x": float(centre_x), "centre_y": float(centre_y), "radius_x": float(radius_x),
            "radius_y": float(radius_y), "angle": float(angle),
        }
        cls.report(images, result, given, combine, settings)
        return io.NodeOutput(result)

    @staticmethod
    def report(images, result, given, combine, settings) -> None:
        """Publish the frames, the masks and the figures the panel draws."""
        shown = local.shown_frames(int(images.shape[0]))
        preview.publish_frames(local.thumbnails(images, shown), slot="images")
        if given is not None:
            preview.publish_mask_frames(local.thumbnails(given, shown), slot="mask")
        preview.publish_mask_output_frames(local.thumbnails(result, shown), slot="mask")
        warn, summary, joined = local.summary(result, given, combine)
        figures = local.figures(result)
        run_result.publish(
            status=run_result.WARNING if warn else run_result.OK,
            summary=summary,
            counts={
                "frames": int(images.shape[0]), "width": int(images.shape[2]), "height": int(images.shape[1]),
                "coverage": figures["coverage"], "selected": figures["selected"],
            },
            facts={
                "settings": json.dumps(settings, separators=(",", ":")),
                "shown": json.dumps(shown, separators=(",", ":")),
                "combine": joined,
                "mask_wired": "yes" if given is not None else "no",
            },
        )
