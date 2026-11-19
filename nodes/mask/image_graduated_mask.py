"""A linear gradient mask placed by dragging its lines on the picture."""

from __future__ import annotations

import json

from comfy_api.latest import io

from ...modules.interface import preview, run_result
from ...modules.mask import local

NODE = "Image Graduated Mask"

#: Furthest a line may sit outside the picture, as a share of its width or height.
REACH = (-1.0, 2.0)


class ImageGraduatedMask(io.ComfyNode):
    """Linear gradient mask: full on the start line, none past the end line."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageGraduatedMask",
            display_name="Image Graduated Mask",
            search_aliases=[
                "WASImageGraduatedMask", "Image Graduated Mask", "linear gradient mask", "graduated filter",
                "gradient mask", "sky mask", "nd grad", "local adjustment",
            ],
            category="WAS Suite/Image/Masking",
            description=(
                "A linear gradient mask, like a photo editor's graduated filter: full effect on the start "
                "line, fading to none at the end line. Drag the lines on the node to place and turn it, "
                "then feed the mask to Image Develop or another adjustment's mask."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="The pictures the mask is for. One mask is made per frame, at the pictures' size.",
                ),
                io.Combo.Input(
                    "profile",
                    options=list(local.PROFILES),
                    tooltip="How the fade runs between the lines. `smooth` = eases in and out, 0.5 on the middle line; `linear` = an even ramp.",
                ),
                io.Boolean.Input(
                    "invert", default=False,
                    tooltip="Turns the mask over: 0 before the start line, 1 past the end line.",
                ),
                io.Combo.Input(
                    "combine",
                    options=list(local.COMBINE_MODES),
                    tooltip="How this gradient joins `mask`. `replace` = the gradient alone; `add` = the larger of the two; `subtract` = mask x (1 - gradient); `intersect` = mask x gradient.",
                ),
                io.Float.Input(
                    "start_x", default=0.5, min=REACH[0], max=REACH[1], step=0.001, advanced=True,
                    tooltip="Where the start line (full effect) crosses, as a share of the width: 0 = left edge, 1 = right edge. Drag it on the node.",
                ),
                io.Float.Input(
                    "start_y", default=0.2, min=REACH[0], max=REACH[1], step=0.001, advanced=True,
                    tooltip="Where the start line crosses, as a share of the height: 0 = top, 1 = bottom. Below 0 sits above the picture.",
                ),
                io.Float.Input(
                    "end_x", default=0.5, min=REACH[0], max=REACH[1], step=0.001, advanced=True,
                    tooltip="Where the end line (no effect) crosses, as a share of the width: 0.5 = the middle. Both lines stand square to the way from start to end.",
                ),
                io.Float.Input(
                    "end_y", default=0.6, min=REACH[0], max=REACH[1], step=0.001, advanced=True,
                    tooltip="Where the end line crosses, as a share of the height. 0.2 to 0.6 = full at the top fifth, gone by 60% down.",
                ),
                io.Mask.Input(
                    "mask", optional=True,
                    tooltip="A mask to join this gradient to, as `combine` says, such as another Graduated, Radial or Color Range mask.",
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
    def execute(cls, images, profile="smooth", invert=False, combine="replace", start_x=0.5, start_y=0.2,
                end_x=0.5, end_y=0.6, mask=None) -> io.NodeOutput:
        if images is None or images.ndim != 4 or int(images.shape[0]) == 0:
            raise ValueError(f"{NODE} needs at least one picture on `images`.")
        if profile not in local.PROFILES:
            raise ValueError(f"{NODE}: profile must be one of {', '.join(local.PROFILES)}, not {profile!r}.")
        if combine not in local.COMBINE_MODES:
            raise ValueError(f"{NODE}: combine must be one of {', '.join(local.COMBINE_MODES)}, not {combine!r}.")
        height, width = int(images.shape[1]), int(images.shape[2])
        made = local.linear_gradient(height, width, (start_x, start_y), (end_x, end_y), profile, NODE)
        result, given = local.finish(images, made, mask, combine, bool(invert))
        settings = {
            "profile": profile, "invert": bool(invert), "combine": combine, "start_x": float(start_x),
            "start_y": float(start_y), "end_x": float(end_x), "end_y": float(end_y),
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
