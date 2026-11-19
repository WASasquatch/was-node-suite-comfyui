"""A mask of the colours or the band of lightness clicked on the picture, made per frame."""

from __future__ import annotations

import json

import torch
from comfy_api.latest import io

from ...modules.image.accelerate import run_frames
from ...modules.image.develop import frame_by_frame
from ...modules.interface import preview, run_result
from ...modules.mask import local

NODE = "Image Color Range Mask"


class ImageColorRangeMask(io.ComfyNode):
    """Colour range or luminance range mask, measured on every frame."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageColorRangeMask",
            display_name="Image Color Range Mask",
            search_aliases=[
                "WASImageColorRangeMask", "Image Color Range Mask", "color range mask", "colour range",
                "luminance range mask", "select color", "eyedropper mask", "sky mask", "local adjustment",
            ],
            category="WAS Suite/Image/Masking",
            description=(
                "Selects colours, or a band of lightness, clicked on the picture on the node, like a photo "
                "editor's Color Range and Luminance Range masks. Every frame is measured on its own, so the "
                "selection follows the colour through a video. Feed the mask to Image Develop or another "
                "adjustment's mask."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="The pictures to select from. Each frame is measured on its own.",
                ),
                io.Combo.Input(
                    "mode",
                    options=list(local.RANGE_MODES),
                    tooltip="`color` = the colours clicked on the picture; `luminance` = a band of lightness, set on the bar under the picture.",
                ),
                io.Float.Input(
                    "range", default=0.15, min=0.005, max=1.0, step=0.005,
                    tooltip="Color mode: how far from a sampled colour the selection reaches, as Oklab distance. 0.05 = that colour only; 0.15 = its shades; 0.4 = neighbouring hues too.",
                ),
                io.Float.Input(
                    "smoothness", default=0.5, min=0.0, max=1.0, step=0.01,
                    tooltip="How gradually the selection fades. 0 = a hard edge; 0.5 = over the outer half of `range`, or 0.125 of lightness past the band; 1 = widest.",
                ),
                io.Float.Input(
                    "lightness_weight", default=0.5, min=0.0, max=1.0, step=0.01,
                    tooltip="Color mode: how much a difference in lightness counts. 1 = plain Oklab distance; 0.5 = lighter and darker shades still match; 0 = hue and saturation only.",
                ),
                io.Float.Input(
                    "refine", default=0.3, min=0.0, max=5.0, step=0.05,
                    tooltip="Snaps the selection's edge to the picture's own edges, over this many hundredths of the short side. 0 = off; 0.3 = tidy; 1 = also smooths speckle in noise.",
                ),
                io.Boolean.Input(
                    "invert", default=False,
                    tooltip="`true` = everything but the selected colours or lightness; `false` = the selection itself.",
                ),
                io.Combo.Input(
                    "combine",
                    options=list(local.COMBINE_MODES),
                    tooltip="How this selection joins `mask`. `replace` = the selection alone; `add` = the larger of the two; `subtract` = mask x (1 - selection); `intersect` = mask x selection.",
                ),
                io.String.Input(
                    "samples", default="", advanced=True,
                    tooltip="Colours to select, set by clicking the picture: x,y or an area x,y,w,h in shares of the picture, then @frame, or #rrggbb; split by ;. Example: 0.42,0.18@0; #6fa0e8",
                ),
                io.Float.Input(
                    "luminance_low", default=0.5, min=0.0, max=1.0, step=0.001, advanced=True,
                    tooltip="Luminance mode: the darkest Oklab lightness fully selected. 0 = black; 0.6 = middle grey; 1 = white. Drag the band under the picture.",
                ),
                io.Float.Input(
                    "luminance_high", default=1.0, min=0.0, max=1.0, step=0.001, advanced=True,
                    tooltip="Luminance mode: the lightest Oklab lightness fully selected. 1 = up to white; 0.3 = shadows only with luminance_low at 0.",
                ),
                io.Mask.Input(
                    "mask", optional=True,
                    tooltip="A mask to join this selection to, as `combine` says, such as a Graduated or Radial mask.",
                ),
            ],
            outputs=[
                io.Mask.Output(
                    display_name="mask",
                    tooltip="One mask per frame at the pictures' size: 1 = selected, 0 = not.",
                ),
            ],
        )

    @classmethod
    def execute(cls, images, mode="color", range=0.15, smoothness=0.5, lightness_weight=0.5, refine=0.3,
                invert=False, combine="replace", samples="", luminance_low=0.5, luminance_high=1.0,
                mask=None) -> io.NodeOutput:
        if images is None or images.ndim != 4 or int(images.shape[0]) == 0:
            raise ValueError(f"{NODE} needs at least one picture on `images`.")
        if mode not in local.RANGE_MODES:
            raise ValueError(f"{NODE}: mode must be one of {', '.join(local.RANGE_MODES)}, not {mode!r}.")
        if combine not in local.COMBINE_MODES:
            raise ValueError(f"{NODE}: combine must be one of {', '.join(local.COMBINE_MODES)}, not {combine!r}.")
        count, height, width = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
        picked = local.parse_samples(samples, NODE) if mode == "color" else []
        colours, described = local.sample_colors(images, picked)
        reach, soft, weight, amount = float(range), float(smoothness), float(lightness_weight), float(refine)
        low, high = float(luminance_low), float(luminance_high)

        def work(chunk: torch.Tensor) -> torch.Tensor:
            if mode == "color":
                selected = local.color_range(chunk, colours, reach, soft, weight)
            else:
                selected = local.luminance_range(chunk, low, high, soft)
            return local.refine(selected, chunk, amount)

        if mode == "color" and not picked:
            made = torch.zeros((count, height, width), dtype=torch.float32)
        else:
            made = run_frames(images, frame_by_frame(work), factor=48.0, node=NODE,
                              advice="Select from fewer frames at a time.")
        result, given = local.finish(images, made, mask, combine, bool(invert))
        settings = {
            "mode": mode, "range": reach, "smoothness": soft, "lightness_weight": weight, "refine": amount,
            "invert": bool(invert), "combine": combine,
            "samples": [
                f"#{''.join(f'{round(v * 255):02x}' for v in one.code)}" if one.code is not None
                else [round(one.x, 4), round(one.y, 4), round(one.width, 4), round(one.height, 4), one.frame]
                for one in picked
            ],
            "luminance_low": low, "luminance_high": high,
        }
        cls.report(images, result, given, combine, settings, described, mode == "color" and not picked)
        return io.NodeOutput(result)

    @staticmethod
    def report(images, result, given, combine, settings, described, unsampled) -> None:
        """Publish the frames, the masks, the sampled colours and the figures the panel draws."""
        shown = local.shown_frames(int(images.shape[0]))
        preview.publish_frames(local.thumbnails(images, shown), slot="images")
        if given is not None:
            preview.publish_mask_frames(local.thumbnails(given, shown), slot="mask")
        preview.publish_mask_output_frames(local.thumbnails(result, shown), slot="mask")
        warn, summary, joined = local.summary(result, given, combine)
        if unsampled:
            warn, summary = True, "no colour is sampled yet: click the picture on the node, then run again"
        figures = local.figures(result)
        run_result.publish(
            status=run_result.WARNING if warn else run_result.OK,
            summary=summary,
            counts={
                "frames": int(images.shape[0]), "width": int(images.shape[2]), "height": int(images.shape[1]),
                "coverage": figures["coverage"], "selected": figures["selected"], "samples": len(described),
            },
            facts={
                "settings": json.dumps(settings, separators=(",", ":")),
                "shown": json.dumps(shown, separators=(",", ":")),
                "combine": joined,
                "mask_wired": "yes" if given is not None else "no",
                "colors": json.dumps([[one["hex"], one["frame"]] for one in described], separators=(",", ":")),
            },
        )
