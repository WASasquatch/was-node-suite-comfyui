"""Background and foreground removal driven by a MiDaS depth map."""

from __future__ import annotations

import numpy as np
from comfy_api.latest import io

from ....modules import log
from ....modules.compat.types import MIDAS_MODEL
from ....modules.convert.tensors import image_planes, stack_images, tensor2pil

REQUIRES = "superseded"

logger = log.get_logger("nodes.legacy.ai.midas_mask")


def adjust_levels(image, min_level, mid_level, max_level):
    """Stretch an image's tonal range between two levels.

    Args:
        image: Source PIL image.
        min_level: Intensity that becomes black, 0-255.
        mid_level: Intensity that becomes mid grey, 0-255. At or below ``min_level`` the
            gamma curve is skipped.
        max_level: Intensity that becomes white, 0-255, and the autocontrast cutoff.

    Returns:
        A PIL image in the source's mode.

    Raises:
        ZeroDivisionError: ``max_level`` equals ``min_level``, leaving no range to scale.
    """
    from PIL import Image, ImageOps

    im_arr = np.array(image)
    im_arr[im_arr < min_level] = min_level
    im_arr = (im_arr - min_level) * (255 / (max_level - min_level))
    im_arr[im_arr < 0] = 0
    im_arr[im_arr > 255] = 255
    if mid_level > min_level:
        import math

        gamma = math.log(0.5) / math.log((mid_level - min_level) / (max_level - min_level))
        im_arr = np.power(im_arr / 255, gamma) * 255
    im_arr = im_arr.astype(np.uint8)

    # A cutoff of 100 or more removes the whole histogram from both ends, which
    # autocontrast reports as nothing to stretch and answers with an identity mapping.
    return ImageOps.autocontrast(Image.fromarray(im_arr), cutoff=max_level)


class MidasMaskImage(io.ComfyNode):
    """Replace the far or the near half of an image with a flat colour."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiDaS Mask Image",
            display_name="MiDaS Mask Image",
            search_aliases=[
                "MiDaS Mask Image",
                "remove background depth",
                "midas",
                "depth matte",
            ],
            category="WAS Suite/Image/AI",
            is_deprecated=True,
            description=(
                "Deprecated: use Image Remove Background, CLIPSeg Masking or SAM Image Mask "
                "for cutouts. Keeps the near or far part of an image by MiDaS depth, on "
                "transparency or a flat colour, with a matching mask, for fading a "
                "background by distance or matting in fog. Needs features.midas enabled."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The image to split. A batch is depth-mapped one image at a time and "
                        "comes back as a batch of the same length."
                    ),
                ),
                MIDAS_MODEL.Input(
                    "midas_model",
                    tooltip=(
                        "The depth model, from MiDaS Model Loader, which is where the "
                        "DPT_Large, DPT_Hybrid or DPT_Small choice is made. One loader can "
                        "feed several nodes so the weights are built once."
                    ),
                ),
                io.Boolean.Input(
                    "use_cpu",
                    default=False,
                    tooltip=(
                        "`off` = the graphics card, which is much faster and costs VRAM; `on` = "
                        "the processor instead."
                    ),
                ),
                io.Combo.Input(
                    "remove",
                    options=["background", "foregroud"],
                    tooltip=(
                        "Which half to replace with the background colour: `background` keeps "
                        "what is near the camera, `foregroud` keeps what is far away. The "
                        "second option is spelled as it was in the workflows that store it."
                    ),
                ),
                io.Boolean.Input(
                    "threshold",
                    default=False,
                    tooltip=(
                        "`on` pushes the depth map towards black and white using the three "
                        "threshold values below, which gives a harder edge; `off` composites "
                        "with the smooth depth map and leaves a gradual fade."
                    ),
                ),
                io.Float.Input(
                    "threshold_low",
                    default=10,
                    min=0,
                    max=255,
                    step=1,
                    tooltip=(
                        "Depth value that becomes fully background, 0-255. Raise it to pull "
                        "more of the middle distance into the background. Only used when "
                        "threshold is on."
                    ),
                ),
                io.Float.Input(
                    "threshold_mid",
                    default=200,
                    min=0,
                    max=255,
                    step=1,
                    tooltip=(
                        "Depth value that becomes mid grey, 0-255. Below threshold_low it "
                        "is skipped; between the two it bends the falloff, so 150 keeps more "
                        "of the near half and 230 keeps less. Only used when threshold is on."
                    ),
                ),
                io.Float.Input(
                    "threshold_high",
                    default=210,
                    min=0,
                    max=255,
                    step=1,
                    tooltip=(
                        "Depth value that becomes fully foreground, 0-255. Lower it to keep "
                        "more of the middle distance. Only used when threshold is on, and "
                        "it must not equal threshold_low."
                    ),
                ),
                io.Float.Input(
                    "smoothing",
                    default=0.25,
                    min=0.0,
                    max=16.0,
                    step=0.01,
                    tooltip=(
                        "Blur applied to the depth map before compositing, in pixels. Softens "
                        "the edge between the two halves; 0 turns it off and leaves the edge "
                        "as the depth map drew it."
                    ),
                ),
                io.Int.Input(
                    "background_red",
                    default=0,
                    min=0,
                    max=255,
                    step=1,
                    tooltip="Red channel of the replacement colour, 0-255.",
                ),
                io.Int.Input(
                    "background_green",
                    default=0,
                    min=0,
                    max=255,
                    step=1,
                    tooltip=(
                        "Green channel of the replacement colour, 0-255. 255 with the other "
                        "two at 0 gives a green screen."
                    ),
                ),
                io.Int.Input(
                    "background_blue",
                    default=0,
                    min=0,
                    max=255,
                    step=1,
                    tooltip="Blue channel of the replacement colour, 0-255.",
                ),
                io.Boolean.Input(
                    "transparency",
                    default=True,
                    tooltip=(
                        "`on` = the removed half is transparent and RESULT carries four "
                        "channels, ready to composite over anything; `off` = RESULT is three "
                        "channels and the removed half is filled with the background colour "
                        "above."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="RESULT",
                    tooltip=(
                        "The kept half of the image. Four channels with the removed half "
                        "transparent when transparency is on, three with it filled by the "
                        "background colour when off."
                    ),
                ),
                io.Image.Output(
                    display_name="DEPTH",
                    tooltip=(
                        "The depth map used to make the split, after levelling and blurring, as "
                        "a greyscale image. White is the half that was kept."
                    ),
                ),
                io.Mask.Output(
                    display_name="MASK",
                    tooltip=(
                        "The same split as a mask, white over the half that was kept. Wire it "
                        "into Image Paste Crop, a mask input or Mask Dilate Region without "
                        "converting DEPTH first."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        midas_model,
        use_cpu,
        remove,
        threshold,
        threshold_low,
        threshold_mid,
        threshold_high,
        smoothing,
        background_red,
        background_green,
        background_blue,
        transparency,
    ) -> io.NodeOutput:
        import torch
        from PIL import Image, ImageFilter

        # Loaded once for the whole batch.
        backend = midas_model.on("cpu" if use_cpu else None)
        device = backend.load()
        logger.info("MiDaS is using device: %s", device)

        background_color = (int(background_red), int(background_green), int(background_blue))
        results = []
        depths = []
        masks = []
        for plane in image_planes(image):
            source = 255.0 * plane.cpu().numpy().squeeze()
            # The channels are reversed before the model sees them, as they were when an
            # RGB array was run through OpenCV's BGR-to-RGB conversion. The depth map
            # depends on it.
            source = np.ascontiguousarray(source[:, :, ::-1])
            img_original = tensor2pil(plane).convert("RGB")

            logger.info("Approximating depth from image.")
            inputs = backend.processor(images=source, return_tensors="pt").to(device)
            with torch.no_grad():
                prediction = backend.model(**inputs).predicted_depth
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=source.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            # Stretched between the map's own nearest and farthest points, then optionally
            # levelled and blurred, before it splits the image.
            depth = prediction.cpu().numpy().astype(np.float32)
            span = np.max(depth) - np.min(depth)
            depth = (depth - np.min(depth)) / span if span else np.zeros_like(depth)
            if remove == "foregroud":
                depth = 1.0 - depth
            depth = Image.fromarray(np.uint8(depth * 255))

            if threshold:
                depth = adjust_levels(
                    depth.convert("RGB"), threshold_low, threshold_mid, threshold_high
                ).convert("L")
            if smoothing > 0:
                depth = depth.filter(ImageFilter.GaussianBlur(radius=smoothing))
            depth = depth.resize(img_original.size).convert("L")

            if transparency:
                kept = img_original.convert("RGBA")
                kept.putalpha(depth)
                results.append(kept)
            else:
                background = Image.new(
                    mode="RGB", size=img_original.size, color=background_color
                )
                results.append(Image.composite(img_original, background, depth))
            depths.append(depth.convert("RGB"))
            masks.append(depth)

        return io.NodeOutput(stack_images(results), stack_images(depths), stack_images(masks))
