"""Defocus pictures through a depth map, as a lens focused at one depth would."""

from __future__ import annotations

import time

from comfy_api.latest import io

from ....modules import log
from ....modules.image import dynamic, lens_blur

logger = log.get_logger("nodes.image.filter")

NODE = "Image Lens Blur"


class ImageLensBlur(io.ComfyNode):
    """Blur a picture by depth, sharp at the focus and softer with distance from it."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageLensBlur",
            display_name=NODE,
            search_aliases=[
                "WASImageLensBlur",
                "Image Lens Blur",
                "lens blur",
                "depth of field",
                "bokeh",
                "defocus",
                "portrait mode",
                "background blur",
            ],
            category="WAS Suite/Image/Filter",
            description=(
                "Defocus a picture the way a fast lens does, from a depth map: what lies at the "
                "focus stays sharp, nearer and farther things blur by their distance from it, and "
                "small lights open into discs. Click the picture on the node to choose what stays "
                "sharp."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The pictures to defocus, as picture codes; the blur is worked out in linear "
                        "light. A batch runs frame by frame, and video needs a depth map that holds "
                        "steady from frame to frame."
                    ),
                ),
                io.MultiType.Input(
                    "depth",
                    [io.Image, io.Mask],
                    tooltip=(
                        "Grey depth map of the same pictures, as an IMAGE or a MASK: Power "
                        "Preprocessor's depth_map, Depth Anything or Marigold. One map for all frames "
                        "or one per frame. A map past 0 to 1 is stretched to it over the batch."
                    ),
                ),
                io.Combo.Input(
                    "near_is",
                    options=list(lens_blur.NEAR_ENDS),
                    default=lens_blur.NEAR_ENDS[0],
                    tooltip=(
                        "Which end of the depth map is near the camera. `bright` = Depth Anything, "
                        "Marigold and Power Preprocessor's depth_map; `dark` = a Z pass or a "
                        "distance render."
                    ),
                ),
                io.Combo.Input(
                    "focus_from",
                    options=list(lens_blur.FOCUS_SOURCES),
                    default=lens_blur.FOCUS_SOURCES[0],
                    tooltip=(
                        "`picked point` = sharp at the depth under focus_x and focus_y, set by "
                        "clicking the picture on the node; `focus depth` = sharp at the focus_depth "
                        "value."
                    ),
                ),
                io.Float.Input(
                    "focus_depth",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Depth kept sharp when focus_from is `focus depth`: 1 = the nearest thing in "
                        "the map; 0 = the farthest; 0.5 = halfway."
                    ),
                ),
                io.Float.Input(
                    "blur_amount",
                    default=0.03,
                    min=0.0,
                    max=0.25,
                    step=0.001,
                    tooltip=(
                        "Widest blur, as a share of the short side, reached a full depth range from "
                        "the focus. 0 = off; 0.02 = 22 px on a 1080 frame, a portrait lens; 0.08 = "
                        "very shallow."
                    ),
                ),
                io.Float.Input(
                    "focus_range",
                    default=0.1,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Depth of the sharp zone, in depth map units across both sides of the focus. "
                        "0 = only the focus depth is sharp; 0.1 = a tenth of the depth range; 1 = "
                        "nothing blurs."
                    ),
                ),
                io.Combo.Input(
                    "aperture",
                    options=list(lens_blur.APERTURES),
                    default="circle",
                    tooltip=(
                        "Shape of the lens opening, which is the shape out-of-focus lights take: "
                        "`circle` = round discs; `6 blades` = hexagons; more blades = rounder "
                        "polygons."
                    ),
                ),
                io.Float.Input(
                    "blade_rotation",
                    default=0.0,
                    min=-180.0,
                    max=180.0,
                    step=1.0,
                    advanced=True,
                    tooltip=(
                        "Turn of the blade polygon in degrees. 0 = one flat edge at the bottom; 30 = "
                        "a corner at the bottom on `6 blades`. No effect on `circle`."
                    ),
                ),
                io.Float.Input(
                    "highlight_bloom",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Lift given to lights, near-white points standing above their surroundings, "
                        "before they blur, so they open into bright discs rather than grey smudges. "
                        "0 = off; 0.5 = 4x on a clipped light; 1 = 16x. Sharp pixels are never lifted."
                    ),
                ),
                io.Float.Input(
                    "focus_x",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    advanced=True,
                    tooltip=(
                        "Focus point across the picture: 0 = left edge; 1 = right edge. Set by "
                        "clicking the picture on the node."
                    ),
                ),
                io.Float.Input(
                    "focus_y",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    advanced=True,
                    tooltip=(
                        "Focus point down the picture: 0 = top edge; 1 = bottom edge. Set by "
                        "clicking the picture on the node."
                    ),
                ),
                io.Mask.Input(
                    "subject_mask",
                    optional=True,
                    tooltip=(
                        "Optional subject matte, white on the subject, from Image Matte or Image "
                        "Track Subject. Its edge replaces the depth map's soft one, so hair stays "
                        "sharp over a blurred background with no halo. One matte per frame or one "
                        "for all."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip=(
                        "The defocused pictures as picture codes, held to 0 to 1 unless the input "
                        "carried light above white."
                    ),
                ),
                io.Mask.Output(
                    display_name="focus_mask",
                    tooltip=(
                        "1 where the picture stayed sharp, falling to 0 at the widest blur. Wire it "
                        "into a sharpen or a grade to touch only what is in focus."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls, images, depth, near_is="bright", focus_from="picked point", focus_depth=0.5,
        blur_amount=0.03, focus_range=0.1, aperture="circle", blade_rotation=0.0,
        highlight_bloom=0.5, focus_x=0.5, focus_y=0.5, subject_mask=None,
    ) -> io.NodeOutput:
        import torch

        import comfy.model_management
        import comfy.utils

        from ....modules.image import accelerate, scratch, sizing
        from ....modules.interface import preview, run_result

        if images is None or images.ndim != 4 or int(images.shape[0]) == 0:
            raise ValueError(f"{NODE} needs at least one picture on `images`.")
        if near_is not in lens_blur.NEAR_ENDS:
            raise ValueError(f"{NODE}: near_is must be one of {', '.join(lens_blur.NEAR_ENDS)}.")
        if focus_from not in lens_blur.FOCUS_SOURCES:
            raise ValueError(f"{NODE}: focus_from must be one of {', '.join(lens_blur.FOCUS_SOURCES)}.")
        if aperture not in lens_blur.APERTURES:
            raise ValueError(f"{NODE}: aperture must be one of {', '.join(lens_blur.APERTURES)}.")

        count, height, width, channels = (int(side) for side in images.shape)
        largest = float(blur_amount) * min(height, width)
        margin = lens_blur.canvas_margin(lens_blur.diameter_grid(largest)[-1])
        sizing.check_canvas(
            width + 2 * margin, height + 2 * margin, f"{NODE}'s working canvas",
            "Lower blur_amount, or pass a smaller picture.",
        )
        maps = lens_blur.depth_planes(depth, count, height, width, NODE)
        mattes = lens_blur.matte_planes(subject_mask, count, height, width, NODE)

        advice = "Passing fewer or smaller pictures also fits it."
        result = scratch.allocate((count, height, width, channels), node=NODE, advice=advice)
        focus_masks = scratch.allocate((count, height, width), node=NODE, advice=advice)
        device = accelerate.working_device()
        progress = comfy.utils.ProgressBar(count)
        started = time.perf_counter()
        first = {}
        for index in range(count):
            plane = lens_blur.depth_plane(maps[index], near_is)
            matte = mattes[index] if mattes is not None else None
            if focus_from == lens_blur.FOCUS_SOURCES[0]:
                focus = lens_blur.focus_depth_at(plane, focus_x, focus_y, matte)
            else:
                focus = float(focus_depth)
            settings = lens_blur.Settings(
                focus=focus,
                largest=largest,
                focus_range=float(focus_range),
                blades=lens_blur.APERTURES[aperture],
                rotation=float(blade_rotation),
                bloom=float(highlight_bloom),
            )
            frame = images[index, :, :, :3]
            answer, sharp, facts = cls.render_on(device, frame, plane, matte, settings, comfy.model_management)
            result[index, :, :, :3] = answer
            if channels > 3:
                result[index, :, :, 3:] = images[index, :, :, 3:].cpu()
            focus_masks[index] = sharp
            if index == 0:
                first = dict(facts, focus=focus)
            progress.update(1)

        result = dynamic.hold(result, images)
        elapsed = (time.perf_counter() - started) * 1000.0
        preview.publish(images, slot="images")
        preview.publish_output(result, slot="images")
        preview.publish_mask(maps[:1], slot="depth")
        run_result.publish(
            status=run_result.OK,
            summary=(
                f"focused at depth {first.get('focus', 0.0):.3f} with blur up to "
                f"{first.get('widest', 0.0):.0f} px over {first.get('layers', 0)} layers"
            ),
            counts={
                "frames": count,
                "layers": int(first.get("layers", 0)),
                "widest": round(float(first.get("widest", 0.0)), 1),
                "milliseconds": round(elapsed),
            },
            facts={
                "focus": round(float(first.get("focus", 0.0)), 4),
                "focus_from": focus_from,
                "focus_x": round(float(focus_x), 4),
                "focus_y": round(float(focus_y), 4),
                "focus_depth": round(float(focus_depth), 4),
                "near_is": near_is,
                "blur_amount": round(float(blur_amount), 4),
                "focus_range": round(float(focus_range), 4),
                "aperture": aperture,
                "matte": "yes" if mattes is not None else "no",
                "size": f"{width}x{height}",
            },
        )
        logger.info("%s defocused %d frame(s) in %.0f ms", NODE, count, elapsed)
        return io.NodeOutput(result, focus_masks)

    @staticmethod
    def render_on(device, frame, plane, matte, settings, management):
        """Render one frame on a device, falling back to the CPU when the device runs out of memory.

        Args:
            device: The device to try first.
            frame: ``(height, width, 3)`` picture codes.
            plane: ``(height, width)`` depth with 1 nearest.
            matte: ``(height, width)`` subject matte, or None.
            settings: A :class:`~modules.image.lens_blur.Settings`.
            management: ComfyUI's model management module.

        Returns:
            ``(codes, focus, facts)`` on the CPU, as :func:`~modules.image.lens_blur.render` answers.
        """
        import torch

        def run(on):
            with torch.inference_mode():
                answer, sharp, facts = lens_blur.render(
                    frame.to(device=on, dtype=torch.float32),
                    plane.to(device=on, dtype=torch.float32),
                    settings,
                    matte.to(device=on, dtype=torch.float32) if matte is not None else None,
                    management.throw_exception_if_processing_interrupted,
                )
                return answer.cpu(), sharp.cpu(), facts

        try:
            return run(device)
        except torch.cuda.OutOfMemoryError:
            if device.type != "cuda":
                raise
            logger.warning("%s ran out of graphics memory on one frame, so it ran on the CPU", NODE)
            torch.cuda.empty_cache()
            return run(torch.device("cpu"))
