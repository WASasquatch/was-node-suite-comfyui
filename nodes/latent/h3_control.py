"""Per-segment structural control and inpainting for a MiniMax H3 extend loop."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log

logger = log.get_logger("nodes.h3_control")

NODE_NAME = "H3 Control"

CONTROL_HINT = (
    "The whole video's control frames at 24 fps, frame 0 lining up with the finished clip's "
    "frame 0: pose, depth, canny, lines or any map the patch takes. Cropped to the window's "
    "canvas; a short video holds its last frame."
)

STRENGTH_HINT = (
    "How hard the control frames steer: `0.0` = off, model passed through; `1.0` = as "
    "trained; `0.5` = a looser follow the prompt can bend; `1.5` = stricter."
)

START_HINT = (
    "Where in the noise schedule control starts: `0.0` = from the first step; `0.2` = a "
    "little later, leaving the opening steps to the prompt."
)

END_HINT = (
    "Where in the noise schedule control stops: `1.0` = to the last step; `0.6` = released "
    "for the final steps, so fine detail follows the prompt rather than the control frames."
)

MASK_HINT = (
    "Where to regenerate, the whole video's length at 24 fps: `1` = regenerate, `0` = keep "
    "source_video. One frame holds for every frame. Needs source_video."
)

SOURCE_HINT = (
    "The video the mask was drawn on, the whole length at 24 fps. Kept outside the mask and "
    "regenerated inside it. Read only with a mask."
)


class H3Control(io.ComfyNode):
    """Steer one H3 segment with the stretch of a control video that lands on its window."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASH3Control",
            display_name=NODE_NAME,
            search_aliases=[
                "WASH3Control", NODE_NAME,
                "minimax h3 controlnet",
                "fun controlnet union",
                "pose control",
                "depth control",
                "canny control",
                "video inpaint",
            ],
            category="WAS Suite/Latent/Video",
            description=(
                "Drive each segment of a MiniMax H3 extend loop with its own stretch of one "
                "control video: pose, depth, canny, lines or any other map the H3 Fun "
                "ControlNet-Union patch takes. Wire the whole video's control frames once; "
                "every pass reads the frames that land on its window, carried frames "
                "included, so each scene follows the video in step. A mask and source video "
                "add inpainting, regenerating inside the mask and keeping the rest. Place it "
                "between H3 Extend Window's model output and the guider."
            ),
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip=(
                        "This segment's MiniMax H3 model, from H3 Extend Window's model output."
                    ),
                ),
                io.ModelPatch.Input(
                    "model_patch",
                    tooltip=(
                        "The H3 Fun ControlNet-Union patch, from Load Model Patch set to a "
                        "`minimax_h3_fun_controlnet_union` file in models/model_patches."
                    ),
                ),
                io.Vae.Input(
                    "vae",
                    tooltip="The H3 video VAE, which encodes this segment's control frames.",
                ),
                io.Latent.Input(
                    "window",
                    tooltip=(
                        "The window this segment samples, from H3 Extend Window. Its place on "
                        "the finished clip picks the control frames; a latent with no place "
                        "reads from frame 0."
                    ),
                ),
                io.Float.Input(
                    "strength", default=1.0, min=0.0, max=10.0, step=0.05, tooltip=STRENGTH_HINT,
                ),
                io.Float.Input(
                    "start_percent", default=0.0, min=0.0, max=1.0, step=0.01, advanced=True,
                    tooltip=START_HINT,
                ),
                io.Float.Input(
                    "end_percent", default=1.0, min=0.0, max=1.0, step=0.01, advanced=True,
                    tooltip=END_HINT,
                ),
                io.Image.Input("control_video", optional=True, tooltip=CONTROL_HINT),
                io.Mask.Input("mask", optional=True, tooltip=MASK_HINT),
                io.Image.Input("source_video", optional=True, tooltip=SOURCE_HINT),
            ],
            outputs=[
                io.Model.Output(
                    display_name="model",
                    tooltip="The segment's model steered by its control frames, for the guider.",
                ),
                io.String.Output(
                    display_name="report",
                    tooltip=(
                        "Which control frames the window reads, at what strength, and whether "
                        "it inpaints."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, model, model_patch, vae, window, strength, start_percent=0.0,
                end_percent=1.0, control_video=None, mask=None,
                source_video=None) -> io.NodeOutput:
        """Encode the window's stretch of the control video and patch the model with it.

        Raises:
            ValueError: Nothing to control with, a mask without source_video, an empty
                schedule range, a patch or model of the wrong kind, or a window that is not
                an H3 latent.
        """
        from ...modules.latent import h3_control, h3_extend

        if float(strength) <= 0.0:
            return io.NodeOutput(
                model, "strength 0.00: control off, the model passes through unchanged"
            )
        place = h3_control.placement(window)
        if float(end_percent) <= float(start_percent):
            raise ValueError(
                f"{NODE_NAME} has end_percent {float(end_percent):.2f} at or before "
                f"start_percent {float(start_percent):.2f}, so control would never run. Raise "
                f"end_percent above start_percent"
            )
        network = h3_control.checked(model, model_patch)
        if mask is not None and int(network.control_in_dim) < h3_control.INPAINT_CHANNELS:
            raise ValueError(
                f"this control patch takes {int(network.control_in_dim)} channels and "
                f"inpainting needs {h3_control.INPAINT_CHANNELS}. Disconnect mask, or load the "
                f"ControlNet-Union patch that inpaints"
            )

        video, _ = h3_extend.split(window)
        latent, regenerated = h3_control.control_latent(
            vae, video.shape, place["first"], control_video, mask, source_video,
            post_norm=bool(getattr(network, "inpaint_post_norm", False)),
        )
        patched = h3_control.controlled(model, model_patch, latent, float(strength),
                                        float(start_percent), float(end_percent))

        read = "control video" if control_video is not None else "mask"
        available = int((control_video if control_video is not None else mask).shape[0])
        report = h3_control.describe(place, available, read, strength, float(start_percent),
                                     float(end_percent), regenerated)
        logger.info("%s: %s", NODE_NAME, report)
        return io.NodeOutput(patched, report)
