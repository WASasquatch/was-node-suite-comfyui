"""Depth estimation with MiDaS."""

from __future__ import annotations

import numpy as np
from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import MIDAS_MODEL
from ...modules.convert.tensors import pil2tensor, tensor2pil

REQUIRES = "midas"

logger = log.get_logger("nodes.ai.midas_depth")


class MidasDepthApproximation(io.ComfyNode):
    """Estimate a depth map for every image in a batch."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiDaS Depth Approximation",
            display_name="MiDaS Depth Approximation",
            search_aliases=["MiDaS Depth Approximation", "depth map", "midas", "depth"],
            category="WAS Suite/Image/AI",
            description=(
                "Estimate how far away each part of an image is and return it as a greyscale "
                "depth map, for a depth ControlNet or a displacement effect. Enable "
                "features.midas to load this node."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="The images to estimate depth for. A whole batch is processed.",
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
                        "the processor instead, for a machine with no room left on the card."
                    ),
                ),
                io.Boolean.Input(
                    "invert_depth",
                    default=False,
                    tooltip=(
                        "`off` = near things white and far things black, which is what depth "
                        "ControlNets expect; `on` = flipped, for a model or effect that "
                        "wants near things dark."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip=(
                        "The depth maps, as a greyscale batch the same length and size as the "
                        "input."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, image, midas_model, use_cpu, invert_depth) -> io.NodeOutput:
        import torch
        from PIL import Image, ImageOps

        import comfy.utils

        from ...modules.image import scratch

        device_name = "cpu" if use_cpu else None
        backend = midas_model.on(device_name)

        device = backend.load()
        logger.info("MiDaS is using device: %s", device)

        processor = backend.processor
        model = backend.model
        progress = comfy.utils.ProgressBar(len(image))

        estimates = []
        for index, frame in enumerate(image):
            # The frame is handed to the processor in BGR channel order.
            source = np.ascontiguousarray(np.array(tensor2pil(frame))[:, :, ::-1])

            logger.info("Approximating depth for image %s/%s", index + 1, len(image))

            inputs = processor(images=source, return_tensors="pt").to(device)
            with torch.no_grad():
                prediction = model(**inputs).predicted_depth
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=source.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            estimates.append(prediction)
            progress.update(1)

        # The floor and span are taken across the batch rather than per frame.
        floor = torch.stack([found.min() for found in estimates]).min()
        span = (torch.stack([found.max() for found in estimates]).max() - floor).clamp_min(1e-6)

        depths = None
        for index, found in enumerate(estimates):
            scaled = (found - floor) / span
            scaled = (scaled * 255).clamp(0, 255).round().cpu().numpy().astype(np.uint8)

            depth = Image.fromarray(scaled)
            if invert_depth:
                depth = ImageOps.invert(depth)

            frame = pil2tensor(depth.convert("RGB"))
            if depths is None:
                depths = scratch.allocate(
                    (len(estimates),) + tuple(frame.shape[1:]),
                    node="MiDaS Depth Approximation",
                    advice="Passing fewer or smaller frames also fits it.",
                )
            depths[index] = frame[0]

        return io.NodeOutput(depths)
