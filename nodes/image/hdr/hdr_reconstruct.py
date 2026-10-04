"""Recovering the light a clipped highlight lost, as linear light above one."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ....modules import log
from ....modules.convert.tensors import image_planes, plane_shape
from ....modules.image import hdr
from ....modules.interface import run_result
from ....modules.interface.progress import progress_bar

REQUIRES = "preprocessors"

logger = log.get_logger("nodes.image.hdr")

#: Level a sample is counted as carrying recovered headroom above.
WHITE = 1.0


def _rgb(frame: torch.Tensor) -> torch.Tensor:
    """One image plane as a ``(1, height, width, 3)`` tensor on a 0 to 1 scale.

    Args:
        frame: Image plane, as :func:`modules.convert.tensors.image_planes` answers it.

    Returns:
        A tensor holding three colour channels, a lone channel repeated across all three
        and anything past the third dropped.
    """
    height, width, channels = plane_shape(frame)
    planes = frame.reshape(1, height, width, channels).float().clamp(0.0, 1.0)
    if channels >= 3:
        return planes[..., :3]
    return planes[..., :1].expand(1, height, width, 3)


def _publish_report(
    reconstructed: torch.Tensor, peak: float, above: float, dequantised: bool
) -> None:
    """Report the headroom that was recovered to the node's own interface.

    Never raises, and never changes what the node returns.

    Args:
        reconstructed: The linear light the node answers with.
        peak: The largest value in it.
        above: Share of its pixels over 1.0, from 0 to 1.
        dequantised: Whether the frames were dequantised before the network ran.
    """
    try:
        if not run_result.watching():
            return
        frames, height, width = (int(size) for size in reconstructed.shape[:3])
        share = above * 100.0
        recovered = peak > WHITE
        line = (
            f"peak {peak:.2f}, {share:.2f}% of the frame above 1.0"
            if recovered
            else f"peak {peak:.2f}, nothing above 1.0"
        )
        run_result.publish(
            # A peak at or below 1.0 draws the panel in the warning colour.
            status=run_result.OK if recovered else run_result.WARNING,
            summary=line if frames == 1 else f"{frames} frames, {line}",
            counts={
                "peak": round(peak, 2),
                "above one %": round(share, 2),
                "frames": frames,
            },
            facts={
                "size": f"{width} x {height}",
                "input": "dequantised" if dequantised else "as given",
            },
        )
    except Exception as error:
        logger.debug("no HDR reconstruction report was published (%s)", error)


class HDRReconstruct(io.ComfyNode):
    """Recover the linear light a clipped highlight lost."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASHDRReconstruct",
            display_name="HDR Reconstruct",
            search_aliases=[
                "WASHDRReconstruct", "HDR Reconstruct",
                "hdr",
                "highlight recovery",
                "inverse tone mapping",
                "linear light",
                "blown highlights",
                "hdrcnn",
            ],
            category="WAS Suite/Image/HDR",
            description=(
                "Rebuild the light a clipped highlight lost, answering linear light with "
                "everything above one kept. A blown sky, a lamp or a specular hit comes "
                "back as a gradient rather than one flat white, so a sun and its glow stay "
                "apart under a grade or in a 32-bit EXR. Every frame of a batch is "
                "reconstructed, and the largest value reached is answered beside the image."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The frames to reconstruct. Each one is run on its own and comes "
                        "back at the size it went in at."
                    ),
                ),
                io.Boolean.Input(
                    "dequantise",
                    default=True,
                    tooltip=(
                        "'true' = rebuild the levels an 8-bit file threw away first; "
                        "'false' = run the frames as they arrived. Leave it on for a PNG "
                        "or a JPEG, off for footage already in float."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip=(
                        "The reconstruction, as linear light on a 0 to 1 scale with the "
                        "recovered highlights above it. A preview clips it back to white, "
                        "so tone map it or write it to EXR to see the range."
                    ),
                ),
                io.Float.Output(
                    display_name="peak",
                    tooltip=(
                        "The largest value anywhere in the result. 40.07 = a clipped disc "
                        "rebuilt to 40 times white; 1.0 = nothing above white was "
                        "recovered. Divide by it to scale the frame back into range."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, images, dequantise=True) -> io.NodeOutput:
        """Reconstruct every frame and answer them as one batch beside its peak.

        Raises:
            ValueError: No frames were given.
            ModelUnavailable: The checkpoint is absent and ``features.network`` is off.
            MemoryError: Neither free memory nor a scratch drive can hold the result.
        """
        from ....modules.image import scratch
        from ....modules.model import hdrcnn

        frames = image_planes(images)
        if not frames:
            raise ValueError(
                "HDR Reconstruct was given no frames. Connect an image or a batch of them."
            )

        backend = hdrcnn.load()
        device = backend.load()
        network = backend.model

        bar = progress_bar(len(frames))
        reconstructed = None
        lifted = 0
        for index, frame in enumerate(frames):
            planes = _rgb(frame).to(device=device)
            if dequantise:
                planes = hdr.dequantise(planes)
            with torch.no_grad():
                linear = network(planes.permute(0, 3, 1, 2))[0].permute(1, 2, 0)
            if reconstructed is None:
                reconstructed = scratch.allocate(
                    (len(frames),) + tuple(linear.shape),
                    node="HDR Reconstruct",
                    advice="Passing fewer frames also fits it.",
                )
            reconstructed[index] = linear
            # A pixel counts once however many of its channels carry the headroom.
            lifted += int((reconstructed[index].amax(dim=-1) > WHITE).sum())
            bar.update(1)

        peak = float(reconstructed.amax())
        above = lifted / max(1, reconstructed[..., 0].numel())
        _publish_report(reconstructed, peak, above, bool(dequantise))
        return io.NodeOutput(reconstructed, peak)
