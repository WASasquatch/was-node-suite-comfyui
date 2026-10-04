"""Inventing frames between the frames of a sequence with EMA-VFI."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import EMA_VFI_MODEL
from ...modules.interface import batch_report
from ...modules.model import frame_interpolation

NODE_NAME = "EMA-VFI Frame Interpolation"

#: Most frames a run will produce before it refuses.
MAX_FRAMES = 4096


class EMAVFIFrameInterpolation(io.ComfyNode):
    """Interpolate a sequence to a higher frame rate."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASEMAVFIFrameInterpolation",
            display_name=NODE_NAME,
            search_aliases=[
                "WASEMAVFIFrameInterpolation", NODE_NAME,
                "Image Frame Interpolate",
                "frame interpolation",
                "EMA-VFI",
                "interpolate frames",
                "slow motion",
                "increase frame rate",
                "fps",
                "inbetween frames",
                "smooth video",
            ],
            category="WAS Suite/Animation",
            description=(
                "Raise a sequence's frame rate by inventing frames between the ones it has, "
                "using EMA-VFI's motion estimate rather than fading one frame into the next. "
                "The weights come from EMA-VFI Video Model Loader. A multiplier above 2 needs one of "
                "the 'ours_t' checkpoints, which were trained to land anywhere between two "
                "frames rather than only halfway."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The sequence to interpolate, in order. Needs at least two frames, all "
                        "the same size."
                    ),
                ),
                EMA_VFI_MODEL.Input(
                    "ema_vfi_model",
                    tooltip=(
                        "The interpolation network, from EMA-VFI Video Model Loader, which is where "
                        "the checkpoint is chosen. One loader can feed several nodes so the "
                        "network is built once."
                    ),
                ),
                io.Int.Input(
                    "multiplier",
                    default=2,
                    min=2,
                    max=8,
                    tooltip=(
                        "How many times the frame rate goes up. 2 puts one new frame in each "
                        "gap, 4 puts three. Above 2 needs an 'ours_t' checkpoint."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip=(
                        "The longer sequence, originals included. A run of n frames at "
                        "multiplier m answers (n - 1) * m + 1 frames."
                    ),
                ),
                io.Int.Output(
                    display_name="frame_count",
                    tooltip="How many frames came back, counting the originals.",
                ),
            ],
        )

    @classmethod
    def execute(cls, images, ema_vfi_model, multiplier=2) -> io.NodeOutput:
        """Interpolate the sequence.

        Raises:
            ValueError: Fewer than two frames arrived, the multiplier needs a timestep the
                chosen weights were not trained for, or the run would produce more frames
                than :data:`MAX_FRAMES`.
            MemoryError: Neither free memory nor a scratch drive can hold the result.
        """
        from ...modules.image import scratch

        if getattr(images, "ndim", 0) != 4:
            raise ValueError(f"{NODE_NAME} needs a batch of images shaped (frames, H, W, C).")
        frames = int(images.shape[0])
        if frames < 2:
            raise ValueError(
                f"{NODE_NAME} needs at least two frames to interpolate between; it was given "
                f"{frames}. Wire a batch, not a single image."
            )
        checkpoint = ema_vfi_model.name
        multiplier = int(multiplier)
        spec = frame_interpolation.spec_for(checkpoint)
        if multiplier > 2 and not spec.get("any_timestep", False):
            timestep_files = ", ".join(
                name for name, entry in frame_interpolation.CHECKPOINTS.items()
                if entry["any_timestep"]
            )
            raise ValueError(
                f"{checkpoint} was only trained to land halfway between two frames, so it can "
                f"only do a multiplier of 2. For {multiplier} use one of: {timestep_files}."
            )
        produced = (frames - 1) * multiplier + 1
        if produced > MAX_FRAMES:
            raise ValueError(
                f"{frames} frames at multiplier {multiplier} would produce {produced} frames, "
                f"over the {MAX_FRAMES} this node will attempt. Lower the multiplier or split "
                f"the sequence."
            )

        answer = scratch.allocate(
            (produced, int(images.shape[1]), int(images.shape[2]), 3),
            images.dtype,
            node=NODE_NAME,
            advice="Lowering multiplier or passing fewer frames also fits it.",
        )

        backend = ema_vfi_model.backend
        backend.load()
        net = backend.model
        device = next(net.parameters()).device

        # ComfyUI hands frames over as (frames, height, width, channels); the network wants
        # (1, channels, height, width) per frame, and only ever three channels.
        def as_planes(index):
            return images[index:index + 1, :, :, :3].permute(0, 3, 1, 2).to(device)

        def keep(at, planes):
            answer[at].copy_(planes[0].permute(1, 2, 0).clamp(0.0, 1.0))

        try:
            from comfy.utils import ProgressBar

            progress = ProgressBar(frames - 1)
        except Exception:
            progress = None

        for index in range(frames - 1):
            first, second = as_planes(index), as_planes(index + 1)
            keep(index * multiplier, first)
            for step in range(1, multiplier):
                keep(
                    index * multiplier + step,
                    frame_interpolation.interpolate(net, first, second, step / multiplier),
                )
            if progress is not None:
                progress.update(1)
        # The last frame closes the sequence; every loop above wrote only its own left edge.
        keep(produced - 1, as_planes(frames - 1))

        size, mode = batch_report.describe_images(answer)
        batch_report.publish(
            frames=int(answer.shape[0]),
            slots=1,
            size=size,
            mode=mode,
            memory=batch_report.memory_of(answer),
        )
        return io.NodeOutput(answer, int(answer.shape[0]))
