"""Raising a video's resolution with PS-SR."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ...modules import log
from ...modules.convert.tensors import image_planes
from ...modules.interface import size_report
from ...modules.interface.progress import progress_bar

REQUIRES = "pssr"

logger = log.get_logger("nodes.ai.pssr")

#: Resampling used to reach the target grid before the model restores detail into it.
INTERPOLATION = ("lanczos", "bicubic", "bilinear")

#: Name the node goes by in the log and in its refusals.
NAME = "Video Super Resolution (PS-SR)"

#: Closing sentence of a refusal for want of memory.
ADVICE = "Lowering scale or passing fewer frames also fits it."


class PSSRSuperResolution(io.ComfyNode):
    """Raise a video's resolution and restore the detail a plain resize cannot invent."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASPSSRSuperResolution",
            display_name="Video Super Resolution (PS-SR)",
            search_aliases=[
                "WASPSSRSuperResolution", "PS-SR", "PSSR",
                "video super resolution", "upscale video", "restore video", "video upscaler",
            ],
            category="WAS Suite/Image/Process",
            description=(
                "Raise a video's resolution with PS-SR, which resamples to the target size and "
                "then puts detail back with a diffusion pass, twice: once for a steady result "
                "and once for a sharp one, blended by frequency. Weights are placed by hand and "
                "never downloaded. Long clips are covered by sliding windows, so memory depends "
                "on the window rather than the length."
            ),
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip=(
                        "Wan 2.1 T2V-1.3B, from Load Diffusion Model. Finetunes of it work; 14B "
                        "and other families are refused, since the restoration weights are 1.3B."
                    ),
                ),
                io.Conditioning.Input(
                    "positive",
                    tooltip=(
                        "What to restore towards, from CLIP Text Encode on a wan CLIP Loader. "
                        "Name the subject and the finish, eg 'a red car, sharp, fine detail'."
                    ),
                ),
                io.Conditioning.Input(
                    "negative",
                    tooltip=(
                        "What to avoid, eg 'blurry, jpeg artifacts, over-smooth'. Carried but "
                        "not read at this method's guidance; wire Conditioning Zero Out if unused."
                    ),
                ),
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The frames to raise, in order. Treated as one continuous shot, so a cut "
                        "inside the batch is blended across rather than respected."
                    ),
                ),
                io.Float.Input(
                    "scale",
                    default=1.5,
                    min=1.0,
                    max=4.0,
                    step=0.25,
                    tooltip=(
                        "Size multiplier. 1.5 = half again as large; 1.0 = restore at the "
                        "current size. Cost rises with the square of it."
                    ),
                ),
                io.Combo.Input(
                    "interpolation",
                    options=list(INTERPOLATION),
                    default="lanczos",
                    tooltip=(
                        "How the frames reach the target size first. `lanczos` keeps the "
                        "most detail for the model to build on, `bicubic` is smoother over "
                        "flat areas, and `bilinear` is the cheapest."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=1,
                    min=0,
                    max=0xFFFFFFFF,
                    tooltip=(
                        "Seeds the diffusion noise, so the same seed restores the same way. "
                        "Any whole number; `0` is as good a seed as any."
                    ),
                ),
                io.Int.Input(
                    "window_frames",
                    default=33,
                    min=9,
                    max=81,
                    step=4,
                    tooltip=(
                        "Frames per pass, eg 33. Larger is steadier over time and costs more "
                        "memory. This, not the clip length, sets peak VRAM."
                    ),
                ),
                io.Int.Input(
                    "overlap_frames",
                    default=8,
                    min=0,
                    max=32,
                    step=1,
                    tooltip=(
                        "Frames shared between passes, eg 8. More hides the joins and costs "
                        "proportionally more; 0 = no sharing."
                    ),
                ),
                io.Int.Input(
                    "tile_size",
                    default=0,
                    min=0,
                    max=4096,
                    step=64,
                    tooltip=(
                        "0 = whole frame, halving only if it will not fit; 1280 = 1280px "
                        "patches. Tiling is for memory, not speed: overlaps average two passes "
                        "and band against the edges."
                    ),
                ),
                io.Int.Input(
                    "tile_overlap",
                    default=128,
                    min=0,
                    max=512,
                    step=16,
                    tooltip=(
                        "How much neighbouring patches share. The shared band is feathered "
                        "between them, so more hides the seams at proportionally more cost."
                    ),
                ),
                io.Float.Input(
                    "detail_strength",
                    default=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.05,
                    tooltip=(
                        "How much sharp pass to blend in. 0 = steady only; 1 = intended; "
                        "2 = exaggerated."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip="The frames at the new size, detail restored.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls, model, positive, negative, images, scale, interpolation, seed, window_frames,
        overlap_frames, tile_size, tile_overlap, detail_strength,
    ) -> io.NodeOutput:
        """Raise every frame and answer them as one batch.

        Raises:
            ValueError: No frames were given, they are not colour frames, or the overlap is not
                smaller than the window.
            FileNotFoundError: The checkout or its weights are not where they should be.
            MemoryError: Neither free memory nor a scratch drive has room for the result.
        """
        from ...modules.model import pssr

        frames = image_planes(images)
        if not frames:
            raise ValueError(f"{NAME} was given no frames.")
        if overlap_frames >= window_frames:
            raise ValueError(
                f"overlap_frames ({overlap_frames}) has to be smaller than window_frames "
                f"({window_frames}), or the windows would never advance."
            )
        source = images.contiguous() if images.ndim >= 4 else torch.stack(frames, dim=0)
        if source.ndim != 4 or source.shape[-1] < 3:
            raise ValueError(
                f"{NAME} restores colour frames shaped (frames, height, width, 3 or 4) and was "
                f"given {tuple(source.shape)}. Pass them through Images to RGB first."
            )

        root = pssr.find_root()
        target = cls.resize(source, scale, interpolation)
        logger.info(
            "PS-SR: %d frame(s), %dx%d -> %dx%d",
            target.shape[0], source.shape[2], source.shape[1], target.shape[2], target.shape[1],
        )

        device = cls.pick_device()
        supplied = pssr.dit_state_dict(model)
        context = pssr.conditioning_tensor(positive).detach().to("cpu", copy=True)
        against = pssr.conditioning_tensor(negative).detach().to("cpu", copy=True)
        pssr.release_comfy_models()
        # Built pipelines are reused while this fingerprint of the weights matches.
        key = pssr.fingerprint(supplied)
        base, draft = pssr.load_pipelines(
            root, torch.bfloat16, device, k_select=1.5, dit_state=supplied, dit_key=key,
        )
        # The wired conditioning stands in for the pipelines' own prompt encoding.
        with pssr.supplied_conditioning((base, draft), context, against):
            restored = cls.with_a_tile_that_fits(
                pssr, base, draft, target, seed, window_frames, overlap_frames,
                tile_size, tile_overlap, detail_strength,
            )
        size_report.publish(images, restored, action="restored", requested=target)
        return io.NodeOutput(restored)

    @staticmethod
    def pick_device() -> str:
        """The device to run on, preferring the one ComfyUI is using."""
        try:
            import comfy.model_management

            return str(comfy.model_management.get_torch_device())
        except Exception:
            return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def resize(frames: torch.Tensor, scale: float, interpolation: str) -> torch.Tensor:
        """Resample a batch to the target grid, rounded so the model's patching divides evenly.

        Args:
            frames: ``(count, height, width, channels)`` in 0 to 1.
            scale: Multiplier for both sides.
            interpolation: One of :data:`INTERPOLATION`.

        Returns:
            The resampled batch, sides rounded to a multiple of 16, or ``frames`` itself where
            that is its size already.

        Raises:
            MemoryError: Neither free memory nor a scratch drive has room for the batch.
        """
        import torch.nn.functional as functional

        from ...modules.image import scratch

        count, height, width, channels = frames.shape
        # Each side rounds to the nearest multiple of 16, the latent grid's step.
        target_h = max(16, int(round(height * scale / 16)) * 16)
        target_w = max(16, int(round(width * scale / 16)) * 16)
        if (target_h, target_w) == (height, width):
            return frames

        lanczos = interpolation == "lanczos"
        resized = scratch.allocate(
            (count, target_h, target_w, channels), torch.float32 if lanczos else frames.dtype,
            node=NAME, advice=ADVICE,
        )
        for index in range(count):
            if lanczos:
                resized[index] = PSSRSuperResolution.lanczos(frames[index], target_h, target_w)
            else:
                plane = frames[index : index + 1].permute(0, 3, 1, 2)
                resized[index] = functional.interpolate(
                    plane, size=(target_h, target_w), mode=interpolation, align_corners=False,
                )[0].permute(1, 2, 0)
            resized[index].clamp_(0, 1)
        return resized

    @staticmethod
    def lanczos(frame: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Resize one frame with a Lanczos kernel, through PIL.

        Args:
            frame: ``(height, width, channels)`` in 0 to 1.
            height: Target height.
            width: Target width.

        Returns:
            A float32 ``(height, width, channels)`` frame.
        """
        import numpy as np
        from PIL import Image

        array = (frame.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        resized = Image.fromarray(array).resize((width, height), Image.LANCZOS)
        return torch.from_numpy(np.asarray(resized, dtype=np.float32) / 255.0)

    @classmethod
    def with_a_tile_that_fits(
        cls, pssr, base, draft, frames, seed, window, overlap, tile, tile_overlap, strength,
    ) -> torch.Tensor:
        """Run at the largest tile the card accepts, halving whenever it refuses.

        Args:
            pssr: The runtime module.
            base: The steady pipeline.
            draft: The sharp pipeline.
            frames: The resampled batch to restore.
            seed: Diffusion seed.
            window: Frames per pass.
            overlap: Frames shared between consecutive passes.
            tile: Longest side to try first, or 0 to begin with the whole frame.
            tile_overlap: Pixels shared between neighbouring patches.
            strength: How much of the sharp pass to blend in.

        Returns:
            The restored batch.

        Raises:
            torch.OutOfMemoryError: Even the smallest tile did not fit.
        """
        longest = max(frames.shape[1], frames.shape[2])
        attempt = longest if tile <= 0 else min(tile, longest)
        if tile <= 0:
            logger.info("PS-SR: tile_size 0, starting with the whole frame at %d", attempt)

        while True:
            try:
                return cls.run_windows(
                    pssr, base, draft, frames, seed, window, overlap,
                    attempt, tile_overlap, strength,
                )
            except torch.OutOfMemoryError:
                if attempt <= 256:
                    raise
                pssr.release_comfy_models()
                attempt = max(256, (attempt // 2 // 16) * 16)
                logger.warning(
                    "PS-SR: out of memory, retrying with a %d tile. Tiling averages the "
                    "overlaps, so expect the joins to be softer than a single pass.",
                    attempt,
                )

    @classmethod
    def run_windows(
        cls, pssr, base, draft, frames, seed, window, overlap, tile, tile_overlap, strength,
    ) -> torch.Tensor:
        """Restore every frame, covering time and the frame itself with overlapping windows.

        Args:
            pssr: The runtime module.
            base: The steady pipeline.
            draft: The sharp pipeline.
            frames: The resampled batch to restore.
            seed: Diffusion seed.
            window: Frames per pass.
            overlap: Frames shared between consecutive passes.
            tile: Longest side of a spatial patch.
            tile_overlap: Pixels shared between neighbouring patches.
            strength: How much of the sharp pass to blend in.

        Returns:
            The restored batch, the same shape as ``frames``, any alpha carried from ``frames``.

        Raises:
            MemoryError: Neither free memory nor a scratch drive has room for the batch.
        """
        from ...modules.image import scratch
        from ...nodes.image.process.image_frequency_blend import ImageFrequencyBlend

        count, height, width, channels = frames.shape
        window = min(window, count)
        # Patches are 16:9 along the longer side of the frame.
        if width >= height:
            tile_w, tile_h = min(tile, width), min(round(tile * 9 / 16 / 16) * 16, height)
        else:
            tile_h, tile_w = min(tile, height), min(round(tile * 9 / 16 / 16) * 16, width)
        tile_h, tile_w = max(64, tile_h), max(64, tile_w)

        times = pssr.window_starts(count, window, overlap)
        rows = pssr.window_starts(height, tile_h, min(tile_overlap, tile_h - 1))
        columns = pssr.window_starts(width, tile_w, min(tile_overlap, tile_w - 1))
        # Feathers span what neighbouring windows actually share.
        time_feather = pssr.shared_extent(times, window)
        row_feather = pssr.shared_extent(rows, tile_h)
        column_feather = pssr.shared_extent(columns, tile_w)
        patches = len(times) * len(rows) * len(columns)
        logger.info(
            "PS-SR: %d temporal window(s) of %d frame(s), %dx%d patches of %dx%d, %d in all",
            len(times), window, len(rows), len(columns), tile_w, tile_h, patches,
        )

        # The steady sum becomes the result and carries every channel of ``frames``.
        steady_total = scratch.allocate(
            frames.shape, frames.dtype, node=NAME, advice=ADVICE,
        ).zero_()
        sharp_total = scratch.allocate(
            (count, height, width, 3), frames.dtype, node=NAME, advice=ADVICE,
        ).zero_()
        weight = scratch.allocate(
            (count, height, width, 1), frames.dtype, node=NAME, advice=ADVICE,
        ).zero_()
        bar = progress_bar(patches)
        done = 0

        for start in times:
            # The outer side of the first and the last window is left unfeathered.
            frame_ramp = pssr.blend_weights(
                window, time_feather, frames.device, frames.dtype,
                lead=start != times[0], trail=start != times[-1],
            )
            for top in rows:
                for left in columns:
                    done += 1
                    logger.info("PS-SR: patch %d/%d, frames %d-%d at %d,%d",
                                done, patches, start, start + window - 1, left, top)
                    chunk = frames[start : start + window, top : top + tile_h,
                                   left : left + tile_w, :3]
                    steady, sharp = cls.restore_chunk(pssr, base, draft, chunk, seed)
                    ramp = (
                        frame_ramp.view(-1, 1, 1, 1)
                        * pssr.blend_weights(
                            tile_h, row_feather, steady.device, steady.dtype,
                            lead=top != rows[0], trail=top != rows[-1],
                        ).view(1, -1, 1, 1)
                        * pssr.blend_weights(
                            tile_w, column_feather, steady.device, steady.dtype,
                            lead=left != columns[0], trail=left != columns[-1],
                        ).view(1, 1, -1, 1)
                    )
                    window_slice = (slice(start, start + window),
                                    slice(top, top + tile_h),
                                    slice(left, left + tile_w))
                    steady_total[window_slice + (slice(0, 3),)] += steady * ramp
                    sharp_total[window_slice] += sharp * ramp
                    weight[window_slice] += ramp
                    shown = min(start, count - 1)
                    settled = steady_total[shown, ..., :3] / weight[shown].clamp(min=1e-6)
                    bar.update(1, preview=settled.clamp_(0, 1))

        # Each finished frame blends its two passes by frequency and replaces its steady sum.
        for index in range(count):
            divisor = weight[index].clamp(min=1e-6)
            steady = (steady_total[index, ..., :3] / divisor).clamp_(0, 1)
            sharp = (sharp_total[index] / divisor).clamp_(0, 1)
            blended = ImageFrequencyBlend.blend_one(
                steady, sharp, cutoff=0.20, order=2, strength=strength, border=2,
            )
            steady_total[index, ..., :3] = blended.clamp_(0, 1)
            if channels > 3:
                steady_total[index, ..., 3:] = frames[index, ..., 3:]
        return steady_total

    @staticmethod
    def restore_chunk(pssr, base, draft, chunk, seed):
        """Run one window through both pipelines.

        Args:
            pssr: The runtime module.
            base: The steady pipeline.
            draft: The sharp pipeline.
            chunk: One patch, ``(frames, height, width, 3)``.
            seed: Diffusion seed.

        Returns:
            ``(steady frames, sharp frames)``, each a batch shaped like ``chunk``.
        """
        images = pssr.to_pil(chunk)
        shared = dict(
            prompt="", negative_prompt="", input_video=images,
            denoising_strength=1.0, seed=seed, tiled=False,
            width=chunk.shape[2], height=chunk.shape[1], num_frames=chunk.shape[0],
            cfg_scale=1.0, timestep_draft_list=[599, 499, 399],
        )
        steady, draft_latents, features = base(
            timestep_base=699, k_select=1.5, **shared,
        )
        sharp_list = draft(
            latents_next=draft_latents, latents_feature_list=features, **shared,
        )
        # The last draft branch is the sharp result.
        return pssr.from_pil(steady), pssr.from_pil(sharp_list[-1])
