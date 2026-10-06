"""KSampler for MiniMax H3 latents, run over overlapping tiles: the work-in-progress plan."""

from __future__ import annotations

import comfy.sample
import comfy.samplers
import comfy.utils
from comfy_api.latest import io

from ...modules.interface import preview as published
from ...modules.interface import run_result
from ...modules.latent import h3_derope, h3_extend
from ...modules.model import h3_tiles_wip as h3_tiles
from ...modules.sampling import preview


def _with_audio(latent: dict, audio) -> dict:
    """A video-only H3 latent joined with an audio half that sampling holds as it is."""
    import torch

    video = latent["samples"]
    span = h3_extend.audio_span(h3_extend.frames_for(int(video.shape[2])))
    sound = audio.get("samples") if isinstance(audio, dict) else None
    if getattr(sound, "tensors", None) is not None:
        sound = sound.tensors[-1]
    if not isinstance(sound, torch.Tensor) or sound.ndim != 4:
        sound = torch.zeros([video.shape[0], 32, 2, span], dtype=video.dtype, device=video.device)
    sound = sound[..., :span].to(video)
    if sound.shape[-1] < span:
        sound = torch.nn.functional.pad(sound, (0, span - sound.shape[-1]))
    joined = dict(latent)
    joined.update(h3_extend.join(video, sound))
    joined["noise_mask"] = h3_derope.audio_row_strength(0.0, video, sound)
    return joined


#: Choices for when tiles are guided by latent_image.
ANCHOR_MODES = ("auto", "on", "off")


def _start_sigmas(model, scheduler: str, steps: int, strength: float):
    """The scheduler's sigmas for ``steps`` steps starting nearest ``strength``.

    Args:
        model: The model patcher sampling.
        scheduler: Scheduler name.
        steps: Steps to run.
        strength: Noise level to start from, ``0`` to ``1``.

    Returns:
        ``steps + 1`` sigmas, the last ``0``.
    """
    import torch

    sampling = model.get_model_object("model_sampling")
    steps = max(1, int(steps))
    if strength >= 0.9999:
        return comfy.samplers.calculate_sigmas(sampling, scheduler, steps)

    def tail(total):
        return comfy.samplers.calculate_sigmas(sampling, scheduler, total)[-(steps + 1):]

    seen = {}

    def miss(total):
        if total not in seen:
            seen[total] = float(tail(total)[0])
        return abs(seen[total] - strength)

    totals = sorted({min(steps * 1000, int(round(steps * 1.15 ** k))) for k in range(0, 50)} | {steps})
    best = min(totals, key=miss)
    low, high = max(steps, int(best / 1.15)), int(best * 1.15) + 1
    while high - low > 1:
        middle = (low + high) // 2
        if miss(middle) < miss(best):
            best = middle
        if seen[middle] > strength:
            low = middle
        else:
            high = middle
    best = min((low, high, best), key=miss)
    sigmas = tail(best)
    sigmas = sigmas if isinstance(sigmas, torch.Tensor) else torch.tensor(sigmas)
    if abs(float(sigmas[0]) - strength) > 0.02 and scheduler in ("karras", "exponential", "kl_optimal"):
        import comfy.k_diffusion.sampling as k_sampling

        builders = {"karras": k_sampling.get_sigmas_karras,
                    "exponential": k_sampling.get_sigmas_exponential,
                    "kl_optimal": comfy.samplers.kl_optimal_scheduler}
        low = float(sampling.sigma_min)
        sigmas = builders[scheduler](n=steps, sigma_min=low, sigma_max=max(low * 1.01, strength))
    return sigmas


def _anchored(positive, video, rows: int):
    """``positive`` with the first frame of every chunk of ``video`` held as a guide.

    Args:
        positive: Positive conditioning.
        video: The H3 video latent ``[B, 24, rows, H, W]`` the run refines.
        rows: Latent rows of the clip.

    Returns:
        The conditioning, with one single-frame guide per chunk added to any it carried.
    """
    import node_helpers

    held = list((positive[0][1] if positive else {}).get("minimax_keyframes", []))
    for row in range(0, int(rows), h3_tiles.CHUNK_ROWS):
        held.append({"resolved_frame_index": _frames(row),
                     "latent": video[:, :, row:row + 1].clone()})
    return node_helpers.conditioning_set_values(positive, {"minimax_keyframes": held})


def _frames(rows: int) -> int:
    """Video frames ``rows`` latent rows cover from the start of a chunk."""
    return sum(h3_tiles.ROW_FRAMES[k % h3_tiles.CHUNK_ROWS] for k in range(int(rows)))


def _publish(plan, rows: int, height: int, width: int, start, anchored: bool,
             starts=None) -> None:
    """Publish the tile plan and its figures, for the node's own panel.

    Args:
        plan: The tiling the run used.
        rows: Latent rows of the clip.
        height: Latent height, even.
        width: Latent width, even.
        start: Sigma the run started from, or None.
        anchored: Whether every chunk's first frame guided the run.
        starts: Latent rows each scene of the clip opens on, or None for one scene.
    """
    try:
        published.publish_output(h3_tiles.plot(plan, rows, height, width, start=start, starts=starts))
        if not run_result.watching():
            return
        across = len(plan.heights) * len(plan.widths)
        window = max(b - a for a, b in plan.rows)
        shared = max((plan.rows[i][1] - plan.rows[i + 1][0] for i in range(len(plan.rows) - 1)),
                     default=0)
        counts = {"windows": len(plan.rows), "tiles across": across, "frames a window": _frames(window)}
        if shared:
            counts["frames shared"] = _frames(shared)
        cuts = len(h3_tiles.scene_spans(starts, rows)) - 1
        if cuts:
            counts["cuts"] = cuts
        if start is not None:
            counts["start sigma"] = round(float(start), 3)
        crowded = (across > 1 and not anchored and start is not None
                   and float(start) > h3_tiles.TILED_SIGMA)
        summary = (f"{len(plan.rows)} window(s), the frame whole" if across == 1
                   else f"{len(plan.rows)} window(s) x {across} tiles across the frame")
        note = h3_tiles.cut_report(starts, rows, len(plan.rows))
        if note:
            summary += f"; {note}"
        if crowded:
            summary += f"; above sigma {h3_tiles.TILED_SIGMA} a subject can repeat across tiles"
        run_result.publish(
            status=run_result.WARNING if crowded else run_result.OK,
            summary=summary,
            counts=counts,
            facts={"frame": f"{width * h3_tiles.LATENT_SCALE}x{height * h3_tiles.LATENT_SCALE} px",
                   "anchor": "first frame of every 17" if anchored else "off"},
        )
    except Exception:
        run_result.publish(status=run_result.WARNING, summary="the tile plan could not be drawn")


def _video_shape(samples):
    """``(rows, height, width)`` of an H3 latent's video, or ``None`` when it has none."""
    video = (getattr(samples, "tensors", None) or [samples])[0]
    if getattr(video, "ndim", 0) != 5:
        return None
    return int(video.shape[2]), int(video.shape[3]), int(video.shape[4])


def _tiling_options() -> list:
    """The ``auto`` and ``manual`` tilings with their settings."""
    return [
        io.DynamicCombo.Option(key="auto", inputs=[
            io.Int.Input(
                "max_tokens", default=h3_tiles.AUTO_TOKENS, min=4000, max=1048000, step=1000,
                tooltip="Most tokens one tile holds, as `64000` on a 24 GB card or `24000` on a "
                        "12 to 16 GB card.",
            ),
        ]),
        io.DynamicCombo.Option(key="manual", inputs=[
            io.Int.Input(
                "tile_width", default=1024, min=64, max=16384, step=32,
                tooltip="Pixels a tile spans across the frame, as `1024`.",
            ),
            io.Int.Input(
                "tile_height", default=576, min=64, max=16384, step=32,
                tooltip="Pixels a tile spans down the frame, as `576`.",
            ),
            io.Int.Input(
                "window_frames", default=0, min=0, max=100000, step=1,
                tooltip="Frames a time window spans, as `141`, or `0` for the whole clip.",
            ),
            io.Int.Input(
                "overlap", default=128, min=32, max=4096, step=32,
                tooltip="Pixels neighbouring tiles share, as `128`.",
            ),
            io.Int.Input(
                "window_overlap", default=17, min=1, max=1000, step=1,
                tooltip="Frames neighbouring time windows share, as `17`.",
            ),
        ]),
    ]


class H3TiledSamplerWIP(io.ComfyNode):
    """Denoise a MiniMax H3 latent as KSampler does, every model call run over overlapping tiles."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASH3TiledSamplerWIP",
            display_name="H3 Tiled Sampler [WIP]",
            search_aliases=[
                "WASH3TiledSamplerWIP",
                "H3 Tiled Sampler [WIP]",
                "H3 Tiled Sampler",
                "tiled sampler",
                "ultimate upscale",
                "tiled diffusion",
                "minimax h3",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Work in progress. Denoise a MiniMax H3 latent with KSampler's settings in overlapping tiles across "
                "the frame and overlapping windows along the clip, blended at every step so the "
                "tiles stay one clip. For refining a long or upscaled H3 clip in tiles that each "
                "fit the card, run as H3 Low VRAM runs them; it does no upscaling itself. A clip "
                "joined by H3 Extend Append keeps its cuts: where it takes several time windows, "
                "each stays inside one scene. The panel shows where the tiles sit, marks each "
                "cut and gives the sigma the run started from."
            ),
            inputs=[
                io.Model.Input("model", tooltip="The MiniMax H3 model, after any LoRA."),
                io.Int.Input(
                    "seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True,
                    tooltip="The noise seed, `0` as good as any.",
                ),
                io.Int.Input(
                    "steps", default=8, min=1, max=10000,
                    tooltip="Denoising steps over the whole schedule, as `8` for a turbo LoRA or "
                            "`25` for the base model.",
                ),
                io.Float.Input(
                    "cfg", default=1.0, min=0.0, max=100.0, step=0.1, round=0.01,
                    tooltip="Guidance, `1` for turbo LoRAs and distilled models.",
                ),
                io.Combo.Input(
                    "sampler_name", options=comfy.samplers.KSampler.SAMPLERS,
                    tooltip="The solver each step runs, such as `euler` or `res_multistep`.",
                ),
                io.Combo.Input(
                    "scheduler", options=comfy.samplers.KSampler.SCHEDULERS,
                    tooltip="How the noise level falls from step to step, such as `simple`.",
                ),
                io.Conditioning.Input("positive", tooltip="The clip's positive conditioning."),
                io.Conditioning.Input("negative", tooltip="The clip's negative conditioning."),
                io.Latent.Input(
                    "latent_image",
                    tooltip="The H3 latent to refine, such as an upscaled clip encoded again. A "
                            "video-only latent has its audio held; a joined audio and video "
                            "latent has both refined.",
                ),
                io.Float.Input(
                    "strength", default=0.45, min=0.0, max=1.0, step=0.01,
                    tooltip="Noise level the refine starts from, on any scheduler: `0.3` keeps the "
                            "clip and sharpens it, `0.5` adds detail, `0.8` redraws detail on the "
                            "clip's own layout, `0.93` redraws nearly all of it, `1.0` from noise.",
                ),
                io.DynamicCombo.Input(
                    "tiling", options=_tiling_options(), display_name="tiling",
                    tooltip="`auto` keeps the whole clip in one time window and splits the frame into "
                            "the fewest tiles under the token budget, adding time windows only where no "
                            "tiling of one window fits or the clip runs past 362 frames; `manual` sets tile "
                            "and window sizes. Added windows never cross a cut the latent records.",
                ),
                io.Latent.Input(
                    "audio", optional=True,
                    tooltip="The clip's audio latent, from VAE Encode Audio with the H3 audio VAE, "
                            "held as it is beside a video-only latent_image. Left empty, silence "
                            "is held.",
                ),
                io.Combo.Input(
                    "anchor", options=list(ANCHOR_MODES), default="auto",
                    tooltip="Guides each tile with its own part of latent_image at the first frame "
                            "of every 17: `auto` when tiles split the frame, `on` always, `off` "
                            "never.",
                ),
            ],
            outputs=[io.Latent.Output(
                display_name="LATENT",
                tooltip="The refined latent, video only when latent_image was video only.",
            )],
        )

    @classmethod
    def execute(cls, model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                latent_image, strength, tiling, audio=None, anchor="auto") -> io.NodeOutput:
        """Sample the latent over tiles.

        Args:
            model: The model.
            seed: Noise seed.
            steps: Denoising steps.
            cfg: Guidance scale.
            sampler_name: Solver name.
            scheduler: Scheduler name.
            positive: Positive conditioning.
            negative: Negative conditioning.
            latent_image: The latent to refine.
            strength: Noise level the run starts from.
            tiling: The tiling widget's value.
            audio: An H3 audio latent held beside a video-only latent, or None.
            anchor: A value of :data:`ANCHOR_MODES`.

        Returns:
            The refined latent, with a picture of the tiles.

        Raises:
            ValueError: The latent holds no video.
        """
        latent = latent_image
        shape = _video_shape(latent["samples"])
        if shape is None:
            raise ValueError(
                "H3 Tiled Sampler [WIP] needs a MiniMax H3 video latent, from an H3 sampler or VAE "
                "Encode with the H3 video VAE"
            )
        if float(strength) <= 0.0:
            return io.NodeOutput(latent_image)
        rows, height, width = shape
        even_h, even_w = height + height % 2, width + width % 2
        settings = h3_tiles.tiling_settings(tiling)
        starts = h3_extend.scene_starts(latent_image)
        text = max((int(cond[0].shape[1]) for cond in (positive or []) + (negative or [])
                    if hasattr(cond[0], "shape") and cond[0].ndim >= 2), default=0)
        if settings[0] == "manual":
            planned = h3_tiles.manual_tiling(rows, even_h, even_w, *settings[1:])
        else:
            planned = h3_tiles.auto_tiling(rows, even_h, even_w, text, max_tokens=settings[1])
        planned = h3_tiles.scene_tiling(planned, settings, starts, rows)
        split = len(planned.heights) * len(planned.widths) > 1
        anchored = anchor == "on" or (anchor == "auto" and split)
        if anchored:
            video = (getattr(latent["samples"], "tensors", None) or [latent["samples"]])[0]
            positive = _anchored(positive, video, rows)
        video_only = getattr(latent["samples"], "tensors", None) is None
        if video_only:
            latent = _with_audio(latent, audio)
        tiled = h3_tiles.tiled_model(model, settings, starts)
        samples = comfy.sample.fix_empty_latent_channels(
            tiled, latent["samples"],
            latent.get("downscale_ratio_spacial", None),
            latent.get("downscale_ratio_temporal", None),
        )
        noise = comfy.sample.prepare_noise(samples, seed, latent.get("batch_index"))
        sigmas = _start_sigmas(tiled, scheduler, steps, float(strength))
        callback = preview.prepare_callback(tiled, steps)
        result = comfy.sample.sample(
            tiled, noise, steps, cfg, sampler_name, scheduler, positive, negative, samples,
            noise_mask=latent.get("noise_mask"), sigmas=sigmas, callback=callback,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed,
        )
        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = result
        if video_only:
            out.pop("noise_mask", None)
            out["samples"] = h3_extend.split(out)[0]
        plan, start = h3_tiles.last_run(tiled)
        _publish(plan or planned, rows, even_h, even_w, start, anchored, starts)
        return io.NodeOutput(out)
