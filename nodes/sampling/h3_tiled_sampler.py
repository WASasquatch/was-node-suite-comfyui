"""KSampler for MiniMax H3 latents, run over overlapping tiles."""

from __future__ import annotations

import comfy.sample
import comfy.samplers
import comfy.utils
from comfy_api.latest import io, ui

from ...modules.latent import h3_derope, h3_extend
from ...modules.model import h3_tiles
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
                tooltip="Most tokens one tile holds, as `48000`, about 200 frames at 0.8 MP, "
                        "or `24000` on a 12 to 16 GB card.",
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


class H3TiledSampler(io.ComfyNode):
    """Denoise a MiniMax H3 latent as KSampler does, every model call run over overlapping tiles."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASH3TiledSampler",
            display_name="H3 Tiled Sampler",
            search_aliases=[
                "WASH3TiledSampler",
                "H3 Tiled Sampler",
                "tiled sampler",
                "ultimate upscale",
                "tiled diffusion",
                "minimax h3",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Denoise a MiniMax H3 latent with KSampler's settings in overlapping tiles across "
                "the frame and overlapping windows along the clip, blended at every step so the "
                "tiles stay one clip. For refining a long or upscaled H3 clip in tiles that each "
                "fit the card, run as H3 Low VRAM runs them; it does no upscaling itself. The "
                "panel shows where the tiles sit."
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
                    tooltip="The H3 latent to refine, such as an upscaled clip encoded again.",
                ),
                io.Float.Input(
                    "denoise", default=0.4, min=0.0, max=1.0, step=0.01,
                    tooltip="Share of the schedule run, `0.3` to `0.5` to refine, `1.0` from noise.",
                ),
                io.DynamicCombo.Input(
                    "tiling", options=_tiling_options(), display_name="tiling",
                    tooltip="`auto` splits the clip into windows along time under a token budget, "
                            "and across the frame only where a window will not fit; "
                            "`manual` sets tile and window sizes.",
                ),
                io.Latent.Input(
                    "audio", optional=True,
                    tooltip="The clip's audio latent, from VAE Encode Audio with the H3 audio VAE, "
                            "held as it is beside a video-only latent_image. Left empty, silence "
                            "is held.",
                ),
            ],
            outputs=[io.Latent.Output(
                display_name="LATENT",
                tooltip="The refined latent, video only when latent_image was video only.",
            )],
        )

    @classmethod
    def execute(cls, model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                latent_image, denoise, tiling, audio=None) -> io.NodeOutput:
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
            denoise: Share of the schedule run.
            tiling: The tiling widget's value.
            audio: An H3 audio latent held beside a video-only latent, or None.

        Returns:
            The refined latent, with a picture of the tiles.

        Raises:
            ValueError: The latent holds no video.
        """
        latent = latent_image
        shape = _video_shape(latent["samples"])
        if shape is None:
            raise ValueError(
                "H3 Tiled Sampler needs a MiniMax H3 video latent, from an H3 sampler or VAE "
                "Encode with the H3 video VAE"
            )
        video_only = getattr(latent["samples"], "tensors", None) is None
        if video_only:
            latent = _with_audio(latent, audio)
        settings = h3_tiles.tiling_settings(tiling)
        tiled = h3_tiles.tiled_model(model, settings)
        samples = comfy.sample.fix_empty_latent_channels(
            tiled, latent["samples"],
            latent.get("downscale_ratio_spacial", None),
            latent.get("downscale_ratio_temporal", None),
        )
        noise = comfy.sample.prepare_noise(samples, seed, latent.get("batch_index"))
        callback = preview.prepare_callback(tiled, steps)
        result = comfy.sample.sample(
            tiled, noise, steps, cfg, sampler_name, scheduler, positive, negative, samples,
            denoise=denoise, noise_mask=latent.get("noise_mask"), callback=callback,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed,
        )
        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = result
        if video_only:
            out.pop("noise_mask", None)
            out["samples"] = h3_extend.split(out)[0]
        rows, height, width = shape
        even_h, even_w = height + height % 2, width + width % 2
        text = max((int(cond[0].shape[1]) for cond in (positive or []) + (negative or [])
                    if hasattr(cond[0], "shape") and cond[0].ndim >= 2), default=0)
        if settings[0] == "manual":
            plan = h3_tiles.manual_tiling(rows, even_h, even_w, *settings[1:])
        else:
            plan = h3_tiles.auto_tiling(rows, even_h, even_w, text, max_tokens=settings[1])
        picture = h3_tiles.plot(plan, rows, even_h, even_w)
        return io.NodeOutput(out, ui=ui.PreviewImage(picture, cls=cls))
