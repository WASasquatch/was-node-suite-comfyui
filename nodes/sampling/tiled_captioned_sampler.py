"""Samplers that refine a picture over overlapping tiles, each tile prompted by its own caption."""

from __future__ import annotations

import copy

from comfy_api.latest import io

from ...modules.compat.types import DICT
from ...modules.sampling import tile_captions, tile_refine, tiles
from ...modules.sampling.affine import ACTS_ON, patch_sampler
from ..latent.latent_affine import OPTIONS_HINT
from .affine_sampler import (
    ACTS_ON_HINT,
    AFFINE_INPUTS,
    SAMPLER_HINT,
    SCHEDULE_HINT,
    SPACE_HINT,
    build_spec,
    sampler_label,
    sampler_names,
    scheduler_names,
    warn_unsupported,
)

KSAMPLER_NAME = "Tiled Captioned KSampler"
CUSTOM_NAME = "Tiled Captioned Custom Sampler"

#: The tiling choices both samplers offer.
TILING_INPUTS = [io.DynamicCombo.Input(
    "tiling",
    options=[
        io.DynamicCombo.Option(key="auto", inputs=[]),
        io.DynamicCombo.Option(key="manual", inputs=[
            io.Int.Input(
                "tile_width", default=1024, min=128, max=16384, step=16,
                tooltip="Pixels a tile spans across the picture, as `1024`, or `512` for an SD1 model.",
            ),
            io.Int.Input(
                "tile_height", default=1024, min=128, max=16384, step=16,
                tooltip="Pixels a tile spans down the picture, as `1024`.",
            ),
            io.Int.Input(
                "overlap", default=256, min=0, max=8192, step=16,
                tooltip=(
                    "Least pixels neighbouring tiles share, blended across. `256` suits 1024 "
                    "tiles; raise it if a seam shows at a large upscale."
                ),
            ),
        ]),
    ],
    tooltip=(
        "`auto` = tiles the size the model was trained at, 512 for SD1 and 1024 for the rest, "
        "sharing a quarter of a tile; `manual` = set the tile size and overlap."
    ),
)]

#: The affine choices both samplers offer.
AFFINE_INPUTS_COMBO = [io.DynamicCombo.Input(
    "affine",
    options=[
        io.DynamicCombo.Option(key="off", inputs=[]),
        io.DynamicCombo.Option(key="on", inputs=[
            *AFFINE_INPUTS,
            io.Combo.Input("affine_space", options=["latent", "model"], default="latent",
                           tooltip=SPACE_HINT),
            io.Combo.Input("affine_acts_on", options=ACTS_ON, default="content",
                           tooltip=ACTS_ON_HINT),
        ]),
    ],
    tooltip=(
        "`off` = sample as it is; `on` = scale and offset the whole picture from inside the "
        "sampler's loop, which pushes texture and contrast in as the tiles refine."
    ),
)]

#: The picture and its enlargement, which both samplers take.
PICTURE_INPUTS = [
    io.Vae.Input("vae", tooltip="The VAE of the model, which encodes the picture and decodes the answer."),
    io.Image.Input(
        "image",
        tooltip="The picture or batch of pictures to refine, each sampled on its own.",
    ),
    io.Float.Input(
        "upscale_by", default=2.0, min=0.25, max=8.0, step=0.05,
        tooltip=(
            "Size of the answer against the picture. `1.0` = refine at the same size; `2.0` = "
            "twice as wide and tall. The upscale model runs first where one is wired."
        ),
    ),
]

#: The captioning sockets and settings both samplers take.
CAPTION_INPUTS = [
    io.Clip.Input(
        "clip", optional=True,
        tooltip=(
            "The text encoder from the loader the model came from. It encodes each tile's "
            "caption; with no captioner it is not used."
        ),
    ),
    io.Clip.Input(
        "captioner", optional=True,
        tooltip=(
            "A vision language model from Load CLIP that writes each tile's prompt, as "
            "`qwen3vl_8b_fp8_scaled.safetensors`. Unwired, every tile takes the positive prompt."
        ),
    ),
    io.String.Input(
        "prompt", multiline=True, default="", optional=True,
        tooltip=(
            "The prompt the picture was made from, or what it shows, as `a top-down view of a "
            "space station`. The captioner reads it to name what each tile holds; it is not "
            "sampled."
        ),
    ),
    io.String.Input(
        "style", default="", optional=True,
        tooltip="Words added to the end of every tile's caption, as `oil painting, sharp focus`.",
    ),
    io.UpscaleModel.Input(
        "upscale_model", optional=True,
        tooltip="An upscale model run on the picture before it is sampled, as `4x-UltraSharp`.",
    ),
    DICT.Input("affine_schedule", optional=True, tooltip=SCHEDULE_HINT),
    DICT.Input("affine_options", optional=True, tooltip=OPTIONS_HINT),
    io.String.Input(
        "system_prompt", multiline=True, default=tile_captions.DEFAULT_SYSTEM_PROMPT,
        optional=True, advanced=True,
        tooltip=(
            "The rules each tile's caption is written under. A line such as `Name every plant "
            "species you can identify.` adds a rule."
        ),
    ),
]

#: What both samplers answer.
OUTPUTS = [
    io.Image.Output(display_name="image", tooltip="The refined pictures."),
    io.Latent.Output(display_name="latent", tooltip="The refined latents, for a further pass."),
    io.String.Output(
        display_name="captions",
        tooltip="The overview and every tile's caption, one numbered line each. Empty with no captioner.",
    ),
]


def _affine_spec(affine: dict, affine_schedule, affine_options, total_steps=None):
    """The affine settings from the combo's value and its sockets, or None when it is off."""
    affine = dict(affine or {})
    if affine.get("affine") != "on":
        return None
    affine["affine_schedule"] = affine_schedule
    affine["affine_options"] = affine_options
    return build_spec(affine, total_steps=total_steps)


def _wrap(sampler, spec, name: str):
    """A sampler carrying the affine where one is set, with what the wrap reported."""
    if spec is None:
        return sampler, None
    patched, holder = patch_sampler(sampler, spec)
    warn_unsupported(holder, name)
    return patched, holder


def _affine_report(holder) -> dict | None:
    """What the affine did over a run, for the panel."""
    if holder is None:
        return None
    return {"applied": holder.get("applications", 0), "gain": holder.get("gain", 1.0)}


def _frames(image):
    """The pictures of a batch, one ``[1, H, W, C]`` at a time."""
    for index in range(int(image.shape[0])):
        yield index, image[index:index + 1]


def _answer(frames: list, latents: list, vae, name: str, holder) -> io.NodeOutput:
    """The node's outputs from the sampled latents, with the tile report published."""
    import torch

    tile_refine.release(vae)
    pictures = torch.cat([tile_refine.decode(vae, latent) for latent in latents], dim=0)
    tile_refine.publish(frames, name, _affine_report(holder), pictures[-1:])
    return io.NodeOutput(pictures, {"samples": torch.cat(latents, dim=0)},
                         tile_refine.captions_text(frames))


class TiledCaptionedKSampler(io.ComfyNode):
    """Refine a picture over overlapping tiles with KSampler's settings, each tile prompted by its own caption."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASTiledCaptionedKSampler",
            display_name=KSAMPLER_NAME,
            search_aliases=[
                "WASTiledCaptionedKSampler",
                KSAMPLER_NAME,
                "tiled sampler",
                "tiled upscale",
                "ultimate upscale",
                "tile refine",
                "tile captions",
                "regional prompt",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Upscale and refine a picture in overlapping tiles sampled together, so no seam "
                "shows and any sampler works. A vision model can caption each tile, so a tile "
                "is prompted only with what it holds. Affine can push texture in as it runs."
            ),
            inputs=[
                io.Model.Input("model", tooltip="The model to refine with."),
                io.Conditioning.Input(
                    "positive",
                    tooltip=(
                        "The prompt for the whole picture. With a captioner each tile takes its "
                        "own caption in place of this text, keeping its other settings."
                    ),
                ),
                io.Conditioning.Input("negative", tooltip="What every tile should avoid."),
                *PICTURE_INPUTS,
                io.Int.Input(
                    "seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True,
                    tooltip="Seeds the noise laid over the whole picture, `0` as good as any.",
                ),
                io.Int.Input(
                    "steps", default=20, min=1, max=10000,
                    tooltip="Steps over the whole schedule, as `20`, or `8` for a turbo model.",
                ),
                io.Float.Input(
                    "cfg", default=4.5, min=0.0, max=100.0, step=0.1, round=0.01,
                    tooltip="Guidance. `1.0` for a distilled model, `4.5` for most, `7.0` for SD1 and SDXL.",
                ),
                io.Combo.Input(
                    "sampler_name", options=sampler_names(), default="euler",
                    tooltip="The algorithm the steps are taken with, as `euler` or `dpmpp_2m`.",
                ),
                io.Combo.Input(
                    "scheduler", options=scheduler_names(), default="normal",
                    tooltip="How the noise level falls across the steps, as `normal`, `simple` or `beta`.",
                ),
                io.Float.Input(
                    "denoise", default=0.35, min=0.0, max=1.0, step=0.01,
                    tooltip=(
                        "How much of the picture is redrawn. `0.2` = sharpens and repairs; `0.35` "
                        "= adds detail; `0.6` = repaints, which captions keep on subject."
                    ),
                ),
                *TILING_INPUTS,
                *AFFINE_INPUTS_COMBO,
                *CAPTION_INPUTS,
            ],
            outputs=OUTPUTS,
        )

    @classmethod
    def execute(cls, model, positive, negative, vae, image, upscale_by, seed, steps, cfg,
                sampler_name, scheduler, denoise, tiling, affine, clip=None, captioner=None,
                prompt="", style="", upscale_model=None, affine_schedule=None, affine_options=None,
                system_prompt=tile_captions.DEFAULT_SYSTEM_PROMPT) -> io.NodeOutput:
        import comfy.model_management
        import comfy.sample
        import comfy.samplers
        import comfy.utils
        from ...modules.sampling import preview

        if not hasattr(model, "get_model_object"):
            raise ValueError(
                f"{KSAMPLER_NAME} needs a diffusion model in model, as Load Checkpoint answers one. "
                f"It was handed {type(model).__name__}"
            )
        spec = _affine_spec(affine, affine_schedule, affine_options, total_steps=int(steps))
        sampler, holder = _wrap(comfy.samplers.sampler_object(sampler_name), spec, str(sampler_name))
        sigmas = comfy.samplers.KSampler(
            model, steps=int(steps), device=model.load_device, sampler=sampler_name,
            scheduler=scheduler, denoise=float(denoise), model_options=model.model_options,
        ).sigmas

        frames, latents = [], []
        for index, picture in _frames(image):
            frame = tile_refine.prepare(picture, model, vae, upscale_by, upscale_model, tiling,
                                        clip, captioner, style, system_prompt, KSAMPLER_NAME, prompt)
            frames.append(frame)
            samples = comfy.sample.fix_empty_latent_channels(model, frame.latent)
            if sigmas.shape[-1] < 2:
                latents.append(samples)
                continue
            controls = {}
            tiled = tiles.tiled_model(model, frame.grid, frame.announce)
            tile_positive = tiles.tiled_conditioning(positive, frame.grid, frame.masks, frame.texts, controls)
            tile_negative = tiles.tiled_conditioning(negative, frame.grid, frame.masks, None, controls)
            noise = comfy.sample.prepare_noise(samples, int(seed), [index])
            tile_refine.release(model)
            callback = tile_refine.tracked(preview.prepare_callback(tiled, int(sigmas.shape[-1]) - 1),
                                           frame.announce, int(sigmas.shape[-1]) - 1)
            result = comfy.samplers.sample(
                tiled, noise, tile_positive, tile_negative, float(cfg), tiled.load_device, sampler,
                sigmas, model_options=tiled.model_options, latent_image=samples, callback=callback,
                disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=int(seed),
            )
            latents.append(result.to(comfy.model_management.intermediate_device()))
            frame.announce("done")
        return _answer(frames, latents, vae, str(sampler_name), holder)


class TiledCaptionedCustomSampler(io.ComfyNode):
    """Refine a picture over overlapping tiles from a guider and a sigma schedule, each tile prompted by its own caption."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASTiledCaptionedCustomSampler",
            display_name=CUSTOM_NAME,
            search_aliases=[
                "WASTiledCaptionedCustomSampler",
                CUSTOM_NAME,
                "tiled custom sampler",
                "tiled upscale",
                "ultimate upscale",
                "tile refine",
                "tile captions",
                "guider",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Upscale and refine a picture in overlapping tiles from a guider, sampler and "
                "sigmas, every tile sampled together so no seam shows. A vision model can "
                "caption each tile in place of the guider's positive prompt."
            ),
            inputs=[
                io.Noise.Input("noise", tooltip="Where the noise laid over the whole picture comes from."),
                io.Guider.Input(
                    "guider",
                    tooltip=(
                        "What steers the denoising, as CFGGuider. With a captioner each tile's "
                        "caption replaces the text of its positive prompt."
                    ),
                ),
                io.Sampler.Input("sampler", tooltip=SAMPLER_HINT),
                io.Sigmas.Input(
                    "sigmas",
                    tooltip=(
                        "The noise levels to step through, as BasicScheduler gives them. Its "
                        "denoise sets how much of the picture is redrawn."
                    ),
                ),
                *PICTURE_INPUTS,
                *TILING_INPUTS,
                *AFFINE_INPUTS_COMBO,
                *CAPTION_INPUTS,
            ],
            outputs=OUTPUTS,
        )

    @classmethod
    def execute(cls, noise, guider, sampler, sigmas, vae, image, upscale_by, tiling, affine,
                clip=None, captioner=None, prompt="", style="", upscale_model=None, affine_schedule=None,
                affine_options=None, system_prompt=tile_captions.DEFAULT_SYSTEM_PROMPT) -> io.NodeOutput:
        import comfy.model_management
        import comfy.sample
        import comfy.utils
        from ...modules.sampling import preview

        if not hasattr(guider, "model_patcher") or not hasattr(guider, "original_conds"):
            raise ValueError(
                f"{CUSTOM_NAME} needs a guider in guider, as CFGGuider, BasicGuider and DualCFGGuider "
                f"answer one. It was handed {type(guider).__name__}"
            )
        model = guider.model_patcher
        name = sampler_label(sampler)
        sampler, holder = _wrap(sampler, _affine_spec(affine, affine_schedule, affine_options), name)
        conds = dict(guider.original_conds)
        text_key = "positive" if "positive" in conds else next((key for key in conds if key != "negative"), None)
        if captioner is not None and text_key is None:
            raise ValueError(
                f"{CUSTOM_NAME} has a captioner, and this guider carries no positive prompt for the "
                "captions to replace. Use CFGGuider or BasicGuider, or disconnect captioner"
            )

        frames, latents = [], []
        for index, picture in _frames(image):
            frame = tile_refine.prepare(picture, model, vae, upscale_by, upscale_model, tiling,
                                        clip, captioner, style, system_prompt, CUSTOM_NAME, prompt)
            frames.append(frame)
            samples = comfy.sample.fix_empty_latent_channels(model, frame.latent)
            if sigmas.shape[-1] < 2:
                latents.append(samples)
                continue
            controls = {}
            tiled_guider = copy.copy(guider)
            tiled_guider.model_patcher = tiles.tiled_model(model, frame.grid, frame.announce)
            tiled_guider.model_options = tiled_guider.model_patcher.model_options
            tiled_guider.original_conds = {}
            for key, entries in conds.items():
                conditioning = [[entry.get("cross_attn"),
                                 {k: v for k, v in entry.items() if k not in ("cross_attn", "uuid")}]
                                for entry in entries]
                texts = frame.texts if key == text_key else None
                tiled_guider.inner_set_conds(
                    {key: tiles.tiled_conditioning(conditioning, frame.grid, frame.masks, texts, controls)})
            latent = {"samples": samples, "batch_index": [index]}
            tile_refine.release(model)
            callback = tile_refine.tracked(
                preview.prepare_callback(tiled_guider.model_patcher, int(sigmas.shape[-1]) - 1),
                frame.announce, int(sigmas.shape[-1]) - 1)
            result = tiled_guider.sample(
                noise.generate_noise(latent), samples, sampler, sigmas, callback=callback,
                disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=noise.seed,
            )
            latents.append(result.to(comfy.model_management.intermediate_device()))
            frame.announce("done")
        return _answer(frames, latents, vae, name, holder)
