"""Iterative sample-upscale-resample sampling."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ...modules import log
from ...modules.compat import operations
from ...modules.convert.tensors import pil2tensor, tensor2pil
from ...modules.image.blend_modes import ceiling_of

logger = log.get_logger("sampling.ksampler_cycle")

#: Resampling filter names to their ``PIL.Image.Resampling`` values.
RESAMPLE_FILTERS = {"nearest": 0, "bilinear": 2, "bicubic": 3, "lanczos": 1}

#: Latent-space upscale methods, prefixed with the option that turns the pixel-space path
#: on instead. Everything after ``disable`` is passed straight to core's LatentUpscaleBy.
LATENT_UPSCALE_METHODS = ["disable", "nearest-exact", "bilinear", "area", "bicubic", "bislerp"]


def sampler_names() -> list[str]:
    """The sampler names this ComfyUI offers."""
    # Read when the schema is built. The list grows with every ComfyUI release, so a copy
    # frozen into this source would be wrong on the next one.
    import comfy.samplers

    return comfy.samplers.KSampler.SAMPLERS


def scheduler_names() -> list[str]:
    """The scheduler names this ComfyUI offers."""
    import comfy.samplers

    return comfy.samplers.KSampler.SCHEDULERS


def resampling_filter(name: str):
    """``PIL.Image.Resampling`` member for one of :data:`RESAMPLE_FILTERS`."""
    from PIL import Image

    return Image.Resampling(RESAMPLE_FILTERS[name])


def gaussian_kernel(sigma: float) -> torch.Tensor:
    """One axis of a Gaussian blur, truncated at four standard deviations.

    Args:
        sigma: Standard deviation in pixels.

    Returns:
        A ``(2 * int(4 * sigma + 0.5) + 1,)`` float64 tensor summing to 1.
    """
    span = int(4.0 * sigma + 0.5)
    offsets = torch.arange(-span, span + 1, dtype=torch.float64)
    weights = torch.exp(-0.5 * offsets ** 2 / (sigma * sigma))
    return weights / weights.sum()


def mirrored_index(length: int, pad: int, device) -> torch.Tensor:
    """Read positions for one axis extended by ``pad`` samples mirrored at each end.

    Args:
        length: Length of the axis being extended.
        pad: Samples added at each end.
        device: Device the indices are built on.

    Returns:
        A ``(length + 2 * pad,)`` long tensor, the edge sample repeated as the mirror's
        first step.
    """
    period = 2 * length
    index = torch.arange(-pad, length + pad, device=device) % period
    return torch.where(index >= length, period - 1 - index, index)


def gaussian_blur(images_bchw: torch.Tensor, sigma: float) -> torch.Tensor:
    """Blur a batch of planes with a separable Gaussian, mirroring at the edges.

    Args:
        images_bchw: ``(batch, channels, height, width)`` float tensor.
        sigma: Standard deviation in pixels.

    Returns:
        A tensor of the same shape and dtype.
    """
    import torch.nn.functional as functional

    device = images_bchw.device
    channels = images_bchw.shape[1]
    weights = gaussian_kernel(sigma).to(images_bchw)
    pad = weights.shape[0] // 2
    rows = weights.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    columns = weights.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)
    spread = images_bchw.index_select(3, mirrored_index(images_bchw.shape[3], pad, device))
    blurred = functional.conv2d(spread, rows, groups=channels)
    spread = blurred.index_select(2, mirrored_index(blurred.shape[2], pad, device))
    return functional.conv2d(spread, columns, groups=channels)


def unsharp_filter(images: torch.Tensor, radius: float = 2, amount: float = 1.0) -> torch.Tensor:
    """Sharpen an image batch with an unsharp mask.

    Args:
        images: ``(batch, height, width, channels)`` float tensor.
        radius: Standard deviation of the blur the mask is taken against, in pixels. Zero
            and below leave the batch alone.
        amount: How far the detail the mask found is amplified.

    Returns:
        A tensor of the same shape, held inside 0 to the batch's own peak.
    """
    if radius <= 0:
        return images
    planes = images.permute(0, 3, 1, 2)
    detail = planes - gaussian_blur(planes, float(radius))
    sharpened = (planes + detail * amount).clamp(0.0, ceiling_of(images))
    return sharpened.permute(0, 2, 3, 1)


def vae_encode_crop_pixels(pixels):
    """Centre-crop an image batch to a multiple of eight in both spatial axes."""
    x = (pixels.shape[1] // 8) * 8
    y = (pixels.shape[2] // 8) * 8
    if pixels.shape[1] != x or pixels.shape[2] != y:
        x_offset = (pixels.shape[1] % 8) // 2
        y_offset = (pixels.shape[2] % 8) // 2
        pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
    return pixels


def rescale_tensor(tensor, factor: float, resample: str):
    """Scale one image by a factor, through an 8x supersample.

    The image is first resized to eight times the target and then down to it.

    Args:
        tensor: A single image tensor.
        factor: Multiplier applied to both axes.
        resample: One of :data:`RESAMPLE_FILTERS`.

    Returns:
        A one-image batch tensor.
    """
    image = tensor2pil(tensor)
    new_width, new_height = int(image.size[0] * factor), int(image.size[1] * factor)
    filter_ = resampling_filter(resample)
    # The same supersample path the Image Rescale node takes.
    image = image.resize((new_width * 8, new_height * 8), resample=filter_)
    return pil2tensor(image.resize((new_width, new_height), resample=filter_))


def additive_strength(cycle: int, mode: str, strength: float, scaling: bool, cutoff: float) -> float:
    """Strength the additive conditioning is averaged in at on one cycle.

    Args:
        cycle: Zero-based cycle index.
        mode: ``increment`` doubles the strength each cycle, anything else halves it.
        strength: Strength on the first cycle.
        scaling: Whether the strength changes at all across cycles.
        cutoff: Bound the scaled strength is held to, an upper bound while incrementing,
            a lower bound while decrementing.

    Returns:
        The strength to use on this cycle.
    """
    if mode == "increment":
        scaled = (round(strength * (2 ** (cycle - 1)), 2) if cycle > 0 else strength) if scaling else strength
        return cutoff if scaled > cutoff else scaled
    scaled = (round(strength / (2 ** (cycle - 1)), 2) if cycle > 0 else strength) if scaling else strength
    return cutoff if scaled < cutoff else scaled


class KSamplerCycle(io.ComfyNode):
    """Sample a latent repeatedly, enlarging it between passes."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="KSampler Cycle",
            display_name="KSampler Cycle",
            search_aliases=[
                "KSampler Cycle",
                "iterative upscale",
                "hires fix",
                "sampler",
                "refine",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Sample a latent over several cycles, enlarging it between passes by an "
                "even share of upscale_factor. Scaling runs in latent space, or through a "
                "VAE round trip with an optional upscale model and unsharp sharpening."
            ),
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip=(
                        "The diffusion model every cycle samples with, unless "
                        "secondary_model takes over partway through."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    tooltip=(
                        "Seed for the noise. Every cycle uses this same seed, so the whole run "
                        "is repeatable; change it for a different image. Any whole number; `0` "
                        "is as good a seed as any."
                    ),
                ),
                io.Int.Input(
                    "steps",
                    default=20,
                    min=1,
                    max=10000,
                    tooltip=(
                        "Sampling steps on the first cycle. More steps take longer and "
                        "resolve more detail, with little to gain past about 30 for most "
                        "models. Turning steps_scaling on changes this figure on later "
                        "cycles."
                    ),
                ),
                io.Float.Input(
                    "cfg",
                    default=8.0,
                    min=0.0,
                    max=100.0,
                    tooltip=(
                        "How closely the image is held to the prompt. Around 7-8 suits most "
                        "models; lower is looser and softer, much higher tends to burn "
                        "contrast and flatten detail."
                    ),
                ),
                io.Combo.Input(
                    "sampler_name",
                    options=sampler_names(),
                    tooltip=(
                        "The sampling algorithm. 'euler' is the plain, predictable choice; "
                        "the 'ancestral' and 'sde' variants add fresh noise as they go and "
                        "keep changing the image at high step counts; the 'dpmpp' family "
                        "converges in fewer steps. The list is whatever this ComfyUI offers."
                    ),
                ),
                io.Combo.Input(
                    "scheduler",
                    options=scheduler_names(),
                    tooltip=(
                        "How the noise level is stepped down over the run. 'normal' and "
                        "'karras' are the usual choices, karras spending more steps at low "
                        "noise where fine detail is decided. The list is whatever this "
                        "ComfyUI offers."
                    ),
                ),
                io.Conditioning.Input(
                    "positive",
                    tooltip="Encoded prompt describing what the image should contain.",
                ),
                io.Conditioning.Input(
                    "negative",
                    tooltip="Encoded prompt describing what to keep out of the image.",
                ),
                io.Latent.Input(
                    "latent_image",
                    tooltip=(
                        "The latent the first cycle works on: an empty one to generate from "
                        "scratch, or an encoded image to work up from. Its size sets where "
                        "the enlargement starts."
                    ),
                ),
                io.Combo.Input(
                    "tiled_vae",
                    options=["disable", "enable"],
                    tooltip=(
                        "`enable` converts between latent and pixels a tile at a time, which "
                        "needs far less VRAM at large sizes and can leave faint seams. Only "
                        "matters when latent_upscale is `disable`, since that is the only "
                        "path that goes through pixels."
                    ),
                ),
                io.Combo.Input(
                    "latent_upscale",
                    options=LATENT_UPSCALE_METHODS,
                    tooltip=(
                        "`disable` = through pixels, slower and sharper, using vae, "
                        "upscale_model, processor_model and scale_sampling. Any other entry stays "
                        "in latent space, much faster, ignoring all four: `nearest-exact` = "
                        "blockiest; `bilinear`, `bicubic` = smoother; `area` = averages; "
                        "`bislerp` = a blend for latents."
                    ),
                ),
                io.Float.Input(
                    "upscale_factor",
                    default=2.0,
                    min=0.1,
                    max=8.0,
                    step=0.1,
                    tooltip=(
                        "Total enlargement across the whole run, not per cycle: 2.0 means the "
                        "result is twice the size it started at, and the cycles share that "
                        "growth evenly between them."
                    ),
                ),
                io.Int.Input(
                    "upscale_cycles",
                    default=2,
                    min=2,
                    max=12,
                    step=1,
                    tooltip=(
                        "How many sample passes to run. Enlargement happens between passes, "
                        "so 2 grows once and 4 grows three times in smaller jumps, which is "
                        "gentler but slower. Capped at steps, since a pass needs at least one "
                        "step."
                    ),
                ),
                io.Float.Input(
                    "starting_denoise",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How much of the first pass is redrawn. 1.0 ignores latent_image's "
                        "content and generates from noise; around 0.5 keeps its composition "
                        "and changes the detail; 0.0 changes nothing."
                    ),
                ),
                io.Float.Input(
                    "cycle_denoise",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How much every pass after the first redraws. This is what keeps the "
                        "enlarged image recognisable: 0.5 refines it, and much above that "
                        "starts inventing new content at each size."
                    ),
                ),
                io.Combo.Input(
                    "scale_denoise",
                    options=["enable", "disable"],
                    tooltip=(
                        "`enable` halves cycle_denoise again on each pass after the second, "
                        "so later passes only polish; denoise_cutoff sets how low it may go. "
                        "`disable` uses cycle_denoise unchanged on every pass."
                    ),
                ),
                io.Combo.Input(
                    "scale_sampling",
                    options=["bilinear", "bicubic", "nearest", "lanczos"],
                    tooltip=(
                        "Which filter resizes the decoded picture on the pixel-space route. "
                        "`lanczos` and `bicubic` are the sharpest, `bilinear` softer, "
                        "`nearest` blocky. Ignored unless latent_upscale is `disable`."
                    ),
                ),
                io.Vae.Input(
                    "vae",
                    tooltip=(
                        "The VAE used to decode to pixels and encode back between cycles. "
                        "Required even when latent_upscale keeps the work in latent space and "
                        "nothing is decoded."
                    ),
                ),
                io.Model.Input(
                    "secondary_model",
                    optional=True,
                    tooltip=(
                        "A second diffusion model to hand the later cycles to, so one model "
                        "lays out the image and another finishes it. Disconnected, one model "
                        "does the whole run."
                    ),
                ),
                io.Int.Input(
                    "secondary_start_cycle",
                    default=2,
                    min=2,
                    max=16,
                    step=1,
                    optional=True,
                    tooltip=(
                        "Which pass secondary_model takes over on, counting from 1, so 2 hands "
                        "over straight after the first. That pass also uses cycle_denoise "
                        "rather than any scaled-down value."
                    ),
                ),
                io.UpscaleModel.Input(
                    "upscale_model",
                    optional=True,
                    tooltip=(
                        "An upscale model such as ESRGAN to do the enlarging, which recovers "
                        "far more detail than a plain resize. Its result is fitted to the "
                        "target size, rounded to a multiple of 32. Disconnected, the picture "
                        "is simply resampled. Ignored unless latent_upscale is `disable`."
                    ),
                ),
                io.UpscaleModel.Input(
                    "processor_model",
                    optional=True,
                    tooltip=(
                        "An upscale model run before the enlargement and shrunk straight back "
                        "to the size it started at, so it cleans up artefacts and restores "
                        "detail without changing the size. Ignored unless latent_upscale is "
                        "`disable`."
                    ),
                ),
                io.Conditioning.Input(
                    "pos_additive",
                    optional=True,
                    tooltip=(
                        "A second positive prompt mixed into the first a little more, or a "
                        "little less, on every cycle, a way to steer the image somewhere new "
                        "as it grows. Disconnected, the positive prompt stays as it is."
                    ),
                ),
                io.Conditioning.Input(
                    "neg_additive",
                    optional=True,
                    tooltip=(
                        "A second negative prompt mixed into the first a little more, or a "
                        "little less, on every cycle. Disconnected, the negative prompt stays "
                        "as it is."
                    ),
                ),
                io.Combo.Input(
                    "pos_add_mode",
                    options=["increment", "decrement"],
                    optional=True,
                    tooltip=(
                        "Which way pos_add_strength moves between cycles: `increment` doubles "
                        "it each pass, so pos_additive takes over gradually; `decrement` "
                        "halves it, so its influence fades out. Only used when "
                        "pos_add_strength_scaling is enabled."
                    ),
                ),
                io.Float.Input(
                    "pos_add_strength",
                    default=0.25,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    optional=True,
                    tooltip=(
                        "How much of pos_additive is mixed in on the first cycle. 0.25 is a "
                        "quarter of the way towards it, 1.0 replaces the positive prompt "
                        "outright."
                    ),
                ),
                io.Combo.Input(
                    "pos_add_strength_scaling",
                    options=["enable", "disable"],
                    optional=True,
                    tooltip=(
                        "`enable` lets pos_add_mode change the strength from cycle to cycle. "
                        "`disable` holds pos_add_strength steady for the whole run."
                    ),
                ),
                io.Float.Input(
                    "pos_add_strength_cutoff",
                    default=2.0,
                    min=0.01,
                    max=10.0,
                    step=0.01,
                    optional=True,
                    tooltip=(
                        "The limit the scaled strength is held to: a ceiling while "
                        "incrementing, a floor while decrementing. At the default of 2.0 in "
                        "increment mode the strength is effectively unbounded, since 1.0 "
                        "already means full replacement."
                    ),
                ),
                io.Combo.Input(
                    "neg_add_mode",
                    options=["increment", "decrement"],
                    optional=True,
                    tooltip=(
                        "Which way neg_add_strength moves between cycles: `increment` doubles "
                        "it each pass, `decrement` halves it. Only used when "
                        "neg_add_strength_scaling is enabled."
                    ),
                ),
                io.Float.Input(
                    "neg_add_strength",
                    default=0.25,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    optional=True,
                    tooltip=(
                        "How much of neg_additive is mixed in on the first cycle. 0.25 is a "
                        "quarter of the way towards it, 1.0 replaces the negative prompt "
                        "outright."
                    ),
                ),
                io.Combo.Input(
                    "neg_add_strength_scaling",
                    options=["enable", "disable"],
                    optional=True,
                    tooltip=(
                        "`enable` lets neg_add_mode change the strength from cycle to cycle. "
                        "`disable` holds neg_add_strength steady for the whole run."
                    ),
                ),
                io.Float.Input(
                    "neg_add_strength_cutoff",
                    default=2.0,
                    min=0.01,
                    max=10.0,
                    step=0.01,
                    optional=True,
                    tooltip=(
                        "The limit the scaled strength is held to: a ceiling while "
                        "incrementing, a floor while decrementing."
                    ),
                ),
                io.Float.Input(
                    "sharpen_strength",
                    default=0.0,
                    min=0.0,
                    max=10.0,
                    step=0.01,
                    optional=True,
                    tooltip=(
                        "How hard to sharpen the enlarged picture before it is encoded again. "
                        "0.0 skips sharpening entirely; around 1.0 counteracts the softness "
                        "of a resize; far above that leaves halos. Ignored unless "
                        "latent_upscale is `disable`."
                    ),
                ),
                io.Int.Input(
                    "sharpen_radius",
                    default=2,
                    min=1,
                    max=12,
                    step=1,
                    optional=True,
                    tooltip=(
                        "How wide the sharpening reaches, in pixels. Small values pick out "
                        "fine texture, large ones lift broad edges and coarsen the picture. "
                        "Only used when sharpen_strength is above 0."
                    ),
                ),
                io.Combo.Input(
                    "steps_scaling",
                    options=["enable", "disable"],
                    optional=True,
                    tooltip=(
                        "`enable` changes the step count on every pass after the first, by "
                        "steps_scaling_value and in the direction steps_control names. "
                        "`disable` keeps steps the same throughout."
                    ),
                ),
                io.Combo.Input(
                    "steps_control",
                    options=["decrement", "increment"],
                    optional=True,
                    tooltip=(
                        "Which way the step count moves. `decrement` spends fewer steps on "
                        "each larger pass, which is the cheaper choice since low-denoise "
                        "passes need fewer; `increment` spends more."
                    ),
                ),
                io.Int.Input(
                    "steps_scaling_value",
                    default=10,
                    min=1,
                    max=20,
                    step=1,
                    optional=True,
                    tooltip=(
                        "How many steps are added or taken away on each pass after the first. "
                        "Only used when steps_scaling is enabled."
                    ),
                ),
                io.Int.Input(
                    "steps_cutoff",
                    default=20,
                    min=4,
                    max=1000,
                    step=1,
                    optional=True,
                    tooltip=(
                        "The step count the scaling is not allowed past: a ceiling while "
                        "incrementing, a floor while decrementing. At the default of 20, with "
                        "steps also 20 and steps_control on `decrement`, the count never "
                        "moves."
                    ),
                ),
                io.Float.Input(
                    "denoise_cutoff",
                    default=0.25,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    optional=True,
                    tooltip=(
                        "The lowest denoise the halving under scale_denoise may reach, so "
                        "later passes still do some work. Ignored when scale_denoise is "
                        "disabled."
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output(
                    display_name="latent(s)",
                    tooltip=(
                        "The latent after the final pass, at the full enlarged size. Decode it "
                        "with a VAE Decode to see the picture."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        tiled_vae,
        latent_upscale,
        upscale_factor,
        upscale_cycles,
        starting_denoise,
        cycle_denoise,
        scale_denoise,
        scale_sampling,
        vae,
        secondary_model=None,
        secondary_start_cycle=None,
        upscale_model=None,
        processor_model=None,
        pos_additive=None,
        neg_additive=None,
        pos_add_mode=None,
        pos_add_strength=None,
        pos_add_strength_scaling=None,
        pos_add_strength_cutoff=None,
        neg_add_mode=None,
        neg_add_strength=None,
        neg_add_strength_scaling=None,
        neg_add_strength_cutoff=None,
        sharpen_strength=0,
        sharpen_radius=2,
        steps_scaling=None,
        steps_control=None,
        steps_scaling_value=None,
        steps_cutoff=None,
        denoise_cutoff=0.25,
    ) -> io.NodeOutput:
        import comfy.utils

        # ComfyUI's root nodes.py. This pack's own nodes/ package is reachable only as a
        # submodule of the pack, so the bare name resolves to ComfyUI's module.
        from nodes import ConditioningAverage, LatentUpscaleBy, common_ksampler

        division_factor = upscale_cycles if steps >= upscale_cycles else steps
        # One cycle never upscales, and the even share of upscale_factor is undefined for
        # it, so the root is only taken where there is a gap to spread across.
        current_upscale_factor = (
            upscale_factor ** (1 / (division_factor - 1)) if division_factor > 1 else upscale_factor
        )
        tiled_vae = tiled_vae == "enable"
        scale_denoise = scale_denoise == "enable"
        pos_add_strength_scaling = pos_add_strength_scaling == "enable"
        neg_add_strength_scaling = neg_add_strength_scaling == "enable"
        steps_scaling = steps_scaling == "enable"
        run_model = model
        secondary_switched = False
        latent_image_result = latent_image
        progress = comfy.utils.ProgressBar(division_factor)

        for cycle in range(division_factor):
            logger.info("Cycle pass %d/%d", cycle + 1, division_factor)

            if scale_denoise:
                denoise = (
                    round(cycle_denoise * (2 ** (-(cycle - 1))), 2)
                    if cycle > 0
                    else round(starting_denoise, 2)
                )
            else:
                denoise = round(cycle_denoise if cycle > 0 else starting_denoise, 2)

            if denoise < denoise_cutoff and scale_denoise:
                denoise = denoise_cutoff

            if (
                secondary_model
                and secondary_start_cycle is not None
                and cycle >= (secondary_start_cycle - 1)
                and not secondary_switched
            ):
                run_model = secondary_model
                denoise = cycle_denoise
                secondary_switched = True

            if steps_scaling and cycle > 0:
                if steps_control == "increment":
                    steps = min(steps + steps_scaling_value, steps_cutoff)
                else:
                    steps = max(steps - steps_scaling_value, steps_cutoff)

            logger.info("Steps: %s, denoise: %s", steps, denoise)

            if pos_additive:
                pos_strength = additive_strength(
                    cycle,
                    pos_add_mode,
                    pos_add_strength,
                    pos_add_strength_scaling,
                    pos_add_strength_cutoff,
                )
                positive = ConditioningAverage().addWeighted(pos_additive, positive, pos_strength)[0]
                logger.info("Positive additive strength: %s", pos_strength)

            if neg_additive:
                neg_strength = additive_strength(
                    cycle,
                    neg_add_mode,
                    neg_add_strength,
                    neg_add_strength_scaling,
                    neg_add_strength_cutoff,
                )
                negative = ConditioningAverage().addWeighted(neg_additive, negative, neg_strength)[0]
                logger.info("Negative additive strength: %s", neg_strength)

            if cycle != 0:
                latent_image = latent_image_result

            samples = common_ksampler(
                run_model,
                seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                latent_image,
                denoise=denoise,
            )

            if cycle >= division_factor - 1:
                latent_image_result = samples[0]
                progress.update(1)
                continue

            if latent_upscale != "disable":
                latent_image_result = LatentUpscaleBy().upscale(
                    samples[0], latent_upscale, current_upscale_factor
                )[0]
                progress.update(1)
                continue

            latent_image_result = cls.upscale_in_pixel_space(
                samples[0],
                vae=vae,
                tiled_vae=tiled_vae,
                upscale_model=upscale_model,
                processor_model=processor_model,
                factor=current_upscale_factor,
                resample=scale_sampling,
                sharpen_strength=sharpen_strength,
                sharpen_radius=sharpen_radius,
            )
            progress.update(1)

        return io.NodeOutput(latent_image_result)

    @classmethod
    def upscale_in_pixel_space(
        cls,
        latent,
        vae,
        tiled_vae: bool,
        upscale_model,
        processor_model,
        factor: float,
        resample: str,
        sharpen_strength: float,
        sharpen_radius: int,
    ):
        """Decode a latent, enlarge it in pixel space, and encode it again.

        Args:
            latent: LATENT dict to decode.
            vae: VAE used for both halves of the round trip.
            tiled_vae: Whether to decode and encode tiled.
            upscale_model: UPSCALE_MODEL doing the enlargement, or None.
            processor_model: UPSCALE_MODEL run at the original size first, or None.
            factor: Multiplier applied to both axes.
            resample: Filter used for every PIL resize.
            sharpen_strength: Unsharp mask amount. Zero skips sharpening.
            sharpen_radius: Unsharp mask radius.

        Returns:
            A LATENT dict holding the re-encoded image.
        """
        tensors = vae.decode_tiled(latent["samples"]) if tiled_vae else vae.decode(latent["samples"])

        if processor_model:
            original_size = tensor2pil(tensors[0]).size
            processed = []
            for tensor in operations.upscale_with_model(processor_model, tensors):
                pil = tensor2pil(tensor)
                if pil.size != original_size:
                    pil = pil.resize(original_size, resampling_filter(resample))
                frame = pil2tensor(pil)
                if sharpen_strength != 0.0:
                    frame = unsharp_filter(frame, sharpen_radius, sharpen_strength)
                processed.append(frame)
            processed = torch.cat(processed, dim=0)

        if upscale_model:
            # The only consumer of the processed frames, so a processor model with no
            # upscale model does its work and has it discarded.
            if processor_model:
                tensors = processed
            original_size = tensor2pil(tensors[0]).size
            new_width = int(round(round(original_size[0] * factor) / 32) * 32)
            new_height = int(round(round(original_size[1] * factor) / 32) * 32)
            tensor_images = []
            for tensor in operations.upscale_with_model(upscale_model, tensors):
                pil = tensor2pil(tensor).resize((new_width, new_height), resampling_filter(resample))
                frame = pil2tensor(pil)
                if sharpen_strength != 0.0:
                    frame = unsharp_filter(frame, sharpen_radius, sharpen_strength)
                tensor_images.append(frame)
            tensor_images = torch.cat(tensor_images, dim=0)
        else:
            tensor_images = []
            for tensor in tensors:
                scaled = rescale_tensor(tensor.unsqueeze(0), factor, resample)
                if sharpen_strength > 0.0:
                    scaled = unsharp_filter(scaled, sharpen_radius, sharpen_strength)
                tensor_images.append(scaled)
            tensor_images = torch.cat(tensor_images, dim=0)

        pixels = vae_encode_crop_pixels(tensor_images)[:, :, :, :3]
        return {"samples": vae.encode_tiled(pixels) if tiled_vae else vae.encode(pixels)}
