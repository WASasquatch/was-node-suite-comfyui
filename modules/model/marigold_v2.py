"""Marigold v2 dense prediction, taken in one sampler step.

The transformer, its adapter, the decoder and the prompt embedding are all the caller's.
This holds the step that turns a picture into a map and the reading of that map.
"""

from __future__ import annotations

__all__ = [
    "ALREADY_APPLIED",
    "AUTOMATIC",
    "MODALITIES",
    "MODEL_NAME",
    "MULTIPLE",
    "TIMESTEP",
    "adapted",
    "adapters",
    "albedo",
    "conditioning",
    "depth",
    "embeddings",
    "normals",
    "published",
    "published_adapter",
    "predict",
]

import torch

from .. import log

logger = log.get_logger("model.marigold_v2")

#: The maps this reads, against the preprocessor that answers each one.
MODALITIES = {
    "depth_map": "depth",
    "normal_map": "normals",
    "albedo": "albedo",
}

#: What the model menu calls this route.
MODEL_NAME = "Marigold v2"

#: Where on the flow the step is taken.
TIMESTEP = 499.0 / 1000.0

#: Pixels a latent sample covers, times the pair of samples a token carries.
MULTIPLE = 16

#: Narrowest range that is stretched.
EPSILON = 1e-6

#: Key a saved prompt embedding is stored under.
_TENSOR = "conditioning"

#: The name a map's prompt embedding is published under.
_PUBLISHED = "marigold_v2_{}_conditioning"

#: The prefix a map's adapter is published under.
_ADAPTER = "marigold_v2_{}"

#: The menu entry naming the file the map publishes.
AUTOMATIC = "auto"

#: The adapter menu entry naming a model that carries one already.
ALREADY_APPLIED = "already on the model"

#: What a sigma is multiplied by to read it as a timestep.
_MULTIPLIER = 1.0


def embeddings() -> list[str]:
    """Every file ComfyUI's embeddings folder offers.

    Returns:
        :data:`AUTOMATIC` first, then the file names. Just :data:`AUTOMATIC` outside ComfyUI.
    """
    try:
        import folder_paths

        return [AUTOMATIC, *folder_paths.get_filename_list("embeddings")]
    except Exception as error:
        logger.debug("the embeddings folder could not be read: %s", error)
        return [AUTOMATIC]


def adapters() -> list[str]:
    """Every file ComfyUI's loras folder offers.

    Returns:
        :data:`AUTOMATIC` and :data:`ALREADY_APPLIED` first, then the file names.
    """
    try:
        import folder_paths

        return [AUTOMATIC, ALREADY_APPLIED, *folder_paths.get_filename_list("loras")]
    except Exception as error:
        logger.debug("the loras folder could not be read: %s", error)
        return [AUTOMATIC, ALREADY_APPLIED]


def published_adapter(modality: str) -> str:
    """The adapter file one map is published under, where the folder holds one.

    Args:
        modality: One of the values of :data:`MODALITIES`.

    Returns:
        The file name, or an empty string where nothing there matches.
    """
    wanted = _ADAPTER.format(modality)
    for name in adapters():
        stem = name.rsplit(".", 1)[0].replace("\\", "/").rsplit("/", 1)[-1]
        if stem == wanted or stem.startswith(wanted + "_"):
            return name
    return ""


def adapted(model, name: str, modality: str):
    """The model carrying one map's adapter.

    Args:
        model: The ``MODEL`` wired in.
        name: A file the loras folder offers, :data:`AUTOMATIC`, or
            :data:`ALREADY_APPLIED` to use the model as it arrived.
        modality: The map being read.

    Returns:
        A patched clone, or the model itself where it already carries its adapter.

    Raises:
        ValueError: Nothing is picked and nothing matches, or the name names nothing there.
    """
    import comfy.sd
    import comfy.utils
    import folder_paths

    picked = (name or "").strip()
    if picked == ALREADY_APPLIED:
        return model
    chosen = published_adapter(modality) if picked in ("", AUTOMATIC) else picked
    if not chosen:
        raise ValueError(
            f"{MODEL_NAME} found no adapter for {modality}.\n"
            f"  Put {_ADAPTER.format(modality)}*.safetensors in ComfyUI/models/loras and "
            f"reload the page,\n"
            f"  pick one in the adapter input, or choose '{ALREADY_APPLIED}' where the "
            f"model already carries it."
        )
    path = folder_paths.get_full_path("loras", chosen)
    if not path:
        raise ValueError(
            f"ComfyUI's loras folder holds no {chosen!r}. Reload the page after putting "
            f"it there."
        )
    patched, _ = comfy.sd.load_lora_for_models(
        model, None, comfy.utils.load_torch_file(path, safe_load=True), 1.0, 0.0
    )
    logger.debug("%s applied %s for %s", MODEL_NAME, chosen, modality)
    return patched


def published(modality: str) -> str:
    """The embedding file one map is published under, where the folder holds it.

    Args:
        modality: One of the values of :data:`MODALITIES`.

    Returns:
        The file name, or an empty string where nothing there matches.
    """
    wanted = _PUBLISHED.format(modality)
    for name in embeddings():
        stem = name.rsplit(".", 1)[0].replace("\\", "/").rsplit("/", 1)[-1]
        if stem == wanted:
            return name
    return ""


def conditioning(name: str, modality: str = ""):
    """A saved prompt embedding from ComfyUI's embeddings folder.

    Args:
        name: A file name the folder offers, or :data:`AUTOMATIC` for the map's own.
        modality: The map being read, used only when no name is given.

    Returns:
        A ``(1, tokens, width)`` embedding.

    Raises:
        ValueError: Nothing is picked and nothing matches, the name names nothing there, or
            the file holds no embedding.
    """
    import comfy.utils
    import folder_paths

    picked = (name or "").strip()
    chosen = published(modality) if picked in ("", AUTOMATIC) else picked
    if not chosen:
        raise ValueError(
            f"{MODEL_NAME} found no prompt embedding for {modality or 'this map'}.\n"
            f"  Put {_PUBLISHED.format(modality or 'depth')}.safetensors in "
            f"ComfyUI/models/embeddings and reload the page,\n"
            f"  or pick one in the conditioning input."
        )
    path = folder_paths.get_full_path("embeddings", chosen)
    if not path:
        raise ValueError(
            f"ComfyUI's embeddings folder holds no {chosen!r}. Reload the page after "
            f"putting it there."
        )
    held = comfy.utils.load_torch_file(path, safe_load=True)
    if _TENSOR not in held:
        raise ValueError(
            f"{chosen!r} holds no {_TENSOR!r} tensor, so it is not a prompt embedding. "
            f"Pick one of the marigold_v2 conditioning files."
        )
    return held[_TENSOR]


def _velocity(model):
    """A clone of the model whose one step runs from the picture toward the map.

    Args:
        model: The ``MODEL`` wired in.

    Returns:
        A clone. The original is left alone.
    """
    import comfy.model_sampling

    class _Sampling(comfy.model_sampling.ModelSamplingDiscreteFlow, comfy.model_sampling.CONST):
        """A flow whose prediction is the distance from the picture to the map."""

        def calculate_denoised(self, sigma, model_output, model_input):
            return model_input - model_output

        def noise_scaling(self, sigma, noise, latent_image, max_denoise=False):
            return latent_image

        def inverse_noise_scaling(self, sigma, latent):
            return latent

    patched = model.clone()
    sampling = _Sampling(model.model.model_config)
    sampling.set_parameters(shift=1.0, multiplier=_MULTIPLIER)
    patched.add_object_patch("model_sampling", sampling)
    return patched


def depth(frame, per_frame: bool = False):
    """A decoded prediction as a depth map, near bright and far dark.

    Args:
        frame: ``(batch, height, width, 3)`` as the decoder answered it.
        per_frame: Stretch each frame over its own range rather than the batch's.

    Returns:
        ``(batch, height, width, 3)`` on a 0 to 1 scale.
    """
    held = frame.mean(dim=-1, keepdim=True)
    dims = (1, 2, 3) if per_frame else (0, 1, 2, 3)
    low = held.amin(dim=dims, keepdim=True)
    high = held.amax(dim=dims, keepdim=True)
    # Stretched over the range, and turned over.
    return ((high - held) / (high - low).clamp(min=EPSILON)).repeat(1, 1, 1, 3)


def normals(frame):
    """A decoded prediction as unit surface normals.

    Args:
        frame: ``(batch, height, width, 3)`` as the decoder answered it.

    Returns:
        ``(batch, height, width, 3)`` on a 0 to 1 scale, each pixel a unit vector.
    """
    vectors = frame * 2.0 - 1.0
    length = vectors.pow(2).sum(dim=-1, keepdim=True).sqrt().clamp(min=EPSILON)
    return ((vectors / length) + 1.0) * 0.5


def albedo(frame, picture_codes: bool = True):
    """A decoded prediction as albedo.

    Args:
        frame: ``(batch, height, width, 3)`` as the decoder answered it, linear light.
        picture_codes: Answer picture codes rather than the linear light it arrived as.

    Returns:
        ``(batch, height, width, 3)`` on a 0 to 1 scale.
    """
    held = frame.clamp(0.0, 1.0)
    if not picture_codes:
        return held
    high = 1.055 * held.clamp(min=0.0031308) ** (1.0 / 2.4) - 0.055
    return torch.where(held <= 0.0031308, held * 12.92, high)


def predict(model, vae, image, prompt):
    """Read one map off a picture, in a single sampler step.

    Args:
        model: A ``MODEL`` carrying the transformer and the map's adapter.
        vae: The ``VAE`` holding the map's decoder.
        image: ``(batch, height, width, 3)`` picture codes.
        prompt: The ``(1, tokens, width)`` embedding the map is conditioned on.

    Returns:
        ``(batch, height, width, 3)`` as the decoder answered it, before any map is read
        out of it.
    """
    import comfy.sample
    import comfy.samplers

    patched = _velocity(model)
    sampler = comfy.samplers.sampler_object("euler")
    sigmas = torch.tensor([TIMESTEP, 0.0], dtype=torch.float32)
    context = [[prompt, {}]]

    answers = []
    with torch.no_grad():
        # One frame at a time, each as a single still.
        for index in range(int(image.shape[0])):
            latent = vae.encode(image[index : index + 1])
            answered = comfy.sample.sample_custom(
                patched,
                torch.zeros_like(latent),
                1.0,
                sampler,
                sigmas,
                context,
                context,
                latent,
                disable_pbar=True,
            )
            decoded = vae.decode(answered)
            while decoded.ndim > 4:
                decoded = decoded[:, 0]
            answers.append(decoded)

    stacked = torch.cat(answers, dim=0)
    logger.debug(
        "%s read %d frame(s) into %s", MODEL_NAME, int(image.shape[0]), tuple(stacked.shape)
    )
    return stacked
