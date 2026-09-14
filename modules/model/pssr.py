"""Driving PS-SR video super resolution from frames already in memory.

:func:`load_pipelines` builds the two pipelines the method needs, :func:`window_starts` and
:func:`blend_weights` cover a clip longer than one pass, and :func:`to_pil` and
:func:`from_pil` carry frames across.
"""

from __future__ import annotations

import gc
import math
import pathlib
import sys
from typing import Iterable

import torch

from .. import log
from ..config.paths import env_value
from . import model_directories

logger = log.get_logger("modules.model.pssr")

#: Folder names looked for under ComfyUI's models directory, in order.
MODEL_DIRS = ("PS-SR", "ps-sr", "pssr")

#: What the checkout must carry. The transformer is not among them: it comes from ComfyUI. Nor
#: are the tagger and its LoRA, whose captions conditioning replaced, nor the 11 GB text encoder.
REQUIRED = (
    "checkpoints/pretrained_models/base.safetensors",
    "checkpoints/pretrained_models/draft.safetensors",
    "dependent_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth",
    "models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl",
)

#: The one set of pipelines currently built, as ``(key, base, draft)``, or ``None``. Kept so a
#: repeated run skips a minute of construction. Deliberately a single entry: each pair is several
#: gigabytes, and keeping more than one is how a card fills up over a few runs.
_CACHED: tuple | None = None


def find_root(explicit: str | None = None) -> pathlib.Path:
    """Locate the PS-SR checkout.

    Args:
        explicit: A path to use instead of searching, or ``None`` to search.

    Returns:
        The checkout directory.

    Raises:
        FileNotFoundError: No directory was found, or the one found is missing a required file.
    """
    candidates: list[pathlib.Path] = []
    if explicit:
        candidates.append(pathlib.Path(explicit))
    root = env_value("PSSR_ROOT")
    if root:
        candidates.append(pathlib.Path(root))
    # Registering each name means an extra_model_paths entry spelling it takes effect, so a
    # checkout kept on another drive is found where it lies.
    for name in MODEL_DIRS:
        candidates.extend(model_directories(name))

    for candidate in candidates:
        if candidate.is_dir():
            missing = [part for part in REQUIRED if not (candidate / part).exists()]
            if missing:
                raise FileNotFoundError(
                    f"{candidate} looks like a PS-SR checkout but is missing "
                    f"{', '.join(missing)}. See docs/MODELS.md for the layout."
                )
            return candidate

    looked = ", ".join(str(c) for c in candidates) or "nowhere"
    raise FileNotFoundError(
        f"PS-SR was not found. Put the checkout and its weights in one of {MODEL_DIRS} under "
        f"ComfyUI/models, name one of those in extra_model_paths.yaml to keep it on "
        f"another drive, or set PSSR_ROOT. Looked in: {looked}. See docs/MODELS.md."
    )


def _import_upstream(root: pathlib.Path):
    """Put the checkout on the path and return the pieces needed to build its pipelines."""
    import sys

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from Wan_SR.pipelines.wan_sr_base import ModelConfig, WanVideoSRPipeline_base
    from Wan_SR.pipelines.wan_sr_draft import WanVideoSRPipeline_draft

    return ModelConfig, WanVideoSRPipeline_base, WanVideoSRPipeline_draft


def _tokenizer_config(root: pathlib.Path, ModelConfig):
    """The umt5 tokenizer, as an absolute path.

    Args:
        root: The PS-SR checkout.
        ModelConfig: Upstream's config class.

    Returns:
        A config pointing at the tokenizer already in the checkout.
    """
    path = root / "models" / "Wan-AI" / "Wan2.1-T2V-1.3B" / "google" / "umt5-xxl"
    if not path.is_dir():
        raise FileNotFoundError(
            f"the umt5 tokenizer is missing from the checkout at {path}. It ships with the "
            f"PS-SR repository, so a partial clone is the usual cause."
        )
    return ModelConfig(path=str(path), skip_download=True)


def _vae_file(root: pathlib.Path) -> str:
    """The VAE, the only Wan file still read from the checkout."""
    return str(root / "dependent_models" / "Wan2.1-T2V-1.3B" / "Wan2.1_VAE.pth")


class _NoCaptions:
    """Stands in for the tagger's inference so it is never run."""

    def __init__(self, original):
        self.original = original

    def __call__(self, images, model, *args, **kwargs):
        count = images.shape[0] if hasattr(images, "shape") else 1
        return [""] * count


def load_pipelines(root: pathlib.Path, dtype: torch.dtype, device: str, k_select: float,
                   dit_state=None, dit_key=None):
    """Build the base and draft pipelines, or return the ones already built.

    Args:
        root: The PS-SR checkout.
        dtype: Working dtype for both pipelines.
        device: Device to run on.
        k_select: How much of the draft transformer to keep, which changes its shape and so
            cannot be varied without rebuilding.
        dit_state: Transformer weights, as ComfyUI's loader provides them.
        dit_key: Something identifying those weights, so a different model is not served from
            the cache built for the last one.

    Returns:
        ``(base pipeline, draft pipeline)``.
    """
    global _CACHED
    key = (str(root), str(dtype), device, float(k_select), dit_key)
    if _CACHED is not None and _CACHED[0] == key:
        return _CACHED[1], _CACHED[2]
    # Anything held from a previous run goes before more is built, or the two sets overlap on the
    # card and the second one has nowhere to fit.
    release_pipelines()

    import torch.nn as nn
    from diffsynth.models import load_state_dict

    ModelConfig, BasePipeline, DraftPipeline = _import_upstream(root)
    # Constructed on the CPU so diffsynth streams layers up as they are needed. Built straight
    # onto the GPU, the two pipelines together exceed 24 GB before a frame is touched.
    configs = [ModelConfig(path=_vae_file(root), offload_device="cpu")]
    # The tagger's captioning is replaced before either pipeline is built, so neither runs it.
    _silence_tagger()
    # Left empty on purpose: ram() only loads a checkpoint when given one, and its captions are
    # discarded. The 5 GB read and the DAPE LoRA are both skipped.
    ram_path = None
    dape_path = None
    checkpoints = root / "checkpoints" / "pretrained_models"

    # Upstream builds both pipelines with their own copy of every model, which puts two 11 GB
    # text encoders on the card and exceeds 24 GB before a frame is touched. Constructing on the
    # CPU and sharing the read-only models is the difference between about 8 GB and not running.
    logger.info("building the PS-SR base pipeline")
    tokenizer = _tokenizer_config(root, ModelConfig)
    base = BasePipeline.from_pretrained(
        torch_dtype=dtype, device=device, model_configs=configs,
        tokenizer_config=tokenizer, ram_path=ram_path, DAPE_path=dape_path,
    )
    base.dit = build_dit(dit_state, dtype, device)
    base.dit.patch_embedding = expand_patch_embedding(base.dit.patch_embedding, factor=2)
    base.load_lora(base.dit, str(checkpoints / "base.safetensors"), alpha=1)
    base.enable_vram_management()
    base.net_lpips = None

    logger.info("building the PS-SR draft pipeline")
    draft = DraftPipeline.from_pretrained(
        torch_dtype=dtype, device=device, model_configs=configs,
        tokenizer_config=tokenizer, ram_path=ram_path, DAPE_path=dape_path,
    )
    draft.dit = build_dit(dit_state, dtype, device)
    draft.dit.patch_embedding = expand_patch_embedding(draft.dit.patch_embedding, factor=2)
    draft.shave_dit_draft(k_select=k_select)
    blocks = len(draft.dit.blocks)
    draft.dit.fc_layers = nn.ModuleList([nn.Linear(1536 * 2, 1536) for _ in range(blocks)])
    draft.dit.load_state_dict(load_state_dict(str(checkpoints / "draft.safetensors")))
    draft.enable_vram_management()
    draft.net_lpips = None

    # The text encoder, VAE and tagger are identical between the two and only ever read, so the
    # draft borrows the base's rather than holding a second copy of each.
    draft.text_encoder = base.text_encoder
    draft.prompter.fetch_models(draft.text_encoder)
    draft.vae = base.vae
    draft.model_vlm = base.model_vlm

    gc.collect()
    torch.cuda.empty_cache()
    _CACHED = (key, base, draft)
    return base, draft


def expand_patch_embedding(old, factor: int = 2):
    """Widen a patch embedding so it takes the noisy latent and the source latent together.

    Args:
        old: The ``Conv3d`` to widen.
        factor: How many times wider the input becomes.

    Returns:
        The replacement layer.
    """
    import torch.nn as nn

    widened = nn.Conv3d(
        in_channels=old.in_channels * factor,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
    )
    weight = torch.zeros(
        (old.out_channels, old.in_channels * factor, *old.kernel_size),
        dtype=old.weight.dtype, device=old.weight.device,
    )
    weight[:, : old.in_channels, ...] = old.weight.data.clone()
    widened.weight = nn.Parameter(weight)
    if old.bias is not None:
        widened.bias = nn.Parameter(old.bias.data.clone())
    return widened.to(dtype=old.weight.dtype, device=old.weight.device)


def window_starts(length: int, window: int, overlap: int) -> list[int]:
    """Where each sliding window begins, covering ``length`` with the last one flush to the end.

    Args:
        length: Extent to cover.
        window: Window size.
        overlap: How much each window shares with the one before it.

    Returns:
        Start offsets, ascending, the last chosen so its window ends exactly at ``length``.
    """
    if length <= window:
        return [0]
    stride = max(1, window - overlap)
    # Spaced evenly rather than at a fixed stride with the last one snapped flush. Snapping can
    # leave the final window sharing far more than ``overlap`` with its neighbour, and since only
    # ``overlap`` of it is feathered the rest is counted twice at full weight: a wide strip
    # averaging two separate passes, abutting pixels only one pass covered, which reads as a band.
    count = max(2, -(-(length - overlap) // stride))
    span = length - window
    return [round(i * span / (count - 1)) for i in range(count)]


def shared_extent(starts: list[int], window: int) -> int:
    """How much neighbouring windows actually share, which is what the feather has to span.

    Args:
        starts: Window offsets as :func:`window_starts` returns them.
        window: Window size.

    Returns:
        The smallest overlap between any two neighbours, or 0 for a single window.
    """
    if len(starts) < 2:
        return 0
    widest = max(b - a for a, b in zip(starts, starts[1:]))
    return max(0, window - widest)


def blend_weights(
    count: int, overlap: int, device, dtype, lead: bool = True, trail: bool = True,
) -> torch.Tensor:
    """A raised-cosine ramp for feathering one window into the next.

    Args:
        count: Length of the window.
        overlap: How much it shares with its neighbour.
        device: Where the ramp is built.
        dtype: Working dtype.
        lead: Taper the start, ie another window precedes this one.
        trail: Taper the end, ie another window follows it.

    Returns:
        A ``count``-long ramp, flat at any edge that was not tapered.
    """
    weights = torch.ones(count, device=device, dtype=dtype)
    if overlap > 0 and (lead or trail):
        ramp = torch.linspace(0, math.pi, min(overlap, count), device=device, dtype=dtype)
        taper = (1 - torch.cos(ramp)) / 2
        if lead:
            weights[: taper.numel()] = torch.minimum(weights[: taper.numel()], taper)
        if trail:
            weights[-taper.numel():] = torch.minimum(weights[-taper.numel():], taper.flip(0))
    return weights.clamp(min=1e-3)


def to_pil(frames: torch.Tensor) -> list:
    """ComfyUI frames to the PIL list the pipelines take."""
    from PIL import Image

    out = []
    for frame in frames:
        array = (frame.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
        out.append(Image.fromarray(array))
    return out


def from_pil(images: Iterable) -> torch.Tensor:
    """A pipeline's PIL frames back to a ComfyUI batch."""
    import numpy as np

    stacked = [torch.from_numpy(np.asarray(i.convert("RGB"), dtype=np.float32) / 255.0)
               for i in images]
    return torch.stack(stacked, dim=0)


class supplied_conditioning:
    """Make the pipelines use conditioning from outside instead of encoding a prompt.

    Args:
        pipes: The pipelines to redirect.
        positive: The embedding to use for the positive pass.
        negative: The embedding for the negative pass, or ``None`` to reuse ``positive``. The
            negative pass only runs when cfg is above 1.
    """

    # The pipelines build their own embedding: a tagger reads the frames, its caption is appended
    # to the prompt string, and the result goes through the pipeline's own copy of umt5. ComfyUI
    # has that same encoder, so its CONDITIONING is the same kind of tensor. Replacing the encoder
    # skips the caption too, since the string it was appended to is never read.
    def __init__(self, pipes, positive, negative=None):
        self.pipes = list(pipes)
        self.positive = positive
        self.negative = negative if negative is not None else positive
        self.restore = []

    def __enter__(self):
        for pipe in self.pipes:
            prompter = pipe.prompter
            self.restore.append((prompter, prompter.encode_prompt))

            def encode(prompt, positive=True, device=None, _pipe=pipe, **kwargs):
                chosen = self.positive if positive else self.negative
                return chosen.to(device=_pipe.device, dtype=_pipe.torch_dtype)

            prompter.encode_prompt = encode
        return self

    def __exit__(self, *unused):
        for prompter, original in self.restore:
            prompter.encode_prompt = original
        self.restore.clear()
        return False


def conditioning_tensor(conditioning):
    """The embedding out of a ComfyUI CONDITIONING.

    Args:
        conditioning: ComfyUI's ``[[tensor, dict], ...]`` conditioning.

    Returns:
        The first entry's tensor.

    Raises:
        ValueError: The conditioning is empty or not shaped as ComfyUI produces it.
    """
    try:
        tensor = conditioning[0][0]
    except (TypeError, IndexError) as bad:
        raise ValueError(
            "conditioning must be ComfyUI's [[tensor, dict]] form, as CLIP Text Encode gives it."
        ) from bad
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"conditioning holds {type(tensor).__name__}, not a tensor.")
    return tensor


def dit_state_dict(model):
    """The transformer weights out of a ComfyUI MODEL, keyed as diffsynth names them.

    Args:
        model: A ComfyUI ``MODEL``, as its loaders produce.

    Returns:
        A state dict with ComfyUI's ``model.diffusion_model.`` prefix removed, which is exactly
        how diffsynth's ``WanModel`` names the same parameters.

    Raises:
        ValueError: Nothing in the model looks like a Wan transformer.
    """
    inner = getattr(model, "model", model)
    state = inner.state_dict() if hasattr(inner, "state_dict") else {}
    # Only what sits under the transformer's own prefix. ComfyUI keeps its own buffers alongside
    # it, `model_sampling.sigmas` among them, and diffsynth recognises a checkpoint by hashing
    # the key set, so one stray name is enough to make it unrecognisable.
    stripped = {}
    for name, tensor in state.items():
        for prefix in ("model.diffusion_model.", "diffusion_model."):
            if name.startswith(prefix):
                stripped[name[len(prefix):]] = tensor
                break
    if not stripped:
        # Already a bare transformer, with nothing wrapped around it.
        stripped = dict(state)
    # Detached onto the CPU: ComfyUI's copy is freed before the pipelines are built, and a
    # reference into weights it has released would be a use after free.
    stripped = {name: tensor.detach().to("cpu", copy=True) for name, tensor in stripped.items()}
    if not any(key.startswith("blocks.") for key in stripped):
        raise ValueError(
            "the MODEL input does not look like a Wan transformer: no 'blocks.' parameters. "
            "Load a Wan 2.1 model with Load Diffusion Model."
        )
    return stripped


def build_dit(state, dtype, device):
    """Build a Wan transformer from weights ComfyUI loaded.

    Args:
        state: Weights as :func:`dit_state_dict` returns them.
        dtype: Working dtype.
        device: Where the model runs.

    Returns:
        The transformer, ready for the patch embedding to be widened and the LoRA applied.

    Raises:
        ValueError: diffsynth does not recognise the checkpoint as a Wan transformer.
    """
    from diffsynth.models.wan_video_dit import WanModel

    converted, config = WanModel.state_dict_converter().from_civitai(state)
    if not config:
        raise ValueError(
            "the MODEL input was not recognised as a Wan transformer. PS-SR is trained against "
            "Wan 2.1 T2V-1.3B, so load one of those with Load Diffusion Model."
        )
    model = WanModel(**config)
    model.load_state_dict(converted)
    return model.to(device=device, dtype=dtype).eval()


def _silence_tagger():
    """Replace the tagger's captioning with one that returns nothing."""
    for module_name in ("Wan_SR.pipelines.wan_sr_base", "Wan_SR.pipelines.wan_sr_draft"):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        current = getattr(module, "inference", None)
        if current is None or isinstance(current, _NoCaptions):
            continue
        module.inference = _NoCaptions(current)


def release_comfy_models() -> None:
    """Let ComfyUI free what it is holding on the card before the pipelines are built."""
    # ComfyUI keeps the transformer and the text encoder resident after a node has read them, and
    # PS-SR builds its own two pipelines beside them, which does not fit on a 24 GB card. The
    # weights and the conditioning have already been copied out by the time this runs.
    try:
        import comfy.model_management as management

        management.unload_all_models()
        management.soft_empty_cache(force=True)
    except Exception as unavailable:
        logger.debug("could not release ComfyUI's models: %s", unavailable)


def release_pipelines() -> None:
    """Free the pipelines currently held, if any."""
    global _CACHED
    if _CACHED is None:
        return
    logger.info("releasing the previously built PS-SR pipelines")
    _CACHED = None
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def fingerprint(state) -> str:
    """Identify a set of weights cheaply enough to check on every run.

    Args:
        state: A transformer state dict.

    Returns:
        A short string, stable for the same weights across reloads.
    """
    from diffsynth.models.utils import hash_state_dict_keys

    shape = hash_state_dict_keys(state)
    marker = "blocks.0.self_attn.q.weight"
    tensor = state.get(marker)
    if tensor is None:
        return f"{shape}:no-marker"
    return f"{shape}:{float(tensor.detach().float().sum()):.6e}"
