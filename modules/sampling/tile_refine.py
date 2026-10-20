"""One picture refined over overlapping tiles: upscaled, encoded, captioned per tile, sampled whole, decoded.

Pictures are IMAGE tensors ``[1, H, W, C]`` in ``[0, 1]``; latents are ``[1, C, H, W]`` or
``[1, C, T, H, W]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .. import log
from ..interface import tile_activity
from . import tile_captions, tiles

__all__ = [
    "Frame",
    "LARGE_PIXELS",
    "VAE_OVERLAP",
    "VAE_TILE",
    "caption_line",
    "captions_text",
    "decode",
    "encode",
    "fitted",
    "model_channels",
    "prepare",
    "publish",
    "release",
    "source_size",
    "stage_sizes",
    "tracked",
    "tile_settings",
    "upscaled",
]

logger = log.get_logger("sampling.tile_refine")

#: Pixels in a picture above which it is encoded and decoded in tiles.
LARGE_PIXELS = 1536 * 1536

#: Side and overlap in pixels of the tiles a large picture is encoded and decoded in.
VAE_TILE = 512
VAE_OVERLAP = 64


@dataclass
class Frame:
    """One picture made ready for tiled sampling.

    Attributes:
        pixels: The picture at the size it is sampled at, ``[1, H, W, C]``.
        latent: Its latent.
        grid: The tiling of the latent.
        masks: One blend mask per tile.
        overview: What the captioner wrote about the whole picture, empty without one.
        captions: One caption per tile, empty without a captioner.
        texts: One CONDITIONING per tile replacing the positive text, or None.
        announce: Sends which tiles are being worked on to the node's panel.
    """

    pixels: torch.Tensor
    latent: torch.Tensor
    grid: tiles.Grid
    masks: list
    overview: str = ""
    captions: list = field(default_factory=list)
    texts: list | None = None
    announce: object = None


def tile_settings(choice: dict, model) -> tuple:
    """Tile size and overlap in pixels from the tiling widget's value.

    Args:
        choice: The widget's value, its option under ``tiling`` and that option's inputs.
        model: The model patcher sampled, which sets the automatic size.

    Returns:
        ``((tile_height, tile_width), overlap)``.
    """
    choice = choice or {}
    if choice.get("tiling") == "manual":
        return ((int(choice.get("tile_height", 1024)), int(choice.get("tile_width", 1024))),
                int(choice.get("overlap", 256)))
    side = tiles.auto_tile(model)
    return ((side, side), int(side * tiles.AUTO_OVERLAP))


def _multiple(vae) -> int:
    """Pixels a picture's sides are rounded to: the VAE's downscale times the tile alignment."""
    try:
        ratio = vae.spacial_compression_encode()
    except Exception:
        ratio = 8
    if not isinstance(ratio, int) or ratio <= 0:
        ratio = 8
    return ratio * tiles.ALIGN


def source_size(picture, latent, vae) -> tuple[int, int]:
    """A source's size in pixels.

    Args:
        picture: ``[1, H, W, C]``, or None for a latent.
        latent: ``[1, C, h, w]`` read where ``picture`` is None.
        vae: The VAE the latent decodes with.

    Returns:
        ``(height, width)``.
    """
    if picture is not None:
        return int(picture.shape[1]), int(picture.shape[2])
    try:
        ratio = vae.spacial_compression_decode()
    except Exception:
        ratio = 8
    ratio = ratio if isinstance(ratio, int) and ratio > 0 else 8
    return int(latent.shape[-2]) * ratio, int(latent.shape[-1]) * ratio


def stage_sizes(height: int, width: int, upscale_by: float, stages: int, vae) -> list[tuple[int, int]]:
    """The size each pass of a staged refine lands on, the last being the size one pass reaches.

    Args:
        height: The source picture's height in pixels.
        width: Its width in pixels.
        upscale_by: Size of the result against the source.
        stages: How many passes reach it, each enlarging by the same factor.
        vae: The VAE the passes encode with.

    Returns:
        One ``(height, width)`` per pass, each side a multiple of the VAE's downscale.
    """
    multiple = _multiple(vae)
    count = max(1, int(stages))
    sizes = []
    for stage in range(1, count + 1):
        factor = float(upscale_by) ** (stage / count)
        sizes.append((max(multiple, int(round(height * factor / multiple)) * multiple),
                      max(multiple, int(round(width * factor / multiple)) * multiple)))
    return sizes


def upscaled(image: torch.Tensor, upscale_by: float, upscale_model, vae, size=None) -> torch.Tensor:
    """One picture enlarged by a model where given, then resized to ``upscale_by`` on whole latent pixels.

    Args:
        image: ``[1, H, W, C]``.
        upscale_by: Size of the answer against the picture.
        upscale_model: An UPSCALE_MODEL run first, or None.
        vae: The VAE the answer is encoded with.
        size: ``(height, width)`` to land on in place of ``upscale_by``, or None.

    Returns:
        ``[1, h, w, 3]`` with each side a multiple of the VAE's downscale.
    """
    from ..compat import operations

    picture = image[..., :3].float()
    height, width = int(picture.shape[1]), int(picture.shape[2])
    target_h, target_w = size if size is not None else stage_sizes(height, width, upscale_by, 1, vae)[0]
    if upscale_model is not None:
        picture = operations.upscale_with_model(upscale_model, picture)
    if tuple(picture.shape[1:3]) == (target_h, target_w):
        return picture.clamp(0.0, 1.0)
    grow = target_h * target_w > picture.shape[1] * picture.shape[2]
    resized = torch.nn.functional.interpolate(
        picture.movedim(-1, 1), size=(target_h, target_w), mode="bicubic", align_corners=False,
        antialias=not grow,
    )
    return resized.movedim(1, -1).clamp(0.0, 1.0)


def encode(vae, pixels: torch.Tensor) -> torch.Tensor:
    """A picture's latent, encoded in tiles above :data:`LARGE_PIXELS`."""
    if pixels.shape[1] * pixels.shape[2] > LARGE_PIXELS:
        return vae.encode_tiled(pixels, tile_x=VAE_TILE, tile_y=VAE_TILE, overlap=VAE_OVERLAP)
    return vae.encode(pixels)


def decode(vae, latent: torch.Tensor) -> torch.Tensor:
    """A latent's picture as ``[N, H, W, C]``, decoded in tiles above :data:`LARGE_PIXELS`."""
    try:
        ratio = vae.spacial_compression_decode()
    except Exception:
        ratio = 8
    ratio = ratio if isinstance(ratio, int) and ratio > 0 else 8
    if latent.shape[-2] * latent.shape[-1] * ratio * ratio > LARGE_PIXELS:
        tile = max(8, VAE_TILE // ratio)
        images = vae.decode_tiled(latent, tile_x=tile, tile_y=tile, overlap=max(1, VAE_OVERLAP // ratio))
    else:
        images = vae.decode(latent)
    if images.ndim == 5:
        images = images.reshape(-1, *images.shape[-3:])
    return images


def fitted(vae, latent: torch.Tensor, node: str = "") -> torch.Tensor:
    """A source latent checked against the VAE and shaped for it.

    Args:
        vae: The VAE the latent decodes with.
        latent: ``[1, C, H, W]`` or ``[1, C, T, H, W]``.
        node: The node's display name, for messages.

    Returns:
        The latent, with a frame axis added where the VAE decodes video latents.

    Raises:
        ValueError: The latent's channel count is not the one the VAE decodes.
    """
    channels = getattr(vae, "latent_channels", None)
    if isinstance(channels, int) and latent.ndim in (4, 5) and int(latent.shape[1]) != channels:
        raise ValueError(
            f"{node} decodes a LATENT source with the VAE wired to vae, which reads {channels}-channel "
            f"latents, and this latent has {int(latent.shape[1])} channels, so another model's VAE "
            "made it. Keep vae as it is, decode this latent with the VAE that made it, and wire that "
            "picture into source as IMAGE"
        )
    if getattr(vae, "latent_dim", 2) == 3 and latent.ndim == 4:
        return latent.unsqueeze(2)
    return latent


def model_channels(model) -> int | None:
    """The channel count of the latents a model samples, or None where it is not known."""
    try:
        return int(model.get_model_object("latent_format").latent_channels)
    except Exception:
        return None


def prepare(image: torch.Tensor | None, model, vae, upscale_by: float, upscale_model, tiling: dict,
            clip=None, captioner=None, style: str = "", system_prompt: str = "",
            node: str = "", context: str = "", seed: int = 0, latent: torch.Tensor | None = None,
            size=None, stage: tuple = (1, 1)) -> Frame:
    """Upscale, encode, tile and caption one picture, given as pixels or as a latent.

    Args:
        image: ``[1, H, W, C]``, or None where ``latent`` is given.
        model: The model patcher sampled.
        vae: The VAE.
        upscale_by: Size of the sampled picture against ``image``.
        upscale_model: An UPSCALE_MODEL run first, or None.
        tiling: The tiling widget's value.
        clip: The model's text encoder, for the captions.
        captioner: A vision language model writing a caption per tile, or None.
        style: Words added to the end of every caption.
        system_prompt: The rules each caption is written under.
        node: The node's display name, for messages.
        context: The prompt the picture was made from, read by the captioner.
        seed: Seeds the captions; the same seed on the same picture reuses them.
        latent: A ``[1, C, H, W]`` latent refined in place of ``image``, sampled as it is at
            ``upscale_by`` 1.0 or a ``size`` it already has, with no upscale model, and decoded
            for the captions.
        size: ``(height, width)`` to land on in place of ``upscale_by``, or None.
        stage: ``(pass, passes)`` of a staged refine, announced to the node's panel.

    Returns:
        The frame.

    Raises:
        ValueError: The latent is not a picture latent, came from another VAE or holds several
            frames, the VAE does not belong to the model, or the captioner cannot be used.
    """
    import comfy.model_management
    import comfy.utils

    announce = tile_activity.Announcer([], stage=stage)
    announce(tile_activity.PREPARE)
    as_is = False
    if latent is not None:
        latent = fitted(vae, latent, node)
        image = decode(vae, latent)
        if int(image.shape[0]) != 1:
            raise ValueError(
                f"{node} refines one picture at a time, and this latent decodes to "
                f"{int(image.shape[0])} frames. Pick one frame of it, or decode it and wire the "
                "frames into source"
            )
        kept = (tuple(int(v) for v in image.shape[1:3]) == tuple(size) if size is not None
                else abs(float(upscale_by) - 1.0) < 1e-6)
        as_is = kept and upscale_model is None
    if as_is:
        pixels = image[..., :3].float().clamp(0.0, 1.0)
    else:
        pixels = upscaled(image, upscale_by, upscale_model, vae, size)
        latent = encode(vae, pixels)
    if getattr(latent, "is_nested", False) or latent.ndim not in (4, 5):
        raise ValueError(
            f"{node} samples picture latents, and this VAE encodes to a {latent.ndim}-axis latent. "
            "Use the VAE of an image or video model"
        )
    expected = model_channels(model)
    if expected is not None and int(latent.shape[1]) != expected and bool(torch.count_nonzero(latent)):
        raise ValueError(
            f"{node} samples {expected}-channel latents with this model, and the VAE wired to vae "
            f"makes {int(latent.shape[1])}-channel ones. Wire the VAE that belongs to the model into vae"
        )
    scale = pixels.shape[1] / latent.shape[-2]
    tile, overlap = tile_settings(tiling, model)
    grid = tiles.plan(tuple(latent.shape[-2:]), scale, tile, overlap)
    announce.boxes = [[round(float(v), 5) for v in box] for box in grid.fractions()]
    frame = Frame(pixels, latent, grid, tiles.masks(grid, tuple(latent.shape[2:])), announce=announce)
    _publish_plan(pixels, grid)
    if captioner is None:
        return frame

    tile_captions.require_captioner(captioner, clip, node)
    progress = comfy.utils.ProgressBar(grid.count + 1)
    release(captioner)
    frame.overview, frame.captions = tile_captions.caption_tiles(
        captioner, pixels, grid, system_prompt, progress, frame.announce, context, seed,
        source=image, written=lambda tile, text: announce.wrote(tile, caption_line(grid, tile, text)))
    release(clip)
    suffix = str(style or "").strip()
    texts = []
    for caption in frame.captions:
        comfy.model_management.throw_exception_if_processing_interrupted()
        if not caption:
            texts.append(None)
            continue
        text = f"{caption.rstrip(' .,;')}. {suffix}" if suffix else caption
        texts.append(tile_captions.encode(clip, text))
    frame.texts = texts
    return frame


def release(*keep) -> None:
    """Free the VRAM every loaded model holds except the ones named, so the next phase loads whole.

    Args:
        *keep: Model patchers, or CLIP and VAE objects carrying one, to leave loaded.
    """
    from ..model import release_others

    try:
        release_others([getattr(item, "patcher", item) for item in keep if item is not None], unpin=True)
    except Exception as error:
        logger.debug("The other models could not be released: %s", error)


def tracked(callback, announce, steps: int):
    """A sampler callback that also counts finished steps for the node's panel.

    Args:
        callback: The callback the sampler is handed, called as ``callback(step, x0, x, total)``.
        announce: The frame's announcer.
        steps: Steps in the run.

    Returns:
        The wrapped callback.
    """
    announce.begin(steps)

    def counted(step, x0, x, total_steps):
        announce.finished(step)
        return callback(step, x0, x, total_steps)

    return counted


def _publish_plan(picture: torch.Tensor, grid: tiles.Grid) -> None:
    """Publish the tile plan over a picture, for the node's own panel."""
    from ..interface import preview

    try:
        preview.publish_output(tiles.plot(picture, grid))
    except Exception as error:
        logger.debug("The tile plan could not be published: %s", error)


def publish(frames: list, sampler: str, affine: dict | None = None, picture=None, stages: int = 1) -> None:
    """Publish the tile plan of the last frame and its captions, for the node's own panel.

    Args:
        frames: The last stage's frames, in order.
        sampler: The sampler's name.
        affine: What the affine reported, ``{"applied": steps, "gain": factor, "acts_on": name}``,
            or None.
        picture: The last refined picture, ``[1, H, W, C]``, the tiles are drawn over; the
            picture sampled when left out.
        stages: Passes each picture took to reach its size.
    """
    from ..interface import preview, run_result

    try:
        if not frames:
            return
        last = frames[-1]
        grid = last.grid
        preview.publish_output(tiles.plot(last.pixels if picture is None else picture, grid))
        if not run_result.watching():
            return
        height, width = int(last.pixels.shape[1]), int(last.pixels.shape[2])
        tile_h, tile_w = (int(round(v * grid.scale)) for v in grid.tile)
        shared = max(grid.overlap)
        counts = {
            "tiles": grid.count,
            "tile width": tile_w,
            "tile height": tile_h,
            "overlap": int(round(shared * grid.scale)),
            "pictures": len(frames),
        }
        if stages > 1:
            counts["stages"] = int(stages)
        facts = {"size": f"{width}x{height} px", "sampler": sampler}
        if affine is not None:
            counts["affine steps"] = int(affine.get("applied", 0))
            scaled = "noise" if affine.get("acts_on") == "noise" else "picture"
            facts["affine"] = f"{scaled} {(float(affine.get('gain', 1.0)) - 1.0) * 100.0:+.1f}%"
        captioned = sum(1 for caption in last.captions if caption)
        if last.captions:
            summary = f"{grid.count} tiles, {captioned} captioned, over {width}x{height} px"
        else:
            summary = f"{grid.count} tiles over {width}x{height} px, every tile on the positive prompt"
        text = captions_text([last])
        run_result.publish(
            status=run_result.OK if captioned == len(last.captions) else run_result.WARNING,
            summary=summary,
            counts=counts,
            facts=facts,
            bodies=[run_result.body("captions", text)] if text else None,
        )
    except Exception as error:
        logger.debug("The tile report could not be published: %s", error)


def captions_text(frames: list) -> str:
    """Every frame's overview and tile captions as numbered lines.

    Args:
        frames: The frames sampled, in order.

    Returns:
        The text, empty where nothing was captioned.
    """
    blocks = []
    for number, frame in enumerate(frames, start=1):
        if not frame.captions:
            continue
        lines = [f"picture {number}" if len(frames) > 1 else "",
                 caption_line(frame.grid, -1, frame.overview)]
        lines += [caption_line(frame.grid, index, caption) for index, caption in enumerate(frame.captions)]
        blocks.append("\n".join(line for line in lines if line))
    return "\n\n".join(blocks)


def caption_line(grid: tiles.Grid, tile: int, caption: str) -> str:
    """One numbered line of the captions output.

    Args:
        grid: The tiling the caption belongs to.
        tile: The tile from 0, or ``-1`` for the overview.
        caption: What the captioner wrote, empty where it wrote nothing.

    Returns:
        The line.
    """
    if tile < 0:
        return f"overview: {caption}"
    return f"{tile + 1} {tiles.position(grid, tile)}: {caption or '(positive prompt)'}"
