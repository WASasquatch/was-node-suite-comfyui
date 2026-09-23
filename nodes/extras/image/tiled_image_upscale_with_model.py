"""Running an upscale model tile by tile to any target magnification."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.compat.sockets import require_input
from ....modules.interface import size_report

REQUIRES = "extras"

logger = log.get_logger("nodes.extras.image")

#: Smallest tile the out-of-memory retry will fall back to before giving up.
MINIMUM_TILE = 64

#: What the model may run in, `auto` following what the model declares it supports.
PRECISIONS = ("auto", "32 bit float", "16 bit float", "bfloat16")


class TiledImageUpscaleWithModel(io.ComfyNode):
    """Upscale with a model in overlapping tiles, cross-faded so no seam shows."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASNSTiledImageUpscaleWithModel",
            display_name="Tiled Image Upscale (With Model)",
            search_aliases=[
                "WASNSTiledImageUpscaleWithModel", "upscale", "tiled", "esrgan", "seam",
            ],
            category="WAS Suite/Image/Upscaling",
            description=(
                "Upscale pictures with a loaded upscale model, one overlapping tile at a "
                "time, so a large frame fits in the memory a single pass would not. The "
                "overlaps are cross-faded, so no tile seams show, and the result is "
                "resampled to whatever magnification is asked for rather than the model's "
                "own fixed scale."
            ),
            inputs=[
                io.UpscaleModel.Input(
                    "upscale_model",
                    tooltip=(
                        "The upscale model to run, from a Load Upscale Model node. Its own "
                        "scale does not have to match upscale_factor: a 4x model can produce "
                        "a 2x result."
                    ),
                ),
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The pictures to enlarge. Each frame of a batch is upscaled in turn, "
                        "so memory use is set by the tile size rather than by the batch."
                    ),
                ),
                io.Float.Input(
                    "upscale_factor", default=4.0, min=1.0, max=16.0, step=0.1,
                    tooltip=(
                        "Final size relative to the input. 2.0 doubles both sides, 4.0 "
                        "quadruples them, 1.0 keeps the original size while still passing "
                        "the picture through the model."
                    ),
                ),
                io.Int.Input(
                    "tile_size", default=512, min=64, max=4096, step=16,
                    tooltip=(
                        "Tile edge in input pixels. `256` and `512` both suit most cards and "
                        "run at much the same speed; a larger tile needs more video memory "
                        "without being faster. If the card runs out, the tile is halved and "
                        "the run retried."
                    ),
                ),
                io.Int.Input(
                    "overlap", default=32, min=0, max=1024, step=1,
                    tooltip=(
                        "How far neighbouring tiles overlap, in input pixels. This is the "
                        "material the cross-fade is made from, so 0 puts a hard join between "
                        "tiles; 32 to 64 hides it on most models."
                    ),
                ),
                io.Int.Input(
                    "feather", default=0, min=0, max=4096, step=1,
                    tooltip=(
                        "Width of the cross-fade in output pixels. 0 works it out from the "
                        "overlap, which is the right answer almost always; raise it only "
                        "when a faint grid still shows on flat areas such as sky."
                    ),
                ),
                io.Combo.Input(
                    "resample_method",
                    options=["nearest-exact", "bilinear", "area", "bicubic", "lanczos"],
                    default="lanczos",
                    tooltip=(
                        "How a tile is resized when the model's own scale does not match "
                        "upscale_factor. `lanczos` keeps the most detail, `area` is the "
                        "gentlest when shrinking, `nearest-exact` keeps hard pixel edges for "
                        "pixel art."
                    ),
                ),
                io.Combo.Input(
                    "precision",
                    options=list(PRECISIONS),
                    default=PRECISIONS[0],
                    optional=True,
                    tooltip=(
                        "What the model runs in. `auto` = half precision where the model "
                        "declares it safe, which is about twice as fast; `32 bit float` = "
                        "every model's safest; `16 bit float` and `bfloat16` force one, "
                        "whatever the model says."
                    ),
                ),
                io.Boolean.Input(
                    "clear_comfy_memory", default=False,
                    tooltip=(
                        "Whether to unload every other model and empty the caches before "
                        "upscaling. Turn this on when a large upscale runs out of memory "
                        "next to a checkpoint that is still resident; it costs the time to "
                        "reload those models afterwards."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    tooltip=(
                        "The enlarged pictures, at roughly the input size times "
                        "upscale_factor, clamped to the displayable range."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        upscale_model,
        image,
        upscale_factor,
        tile_size,
        overlap,
        feather,
        resample_method,
        clear_comfy_memory,
        precision="auto",
    ) -> io.NodeOutput:
        """Upscale the batch tile by tile.

        Raises:
            ValueError: Nothing is connected to the upscale_model input.
        """
        import torch

        import comfy.utils
        from comfy import model_management

        from ....modules.image.tiled_upscale import tiled_upscale

        require_input(
            upscale_model,
            "Tiled Image Upscale (With Model)",
            "upscale_model",
            "model",
            "Load Upscale Model",
            "UPSCALE_MODEL",
        )

        device = model_management.get_torch_device()

        if clear_comfy_memory:
            try:
                model_management.unload_all_models()
                model_management.soft_empty_cache(True)
                torch.cuda.empty_cache()
                logger.info("unloaded every model and emptied the caches before upscaling")
            except Exception as error:
                logger.warning("the models and caches could not be cleared: %s", error)

        scale_estimate = getattr(upscale_model, "scale", 4.0)
        element_size = image.element_size()
        memory_required = model_management.module_size(upscale_model.model)
        memory_required += (
            (tile_size * tile_size * 3) * element_size * max(scale_estimate, 1.0) * 384.0
        )
        memory_required += image.nelement() * element_size
        model_management.free_memory(memory_required, device)

        upscale_model.to(device)
        working = cls.pick_dtype(precision, upscale_model, device)
        if working is not torch.float32:
            upscale_model.model.to(working)
            logger.info("running the upscale model in %s", working)

        _batch, in_height, in_width, _channels = image.shape
        upscale_factor = max(float(upscale_factor), 1.0)
        target_height = max(1, int(round(in_height * upscale_factor)))
        target_width = max(1, int(round(in_width * upscale_factor)))

        source = image.movedim(-1, -3).to("cpu")
        current_tile = int(tile_size)
        result = None

        while result is None:
            try:
                steps = source.shape[0] * comfy.utils.get_tiled_scale_steps(
                    source.shape[3], source.shape[2],
                    tile_x=current_tile, tile_y=current_tile, overlap=overlap,
                )
                result = tiled_upscale(
                    samples=source,
                    function=lambda tile: upscale_model(
                        tile.to(working).contiguous(memory_format=torch.channels_last)
                    ).float(),
                    tile_size=current_tile,
                    overlap=overlap,
                    output_device="cpu",
                    pbar=comfy.utils.ProgressBar(steps),
                    feather=feather,
                    target_height=target_height,
                    target_width=target_width,
                    resample_method=resample_method,
                    device=device,
                )
            except model_management.OOM_EXCEPTION:
                current_tile //= 2
                if current_tile < MINIMUM_TILE:
                    upscale_model.to("cpu")
                    raise
                logger.warning(
                    "the upscale ran out of memory; retrying with %d pixel tiles", current_tile
                )

        upscale_model.model.to(torch.float32)
        upscale_model.to("cpu")
        upscaled = torch.clamp(result, min=0.0, max=1.0).movedim(-3, -1)
        size_report.publish(
            image,
            upscaled,
            action="upscaled",
            requested=(target_width, target_height),
            facts={"tile": f"{current_tile} px", "precision": str(working).replace("torch.", "")},
        )
        return io.NodeOutput(upscaled)

    @staticmethod
    def pick_dtype(precision: str, upscale_model, device):
        """What the model runs in.

        Args:
            precision: An entry from :data:`PRECISIONS`.
            upscale_model: The loaded upscale model, read for the dtypes it declares.
            device: The device the model runs on.

        Returns:
            A ``torch.dtype``.
        """
        import torch

        from comfy import model_management

        named = {
            "32 bit float": torch.float32,
            "16 bit float": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        if precision in named:
            return named[precision]
        if not model_management.should_use_fp16(device):
            return torch.float32
        if getattr(upscale_model, "supports_half", False):
            return torch.float16
        if getattr(upscale_model, "supports_bfloat16", False):
            return torch.bfloat16
        return torch.float32
