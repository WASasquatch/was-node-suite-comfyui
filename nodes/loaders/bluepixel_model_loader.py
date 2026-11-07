"""Loading the BluePixel image editing transformer."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.interface import run_result
from ...modules.io import picker
from ...modules.model.bluepixel import model as bluepixel

NODE_NAME = "BluePixel Model Loader"

#: Shown in the file list when the diffusion_models folders hold nothing.
NO_FILE = "put a BluePixel transformer in models/diffusion_models"


class BluePixelModelLoader(io.ComfyNode):
    """Load the BluePixel transformer as a MODEL sampling DC-AE f64c128 latents."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        files, likely = bluepixel.offered()
        return io.Schema(
            node_id="WASBluePixelModelLoader",
            display_name=NODE_NAME,
            search_aliases=[
                "WASBluePixelModelLoader", NODE_NAME,
                "bluepixel", "blue pixel", "bluepixel turbo", "vivo", "4k image edit", "load bluepixel",
            ],
            category="WAS Suite/Loaders",
            description=(
                "Load BluePixel-Turbo, the 4K instruction image editor built on FLUX.2 Klein 9B, for "
                "Text Encode BluePixel and BluePixel Scheduler. The panel shows the file's size, "
                "its parameter count and how its weights are stored."
            ),
            inputs=[
                picker.path_input(io.Combo.Input(
                    "unet_name",
                    options=files or [NO_FILE],
                    default=likely,
                    tooltip=(
                        "A BluePixel transformer in models/diffusion_models: bluepixel_turbo_int8_convrot "
                        "(9.4 GB), _w6a8 (7.5 GB), _w4a8 (5.6 GB) or _bf16 (18 GB). A Diffusers single file "
                        "is converted on loading."
                    ),
                ), model=True),
                io.Combo.Input(
                    "weight_dtype", options=list(bluepixel.WEIGHT_DTYPES), default="default", advanced=True,
                    tooltip=(
                        "default = as stored; fp8_e4m3fn or fp8_e5m2 = store a bf16 file at 8 bits, half "
                        "the memory; fp8_e4m3fn_fast = also multiply at 8 bits on cards that can."
                    ),
                ),
            ],
            outputs=[
                io.Model.Output(
                    tooltip="BluePixel, for a CFG Guider at 3 with Text Encode BluePixel's conditioning and BluePixel Scheduler's sigmas.",
                ),
            ],
        )

    @classmethod
    def execute(cls, unet_name, weight_dtype="default") -> io.NodeOutput:
        """Load the transformer in ``unet_name``.

        Raises:
            ValueError: No file is chosen, or the file is not a BluePixel transformer.
            PathNotAllowed: A wired path lies outside the diffusion_models folders.
        """
        import os

        import folder_paths

        unet_name = picker.model_name(unet_name, (bluepixel.FOLDER,))
        if not unet_name or unet_name == NO_FILE:
            raise ValueError(
                f"{NODE_NAME} has no file to load. Put bluepixel_turbo_int8_convrot.safetensors or "
                "another BluePixel transformer in ComfyUI/models/diffusion_models, then refresh the "
                "node list."
            )
        path = folder_paths.get_full_path_or_raise(bluepixel.FOLDER, unet_name)
        patcher = bluepixel.load(path, weight_dtype)
        formats = bluepixel.weight_formats(path)
        stored = ", ".join(f"{count} {name}" for name, count in sorted(formats.items(), key=lambda item: -item[1])) or "not read"
        run_result.publish(
            status=run_result.OK,
            summary=f"{os.path.basename(unet_name)} is ready to edit",
            counts={
                "parameters (B)": round(bluepixel.parameters(patcher) / 1e9, 2),
                "file (GB)": round(os.path.getsize(path) / 1e9, 2),
            },
            facts={
                "model": "BluePixel-Turbo on FLUX.2 Klein 9B",
                "weights": stored,
                "storage": weight_dtype,
                "latent": f"{bluepixel.CHANNELS} channels at 1/{bluepixel.COMPRESSION}, scaled {bluepixel.SCALING_FACTOR}",
            },
        )
        return io.NodeOutput(patcher)
