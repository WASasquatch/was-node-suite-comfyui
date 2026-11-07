"""Loading DC-AE f64c128, the autoencoder BluePixel edits on."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.interface import run_result
from ...modules.io import picker
from ...modules.model.bluepixel import vae as dcae

NODE_NAME = "BluePixel VAE Loader"

#: Shown in the file list when the vae folders hold nothing.
NO_FILE = "put the DC-AE f64c128 file in models/vae"


class BluePixelVAELoader(io.ComfyNode):
    """Load DC-AE f64c128 as the VAE BluePixel encodes and decodes with."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        files, likely = dcae.offered()
        return io.Schema(
            node_id="WASBluePixelVAELoader",
            display_name=NODE_NAME,
            search_aliases=[
                "WASBluePixelVAELoader", NODE_NAME,
                "bluepixel", "dc-ae", "dcae", "f64c128", "deep compression autoencoder", "load bluepixel vae",
            ],
            category="WAS Suite/Loaders",
            description=(
                "Load DC-AE f64c128, the autoencoder BluePixel edits on, as a VAE for Text Encode "
                "BluePixel, VAE Encode, VAE Decode and VAE Decode (Tiled). One latent unit covers 64 "
                "by 64 pixels in 128 channels."
            ),
            inputs=[
                picker.path_input(io.Combo.Input(
                    "vae_name",
                    options=files or [NO_FILE],
                    default=likely,
                    tooltip=(
                        "The DC-AE f64c128 file in models/vae, such as dc-ae-f64c128-mix-1.0.safetensors: "
                        "vae/diffusion_pytorch_model.safetensors from openvivo/BluePixel-Turbo."
                    ),
                ), model=True),
            ],
            outputs=[
                io.Vae.Output(
                    tooltip=(
                        "The autoencoder, 64 times smaller on each side. A 4096 square decodes in about "
                        "12 GB at bf16; VAE Decode tiles when that does not fit."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, vae_name) -> io.NodeOutput:
        """Build the autoencoder from ``vae_name``.

        Raises:
            ValueError: No file is chosen, or the file is not DC-AE f64c128.
            PathNotAllowed: A wired path lies outside the vae folders.
        """
        import os

        import folder_paths

        vae_name = picker.model_name(vae_name, (dcae.FOLDER,))
        if not vae_name or vae_name == NO_FILE:
            raise ValueError(
                f"{NODE_NAME} has no file to load. Download vae/diffusion_pytorch_model.safetensors "
                "from openvivo/BluePixel-Turbo into ComfyUI/models/vae, then refresh the node list."
            )
        path = folder_paths.get_full_path_or_raise(dcae.FOLDER, vae_name)
        vae = dcae.load(path)
        run_result.publish(
            status=run_result.OK,
            summary=f"{os.path.basename(vae_name)} is ready",
            counts={
                "parameters (M)": round(sum(t.numel() for t in vae.first_stage_model.state_dict().values()) / 1e6),
                "file (GB)": round(os.path.getsize(path) / 1e9, 2),
            },
            facts={
                "autoencoder": "DC-AE f64c128",
                "latent": f"{dcae.CHANNELS} channels at 1/{dcae.COMPRESSION}",
                "working dtype": str(vae.vae_dtype).replace("torch.", ""),
                "device": str(vae.device),
            },
        )
        return io.NodeOutput(vae)
