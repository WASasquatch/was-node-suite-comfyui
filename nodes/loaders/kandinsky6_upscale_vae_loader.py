"""Loading the Kandinsky 6 video super-resolution K-VAE."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.interface import run_result
from ...modules.io import picker
from ...modules.model.kandinsky6 import sr

NODE_NAME = "Kandinsky 6 Upscale VAE Loader"

#: Shown in the file list when the vae folders hold nothing.
NO_FILE = "put the Kandinsky 6 VSR K-VAE in models/vae"


def offered() -> tuple[list[str], str | None]:
    """The files in the vae folders, and the one most likely to be the K-VAE."""
    import folder_paths

    files = list(folder_paths.get_filename_list(sr.VAE_FOLDER))
    likely = [name for name in files if "kvae" in name.lower() or ("vsr" in name.lower() and "kandinsky" in name.lower())]
    return files, (likely[0] if likely else None)


class Kandinsky6UpscaleVAELoader(io.ComfyNode):
    """Load the Kandinsky 6 K-VAE as a VAE."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        files, likely = offered()
        return io.Schema(
            node_id="WASKandinsky6UpscaleVAELoader",
            display_name=NODE_NAME,
            search_aliases=[
                "WASKandinsky6UpscaleVAELoader", NODE_NAME,
                "kvae", "k-vae", "kandinsky vae", "kandinsky vsr vae", "load kvae",
            ],
            category="WAS Suite/Loaders",
            description=(
                "Load the K-VAE that Kandinsky 6 video super-resolution reads and writes video through. "
                "Wire it into Kandinsky 6 Upscale Video; VAE Encode and VAE Decode take it too. The panel "
                "shows the file's size and how the autoencoder runs."
            ),
            inputs=[
                picker.path_input(io.Combo.Input(
                    "vae_name",
                    options=files or [NO_FILE],
                    default=likely,
                    tooltip=(
                        "The K-VAE in models/vae: kandinsky6_vsr_kvae_bf16.safetensors (3.5 GB), or the vae "
                        "file of a kandinskylab/Kandinsky-6.0-VSR repository (7 GB, float32)."
                    ),
                ), model=True),
            ],
            outputs=[
                io.Vae.Output(
                    tooltip="The K-VAE: 64 channels at a 16th of the width and height and a quarter of the frames.",
                ),
            ],
        )

    @classmethod
    def execute(cls, vae_name) -> io.NodeOutput:
        """Build the K-VAE from ``vae_name``.

        Raises:
            ValueError: No file is chosen, or the file is not the K-VAE.
            PathNotAllowed: A wired path lies outside the vae folders.
        """
        import os

        import folder_paths

        vae_name = picker.model_name(vae_name, (sr.VAE_FOLDER,))
        if not vae_name or vae_name == NO_FILE:
            raise ValueError(
                f"{NODE_NAME} has no file to load. Put kandinsky6_vsr_kvae_bf16.safetensors in "
                "ComfyUI/models/vae, then refresh the node list."
            )
        path = folder_paths.get_full_path_or_raise(sr.VAE_FOLDER, vae_name)
        vae = sr.load_vae(path)
        run_result.publish(
            status=run_result.OK,
            summary=f"{os.path.basename(vae_name)} is ready",
            counts={"file (GB)": round(os.path.getsize(path) / 1e9, 2)},
            facts={
                "latent": f"{sr.CHANNELS} channels at 1/{sr.SPATIAL_FACTOR}, 1/{sr.TEMPORAL_FACTOR} of the frames",
                "runs in": str(vae.vae_dtype).replace("torch.", ""),
            },
        )
        return io.NodeOutput(vae)
