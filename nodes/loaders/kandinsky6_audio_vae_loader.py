"""Loading the Kandinsky 6 audio VAE."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.model.kandinsky6 import audio_vae

NODE_NAME = "Kandinsky 6 Audio VAE Loader"

#: Shown in the file list when the vae folders hold nothing, so the widget has something to draw.
NO_FILE = "put the Kandinsky 6 audio_vae file in models/vae"


class Kandinsky6AudioVAELoader(io.ComfyNode):
    """Load the Kandinsky 6 audio autoencoder and vocoder as a VAE."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        files, likely = audio_vae.offered()
        return io.Schema(
            node_id="WASKandinsky6AudioVAELoader",
            display_name=NODE_NAME,
            search_aliases=[
                "WASKandinsky6AudioVAELoader", NODE_NAME,
                "kandinsky audio vae",
                "k6 audio",
                "mmaudio",
                "bigvgan",
                "load audio vae",
            ],
            category="WAS Suite/Loaders",
            description=(
                "Load the Kandinsky 6 audio VAE, which turns the sound half of a Kandinsky 6 latent "
                "into a waveform through VAE Decode Audio. The one file holds the autoencoder and "
                "its vocoder."
            ),
            inputs=[
                io.Combo.Input(
                    "vae_name",
                    options=files or [NO_FILE],
                    default=likely,
                    tooltip=(
                        "The audio_vae file from a Kandinsky-6.0 repository, saved under models/vae, "
                        "as kandinsky6_audio_vae.safetensors. One file serves every Kandinsky 6 "
                        "checkpoint."
                    ),
                ),
            ],
            outputs=[
                io.Vae.Output(
                    tooltip="The audio VAE, for VAE Decode Audio and VAE Encode Audio. 44.1 kHz, mono.",
                ),
            ],
        )

    @classmethod
    def execute(cls, vae_name) -> io.NodeOutput:
        """Build the audio VAE from ``vae_name``.

        Raises:
            ValueError: No file is chosen, or the file is not the Kandinsky 6 audio VAE.
        """
        if not vae_name or vae_name == NO_FILE:
            raise ValueError(
                f"{NODE_NAME} has no file to load. Download audio_vae/diffusion_pytorch_model."
                "safetensors from a kandinskylab/Kandinsky-6.0 repository into ComfyUI/models/vae, "
                "then refresh the node list."
            )
        return io.NodeOutput(audio_vae.load(vae_name))
