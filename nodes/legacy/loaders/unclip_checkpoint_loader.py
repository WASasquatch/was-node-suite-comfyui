"""unCLIP checkpoint loading, plus the checkpoint's name."""

from __future__ import annotations

import os

from comfy_api.latest import io

from ....modules.io import picker

REQUIRES = "loaders"


def checkpoint_names() -> list[str]:
    """The checkpoints this install offers."""
    import folder_paths

    return folder_paths.get_filename_list("checkpoints")


class UnCLIPCheckpointLoader(io.ComfyNode):
    """Load an unCLIP checkpoint, including its CLIP vision tower, and report its name."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="unCLIP Checkpoint Loader",
            display_name="unCLIP Checkpoint Loader (Advanced)",
            search_aliases=[
                "unCLIP Checkpoint Loader",
                "unclip",
                "checkpoint",
                "clip vision",
            ],
            category="WAS Suite/Loaders",
            description=(
                "Deprecated: use ComfyUI's unCLIP Checkpoint Loader instead. Loads an "
                "unCLIP checkpoint and returns the model, CLIP, VAE and CLIP vision tower, "
                "plus the checkpoint's file name as a string. That name is already on the "
                "ckpt_name widget."
            ),
            inputs=[
                picker.path_input(io.Combo.Input(
                    "ckpt_name",
                    options=checkpoint_names(),
                    tooltip=(
                        "The checkpoint file in models/checkpoints to load. An unCLIP "
                        "checkpoint is needed for the CLIP_VISION output to carry anything."
                    ),
                ), model=True),
            ],
            outputs=[
                io.Model.Output(
                    display_name="MODEL",
                    tooltip="The diffusion model, for a sampler.",
                ),
                io.Clip.Output(
                    display_name="CLIP",
                    tooltip="The text encoder, for the prompt encoding nodes.",
                ),
                io.Vae.Output(
                    display_name="VAE",
                    tooltip="The autoencoder that turns a latent into an image.",
                ),
                io.ClipVision.Output(
                    display_name="CLIP_VISION",
                    tooltip=(
                        "The image encoder an unCLIP conditioning node needs. Empty when "
                        "the checkpoint is not an unCLIP model."
                    ),
                ),
                io.String.Output(
                    display_name="NAME_STRING",
                    tooltip=(
                        "The checkpoint's file name without its folder or extension, for "
                        "captions, file names and log lines."
                    ),
                ),
            ],
            is_deprecated=True,
        )

    @classmethod
    def execute(cls, ckpt_name) -> io.NodeOutput:
        import comfy.sd
        import folder_paths

        ckpt_name = picker.model_name(ckpt_name, ("checkpoints",))
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            output_clipvision=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        name = os.path.splitext(os.path.basename(ckpt_name))[0]
        return io.NodeOutput(out[0], out[1], out[2], out[3], name)
