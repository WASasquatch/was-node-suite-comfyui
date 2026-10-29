"""Checkpoint loading against an explicit model config, plus the checkpoint's name."""

from __future__ import annotations

import os

from comfy_api.latest import io

from ....modules.io import picker

REQUIRES = "loaders"


def config_names() -> list[str]:
    """The model config files this install offers."""
    import folder_paths

    return folder_paths.get_filename_list("configs")


def checkpoint_names() -> list[str]:
    """The checkpoints this install offers. Read live, as in :func:`config_names`."""
    import folder_paths

    return folder_paths.get_filename_list("checkpoints")


class CheckpointLoader(io.ComfyNode):
    """Load a checkpoint described by a model config file, and report its name."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Checkpoint Loader",
            display_name="Checkpoint Loader (Advanced)",
            search_aliases=["Checkpoint Loader", "checkpoint", "ckpt", "load model"],
            category="WAS Suite/Loaders",
            description=(
                "Deprecated: use ComfyUI's Load Checkpoint instead, which reads the "
                "architecture out of the weights rather than asking for a config file. "
                "Loads a checkpoint against a chosen model config and returns the model, "
                "CLIP and VAE, plus the checkpoint's file name as a string. That name is "
                "already on the ckpt_name widget."
            ),
            inputs=[
                picker.path_input(io.Combo.Input(
                    "config_name",
                    options=config_names(),
                    tooltip=(
                        "The .yaml config in models/configs describing the checkpoint's "
                        "architecture. Only original Stable Diffusion 1.x and 2.x weights "
                        "need one; ComfyUI's Load Checkpoint loads anything newer."
                    ),
                ), model=True),
                picker.path_input(io.Combo.Input(
                    "ckpt_name",
                    options=checkpoint_names(),
                    tooltip="The checkpoint file in models/checkpoints to load.",
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
    def execute(cls, config_name, ckpt_name) -> io.NodeOutput:
        import comfy.sd
        import folder_paths

        config_name = picker.model_name(config_name, ("configs",))
        ckpt_name = picker.model_name(ckpt_name, ("checkpoints",))
        config_path = folder_paths.get_full_path_or_raise("configs", config_name)
        ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
        model, clip, vae = comfy.sd.load_checkpoint(
            config_path,
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        name = os.path.splitext(os.path.basename(ckpt_name))[0]
        return io.NodeOutput(model, clip, vae, name)
