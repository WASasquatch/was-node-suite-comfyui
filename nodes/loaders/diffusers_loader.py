"""Diffusers-format checkpoint loading from the models/diffusers directory."""

from __future__ import annotations

import os

from comfy_api.latest import io

from ...modules import log
from ...modules.io import picker
from ...modules.util import sandbox

REQUIRES = "diffusers"

logger = log.get_logger("nodes.loaders")

#: ``((path, mtime), ...)`` the model listing was built from, and the listing itself.
_listing: tuple[tuple[tuple[str, float], ...], list[str]] | None = None


def _stamps(roots: list[str]) -> tuple[tuple[str, float], ...]:
    """``(path, mtime)`` for each search directory that exists, in search order.

    A path that cannot be stat'ed is left out.
    """
    stamps = []
    for path in roots:
        try:
            stamps.append((path, os.path.getmtime(path)))
        except OSError:
            # A configured model directory that was never created is the normal state of
            # most installs, so it is skipped rather than raised on.
            continue
    return tuple(stamps)


def model_paths() -> list[str]:
    """The directory names under every configured models/diffusers path."""
    global _listing

    import folder_paths

    signature = _stamps(folder_paths.get_folder_paths("diffusers"))
    # define_schema runs again for every /object_info request, so an unmemoized directory
    # scan here is paid on every browser refresh.
    if _listing is not None and _listing[0] == signature:
        return list(_listing[1])

    # A diffusers model is a directory of subdirectories rather than a single file, so
    # folder_paths.get_filename_list, which lists files by extension, cannot see one.
    names: list[str] = []
    for path, _ in signature:
        try:
            with os.scandir(path) as entries:
                names += sorted(entry.name for entry in entries if entry.is_dir())
        except OSError as error:
            logger.warning("the diffusers model directory %s could not be listed: %s", path, error)
    _listing = (signature, names)
    return list(names)


def resolve_model_path(model_path: str) -> str:
    """The absolute directory a ``model_path`` widget value names.

    Args:
        model_path: A directory name offered by :func:`model_paths`, a folder below one
            such as ``org/model``, or a full path inside a models/diffusers directory.

    Returns:
        The first search directory holding it.

    Raises:
        ValueError: No models/diffusers directory holds a model folder of that name, which
            is what a workflow saved against a model since removed or renamed carries.
        PathNotAllowed: A full path lies outside every models/diffusers directory.
    """
    import folder_paths

    model_path = picker.model_name(model_path, ("diffusers",), folder=True)
    if model_path.strip(" ./\\") and sandbox.leaves_folder(model_path) is None:
        for root in folder_paths.get_folder_paths("diffusers"):
            candidate = os.path.join(root, model_path)
            if os.path.isdir(candidate):
                return candidate
    searched = ", ".join(folder_paths.get_folder_paths("diffusers")) or "no directory"
    raise ValueError(
        f"there is no diffusers model directory named {model_path!r}; searched {searched}"
    )


class DiffusersLoader(io.ComfyNode):
    """Load a diffusers-format model directory as a MODEL, CLIP and VAE.

    A diffusers model is a directory of ``unet``, ``vae`` and ``text_encoder`` subdirectories.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Diffusers Model Loader",
            display_name="Diffusers Model Loader",
            search_aliases=["Diffusers Model Loader", "diffusers", "huggingface", "load model"],
            category="WAS Suite/Loaders",
            description=(
                "Load a diffusers-format model directory from models/diffusers and emit its "
                "name alongside the model, CLIP and VAE. Enable features.diffusers to load "
                "this node."
            ),
            inputs=[
                picker.path_input(io.Combo.Input(
                    "model_path",
                    options=model_paths(),
                    tooltip=(
                        "The model folder in models/diffusers to load, holding unet, vae and "
                        "text_encoder subfolders as a cloned Hugging Face model repository does."
                    ),
                ), folder=True, model=True),
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
                        "The name of the directory the model was loaded from, for captions, "
                        "file names and log lines."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, model_path) -> io.NodeOutput:
        import comfy.diffusers_load
        import folder_paths

        resolved = resolve_model_path(model_path)
        model, clip, vae = comfy.diffusers_load.load_diffusers(
            resolved,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        return io.NodeOutput(model, clip, vae, os.path.basename(resolved))
