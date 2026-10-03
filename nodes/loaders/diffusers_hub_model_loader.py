"""Downloading a diffusers model from the Hugging Face Hub, then loading it."""

from __future__ import annotations

import os
from pathlib import Path

from comfy_api.latest import io

from ...modules import deps, log
from ...modules.util import sandbox

REQUIRES = "diffusers"

logger = log.get_logger("nodes.loaders")

#: Config key of the feature group this node is gated on.
FEATURE = "features.diffusers"

#: File patterns the download skips. A single-file checkpoint and an ONNX export are both
#: unreadable by the diffusers loader, and either one can be most of the repository's size.
IGNORE_PATTERNS = ["*.ckpt", "*.onnx"]

#: Widget values that mean "no revision was given", so the repository's default branch is
#: fetched. The widget defaults to the string "None".
UNSET_REVISIONS = ("", "None", "none")


def download_directory(repo_id: str) -> str:
    """The absolute directory a repository is downloaded into.

    Args:
        repo_id: A Hugging Face repository id, ``owner/name``.

    Returns:
        ``<models/diffusers>/<repo_id>``.

    Raises:
        ValueError: The repository id is not ``owner/name``, no diffusers model directory is
            configured, or the id resolves outside it.
    """
    import folder_paths

    from ...modules.model import repository_id

    repo_id = repository_id(repo_id)
    roots = folder_paths.get_folder_paths("diffusers")
    if not roots:
        raise ValueError(
            "no models/diffusers directory is configured, so there is nowhere to download to"
        )
    root = os.path.abspath(roots[0])
    target = os.path.abspath(os.path.join(root, repo_id))
    if not sandbox.contains(Path(root).resolve(), Path(target).resolve()):
        raise ValueError(
            f"refusing to download {repo_id!r}, which resolves to {target}, outside the "
            f"diffusers model directory {root}"
        )
    return target


class DiffusersHubModelLoader(io.ComfyNode):
    """Fetch a diffusers model from the Hugging Face Hub and load it.

    The repository is downloaded into ``models/diffusers/<repo_id>`` and loaded from there.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Diffusers Hub Model Down-Loader",
            display_name="Diffusers Hub Model Down-Loader",
            search_aliases=[
                "Diffusers Hub Model Down-Loader",
                "huggingface",
                "download model",
                "diffusers",
            ],
            category="WAS Suite/Loaders",
            description=(
                "Download a diffusers model from the Hugging Face Hub into "
                "models/diffusers and load it. Enable features.diffusers to load this node."
            ),
            inputs=[
                io.String.Input(
                    "repo_id",
                    multiline=False,
                    tooltip=(
                        "The Hugging Face repository to fetch, owner and name, such as "
                        "'stabilityai/stable-diffusion-2-1'. It must be a diffusers-format "
                        "repository: one holding unet, vae and text_encoder folders. "
                        "Fetching needs features.network on in config.yaml; without it, a "
                        "repository already in models/diffusers still loads."
                    ),
                ),
                io.String.Input(
                    "revision",
                    default="None",
                    multiline=False,
                    tooltip=(
                        "Branch, tag or commit to fetch, such as 'fp16' or 'refs/pr/2'. "
                        "Leave it as 'None' for the repository's default branch, which then "
                        "means the files can change under a saved workflow; a commit hash "
                        "pins them."
                    ),
                ),
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
                        "The repository id that was loaded, for captions, file names and log "
                        "lines."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, repo_id=None, revision=None) -> io.NodeOutput:
        import comfy.diffusers_load
        import folder_paths

        from ...modules.model import ModelUnavailable, NETWORK_FEATURE, network_enabled

        repo_id = (repo_id or "").strip().strip("/")
        if not repo_id:
            raise ValueError("no repository id was given, so there is nothing to download")
        revision = revision.strip() if isinstance(revision, str) else revision
        if revision in UNSET_REVISIONS or revision is None:
            revision = None

        target = download_directory(repo_id)
        if revision is None and os.path.isfile(os.path.join(target, "model_index.json")):
            logger.info("%s is on disk at %s, so the hub is not contacted", repo_id, target)
        elif network_enabled():
            hub = deps.require("huggingface_hub", feature=FEATURE)
            logger.info("downloading %s into %s", repo_id, target)
            hub.snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                local_dir=target,
                revision=revision,
                token=False,
                ignore_patterns=IGNORE_PATTERNS,
            )
        elif os.path.isdir(target):
            logger.info(
                "%s is off, so %s is loaded from %s without contacting the hub",
                NETWORK_FEATURE, repo_id, target,
            )
        else:
            raise ModelUnavailable(
                f"{repo_id} is not on disk and {NETWORK_FEATURE} is off, so nothing was "
                f"downloaded.\n"
                f"    Turn {NETWORK_FEATURE} on in config.yaml to let this node fetch it,\n"
                f"    or put the repository at {target} yourself and run this node again."
            )

        model, clip, vae = comfy.diffusers_load.load_diffusers(
            target,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        return io.NodeOutput(model, clip, vae, repo_id)
