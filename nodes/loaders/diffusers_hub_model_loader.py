"""Downloading a diffusers model from the Hugging Face Hub, then loading it."""

from __future__ import annotations

import os
import re
from pathlib import Path

from comfy_api.latest import io

from ...modules import deps, log
from ...modules.util import sandbox

REQUIRES = "diffusers"

logger = log.get_logger("nodes.loaders")

#: Config key of the feature group this node is gated on.
FEATURE = "features.diffusers"

#: Repository files the download fetches: configuration and tokenizer text, and the
#: safetensors weights of the folders the loader reads.
ALLOW_PATTERNS = [
    "*.json",
    "*.txt",
    "unet/*.safetensors",
    "vae/*.safetensors",
    "text_encoder/*.safetensors",
    "text_encoder_2/*.safetensors",
]

#: Widget values that select the repository's default branch. The widget defaults to the
#: string "None".
DEFAULT_REVISIONS = ("", "None", "none", "main")

#: A pinned revision: a full commit hash.
COMMIT_HASH = re.compile(r"[0-9a-fA-F]{40}")

#: Weight files the loader tries in a ``unet`` or ``vae`` folder, the first present winning.
DIFFUSION_WEIGHTS = (
    "diffusion_pytorch_model.fp16.safetensors",
    "diffusion_pytorch_model.safetensors",
    "diffusion_pytorch_model.fp16.bin",
    "diffusion_pytorch_model.bin",
)

#: Weight files the loader tries in a text encoder folder, the first present winning.
TEXT_ENCODER_WEIGHTS = (
    "model.fp16.safetensors",
    "model.safetensors",
    "pytorch_model.fp16.bin",
    "pytorch_model.bin",
)

#: ``(folder, weight file names, required)`` for every folder the loader reads.
COMPONENTS = (
    ("unet", DIFFUSION_WEIGHTS, True),
    ("vae", DIFFUSION_WEIGHTS, True),
    ("text_encoder", TEXT_ENCODER_WEIGHTS, True),
    ("text_encoder_2", TEXT_ENCODER_WEIGHTS, False),
)


def pinned_commit(revision) -> str | None:
    """The commit a revision widget names, or None for the default branch.

    Args:
        revision: The raw widget value.

    Returns:
        A lowercase 40-character commit hash, or None for an empty value, ``None`` or
        ``main``.

    Raises:
        ValueError: The value names a branch, tag or pull request other than ``main``.
    """
    text = "" if revision is None else str(revision).strip()
    if text in DEFAULT_REVISIONS:
        return None
    if COMMIT_HASH.fullmatch(text):
        return text.lower()
    raise ValueError(
        f"revision {text!r} is not a commit hash, so nothing was downloaded.\n"
        f"    This node fetches the default branch ('None' or 'main') or one commit named by "
        f"its full 40-character hash. A branch, tag or pull request such as 'fp16' or "
        f"'refs/pr/2' can change after a workflow is saved, and is refused.\n"
        f"    Copy the commit's full hash from the repository's history on huggingface.co, "
        f"or set revision back to 'None'."
    )


def download_directory(repo_id: str, commit: str | None = None) -> str:
    """The absolute directory a repository is downloaded into.

    Args:
        repo_id: A Hugging Face repository id, ``owner/name``.
        commit: A 40-character commit hash from :func:`pinned_commit`, or None for the
            default branch.

    Returns:
        ``<models/diffusers>/<repo_id>``, or ``<models/diffusers>/<repo_id>@<commit>`` for
        a pinned commit.

    Raises:
        ValueError: The repository id is not ``owner/name``, the commit is not a hash, no
            diffusers model directory is configured, or the id resolves outside it.
    """
    import folder_paths

    from ...modules.model import repository_id

    repo_id = repository_id(repo_id)
    if commit is not None and not COMMIT_HASH.fullmatch(commit):
        raise ValueError(f"{commit!r} is not a 40-character commit hash")
    roots = folder_paths.get_folder_paths("diffusers")
    if not roots:
        raise ValueError(
            "no models/diffusers directory is configured, so there is nowhere to download to"
        )
    root = os.path.abspath(roots[0])
    name = repo_id if commit is None else f"{repo_id}@{commit}"
    target = os.path.abspath(os.path.join(root, name))
    if not sandbox.contains(Path(root).resolve(), Path(target).resolve()):
        raise ValueError(
            f"refusing to download {repo_id!r}, which resolves to {target}, outside the "
            f"diffusers model directory {root}"
        )
    return target


def safetensors_weights(directory: str) -> list[str]:
    """The weight file the loader reads from each folder, every one a safetensors file.

    Args:
        directory: A diffusers repository on disk.

    Returns:
        The absolute weight files, one per folder that holds one.

    Raises:
        ValueError: ``unet``, ``vae`` or ``text_encoder`` holds no weight file, or the file
            the loader would read from any folder is not ``.safetensors``.
    """
    found, problems = [], []
    for folder, names, required in COMPONENTS:
        candidates = (os.path.join(directory, folder, name) for name in names)
        chosen = next((path for path in candidates if os.path.exists(path)), None)
        if chosen is None:
            if required:
                problems.append(f"{folder} holds no weight file")
            continue
        if not chosen.endswith(".safetensors") or not os.path.isfile(chosen):
            problems.append(f"{folder} holds {os.path.basename(chosen)}, not .safetensors")
            continue
        found.append(chosen)
    if problems:
        raise ValueError(
            f"refusing to load {directory}: {'; '.join(problems)}.\n"
            f"    This node loads .safetensors weights only, from the unet, vae and "
            f"text_encoder folders.\n"
            f"    Pick a repository that publishes them, or remove {directory} so the next "
            f"run fetches it again."
        )
    return found


class DiffusersHubModelLoader(io.ComfyNode):
    """Fetch a diffusers model from the Hugging Face Hub and load it.

    Files land in ``models/diffusers/<repo_id>``, or ``<repo_id>@<commit>`` for a pinned
    commit.
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
                        "repository whose unet, vae and text_encoder folders hold "
                        ".safetensors weights. Fetching needs features.network on in config.yaml; without it, a "
                        "repository already in models/diffusers still loads."
                    ),
                ),
                io.String.Input(
                    "revision",
                    default="None",
                    multiline=False,
                    tooltip=(
                        "'None' or 'main' = the default branch, which can change under a "
                        "saved workflow; a full 40-character commit hash = that commit, "
                        "pinned, in its own '<name>@<hash>' folder. Other branches, tags and "
                        "pull requests, such as 'fp16' or 'refs/pr/2', are refused."
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
        commit = pinned_commit(revision)

        target = download_directory(repo_id, commit)
        if os.path.isfile(os.path.join(target, "model_index.json")):
            logger.info("%s is on disk at %s, so the hub is not contacted", repo_id, target)
        elif network_enabled():
            hub = deps.require("huggingface_hub", feature=FEATURE)
            logger.info("downloading %s into %s", repo_id, target)
            hub.snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                local_dir=target,
                revision=commit,
                token=False,
                allow_patterns=ALLOW_PATTERNS,
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

        safetensors_weights(target)
        model, clip, vae = comfy.diffusers_load.load_diffusers(
            target,
            output_vae=True,
            output_clip=True,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        return io.NodeOutput(model, clip, vae, repo_id)
