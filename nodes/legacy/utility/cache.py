"""Write latents, image batches and conditioning to disk between prompts."""

from __future__ import annotations

import os
import random

from comfy_api.latest import io

from ....modules import config, log
from ....modules.io import rooted
from ....modules.util import sandbox

REQUIRES = "cache"

logger = log.get_logger("nodes.legacy.utility")


#: The name the root widget gives the pack's own cache directory.
CACHE = "cache"


def cache_directory() -> str:
    """The directory cache files are written to, under the pack's config directory."""
    return str(config.config_directory() / "cache")


def suffix_default() -> str:
    """A fresh file-name suffix: a random six to eight digit number and ``_cache``."""
    # The range and the trailing `_cache` match the v2 widget default.
    return str(random.randint(999999, 99999999)) + "_cache"


def cache_root() -> tuple[tuple[str, str], ...]:
    """The pack's cache directory, as the one extra root these widgets offer."""
    return ((CACHE, cache_directory()),)


def cache_name(suffix: str, extension: str) -> str:
    """``<suffix>.<extension>``, rejected if the suffix carries a directory.

    Raises:
        ValueError: The suffix holds a path separator, which would place the file
            somewhere other than the directory it was meant for.
    """
    name = f"{suffix}{extension}"
    if os.path.basename(name) != name or name != name.strip():
        raise ValueError(f"`{suffix}` is not a file name")
    return name


class CacheNode(io.ComfyNode):
    """Write the connected latent, image batch and conditioning to the cache directory."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Cache Node",
            display_name="Cache Node",
            search_aliases=["Cache Node", "cache latent", "save latent"],
            category="WAS Suite/IO",
            description=(
                "Deprecated. Nothing replaces it directly: it writes a latent, image batch "
                "or conditioning to a file between prompts, and Load Cache reads it back. "
                "Off by default: enable legacy.cache to load it. Each suffix defaults to a "
                "fresh random number followed by '_cache', so two Cache Nodes do not "
                "overwrite each other; set one by hand to write a predictable name a Load "
                "Cache node can be pointed at."
            ),
            inputs=[
                io.String.Input(
                    "latent_suffix",
                    default=suffix_default(),
                    multiline=False,
                    tooltip=(
                        "Name for the latent's file, without an extension, '.latent' is "
                        "added. It has to be a plain file name with no folders in it."
                    ),
                ),
                io.String.Input(
                    "image_suffix",
                    default=suffix_default(),
                    multiline=False,
                    tooltip=(
                        "Name for the image batch's file, without an extension, '.image' is "
                        "added. Otherwise as latent_suffix."
                    ),
                ),
                io.String.Input(
                    "conditioning_suffix",
                    default=suffix_default(),
                    multiline=False,
                    tooltip=(
                        "Name for the conditioning's file, without an extension, "
                        "'.conditioning' is added. Otherwise as latent_suffix."
                    ),
                ),
                io.Combo.Input(
                    "root",
                    options=rooted.options(cache_root()),
                    optional=True,
                    tooltip=(
                        "Which folder the files land in: 'cache', the pack's own cache "
                        "directory, ComfyUI's 'output' or 'temp', or any folder added under "
                        "paths.allow_write in config.yaml. folder names the part below it."
                    ),
                ),
                io.String.Input(
                    "folder",
                    default="",
                    multiline=False,
                    optional=True,
                    tooltip=(
                        "Folder below the root the files land in, created if it is not "
                        "there. Tokens expand, so '[time(%Y-%m-%d)]' files each day's cache "
                        "under a dated folder. Empty writes into the root itself."
                    ),
                ),
                io.Latent.Input(
                    "latent",
                    optional=True,
                    tooltip=(
                        "A latent to write out. Disconnected, no latent file is written and "
                        "latent_filename comes back empty."
                    ),
                ),
                io.Image.Input(
                    "image",
                    optional=True,
                    tooltip=(
                        "An image batch to write out. Disconnected, no image file is written "
                        "and image_filename comes back empty."
                    ),
                ),
                io.Conditioning.Input(
                    "conditioning",
                    optional=True,
                    tooltip=(
                        "Conditioning to write out. Disconnected, no conditioning file is "
                        "written and conditioning_filename comes back empty."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="latent_filename",
                    tooltip=(
                        "Name of the latent file that was written, to paste into a Load Cache "
                        "node. Empty when no latent was connected."
                    ),
                ),
                io.String.Output(
                    display_name="image_filename",
                    tooltip=(
                        "Name of the image file that was written. Empty when no image was "
                        "connected."
                    ),
                ),
                io.String.Output(
                    display_name="conditioning_filename",
                    tooltip=(
                        "Name of the conditioning file that was written. Empty when no "
                        "conditioning was connected."
                    ),
                ),
            ],
            is_output_node=True,
            is_deprecated=True,
        )

    @classmethod
    def execute(
        cls,
        latent_suffix="_cache",
        image_suffix="_cache",
        conditioning_suffix="_cache",
        root=CACHE,
        folder="",
        latent=None,
        image=None,
        conditioning=None,
    ) -> io.NodeOutput:
        import torch

        directory = rooted.destination(root, folder, cache_root())
        os.makedirs(directory, exist_ok=True)

        written = {"latent": "", "image": "", "conditioning": ""}
        for kind, payload, suffix in (
            ("latent", latent, latent_suffix),
            ("image", image, image_suffix),
            ("conditioning", conditioning, conditioning_suffix),
        ):
            if payload is None:
                continue
            name = cache_name(suffix, f".{kind}")
            out_file = sandbox.resolve_write_file(directory, name)
            torch.save(payload, out_file)
            written[kind] = name
            logger.info("%s saved to: %s", kind, out_file)

        return io.NodeOutput(written["latent"], written["image"], written["conditioning"])
