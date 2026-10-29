"""Read latents, image batches and conditioning back out of the cache directory."""

from __future__ import annotations

import re
from pathlib import Path

from comfy_api.latest import io

from ....modules import config, log
from ....modules.io import picker

REQUIRES = "cache"

logger = log.get_logger("nodes.legacy.utility")


#: What the menus say when the cache holds nothing of that kind.
NOTHING_CACHED = "nothing cached"

#: Oldest torch ``(major, minor)`` this node reads a cache file with.
SAFE_TORCH = (2, 6)


def torch_release(version: str) -> tuple[int, int]:
    """The ``(major, minor)`` release a torch version string names.

    Args:
        version: A version such as ``"2.6.0+cu124"`` or ``"2.13.0"``.

    Returns:
        ``(major, minor)``, or ``(0, 0)`` where the string names no release.
    """
    match = re.match(r"\s*(\d+)\.(\d+)", str(version or ""))
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)


def refuse_old_torch(version: str, path) -> None:
    """Refuse to read a cache file on a torch older than :data:`SAFE_TORCH`.

    Args:
        version: The running torch's ``__version__``.
        path: The cache file about to be read, named in the error.

    Raises:
        RuntimeError: The torch release is older than :data:`SAFE_TORCH`.
    """
    if torch_release(version) >= SAFE_TORCH:
        return
    wanted = ".".join(str(part) for part in SAFE_TORCH)
    raise RuntimeError(
        f"Load Cache refuses to read {path}: this ComfyUI runs torch {version}.\n"
        f"    Below torch {wanted}, a crafted cache file can run code while it is read, "
        f"even in the safe loading mode (CVE-2025-32434).\n"
        f"    Upgrade torch to {wanted} or newer in the Python this ComfyUI runs on, then "
        f"run the node again."
    )


def cache_roots() -> list[tuple[str, str]]:
    """The pack's cache directory, as the one extra folder these menus offer."""
    return [("cache", cache_directory())]


def cache_options(extensions) -> list[str]:
    """The menu's entries for one kind of cache file, with an empty choice first."""
    return [NOTHING_CACHED, *picker.labels(extensions, extra=cache_roots())]


def chosen_cache(value: str, suffix: str):
    """The cache file one menu entry or wired path names, resolved, or None where nothing was chosen."""
    entry = str(value or "").strip()
    if not entry or entry == NOTHING_CACHED:
        return None
    wired = picker.wired_file(entry, (suffix,), roots=[cache_directory()])
    if wired is not None:
        return wired
    found = picker.resolve(entry, (suffix,), extra=cache_roots())
    return Path(found) if found else None


def cache_directory() -> str:
    """The directory cache files are written to, under the pack's config directory."""
    return str(config.config_directory() / "cache")


class LoadCache(io.ComfyNode):
    """Read back what `Cache Node` wrote."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Load Cache",
            display_name="Load Cache",
            search_aliases=["Load Cache", "load latent", "cache"],
            category="WAS Suite/IO",
            description=(
                "Deprecated, with no direct replacement. Reads back a latent, image batch or "
                "conditioning that Cache Node wrote in an earlier prompt; a file holding "
                "anything but tensor data is refused, and reading needs torch 2.6 or newer. "
                "Enable legacy.cache to load it."
            ),
            inputs=[
                picker.path_input(io.Combo.Input(
                    "latent",
                    options=cache_options((".latent",)),
                    optional=True,
                    tooltip=(
                        "Which '.latent' file to read, as Cache Node named it. The menu "
                        "lists the pack's cache directory, tagged '[cache]', beside "
                        "ComfyUI's own folders. Left on the empty entry, the LATENT "
                        "output is nothing at all."
                    ),
                )),
                picker.path_input(io.Combo.Input(
                    "image",
                    options=cache_options((".image",)),
                    optional=True,
                    tooltip=(
                        "Which '.image' file to read, as Cache Node named it. The menu "
                        "lists the pack's cache directory, tagged '[cache]', beside "
                        "ComfyUI's own folders. Left on the empty entry, the IMAGE "
                        "output is nothing at all."
                    ),
                )),
                picker.path_input(io.Combo.Input(
                    "conditioning",
                    options=cache_options((".conditioning",)),
                    optional=True,
                    tooltip=(
                        "Which '.conditioning' file to read, as Cache Node named it. The menu "
                        "lists the pack's cache directory, tagged '[cache]', beside "
                        "ComfyUI's own folders. Left on the empty entry, the CONDITIONING "
                        "output is nothing at all."
                    ),
                )),
            ],
            outputs=[
                io.Latent.Output(
                    display_name="LATENT",
                    tooltip=(
                        "The latent read from the file the latent menu names. Nothing at all "
                        "when the menu is on its empty entry or the file is missing, which "
                        "will fail whatever it is connected to."
                    ),
                ),
                io.Image.Output(
                    display_name="IMAGE",
                    tooltip=(
                        "The image batch read from the file the image menu names. Nothing at "
                        "all when the menu is on its empty entry or the file is missing."
                    ),
                ),
                io.Conditioning.Output(
                    display_name="CONDITIONING",
                    tooltip=(
                        "The conditioning read from the file the conditioning menu names. "
                        "Nothing at all when the menu is on its empty entry or the file is "
                        "missing."
                    ),
                ),
            ],
            is_deprecated=True,
        )

    @classmethod
    def execute(cls, latent="", image="", conditioning="") -> io.NodeOutput:
        import torch

        loaded = []
        for value, suffix in (
            (latent, ".latent"), (image, ".image"), (conditioning, ".conditioning")
        ):
            path = chosen_cache(value, suffix)
            if path is None:
                loaded.append(None)
            elif path.is_file():
                refuse_old_torch(torch.__version__, path.name)
                loaded.append(torch.load(path, map_location="cpu", weights_only=True))
            else:
                logger.error("unable to locate cache file %s", path.name)
                loaded.append(None)
        return io.NodeOutput(*loaded)
