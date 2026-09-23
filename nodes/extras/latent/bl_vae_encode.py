"""VAE encode that can keep its latent inside the workflow it was saved with."""

from __future__ import annotations

import base64
import hashlib
import inspect
import json
import zlib

import torch
from comfy_api.latest import io

from ....modules.compat.sockets import require_input
from ....modules.log import get_logger

REQUIRES = "extras"

logger = get_logger("nodes.extras.latent")

#: How far neighbouring tiles overlap during a tiled encode, in pixels. The value
#: ComfyUI's own tiled encode offers, so a tiled result here matches one taken from that
#: node at its defaults.
TILE_OVERLAP = 64

#: ``graph node id -> digest of the image that node last encoded``, for the lifetime of the
#: process. A bundled latent from an earlier run is stale the moment an image is connected,
#: and is replaced rather than reloaded. ``execute`` runs on a per-execution clone of
#: the class which is discarded afterwards, so a module attribute is what has the right
#: lifetime. Nothing is written to disk: the bundle itself lives in the workflow.
_last_image_hash: dict[str, str] = {}


def sha256_tensor(tensor) -> str:
    """Digest a tensor's contents.

    Args:
        tensor: Any tensor. It is moved to the CPU and made contiguous first, so the digest
            covers the values rather than the memory layout they happened to arrive in.

    Returns:
        The SHA-256 digest as a lowercase hex string.
    """
    tensor_bytes = tensor.cpu().contiguous().numpy().tobytes()
    hash_obj = hashlib.sha256()
    hash_obj.update(tensor_bytes)
    return hash_obj.hexdigest()


def serialize(obj) -> str:
    """Pack a latent into one text string a workflow can carry.

    Args:
        obj: The latent dict, or anything else built from tensors, dicts and lists.

    Returns:
        The base64 text.
    """
    json_str = json.dumps(
        obj,
        default=lambda o: (
            {"__tensor__": True, "value": o.cpu().numpy().tolist()}
            if torch.is_tensor(o)
            else o.__dict__
        ),
    )
    compressed_data = zlib.compress(json_str.encode("utf-8"))
    return base64.b64encode(compressed_data).decode("utf-8")


def deserialize(base64_str: str):
    """Unpack what :func:`serialize` produced.

    Args:
        base64_str: The base64 text read back out of the workflow.

    Returns:
        The original structure, with every marked entry rebuilt as a CPU tensor.

    Raises:
        Exception: The text is not valid base64, does not inflate, or is not JSON, which
            is what a truncated or edited metadata field looks like.
    """
    compressed_data = base64.b64decode(base64_str)
    json_str = zlib.decompress(compressed_data).decode("utf-8")
    return json.loads(
        json_str,
        object_hook=lambda d: torch.tensor(d["value"]) if "__tensor__" in d else d,
    )


def workflow_extra(extra_pnginfo):
    """The workflow's own scratch dict, where a bundled latent is kept.

    Args:
        extra_pnginfo: The hidden ``EXTRA_PNGINFO`` value. It is ``None`` on a prompt queued
            through the API, which carries no workflow document at all.

    Returns:
        The mutable ``workflow.extra`` mapping, or ``None`` when this prompt has nowhere to
        keep a bundle. Writing into the returned dict is what puts the latent in the file
        the run saves.
    """
    if not isinstance(extra_pnginfo, dict):
        return None
    workflow = extra_pnginfo.get("workflow")
    if not isinstance(workflow, dict):
        return None
    extra = workflow.get("extra")
    return extra if isinstance(extra, dict) else None


def encode_image(vae, image, tiled: bool, tile_size: int):
    """Encode pixels to a latent through ComfyUI's own encoders.

    Args:
        vae: The VAE to encode with.
        image: An IMAGE tensor.
        tiled: Whether to encode in tiles instead of in one pass.
        tile_size: Tile edge in pixels, read only when ``tiled`` is set.

    Returns:
        The latent dict, exactly as ComfyUI's own VAE encode nodes build it.
    """
    # ComfyUI's node library, imported here, which resolves the encoders against the
    # running install at the moment they are used.
    import nodes

    if not tiled:
        return nodes.VAEEncode().encode(pixels=image, vae=vae)[0]

    encoder = nodes.VAEEncodeTiled()
    arguments = {"pixels": image, "tile_size": tile_size, "vae": vae}
    # `overlap` carries no default on installs where it is a widget of its own, and the
    # call fails outright without it. Supplying the value that widget offers keeps a tile
    # overlap of 64 whether the running install asks for it or fills it in.
    parameters = inspect.signature(encoder.encode).parameters
    overlap = parameters.get("overlap")
    if overlap is not None and overlap.default is inspect.Parameter.empty:
        arguments["overlap"] = TILE_OVERLAP
    return encoder.encode(**arguments)[0]


class BundleLatentVAEEncode(io.ComfyNode):
    """Encode an image, and optionally store the latent in the workflow document."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASBLVAEEncode",
            display_name="VAEEncode (Bundle Latent)",
            search_aliases=[
                "WASBLVAEEncode",
                "VAEEncode (Bundle Latent)",
                "bundle latent",
                "vae encode",
                "embed latent in workflow",
            ],
            category="WAS Suite/Latent",
            description=(
                "Encode an image to a latent and, if asked, keep a copy of that latent "
                "inside the workflow itself. A workflow saved with a bundled latent can be "
                "shared or reopened without the source image and still start from the same "
                "point, which is how a starting latent travels in one file instead of two."
            ),
            inputs=[
                io.Vae.Input(
                    "vae",
                    tooltip=(
                        "The VAE that turns the image into a latent. Use the one belonging "
                        "to the checkpoint that will sample it."
                    ),
                ),
                io.Boolean.Input(
                    "tiled",
                    default=False,
                    tooltip=(
                        "Whether the image is encoded a tile at a time instead of all at "
                        "once. Tiling holds far less in VRAM, which is what makes a very "
                        "large image encodable on a small card, at the cost of being "
                        "slower and of faint seams where tiles meet."
                    ),
                ),
                io.Int.Input(
                    "tile_size",
                    default=512,
                    min=320,
                    max=4096,
                    step=64,
                    tooltip=(
                        "Edge of one tile in pixels, read only when tiled is on. Smaller "
                        "tiles use less VRAM and take longer: 512 is a safe starting point, "
                        "and 1024 or more is worth trying if the card has room."
                    ),
                ),
                io.Boolean.Input(
                    "store_or_load_latent",
                    default=True,
                    tooltip=(
                        "Whether the workflow is used as the latent's home. On, the node "
                        "reads a latent already bundled in the workflow rather than "
                        "encoding, and writes the one it encodes back into it so the next "
                        "save carries it. Off, the node is an ordinary VAE encode and "
                        "touches nothing."
                    ),
                ),
                io.Boolean.Input(
                    "remove_latent_on_load",
                    default=True,
                    tooltip=(
                        "Whether a bundled latent is taken out of the workflow once it has "
                        "been read. On, it is used once and the saved file is left clean, "
                        "which suits carrying a starting point into a run. Off, it stays in "
                        "the workflow and every later save keeps carrying it."
                    ),
                ),
                io.Boolean.Input(
                    "delete_workflow_latent",
                    default=False,
                    tooltip=(
                        "Turn on for one run to throw away whatever this node has bundled "
                        "and encode the image again. That is the way out when the stored "
                        "latent no longer matches the image, or when a shared workflow "
                        "arrived with one that is not wanted."
                    ),
                ),
                io.Image.Input(
                    "image",
                    optional=True,
                    tooltip=(
                        "The image to encode. It can be left unconnected when the workflow "
                        "already carries a bundled latent, which is what lets a workflow be "
                        "reopened and run without the picture it started from."
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output(
                    display_name="latent",
                    tooltip=(
                        "The encoded latent, or the one that was bundled in the workflow "
                        "when there was one to read."
                    ),
                ),
            ],
            hidden=[io.Hidden.extra_pnginfo, io.Hidden.unique_id],
        )

    @classmethod
    def execute(
        cls,
        vae,
        tiled,
        tile_size,
        store_or_load_latent,
        remove_latent_on_load,
        delete_workflow_latent,
        image=None,
    ) -> io.NodeOutput:
        """Encode the image, or hand back the latent bundled in the workflow.

        Raises:
            ValueError: There is no image on the image input and no bundled latent to read,
                or nothing is connected to the vae input.
        """
        # `cls.hidden` is None when the body runs outside a prompt, and `extra_pnginfo` is
        # absent from a prompt queued through the API. Both address the bundle and nothing
        # else, so the encode goes ahead without either.
        hidden = getattr(cls, "hidden", None)
        unique_id = getattr(hidden, "unique_id", None)
        extra = workflow_extra(getattr(hidden, "extra_pnginfo", None))
        workflow_latent = None
        latent_key = f"latent_{unique_id}"
        has_image = torch.is_tensor(image)
        name = "VAEEncode (Bundle Latent)"
        node_label = f"{name} node {unique_id}" if unique_id else name

        # Anything bundled while this session has been running belongs to an earlier run of
        # this node, so a connected image replaces it instead of being ignored in its
        # favour. Only a bundle that arrived with the workflow is loaded.
        if unique_id is not None:
            if _last_image_hash.get(str(unique_id)) is not None and has_image:
                delete_workflow_latent = True
            if has_image:
                _last_image_hash[str(unique_id)] = sha256_tensor(image)

        if delete_workflow_latent and extra is not None and latent_key in extra:
            try:
                del extra[latent_key]
            except Exception:
                logger.warning("Unable to delete latent image from workflow node: %s", unique_id)

        if store_or_load_latent and not (unique_id and extra is not None):
            logger.info(
                "This prompt carries no workflow document to keep a latent in, so %s can "
                "neither read a bundled latent nor write one.", node_label
            )

        if store_or_load_latent and unique_id and extra is not None:
            if latent_key in extra:
                logger.info("Loading latent image from workflow node: %s", unique_id)
                try:
                    workflow_latent = deserialize(extra[latent_key])
                except Exception:
                    logger.error(
                        "There was an issue extracting the latent tensor from the workflow. "
                        "Is it corrupted?"
                    )
                    workflow_latent = None
                    if not has_image:
                        raise ValueError(
                            f"{node_label} could not read the latent bundled in the "
                            f"workflow, and has no image on its image input to encode "
                            f"instead. Connect an image, or turn on "
                            f"delete_workflow_latent for one run to discard the bundle."
                        )

                if workflow_latent and remove_latent_on_load:
                    try:
                        del extra[latent_key]
                    except Exception:
                        pass

        if workflow_latent:
            logger.info("Loaded workflow latent from node: %s", unique_id)
            return io.NodeOutput(workflow_latent)

        if not has_image:
            raise ValueError(
                f"{node_label} has no image on its image input, and no latent bundled in "
                f"the workflow to read instead. Connect an image to encode, or open a "
                f"workflow that carries a bundled latent."
            )

        require_input(
            vae,
            name,
            "vae",
            "VAE",
            "Load VAE or a checkpoint loader",
            "VAE",
        )
        encoded = encode_image(vae, image, tiled, tile_size)

        if store_or_load_latent and unique_id and extra is not None:
            logger.info("Saving latent to workflow node %s", unique_id)
            extra[latent_key] = serialize(encoded)

        return io.NodeOutput(encoded)
