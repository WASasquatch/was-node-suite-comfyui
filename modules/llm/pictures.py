"""Pictures for the LLM nodes: a picture embedding model, picture documents, pictures read back.

A picture document is an ``ImageNode`` holding a JPEG copy :data:`STORED_EDGE` pixels a side at
most, its folder label, relative path and prompt.
"""

from __future__ import annotations

import base64
import hashlib
import inspect
import io
import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field

from .. import deps, log
from . import FEATURE, embeddings, generate, messages, providers, runtime, wires

__all__ = [
    "IMAGE_EXTENSIONS",
    "MAX_DESCRIBE_TOKENS",
    "PICTURE_TYPES",
    "STORED_EDGE",
    "SUGGESTED",
    "Read",
    "choices",
    "describe",
    "frames_of",
    "is_picture",
    "load_folder",
    "picture_embedding",
    "png_bytes",
    "prompt_of",
    "stored_picture",
    "tensor_nodes",
    "to_batch",
    "video_nodes",
]

logger = log.get_logger("llm.pictures")

#: Repository ids offered beside the local folders, fetched only with ``features.network`` on.
SUGGESTED = (
    "google/siglip2-base-patch16-224",
    "google/siglip2-so400m-patch14-384",
    "openai/clip-vit-large-patch14",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
)

#: Files fetched from a suggested repository.
DOWNLOAD_FILES = (
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.model",
    "vocab.json",
    "merges.txt",
    "model.safetensors",
)

#: ``model_type`` values in a checkpoint's ``config.json`` that embed pictures and text together.
PICTURE_TYPES = embeddings.PICTURE_TYPES

#: Picture files a folder load reads.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff")

#: Longest side of the copy a picture document stores, in pixels.
STORED_EDGE = 384

#: JPEG quality of the stored copy.
STORED_QUALITY = 85

#: Text a picture model reads per text at most, in tokens, where its config names none.
TEXT_TOKENS = 64

#: Characters of a prompt read from a file kept.
MAX_PROMPT_CHARS = 2000

#: Characters of a ComfyUI prompt graph parsed at most.
MAX_GRAPH_CHARS = 4 * 1024 * 1024

#: Graph nodes visited while following a prompt back from a sampler.
MAX_GRAPH_STEPS = 400

#: Tokens a description is written in at most.
MAX_DESCRIBE_TOKENS = 300

#: Pictures one batch output holds at most.
MAX_BATCH = 256

#: Input names whose text is a prompt in a ComfyUI graph.
_TEXT_KEYS = ("text", "text_g", "text_l", "prompt", "positive_prompt", "string", "value", "t5xxl", "clip_l")

#: Input names whose text names a model file in a ComfyUI graph.
_MODEL_KEYS = ("ckpt_name", "unet_name", "model_name")

#: Metadata kept from the embedding and from the language model's view.
_HIDDEN = ("folder", "relative_path", "size", "modified", "width", "height", "source", "extension")

_classes: dict = {}


# ---------------------------------------------------------------------- the model


def choices() -> list[str]:
    """What the picture model menu offers: picture checkpoints on disk, then suggested ids."""
    found = []
    for label in embeddings.local_folders():
        folder = embeddings._local_folder(label)
        if folder and embeddings.model_type(folder) in PICTURE_TYPES:
            found.append(label)
    return found + [name for name in SUGGESTED if name not in found]


def _backend(folder: str, device):
    """The managed processor and model of a picture checkpoint folder, built on first use."""
    from ..model import compute_device, managed

    def build():
        import torch

        transformers = deps.require("transformers", feature=FEATURE)
        target = compute_device(device)
        if target.type == "cpu":
            dtype = torch.float32
        elif target.type == "cuda" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
        processor = transformers.AutoProcessor.from_pretrained(
            folder, token=False, trust_remote_code=False, local_files_only=True,
        )
        network = transformers.AutoModel.from_pretrained(
            folder, token=False, trust_remote_code=False, use_safetensors=True,
            local_files_only=True, dtype=dtype,
        )
        return processor, network

    key = ("llm-picture-embedding", os.path.normcase(os.path.abspath(folder)))
    return managed(key, build, device=device, name="LLMImageEmbeddingModel")


def _accepted(method) -> frozenset[str]:
    """The named arguments a model method takes."""
    try:
        parameters = inspect.signature(method).parameters.values()
    except (TypeError, ValueError):
        return frozenset()
    return frozenset(p.name for p in parameters if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY))


def _pooled(output):
    """The projected vectors a feature call answers, as a tensor or in ``pooler_output``."""
    import torch

    if isinstance(output, torch.Tensor):
        return output
    for name in ("pooler_output", "image_embeds", "text_embeds"):
        value = getattr(output, name, None)
        if isinstance(value, torch.Tensor):
            return value
    raise RuntimeError("the picture model answered no vectors")


def _opened(picture):
    """A PIL picture from what llama-index hands an embedding: a byte stream or raw bytes.

    Raises:
        ValueError: It is a path. Pictures are embedded from the copy a document stores.
    """
    from ..image import sizing

    if isinstance(picture, io.BytesIO):
        data = picture.getvalue()
    elif isinstance(picture, (bytes, bytearray)):
        data = bytes(picture)
    else:
        raise ValueError("a picture document holds no stored copy to embed. Load it again with LLM Load Images.")
    return sizing.open_bytes(data).convert("RGB")


def _picture_class():
    """The embedding class over a CLIP or SigLIP checkpoint folder."""
    if "picture" in _classes:
        return _classes["picture"]
    runtime.core()
    from llama_index.core.bridge.pydantic import Field
    from llama_index.core.embeddings.multi_modal_base import MultiModalEmbedding

    class PictureEmbedding(MultiModalEmbedding):
        """Text and picture vectors in one space, from a CLIP or SigLIP checkpoint folder."""

        folder: str = Field(description="The checkpoint folder.")
        family: str = Field(default="clip", description="clip, siglip or siglip2.")
        device: str | None = Field(default=None, description="Device name, None for ComfyUI's.")

        @classmethod
        def class_name(cls) -> str:
            return "WAS_PictureEmbedding"

        def backend(self):
            """The managed processor and model."""
            return _backend(self.folder, self.device)

        def _feed(self, encoded, accepted, device, dtype) -> dict:
            """Processor output moved to the device, floating tensors in the model's precision."""
            feed = {}
            for name, value in encoded.items():
                if name not in accepted:
                    continue
                value = value.to(device)
                feed[name] = value.to(dtype) if value.is_floating_point() else value
            return feed

        def text_vectors(self, texts: list[str]) -> list[list[float]]:
            """Embed texts in batches of ``embed_batch_size``."""
            import torch

            backend = self.backend()
            processor, network = backend.processor, backend.model
            tokenizer = getattr(processor, "tokenizer", processor)
            accepted = _accepted(network.get_text_features)
            config = getattr(network.config, "text_config", network.config)
            limit = int(getattr(config, "max_position_embeddings", 0) or TEXT_TOKENS)
            siglip = self.family.startswith("siglip")
            found: list[list[float]] = []
            for start in range(0, len(texts), self.embed_batch_size):
                runtime.check_interrupt()
                device = backend.load()
                batch = texts[start:start + self.embed_batch_size]
                if siglip:
                    batch = [text.lower() for text in batch]
                encoded = tokenizer(
                    batch, padding="max_length" if siglip else True, truncation=True,
                    max_length=limit, return_tensors="pt",
                )
                with torch.inference_mode():
                    vectors = _pooled(network.get_text_features(**self._feed(encoded, accepted, device, network.dtype)))
                vectors = torch.nn.functional.normalize(vectors.float(), p=2, dim=-1)
                found.extend(vectors.cpu().tolist())
            return found

        def picture_vectors(self, pictures: list) -> list[list[float]]:
            """Embed pictures in batches of ``embed_batch_size``."""
            import torch

            backend = self.backend()
            processor, network = backend.processor, backend.model
            accepted = _accepted(network.get_image_features)
            found: list[list[float]] = []
            for start in range(0, len(pictures), self.embed_batch_size):
                runtime.check_interrupt()
                device = backend.load()
                opened = [_opened(item) for item in pictures[start:start + self.embed_batch_size]]
                encoded = processor(images=opened, return_tensors="pt")
                with torch.inference_mode():
                    vectors = _pooled(network.get_image_features(**self._feed(encoded, accepted, device, network.dtype)))
                vectors = torch.nn.functional.normalize(vectors.float(), p=2, dim=-1)
                found.extend(vectors.cpu().tolist())
            return found

        def _get_query_embedding(self, query: str) -> list[float]:
            return self.text_vectors([query])[0]

        async def _aget_query_embedding(self, query: str) -> list[float]:
            return self._get_query_embedding(query)

        def _get_text_embedding(self, text: str) -> list[float]:
            return self.text_vectors([text])[0]

        def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
            return self.text_vectors(list(texts))

        def _get_image_embedding(self, img_file_path) -> list[float]:
            return self.picture_vectors([img_file_path])[0]

        async def _aget_image_embedding(self, img_file_path) -> list[float]:
            return self._get_image_embedding(img_file_path)

        def _get_image_embeddings(self, img_file_paths) -> list[list[float]]:
            return self.picture_vectors(list(img_file_paths))

    _classes["picture"] = PictureEmbedding
    return PictureEmbedding


def picture_embedding(choice: str, batch_size: int = 16, node: str = "LLM Image Embedding Model") -> tuple[wires.EmbeddingModel, dict]:
    """A picture embedding model from a local folder or a suggested repository.

    Args:
        choice: An entry of :func:`choices`.
        batch_size: Pictures or texts embedded together.
        node: The node's display name, for messages.

    Returns:
        The wire value, and ``family`` and ``device`` as used.

    Raises:
        ValueError: The choice names nothing on disk or in :data:`SUGGESTED`, holds no safetensors
            weights, or is not a CLIP or SigLIP checkpoint.
        ModelUnavailable: A suggested repository is not here and ``features.network`` is off.
    """
    label = str(choice or "").strip()
    folder = embeddings._local_folder(label)
    if folder is None:
        if label not in SUGGESTED:
            raise ValueError(
                f"{node} cannot find `{label[:80]}` in models/LLM. Pick a model from the menu, or put "
                f"a CLIP or SigLIP checkpoint folder holding config.json in models/LLM."
            )
        folder = embeddings._fetched(label, DOWNLOAD_FILES)
    embeddings._require_safetensors(folder, label, node)
    family = embeddings.model_type(folder)
    if family not in PICTURE_TYPES:
        raise ValueError(
            f"{node}: {label} is a `{family or 'unknown'}` model, which does not embed pictures. Pick a "
            f"CLIP or SigLIP model, such as {SUGGESTED[0]}."
        )
    embedder = _picture_class()(
        model_name=label, folder=folder, family=family, embed_batch_size=max(1, int(batch_size)),
    )
    started = time.monotonic()
    dimensions = len(providers.construct(node, lambda: embedder.get_text_embedding("a picture")))
    used = str(embedder.backend().load_device)
    logger.info("picture embedding model %s ready: %d dimensions, on %s, %.2fs", label, dimensions, used, time.monotonic() - started)
    wired = wires.EmbeddingModel(
        model=embedder, label=label, provider="transformers", local=True, dimensions=dimensions, pictures=True,
    )
    return wired, {"family": family, "device": used}


# ---------------------------------------------------------------------- documents


def is_picture(item) -> bool:
    """Whether a llama-index node is a picture document."""
    runtime.core()
    from llama_index.core.schema import ImageNode

    return isinstance(item, ImageNode)


def stored_picture(picture) -> tuple[str, int, int]:
    """The copy a picture document stores.

    Args:
        picture: A PIL picture.

    Returns:
        ``(base64 JPEG, width, height)``, the size being the picture's own.
    """
    from PIL import Image

    width, height = picture.size
    copy = picture.convert("RGB")
    if max(copy.size) > STORED_EDGE:
        copy.thumbnail((STORED_EDGE, STORED_EDGE), Image.LANCZOS)
    buffer = io.BytesIO()
    copy.save(buffer, format="JPEG", quality=STORED_QUALITY)
    return base64.b64encode(buffer.getvalue()).decode("ascii"), int(width), int(height)


def _node(picture, text: str, metadata: dict, node_id: str):
    """One picture document."""
    runtime.core()
    from llama_index.core.schema import ImageNode

    encoded, width, height = stored_picture(picture)
    kept = {**metadata, "width": width, "height": height}
    hidden = [key for key in _HIDDEN if key in kept]
    return ImageNode(
        id_=node_id,
        image=encoded,
        image_mimetype="image/jpeg",
        text=text,
        metadata=kept,
        excluded_embed_metadata_keys=hidden,
        excluded_llm_metadata_keys=hidden,
    )


def _graph_prompt(graph: dict) -> tuple[str, str]:
    """The positive prompt and the model a ComfyUI prompt graph names.

    Args:
        graph: The API prompt: node id to ``{"class_type", "inputs"}``.

    Returns:
        ``(prompt, model file name)``, either empty.
    """
    texts: list[str] = []
    seen: set = set()
    steps = 0

    def follow(node_id) -> None:
        nonlocal steps
        key = str(node_id)
        entry = graph.get(key)
        if key in seen or not isinstance(entry, dict) or steps >= MAX_GRAPH_STEPS:
            return
        seen.add(key)
        steps += 1
        inputs = entry.get("inputs") if isinstance(entry.get("inputs"), dict) else {}
        for name, value in inputs.items():
            if name in _TEXT_KEYS and isinstance(value, str) and value.strip():
                texts.append(value.strip())
            elif isinstance(value, list) and len(value) == 2 and name not in ("negative", "model", "clip", "vae"):
                follow(value[0])

    for entry in graph.values():
        inputs = entry.get("inputs") if isinstance(entry, dict) and isinstance(entry.get("inputs"), dict) else {}
        positive = inputs.get("positive")
        if isinstance(positive, list) and len(positive) == 2:
            follow(positive[0])
    model = ""
    for entry in graph.values():
        inputs = entry.get("inputs") if isinstance(entry, dict) and isinstance(entry.get("inputs"), dict) else {}
        for name in _MODEL_KEYS:
            if isinstance(inputs.get(name), str) and not model:
                model = inputs[name].replace("\\", "/").rsplit("/", 1)[-1]
    unique = list(dict.fromkeys(texts))
    return "\n".join(unique)[:MAX_PROMPT_CHARS], model[:200]


def prompt_of(info: dict) -> tuple[str, str]:
    """The prompt and model a picture file's text chunks record, from ComfyUI or A1111.

    Args:
        info: PIL's ``Image.info`` of the opened file.

    Returns:
        ``(prompt, model file name)``, either empty.
    """
    graph = info.get("prompt")
    if isinstance(graph, str) and len(graph) <= MAX_GRAPH_CHARS:
        try:
            parsed = json.loads(graph)
        except (ValueError, RecursionError):
            parsed = None
        if isinstance(parsed, dict):
            prompt, model = _graph_prompt(parsed)
            if prompt or model:
                return prompt, model
    parameters = info.get("parameters")
    if isinstance(parameters, str) and parameters.strip():
        head = parameters.split("\nNegative prompt:", 1)[0].split("\nSteps:", 1)[0]
        model = ""
        for part in parameters.rsplit("\n", 1)[-1].split(","):
            name, _, value = part.partition(":")
            if name.strip() == "Model":
                model = value.strip()
        return head.strip()[:MAX_PROMPT_CHARS], model[:200]
    return "", ""


def _file_info(data: bytes) -> dict:
    """The text chunks a picture file's header holds, read without decoding its pixels."""
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as opened:
            return dict(opened.info)
    except Exception:
        return {}


@dataclass
class Read:
    """What a picture folder load read and what it passed over.

    Attributes:
        items: Picture documents, in path order.
        skipped: Reason to files passed over.
        matched: Files whose extension matched.
        prompts: Pictures a prompt was read from.
    """

    items: list = field(default_factory=list)
    skipped: Counter = field(default_factory=Counter)
    matched: int = 0
    prompts: int = 0


def load_folder(root, folder: str, recursive: bool, max_files: int, max_bytes: int, read_prompts: bool) -> Read:
    """Every picture under a folder as picture documents, passing over what cannot be read.

    Args:
        root: The folder, already resolved inside a permitted read root.
        folder: Its menu label, such as ``output``.
        recursive: Descend into subfolders.
        max_files: Most files read.
        max_bytes: Most bytes one file may hold.
        read_prompts: Read the prompt ComfyUI or A1111 wrote into the file as the document's text.

    Returns:
        What was read.
    """
    from ..image import sizing
    from .documents import walk

    files, _stopped = walk(root, IMAGE_EXTENSIONS, recursive)
    loaded = Read(matched=len(files))
    if len(files) > max_files:
        loaded.skipped["past max_files"] = len(files) - max_files
        files = files[:max_files]
    for found in files:
        runtime.check_interrupt()
        if found.size > max_bytes:
            loaded.skipped["past max_file_mb"] += 1
            continue
        try:
            with open(found.path, "rb") as handle:
                data = handle.read(max_bytes + 1)
            picture = sizing.open_bytes(data)
            prompt, model = prompt_of(_file_info(data)) if read_prompts else ("", "")
        except Exception as error:
            if type(error).__name__ == "InterruptProcessingException":
                raise
            loaded.skipped["unreadable"] += 1
            logger.debug("LLM Load Images could not read %s: %s", found.relative, error)
            continue
        metadata = {
            "file_name": os.path.basename(found.relative),
            "folder": folder,
            "relative_path": found.relative,
            "extension": found.extension,
            "size": int(found.size),
            "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(found.mtime)),
            "source": "picture",
        }
        if prompt:
            metadata["prompt"] = prompt
            loaded.prompts += 1
        if model:
            metadata["model"] = model
        loaded.items.append(_node(picture, prompt, metadata, f"{folder}:{found.relative}"))
    return loaded


def frames_of(images) -> list:
    """The frames of an ``IMAGE`` batch as PIL pictures."""
    import numpy as np
    from PIL import Image

    frames = images if images.ndim == 4 else images[None]
    pictures = []
    for frame in frames:
        values = (frame.detach().float().clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
        if values.shape[-1] == 1:
            pictures.append(Image.fromarray(values[..., 0], mode="L").convert("RGB"))
        else:
            pictures.append(Image.fromarray(values[..., :3], mode="RGB"))
    return pictures


def png_bytes(images) -> bytes:
    """The first frame of an ``IMAGE`` batch as PNG bytes."""
    buffer = io.BytesIO()
    frames_of(images[:1])[0].save(buffer, format="PNG")
    return buffer.getvalue()


def tensor_nodes(images, captions: list[str], name: str, metadata: dict) -> list:
    """Picture documents from an ``IMAGE`` batch.

    Args:
        images: The batch.
        captions: Text for each frame in order; frames past the list get none.
        name: What each document is called, numbered per frame, such as ``reference``.
        metadata: Pairs stored on every document.

    Returns:
        One picture document per frame, its id taken from its pixels.
    """
    made = []
    for number, picture in enumerate(frames_of(images), 1):
        runtime.check_interrupt()
        text = captions[number - 1].strip() if number <= len(captions) else ""
        digest = hashlib.sha256(picture.tobytes()).hexdigest()[:16]
        values = {**metadata, "file_name": f"{name} {number}", "source": "picture"}
        made.append(_node(picture, text, values, f"picture:{digest}:{number}"))
    return made


def video_nodes(video, count: int, name: str, node: str) -> list:
    """Picture documents from frames spread evenly across a video, each naming its time.

    Args:
        video: A ComfyUI ``VideoInput``.
        count: Frames taken at most.
        name: What each document is called, followed by its time, such as ``trailer 00:12.5``.
        node: The node's display name, for messages.

    Returns:
        One picture document per frame taken, in time order.
    """
    from ..media import sampling

    total = int(video.get_frame_count())
    rate = float(video.get_frame_rate()) or 24.0
    indices = sampling.frame_indices(total, max(1, int(count)), "uniform")
    frames = sampling.decode_frames(video, indices, node).get_components().images
    made = []
    for number, (index, picture) in enumerate(zip(indices, frames_of(frames)), 1):
        runtime.check_interrupt()
        seconds = index / rate
        stamp = f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"
        values = {"file_name": f"{name} {stamp}", "time_seconds": round(seconds, 3), "frame": int(index), "source": "video frame"}
        made.append(_node(picture, "", values, f"video:{name}:{index}"))
    return made


def describe(items: list, model: wires.LanguageModel, prompt: str, node: str) -> int:
    """Have a vision model describe each picture document, adding the words to its text.

    Args:
        items: Picture documents, changed in place.
        model: A language model that reads pictures.
        prompt: What it is asked about each one.
        node: The node's display name, for messages.

    Returns:
        How many were described.
    """
    from llama_index.core.llms import ChatMessage, ImageBlock, MessageRole, TextBlock

    generate.require_vision(model, node, len(items))
    described = 0
    for item in items:
        runtime.check_interrupt()
        block = ImageBlock(image=base64.b64decode(item.image), image_mimetype="image/jpeg")
        chat = [ChatMessage(role=MessageRole.USER, blocks=[block, TextBlock(text=prompt)])]
        answer = generate.reply(model, chat, node, max_tokens=MAX_DESCRIBE_TOKENS)
        words = answer.text.strip()
        if words:
            item.text = "\n\n".join(part for part in (words[:MAX_PROMPT_CHARS], item.text) if part)
            described += 1
    return described


# ---------------------------------------------------------------------- reading back


def _full_size(node, max_bytes: int):
    """The picture file a document was read from, at its own size, or ``None``."""
    from ..image import sizing
    from ..util import sandbox
    from ..util.sandbox import leaves_folder
    from .documents import folder_root

    metadata = node.metadata or {}
    folder, relative = metadata.get("folder"), metadata.get("relative_path")
    if not isinstance(folder, str) or not isinstance(relative, str) or leaves_folder(relative) is not None:
        return None
    try:
        root = folder_root(folder)
        if root is None:
            return None
        candidate = os.path.join(str(root), *relative.split("/"))
        path = sandbox.resolve_read(candidate)
        if not path.is_file() or os.path.getsize(path) > max_bytes:
            return None
        with open(path, "rb") as handle:
            return sizing.open_bytes(handle.read(max_bytes + 1)).convert("RGB")
    except Exception as error:
        logger.debug("the picture file %s could not be read back (%s)", relative, error)
        return None


def to_batch(items, full_size: bool, size: tuple[int, int], mode: str, max_bytes: int, node: str):
    """The pictures among scored chunks as one ``IMAGE`` batch.

    Args:
        items: ``NodeWithScore`` objects, best first.
        full_size: Read each picture's file where it is still there, else the stored copy.
        size: ``(width, height)`` every picture is brought to; 0 takes the first picture's.
        mode: One of :data:`modules.image.sizing.MODES`.
        max_bytes: Most bytes one file may hold.
        node: The node's display name, for messages.

    Returns:
        ``(batch, rows)``: the tensor, or ``None`` when no picture was found, and one
        ``(file name, score, read from file)`` row per picture.
    """
    import numpy as np
    import torch

    from ..image import sizing

    pictures, rows = [], []
    for item in items:
        if not is_picture(item.node) or not item.node.image:
            continue
        if len(pictures) >= MAX_BATCH:
            break
        picture = _full_size(item.node, max_bytes) if full_size else None
        from_file = picture is not None
        if picture is None:
            picture = sizing.open_bytes(base64.b64decode(item.node.image)).convert("RGB")
        pictures.append(picture)
        rows.append((str(item.node.metadata.get("file_name", "")), item.score, from_file))
    if not pictures:
        return None, rows
    width = int(size[0]) or pictures[0].size[0]
    height = int(size[1]) or pictures[0].size[1]
    sizing.check_canvas(width, height, f"{node}'s pictures", "Lower width or height.")
    frames = []
    for picture in pictures:
        runtime.check_interrupt()
        placed = picture if picture.size == (width, height) else sizing.fit(picture.convert("RGBA"), width, height, mode)
        values = np.asarray(placed.convert("RGB"), dtype=np.float32) / 255.0
        frames.append(torch.from_numpy(values))
    return torch.stack(frames), rows
