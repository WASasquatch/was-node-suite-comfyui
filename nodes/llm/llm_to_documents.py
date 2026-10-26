"""Turn whatever is wired in, text, pictures, video, a DOC, a conversation, sources or tables, into documents."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import DOC, LIST, LLM, LLM_DOCUMENTS, LLM_MESSAGES, LLM_SOURCES, LLM_TABLES
from ...modules.interface import run_result
from ...modules.llm import documents as loading
from ...modules.llm import messages, pictures, runtime, wires
from ...modules.llm import tables as tabling

REQUIRES = "llm"

logger = log.get_logger("llm.documents")

#: The node's name in messages.
NODE = "LLM To Documents"

#: Items a list or a batch turns into documents at most.
MAX_ITEMS = 10000

#: Frames a video gives at most.
MAX_VIDEO_FRAMES = 1024

#: What a vision model is asked about each picture or frame.
DESCRIBE_PROMPT = (
    "Describe this picture in one short paragraph for a search index: the subject, setting, "
    "style, colours and composition. Write only the description."
)


def _kind(value) -> str:
    """What arrived on the value input, as the word the panel shows."""
    import torch

    from ...modules.document import container

    if isinstance(value, str):
        return "text"
    if isinstance(value, (list, tuple)):
        return "list"
    if torch.is_tensor(value):
        return "image"
    if hasattr(value, "get_components") and hasattr(value, "get_frame_count"):
        return "video"
    if container.is_document(value):
        return "doc"
    if isinstance(value, wires.Messages):
        return "conversation"
    if isinstance(value, wires.Sources):
        return "sources"
    if isinstance(value, tabling.Tables):
        return "tables"
    return ""


class LLMToDocuments(io.ComfyNode):
    """Documents from text, pictures, a video, a DOC, a conversation, found sources or tables."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMToDocuments",
            display_name="LLM To Documents",
            search_aliases=[
                "WASLLMToDocuments",
                "LLM To Documents",
                "To Documents",
                "anything to documents",
                "video to documents",
                "image to documents",
                "text to documents",
            ],
            category="WAS Suite/LLM/Documents",
            description=(
                "Turn whatever is wired in into documents for an index: text, a list of text, pictures, "
                "frames spread across a video, a DOC, a conversation, found sources or table rows. A vision "
                "model can describe the pictures. The panel shows what was made."
            ),
            inputs=[
                io.MultiType.Input(
                    "value",
                    [io.String, LIST, io.Image, io.Video, DOC, LLM_MESSAGES, LLM_SOURCES, LLM_TABLES],
                    tooltip=(
                        "What to turn into documents: a STRING, a list of strings (one each), an IMAGE batch "
                        "(one per frame), a VIDEO, a DOC, a conversation, sources from LLM Retrieve or tables."
                    ),
                ),
                io.String.Input(
                    "title", default="",
                    tooltip=(
                        "What the documents are called, such as `trailer` giving `trailer 00:12.5` for a video frame. "
                        "Blank = the kind wired in. A DOC, sources and tables keep their own names."
                    ),
                ),
                io.Int.Input(
                    "video_frames", default=8, min=1, max=MAX_VIDEO_FRAMES,
                    tooltip="Frames taken from a video, spread evenly from start to end: 8 = a short clip; 64 = a film's scenes.",
                ),
                LLM.Input(
                    "describe", optional=True,
                    tooltip="A vision model that writes a description of each picture or video frame into its text, such as LLM From CLIP with Qwen3 VL.",
                ),
                LLM_DOCUMENTS.Input(
                    "documents", optional=True,
                    tooltip="Documents from another loader. The new ones are added after them.",
                ),
            ],
            outputs=[
                LLM_DOCUMENTS.Output(
                    display_name="documents",
                    tooltip="The documents, for LLM Split Documents or an LLM index; pictures and frames for LLM Image Index.",
                ),
            ],
        )

    @classmethod
    def execute(cls, value, title, video_frames, describe=None, documents=None) -> io.NodeOutput:
        """Make documents from the value.

        Args:
            value: What arrived.
            title: What the documents are called.
            video_frames: Frames taken from a video.
            describe: A vision model describing pictures, or ``None``.
            documents: Documents to add these after.

        Returns:
            The documents.
        """
        previous = None
        if documents is not None:
            previous = wires.expect(documents, wires.Documents, NODE, "documents", "an LLM document loader")
        model = None
        if describe is not None:
            model = wires.expect(describe, wires.LanguageModel, NODE, "describe", "LLM From CLIP with a vision model")
        kind = _kind(value)
        if not kind:
            raise ValueError(
                f"{NODE} cannot turn a {type(value).__name__} into documents. Connect text, a list of text, an "
                f"IMAGE, a VIDEO, a DOC, a conversation, sources or tables."
            )
        name = str(title or "").strip()[:80] or kind
        wires.ready(model)
        if kind == "text":
            made = [loading.text_document(value, name if title else "", None)]
        elif kind == "list":
            texts = [str(item) for item in list(value)[:MAX_ITEMS] if str(item if item is not None else "").strip()]
            if not texts:
                raise ValueError(f"{NODE} was given an empty list. Wire a list holding text.")
            made = [loading.text_document(text, f"{name} {number}", None) for number, text in enumerate(texts, 1)]
        elif kind == "image":
            frames = value if value.ndim == 4 else value[None]
            if frames.shape[0] > MAX_ITEMS:
                raise ValueError(f"{NODE} was given {frames.shape[0]} pictures; {MAX_ITEMS} is the most.")
            made = pictures.tensor_nodes(frames, [], name, {})
        elif kind == "video":
            made = pictures.video_nodes(value, int(video_frames), name, NODE)
        elif kind == "doc":
            made = loading.doc_documents(value, False)
            if not made:
                raise ValueError(f"{NODE}: the DOC holds no text.")
        elif kind == "conversation":
            text = messages.transcript(value.items)
            if not text.strip():
                raise ValueError(f"{NODE}: the conversation holds no messages.")
            made = [loading.text_document(text, name if title else "conversation", {"source": "conversation"})]
        elif kind == "sources":
            made = [item.node.model_copy(update={"metadata": dict(item.node.metadata)}) for item in value.items[:MAX_ITEMS]]
            if not made:
                raise ValueError(f"{NODE}: the sources hold nothing. Check that the search found something.")
        else:
            made = tabling.row_documents(value, "", 1, MAX_ITEMS)
        described = 0
        if model is not None:
            shown = [item for item in made if pictures.is_picture(item)]
            if shown:
                with runtime.guarded():
                    described = pictures.describe(shown, model, DESCRIBE_PROMPT, NODE)
        added = wires.Documents(tuple(made), f"{len(made)} from {kind}")
        result = previous.plus(added) if previous is not None else added
        picture_count = sum(1 for item in made if pictures.is_picture(item))
        logger.info("%s made %d document(s) from %s", NODE, len(made), kind)
        counts = {"documents": len(made)}
        if picture_count:
            counts["pictures"] = picture_count
        if model is not None:
            counts["described"] = described
        if len(result) != len(made):
            counts["with chain"] = len(result)
        facts = {"from": kind}
        if kind not in ("doc", "sources", "tables"):
            facts["named"] = name
        bodies = []
        for item in made[:3]:
            label = str((item.metadata or {}).get("file_name") or (item.metadata or {}).get("title") or kind)
            text = (getattr(item, "text", "") or "")[:2000]
            bodies.append(run_result.body(label, text or f"picture {label}"))
        run_result.publish(
            status=run_result.OK,
            summary=f"made {len(made)} document(s) from {kind}",
            counts=counts,
            facts=facts,
            bodies=bodies,
        )
        return io.NodeOutput(result)
