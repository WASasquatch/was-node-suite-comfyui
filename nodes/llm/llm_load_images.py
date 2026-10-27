"""Read every picture in a folder onto an LLM documents wire, with the prompt each was made from."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM, LLM_DOCUMENTS
from ...modules.interface import run_result
from ...modules.io import picker
from ...modules.llm import documents as loading
from ...modules.llm import pictures, runtime, wires
from ...modules.util import sandbox

REQUIRES = "llm"

logger = log.get_logger("llm.documents")

#: The node's name in messages.
NODE = "LLM Load Images"

#: What a vision model is asked about each picture, unless the node says otherwise.
DESCRIBE_PROMPT = (
    "Describe this picture in one short paragraph for a search index: the subject, setting, "
    "style, colours and composition. Write only the description."
)

#: Prompts shown on the panel.
SAMPLES = 3


class LLMLoadImages(io.ComfyNode):
    """Every picture in a folder, as picture documents for an image index."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        folders = picker.folders()
        return io.Schema(
            node_id="WASLLMLoadImages",
            display_name="LLM Load Images",
            search_aliases=[
                "WASLLMLoadImages",
                "LLM Load Images",
                "ImageDocument",
                "image folder",
                "search outputs",
                "multimodal RAG",
                "picture documents",
            ],
            category="WAS Suite/LLM/Documents",
            description=(
                "Read every picture in a folder, such as ComfyUI's output folder, into documents for "
                "LLM Image Index, each holding the prompt it was made from. A vision model can add a "
                "description of each. The panel shows what was read and a sample of the prompts."
            ),
            inputs=[
                picker.path_input(io.Combo.Input(
                    "folder", options=folders, default="output" if "output" in folders else None,
                    tooltip=(
                        "Which folder to read. `output` = ComfyUI's output folder; `faces [input]` = "
                        "the faces folder inside input. A folder under paths.allow_read in config.yaml "
                        "is listed under its own name."
                    ),
                ), folder=True),
                io.Boolean.Input(
                    "recursive", default=True,
                    tooltip="`true` reads the folders inside it too; `false` reads only the pictures directly in it. Links are not followed.",
                ),
                io.Int.Input(
                    "max_files", default=500, min=1, max=10000,
                    tooltip="Most pictures read, taken in path order: 500 = a few days of renders. Those past it are counted as skipped.",
                ),
                io.Int.Input(
                    "max_file_mb", default=64, min=1, max=512,
                    tooltip="Largest picture file read, in megabytes: `64` = any render; a larger file is counted as skipped.",
                ),
                io.Boolean.Input(
                    "read_prompts", default=True,
                    tooltip="`true` reads the prompt ComfyUI or A1111 saved in each file as its text, so words find it; `false` keeps pictures only.",
                ),
                LLM.Input(
                    "describe", optional=True,
                    tooltip="A vision model that writes a description of each picture into its text, such as LLM From CLIP with Qwen3 VL. Slow: one answer per picture.",
                ),
                io.String.Input(
                    "describe_prompt", multiline=True, default=DESCRIBE_PROMPT, optional=True, advanced=True,
                    tooltip="What the vision model is asked about each picture, such as `List the people and objects in this picture.`",
                ),
                LLM_DOCUMENTS.Input(
                    "documents", optional=True,
                    tooltip="Documents from another loader. The pictures read here are added after them.",
                ),
            ],
            outputs=[
                LLM_DOCUMENTS.Output(
                    display_name="documents",
                    tooltip="One picture document per file, naming its folder, path, size, date and prompt, for LLM Image Index.",
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, folder="", recursive=True, max_files=500, **kwargs):
        """A digest of the pictures the folder holds, so a changed folder is read again."""
        try:
            root = loading.folder_root(folder)
        except (ValueError, sandbox.PathNotAllowed):
            return float("NaN")
        return loading.fingerprint_folder(root, pictures.IMAGE_EXTENSIONS, bool(recursive), int(max_files))

    @classmethod
    def execute(cls, folder, recursive, max_files, max_file_mb, read_prompts, describe=None,
                describe_prompt=DESCRIBE_PROMPT, documents=None) -> io.NodeOutput:
        """Walk the folder and read each picture.

        Args:
            folder: A label from the folder menu.
            recursive: Read subfolders too.
            max_files: Most pictures read.
            max_file_mb: Largest file read, in megabytes.
            read_prompts: Read the prompt saved in each file.
            describe: A vision model describing each picture, or ``None``.
            describe_prompt: What it is asked.
            documents: Documents to add these after.

        Returns:
            The documents.

        Raises:
            ValueError: The folder names nothing, or holds no picture that could be read.
            PathNotAllowed: The folder lies outside every permitted read root.
        """
        previous = None
        if documents is not None:
            previous = wires.expect(documents, wires.Documents, NODE, "documents", "an LLM document loader")
        model = None
        if describe is not None:
            model = wires.expect(describe, wires.LanguageModel, NODE, "describe", "LLM From CLIP with a vision model")
        root = loading.folder_root(folder)
        if root is None or not root.is_dir():
            raise ValueError(
                f"`{folder}` names no folder {NODE} can read. Pick another from the menu, or add its "
                f"folder to paths.allow_read in config.yaml."
            )
        wires.ready(model)
        loaded = pictures.load_folder(
            root, folder, bool(recursive), int(max_files), int(max_file_mb) * 1024 * 1024, bool(read_prompts),
        )
        if not loaded.items:
            reasons = ", ".join(f"{reason} {count}" for reason, count in loaded.skipped.most_common())
            raise ValueError(
                f"{NODE} read no picture from {folder}: {loaded.matched} picture file(s)"
                f"{', passed over as ' + reasons if reasons else ''}"
                f"{'' if recursive else ', and recursive is false'}. Pick another folder."
            )
        described = 0
        if model is not None:
            asked = str(describe_prompt or "").strip() or DESCRIBE_PROMPT
            with runtime.guarded():
                described = pictures.describe(loaded.items, model, asked, NODE)
        added = wires.Documents(tuple(loaded.items), f"{len(loaded.items)} pictures from {folder}")
        result = previous.plus(added) if previous is not None else added
        logger.info(
            "%s read %d of %d picture(s) from %s, %d with a prompt, %d described",
            NODE, len(loaded.items), loaded.matched, folder, loaded.prompts, described,
        )
        counts = {"pictures": len(loaded.items), "with a prompt": loaded.prompts}
        if model is not None:
            counts["described"] = described
        if len(result) != len(added):
            counts["with chain"] = len(result)
        if loaded.skipped:
            counts["skipped"] = sum(loaded.skipped.values())
        facts = {"folder": folder}
        if loaded.skipped:
            facts["skipped"] = ", ".join(f"{reason} {count}" for reason, count in loaded.skipped.most_common())
        if model is not None:
            facts["described by"] = model.label
        bodies = [
            run_result.body(str(item.metadata.get("file_name", "")), item.text[:2000])
            for item in [node for node in loaded.items if node.text][:SAMPLES]
        ]
        run_result.publish(
            status=run_result.WARNING if loaded.skipped else run_result.OK,
            summary=f"read {len(loaded.items)} of {loaded.matched} picture(s) from {folder}",
            counts=counts,
            facts=facts,
            bodies=bodies,
        )
        return io.NodeOutput(result)
