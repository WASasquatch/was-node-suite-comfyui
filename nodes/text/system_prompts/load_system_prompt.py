"""Load one system prompt or skill from the system prompt library."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.compat.types import LLM_DOCUMENTS
from ....modules.config import group_enabled
from ....modules.llm import FEATURE as LLM_FEATURE
from ....modules.llm import wires
from ....modules.prompt import system_prompts

logger = log.get_logger("text.system_prompts")


class LoadSystemPrompt(io.ComfyNode):
    """Emit the text and name of one system prompt in the library, and a skill's files as documents."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLoadSystemPrompt",
            display_name="Load System Prompt",
            search_aliases=[
                "WASLoadSystemPrompt",
                "Load System Prompt",
                "system prompt",
                "system prompt library",
                "skill",
                "load skill",
                "skill references",
                "instructions",
                "LLM",
                "VLM",
            ],
            category="WAS Suite/Text/System Prompts",
            description=(
                "Load a saved system prompt or skill to steer a language or vision model, such "
                "as LLM Complete or Fast Generate Text. A skill also hands its reference files "
                "to the LLM index nodes, so a model can look things up in them while it works."
            ),
            inputs=[
                io.Combo.Input(
                    "system_prompt",
                    options=system_prompts.labels(),
                    tooltip=(
                        "Which saved prompt or skill to load, by its place in the library, such "
                        "as `Captioning/Detailed Caption.txt` or `my-skill/SKILL.md`. Picked from "
                        "the library drawn on the node, where Import adds a skill folder. The "
                        "files are in ComfyUI/user/was-node-suite/system_prompts."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="system_prompt",
                    tooltip="The prompt's text without its front matter, for a model's system_prompt input.",
                ),
                io.String.Output(
                    display_name="name",
                    tooltip="The prompt's name from its front matter, or its file name when it has none.",
                ),
                LLM_DOCUMENTS.Output(
                    display_name="documents",
                    tooltip=(
                        "A skill's other text files, such as `references/layout.md`, one document "
                        "each, for LLM Vector Index or another LLM index. Empty for a plain "
                        "prompt, and while the LLM nodes are off."
                    ),
                ),
            ],
        )

    @classmethod
    def validate_inputs(cls, system_prompt) -> bool | str:
        """Accept any key, so a prompt saved earlier in the same run is found when this node runs."""
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            return "Load System Prompt has no prompt chosen. Pick one from the library on the node"
        return True

    @classmethod
    def fingerprint_inputs(cls, system_prompt) -> str:
        """A stamp that moves whenever the chosen file, or any file of a skill, is edited."""
        entry = system_prompts.find(system_prompt, fresh=True)
        if entry is None:
            return str(system_prompt)
        return f"{entry.key}:{system_prompts.skill_stamp(entry)}"

    @classmethod
    def execute(cls, system_prompt) -> io.NodeOutput:
        """Read the chosen file, and a skill's files as documents while the LLM nodes are on.

        Raises:
            ValueError: The library is empty, the key names no file, or the file cannot be read.
        """
        key = str(system_prompt or "")
        if not key or key == system_prompts.NO_PROMPTS:
            raise ValueError(
                "the system prompt library is empty. Save a prompt with Save System Prompt, "
                "import a skill folder with Import on the node, or put .txt or .md files in "
                "ComfyUI/user/was-node-suite/system_prompts and press R to refresh the menu"
            )
        entry = system_prompts.find(key, fresh=True)
        if entry is None:
            raise ValueError(
                f"`{key}` is not in the system prompt library, "
                "ComfyUI/user/was-node-suite/system_prompts. Pick another prompt, or press R to "
                "refresh the menu"
            )
        text = system_prompts.read(entry).body.strip()
        if not text:
            logger.warning("the system prompt `%s` holds no text after its front matter", key)

        documents = wires.Documents((), entry.name if entry.skill else "")
        if entry.skill and group_enabled(LLM_FEATURE):
            made, skipped = _skill_documents(entry)
            documents = wires.Documents(tuple(made), f"{len(made)} files of {entry.name}")
            logger.info("the skill `%s` handed on %d document(s)", entry.key, len(made))
            if skipped:
                logger.info(
                    "the skill `%s` passed over %s",
                    entry.key, ", ".join(f"{count} {reason}" for reason, count in sorted(skipped.items())),
                )
        return io.NodeOutput(text, entry.name, documents)


def _skill_documents(entry: system_prompts.Entry) -> tuple[list, dict]:
    """One llama-index document per skill file, with the skill and the file's path as metadata."""
    from ....modules.llm import documents as loading

    read_files, skipped = system_prompts.read_skill_files(entry)
    made = []
    for found, body in read_files:
        name = found.relative.rsplit("/", 1)[-1]
        part = found.relative.split("/", 1)[0] if "/" in found.relative else ""
        metadata = {
            "file_name": name,
            "relative_path": found.relative,
            "skill": entry.name,
            "folder": f"{entry.folder} [system prompts]",
            "extension": "." + name.rsplit(".", 1)[-1].lower() if "." in name else "",
            "size": int(found.size),
            "source": "skill",
        }
        if part:
            metadata["section"] = part
        made.append(loading.make_document(body, metadata, f"skill:{entry.key}:{found.relative}"))
    return made, skipped
