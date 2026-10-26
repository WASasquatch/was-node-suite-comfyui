"""Load one system prompt from the system prompt library."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.prompt import system_prompts

logger = log.get_logger("text.system_prompts")


class LoadSystemPrompt(io.ComfyNode):
    """Emit the text and name of one system prompt in the library."""

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
                "instructions",
                "LLM",
                "VLM",
            ],
            category="WAS Suite/Text/System Prompts",
            description=(
                "Load a saved system prompt to steer a language or vision model, such as Fast "
                "Generate Text or Generate Text. The node shows the library to search and pick "
                "from by folder and model family, with the chosen prompt's text. The text arrives "
                "without its front matter."
            ),
            inputs=[
                io.Combo.Input(
                    "system_prompt",
                    options=system_prompts.labels(),
                    tooltip=(
                        "Which saved prompt to load, by its place in the library, such as "
                        "`Captioning/Detailed Caption.txt`. Picked from the library drawn on the "
                        "node. The files are in ComfyUI/user/was-node-suite/system_prompts."
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
        """A stamp that moves whenever the chosen file is edited."""
        entry = system_prompts.find(system_prompt, fresh=True)
        if entry is None:
            return str(system_prompt)
        return f"{entry.key}:{entry.mtime_ns}:{entry.size}"

    @classmethod
    def execute(cls, system_prompt) -> io.NodeOutput:
        """Read the chosen file.

        Raises:
            ValueError: The library is empty, the key names no file, or the file cannot be read.
        """
        key = str(system_prompt or "")
        if not key or key == system_prompts.NO_PROMPTS:
            raise ValueError(
                "the system prompt library is empty. Save a prompt with Save System Prompt, or "
                "put .txt or .md files in ComfyUI/user/was-node-suite/system_prompts and press R "
                "to refresh the menu"
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
        return io.NodeOutput(text, entry.name)
