"""Load one system prompt or skill from the system prompt library."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.compat.types import LLM_DOCUMENTS
from ....modules.config import group_enabled
from ....modules.io import picker
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
                "prompt enhance",
                "prompt upsampler",
                "built-in prompts",
                "LLM",
                "VLM",
            ],
            category="WAS Suite/Text/System Prompts",
            description=(
                "Load a saved or built-in system prompt or skill to steer a language or vision "
                "model, such as Fast Generate Text. The built-in prompts turn a short idea into a "
                "prompt written for one image or video model, such as FLUX.2, LTX-2 or MiniMax H3. "
                "A skill also hands its reference files to the LLM index nodes."
            ),
            inputs=[
                picker.path_input(io.Combo.Input(
                    "system_prompt",
                    options=system_prompts.labels(),
                    tooltip=(
                        "Which prompt or skill to load, by its place in the library, such as "
                        "`Captioning/Detailed Caption.txt` or `Built-in/FLUX/FLUX.2 Prompt "
                        "Upsampler.md`. `Built-in/` entries ship with the pack; Copy on the "
                        "library makes an editable copy in ComfyUI/user/was-node-suite/system_prompts."
                    ),
                )),
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
    def validate_inputs(cls, system_prompt=None) -> bool | str:
        """Accept any key, so a prompt saved earlier in the same run is found when this node runs."""
        if system_prompt is None:
            return True
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            return "Load System Prompt has no prompt chosen. Pick one from the library on the node"
        return True

    @classmethod
    def fingerprint_inputs(cls, system_prompt) -> str:
        """A stamp that moves whenever the chosen file, or any file of a skill, is edited."""
        try:
            entry = _chosen(system_prompt)
        except (OSError, ValueError):
            return str(system_prompt)
        if entry is None:
            return str(system_prompt)
        return f"{entry.key}:{system_prompts.skill_stamp(entry)}"

    @classmethod
    def execute(cls, system_prompt) -> io.NodeOutput:
        """Read the chosen file, and a skill's files as documents while the LLM nodes are on.

        Raises:
            ValueError: The library is empty, the key names no file, or the file cannot be read.
            PathNotAllowed: A wired path lies outside the library and every permitted read
                root, or names no ``.txt`` or ``.md`` file.
        """
        key = str(system_prompt or "")
        if not key or key == system_prompts.NO_PROMPTS:
            raise ValueError(
                "the system prompt library is empty. Save a prompt with Save System Prompt, "
                "import a skill with Import or Import ZIP on the node, or put .txt or .md files in "
                "ComfyUI/user/was-node-suite/system_prompts and press R to refresh the menu"
            )
        try:
            entry = _chosen(key)
        except OSError as error:
            raise ValueError(
                f"`{key}` could not be read ({error.strerror or 'no such file'}). Wire the path "
                "of a .txt or .md file that is there"
            ) from error
        if entry is None and key.startswith(f"{system_prompts.BUILTIN_FOLDER}/"):
            raise ValueError(
                f"`{key}` is not a built-in prompt of this version of WAS Node Suite. Pick another "
                "on the node, or update the pack if the workflow was saved with a newer one"
            )
        if entry is None:
            raise ValueError(
                f"`{key}` is not in the system prompt library, "
                "ComfyUI/user/was-node-suite/system_prompts. Pick another prompt, or press R to "
                "refresh the menu"
            )
        text = system_prompts.read(entry).body.strip()
        if not text:
            logger.warning("the system prompt `%s` holds no text after its front matter", entry.key)

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


def _chosen(value) -> system_prompts.Entry | None:
    """The entry a menu key or a wired path names, or ``None`` for a key the library lacks.

    Raises:
        PathNotAllowed: A wired path lies outside the library and every permitted read root.
        OSError: A wired file cannot be stated.
    """
    wired = picker.wired_file(value, system_prompts.EXTENSIONS, roots=system_prompts.library_roots())
    if wired is not None:
        return system_prompts.entry_at(wired)
    return system_prompts.find(str(value or ""), fresh=True)


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
