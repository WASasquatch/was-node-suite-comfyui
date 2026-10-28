"""Save a system prompt into the system prompt library."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.interface import library_report, run_result
from ....modules.prompt import system_prompts

logger = log.get_logger("text.system_prompts")

#: What each save outcome is drawn as on the panel.
SUMMARIES = {
    "new": "saved `{key}`",
    "replaced": "replaced `{key}`",
    "unchanged": "`{key}` already held this text, so nothing changed",
}


class SaveSystemPrompt(io.ComfyNode):
    """Write a system prompt and its front matter into the library."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASSaveSystemPrompt",
            display_name="Save System Prompt",
            search_aliases=[
                "WASSaveSystemPrompt",
                "Save System Prompt",
                "system prompt",
                "system prompt library",
                "skill",
                "instructions",
            ],
            category="WAS Suite/Text/System Prompts",
            description=(
                "Save a system prompt into the library as a text or Markdown file, with its "
                "name, model family and description written as front matter. Load System "
                "Prompt lists it straight away. The saved text also passes on, so the same run "
                "can use it."
            ),
            inputs=[
                io.String.Input(
                    "system_prompt",
                    default="",
                    multiline=True,
                    placeholder="Eg: You describe pictures for an image model in one plain paragraph.",
                    tooltip=(
                        "The instructions to keep, typed or wired in. Front matter at the top "
                        "is kept, and name, model_family and description below are written "
                        "over it."
                    ),
                ),
                io.String.Input(
                    "name",
                    default="",
                    placeholder="Eg: Detailed Caption",
                    tooltip=(
                        "What the prompt is called in the library, and its file name. Eg: "
                        "`Detailed Caption` saves `Detailed Caption.txt`. Characters a file name "
                        "cannot hold become `_`."
                    ),
                ),
                io.String.Input(
                    "folder",
                    default="",
                    placeholder="Eg: Captioning/Qwen",
                    tooltip=(
                        "A folder inside the library, made when missing. Empty saves at the "
                        "top. Eg: `Captioning/Qwen`. A drive, a leading `/`, `..` or `Built-in`, "
                        "where the pack's own prompts are listed, is refused."
                    ),
                ),
                io.String.Input(
                    "model_family",
                    default="",
                    placeholder="Eg: Qwen, Gemma",
                    tooltip=(
                        "The model families the prompt suits, separated by commas, for "
                        "filtering in Load System Prompt. Eg: `Qwen, Gemma`. Empty keeps the "
                        "family the text's own front matter names, or none."
                    ),
                ),
                io.String.Input(
                    "description",
                    default="",
                    placeholder="Eg: One dense paragraph for training captions",
                    tooltip=(
                        "One line saying what the prompt is for, shown under its name in Load "
                        "System Prompt. Eg: `One dense paragraph for training captions`."
                    ),
                ),
                io.Combo.Input(
                    "file_type",
                    options=system_prompts.FILE_TYPES,
                    default="txt",
                    tooltip="`txt` = a plain text file; `md` = Markdown, as skill files are written.",
                ),
                io.Boolean.Input(
                    "overwrite",
                    default=False,
                    tooltip=(
                        "`true` replaces a saved prompt of the same name and folder. `false` "
                        "stops with an error instead, unless the text is identical."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="system_prompt",
                    tooltip="The text as saved, without front matter, for a model's system_prompt input.",
                ),
                io.String.Output(
                    display_name="name",
                    tooltip="The name the prompt is saved under.",
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        system_prompt="",
        name="",
        folder="",
        model_family="",
        description="",
        file_type="txt",
        overwrite=False,
    ) -> io.NodeOutput:
        """Write the prompt and report where it went.

        Raises:
            PathNotAllowed: The folder or name names somewhere outside the library.
            ValueError: The name or text is empty, or a different prompt has that name and
                overwrite is off.
        """
        saved = system_prompts.save(
            system_prompt,
            name,
            folder=folder,
            families=model_family,
            description=description,
            file_type=file_type,
            overwrite=bool(overwrite),
        )
        text = system_prompts.read(saved.entry).body.strip()
        logger.info("the system prompt `%s` is in the library (%s)", saved.entry.key, saved.status)
        library_report.publish(
            summary=SUMMARIES[saved.status].format(key=saved.entry.key),
            counts={
                "prompts": len(system_prompts.scan()),
                "characters": len(text),
                "lines": len(text.splitlines()),
            },
            facts={
                "file": saved.entry.key,
                "library": system_prompts.library_label(),
                "model family": ", ".join(saved.entry.families) or "any",
                "entry": saved.status,
            },
            lines=text.splitlines(),
            listing="text",
            status=run_result.OK,
        )
        return io.NodeOutput(text, saved.entry.name)
