"""Compile a MOSS-Image caption or layout draft into the prompt MOSS-Image reads."""

from __future__ import annotations

import json

from comfy_api.latest import io

from ...modules import log
from ...modules.image import box_layout
from ...modules.interface import run_result
from ...modules.prompt import moss_draft

logger = log.get_logger("text.moss_image")

#: Characters of the caption drawn on the panel.
SHOWN_CHARS = 4000

#: Boxes listed one per line on the panel.
SHOWN_ITEMS = 24


class MOSSImageCompilePrompt(io.ComfyNode):
    """Turn a model's caption or layout draft into a MOSS-Image prompt."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASMOSSImageCompilePrompt",
            display_name="MOSS-Image Compile Prompt",
            search_aliases=[
                "WASMOSSImageCompilePrompt",
                "MOSS-Image Compile Prompt",
                "MOSS layout",
                "MOSS prompt",
                "prompt enhancer",
                "layout draft",
                "skill",
            ],
            category="WAS Suite/Text",
            description=(
                "Turn the caption or box layout a language model drafted, such as with the "
                "MOSS-Image prompt enhancer skill, into the prompt MOSS-Image reads. Layout "
                "boxes are placed on the image size being generated. The panel lists the "
                "caption and every box."
            ),
            inputs=[
                io.String.Input(
                    "draft",
                    multiline=True,
                    dynamic_prompts=False,
                    default="",
                    tooltip=(
                        "The model's answer holding `{\"caption\": \"...\"}` or a layout with "
                        "`summary`, `items` and boxes in pixels. Words or code fences around the "
                        "JSON are ignored; text with no JSON becomes the caption."
                    ),
                ),
                io.Int.Input(
                    "width",
                    default=1024, min=16, max=box_layout.MAX_FRAME, step=16,
                    tooltip="Width of the image being generated in pixels, such as `2560`. Layout boxes are placed on it.",
                ),
                io.Int.Input(
                    "height",
                    default=1024, min=16, max=box_layout.MAX_FRAME, step=16,
                    tooltip="Height of the image being generated in pixels, such as `1440`. Layout boxes are placed on it.",
                ),
                io.Combo.Input(
                    "on_resize",
                    options=list(box_layout.RESIZE_MODES),
                    default="fit",
                    tooltip=(
                        "How a layout drafted at another size is placed: `fit` = scaled whole "
                        "and centred; `fill` = scaled to cover; `stretch` = each side scaled; "
                        "`keep size` = centred unscaled."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="prompt",
                    tooltip=(
                        "The caption, or the layout as one line of JSON, which MOSS-Image reads "
                        "as it is. Wire it straight into CLIP Text Encode with a MOSS-Image text "
                        "encoder."
                    ),
                ),
                io.String.Output(
                    display_name="caption",
                    tooltip="The caption as plain text, also for a layout, for a preview or a node that reads plain text.",
                ),
            ],
        )

    @classmethod
    def execute(cls, draft, width, height, on_resize="fit") -> io.NodeOutput:
        """Compile the draft and report it.

        Raises:
            ValueError: The draft is empty or breaks the layout format.
        """
        try:
            done = moss_draft.compile_draft(draft, (int(width), int(height)), str(on_resize))
        except ValueError as error:
            raise ValueError(f"MOSS-Image Compile Prompt: {error}.") from None
        counts = {"characters": len(done.prompt)}
        facts = {"draft": done.kind, "image": f"{int(width)}x{int(height)}"}
        if done.kind == moss_draft.LAYOUT:
            counts["boxes"] = len(done.items)
            facts["drafted at"] = f"{done.frame[0]}x{done.frame[1]}"
        if done.notes:
            facts["notes"] = "; ".join(done.notes)[:600]
        if not done.complete:
            logger.warning("MOSS-Image Compile Prompt: %s", moss_draft.CUT_NOTE)
        bodies = [run_result.body("caption", done.caption[:SHOWN_CHARS])]
        if done.items:
            lines = []
            for number, item in enumerate(done.items[:SHOWN_ITEMS], 1):
                written = f" \"{item['text']}\"" if "text" in item else ""
                lines.append(f"{number}. {item['label']} {item['bbox']}{written}: {item.get('caption', '')}")
            if len(done.items) > SHOWN_ITEMS:
                lines.append(f"and {len(done.items) - SHOWN_ITEMS} more")
            bodies.append(run_result.body("boxes", "\n".join(lines)))
            bodies.append(run_result.body("prompt", json.dumps(json.loads(done.prompt), indent=2, ensure_ascii=False)))
        logger.info("MOSS-Image Compile Prompt compiled a %s draft into %d characters", done.kind, len(done.prompt))
        run_result.publish(
            status=run_result.WARNING if done.kind == moss_draft.PLAIN or not done.complete else run_result.OK,
            summary=(
                "The draft was cut off; only the part that closed was compiled" if not done.complete
                else f"Compiled a layout of {len(done.items)} box(es)" if done.kind == moss_draft.LAYOUT
                else "No JSON draft found; the text is the caption" if done.kind == moss_draft.PLAIN
                else "Compiled a caption"
            ),
            counts=counts,
            facts=facts,
            bodies=bodies,
        )
        return io.NodeOutput(done.prompt, done.caption)
