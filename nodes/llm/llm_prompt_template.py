"""Fill the ``{name}`` fields of a prompt template from a dictionary and text inputs."""

from __future__ import annotations

import re

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import DICT
from ...modules.interface import run_result
from ...modules.llm import generate

REQUIRES = "llm"

logger = log.get_logger("llm.generate")

#: The node's name in messages.
NODE = "LLM Prompt Template"

#: A field name a template can use.
FIELD_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")

#: The text inputs, each usable as the field of the same name.
VALUE_SLOTS = ("value_a", "value_b", "value_c", "value_d")


class LLMPromptTemplate(io.ComfyNode):
    """A prompt with its ``{name}`` fields filled in."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMPromptTemplate",
            display_name="LLM Prompt Template",
            search_aliases=[
                "WASLLMPromptTemplate",
                "LLM Prompt Template",
                "PromptTemplate",
                "template",
                "fill in",
                "variables",
            ],
            category="WAS Suite/LLM/Generate",
            description=(
                "Write a prompt once with {name} fields and fill them from a DICT or from text "
                "wired in, such as one template for a whole batch of subjects. Fields with no "
                "value are listed on their own output."
            ),
            inputs=[
                io.String.Input(
                    "template", multiline=True, default="Describe {value_a} in the style of {value_b}.",
                    tooltip=(
                        "The prompt, with fields such as `{subject}` or `{value_a}`. Write `{{` and "
                        "`}}` for a brace that is not a field."
                    ),
                ),
                DICT.Input(
                    "variables", optional=True,
                    tooltip=(
                        "Values for the fields by name, from Text Dictionary New or LLM Structured "
                        "Output: a key `subject` fills `{subject}`. A list is joined with commas."
                    ),
                ),
                io.String.Input(
                    "value_a", force_input=True, optional=True,
                    tooltip="Text that fills `{value_a}`, and wins over a DICT key of the same name.",
                ),
                io.String.Input(
                    "value_b", force_input=True, optional=True,
                    tooltip="Text that fills `{value_b}`, and wins over a DICT key of the same name.",
                ),
                io.String.Input(
                    "value_c", force_input=True, optional=True,
                    tooltip="Text that fills `{value_c}`, and wins over a DICT key of the same name.",
                ),
                io.String.Input(
                    "value_d", force_input=True, optional=True,
                    tooltip="Text that fills `{value_d}`, and wins over a DICT key of the same name.",
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="text",
                    tooltip="The prompt with every field that has a value filled in, for LLM Complete or CLIP Text Encode.",
                ),
                io.String.Output(
                    display_name="missing",
                    tooltip=(
                        "The fields left unfilled, comma separated, such as `style, lens`. Empty "
                        "when every field had a value."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls, template, variables=None, value_a=None, value_b=None, value_c=None, value_d=None
    ) -> io.NodeOutput:
        """Fill the template.

        Args:
            template: The prompt holding ``{name}`` fields.
            variables: Field values by name, or ``None``.
            value_a: Text for ``{value_a}``, or ``None``.
            value_b: Text for ``{value_b}``, or ``None``.
            value_c: Text for ``{value_c}``, or ``None``.
            value_d: Text for ``{value_d}``, or ``None``.

        Returns:
            The filled prompt and the fields left empty.

        Raises:
            ValueError: ``variables`` is not a dictionary, or the result is too long.
        """
        if variables is not None and not isinstance(variables, dict):
            raise ValueError(
                f"{NODE} was handed a {type(variables).__name__} on its variables input, which it "
                f"cannot use. Connect a DICT, such as from Text Dictionary New."
            )
        values = {}
        for key, value in (variables or {}).items():
            if isinstance(key, str) and FIELD_NAME.fullmatch(key):
                values[key] = value
        for name, value in zip(VALUE_SLOTS, (value_a, value_b, value_c, value_d)):
            if value is not None:
                values[name] = str(value)
        text, filled, missing = generate.fill(str(template or ""), values)
        logger.debug("%s filled %d fields; %d left empty", NODE, len(filled), len(missing))
        facts = {"filled": ", ".join(filled) or "none", "missing": ", ".join(missing) or "none"}
        run_result.publish(
            status=run_result.WARNING if missing else run_result.OK,
            summary=(
                f"Filled {len(filled)} field{'s' if len(filled) != 1 else ''}"
                + (f"; {len(missing)} had no value." if missing else ".")
            ),
            counts={"filled": len(filled), "missing": len(missing), "characters": len(text)},
            facts=facts,
            bodies=[run_result.body("text", text)],
        )
        return io.NodeOutput(text, ", ".join(missing))
