"""Fill a schema of named fields from text or pictures with a language model."""

from __future__ import annotations

import json

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import DICT, LLM
from ...modules.interface import run_result
from ...modules.llm import generate, runtime, structured, wires
from ...modules.llm import messages as llm_messages

REQUIRES = "llm"

logger = log.get_logger("llm.generate")

#: The node's name in messages.
NODE = "LLM Structured Output"

#: The nodes that make a language model, for messages.
LLM_HINT = "LLM From CLIP, LLM GGUF Loader or LLM OpenAI"

#: The schema a new node starts with.
DEFAULT_SCHEMA = (
    "subject: string  The main subject, in a few words\n"
    "mood: one of calm, tense, joyful, gloomy  The overall mood\n"
    "colours: list of string  The dominant colours\n"
    "people: integer  How many people appear"
)


class LLMStructuredOutput(io.ComfyNode):
    """Named fields filled by a language model and checked against a schema."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMStructuredOutput",
            display_name="LLM Structured Output",
            search_aliases=[
                "WASLLMStructuredOutput",
                "LLM Structured Output",
                "structured_predict",
                "as_structured_llm",
                "extract",
                "JSON",
                "schema",
                "pydantic",
            ],
            category="WAS Suite/LLM/Generate",
            is_output_node=True,
            description=(
                "Have a language model fill in named fields, such as a subject, a mood and a list "
                "of colours read from a caption or a picture, and get them back as JSON and a DICT. "
                "Every answer is checked against the schema before it leaves the node."
            ),
            inputs=[
                LLM.Input(
                    "llm",
                    tooltip=(
                        "The model that fills the fields, from LLM From CLIP, LLM GGUF Loader, LLM "
                        "OpenAI or another LLM model node."
                    ),
                ),
                io.String.Input(
                    "text", multiline=True, default="",
                    tooltip=(
                        "What to read the fields from, or the request itself, such as a caption, "
                        "an article or `Invent a fantasy character.`"
                    ),
                ),
                io.String.Input(
                    "schema", multiline=True, default=DEFAULT_SCHEMA,
                    tooltip=(
                        "One field per line, `name: type  description`, with types string, number, "
                        "integer, boolean, `list of string` or `one of a, b, c`; or a JSON Schema "
                        "object. At most 64 fields, nested 4 deep."
                    ),
                ),
                io.String.Input(
                    "instructions", multiline=True, default="", optional=True,
                    tooltip=(
                        "Standing instructions placed before the text, such as `Use British "
                        "spelling. Leave a field empty rather than guess.`"
                    ),
                ),
                io.Image.Input(
                    "image", optional=True,
                    tooltip=(
                        "Pictures to read the fields from, for a model that reads them. Every "
                        "frame of a batch is sent, up to 64."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="json",
                    tooltip="The filled fields as indented JSON text, such as for Save Text File or a prompt.",
                ),
                DICT.Output(
                    display_name="dict",
                    tooltip=(
                        "The same fields as a DICT, for Text Dictionary Get and the other "
                        "dictionary nodes."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, llm, text, schema, instructions="", image=None) -> io.NodeOutput:
        """Ask for the fields and check the answer.

        Args:
            llm: The ``LanguageModel`` that fills the fields.
            text: What to read from, or the request.
            schema: The fields, as lines or a JSON Schema object.
            instructions: Standing instructions, ``""`` for none.
            image: Pictures to read from, or ``None``.

        Returns:
            The fields as JSON text and as a dictionary.

        Raises:
            ValueError: The schema cannot be used, there is nothing to read, or the answer did not fit.
            RuntimeError: The model failed.
        """
        model = wires.expect(llm, wires.LanguageModel, NODE, "llm", LLM_HINT)
        wires.ready(model)
        text = str(text or "")
        if not text.strip() and image is None:
            raise ValueError(f"{NODE} has nothing to read. Type text, or connect an image.")
        pictures = (int(image.shape[0]) if image.ndim == 4 else 1) if image is not None else 0
        generate.require_vision(model, NODE, pictures)
        with runtime.guarded():
            parsed = structured.parse_schema(schema)
            request = llm_messages.message("user", text, images=image)
            outcome = structured.predict(model, str(instructions or ""), request, parsed, NODE)
        rendered = json.dumps(outcome.data, ensure_ascii=False, indent=2)
        logger.info(
            "%s: %s filled %d fields by %s in %d request(s), %.2fs",
            NODE, model.label, parsed.fields, outcome.mode, outcome.attempts, outcome.seconds,
        )
        facts = {"model": model.label, "mode": outcome.mode, "schema": parsed.source}
        if outcome.note:
            facts["function calling"] = "did not fit; JSON was asked for"
        bodies = [run_result.body("json", rendered)]
        if outcome.thinking:
            bodies.append(run_result.body("thinking", outcome.thinking))
        tries = "first try" if outcome.attempts == 1 else "second try, after one repair"
        run_result.publish(
            status=run_result.OK,
            summary=f"Filled {parsed.fields} field{'s' if parsed.fields != 1 else ''} by {outcome.mode}, on the {tries}.",
            counts={
                "fields": parsed.fields,
                "depth": parsed.depth,
                "requests": outcome.attempts + (1 if outcome.note else 0),
                "seconds": round(outcome.seconds, 2),
            },
            facts=facts,
            bodies=bodies,
        )
        return io.NodeOutput(rendered, outcome.data)
