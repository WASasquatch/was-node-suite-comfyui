"""Structured output: a schema typed on a node becomes a pydantic model, and an answer is held to it.

:func:`parse_schema` reads a JSON Schema object or ``name: type  description`` lines, and
:func:`predict` asks a model for an object that fits.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

from .. import log
from . import generate, runtime

__all__ = [
    "MAX_DEPTH",
    "MAX_FIELDS",
    "MAX_SCHEMA_CHARS",
    "Outcome",
    "Schema",
    "find_json",
    "parse_schema",
    "predict",
]

logger = log.get_logger("llm.generate")

#: Fields one schema declares at most, counted across every level.
MAX_FIELDS = 64

#: Objects nested inside one another at most, the top object included.
MAX_DEPTH = 4

#: Lists nested directly inside one another at most.
MAX_LIST_DEPTH = 4

#: Characters of schema text read at most.
MAX_SCHEMA_CHARS = 64 * 1024

#: Choices one ``one of`` field offers at most.
MAX_CHOICES = 64

#: Characters of one choice.
MAX_CHOICE_CHARS = 200

#: Characters of one field description.
MAX_DESCRIPTION_CHARS = 1000

#: Validation problems named in one message.
MAX_PROBLEMS = 8

#: Positions a JSON object is looked for at in an answer.
MAX_JSON_STARTS = 64

#: A field name: a letter or underscore, then letters, digits and underscores.
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,63}")

#: One ``name: type  description`` line.
LINE = re.compile(r"\s*([^:]+?)\s*:\s*(.*?)\s*")

#: The type words a line may use, and the JSON type each names.
SCALARS = {
    "string": "string", "str": "string", "text": "string",
    "number": "number", "float": "number",
    "integer": "integer", "int": "integer",
    "boolean": "boolean", "bool": "boolean",
}

#: Keys a JSON Schema may carry that the node does not read.
UNSUPPORTED = ("$ref", "allOf", "oneOf", "not", "if", "patternProperties")

#: The fenced block a model may wrap its JSON in.
FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)

#: What the model is told when it answers in JSON.
JSON_INSTRUCTION = (
    "Reply with one JSON object and nothing else: no prose before or after it and no code "
    "fence. The object must match this JSON Schema:\n"
)

#: What the model is told when its answer did not fit.
REPAIR = "That answer did not fit the schema: "


@dataclass(frozen=True)
class Schema:
    """A schema read off a node.

    Attributes:
        model: The pydantic model an answer is validated against.
        prompt: The same schema as JSON Schema, every object written inline.
        fields: Fields declared, counted across every level.
        depth: Objects nested inside one another, the top object included.
        source: ``JSON Schema`` or ``lines``.
    """

    model: type
    prompt: dict
    fields: int
    depth: int
    source: str


@dataclass(frozen=True)
class Outcome:
    """What a structured request produced.

    Attributes:
        data: The answer, validated, with the field names the schema uses.
        mode: ``function calling`` or ``JSON``.
        attempts: Requests made: 1, or 2 when the first answer was sent back to be repaired.
        thinking: What a reasoning model thought before its last answer.
        seconds: Time across every request.
        note: Why function calling was passed over, ``""`` when it was not.
    """

    data: dict
    mode: str
    attempts: int
    thinking: str
    seconds: float
    note: str = ""


class _Count:
    """Fields seen and the deepest object reached while reading a schema."""

    def __init__(self):
        self.fields = 0
        self.depth = 0
        self.models = 0


def parse_schema(text: str) -> Schema:
    """A schema from a JSON Schema object or from ``name: type  description`` lines.

    Args:
        text: The schema as typed on the node.

    Returns:
        The schema.

    Raises:
        ValueError: The text is empty, too long, or describes something that cannot be used.
    """
    text = str(text or "").strip()
    if not text:
        raise ValueError(
            "The schema is empty. Describe the fields to fill, one per line, such as "
            "`title: string  The film's title`, or paste a JSON Schema object."
        )
    if len(text) > MAX_SCHEMA_CHARS:
        raise ValueError(
            f"The schema is {len(text):,} characters and at most {MAX_SCHEMA_CHARS:,} are read. "
            f"Shorten it."
        )
    count = _Count()
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except RecursionError:
            raise ValueError("The schema nests too deeply to read. Flatten it.") from None
        except ValueError as error:
            raise ValueError(
                f"The schema starts with {{ but is not valid JSON ({error}). Fix the JSON, or "
                f"write one field per line, such as `title: string`."
            ) from None
        spec = _from_json(data, "", 0, 0, count)
        if spec["kind"] != "object":
            raise ValueError(
                "The top of the schema must be an object with properties. For a list, add one "
                "field of type array."
            )
        source = "JSON Schema"
        title = data.get("title") if isinstance(data.get("title"), str) else ""
    else:
        spec = _from_lines(text, count)
        source = "lines"
        title = ""
    title = title if NAME.fullmatch(title or "") else "StructuredOutput"
    description = spec.get("description") or "Record the answer as structured data."
    model = _model(spec, title, description, count)
    return Schema(model, _prompt(spec), count.fields, count.depth, source)


def _problem(where: str, message: str) -> ValueError:
    """A schema error naming where it is."""
    return ValueError(f"In the schema, {where} {message}")


def _description(node: dict) -> str:
    """A node's description, cut to :data:`MAX_DESCRIPTION_CHARS`."""
    value = node.get("description")
    return value.strip()[:MAX_DESCRIPTION_CHARS] if isinstance(value, str) else ""


def _choices(values, where: str) -> list:
    """The choices of an enum, checked."""
    if not isinstance(values, list) or not values:
        raise _problem(where, "has an enum that is not a list of choices.")
    if len(values) > MAX_CHOICES:
        raise _problem(where, f"offers {len(values)} choices and at most {MAX_CHOICES} are allowed.")
    kept = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise _problem(where, "has a choice that is neither text nor a whole number.")
        if isinstance(value, str) and len(value) > MAX_CHOICE_CHARS:
            raise _problem(where, f"has a choice longer than {MAX_CHOICE_CHARS} characters.")
        if value not in kept:
            kept.append(value)
    return kept


def _where(path: str) -> str:
    """How a message names a place in a schema."""
    return f"field '{path}'" if path else "the top object"


def _from_json(node, path: str, depth: int, lists: int, count: _Count) -> dict:
    """One JSON Schema node read into a field description."""
    where = _where(path)
    if not isinstance(node, dict):
        raise _problem(where, "must be an object describing a type, such as {\"type\": \"string\"}.")
    for key in UNSUPPORTED:
        if key in node:
            raise _problem(where, f"uses {key}, which is not supported. Write nested objects inline.")
    nullable = False
    if "anyOf" in node:
        options = node["anyOf"]
        others = [o for o in options if not (isinstance(o, dict) and o.get("type") == "null")] if isinstance(options, list) else []
        if not isinstance(options, list) or len(options) != 2 or len(others) != 1:
            raise _problem(where, "uses anyOf other than for a value that may be null, which is not supported.")
        node = {**{k: v for k, v in node.items() if k != "anyOf"}, **others[0]}
        nullable = True
    kind = node.get("type")
    if isinstance(kind, list):
        named = [k for k in kind if k != "null"]
        if len(named) != 1:
            raise _problem(where, "names several types, and one type, optionally with null, is supported.")
        nullable = nullable or "null" in kind
        kind = named[0]
    description = _description(node)
    if "enum" in node:
        values = _choices(node["enum"], where)
        return {"kind": "enum", "values": values, "description": description, "nullable": nullable}
    if kind is None:
        kind = "object" if "properties" in node else ("array" if "items" in node else None)
    if kind in ("string", "number", "integer", "boolean"):
        return {"kind": kind, "description": description, "nullable": nullable}
    if kind == "array":
        if lists + 1 > MAX_LIST_DEPTH:
            raise _problem(where, f"nests lists more than {MAX_LIST_DEPTH} deep.")
        items = node.get("items", {"type": "string"})
        inner = _from_json(items, f"{path}[]", depth, lists + 1, count)
        return {"kind": "array", "items": inner, "description": description, "nullable": nullable}
    if kind == "object":
        if depth + 1 > MAX_DEPTH:
            raise _problem(where, f"nests objects more than {MAX_DEPTH} deep. Flatten the schema.")
        count.depth = max(count.depth, depth + 1)
        properties = node.get("properties")
        if not isinstance(properties, dict) or not properties:
            raise _problem(where, "is an object with no properties. Give it at least one field.")
        required = node.get("required", [])
        if not isinstance(required, list) or not all(isinstance(name, str) for name in required):
            raise _problem(where, "has a required entry that is not a list of field names.")
        fields = {}
        for name, value in properties.items():
            inner = f"{path}.{name}" if path else str(name)
            _check_name(name, _where(inner))
            count.fields += 1
            if count.fields > MAX_FIELDS:
                raise ValueError(f"The schema declares more than {MAX_FIELDS} fields. Split it into smaller requests.")
            fields[name] = _from_json(value, inner, depth + 1, 0, count)
        return {
            "kind": "object",
            "properties": fields,
            "required": [name for name in properties if name in required],
            "description": description,
            "nullable": nullable,
        }
    raise _problem(
        where,
        f"has type {kind!r}, which is not one of string, number, integer, boolean, array or object."
        if kind is not None else "has no type. Add \"type\", such as \"string\".",
    )


def _check_name(name, label: str) -> None:
    """Refuse a field name that is not an identifier."""
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise _problem(
            label,
            "is not a usable field name. Use letters, digits and underscores, starting with a "
            "letter or underscore, up to 64 characters, such as `release_year`.",
        )


def _from_lines(text: str, count: _Count) -> dict:
    """``name: type  description`` lines read into an object description."""
    fields: dict = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = LINE.fullmatch(line)
        if not match:
            raise ValueError(
                f"Line {number} of the schema is not `name: type  description`, such as "
                f"`title: string  The film's title`."
            )
        name, rest = match.group(1), match.group(2)
        _check_name(name, f"line {number}, '{name}',")
        if name in fields:
            raise ValueError(f"Line {number} of the schema names '{name}' a second time.")
        count.fields += 1
        if count.fields > MAX_FIELDS:
            raise ValueError(f"The schema declares more than {MAX_FIELDS} fields. Split it into smaller requests.")
        fields[name] = _line_type(rest, number)
    if not fields:
        raise ValueError(
            "The schema names no fields. Write one per line, such as `title: string  The film's title`."
        )
    count.depth = 1
    return {"kind": "object", "properties": fields, "required": list(fields), "description": "", "nullable": False}


def _line_type(rest: str, number: int) -> dict:
    """The type and description after a line's colon."""
    parts = re.split(r"\s{2,}|\t", rest, maxsplit=1)
    spec = parts[0].strip()
    description = parts[1].strip()[:MAX_DESCRIPTION_CHARS] if len(parts) > 1 else ""
    lowered = spec.lower()
    if lowered.startswith("one of"):
        values = [value.strip() for value in spec[6:].split(",") if value.strip()]
        if not values:
            raise ValueError(f"Line {number} of the schema says `one of` and lists no choices, such as `one of happy, sad`.")
        return {"kind": "enum", "values": _choices(values, f"line {number}"), "description": description, "nullable": False}
    words = spec.split()
    listed = len(words) >= 3 and words[0].lower() == "list" and words[1].lower() == "of"
    word = (words[2] if listed else words[0] if words else "").lower()
    extra = " ".join(words[3:] if listed else words[1:])
    if extra and not description:
        description = extra[:MAX_DESCRIPTION_CHARS]
    kind = SCALARS.get(word) or (SCALARS.get(word[:-1]) if word.endswith("s") else None)
    if kind is None:
        raise ValueError(
            f"Line {number} of the schema has type `{spec[:40]}`. Use string, number, integer, "
            f"boolean, `list of string` (or of any of those) or `one of a, b, c`."
        )
    scalar = {"kind": kind, "description": "" if listed else description, "nullable": False}
    if listed:
        return {"kind": "array", "items": scalar, "description": description, "nullable": False}
    return scalar


def _model_name(base: str, count: _Count) -> str:
    """A name for a nested pydantic model, unique within one schema."""
    count.models += 1
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", base)[:48] or "Object"
    return f"{cleaned}_{count.models}"


def _python_type(spec: dict, name: str, count: _Count):
    """The Python type a field description stands for."""
    from typing import Literal, Optional

    kind = spec["kind"]
    if kind == "string":
        kind_type = str
    elif kind == "number":
        kind_type = float
    elif kind == "integer":
        kind_type = int
    elif kind == "boolean":
        kind_type = bool
    elif kind == "enum":
        kind_type = Literal[tuple(spec["values"])]
    elif kind == "array":
        kind_type = list[_python_type(spec["items"], f"{name}_item", count)]
    else:
        kind_type = _model(spec, _model_name(name, count), spec.get("description") or "", count)
    return Optional[kind_type] if spec.get("nullable") else kind_type


def _model(spec: dict, title: str, description: str, count: _Count):
    """A pydantic model for an object description, each field aliased to its schema name."""
    from typing import Optional

    from pydantic import Field, create_model

    fields = {}
    for index, (name, field) in enumerate(spec["properties"].items()):
        kind_type = _python_type(field, name, count)
        extra = {"alias": name}
        if field.get("description"):
            extra["description"] = field["description"]
        if name in spec["required"]:
            fields[f"f{index}"] = (kind_type, Field(**extra))
        else:
            fields[f"f{index}"] = (Optional[kind_type], Field(default=None, **extra))
    return create_model(title, __doc__=description or None, **fields)


def _prompt(spec: dict) -> dict:
    """A field description as plain JSON Schema, objects inline."""
    kind = spec["kind"]
    if kind == "enum":
        values = spec["values"]
        named = "integer" if all(isinstance(value, int) for value in values) else "string"
        node: dict = {"type": named, "enum": list(values)}
    elif kind == "array":
        node = {"type": "array", "items": _prompt(spec["items"])}
    elif kind == "object":
        node = {
            "type": "object",
            "properties": {name: _prompt(field) for name, field in spec["properties"].items()},
            "required": list(spec["required"]),
        }
    else:
        node = {"type": kind}
    if spec.get("nullable"):
        node["type"] = [node["type"], "null"]
    if spec.get("description"):
        node["description"] = spec["description"]
    return node


def find_json(text: str):
    """The first JSON object in a model's answer, or ``None``.

    Args:
        text: The answer, thinking already removed.

    Returns:
        A dictionary, or ``None`` when the answer holds no JSON object.
    """
    text = str(text or "").strip()
    candidates = [match.group(1).strip() for match in FENCE.finditer(text)]
    candidates.append(text)
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (ValueError, RecursionError):
            continue
        if isinstance(value, dict):
            return value
    decoder = json.JSONDecoder()
    start = text.find("{")
    tries = 0
    while start != -1 and tries < MAX_JSON_STARTS:
        tries += 1
        try:
            value, _ = decoder.raw_decode(text, start)
        except (ValueError, RecursionError):
            value = None
        if isinstance(value, dict):
            return value
        start = text.find("{", start + 1)
    return None


def _problems(error) -> str:
    """A validation error as short ``field: problem`` phrases."""
    found = []
    for item in error.errors()[:MAX_PROBLEMS]:
        where = ".".join(str(part) for part in item.get("loc", ())) or "the answer"
        found.append(f"{where}: {item.get('msg', 'does not fit')}")
    more = len(error.errors()) - len(found)
    if more > 0:
        found.append(f"and {more} more")
    return "; ".join(found)


def _validated(schema: Schema, value) -> tuple[dict | None, str]:
    """An answer checked against the schema: ``(data, "")`` or ``(None, problem)``."""
    from pydantic import BaseModel, ValidationError

    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", by_alias=True)
    if not isinstance(value, dict):
        return None, "no JSON object was found in the answer"
    try:
        checked = schema.model.model_validate(value)
    except ValidationError as error:
        return None, _problems(error)
    return checked.model_dump(mode="json", by_alias=True), ""


def _interrupted(error: BaseException) -> bool:
    """Whether an exception is ComfyUI stopping the run."""
    return type(error).__name__ == "InterruptProcessingException"


def predict(model, instructions: str, request, schema: Schema, node: str) -> Outcome:
    """Ask for an object fitting a schema, by function call or by JSON with one repair.

    Args:
        model: The ``LanguageModel``.
        instructions: System text placed before the request, ``""`` for none.
        request: The user ``ChatMessage``: what to extract from, with any pictures.
        schema: What the answer must fit.
        node: The node's display name, for errors.

    Returns:
        The validated answer.

    Raises:
        ValueError: The answer did not fit the schema after the repair.
        RuntimeError: The model failed, its message cleared of keys.
    """
    from llama_index.core.llms import ChatMessage, MessageRole

    started = time.monotonic()
    note = ""
    metadata = getattr(model.llm, "metadata", None)
    if model.function_calling and getattr(metadata, "is_function_calling_model", False):
        chat = ([ChatMessage(role=MessageRole.SYSTEM, content=instructions)] if instructions else []) + [request]
        try:
            response = model.llm.as_structured_llm(schema.model).chat(chat)
            data, problem = _validated(schema, getattr(response, "raw", None))
        except (ValueError, TypeError) as error:
            data, problem = None, runtime.describe_error(error)[:300]
        except Exception as error:
            if _interrupted(error):
                raise
            raise RuntimeError(f"{node} failed: {runtime.describe_error(error)}") from None
        if data is not None:
            return Outcome(data, "function calling", 1, "", time.monotonic() - started)
        note = f"the function call did not fit ({problem})"
        logger.debug("%s: function calling did not fit, asking for JSON", node)

    system = (instructions.strip() + "\n\n" if instructions.strip() else "") + JSON_INSTRUCTION
    system += json.dumps(schema.prompt, ensure_ascii=False, indent=2)
    chat = [ChatMessage(role=MessageRole.SYSTEM, content=system), request]
    problem = ""
    thinking = ""
    for attempt in (1, 2):
        answer = generate.reply(model, chat, node)
        thinking = answer.thinking or thinking
        data, problem = _validated(schema, find_json(answer.text))
        if data is not None:
            return Outcome(data, "JSON", attempt, thinking, time.monotonic() - started, note)
        chat = [
            *chat,
            ChatMessage(role=MessageRole.ASSISTANT, content=answer.text or "(no answer)"),
            ChatMessage(
                role=MessageRole.USER,
                content=REPAIR + problem + ". Reply with only the corrected JSON object.",
            ),
        ]
    raise ValueError(
        f"{node}: the model's answer did not fit the schema, even after it was sent back once to "
        f"be fixed. {problem[:1].upper() + problem[1:]}. Try a larger model or one that follows "
        f"instructions closely, simplify the schema, or describe the fields more in instructions."
    )
