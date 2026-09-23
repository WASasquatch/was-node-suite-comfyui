"""Reading a Three.js module file: its header, the inputs it declares and its body.

``parse`` reads the comment header a module opens with, ``screen`` refuses a body reaching
outside the scene, and ``load`` takes a menu label.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "KINDS",
    "MAX_BYTES",
    "MAX_OPTIONS",
    "HEADER_LINES",
    "MARKER",
    "NONE",
    "SUFFIXES",
    "VERSION",
    "VALUE_KINDS",
    "WIRE_KINDS",
    "Declaration",
    "chosen",
    "declares",
    "load",
    "options",
    "parse",
    "screen",
]

#: Suffixes a module file may carry. ``.txt`` is offered so a browser never loads one as a
#: script from a static path.
SUFFIXES = (".js", ".txt")

#: The first line of every module file, followed by the format version.
MARKER = "was-threejs-module"

#: The format this reads.
VERSION = 1

#: Longest a module file may be.
MAX_BYTES = 256 * 1024

#: Most module files a menu offers.
MAX_OPTIONS = 2000

#: What the menu says when the read roots hold no module.
NONE = "no module files found"

#: Lines read from the top of a candidate file while looking for the marker.
HEADER_LINES = 8

#: What a module builds, and the node family that offers it.
KINDS = ("material", "geometry", "object", "update", "module")

#: Directives naming a wired input, and the socket each one takes.
WIRE_KINDS = {
    "texture": "THREE_TEXTURE",
    "material": "THREE_MATERIAL",
    "geometry": "THREE_GEOMETRY",
    "object": "THREE_OBJECT",
}

#: Directives naming a widget, and the value each one holds.
VALUE_KINDS = ("number", "string", "boolean")

#: A header line, as ``// @texture albedo, normal``.
_DIRECTIVE = re.compile(r"^\s*//\s*@(\w+)\s*(.*)$")

#: The marker line, as ``// was-threejs-module 1``.
_MARKER_LINE = re.compile(rf"^\s*//\s*{re.escape(MARKER)}\s+(\d+)\s*$")

#: A comment line carrying no directive.
_COMMENT = re.compile(r"^\s*(//.*)?$")

#: An identifier a declared input may be called. It becomes a javascript parameter name.
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

#: Names a module body may not hold, and what each one reaches. Matched on word boundaries
#: against the source. This catches a body written without care; it is not a boundary, since
#: javascript can build any of these names at runtime.
REFUSED = (
    (r"\bfetch\s*\(", "makes a network request"),
    (r"\bXMLHttpRequest\b", "makes a network request"),
    (r"\bWebSocket\b", "opens a socket"),
    (r"\bEventSource\b", "opens a stream"),
    (r"\bsendBeacon\b", "posts to a server"),
    (r"\bimport\s*\(", "loads another module"),
    (r"\bimportScripts\b", "loads another script"),
    (r"\bWorker\s*\(", "starts a worker"),
    (r"\beval\s*\(", "runs assembled code"),
    (r"\bnew\s+Function\b", "runs assembled code"),
    (r"\bdocument\s*\.\s*cookie\b", "reads cookies"),
    (r"\blocalStorage\b", "reads stored data"),
    (r"\bsessionStorage\b", "reads stored data"),
    (r"\bindexedDB\b", "reads stored data"),
    (r"\bwindow\s*\.\s*parent\b", "reaches the page around the scene"),
    (r"\bframeElement\b", "reaches the page around the scene"),
    (r"(?<![\w.])parent\s*\.", "reaches the page around the scene"),
    (r"(?<![\w.])top\s*\.", "reaches the page around the scene"),
    (r"(?<![\w.])opener\b", "reaches the page around the scene"),
    (r"\bcomfyAPI\b", "reaches ComfyUI"),
    (r"(?<![\w.])api\s*\.", "reaches ComfyUI"),
)

_REFUSED = tuple((re.compile(pattern), reason) for pattern, reason in REFUSED)


@dataclass
class Input:
    """One input a module declares.

    Attributes:
        name: The javascript parameter the value arrives as.
        kind: A key of :data:`WIRE_KINDS`, or one of :data:`VALUE_KINDS`.
        default: The widget's starting value, or None for a wired input.
    """

    name: str
    kind: str
    default: object = None


@dataclass
class Declaration:
    """What a module file asks for and the body it carries.

    Attributes:
        version: The format version its marker named.
        kind: One of :data:`KINDS`.
        inputs: Every declared input, in the order the header named them.
        body: The javascript after the header.
    """

    version: int
    kind: str
    inputs: list[Input] = field(default_factory=list)
    body: str = ""

    @property
    def wires(self) -> list[Input]:
        """The declared inputs that take a wire."""
        return [one for one in self.inputs if one.kind in WIRE_KINDS]

    @property
    def values(self) -> list[Input]:
        """The declared inputs that draw a widget."""
        return [one for one in self.inputs if one.kind in VALUE_KINDS]


def _value(kind: str, text: str, name: str):
    """One widget's starting value, read from its directive.

    Args:
        kind: One of :data:`VALUE_KINDS`.
        text: What followed the name on the directive line.
        name: The input's name, for the message.

    Returns:
        The default, typed for its kind.

    Raises:
        ValueError: The text does not hold that kind of value.
    """
    written = text.strip().strip('"').strip("'")
    if kind == "string":
        return written
    if kind == "boolean":
        if written.lower() in ("true", "1", "yes"):
            return True
        if written.lower() in ("false", "0", "no", ""):
            return False
        raise ValueError(
            f"`@boolean {name} {written}` has no true or false value. Write `true` or `false`"
        )
    if not written:
        return 0.0
    try:
        return float(written)
    except ValueError as error:
        raise ValueError(
            f"`@number {name} {written}` has no number in it. Write a value such as `0.4`"
        ) from error


def parse(source: str) -> Declaration:
    """The header of a module file, and the body under it.

    Args:
        source: The whole file.

    Returns:
        What the file declares.

    Raises:
        ValueError: The marker is missing, the version is not read here, the kind is not one
            of :data:`KINDS`, or a directive is not understood.
    """
    lines = source.splitlines()
    at = 0
    while at < len(lines) and not lines[at].strip():
        at += 1
    if at >= len(lines):
        raise ValueError(
            f"this file is empty. A module starts with `// {MARKER} {VERSION}`"
        )
    marked = _MARKER_LINE.match(lines[at])
    if not marked:
        raise ValueError(
            f"this file does not start with `// {MARKER} {VERSION}`, so it is not a Three "
            f"module. Add that line at the top"
        )
    version = int(marked.group(1))
    if version != VERSION:
        raise ValueError(
            f"this module is written for format {version} and {VERSION} is read here. "
            f"Change its first line to `// {MARKER} {VERSION}`"
        )

    kind = ""
    inputs: list[Input] = []
    seen: set[str] = set()
    at += 1
    while at < len(lines):
        line = lines[at]
        found = _DIRECTIVE.match(line)
        if found is None:
            if _COMMENT.match(line):
                at += 1
                continue
            break
        at += 1
        directive, rest = found.group(1).lower(), found.group(2).strip()
        if directive == "kind":
            if rest not in KINDS:
                raise ValueError(
                    f"`@kind {rest}` is not a kind this builds. Write one of: "
                    f"{', '.join(KINDS)}"
                )
            kind = rest
            continue
        if directive in WIRE_KINDS:
            names = [one.strip() for one in rest.replace(",", " ").split() if one.strip()]
            if not names:
                raise ValueError(f"`@{directive}` names no input. Write `@{directive} albedo`")
            for name in names:
                if not _NAME.match(name):
                    raise ValueError(
                        f"`{name}` is not a name a value can arrive as. Use letters, digits "
                        f"and underscores, starting with a letter"
                    )
                if name in seen:
                    raise ValueError(f"`{name}` is declared twice, and each name arrives once")
                seen.add(name)
                inputs.append(Input(name=name, kind=directive))
            continue
        if directive in VALUE_KINDS:
            parts = rest.split(None, 1)
            if not parts:
                raise ValueError(f"`@{directive}` names no input. Write `@{directive} amount 1`")
            name = parts[0]
            if not _NAME.match(name):
                raise ValueError(
                    f"`{name}` is not a name a value can arrive as. Use letters, digits and "
                    f"underscores, starting with a letter"
                )
            if name in seen:
                raise ValueError(f"`{name}` is declared twice, and each name arrives once")
            seen.add(name)
            inputs.append(
                Input(
                    name=name,
                    kind=directive,
                    default=_value(directive, parts[1] if len(parts) > 1 else "", name),
                )
            )
            continue
        raise ValueError(
            f"`@{directive}` is not a directive this reads. Write one of: kind, "
            f"{', '.join(WIRE_KINDS)}, {', '.join(VALUE_KINDS)}"
        )

    if not kind:
        raise ValueError(
            f"this module names no kind. Add `// @kind material`, or one of: "
            f"{', '.join(KINDS)}"
        )
    return Declaration(
        version=version, kind=kind, inputs=inputs, body="\n".join(lines[at:])
    )


def screen(body: str) -> None:
    """Refuse a body holding a name that reaches outside the scene.

    Args:
        body: The javascript under the header.

    Raises:
        ValueError: The body holds one of :data:`REFUSED`.
    """
    for pattern, reason in _REFUSED:
        found = pattern.search(body)
        if found is None:
            continue
        line = body[: found.start()].count("\n") + 1
        raise ValueError(
            f"line {line} of this module holds `{found.group(0).strip()}`, which "
            f"{reason}. A module builds a scene from what it is given, and the values it "
            f"needs arrive as the inputs its header declares"
        )


def read(path: "Path") -> Declaration:
    """A module file's declaration, with its body screened.

    Args:
        path: The file to read, already resolved inside a permitted root.

    Returns:
        What the file declares.

    Raises:
        ValueError: The file is larger than :data:`MAX_BYTES`, its header is not read here,
            or its body holds a refused name.
        OSError: The file could not be read.
    """
    target = Path(path)
    size = target.stat().st_size
    if size > MAX_BYTES:
        raise ValueError(
            f"`{target.name}` is {size} bytes, longer than the {MAX_BYTES} a module may be"
        )
    declared = parse(target.read_text(encoding="utf-8", errors="replace"))
    screen(declared.body)
    return declared


def declares(path) -> bool:
    """Whether a file opens with the module marker.

    Args:
        path: The file to look at.

    Returns:
        True when its first line that is not blank carries the marker.
    """
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for _ in range(HEADER_LINES):
                line = handle.readline()
                if not line:
                    return False
                if line.strip():
                    return _MARKER_LINE.match(line) is not None
    except OSError:
        return False
    return False


def options() -> list[str]:
    """Every file opening with the marker, as menu labels.

    Returns:
        Labels, or ``[NONE]`` where the read roots hold no module. The marker decides
        membership, not the suffix.
    """
    from ..util import file_listing

    found = [
        entry.label
        for entry in file_listing.view(SUFFIXES, file_listing.ROOTS, MAX_OPTIONS)
        if entry.size <= MAX_BYTES and declares(entry.path)
    ]
    return found or [NONE]


def chosen(label: str) -> bool:
    """Whether a menu value names a module rather than :data:`NONE`."""
    written = (label or "").strip()
    return bool(written) and written != NONE


def load(label: str) -> Declaration:
    """The module one menu label names.

    Args:
        label: The widget's value, as the file menu offered it.

    Returns:
        What the file declares.

    Raises:
        PermissionError: ``threejs.allow_scripts`` is off.
        ValueError: Nothing is chosen, the label names no module that is there, or the file
            is not one this reads.
        OSError: The file could not be read.
    """
    from ..config import group_enabled
    from ..util import file_listing, sandbox
    from .spec import ALLOW_SCRIPTS

    if not group_enabled(ALLOW_SCRIPTS):
        raise PermissionError(
            "a module is javascript this browser runs, and threejs.allow_scripts is off in "
            "config.yaml. Set it to true to load one"
        )
    chosen = (label or "").strip()
    if not chosen:
        raise ValueError(
            "no module was chosen. Pick one from the file list. A folder added under "
            "paths.allow_read in config.yaml appears there under its own name"
        )
    found = file_listing.resolve(chosen, SUFFIXES, tags=file_listing.ROOTS)
    if found is None:
        raise ValueError(
            f"`{chosen}` names no module that is there any more. Pick another from the file "
            f"list, or add its folder to paths.allow_read in config.yaml"
        )
    return read(sandbox.resolve_read(found))
