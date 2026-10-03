"""Locating saved app workflows and reading which inputs and outputs each exposes.

An app workflow is a saved workflow named ``*.app.json``. Its ``extra.linearData``
holds ``inputs`` as ``[widget_id, label]`` pairs and ``outputs`` as node ids. A widget
id reads ``"<definition_id>:<node_id>:<widget_name>"``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..config import paths

#: Name ending that marks a saved workflow as an app.
SUFFIX = ".app.json"

#: Definition id a widget id carries for a node in the root graph.
ROOT_DEFINITION = "00000000-0000-0000-0000-000000000000"

#: Ending shared by the widget names that draw a panel rather than hold a value.
PANEL_SUFFIX = "_ui"

#: What a file is named for each type a wire may stand in for, so a menu of enum words is
#: told from one of files, and a menu of fonts from one of pictures.
READ_EXTENSIONS = {
    "IMAGE": (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".exr"),
    "MASK": (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".exr"),
    "LATENT": (".latent",),
    "AUDIO": (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".mp4", ".webm"),
    "VIDEO": (".mp4", ".webm", ".mkv", ".mov", ".avi", ".m4v", ".gif"),
    "MODEL": (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf"),
    "CLIP": (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf"),
    "VAE": (".safetensors", ".ckpt", ".pt", ".pth"),
    "CLIP_VISION": (".safetensors", ".ckpt", ".pt", ".pth", ".bin"),
    "CONTROL_NET": (".safetensors", ".ckpt", ".pt", ".pth"),
    "STYLE_MODEL": (".safetensors", ".ckpt", ".pt", ".pth"),
    "UPSCALE_MODEL": (".safetensors", ".ckpt", ".pt", ".pth", ".onnx"),
    "GLIGEN": (".safetensors", ".ckpt", ".pt", ".pth"),
}

#: Types a wire may stand in for when a node reads one rather than being given it.
CARRIED_TYPES = tuple(READ_EXTENSIONS)

#: Of those, the ones a graph can only ever load, never build. A menu naming these is
#: read as a file even while the folder it lists is empty.
LOADED_TYPES = (
    "MODEL", "CLIP", "VAE", "CLIP_VISION", "CONTROL_NET", "STYLE_MODEL",
    "UPSCALE_MODEL", "GLIGEN",
)


@dataclass(frozen=True)
class ExposedInput:
    """One input an app offers.

    Attributes:
        widget_id: The exposure key, ``"<definition_id>:<node_id>:<widget_name>"``.
        label: The name the app presents it under.
        definition: Subgraph definition id, or ``None`` for the root graph.
        node_id: Id of the node the widget sits on, within its own graph.
        widget: Name of the widget on that node.
    """

    widget_id: str
    label: str
    definition: str | None
    node_id: str
    widget: str


@dataclass(frozen=True)
class Exposure:
    """What an app workflow presents.

    Attributes:
        inputs: Exposed inputs, in the order the app lists them.
        outputs: Ids of the nodes whose results the app presents.
        panels: Exposed widget names that draw a panel rather than hold a value.
    """

    inputs: tuple[ExposedInput, ...]
    outputs: tuple[str, ...]
    panels: tuple[str, ...]


def directory() -> Path | None:
    """The directory saved workflows are kept in, or ``None`` when ComfyUI is absent."""
    user = paths.comfyui_user_directory()
    return None if user is None else user / "default" / "workflows"


def discover() -> list[str]:
    """Every app workflow found, as a path relative to the workflows directory."""
    root = directory()
    if root is None or not root.is_dir():
        return []
    found = [p.relative_to(root).as_posix() for p in root.rglob(f"*{SUFFIX}") if p.is_file()]
    return sorted(found)


def load(name: str) -> dict:
    """Read one app workflow.

    Args:
        name: A path relative to the workflows directory.

    Returns:
        The saved workflow.

    Raises:
        FileNotFoundError: No workflows directory, or no such file within it.
        ValueError: The path leaves the workflows directory, or the file is not JSON.
    """
    root = directory()
    if root is None:
        raise FileNotFoundError(
            "no ComfyUI user directory was found, so no saved workflow can be read"
        )
    from ..util import sandbox

    reason = sandbox.leaves_folder(name)
    if reason is not None:
        raise ValueError(f"{name!r} {reason}. Name a workflow inside {root}")
    target = (root / name).resolve()
    if root.resolve() not in (target, *target.parents):
        raise ValueError(
            f"{name!r} resolves outside the workflows directory. Name a workflow inside "
            f"{root}"
        )
    if not target.is_file():
        raise FileNotFoundError(
            f"no saved workflow named {name!r} in {root}. Save one from the workflow menu, "
            f"with a name ending {SUFFIX}"
        )
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{name!r} is not a readable workflow: {error}") from error


def exposure(workflow: dict) -> Exposure:
    """Read an app workflow's exposure map.

    Args:
        workflow: A saved workflow.

    Returns:
        Its exposed inputs and outputs. Both are empty for a workflow saved without
        app mode.
    """
    linear = (workflow.get("extra") or {}).get("linearData") or {}
    inputs, panels = [], []
    for entry in linear.get("inputs") or []:
        widget_id, label = (entry + [None])[:2] if isinstance(entry, list) else (entry, None)
        parts = str(widget_id).split(":")
        if len(parts) != 3:
            continue
        definition, node_id, widget = parts
        if widget.endswith(PANEL_SUFFIX):
            panels.append(widget)
            continue
        inputs.append(
            ExposedInput(
                widget_id=str(widget_id),
                label=str(label or widget),
                definition=None if definition == ROOT_DEFINITION else definition,
                node_id=node_id,
                widget=widget,
            )
        )
    outputs = tuple(str(node_id) for node_id in linear.get("outputs") or [])
    return Exposure(tuple(inputs), outputs, tuple(panels))


def targets(exposed: ExposedInput, origins: dict) -> list[str]:
    """API ids an exposed input applies to.

    Args:
        exposed: One exposed input.
        origins: ``{api_id: (definition_id, inner_id)}`` from the conversion.

    Returns:
        Every API id whose node is the one the exposure names. A subgraph placed more
        than once answers with one id per placement.
    """
    return [
        api_id
        for api_id, (definition, inner_id) in origins.items()
        if inner_id == exposed.node_id and definition == exposed.definition
    ]


def carried(class_type, widget) -> str | None:
    """The socket type an exposed input takes instead of a value picked from a menu.

    Args:
        class_type: The node the widget sits on, or ``None`` when it is unknown.
        widget: The input's name.

    Returns:
        A socket type where the menu names something the node reads, such as a picture off
        disk. ``None`` where a value belongs in a widget.
    """
    from nodes import NODE_CLASS_MAPPINGS

    from . import convert

    config = convert.declared_input(class_type, widget) if class_type else None
    if config is None or not convert.is_picker(config):
        return None
    node_class = NODE_CLASS_MAPPINGS.get(class_type)
    if node_class is None or convert.first_picker(class_type) != widget:
        return None
    spec = convert.input_spec(class_type)
    if spec is None:
        return None
    taken = set()
    for section in ("required", "optional"):
        for entry in (spec.get(section) or {}).values():
            kind = entry[0] if isinstance(entry, (list, tuple)) and entry else entry
            # A custom socket type can be a str subclass that will not hash.
            if type(kind) is str:
                taken.add(kind)
    if taken & set(CARRIED_TYPES):
        return None
    # A custom socket type can be a str subclass that equals everything and will not hash.
    answered = [
        kind for kind in (getattr(node_class, "RETURN_TYPES", None) or ())
        if type(kind) is str and kind in CARRIED_TYPES
    ]
    if not answered or not _names_a_read(config, answered, spec):
        return None
    return ",".join(dict.fromkeys(answered))


def _names_a_read(config, answered, spec) -> bool:
    """Whether a menu names files of the kind its node answers.

    Args:
        config: The input's declared configuration.
        answered: The carried types the node answers.
        spec: The node's declared inputs.

    Returns:
        True for a menu marked as an upload, one whose entries are named as files the node
        reads, or an empty menu on a node given nothing to work on. A menu of words, or of
        files of another kind, answers False.
    """
    settings = config[1] if len(config) > 1 and isinstance(config[1], dict) else {}
    if any(key.endswith("_upload") and settings[key] for key in settings):
        return True
    options = config[0] if isinstance(config[0], (list, tuple)) else settings.get("options")
    if not options:
        # A menu is empty where the folder it lists is, so the answer comes from the node's
        # own shape instead, which reads the same on every install.
        return all(kind in LOADED_TYPES for kind in answered) and _reads_only(spec)
    wanted = tuple(
        extension for kind in answered for extension in READ_EXTENSIONS.get(kind, ())
    )
    named = sum(
        1 for option in options
        if isinstance(option, str) and option.lower().endswith(wanted)
    )
    return named * 2 >= len(options)


def _reads_only(spec) -> bool:
    """Whether a node is given nothing but values typed into it.

    Args:
        spec: The node's declared inputs.

    Returns:
        True when every input is a widget kind or a menu, so the node's whole subject comes
        off disk rather than down a wire.
    """
    from . import convert

    for section in ("required", "optional"):
        for entry in (spec.get(section) or {}).values():
            config = entry if isinstance(entry, (list, tuple)) else (entry,)
            kind = config[0] if config else None
            if isinstance(kind, (list, tuple)):
                continue
            if type(kind) is not str or kind not in convert.WIDGET_KINDS:
                return False
    return True


def socketed(exposure, prompt, origins, limit) -> dict:
    """Which exposed inputs are given a socket of their own.

    Args:
        exposure: What the workflow exposes.
        prompt: The workflow in API form.
        origins: Where each converted node came from.
        limit: Sockets available.

    Returns:
        ``{index_in_exposure: socket_type}`` for the inputs a wire replaces, in the order
        the workflow lists them, up to ``limit``.
    """
    found = {}
    for index, entry in enumerate(exposure.inputs):
        if len(found) >= limit:
            break
        reached = targets(entry, origins)
        class_type = prompt[reached[0]]["class_type"] if reached else None
        kind = carried(class_type, entry.widget)
        if kind and not any(_reads_many(prompt, api_id) for api_id in reached):
            found[index] = kind
    return found


def _reads_many(prompt, node_id) -> bool:
    """Whether a workflow takes more than one of a node's outputs.

    Args:
        prompt: The workflow in API form.
        node_id: The node a wire would stand in for.

    Returns:
        True when two or more of its output slots are wired to something. One wire carries
        one of them, so it cannot stand in for the whole read.
    """
    taken = set()
    for node in prompt.values():
        for value in node.get("inputs", {}).values():
            if isinstance(value, list) and len(value) == 2 and value[0] == node_id:
                taken.add(value[1])
    return len(taken) > 1
