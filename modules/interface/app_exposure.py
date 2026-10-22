"""What a saved app workflow offers, as JSON.

``GET /was/interface/api/app_exposure?app=<name>`` answers the exposed inputs, the results
and the node types the workflow needs that are not registered here, read from the asking
user's own workflows folder.
"""

from __future__ import annotations

import os
from pathlib import Path

from .. import log
from ..workflow.apps import MAX_WORKFLOW_BYTES
from .channel import NO_STORE

__all__ = [
    "MAX_WORKFLOW_BYTES",
    "REFUSED",
    "ROUTE",
    "SLOTS",
    "exposure_payload",
    "register_routes",
    "request_directory",
]

logger = log.get_logger("interface.app_exposure")

#: Where a browser asks what one app workflow offers.
ROUTE = "/was/interface/api/app_exposure"

#: Exposed inputs the node gives a socket of its own, matching its declared slots.
SLOTS = 4

#: The one answer a workflow that was refused gets, whatever the cause. The server log names it.
REFUSED = "that app workflow was refused. The ComfyUI log names the reason"


_registered = False


#: Option keys carried through to the browser, so a widget is drawn as its node draws it.
OPTION_KEYS = ("min", "max", "step", "round", "multiline", "placeholder", "default", "tooltip")


def drawable(kind):
    """The widget kind an input is drawn as.

    Args:
        kind: A declared socket type, which may name several separated by commas.

    Returns:
        The first named type a widget is drawn for, or ``kind`` where none is.
    """
    from ..workflow import convert

    if not isinstance(kind, str) or "," not in kind:
        return kind
    for part in kind.split(","):
        if part.strip() in convert.WIDGET_KINDS:
            return part.strip()
    return kind


def declared(class_type, widget):
    """How one input is drawn, for rebuilding its widget in a browser.

    Args:
        class_type: The node the widget sits on, or ``None`` when it is unknown.
        widget: The input's name.

    Returns:
        ``{"kind", "options", ...}``, with ``kind`` ``None`` where nothing is registered.
    """
    from ..workflow import convert

    config = convert.declared_input(class_type, widget) if class_type else None
    if config is None:
        return {"kind": None, "options": None}
    kind = config[0]
    settings = config[1] if len(config) > 1 and isinstance(config[1], dict) else {}
    listed = list(kind) if isinstance(kind, (list, tuple)) else settings.get("options")
    found = {"kind": "COMBO" if listed is not None else drawable(kind), "options": listed}
    for key in OPTION_KEYS:
        if key in settings:
            found[key] = settings[key]
    return found


def request_directory(request) -> Path | None:
    """The workflows directory of the user a request is made as.

    Args:
        request: The request, whose ``Comfy-User`` header names the user when ComfyUI runs
            with ``--multi-user``.

    Returns:
        That user's ``workflows`` folder, or ``None`` when the user is unknown or ComfyUI
        names no folder for them.
    """
    try:
        from server import PromptServer

        found = PromptServer.instance.user_manager.get_request_user_filepath(
            request, "workflows", create_dir=False,
        )
    except Exception as error:
        logger.debug("no workflows folder was found for an app request (%s)", error)
        return None
    return Path(found) if found else None


def file_size(name, root):
    """Size of a saved app workflow file, in bytes.

    Args:
        name: A saved workflow's name, relative to ``root``.
        root: The workflows directory, or ``None`` for none.

    Returns:
        Its size, or ``None`` when the name leaves the directory, resolves outside it or
        names no file.
    """
    from ..util import sandbox

    if root is None or sandbox.leaves_folder(name) is not None:
        return None
    base = os.path.realpath(root)
    target = os.path.realpath(os.path.join(base, name))
    if not sandbox.contains(Path(base), Path(target)) or not os.path.isfile(target):
        return None
    return os.path.getsize(target)


def exposure_payload(name, root):
    """What one app workflow offers.

    Args:
        name: A saved workflow's name, relative to ``root``.
        root: The workflows directory to read in, or ``None``, which reads nothing.

    Returns:
        ``{"app", "inputs", "results", "panels", "nodes", "missing"}``, or ``None`` when
        no such workflow can be read.

    Raises:
        ValueError: The file is larger than ``MAX_WORKFLOW_BYTES``, or the workflow
            cannot be flattened.
    """
    if not name or root is None:
        return None
    from ..workflow import apps, convert

    size = file_size(name, root)
    if size is not None and size > MAX_WORKFLOW_BYTES:
        raise ValueError(
            f"{name!r} is {size / (1024 * 1024):.1f} MiB, larger than the "
            f"{MAX_WORKFLOW_BYTES / (1024 * 1024):.0f} MiB an app workflow may be. Save a "
            f"smaller workflow"
        )
    try:
        workflow = apps.load(name, root)
    except (FileNotFoundError, ValueError):
        return None

    exposure = apps.exposure(workflow)
    prompt, origins, missing = convert.prompt_with_origins(workflow)
    sockets = apps.socketed(exposure, prompt, origins, SLOTS)
    order = {index: place for place, index in enumerate(sockets)}
    inputs = []
    for index, entry in enumerate(exposure.inputs):
        found = apps.targets(entry, origins)
        node = prompt[found[0]]["class_type"] if found else None
        wire = sockets.get(index)
        inputs.append(
            {
                "label": entry.label,
                "widget": entry.widget,
                "node": node,
                "slot": f"input_{order[index] + 1}" if index in order else None,
                "value": prompt[found[0]]["inputs"].get(entry.widget) if found else None,
                **declared(node, entry.widget),
                "wire": wire,
            }
        )
    return {
        "app": name,
        "inputs": inputs,
        "outputs": results(prompt, exposure.outputs),
        "results": len(exposure.outputs),
        "panels": list(exposure.panels),
        "nodes": len(prompt),
        "missing": sorted(missing),
    }


def results(prompt, outputs):
    """What each result an app presents carries.

    Args:
        prompt: The workflow in API form.
        outputs: Ids of the nodes whose results the app presents.

    Returns:
        One entry per result, each ``{"socket", "type", "name", "tooltip", "node"}``, with
        ``type`` ``None`` where nothing feeds that node.
    """
    from ..workflow import convert, expand

    found = []
    for index, node_id in enumerate(outputs):
        link = expand.producer(prompt, node_id)
        source = prompt.get(link[0]) if link else None
        declared = convert.declared_output(source["class_type"], link[1]) if source else None
        found.append(
            {
                "socket": f"output_{index + 1}" if index < SLOTS else None,
                "type": (declared or {}).get("type"),
                "name": (declared or {}).get("name"),
                "tooltip": (declared or {}).get("tooltip"),
                "node": source["class_type"] if source else None,
                "presented_by": prompt.get(node_id, {}).get("class_type"),
            }
        )
    return found


def register_routes():
    """Register the route answering what an app workflow offers.

    Returns:
        True when the route was registered. False when it was registered already, or when
        the server could not be reached, in which case the panel asking gets a failed
        request.
    """
    global _registered
    if _registered:
        return False
    try:
        import asyncio

        from aiohttp import web
        from server import PromptServer

        @PromptServer.instance.routes.get(ROUTE)
        async def get_app_exposure(request):
            name = request.query.get("app")
            root = request_directory(request)
            try:
                payload = await asyncio.to_thread(exposure_payload, name, root)
            except Exception as error:
                logger.warning(
                    "app workflow %r was refused (%s: %s)", name, type(error).__name__, error,
                )
                return web.json_response({"error": REFUSED}, status=422, headers=NO_STORE)
            if payload is None:
                return web.json_response(
                    {"error": "no such app workflow"}, status=404, headers=NO_STORE
                )
            return web.json_response(payload, headers=NO_STORE)

    except Exception as error:
        logger.warning(
            "%s was not registered (%s: %s), so an App Workflow node asking what a saved "
            "workflow offers gets a failed request",
            ROUTE, type(error).__name__, error,
        )
        logger.debug("%s could not be registered", ROUTE, exc_info=True)
        return False
    _registered = True
    return True
