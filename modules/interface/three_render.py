"""Frames a browser draws for a Three.js node that is waiting on them.

A node files a job and blocks; a browser claims it, draws it and posts the PNG back.
"""

from __future__ import annotations

__all__ = [
    "DELIVERED",
    "FAILED",
    "MAX_FRAMES",
    "MAX_FRAME_BYTES",
    "ROUTE",
    "TIMED_OUT",
    "deliver",
    "file_job",
    "pending",
    "report",
    "register_routes",
    "wait_for_frames",
]

import base64
import time
import uuid
from threading import Lock

from ..log import get_logger

logger = get_logger("interface.three_render")

#: Where a browser takes jobs from and posts frames back to.
ROUTE = "/was/threejs/api/render"

#: How the wait ended.
DELIVERED = "delivered"
FAILED = "failed"
TIMED_OUT = "timed_out"

#: Seconds between checks while a node is waiting.
TICK = 0.05

#: Most jobs held at once. A job is small; this only stops an unattended queue growing.
MAX_JOBS = 16

#: Most frames one job may ask for.
MAX_FRAMES = 512

#: Largest one delivered frame may be, in bytes.
MAX_FRAME_BYTES = 64 * 1024 * 1024

_jobs: dict[str, dict] = {}
_lock = Lock()


#: The pictures a browser draws per frame. ``png`` is the scene as it looks; the other two
#: are the same frame drawn with one override material over everything.
PASSES = ("png", "depth", "normal")


def file_job(
    app: dict, width: int, height: int, transparent: bool, times: list[float],
    supersample: int = 1, depth_near: float = 0.0, depth_far: float = 0.0,
    trace: dict | None = None, progress_total: int = 0,
) -> str:
    """Record the frames a browser is being asked to draw.

    Args:
        app: The app descriptor to render.
        width: Frame width in pixels.
        height: Frame height in pixels.
        transparent: Whether the background is left clear.
        times: Seconds into the animation each frame is taken at, in order.
        supersample: How many times oversize each frame is drawn before it is scaled
            back to ``width`` by ``height``.
        depth_near: Distance the depth pass calls white, or 0.0 to fit it to the scene.
        depth_far: Distance the depth pass calls black, or 0.0 to fit it to the scene.
        trace: Path tracer settings, or None to draw the frames the ordinary way.
        progress_total: Units of work the browser will report against, or 0 for the
            frame count.

    Returns:
        The token the job is claimed and answered by.

    Raises:
        ValueError: The scene carries browser code and ``threejs.allow_scripts`` is off.
    """
    from ..threejs.spec import refuse_script

    refuse_script(app, "This render")
    token = uuid.uuid4().hex
    client = _asking_client()
    with _lock:
        while len(_jobs) >= MAX_JOBS:
            oldest = min(_jobs, key=lambda key: _jobs[key]["filed"])
            _jobs.pop(oldest, None)
        _jobs[token] = {
            "filed": time.monotonic(),
            "app": app,
            "width": int(width),
            "height": int(height),
            "transparent": bool(transparent),
            "supersample": int(supersample),
            "depthNear": float(depth_near),
            "depthFar": float(depth_far),
            "times": [float(moment) for moment in times],
            "trace": dict(trace) if trace else None,
            "total": int(progress_total) or len(times),
            "done": 0,
            "note": "",
            "client": client,
            "claimed": False,
            "frames": {kind: {} for kind in PASSES},
            "error": "",
        }
    return token


def _asking_client() -> str:
    """The client id of the prompt being run, or ``""`` outside one."""
    try:
        from server import PromptServer

        return str(getattr(PromptServer.instance, "client_id", "") or "")
    except Exception:
        return ""


def pending(client: str = "") -> list[dict]:
    """The jobs this browser has not claimed yet, and claim them.

    Args:
        client: The asking browser's ComfyUI client id. A job filed for another client is
            not offered, so a token never reaches a caller the prompt did not come from.

    Returns:
        One entry per job, each with its token, descriptor, frame size and moments.
    """
    taken = []
    asking = str(client or "").strip()
    with _lock:
        for token, job in _jobs.items():
            if job["claimed"] or any(job["frames"].values()):
                continue
            if job["client"] and job["client"] != asking:
                continue
            job["claimed"] = True
            taken.append({
                "token": token,
                "app": job["app"],
                "width": job["width"],
                "height": job["height"],
                "transparent": job["transparent"],
                "supersample": job["supersample"],
                "depthNear": job["depthNear"],
                "depthFar": job["depthFar"],
                "times": job["times"],
                "trace": job["trace"],
            })
    return taken


def deliver(
    token: str, index: int, bodies: dict[str, bytes], error: str = ""
) -> bool:
    """Record one frame a browser drew, or why the job could not be drawn.

    Args:
        token: The token the job was filed under.
        index: Which frame of the run this is, counting from 0.
        bodies: PNG bytes per pass name, keyed by one of :data:`PASSES`.
        error: What went wrong, for the message the node raises.

    Returns:
        True where a job was waiting under that token.
    """
    with _lock:
        job = _jobs.get(token)
        if job is None:
            return False
        if error:
            job["error"] = error
            return True
        place = int(index)
        # A frame belongs to a moment the job asked for, and is no larger than one frame.
        if place < 0 or place >= len(job["times"]):
            return False
        for kind, body in bodies.items():
            if kind not in job["frames"] or body is None:
                continue
            if len(body) > MAX_FRAME_BYTES:
                return False
            job["frames"][kind][place] = body
    return True


def report(token: str, done: int, note: str = "") -> bool:
    """Record how far along a browser says it is.

    Args:
        token: The token the job was filed under.
        done: Units of work finished, against the job's total.
        note: A short line naming what it is working on.

    Returns:
        True where a job was waiting under that token.
    """
    with _lock:
        job = _jobs.get(token)
        if job is None:
            return False
        job["done"] = max(int(job["done"]), int(done))
        if note:
            job["note"] = str(note)[:120]
    return True


def _progress_bar(node_id: str):
    """A progress bar for a node, where this ComfyUI offers one.

    Args:
        node_id: Which node the bar is drawn on.

    Returns:
        The bar, or None where it is unavailable.
    """
    if not node_id:
        return None
    try:
        from comfy.utils import ProgressBar

        return ProgressBar(1, node_id=node_id)
    except Exception:
        logger.debug("no progress bar is available", exc_info=True)
        return None


def _tell(node_id: str, note: str) -> None:
    """Put a line under a node's progress bar, where this ComfyUI offers one.

    Args:
        node_id: Which node the line is drawn on.
        note: The line.
    """
    if not node_id:
        return
    try:
        from server import PromptServer

        PromptServer.instance.send_progress_text(note, node_id)
    except Exception:
        logger.debug("no progress line is available", exc_info=True)


def wait_for_frames(
    token: str, timeout: float, node_id: str = ""
) -> tuple[str, dict[str, list[bytes]], str]:
    """Hold the run until every frame arrives, the wait runs out, or the run is cancelled.

    Args:
        token: The token :func:`file_job` answered.
        timeout: Seconds to wait before giving up.
        node_id: Which node the progress is drawn on, or empty for none.

    Returns:
        How it ended, ``{pass name: frames in order}`` where they all arrived, and the
        browser's message where it reported one.

    Raises:
        InterruptProcessingException: The run was cancelled.
    """
    import comfy.model_management

    bar, told = _progress_bar(node_id), ""
    started = time.monotonic()
    try:
        while True:
            # Raising through ComfyUI clears its interrupt flag, which is what stops the flag
            # carrying into whatever runs next.
            comfy.model_management.throw_exception_if_processing_interrupted()
            with _lock:
                job = _jobs.get(token)
                if job is None:
                    return FAILED, {}, "the render job went missing"
                wanted = len(job["times"])
                have = {kind: dict(frames) for kind, frames in job["frames"].items()}
                error = job["error"]
                standing, total, note = job["done"], job["total"], job["note"]
            if bar is not None:
                bar.update_absolute(standing, total)
            if note and note != told:
                told = note
                _tell(node_id, note)
            if error:
                return FAILED, {}, error
            if all(len(frames) >= wanted for frames in have.values()):
                return DELIVERED, {
                    kind: [frames[index] for index in range(wanted)]
                    for kind, frames in have.items()
                }, ""
            if (time.monotonic() - started) >= timeout:
                return TIMED_OUT, {}, ""
            time.sleep(TICK)
    finally:
        with _lock:
            _jobs.pop(token, None)


def register_routes() -> bool:
    """Hand jobs to a browser and take frames back.

    Returns:
        True where the route was registered.
    """
    try:
        from aiohttp import web

        from server import PromptServer

        @PromptServer.instance.routes.get(ROUTE)
        async def get_render_jobs(request):
            asking = str(request.query.get("client_id", "")).strip()
            return web.json_response(
                {"jobs": pending(asking)}, headers={"Cache-Control": "no-store"}
            )

        @PromptServer.instance.routes.post(ROUTE)
        async def post_render_frame(request):
            try:
                sent = await request.json()
            except Exception:
                return web.Response(status=400, text="The body has to be JSON.")
            token = str(sent.get("token", "")).strip()
            if not token:
                return web.Response(status=400, text="The body has to name a token.")
            failure = str(sent.get("error", "")).strip()

            # A body carrying only how far along it is moves the bar and nothing else.
            if not failure and "done" in sent and not any(kind in sent for kind in PASSES):
                try:
                    done = int(sent.get("done", 0))
                except (TypeError, ValueError):
                    return web.Response(status=400, text="done has to be a whole number.")
                if not report(token, done, str(sent.get("note", ""))):
                    return web.Response(status=404, text="No node is waiting under that token.")
                return web.Response(status=204, headers={"Cache-Control": "no-store"})

            try:
                index = int(sent.get("index", 0))
            except (TypeError, ValueError):
                return web.Response(status=400, text="index has to be a whole number.")
            bodies: dict[str, bytes] = {}
            if not failure:
                for kind in PASSES:
                    encoded = str(sent.get(kind, ""))
                    if not encoded:
                        continue
                    marker = "base64,"
                    if marker in encoded:
                        encoded = encoded.split(marker, 1)[1]
                    try:
                        bodies[kind] = base64.b64decode(encoded, validate=True)
                    except Exception:
                        return web.Response(
                            status=400, text=f"The {kind} frame was not valid base64."
                        )
            if not deliver(token, index, bodies, failure):
                return web.Response(status=404, text="No node is waiting under that token.")
            if not failure:
                try:
                    report(token, int(sent.get("done", index + 1)), str(sent.get("note", "")))
                except (TypeError, ValueError):
                    pass
            return web.Response(status=204, headers={"Cache-Control": "no-store"})

    except Exception as error:
        logger.warning(
            "%s was not registered (%s), so Three Render cannot reach a browser and will "
            "time out",
            ROUTE, type(error).__name__,
        )
        logger.debug("%s could not be registered", ROUTE, exc_info=True)
        return False

    logger.debug("%s is carrying render jobs to the browser", ROUTE)
    return True
