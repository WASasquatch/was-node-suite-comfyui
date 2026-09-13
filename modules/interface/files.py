"""ComfyUI's input, output and temp folders as JSON a node interface draws.

``GET /was/interface/api/file_listing`` answers up to :data:`MAX_ENTRIES` entries, each
carrying a menu label, a path below its own root and a tag, never an absolute path.
"""

from __future__ import annotations

from .. import log
from ..util import file_listing
from .channel import NO_STORE

__all__ = ["MAX_ENTRIES", "ROUTE", "listing_payload", "register_routes"]

logger = log.get_logger("interface.files")

#: The one route serving the listing.
ROUTE = "/was/interface/api/file_listing"

#: Entries in one answer. Wider than the combo menus, since a panel scrolls and a menu does
#: not, and narrow enough that the answer stays a few hundred kilobytes.
MAX_ENTRIES = 1000

_registered = False


def listing_payload() -> dict:
    """The whole listing, as the object the route answers with.

    Never raises. The walk is memoized, so a burst of requests costs one directory walk.

    Returns:
        ``{"roots", "entries", "truncated"}``. ``roots`` names each folder that could be
        reached and how many files it contributed, ``entries`` carries ``label``,
        ``relative``, ``tag``, ``size`` and ``mtime`` per file, in the order a menu offers
        them, and ``truncated`` says whether the walk found more than this answer holds.
    """
    try:
        entries = file_listing.view(tags=file_listing.ROOTS, limit=MAX_ENTRIES)
        walked = len(file_listing.scan())
        reachable = [tag for tag, _ in file_listing.roots(file_listing.ROOTS)]
    except Exception as error:
        # A listing nobody can build is an empty panel, never a failed request: the widget
        # beside it still holds whatever was typed into it.
        logger.debug("%s could not build the listing (%s)", ROUTE, error)
        return {"roots": [], "entries": [], "truncated": False}

    counts = {tag: 0 for tag in reachable}
    rows = []
    for entry in entries:
        counts[entry.tag] = counts.get(entry.tag, 0) + 1
        rows.append(
            {
                "label": entry.label,
                "relative": entry.relative,
                "tag": entry.tag,
                "size": entry.size,
                "mtime": entry.mtime,
            }
        )
    return {
        "roots": [{"tag": tag, "files": counts.get(tag, 0)} for tag in reachable],
        "entries": rows,
        "truncated": walked > len(entries),
    }


def register_routes() -> bool:
    """Register the route serving the file listing.

    Returns:
        True when the route was registered. False when it was registered already, or when the
        server could not be reached, in which case a panel asking for the listing gets a
        failed request and draws what the widget holds.
    """
    global _registered
    if _registered:
        return False
    try:
        from aiohttp import web
        from server import PromptServer

        @PromptServer.instance.routes.get(ROUTE)
        async def get_file_listing(request):
            del request
            return web.json_response(listing_payload(), headers=NO_STORE)

    except Exception as error:
        logger.warning(
            "%s was not registered (%s: %s), so a node interface asking for the file "
            "listing gets a failed request",
            ROUTE, type(error).__name__, error,
        )
        logger.debug("%s could not be registered", ROUTE, exc_info=True)
        return False
    _registered = True
    logger.debug("%s is serving the input, output and temp file listing", ROUTE)
    return True
