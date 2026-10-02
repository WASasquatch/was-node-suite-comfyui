"""The pictures a node held, as thumbnails its interface can fetch.

``GET /was/interface/api/preview?node_id=<id>&slot=<name>&side=<input|output>`` answers PNG
bytes, or 204. Pictures are held in memory per ``(node, side, slot)``, bounded in bytes
before the encode and in frames after it.
"""

from __future__ import annotations

import math
import re
import threading
from pathlib import Path
from collections import OrderedDict
from io import BytesIO
from typing import NamedTuple

from .. import log
from .channel import (
    MAX_SUBSCRIPTIONS,
    NO_STORE,
    PROMPT_ID_HEADER,
    REFUSAL_HEADER,
    clear_subscriptions,
    executing_node_id,
    executing_prompt_id,
    node_key,
    subscribe,
    unsubscribe,
    wanted,
    watching,
)

__all__ = [
    "ENCODED_BYTES_HEADER",
    "FRAME_COUNT_HEADER",
    "FRAME_TOTAL_HEADER",
    "IMAGE_KIND",
    "INPUT",
    "KIND_HEADER",
    "MASK_KIND",
    "MAX_EDGE_KEY",
    "MAX_ENTRIES",
    "MAX_FRAMES",
    "MAX_FRAME_BYTES",
    "MAX_KEY_BYTES",
    "MAX_SLOTS",
    "OUTPUT",
    "PROMPT_ID_HEADER",
    "Picture",
    "REFUSAL_HEADER",
    "DISCARD_ROUTE",
    "ROUTE",
    "SIDES",
    "SOURCE_HEIGHT_HEADER",
    "SOURCE_MODE_HEADER",
    "SOURCE_WIDTH_HEADER",
    "SUBSCRIBE_ROUTE",
    "frame_total",
    "frames",
    "held_bytes",
    "publish",
    "publish_frames",
    "publish_mask",
    "publish_mask_frames",
    "publish_mask_output",
    "publish_mask_output_frames",
    "publish_output",
    "publish_output_frames",
    "discard_clipspace",
    "register_routes",
    "thumbnail",
    "watching",
]

logger = log.get_logger("interface.preview")

#: Config key holding the longest edge a published picture is reduced to, in pixels. 0 publishes
#: the image at the size the node received it, which costs held memory and PNG encode time on
#: every run; a positive value trades detail for both.
MAX_EDGE_KEY = "interface.preview_max_edge"

#: What that key holds when the config cannot be read, which is the size the node received.
NO_LIMIT = 0

#: The side of the node a picture was on: what it arrived with.
INPUT = "input"

#: The side of the node a picture was on: what it left with.
OUTPUT = "output"

#: The two sides a picture can be filed under. A publisher naming anything else stores nothing,
#: since the route can never be asked for a side it does not know.
SIDES = (INPUT, OUTPUT)

#: What a picture is, from the publisher that stored it. A mask and a single-channel image are
#: both mode ``L``, so the channel mode cannot tell them apart and this can.
IMAGE_KIND = "image"

#: The other kind, published through the mask entry points.
MASK_KIND = "mask"

#: How many nodes hold thumbnails at once. This process outlives every prompt, so the
#: store is bounded rather than left to grow with the graph. The bound counts nodes, not
#: pictures, so a node publishing several inputs ages out its own oldest slot and never
#: another node's picture.
#:
#: Thirty-two, against families of 43 node ids drawing a before and after, 23 a mask
#: readout and 21 a size band, and every mounted panel registers its node. A graph watching
#: more nodes than this evicts the ones published earliest in the run, and ComfyUI will not
#: re-execute a cached node to publish again, so an evicted panel waits until something makes
#: that node run and reads to the user as a node that did not run. The arithmetic ceiling is
#: :data:`MAX_ENTRIES` times :data:`MAX_SLOTS` times :data:`MAX_KEY_BYTES`, 64 MB a node, so it
#: moves from 1 GB to 2 GB. A filter is the realistic shape and holds two of the eight, one
#: slot name on each side, so a full store of those moves from about 256 MB to about 512 MB.
#: What it costs in practice is far under either: at the 1024 default edge a filter's two
#: sides are nearer 1.4 MB each, so thirty-two of them is about 90 MB.
MAX_ENTRIES = 32

#: How many slots one node holds at once, shared across both sides, so a node publishing an
#: input and an output named alike holds two of the eight. Six images on one node is the widest
#: the pack declares.
MAX_SLOTS = 8

#: Frames one slot of one side holds when a node publishes a whole batch. The backstop behind
#: :data:`MAX_KEY_BYTES`, for a batch of pictures so small that the byte budget never trips.
MAX_FRAMES = 64

#: The most one stored picture may cost uncompressed, in bytes. Decided from the tensor's own
#: shape before anything is encoded: a frame above it is reduced until it fits rather than
#: dropped, so a node publishing a 4096 by 4096 frame with no configured maximum edge still
#: gets a picture. Measured, PNG never exceeded the uncompressed size by more than 0.2%, so
#: this bounds the stored bytes as well as the encode.
MAX_FRAME_BYTES = 8 * 1024 * 1024

#: The encoded bytes one ``(node, side, slot)`` holds before :func:`publish_frames` stops
#: encoding. The budget is per key rather than per node, so one slot cannot spend another's
#: allowance and the second of two slots published back to back is never starved to nothing.
#: The first frame of a key is always stored, whatever it costs.
MAX_KEY_BYTES = 8 * 1024 * 1024

#: Route a panel calls to drop the files ComfyUI's own mask editor uploaded on a save. That
#: editor writes four PNGs into the input directory whatever the caller does with the result, and
#: an interface that uses none of them is leaving litter in a menu every Load Image node draws.
DISCARD_ROUTE = "/was/interface/api/preview/discard"

#: What one of those uploads is named, which is the only shape this will delete: the editor's own
#: prefix, one of its four layer names, and the millisecond stamp it wrote them under.
CLIPSPACE_NAME = re.compile(r"^clipspace-(mask|paint|painted|painted-masked)-\d{6,}\.png$")

#: The one route serving what was published, keyed by a ``node_id`` query parameter, an optional
#: ``slot`` naming one picture of a node holding more than one, an optional ``frame``, and an
#: optional ``side`` of ``input`` or ``output``.
ROUTE = "/was/interface/api/preview"

#: Where a panel registers the node it is open on, so publishing for that node is worth the
#: encode. ``POST`` with one or more ``node_id`` parameters and ``watch=1`` to register or
#: ``watch=0`` to release.
SUBSCRIBE_ROUTE = "/was/interface/api/preview/subscribe"

# A node publishes on the thread running the prompt and the route answers on the server's, so
# every read and write of the store below goes through this.
_lock = threading.Lock()

#: Header carrying the width of the image the node received, before it was reduced.
SOURCE_WIDTH_HEADER = "X-WAS-Source-Width"

#: Header carrying the height of the same image.
SOURCE_HEIGHT_HEADER = "X-WAS-Source-Height"

#: Header carrying how many frames the slot holds, so an interface knows how far it can page
#: without asking for a frame that is not there.
FRAME_COUNT_HEADER = "X-WAS-Frame-Count"

#: Header carrying how many frames the batch had before the budget and :data:`MAX_FRAMES` took
#: their share, so an interface can word what it is not showing rather than reporting a shorter
#: batch than the run made.
FRAME_TOTAL_HEADER = "X-WAS-Frame-Total"

#: Header carrying the picture's channel mode, ``RGB``, ``RGBA`` or ``L``, which an interface
#: cannot recover from the decoded image: a canvas hands back four channels whatever went in.
SOURCE_MODE_HEADER = "X-WAS-Source-Mode"

#: Header carrying :data:`IMAGE_KIND` or :data:`MASK_KIND`. A single-channel image and a mask are
#: both mode ``L``, so this is the only thing in the answer that separates them.
KIND_HEADER = "X-WAS-Kind"

#: Header carrying the encoded picture's own length in bytes. A browser cannot measure this off
#: a decoded image, and it is the one figure that answers to what is in the picture rather than
#: only to how large it is.
ENCODED_BYTES_HEADER = "X-WAS-Encoded-Bytes"


class Picture(NamedTuple):
    """One stored frame, with everything an interface needs to read it.

    Attributes:
        data: The encoded PNG.
        width: Width of the picture the node held, before any reduction.
        height: Height of the picture the node held.
        mode: Channel mode of the encoded picture, ``RGB``, ``RGBA`` or ``L``.
        kind: :data:`IMAGE_KIND` or :data:`MASK_KIND`, from the publisher that stored it.
        prompt: The prompt this was published under, empty for a publish outside a run.
    """

    data: bytes
    width: int
    height: int
    mode: str
    kind: str
    prompt: str


class _Held(NamedTuple):
    """One slot of one side: the frames it holds, and the batch they came from.

    Attributes:
        pictures: The stored frames, in batch order.
        total: How many frames the tensor held before the bounds took their share.
        cost: How many encoded bytes the frames add up to.
    """

    pictures: tuple[Picture, ...]
    total: int
    cost: int


#: Node id to that node's ``(side, slot)`` keys, most recently used node last, each holding a
#: :class:`_Held` with the most recently used key last. The source size travels with the bytes,
#: so an interface writing absolute pixels into a widget can map a gesture on a reduced
#: picture back onto the source.
_thumbnails: OrderedDict[str, OrderedDict[tuple[str, str], _Held]] = OrderedDict()

#: The slot a node with one picture publishes under, and the one a request naming no slot
#: reads. A name is stripped of surrounding space, so a blank one is this.
_DEFAULT_SLOT = ""

_registered = False


def publish(image, node_id=None, slot=None) -> bool:
    """Store an image a node received, as a thumbnail that node's interface can fetch.

    Never raises, and never touches the image it is given.

    Args:
        image: An ``IMAGE`` tensor, ``(batch, height, width, channels)``. The first image
            of a batch is the one stored.
        node_id: The publishing node's graph id. Left out, the id of the node ComfyUI is
            executing is read from its execution context, which is the value
            ``io.Hidden.unique_id`` carries.
        slot: Which of the node's images this is, named after the input it arrived on. A
            node holding one image leaves it out, and that picture is what a request naming
            no slot reads. A node holding several names every one of them.

    Returns:
        True when a thumbnail was stored. False when no browser is connected, when no panel
        is open on that node, when no node id or usable slot name could be found, or when
        the image could not be encoded, each of which costs the picture and nothing else.
    """
    return _publish(image, node_id, slot, INPUT, IMAGE_KIND, first_only=True) > 0


def publish_output(image, node_id=None, slot=None) -> bool:
    """Store an image a node answered with, as a thumbnail its interface can fetch.

    Never raises, and never touches the image it is given.

    Args:
        image: An ``IMAGE`` tensor, ``(batch, height, width, channels)``. The first image
            of a batch is the one stored.
        node_id: The publishing node's graph id, as :func:`publish` reads it.
        slot: Which of the node's images this is, named after the output it left on. An
            output may share the name of an input, since the side is part of the key.

    Returns:
        True when a thumbnail was stored, on the same terms as :func:`publish`.
    """
    return _publish(image, node_id, slot, OUTPUT, IMAGE_KIND, first_only=True) > 0


def publish_mask(mask, node_id=None, slot=None) -> bool:
    """Store a mask a node received, as a mode ``L`` thumbnail its interface can fetch.

    Args:
        mask: A ``MASK`` tensor in any of the four layouts a ``MASK`` socket carries. The
            first mask of a batch is the one stored.
        node_id: The publishing node's graph id, as :func:`publish` reads it.
        slot: Which of the node's masks this is, named after the input it arrived on.

    Returns:
        True when a thumbnail was stored, on the same terms as :func:`publish`.
    """
    return _publish(mask, node_id, slot, INPUT, MASK_KIND, first_only=True) > 0


def publish_mask_output(mask, node_id=None, slot=None) -> bool:
    """Store a mask a node answered with, as a mode ``L`` thumbnail its interface can fetch.

    Args:
        mask: A ``MASK`` tensor in any of the four layouts a ``MASK`` socket carries. The
            first mask of a batch is the one stored.
        node_id: The publishing node's graph id, as :func:`publish` reads it.
        slot: Which of the node's masks this is, named after the output it left on.

    Returns:
        True when a thumbnail was stored, on the same terms as :func:`publish`.
    """
    return _publish(mask, node_id, slot, OUTPUT, MASK_KIND, first_only=True) > 0


def publish_frames(images, node_id=None, slot=None) -> int:
    """Store the frames of a batch a node received, so an interface can page through them.

    Args:
        images: An ``IMAGE`` tensor, ``(batch, height, width, channels)``. Frames are stored
            until :data:`MAX_KEY_BYTES` is spent or :data:`MAX_FRAMES` is reached, whichever
            comes first, and the first frame is always stored.
        node_id: The publishing node's graph id, as :func:`publish` reads it.
        slot: Which of the node's images these are, named after the input they arrived on.

    Returns:
        How many frames were stored, which is what :data:`FRAME_COUNT_HEADER` reports. 0 on
        every condition :func:`publish` answers False for.
    """
    return _publish(images, node_id, slot, INPUT, IMAGE_KIND, first_only=False)


def publish_output_frames(images, node_id=None, slot=None) -> int:
    """Store the frames of a batch a node answered with, so an interface can page through them.

    Args:
        images: An ``IMAGE`` tensor, bounded as :func:`publish_frames` describes.
        node_id: The publishing node's graph id, as :func:`publish` reads it.
        slot: Which of the node's images these are, named after the output they left on.

    Returns:
        How many frames were stored, on the same terms as :func:`publish_frames`.
    """
    return _publish(images, node_id, slot, OUTPUT, IMAGE_KIND, first_only=False)


def publish_mask_frames(masks, node_id=None, slot=None) -> int:
    """Store the masks of a batch a node received, as mode ``L`` pictures.

    Never raises, and never touches the masks it is given.

    Args:
        masks: A ``MASK`` tensor in any of the four layouts a ``MASK`` socket carries,
            bounded as :func:`publish_frames` describes.
        node_id: The publishing node's graph id, as :func:`publish` reads it.
        slot: Which of the node's masks these are, named after the input they arrived on.

    Returns:
        How many frames were stored, on the same terms as :func:`publish_frames`.
    """
    return _publish(masks, node_id, slot, INPUT, MASK_KIND, first_only=False)


def publish_mask_output_frames(masks, node_id=None, slot=None) -> int:
    """Store the masks of a batch a node answered with, as mode ``L`` pictures.

    Never raises, and never touches the masks it is given.

    Args:
        masks: A ``MASK`` tensor in any of the four layouts a ``MASK`` socket carries,
            bounded as :func:`publish_frames` describes.
        node_id: The publishing node's graph id, as :func:`publish` reads it.
        slot: Which of the node's masks these are, named after the output they left on.

    Returns:
        How many frames were stored, on the same terms as :func:`publish_frames`.
    """
    return _publish(masks, node_id, slot, OUTPUT, MASK_KIND, first_only=False)


def _publish(tensor, node_id, slot, side, kind, first_only) -> int:
    """The whole publish path, behind all eight public entry points.

    Args:
        tensor: The tensor to publish, read as ``kind`` says.
        node_id: The publishing node's graph id, or None to read the executing one.
        slot: The slot name, as :func:`_slot` reads it.
        side: :data:`INPUT` or :data:`OUTPUT`.
        kind: :data:`IMAGE_KIND` or :data:`MASK_KIND`.
        first_only: Store the first frame alone rather than as much of the batch as fits.

    Returns:
        How many frames were stored, 0 on every refusal.
    """
    if _idle():
        return 0
    try:
        if side not in SIDES:
            logger.warning(
                "a preview was published under side %r, which is not one of %s, so nothing "
                "was stored: the route can never be asked for it",
                side, ", ".join(SIDES),
            )
            return 0
        key = node_key(node_id if node_id is not None else executing_node_id())
        if key is None:
            logger.debug("a preview was published with no node id to file it under")
            return 0
        # A declared interface is not an open one. Until a panel registers the node it is
        # drawn on, publishing costs nothing at all, which is what keeps the cost of adopting
        # the channel flat as more nodes adopt it.
        if not wanted(key):
            logger.debug("no interface is open on node %s, so no preview was encoded", key)
            return 0
        name = _slot(slot)
        if name is None:
            logger.debug("node %s published a preview under an unusable slot name", key)
            return 0
        pictures, total = _encode_frames(tensor, kind, first_only)
        if not pictures:
            logger.warning(
                "node %s published a %s %s under slot %r and none of it could be encoded, so "
                "its interface draws nothing",
                key, side, kind, name,
            )
            return 0
        _store(key, (side, name), _Held(
            tuple(pictures), total, sum(len(one.data) for one in pictures),
        ))
        return len(pictures)
    except Exception as error:
        logger.warning(
            "no %s preview was published for node %s (%s: %s), so its interface draws nothing",
            side, node_id, type(error).__name__, error,
        )
        logger.debug("the %s preview for node %s could not be stored", side, node_id, exc_info=True)
        return 0


def _idle() -> bool:
    """Whether no browser is attached, releasing anything held for one that has gone.

    Returns:
        True when nothing should be published. The store and the panel registrations are
        both dropped on the way, so a run that carries on after the tab closes reclaims
        what the last run held rather than holding it for the life of the process.
    """
    if watching():
        return False
    released = clear_subscriptions()
    with _lock:
        held = len(_thumbnails)
        _thumbnails.clear()
    if held or released:
        logger.debug(
            "no browser is attached, so %d node(s) of pictures and %d registration(s) were "
            "released",
            held, released,
        )
    return True


def _store(key, inner, held) -> None:
    """Put one key's frames in the store, ageing out the oldest node and key.

    Args:
        key: The publishing node's store key.
        inner: The ``(side, slot)`` pair the frames are filed under.
        held: The frames, the batch length and the encoded bytes.
    """
    with _lock:
        # Reinserted rather than assigned, at both levels, so publishing again makes
        # that node and that key the most recent and the oldest is the one evicted.
        slots = _thumbnails.pop(key, None)
        if slots is None:
            slots = OrderedDict()
        slots.pop(inner, None)
        slots[inner] = held
        while len(slots) > MAX_SLOTS:
            slots.popitem(last=False)
        _thumbnails[key] = slots
        while len(_thumbnails) > MAX_ENTRIES:
            _thumbnails.popitem(last=False)


def thumbnail(node_id, slot=None, frame=0, side=INPUT) -> Picture | None:
    """One thumbnail a node published, with the size of the picture it came from.

    Args:
        node_id: A node's graph id, as a string or an integer. Anything else, including a
            missing or malformed query value, answers None.
        slot: Which of that node's pictures to read, named as it was published. Left out or
            empty, the picture published under no slot name. Anything that is not a string
            or an integer answers None.
        frame: Which frame of that slot to read, for a slot holding a whole batch. Out of
            range answers None.
        side: :data:`INPUT` or :data:`OUTPUT`. Anything else answers None.

    Returns:
        A :class:`Picture`, or None when that node published nothing under that key. Its
        size is the picture the node held, not the reduced one in ``data``.
    """
    record, _, _ = _read(node_id, slot, side, frame)
    return record


def frames(node_id, slot=None, side=INPUT) -> list | None:
    """Every frame a node published under one slot of one side, in order.

    Args:
        node_id: A node's graph id, as a string or an integer.
        slot: Which of that node's pictures to read, named as it was published.
        side: :data:`INPUT` or :data:`OUTPUT`. Anything else answers None.

    Returns:
        A list of :class:`Picture`, or None when that node published nothing under that
        key. The list is a copy, so the store going on changing does not alter it.
    """
    held = _held(node_id, slot, side)
    return None if held is None else list(held.pictures)


def frame_total(node_id, slot=None, side=INPUT) -> int:
    """How many frames the batch behind one slot held before the bounds took their share.

    Args:
        node_id: A node's graph id, as a string or an integer.
        slot: Which of that node's pictures to read, named as it was published.
        side: :data:`INPUT` or :data:`OUTPUT`.

    Returns:
        The batch length, or 0 when that node published nothing under that key. Never below
        the number of frames actually held.
    """
    held = _held(node_id, slot, side)
    return 0 if held is None else held.total


def held_bytes(node_id=None, slot=None, side=None) -> int:
    """How many encoded bytes the store holds, for one key, one node, or all of it.

    Args:
        node_id: A node's graph id. Left out, every node is counted.
        slot: Which of that node's pictures to count, read only when ``side`` is given.
        side: :data:`INPUT` or :data:`OUTPUT` to count one key alone. Left out, every key
            of the node is counted.

    Returns:
        The encoded length of every picture counted, which is the figure
        :data:`MAX_KEY_BYTES` bounds and the one a check can read without reaching into the
        store itself.
    """
    if node_id is not None and side is not None:
        held = _held(node_id, slot, side)
        return 0 if held is None else held.cost
    key = None if node_id is None else node_key(node_id)
    if node_id is not None and key is None:
        return 0
    with _lock:
        if key is not None:
            slots = _thumbnails.get(key)
            return 0 if slots is None else sum(one.cost for one in slots.values())
        return sum(one.cost for slots in _thumbnails.values() for one in slots.values())


def _held(node_id, slot, side) -> _Held | None:
    """One key's whole record, or None.

    Args:
        node_id: A node's graph id.
        slot: The slot name, as :func:`_slot` reads it.
        side: :data:`INPUT` or :data:`OUTPUT`.

    Returns:
        The stored :class:`_Held`, or None when that key holds nothing.
    """
    key = node_key(node_id)
    name = _slot(slot)
    which = _side(side)
    if key is None or name is None or which is None:
        return None
    with _lock:
        slots = _thumbnails.get(key)
        if slots is None:
            return None
        held = slots.get((which, name))
        if held is not None:
            slots.move_to_end((which, name))
            _thumbnails.move_to_end(key)
        return held


def _read(node_id, slot, side, frame) -> tuple[Picture | None, int, int]:
    """One frame, how many the key holds and how many the batch had, under one lock.

    Args:
        node_id: A node's graph id.
        slot: The slot name, as :func:`_slot` reads it.
        side: :data:`INPUT` or :data:`OUTPUT`.
        frame: Which frame to read, counting from 0.

    Returns:
        ``(picture, frames held, batch length)``. The picture is None when the key holds
        nothing and when the frame is missing, out of range or not a number, each of which
        still reports the count the key does hold.
    """
    key = node_key(node_id)
    name = _slot(slot)
    which = _side(side)
    if key is None or name is None or which is None:
        return None, 0, 0
    with _lock:
        slots = _thumbnails.get(key)
        if slots is None:
            return None, 0, 0
        held = slots.get((which, name))
        if held is None:
            return None, 0, 0
        slots.move_to_end((which, name))
        _thumbnails.move_to_end(key)
        count = len(held.pictures)
        total = held.total
        try:
            index = int(frame)
        except (TypeError, ValueError):
            return None, count, total
        if index < 0 or index >= count:
            return None, count, total
        return held.pictures[index], count, total


def register_routes() -> bool:
    """Register the route serving published thumbnails, and the one a panel registers on.

    Returns:
        True when the routes were registered. False when they were registered already, or
        when the server could not be reached, in which case an interface asking for a
        preview gets a failed request.
    """
    global _registered
    if _registered:
        return False
    try:
        from aiohttp import web
        from server import PromptServer

        @PromptServer.instance.routes.get(ROUTE)
        async def get_preview(request):
            # `filename` is read and ignored. It carries no meaning here and resolves no
            # path: the frontend's mask editor refuses a URL without one, so accepting it is
            # what lets that editor be handed a picture this pack holds in memory.
            slot = request.query.get("slot")
            node = request.query.get("node_id")
            side = request.query.get("side")
            record, count, total = None, 0, 0
            try:
                # Asking is a panel saying it is open. A page that outlives a server restart
                # keeps its registration this way, without the panel having to notice.
                subscribe(node)
                record, count, total = _read(node, slot, side, request.query.get("frame", 0))
            except Exception as error:
                logger.debug("a preview request could not be answered (%s)", error)
            if record is None:
                # Named after what the store was asked for rather than after the raw query
                # value, so a slot of nothing but spaces, which reads the picture published
                # under no slot, is not reported as a name the node never published. An
                # unreadable side answers the same words a missing picture gets, never 400.
                words = _refusal(_slot(slot), _side(side)).encode("ascii", "replace").decode("ascii")
                return web.Response(status=204, headers={**NO_STORE, REFUSAL_HEADER: words})
            channel = _channel(request.query.get("channel"))
            # Decided before the headers are built: a channel request is answered with bytes
            # re-encoded here, so the length and the channel mode of the stored picture are
            # not the length and the channel mode of what goes out.
            body, mode = (
                (record.data, record.mode) if channel is None else _as_channel(record, channel)
            )
            headers = {
                # Built fresh for each answer. A shared dict mutated here would carry one
                # request's dimensions into every later response, including its refusals.
                **NO_STORE,
                SOURCE_WIDTH_HEADER: str(record.width),
                SOURCE_HEIGHT_HEADER: str(record.height),
                FRAME_COUNT_HEADER: str(count),
                FRAME_TOTAL_HEADER: str(total),
                SOURCE_MODE_HEADER: str(mode),
                KIND_HEADER: str(record.kind),
                ENCODED_BYTES_HEADER: str(len(body)),
                PROMPT_ID_HEADER: str(record.prompt),
            }
            return web.Response(body=body, content_type="image/png", headers=headers)

        @PromptServer.instance.routes.post(DISCARD_ROUTE)
        async def post_discard_clipspace(request):
            removed = 0
            try:
                body = await request.json()
                removed = discard_clipspace(body.get("names") or [])
            except Exception as error:
                logger.debug("no clipspace leftover was discarded (%s)", error)
            return web.json_response({"removed": removed}, headers=NO_STORE)

        @PromptServer.instance.routes.post(SUBSCRIBE_ROUTE)
        async def post_preview_subscription(request):
            try:
                watch = request.query.get("watch", "1").strip().lower()
                keep = watch not in ("0", "false", "off", "no", "")
                for value in request.query.getall("node_id", [])[:MAX_SUBSCRIPTIONS]:
                    subscribe(value) if keep else unsubscribe(value)
            except Exception as error:
                logger.debug("a preview subscription could not be recorded (%s)", error)
            # No body: the answer is the registration having been taken, and a panel that
            # never hears back simply pays one fetch that answers no picture.
            return web.Response(status=204, headers=NO_STORE)

    except Exception as error:
        logger.warning(
            "%s was not registered (%s: %s), so a node interface asking for the picture its "
            "node held gets a failed request",
            ROUTE, type(error).__name__, error,
        )
        logger.debug("%s could not be registered", ROUTE, exc_info=True)
        return False
    _registered = True
    logger.debug("%s is serving previews of both sides", ROUTE)
    return True


def _refusal(name, side) -> str:
    """The words a request that found no picture is answered with.

    Args:
        name: The slot the store was asked for, empty for the unnamed one.
        side: The side the store was asked for, or None when it could not be read.

    Returns:
        Text naming neither a path nor a store key that was not asked for.
    """
    if side == OUTPUT:
        if name:
            return f"that node has published no output image for {name} in this session"
        return "that node has not published an output image in this session"
    if name:
        return f"that node has published no image for {name} in this session"
    return "that node has not published an image in this session"


def _slot(slot) -> str | None:
    """A slot name as a store key, :data:`_DEFAULT_SLOT` for none, or None for the rest."""
    if slot is None:
        return _DEFAULT_SLOT
    if isinstance(slot, (str, int)):
        return str(slot).strip()
    return None


def _side(side) -> str | None:
    """A side as a store key, :data:`INPUT` for none, or None for anything unknown."""
    if side is None:
        return INPUT
    if not isinstance(side, str):
        return None
    name = side.strip().lower()
    if not name:
        return INPUT
    return name if name in SIDES else None


def _max_edge() -> int:
    """The longest edge a picture is reduced to, from the config.

    Returns:
        The configured limit, or :data:`NO_LIMIT` when it is unset, not a whole number, or
        the config cannot be read, since a picture at its own size is the safe answer for
        what is on screen.
    """
    try:
        from .. import config

        section, _, key = MAX_EDGE_KEY.partition(".")
        value = int(config.load_config()[section][key])
    except Exception:
        return NO_LIMIT
    return value if value > 0 else NO_LIMIT


def _fit_edge(width, height, channels, limit) -> int:
    """The longest edge one picture is drawn down to before it is encoded.

    Args:
        width: The picture's own width in pixels.
        height: The picture's own height in pixels.
        channels: How many samples each of its pixels carries.
        limit: The configured maximum edge, or :data:`NO_LIMIT` for none.

    Returns:
        An edge no longer than the picture's own, no longer than ``limit`` where one is set,
        and short enough that the reduced picture costs at most :data:`MAX_FRAME_BYTES`
        uncompressed.
    """
    longest = max(1, int(width), int(height))
    edge = min(longest, int(limit)) if limit else longest
    edge = max(1, edge)
    source = max(1, int(width) * int(height) * int(channels))
    reduction = edge / longest
    if source * reduction * reduction > MAX_FRAME_BYTES:
        edge = max(1, int(longest * math.sqrt(MAX_FRAME_BYTES / source)))
    return edge


def _encode_frames(tensor, kind, first_only) -> tuple[list[Picture], int]:
    """The frames of one tensor as PNG bytes, bounded before each encode.

    Args:
        tensor: An ``IMAGE`` or ``MASK`` tensor, read as ``kind`` says.
        kind: :data:`IMAGE_KIND` or :data:`MASK_KIND`.
        first_only: Encode the first frame alone.

    Returns:
        ``(frames, batch length)``. The frames are as many as the key's budget allowed, and
        the batch length is what the tensor held before any bound applied.

    Raises:
        ValueError: The tensor holds no picture.
    """
    from ..convert.tensors import image_planes, mask_planes

    planes = mask_planes(tensor) if kind == MASK_KIND else image_planes(tensor)
    if not planes:
        raise ValueError("the tensor holds no picture to preview")
    total = len(planes)
    if first_only:
        planes = planes[:1]
    elif total > MAX_FRAMES:
        logger.info(
            "a node published %d frames and the preview channel holds %d, so the interface "
            "shows the first %d",
            total, MAX_FRAMES, MAX_FRAMES,
        )
        planes = planes[:MAX_FRAMES]

    limit = _max_edge()
    prompt = executing_prompt_id() or ""
    pictures: list[Picture] = []
    spent = 0
    for plane in planes:
        # Asked before the encode rather than after it, so a frame that would not be kept is
        # never paid for. The first frame of a key is exempt: a slot that stored nothing is
        # an empty answer the interface draws as a node that has not run.
        if pictures and spent >= MAX_KEY_BYTES:
            logger.info(
                "a node published %d frames and %d of them fit the preview channel's %d byte "
                "budget for one slot, so the interface shows the first %d",
                total, len(pictures), MAX_KEY_BYTES, len(pictures),
            )
            break
        picture = _encode(plane, kind, limit, prompt)
        pictures.append(picture)
        spent += len(picture.data)
    return pictures, total


def _encode(plane, kind, limit, prompt) -> Picture:
    """One picture as PNG bytes, with the size the node held it at.

    Args:
        plane: One frame, as ``image_planes`` or ``mask_planes`` answers it.
        kind: :data:`IMAGE_KIND` or :data:`MASK_KIND`.
        limit: The configured maximum edge, or :data:`NO_LIMIT` for none.
        prompt: The prompt id to stamp the frame with, empty outside a run.

    Returns:
        A :class:`Picture`, its width and height being the picture the node held, which is
        what a gesture on the reduced picture is measured against.

    Raises:
        ValueError: The plane is not one picture.
    """
    from ..convert.tensors import mask2pil, plane2pil, plane_shape

    # Read off the tensor, so the reduction is decided before any array is materialised and
    # before the encode, which is where both the time and the stored bytes go.
    height, width, channels = plane_shape(plane)
    edge = _fit_edge(width, height, channels, limit)
    picture = mask2pil(plane) if kind == MASK_KIND else plane2pil(plane)
    source = picture.size
    mode = picture.mode
    if edge < max(source):
        picture.thumbnail((edge, edge))
    buffer = BytesIO()
    picture.save(buffer, format="PNG")
    return Picture(buffer.getvalue(), source[0], source[1], mode, kind, prompt)


#: Channels a caller may ask one stored picture for. ``rgb`` drops the alpha and ``a`` answers
#: the alpha alone, which is the pair ComfyUI's own mask editor fetches after it has taken the
#: picture itself. Any other value is ignored and the picture is answered whole.
CHANNELS = ("rgb", "a")


def _channel(value) -> str | None:
    """One of :data:`CHANNELS`, or None for a request that named no channel or an unknown one."""
    if not isinstance(value, str):
        return None
    name = value.strip().lower()
    return name if name in CHANNELS else None


def _as_channel(record, channel: str) -> tuple[bytes, str]:
    """One stored picture as the channel a caller asked for.

    Args:
        record: The stored picture.
        channel: ``rgb`` or ``a``, already checked by :func:`_channel`.

    Returns:
        ``(png bytes, channel mode)``, re-encoded. The stored bytes and the stored mode when
        the picture cannot be re-encoded, since a picture is a better answer than a failure.
    """
    try:
        from PIL import Image

        picture = Image.open(BytesIO(record.data))
        picture.load()
        if channel == "rgb":
            return _png(picture.convert("RGB")), "RGB"
        grey = picture.convert("L")
        if record.kind == MASK_KIND:
            # The editor reads its selection as 255 minus the alpha, so a mask covering an area
            # has to arrive transparent there for that area to open selected.
            alpha = grey.point(lambda level: 255 - level)
        else:
            # An image has no selection to offer, so it answers fully opaque rather than a
            # selection made out of its own brightness, which would be a guess.
            alpha = Image.new("L", grey.size, 255)
        out = picture.convert("RGB")
        out.putalpha(alpha)
        return _png(out), "RGBA"
    except Exception as error:
        logger.debug("channel %r could not be answered (%s)", channel, error)
        return record.data, record.mode


def _png(picture) -> bytes:
    """One Pillow image as PNG bytes."""
    buffer = BytesIO()
    picture.save(buffer, format="PNG")
    return buffer.getvalue()


def discard_clipspace(names) -> int:
    """Delete the uploads ComfyUI's mask editor made for a save nothing kept.

    Args:
        names: Filenames as the editor wrote them, with no directory part.

    Returns:
        How many files were deleted. 0 when the input directory cannot be found, which leaves
        every file exactly where it is.
    """
    try:
        import folder_paths

        root = Path(folder_paths.get_input_directory()).resolve()
    except Exception as error:
        logger.debug("the input directory could not be resolved (%s)", error)
        return 0
    removed = 0
    for value in list(names)[:MAX_DISCARD]:
        if not isinstance(value, str) or not CLIPSPACE_NAME.match(value.strip()):
            continue
        target = (root / value.strip()).resolve()
        # Resolved and compared against the root, so a name that walked out of it is refused
        # even though the pattern above already forbids a separator.
        if target.parent != root or not target.is_file():
            continue
        try:
            target.unlink()
            removed += 1
        except Exception as error:
            logger.debug("%s could not be discarded (%s)", value, error)
    if removed:
        logger.debug("discarded %d mask editor upload(s)", removed)
    return removed


#: Most files one discard call will delete, which is one save's worth with room to spare.
MAX_DISCARD = 8
