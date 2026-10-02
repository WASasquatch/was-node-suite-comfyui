"""What a channel needs before it publishes: a browser, a reader, a node id.

:func:`watching` reads connected clients, :func:`wanted` the panels registered as
interested, :func:`node_key` the key a publisher files under. :data:`NO_STORE` is the cache
header.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

from .. import log

__all__ = [
    "MAX_SUBSCRIPTIONS",
    "NO_STORE",
    "PROMPT_ID_HEADER",
    "REFUSAL_HEADER",
    "clear_subscriptions",
    "executing_class_type",
    "executing_node_id",
    "executing_prompt_id",
    "node_key",
    "subscribe",
    "subscriptions",
    "unsubscribe",
    "wanted",
    "watching",
]

logger = log.get_logger("interface.channel")

#: What a channel publishes is replaced whenever the node runs again, so the browser is told to
#: hold no copy of its own: a stale answer drawn as the last run is worse than none.
NO_STORE = {"Cache-Control": "no-store"}

#: Header carrying the prompt a published answer was made under, empty when the publish happened
#: outside a run. A panel drawing two answers together compares these before it pairs them, since
#: two sides of one node can come from two different runs.
PROMPT_ID_HEADER = "X-WAS-Prompt"

#: Header carrying the words a request that found nothing published is answered with, beside a
#: 204 and no body.
REFUSAL_HEADER = "X-WAS-Refusal"

#: How many nodes may hold a registered panel at once. A page registers one entry per interface
#: on the canvas and drops it when the node goes away, so the bound is a backstop against a
#: registration that is never released rather than a limit anybody meets.
MAX_SUBSCRIPTIONS = 256

# A panel registers on the server's thread and a node reads the registry on the thread running
# the prompt, so every read and write of it goes through this.
_lock = threading.Lock()

#: Execution ids of the nodes a panel is open on, most recently registered last. Values are
#: unused: the mapping is an ordered set, so the oldest registration is the one the bound drops.
_subscribed: OrderedDict[str, None] = OrderedDict()


def watching() -> bool:
    """Whether any browser is connected to this ComfyUI.

    Returns:
        True while the server holds at least one open client socket. False when it holds
        none, when the server is not running, and when it cannot be reached, which is the
        case for a headless run, an API call and a command line run alike.
    """
    try:
        from server import PromptServer

        return bool(PromptServer.instance.sockets)
    except Exception as error:
        logger.debug("no client could be counted (%s), so nothing is published", error)
        return False


def subscribe(key) -> bool:
    """Record that a panel is open on one node, so publishing for it is worth the encode.

    Args:
        key: The node's execution id, as :func:`node_key` reads it.

    Returns:
        True when the id was recorded. False when it is not an id.
    """
    name = node_key(key)
    if name is None:
        return False
    with _lock:
        _subscribed.pop(name, None)
        _subscribed[name] = None
        while len(_subscribed) > MAX_SUBSCRIPTIONS:
            _subscribed.popitem(last=False)
    return True


def unsubscribe(key) -> bool:
    """Forget a panel that has gone, so its node stops paying for pictures nobody reads.

    Args:
        key: The node's execution id, as :func:`node_key` reads it.

    Returns:
        True when a registration was dropped. False when there was none.
    """
    name = node_key(key)
    if name is None:
        return False
    with _lock:
        if name not in _subscribed:
            return False
        del _subscribed[name]
        return True


def wanted(key) -> bool:
    """Whether a panel is open on one node.

    Args:
        key: The node's execution id, as :func:`node_key` reads it.

    Returns:
        True while a panel on that node is registered. False otherwise, which is the answer
        for every node in a graph whose interfaces nobody has open.
    """
    name = node_key(key)
    if name is None:
        return False
    with _lock:
        return name in _subscribed


def subscriptions() -> tuple[str, ...]:
    """Every node id a panel is registered on, oldest registration first.

    Returns:
        The ids, as a snapshot that the registry going on changing does not alter.
    """
    with _lock:
        return tuple(_subscribed)


def clear_subscriptions() -> int:
    """Drop every registration, for the moment the last browser goes away.

    Returns:
        How many registrations were dropped.
    """
    with _lock:
        count = len(_subscribed)
        _subscribed.clear()
    return count


def node_key(node_id) -> str | None:
    """A node id as a store key, or None when it cannot be one.

    Args:
        node_id: A node's graph id, as a string or an integer. Anything else, including a
            missing or malformed query value, answers None.

    Returns:
        The id with surrounding space removed, or None when it is empty or not an id.
    """
    if isinstance(node_id, (str, int)):
        key = str(node_id).strip()
        return key or None
    return None


def executing_node_id() -> str | None:
    """The graph id of the node ComfyUI is executing, from its own execution context.

    Returns:
        The id ``io.Hidden.unique_id`` carries, which is the colon joined path for a node
        inside a subgraph. None when no prompt is running, when the context cannot be read,
        and when the id names an expansion clone, so a body called directly rather than
        through a prompt files under nothing.
    """
    try:
        from comfy_execution.utils import get_executing_context

        node_id = getattr(get_executing_context(), "node_id", None)
    except Exception as error:
        logger.debug("the executing node id could not be read (%s)", error)
        return None
    # A graph expanded at run time, which is what a loop body and an expanding node both
    # become, files its clones under a prefix joined by dots. Those ids exist for one run and
    # for one iteration, no interface composes them, and one loop would fill an entire store
    # with keys nobody can fetch. A subgraph path is joined by colons, so the dot is exact.
    if isinstance(node_id, str) and "." in node_id:
        logger.debug(
            "node %s is an expansion clone, so it publishes nothing to its interface", node_id,
        )
        return None
    return node_id


def executing_prompt_id() -> str | None:
    """The id of the prompt ComfyUI is running, from the same execution context.

    Returns:
        The prompt id, or None when no prompt is running and when the context cannot be
        read, so a body called outside a prompt stamps its answer with nothing.
    """
    try:
        from comfy_execution.utils import get_executing_context

        return getattr(get_executing_context(), "prompt_id", None)
    except Exception as error:
        logger.debug("the executing prompt id could not be read (%s)", error)
        return None


def executing_class_type(node_id=None) -> str | None:
    """The node id of the class ComfyUI is running, as the prompt names it.

    Args:
        node_id: The graph id to look up. Left out, the node being executed.

    Returns:
        The ``class_type`` string, or None when no prompt is running and when the queue
        cannot be read.
    """
    key = node_key(node_id if node_id is not None else executing_node_id())
    if key is None:
        return None
    try:
        from server import PromptServer

        running, _ = PromptServer.instance.prompt_queue.get_current_queue_volatile()
        for item in running:
            prompt = item[2] if len(item) > 2 else None
            entry = prompt.get(key) if isinstance(prompt, dict) else None
            if isinstance(entry, dict):
                declared = entry.get("class_type")
                if isinstance(declared, str) and declared:
                    return declared
    except Exception as error:
        logger.debug("the executing class type could not be read (%s)", error)
    return None
