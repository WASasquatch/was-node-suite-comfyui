"""Error reports for nodes a loop runs as copies.

:func:`register` makes a failure in a copied node name that node and its loop pass, in place of
the node that made the copy.
"""

from __future__ import annotations

from .. import log

__all__ = ["LOOP_CLOSERS", "origin_of", "register"]

logger = log.get_logger("logic.loop_errors")

#: Node ids whose copies are numbered as passes of a loop.
LOOP_CLOSERS = ("WASWhileLoopClose", "WASForLoopClose")

_registered = False


def origin_of(error: dict, prompt: dict, registry) -> dict:
    """An error report naming the copied node that raised it.

    Args:
        error: The report ComfyUI built, its ``node_id`` the node that made the copy.
        prompt: The submitted prompt, keyed by node id.
        registry: The run's progress registry, holding each node's state and the copies.

    Returns:
        A new report naming the copy's own node and pass, or ``error`` itself where no running
        copy shown as another node is found.
    """
    from comfy_execution.progress import NodeState

    reported = error.get("node_id")
    dynprompt = registry.dynprompt
    running = [
        node_id for node_id, state in registry.nodes.items()
        if state.get("state") == NodeState.Running
        and node_id != reported
        and dynprompt.get_real_node_id(node_id) == reported
        and dynprompt.get_display_node_id(node_id) != reported
    ]
    if not running:
        return error
    copy = running[-1]
    shown = dynprompt.get_display_node_id(copy)
    if shown not in prompt:
        return error

    passes = 1
    parent = dynprompt.get_parent_node_id(copy)
    while parent is not None:
        if dynprompt.get_display_node_id(parent) == reported:
            passes += 1
        parent = dynprompt.get_parent_node_id(parent)

    closer = prompt.get(reported, {}).get("class_type", "")
    title = (prompt[shown].get("_meta") or {}).get("title") or prompt[shown].get("class_type", "")
    where = (
        f"pass {passes} of the loop node {reported} closes" if closer in LOOP_CLOSERS
        else f"the nodes node {reported} ({closer}) added to the run"
    )
    report = {**error, "node_id": shown}
    message = error.get("exception_message")
    if isinstance(message, str):
        report["exception_message"] = f"{message.rstrip()}\n\n{title} (node {shown}), on {where}."
    return report


def register() -> bool:
    """Name copied nodes in ComfyUI's error reports.

    Returns:
        True when the reports were hooked. False when they were hooked already, or when this
        ComfyUI has no error report to hook, in which case a failure in a loop's copy is
        reported against the loop's closing node.
    """
    global _registered
    if _registered:
        return False
    try:
        import execution
        from comfy_execution.progress import get_progress_state

        original = execution.PromptExecutor.handle_execution_error
    except Exception as error:
        logger.debug("loop error reports were not hooked (%s: %s)", type(error).__name__, error)
        return False

    def handle_execution_error(self, *args, **kwargs):
        if len(args) == 6 and not kwargs and isinstance(args[4], dict):
            prompt_id, prompt, current_outputs, executed, error, ex = args
            try:
                error = origin_of(error, prompt, get_progress_state())
            except Exception as failure:
                logger.debug("the copied node behind an error was not found (%s)", failure)
            args = (prompt_id, prompt, current_outputs, executed, error, ex)
        return original(self, *args, **kwargs)

    handle_execution_error.__wrapped__ = original
    execution.PromptExecutor.handle_execution_error = handle_execution_error
    _registered = True
    logger.debug("error reports name the copied node a loop ran")
    return True
