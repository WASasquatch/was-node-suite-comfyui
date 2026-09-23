"""Rebuilding a converted workflow inside a running graph, on the node-expansion API.

Nodes are rebuilt under a fresh prefix, and each exposed output node is resolved to
the node feeding it so its result leaves on a wire.
"""

from __future__ import annotations

from comfy_execution.graph_utils import GraphBuilder, is_link


def needed(prompt, wanted):
    """Ids reachable backward from ``wanted``, inclusive.

    Args:
        prompt: An API prompt.
        wanted: Ids to walk back from.

    Returns:
        The set of ids reached.
    """
    seen = set()
    stack = list(wanted)
    while stack:
        node_id = stack.pop()
        if node_id in seen or node_id not in prompt:
            continue
        seen.add(node_id)
        for value in prompt[node_id].get("inputs", {}).values():
            if is_link(value):
                stack.append(value[0])
    return seen


def producer(prompt, node_id):
    """The link feeding an output node, so its result can leave on a wire.

    Args:
        prompt: An API prompt.
        node_id: Id of a node the workflow presents a result from.

    Returns:
        ``[node_id, slot]`` for its first linked input, or ``None`` when it has none.
    """
    node = prompt.get(node_id)
    if node is None:
        return None
    for value in node.get("inputs", {}).values():
        if is_link(value):
            return value
    return None


def matching(value, kinds):
    """Which of a socket's types a value arrived as.

    Args:
        value: A value arriving on a wire.
        kinds: The types that socket carries.

    Returns:
        One of ``kinds``. A socket carrying one type answers it without reading the value;
        a socket carrying several picks by shape, and falls back to the first.
    """
    if not kinds:
        return None
    if len(kinds) == 1:
        return kinds[0]
    shape = getattr(value, "shape", None)
    if shape is not None:
        if len(shape) == 4 and "IMAGE" in kinds:
            return "IMAGE"
        if len(shape) == 3 and "MASK" in kinds:
            return "MASK"
    if isinstance(value, dict):
        if "samples" in value and "LATENT" in kinds:
            return "LATENT"
        if "waveform" in value and "AUDIO" in kinds:
            return "AUDIO"
    return kinds[0]


def overridden(prompt, assignments):
    """Apply widget values to a copy of a prompt.

    Args:
        prompt: An API prompt.
        assignments: ``{(api_id, widget_name): value}``.

    Returns:
        ``(prompt, applied)``: the copy, and the assignments that landed on a real node.
    """
    result = {
        node_id: {"class_type": node["class_type"], "inputs": dict(node.get("inputs", {}))}
        for node_id, node in prompt.items()
    }
    applied = []
    for (node_id, widget), value in assignments.items():
        if node_id in result:
            result[node_id]["inputs"][widget] = value
            applied.append((node_id, widget))
    return result, applied


def build(prompt, outputs, wired=None, swaps=None):
    """Rebuild a prompt as an expansion graph and bind its results.

    Args:
        prompt: An API prompt.
        outputs: Ids of the nodes whose results are wanted.
        wired: ``{(api_id, input_name): value}`` set after rebuilding, for feeding a value
            from the graph the player sits in.
        swaps: ``{(api_id, output_slot): value}``, each standing in for everything that
            node answered on that slot, so the node itself is not run.

    Returns:
        ``(graph, links)``: the finalized expansion, and one ``[node_id, slot]`` per
        entry of ``outputs``, or ``None`` where that output has nothing feeding it.
    """
    wired = wired or {}
    swaps = swaps or {}
    results = [producer(prompt, node_id) for node_id in outputs]
    roots = [link[0] for link in results if link] + [node_id for node_id, _ in wired]
    keep = needed(prompt, roots)

    graph = GraphBuilder()
    for node_id in keep:
        # Each copy is named and drawn as the node it stands for.
        graph.node(prompt[node_id]["class_type"], id=node_id).set_override_display_id(node_id)
    for node_id in keep:
        node = graph.lookup_node(node_id)
        for name, value in prompt[node_id].get("inputs", {}).items():
            if is_link(value):
                upstream = graph.lookup_node(value[0])
                node.set_input(name, None if upstream is None else upstream.out(value[1]))
            else:
                node.set_input(name, value)
    for (node_id, name), value in wired.items():
        node = graph.lookup_node(node_id)
        if node is not None:
            node.set_input(name, value)

    links = []
    for link in results:
        node = graph.lookup_node(link[0]) if link else None
        links.append(None if node is None else node.out(link[1]))

    for (node_id, slot), value in swaps.items():
        if graph.lookup_node(node_id) is None:
            continue
        graph.replace_node_output(node_id, slot, value)
        for index, link in enumerate(links):
            if link and link[0] == graph.prefix + node_id and link[1] == slot:
                links[index] = value

    finalized = graph.finalize()
    return _pruned(finalized, links), links


def _pruned(graph, links):
    """Drop the nodes nothing a result needs still depends on."""
    wanted = [link[0] for link in links if is_link(link)]
    reachable = needed(graph, wanted)
    return {node_id: node for node_id, node in graph.items() if node_id in reachable}
