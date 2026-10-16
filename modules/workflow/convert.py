"""Converting a saved UI workflow into the API prompt form the backend executes.

The API form is ``{node_id: {"class_type": str, "inputs": dict}}``, with every link
written as ``[node_id, output_slot]``. A node inside a subgraph flattens to
``"<instance_id>:<inner_id>"``.
"""

from __future__ import annotations

from .. import log

logger = log.get_logger("workflow.convert")

#: Node modes a saved workflow uses for a node that does not run as itself.
MUTED = 2
BYPASSED = 4

#: Ids the boundary proxies carry inside a subgraph definition.
BOUNDARY_IN = -10
BOUNDARY_OUT = -20

#: Input kinds drawn as a widget rather than a socket.
WIDGET_KINDS = ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO")

#: Deepest subgraph nesting a workflow may flatten through.
MAX_DEPTH = 32

#: Most nodes a workflow may flatten to, counting every expanded subgraph instance.
MAX_NODES = 20000

#: Most nodes, links and input slots read across every expanded level.
MAX_ITEMS = 1_000_000


def input_spec(class_type):
    """A registered node's declared inputs.

    Args:
        class_type: A registered node id.

    Returns:
        Its ``INPUT_TYPES()``, or ``None`` when no node is registered under that id or the
        node could not describe itself.
    """
    from nodes import NODE_CLASS_MAPPINGS

    node_class = NODE_CLASS_MAPPINGS.get(class_type)
    if node_class is None:
        return None
    try:
        return node_class.INPUT_TYPES()
    except Exception as error:
        logger.debug("%s could not describe its inputs: %s", class_type, error)
        return None


def declared_widgets(class_type):
    """Widget input names of a registered node, in the order the canvas draws them.

    Args:
        class_type: A registered node id.

    Returns:
        The widget input names, or ``None`` when no node is registered under that id.
    """
    spec = input_spec(class_type)
    if spec is None:
        return None
    names = []
    for section in ("required", "optional"):
        for name, config in (spec.get(section) or {}).items():
            config = config if isinstance(config, (list, tuple)) else (config,)
            kind = config[0] if config else None
            options = config[1] if len(config) > 1 and isinstance(config[1], dict) else {}
            if options.get("forceInput"):
                continue
            if isinstance(kind, (list, tuple)) or kind in WIDGET_KINDS:
                names.append(name)
    return names


def _links(graph):
    """Map every link id in one graph level to ``(origin_id, origin_slot)``."""
    table = {}
    for link in graph.get("links") or []:
        if isinstance(link, dict):
            table[link["id"]] = (str(link["origin_id"]), int(link["origin_slot"]))
        else:
            table[link[0]] = (str(link[1]), int(link[2]))
    return table


class Converter:
    """Flattens a saved UI workflow, and any subgraph within it, to API prompt form.

    Attributes:
        subgraphs: Subgraph definitions of the workflow, keyed on definition id.
        prompt: The API prompt built so far.
        missing: Class types encountered that no registered node provides.
        origins: ``{api_id: (definition_id, inner_id)}``, where ``definition_id`` is
            ``None`` for a node sitting in the root graph.
        nodes: Nodes read so far, across every expanded level.
        items: Nodes, links and input slots read so far.
    """

    def __init__(self, workflow, lookup=declared_widgets):
        definitions = workflow.get("definitions") or {}
        self.subgraphs = {sg["id"]: sg for sg in definitions.get("subgraphs") or []}
        self.lookup = lookup
        self.prompt = {}
        self.missing = set()
        self.origins = {}
        self.nodes = 0
        self.items = 0
        self._inner = {}

    def convert(self, workflow):
        """Build the API prompt for a whole workflow.

        Args:
            workflow: A saved UI workflow.

        Returns:
            ``{node_id: {"class_type": str, "inputs": dict}}``.

        Raises:
            ValueError: A subgraph contains itself, nesting passes ``MAX_DEPTH``, or the
                expanded workflow passes ``MAX_NODES`` nodes or ``MAX_ITEMS`` items.
        """
        self.prompt = {}
        self.missing = set()
        self.origins = {}
        self.nodes = 0
        self.items = 0
        self._walk(workflow, "", None, None, ())
        return self.prompt

    def _spend(self, nodes=0, items=0):
        """Count nodes and items read, raising once either passes its ceiling."""
        self.nodes += nodes
        self.items += nodes + items
        if self.nodes > MAX_NODES:
            raise ValueError(
                f"the workflow expands to more than {MAX_NODES} nodes once every subgraph "
                f"instance is unpacked. Open it and remove some nested subgraph instances"
            )
        if self.items > MAX_ITEMS:
            raise ValueError(
                f"the workflow holds more than {MAX_ITEMS} nodes, links and inputs once "
                f"every subgraph instance is unpacked. Open it and remove some nested "
                f"subgraph instances"
            )

    def _enter(self, kind, path):
        """The definition path one level into ``kind``, refusing a cycle or deep nesting."""
        name = self.subgraphs[kind].get("name") or kind
        if kind in path:
            raise ValueError(
                f"subgraph {name!r} contains an instance of itself, so the workflow cannot "
                f"be flattened. Open the subgraph and remove that instance"
            )
        if len(path) >= MAX_DEPTH:
            raise ValueError(
                f"subgraph {name!r} sits more than {MAX_DEPTH} subgraphs deep. Open the "
                f"workflow and unpack some of the nesting"
            )
        return (*path, kind)

    def _walk(self, graph, prefix, boundary, definition, path):
        """Emit every node of one graph level, recursing into each subgraph instance."""
        listed = graph.get("nodes") or []
        self._spend(len(listed), len(graph.get("links") or []))
        nodes = {str(n["id"]): n for n in listed}
        links = _links(graph)
        skipped = {i for i, n in nodes.items() if n.get("mode") in (MUTED, BYPASSED)}

        def source(origin, slot):
            seen = set()
            while origin != str(BOUNDARY_IN):
                node = nodes.get(origin)
                if node is None:
                    return None
                if origin not in skipped:
                    if node["type"] in self.subgraphs:
                        return self._through_output(node, prefix, slot, path)
                    return [prefix + origin, slot]
                if node.get("mode") == MUTED or origin in seen:
                    return None
                seen.add(origin)
                self._spend(items=1 + len(node.get("inputs") or []))
                outputs = node.get("outputs") or []
                wanted = outputs[slot].get("type") if slot < len(outputs) else None
                # A bypassed node passes through from its first input of the same type.
                upstream = None
                for candidate in node.get("inputs") or []:
                    if candidate.get("type") != wanted or candidate.get("link") is None:
                        continue
                    upstream = links.get(candidate["link"])
                    if upstream:
                        break
                if not upstream:
                    return None
                origin, slot = upstream
            return boundary.get(slot) if boundary else None

        def wired(node):
            found = {}
            for slot in node.get("inputs") or []:
                if slot.get("link") is None or slot["link"] not in links:
                    continue
                origin, index = links[slot["link"]]
                resolved = source(origin, index)
                if resolved is not None:
                    found[slot["name"]] = resolved
            return found

        for node_id, node in nodes.items():
            if node_id in skipped:
                continue
            self._spend(items=len(node.get("inputs") or []))
            if node["type"] in self.subgraphs:
                inner = self._enter(node["type"], path)
                self._walk(
                    self.subgraphs[node["type"]],
                    f"{prefix}{node_id}:",
                    self._boundary_inputs(node, links, source),
                    node["type"],
                    inner,
                )
                continue

            widgets = self.lookup(node["type"])
            if widgets is None:
                self.missing.add(node["type"])
                continue
            linked = wired(node)
            inputs = dict(linked)
            free = [name for name in widgets if name not in linked]
            for name, value in zip(free, node.get("widgets_values") or []):
                inputs[name] = value
            self.prompt[prefix + node_id] = {"class_type": node["type"], "inputs": inputs}
            self.origins[prefix + node_id] = (definition, node_id)

    def _boundary_inputs(self, instance, links, source):
        """Map each of a subgraph's boundary input slots to what the instance is fed."""
        inner = self.subgraphs[instance["type"]]
        slots = instance.get("inputs") or []
        mapped = {}
        for index in range(min(len(inner.get("inputs") or []), len(slots))):
            if slots[index].get("link") is None:
                continue
            upstream = links.get(slots[index]["link"])
            if upstream:
                resolved = source(upstream[0], upstream[1])
                if resolved is not None:
                    mapped[index] = resolved
        return mapped

    def _definition(self, kind):
        """``(exits, nodes)`` of one subgraph definition, read once.

        Args:
            kind: A subgraph definition id.

        Returns:
            ``exits`` maps each output slot to the first ``(origin_id, origin_slot)`` feeding
            it; ``nodes`` maps each inner node id to its node.
        """
        if kind not in self._inner:
            inner = self.subgraphs[kind]
            wires = inner.get("links") or []
            listed = inner.get("nodes") or []
            self._spend(items=len(wires) + len(listed))
            exits = {}
            for link in wires:
                try:
                    if isinstance(link, dict):
                        target, target_slot = link["target_id"], link["target_slot"]
                        origin, origin_slot = str(link["origin_id"]), int(link["origin_slot"])
                    else:
                        target, target_slot = link[3], link[4]
                        origin, origin_slot = str(link[1]), int(link[2])
                    if int(target) == BOUNDARY_OUT:
                        exits.setdefault(int(target_slot), (origin, origin_slot))
                except (KeyError, IndexError, TypeError, ValueError):
                    continue
            self._inner[kind] = (exits, {str(n["id"]): n for n in listed})
        return self._inner[kind]

    def _through_output(self, instance, prefix, slot, path):
        """Resolve a subgraph instance's output slot to the inner node feeding it."""
        while True:
            path = self._enter(instance["type"], path)
            prefix = f"{prefix}{instance['id']}:"
            exits, inner_nodes = self._definition(instance["type"])
            if slot not in exits:
                return None
            origin, slot = exits[slot]
            node = inner_nodes.get(origin)
            if node is None or node["type"] not in self.subgraphs:
                return [prefix + origin, slot]
            instance = node


def to_prompt(workflow, lookup=declared_widgets):
    """Convert a saved UI workflow to API prompt form.

    Args:
        workflow: A saved UI workflow.
        lookup: Returns a node id's widget input names, or ``None`` when unregistered.

    Returns:
        ``(prompt, missing)``: the API prompt, and the class types no node provides.

    Raises:
        ValueError: A subgraph contains itself, or the workflow expands past its ceilings.
    """
    converter = Converter(workflow, lookup)
    prompt = converter.convert(workflow)
    return prompt, converter.missing


def prompt_with_origins(workflow, lookup=declared_widgets):
    """Convert a workflow, keeping where each node came from.

    Args:
        workflow: A saved UI workflow.
        lookup: Returns a node id's widget input names, or ``None`` when unregistered.

    Returns:
        ``(prompt, origins, missing)``, with ``origins`` mapping each API id to
        ``(definition_id, inner_id)``.

    Raises:
        ValueError: A subgraph contains itself, or the workflow expands past its ceilings.
    """
    converter = Converter(workflow, lookup)
    prompt = converter.convert(workflow)
    return prompt, converter.origins, converter.missing


def declared_input(class_type, name):
    """The declared configuration of one input on a registered node.

    Args:
        class_type: A registered node id.
        name: An input name on that node.

    Returns:
        The declared ``(kind, options)`` configuration, or ``None`` when either the node
        or the input is unknown.
    """
    spec = input_spec(class_type)
    if spec is None:
        return None
    for section in ("required", "optional"):
        config = (spec.get(section) or {}).get(name)
        if config is not None:
            return config if isinstance(config, (list, tuple)) else (config,)
    return None


def _listed(options, most=8):
    """Options written out, cut to ``most`` with the rest counted."""
    shown = ", ".join(repr(option) for option in options[:most])
    return shown if len(options) <= most else f"{shown}, and {len(options) - most} more"


def coerced(value, config, where=""):
    """Bring a value to the type an input declares.

    Args:
        value: The value to convert.
        config: A declared input configuration, or ``None`` to pass the value through.
        where: Names the input, for the message raised when the value will not convert.

    Returns:
        The value as the declared type. A list or dict passes through untouched, as does
        anything whose declared kind is a socket type rather than a widget kind.

    Raises:
        ValueError: The value cannot be brought to the declared type.
    """
    if config is None or isinstance(value, (list, dict)) or value is None:
        return value
    kind = config[0]
    settings = config[1] if len(config) > 1 and isinstance(config[1], dict) else {}
    options = list(kind) if isinstance(kind, (list, tuple)) else settings.get("options")
    if options is not None:
        if value in options:
            return value
        raise ValueError(
            f"{where or 'the value'} is {value!r}, which is not one of the "
            f"{len(options)} it offers: {_listed(options)}"
        )
    if kind not in WIDGET_KINDS:
        return value
    try:
        if kind == "INT":
            return int(round(float(value)))
        if kind == "FLOAT":
            return float(value)
        if kind == "BOOLEAN":
            return value if isinstance(value, bool) else str(value).strip().lower() in ("1", "true", "yes", "on")
        if kind == "STRING":
            return value if isinstance(value, str) else str(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{where or 'the value'} is {value!r}, which will not read as {kind}: {error}"
        ) from error
    return value


def declared_output(class_type, slot):
    """What one output of a registered node carries.

    Args:
        class_type: A registered node id.
        slot: The output's index.

    Returns:
        ``{"type", "name", "tooltip"}``, or ``None`` when the node or the slot is unknown.
    """
    from nodes import NODE_CLASS_MAPPINGS

    node_class = NODE_CLASS_MAPPINGS.get(class_type)
    types = getattr(node_class, "RETURN_TYPES", None) if node_class else None
    if not types or slot >= len(types):
        return None
    names = getattr(node_class, "RETURN_NAMES", None) or ()
    tips = getattr(node_class, "OUTPUT_TOOLTIPS", None) or ()
    kind = types[slot]
    return {
        "type": str(kind) if isinstance(kind, str) else "COMBO",
        "name": names[slot] if slot < len(names) else (kind if isinstance(kind, str) else "value"),
        "tooltip": tips[slot] if slot < len(tips) else None,
    }


def output_slot(class_type, wanted):
    """Where a node answers a given socket type.

    Args:
        class_type: A registered node id.
        wanted: The socket type to find.

    Returns:
        The index of its first output of that type, or ``None`` when it has none.
    """
    from nodes import NODE_CLASS_MAPPINGS

    node_class = NODE_CLASS_MAPPINGS.get(class_type)
    types = getattr(node_class, "RETURN_TYPES", None) if node_class else None
    for index, kind in enumerate(types or ()):
        if type(kind) is str and kind == wanted:
            return index
    return None


def is_picker(config):
    """Whether an input is a menu of choices rather than a typed value.

    Args:
        config: A declared input configuration, or ``None``.

    Returns:
        True when the input is drawn as a list of options.
    """
    if not config:
        return False
    kind = config[0]
    if isinstance(kind, (list, tuple)):
        return True
    settings = config[1] if len(config) > 1 and isinstance(config[1], dict) else {}
    return settings.get("options") is not None


def first_picker(class_type):
    """The first input of a node that is a menu of choices.

    Args:
        class_type: A registered node id.

    Returns:
        Its name, or ``None`` when the node has no menu input.
    """
    spec = input_spec(class_type)
    if spec is None:
        return None
    for section in ("required", "optional"):
        for name, config in (spec.get(section) or {}).items():
            if is_picker(config if isinstance(config, (list, tuple)) else (config,)):
                return name
    return None
