"""A bus that carries named extras beside the five standard members."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import BUS, LIST

#: Most extra slots the list grows to.
MAX_SLOTS = 16


class BusNodeDynamic(io.ComfyNode):
    """Bus Node with an autogrow slot list of extra named values on the same wire."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        template = io.Autogrow.TemplatePrefix(
            input=io.AnyType.Input(
                "slot",
                tooltip=(
                    "A value of any type to carry on the bus, named after this slot. A new "
                    "empty slot appears as soon as this one is connected."
                ),
            ),
            prefix="slot",
            min=0,
            max=MAX_SLOTS,
        )
        return io.Schema(
            node_id="WASBusNodeDynamic",
            display_name="Bus Node (Dynamic)",
            search_aliases=[
                "WASBusNodeDynamic", "Bus Node (Dynamic)",
                "bus",
                "reroute",
                "pipe",
                "dynamic bus",
                "carry",
            ],
            category="WAS Suite/Utilities",
            description=(
                "Bundle model, clip, vae, positive and negative onto one wire, along with "
                "any number of extra named values. Reads and writes the same BUS as Bus "
                "Node, with the extras carried alongside."
            ),
            inputs=[
                BUS.Input(
                    "bus",
                    optional=True,
                    tooltip=(
                        "An incoming bundle from either bus node. Its five members and any "
                        "extras it carries pass through unless something here replaces them. "
                        "Leave it disconnected on the first node of a chain."
                    ),
                ),
                io.Model.Input(
                    "model",
                    optional=True,
                    tooltip=(
                        "Diffusion model to put on the bus. Connected, it replaces whatever "
                        "model arrived on the bus input; disconnected, the bus keeps its own. "
                        "Unlike Bus Node, this node does not insist on one being present."
                    ),
                ),
                io.Clip.Input(
                    "clip",
                    optional=True,
                    tooltip=(
                        "Text encoder to put on the bus. Connected, it replaces whatever clip "
                        "arrived on the bus input; disconnected, the bus keeps its own."
                    ),
                ),
                io.Vae.Input(
                    "vae",
                    optional=True,
                    tooltip=(
                        "VAE to put on the bus. Connected, it replaces whatever vae arrived "
                        "on the bus input; disconnected, the bus keeps its own."
                    ),
                ),
                io.Conditioning.Input(
                    "positive",
                    optional=True,
                    tooltip=(
                        "Positive conditioning to put on the bus. Connected, it replaces "
                        "whatever arrived on the bus input."
                    ),
                ),
                io.Conditioning.Input(
                    "negative",
                    optional=True,
                    tooltip=(
                        "Negative conditioning to put on the bus. Connected, it replaces "
                        "whatever arrived on the bus input."
                    ),
                ),
                io.Int.Input(
                    "unpack_slot",
                    default=0,
                    min=0,
                    max=MAX_SLOTS - 1,
                    step=1,
                    tooltip=(
                        "Which extra comes out on slot, counted from 0 over slot_names; past "
                        "the end gives nothing. Chain another of these nodes to take a second "
                        "extra off."
                    ),
                ),
                io.Autogrow.Input(
                    "slots",
                    template=template,
                    tooltip=(
                        "Extra values to put on the bus, named slot0, slot1 and so on. A slot "
                        "replaces an extra of the same name already on the bus, so it can be "
                        f"updated part way along a chain. Up to {MAX_SLOTS} slots."
                    ),
                ),
            ],
            outputs=[
                BUS.Output(
                    display_name="bus",
                    tooltip=(
                        "Everything bundled onto one wire: the five members and every extra. "
                        "Readable by Bus Node, which will see the five members and drop the "
                        "extras."
                    ),
                ),
                io.Model.Output(
                    display_name="model",
                    tooltip="The model now on the bus: the model input, or the incoming bus's.",
                ),
                io.Clip.Output(
                    display_name="clip",
                    tooltip="The clip now on the bus: the clip input, or the incoming bus's.",
                ),
                io.Vae.Output(
                    display_name="vae",
                    tooltip="The vae now on the bus: the vae input, or the incoming bus's.",
                ),
                io.Conditioning.Output(
                    display_name="positive",
                    tooltip=(
                        "The positive conditioning now on the bus: the positive input, or "
                        "the incoming bus's."
                    ),
                ),
                io.Conditioning.Output(
                    display_name="negative",
                    tooltip=(
                        "The negative conditioning now on the bus: the negative input, or "
                        "the incoming bus's."
                    ),
                ),
                io.AnyType.Output(
                    display_name="slot",
                    tooltip=(
                        "The extra named by unpack_slot, on a socket that accepts any type. "
                        "Empty when the bus carries no extra at that position."
                    ),
                ),
                LIST.Output(
                    display_name="slot_names",
                    tooltip=(
                        "The names of every extra on the bus, in the order unpack_slot "
                        "counts them. Wire it into Text List to Text to see what is being "
                        "carried."
                    ),
                ),
                io.Int.Output(
                    display_name="slot_count",
                    tooltip="How many extras the bus carries.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        bus=None,
        model=None,
        clip=None,
        vae=None,
        positive=None,
        negative=None,
        unpack_slot=0,
        slots=None,
    ) -> io.NodeOutput:
        from ...modules.compat.bus import DynamicBus, extras_of, members_of

        incoming = members_of(bus)
        supplied = (model, clip, vae, positive, negative)
        members = [given if given is not None else carried for given, carried in zip(supplied, incoming)]

        extras = extras_of(bus)
        for name, value in sorted((slots or {}).items(), key=lambda entry: (len(entry[0]), entry[0])):
            if value is not None:
                extras[name] = value

        names = list(extras)
        chosen = extras[names[unpack_slot]] if 0 <= unpack_slot < len(names) else None
        return io.NodeOutput(
            DynamicBus(members, extras), *members, chosen, names, len(names)
        )
