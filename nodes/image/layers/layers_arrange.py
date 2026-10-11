"""Place, size and order the layers of a stack, and pass the stack on."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.image import layer_arrange
from ....modules.interface import preview, run_result
from ....modules.log import get_logger

logger = get_logger("nodes.image.layers")

#: Slot the layer thumbnails are published under.
THUMBNAIL_SLOT = "layers"

#: Name every published table body carries.
TABLE_BODY = "layers"


class LayersArrange(io.ComfyNode):
    """Apply a saved arrangement to a ``LAYERS`` document and answer the document."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLayersArrange",
            display_name="Layers Arrange",
            search_aliases=[
                "WASLayersArrange",
                "Layers Arrange",
                "arrange layers",
                "move layer",
                "resize layer",
                "reorder layers",
                "layer stack",
                "compositor",
            ],
            category="WAS Suite/Image/Layers",
            description=(
                "Move, resize, reorder, hide and fade the layers of a stack, and pass the "
                "stack on rather than flattening it, so the result goes on to Create "
                "Layered Image, Layers to Image Batch or another arrange. The panel lists "
                "the layers and writes what it changes into the arrangement box, which a "
                "run with no browser can hold instead."
            ),
            inputs=[
                io.Layers.Input(
                    "layers",
                    tooltip=(
                        "The stack to arrange. Wire in Add Layer, Layer Edit, Layers From "
                        "Bounding Boxes or anything else answering a LAYERS document; layer "
                        "0 is the bottom of the stack."
                    ),
                ),
                io.String.Input(
                    "arrangement",
                    multiline=True,
                    default="{}",
                    tooltip=(
                        "Where each layer goes, as JSON keyed on its index from the bottom: "
                        '{"0": {"x": 64, "z_index": 2}}. Optional keys: x, y, w, h in '
                        "pixels, rotation in degrees, opacity 0.0 to 1.0, visible, z_index; "
                        "omitted keys are unchanged. The panel writes this."
                    ),
                ),
            ],
            outputs=[
                io.Layers.Output(
                    display_name="layers",
                    tooltip=(
                        "The same stack with the arrangement applied, for Create Layered "
                        "Image or Layers to Image Batch. Layer 0 is still the bottom."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, layers, arrangement="{}") -> io.NodeOutput:
        changes = layer_arrange.arrangement(arrangement)
        source = layer_arrange.entries(layers)
        if not source:
            raise ValueError(
                "Layers Arrange was handed a stack with no layer carrying a picture. Wire a "
                "LAYERS document that Add Layer, Layer Edit or Layers From Bounding Boxes "
                "has put a layer into."
            )

        # An index past the top of the stack is counted and skipped.
        ignored = sorted(index for index in changes if index >= len(source))
        placed = layer_arrange.arranged(
            source, {index: changes[index] for index in changes if index < len(source)}
        )
        document = layer_arrange.rebuilt(layers, placed)

        cls.report(source, placed, document, ignored)
        logger.info(
            "Layers Arrange placed %d layer(s) from %d arrangement entry(s)",
            len(placed), len(changes) - len(ignored),
        )
        return io.NodeOutput(document)

    @classmethod
    def report(cls, source, placed, document, ignored) -> None:
        """Draw the arranged stack on the node. Never raises.

        Args:
            source: The layers as they arrived, lowest in the stack first.
            placed: The same layers with the arrangement applied.
            document: The document the node answered with.
            ignored: Indices the arrangement named that the stack has no layer for.
        """
        try:
            if not run_result.watching():
                return
            width, height = layer_arrange.canvas_size(document, placed)
            moved = sum(
                1
                for before, after in zip(source, placed)
                if layer_arrange.placement(before) != layer_arrange.placement(after)
            )
            hidden = sum(1 for entry in placed if not bool(entry.get("visible", True)))
            lines = layer_arrange.rows(placed)
            bodies = [
                run_result.body(TABLE_BODY, chunk)
                for chunk in layer_arrange.chunks(
                    lines, run_result.MAX_BODY_CHARS, run_result.MAX_BODIES
                )
            ]
            listed = sum(len(part["text"].split("\n")) for part in bodies)
            run_result.publish(
                status=run_result.WARNING if ignored else run_result.OK,
                summary=cls.summary(len(placed), moved, listed, ignored),
                counts={"layers": len(placed), "moved": moved, "hidden": hidden},
                facts={
                    "canvas": f"{width}x{height}",
                    "listed": f"{listed} of {len(lines)}",
                },
                bodies=bodies,
            )
            drawn = min(listed, preview.MAX_FRAMES)
            frames = layer_arrange.thumbnails(placed[:drawn], width, height)
            if frames is not None:
                preview.publish_frames(frames, slot=THUMBNAIL_SLOT)
        except Exception as error:
            logger.debug("Layers Arrange published no report (%s)", error)

    @classmethod
    def summary(cls, total, moved, listed, ignored) -> str:
        """One line saying what the arrangement did.

        Args:
            total: How many layers the stack holds.
            moved: How many of them the arrangement changed.
            listed: How many the panel was given rows for.
            ignored: Indices the arrangement named that name no layer.

        Returns:
            The line, written for the person running the pack.
        """
        if ignored:
            return (
                f"{moved} of {total} layer(s) arranged, and {len(ignored)} entry(s) name no "
                f"layer, the first of them layer {ignored[0]} of a {total} layer stack"
            )
        if not moved:
            return f"{total} layer(s) passed through as they arrived"
        if listed < total:
            return f"{moved} of {total} layer(s) arranged, {listed} listed below"
        return f"{moved} of {total} layer(s) arranged"
