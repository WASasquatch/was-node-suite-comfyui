"""How much of a mask is covered, as numbers a graph can branch on."""

from __future__ import annotations

from comfy_api.latest import io, ui

from ...modules.compat import limits
from ...modules.compat.sockets import require_input
from ...modules.compat.types import NUMBER
from ...modules.logic.switch_index import OUT_OF_RANGE
from ...modules.mask.statistics import DEFAULT_THRESHOLD, measure, summarise

#: The node's display name, spelled as its title bar spells it, for every message it raises.
NODE = "Mask Statistics"


class MaskStatistics(io.ComfyNode):
    """Measure what a mask covers and answer the figures as separate numbers."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASMaskStatistics",
            display_name="Mask Statistics",
            search_aliases=[
                "WASMaskStatistics",
                "Mask Statistics",
                "mask coverage",
                "mask area",
                "is mask empty",
                "empty mask",
                "measure mask",
                "mask pixels",
            ],
            category="WAS Suite/Image/Masking",
            description=(
                "Measure what a mask covers: the fraction and count of pixels above the "
                "threshold, the value range, and whether it found anything at all. Wire "
                "is_empty into a gate or a switch to skip an inpaint, a crop or a save when "
                "a segmenter matched nothing."
            ),
            inputs=[
                io.Mask.Input(
                    "mask",
                    tooltip=(
                        "The mask to measure, from CLIPSeg, SAM, a threshold or anything "
                        "else. It is read, never changed, so it can be tapped off a wire "
                        "on its way somewhere else. A batch is measured together unless "
                        "index picks one mask of it."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input(
                        "threshold",
                        default=DEFAULT_THRESHOLD,
                        min=0.0,
                        max=1.0,
                        step=0.01,
                    ),
                    [io.Float, NUMBER, io.Int],
                    tooltip=(
                        "Value a pixel must be above to count as covered. 0.5 = halfway, "
                        "which the pack's other mask operations use; 0.1 counts faint "
                        "feathering in; 1.0 counts nothing, since no mask value goes above "
                        "it. min, max and mean ignore it."
                    ),
                ),
                io.Combo.Input(
                    "scope",
                    options=["whole batch", "one mask"],
                    default="whole batch",
                    tooltip=(
                        "What to measure. `whole batch` answers one set of figures for every "
                        "mask together; `one mask` measures the one the index picks."
                    ),
                ),
                io.MultiType.Input(
                    io.Int.Input(
                        "index",
                        default=0,
                        min=-limits.max_resolution(),
                        max=limits.max_resolution(),
                        step=1,
                    ),
                    [io.Int, NUMBER, io.Float],
                    tooltip=(
                        "Which mask to measure, read only when scope is `one mask`. "
                        "Counts from 0, and negatives count from the end: -1 = last, -2 the "
                        "one before it. A decimal is truncated: 2.7 = 2."
                    ),
                ),
                io.Combo.Input(
                    "out_of_range",
                    options=list(OUT_OF_RANGE),
                    default="error",
                    tooltip=(
                        "Index outside 0..batch_size-1, which index -1 never reaches. With "
                        "3 masks and index 4: `wrap` = mask 1, `clamp` = mask 2, `error` "
                        "stops the prompt and names the batch size."
                    ),
                ),
            ],
            outputs=[
                io.Float.Output(
                    display_name="coverage",
                    tooltip=(
                        "Fraction of the pixels measured that are above the threshold, 0.0 "
                        "to 1.0. 0.25 = a quarter of the frame is masked. Multiply by 100 "
                        "for a percentage, or compare it against a minimum to reject a "
                        "mask that found next to nothing."
                    ),
                ),
                io.Int.Output(
                    display_name="covered_pixels",
                    tooltip=(
                        "How many pixels are above the threshold, counted exactly. The "
                        "figure to test where an area in pixels matters more than a "
                        "fraction of the frame, such as refusing a detection only a few "
                        "hundred pixels across."
                    ),
                ),
                io.Int.Output(
                    display_name="total_pixels",
                    tooltip=(
                        "Pixels measured: width times height for one mask, and that times "
                        "batch_size at index -1. coverage is covered_pixels divided by "
                        "this."
                    ),
                ),
                io.Float.Output(
                    display_name="min",
                    tooltip=(
                        "Smallest value measured, normally 0.0 to 1.0. Above 0.0 means no "
                        "pixel is fully outside the mask, which a blur or a lifted floor "
                        "causes and which makes a hard-edged paste bleed."
                    ),
                ),
                io.Float.Output(
                    display_name="max",
                    tooltip=(
                        "Largest value measured, normally 0.0 to 1.0. 1.0 means at least "
                        "one pixel is fully inside. Below the threshold means nothing was "
                        "found at all, whatever the mask looks like on a preview."
                    ),
                ),
                io.Float.Output(
                    display_name="mean",
                    tooltip=(
                        "Average of every value measured, before the threshold is applied. "
                        "A feathered mask reads well below its coverage and a hard-edged "
                        "one reads about the same, so the gap between the two says how "
                        "soft the edges are."
                    ),
                ),
                io.Boolean.Output(
                    display_name="is_empty",
                    tooltip=(
                        "true when no pixel is above the threshold. Wire it into Any Gate "
                        "or a switch so an inpaint, a crop or a save is skipped when a "
                        "detector came back with nothing instead of running on a black "
                        "mask."
                    ),
                ),
                io.Int.Output(
                    display_name="batch_size",
                    tooltip=(
                        "How many masks arrived on the wire, whatever index was set. Wire "
                        "it into a loop's iterations to walk the batch one mask at a time."
                    ),
                ),
                io.Int.Output(
                    display_name="width",
                    tooltip=(
                        "Mask width in pixels. Feed it to a crop, a paste or an Empty "
                        "Latent Image so the size follows the mask rather than being typed "
                        "twice."
                    ),
                ),
                io.Int.Output(
                    display_name="height",
                    tooltip=(
                        "Mask height in pixels. With width it gives the frame size a crop "
                        "or an Empty Latent Image needs."
                    ),
                ),
                io.String.Output(
                    display_name="summary",
                    tooltip=(
                        "Every figure on one line, as `index=all  batch_size=1  512x512  "
                        "threshold=0.500  coverage=25.00%  covered=65536/262144  ...`. For "
                        "a log, a console print, or burning into a frame with Image Draw "
                        "Text."
                    ),
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls, mask, threshold=DEFAULT_THRESHOLD, scope="whole batch", index=0,
        out_of_range="error",
    ) -> io.NodeOutput:
        """Measure the whole batch, or one mask of it.

        Args:
            mask: The MASK to read.
            threshold: Value a pixel must be above to count as covered.
            scope: Measure every mask together, or the one the index picks.
            index: Which mask to measure, counting from 0, negatives from the end.
            out_of_range: ``wrap``, ``clamp`` or ``error``.

        Returns:
            The six figures, the batch size, the frame size, whether the mask is empty,
            and the lot as one line of text.

        Raises:
            ValueError: Nothing is connected to mask, the batch holds nothing, or index is
                outside it and ``out_of_range`` is ``error``.
        """
        require_input(
            mask,
            NODE,
            "mask",
            "mask",
            "mask source such as CLIPSeg Masking, SAM Image Mask or Convert Image to Mask",
            "MASK",
        )
        stats = measure(
            mask, threshold, int(index), out_of_range, NODE, whole=scope == "whole batch"
        )
        summary = summarise(stats)
        return io.NodeOutput(
            stats.coverage,
            stats.covered_pixels,
            stats.total_pixels,
            stats.lowest,
            stats.highest,
            stats.mean,
            stats.is_empty,
            stats.batch_size,
            stats.width,
            stats.height,
            summary,
            ui=ui.PreviewText(summary),
        )
