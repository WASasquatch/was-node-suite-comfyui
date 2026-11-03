"""Join two datasets into one, numbering a name the first already holds."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import DATASET
from ...modules.dataset import dataset as datasets
from ...modules.interface import run_result

logger = log.get_logger("nodes.dataset")

#: Display name, written into the messages this node raises.
NAME = "Dataset Merge"


class DatasetMerge(io.ComfyNode):
    """Append one dataset's items after another's."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASDatasetMerge",
            display_name="Dataset Merge",
            search_aliases=[
                "WASDatasetMerge", "Dataset Merge",
                "join datasets",
                "combine datasets",
                "append dataset",
                "training set",
            ],
            category="WAS Suite/Dataset",
            description=(
                "Join two datasets into one, the second's items after the first's, so sets "
                "loaded from several folders are checked, edited and saved together. A name the "
                "first already holds is numbered, as `cat_2`, so no item is lost."
            ),
            inputs=[
                DATASET.Input(
                    "dataset_a",
                    tooltip="The first dataset. Its items come first and keep their names.",
                ),
                DATASET.Input(
                    "dataset_b",
                    tooltip=(
                        "The second dataset. Its items follow; one whose name dataset_a already "
                        "holds, case ignored, is renamed `cat_2`, `cat_3` and so on."
                    ),
                ),
            ],
            outputs=[
                DATASET.Output(
                    display_name="dataset",
                    tooltip=(
                        "Every item of both, dataset_a's first. Save Dataset writes a renamed "
                        "item under its new name, as cat_2.png with cat_2.txt."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip="How many items the joined dataset holds.",
                ),
                io.Int.Output(
                    display_name="renamed",
                    tooltip="How many of dataset_b's items took a number. 0 = no name was shared.",
                ),
            ],
        )

    @classmethod
    def execute(cls, dataset_a, dataset_b) -> io.NodeOutput:
        """Join the two datasets and publish the counts and the renamed items.

        Raises:
            ValueError: Either input is not a dataset, or the two hold more items together than
                one dataset holds.
        """
        first = datasets.from_value(dataset_a, NAME)
        second = datasets.from_value(dataset_b, NAME)
        joined, renamed = datasets.merge(first, second)
        _publish(first, second, joined, renamed)
        return io.NodeOutput(joined, len(joined), renamed)


def _publish(first, second, joined, renamed: int) -> None:
    """Report the counts, the two sources and the first renamed items to the node's panel.

    Never raises, and never changes what the node returns.

    Args:
        first: dataset_a.
        second: dataset_b.
        joined: The merged dataset.
        renamed: How many of the second's items took a number.
    """
    try:
        if not run_result.watching():
            return
        moved = [
            {"text": item.name, "note": f"was {given.name}"}
            for given, item in zip(second.items, joined.items[len(first):])
            if item.name != given.name
        ]
        summary = f"joined {len(first)} and {len(second)} items into {len(joined)}"
        if renamed:
            summary += f", {renamed} renamed"
        run_result.publish(
            status=run_result.OK if len(joined) else run_result.WARNING,
            summary=summary if len(joined) else "both datasets are empty",
            counts={"dataset_a": len(first), "dataset_b": len(second), "count": len(joined), "renamed": renamed},
            facts={"dataset_a": first.source or "unnamed", "dataset_b": second.source or "unnamed"},
            items=moved[: run_result.MAX_ITEMS],
            items_total=len(moved),
        )
    except Exception as error:
        logger.debug("%s did not report the join (%s)", NAME, error)
