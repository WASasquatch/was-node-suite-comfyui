"""Put edited captions or pictures back into a dataset, as a new dataset."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import DATASET, LIST

logger = log.get_logger("nodes.dataset")

#: Sample rows the report carries.
SAMPLE_ROWS = 8

#: Characters of a caption a sample row shows.
SNIPPET_CHARS = 90


def flat_texts(value) -> list[str] | None:
    """Every text out of a STRING list or LIST input, a missing one kept as ``""``, or None."""
    if value is None:
        return None
    found, arrived = [], False
    for item in value if isinstance(value, list) else [value]:
        if item is None:
            found.append("")
            continue
        arrived = True
        for entry in item if isinstance(item, (list, tuple)) else [item]:
            found.append("" if entry is None else str(entry))
    return found if arrived else None


class DatasetUpdate(io.ComfyNode):
    """Answer a copy of a dataset carrying new captions, new pictures, or both."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASDatasetUpdate",
            display_name="Dataset Update",
            search_aliases=["WASDatasetUpdate", "Dataset Update", "set captions", "replace captions"],
            category="WAS Suite/Dataset",
            description=(
                "Put edited captions or pictures back into a dataset, such as captions from Text "
                "Replace Words or pictures from a resize, and hand on the updated copy. The "
                "dataset wired in is left as it was, so other nodes reading it are unaffected."
            ),
            is_input_list=True,
            is_output_node=True,
            inputs=[
                DATASET.Input("dataset", tooltip="The dataset to update, such as from Load Dataset."),
                io.MultiType.Input(
                    "captions",
                    [io.String, LIST],
                    optional=True,
                    tooltip=(
                        "New captions, one per picture in the dataset's order, or one per name "
                        "when names is wired. A blank caption leaves its picture uncaptioned."
                    ),
                ),
                io.Image.Input(
                    "images",
                    optional=True,
                    tooltip="New pictures, one per picture in the dataset's order, or one per name.",
                ),
                io.MultiType.Input(
                    "names",
                    [io.String, LIST],
                    optional=True,
                    tooltip=(
                        "The names the captions and images belong to, such as Dataset to Lists' "
                        "names after a filter. Unwired, they pair by position."
                    ),
                ),
            ],
            outputs=[
                DATASET.Output(display_name="dataset", tooltip="The updated copy of the dataset."),
                io.Int.Output(
                    display_name="changed",
                    tooltip="How many pictures took a new caption or a new picture. 0 = nothing changed.",
                ),
            ],
        )

    @classmethod
    def execute(cls, dataset, captions=None, images=None, names=None) -> io.NodeOutput:
        """Copy the dataset with the new captions and pictures.

        Raises:
            ValueError: No dataset arrived, or the captions, pictures or names do not line up
                with the dataset.
        """
        from ...modules.dataset import dataset as datasets

        data = datasets.from_value(dataset, "Dataset Update")
        texts = flat_texts(captions)
        named = flat_texts(names)
        named = [name for name in named if name] if named else None
        updated, caption_changes, picture_changes = data, 0, 0
        if texts is not None:
            updated, caption_changes = datasets.with_captions(updated, texts, by_name=named)
        frames = [image for image in (images or []) if image is not None]
        if frames:
            updated, picture_changes = datasets.with_images(updated, frames, by_name=named)
        changed = {index for index, (old, new) in enumerate(zip(data.items, updated.items)) if old is not new}
        cls._publish(updated, caption_changes, picture_changes, changed)
        return io.NodeOutput(updated, len(changed))

    @staticmethod
    def _publish(updated, caption_changes: int, picture_changes: int, changed: set) -> None:
        """Report what changed, for the panel on this node. Never raises."""
        try:
            from ...modules.interface import run_result

            rows = []
            for index in sorted(changed)[:SAMPLE_ROWS]:
                item = updated.items[index]
                text = " ".join((item.caption or "(no caption)").split())
                if len(text) > SNIPPET_CHARS:
                    text = text[: SNIPPET_CHARS - 3] + "..."
                rows.append({"text": text, "note": item.name})
            run_result.publish(
                status=run_result.OK if changed else run_result.WARNING,
                summary=(
                    f"{len(changed)} of {len(updated)} pictures updated"
                    if changed else f"nothing changed in {len(updated)} pictures"
                ),
                counts={
                    "pictures": len(updated), "updated": len(changed),
                    "captions": caption_changes, "images": picture_changes,
                },
                items=rows,
                items_total=len(changed),
            )
        except Exception as error:
            logger.debug("Dataset Update published no report (%s)", error)
