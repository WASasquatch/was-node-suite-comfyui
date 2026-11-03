"""Open a dataset into its pictures, captions and names, as lists the other nodes run once per entry."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import DATASET, LIST


class DatasetToLists(io.ComfyNode):
    """Hand a dataset's pictures, captions and names on as lists in the dataset's order."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASDatasetToLists",
            display_name="Dataset to Lists",
            search_aliases=["WASDatasetToLists", "Dataset to Lists", "dataset captions", "dataset images"],
            category="WAS Suite/Dataset",
            description=(
                "Open a dataset into lists of its pictures, captions and names, in the same "
                "order, so any image or text node can work on them. Dataset Update puts the "
                "results back into the dataset."
            ),
            inputs=[
                DATASET.Input("dataset", tooltip="The dataset to open, such as from Load Dataset."),
                io.Boolean.Input(
                    "pictures",
                    default=True,
                    tooltip=(
                        "`on` = decode every picture onto the images output; `off` = captions and "
                        "names only, and images stops the nodes it feeds."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    is_output_list=True,
                    tooltip="One picture per entry, so a node wired here runs once per picture.",
                ),
                io.String.Output(
                    display_name="captions",
                    is_output_list=True,
                    tooltip="One caption per picture, empty where a picture has none.",
                ),
                io.String.Output(
                    display_name="names",
                    is_output_list=True,
                    tooltip=(
                        "Each picture's name below its folder without extension, such as "
                        "`set_b/cat`, for Dataset Update's names or Save Text File's filenames."
                    ),
                ),
                LIST.Output(display_name="caption_list", tooltip="The captions on one wire, as a LIST."),
                LIST.Output(display_name="name_list", tooltip="The names on one wire, as a LIST."),
                io.Int.Output(display_name="count", tooltip="How many pictures the dataset holds."),
            ],
        )

    @classmethod
    def execute(cls, dataset, pictures=True) -> io.NodeOutput:
        """Open the dataset.

        Raises:
            ValueError: The dataset input holds no dataset, the dataset is empty, or a picture
                cannot be read.
        """
        from ...modules.compat.lists import block_if_empty, require_values
        from ...modules.dataset import dataset as datasets

        data = datasets.from_value(dataset, "Dataset to Lists")
        require_values(
            list(data.items),
            "the dataset is empty, so Dataset to Lists has nothing to hand on. A filter "
            "upstream may have removed every picture",
        )
        texts = datasets.captions(data)
        named = datasets.names(data)
        images = [datasets.pixels(item) for item in data.items] if pictures else []
        images = block_if_empty(
            images,
            "Dataset to Lists has pictures off, so its images output is empty. Turn it on to "
            "decode the pictures",
        )
        return io.NodeOutput(images, texts, named, list(texts), list(named), len(data))
