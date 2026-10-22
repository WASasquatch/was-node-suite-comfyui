"""Two images held side by side under a divider, for comparing them frame by frame."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.interface import preview
from ....modules.interface.compare_files import keep_ui

#: What each side's frames are written under in the temp folder.
PREFIX_A = "was.compare.image.a"
PREFIX_B = "was.compare.image.b"


class ImageCompare(io.ComfyNode):
    """Write two images so the node's own interface can draw one over the other."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageCompare",
            display_name="Image Compare (Advanced)",
            search_aliases=[
                "WASImageCompare", "Image Compare",
                "compare images",
                "before after",
                "a b compare",
                "difference",
                "slider",
                "split view",
            ],
            category="WAS Suite/Image/Analyze",
            description=(
                "Compare two images on the node, with a divider you drag across to reveal one "
                "under the other. A batch is compared pair by pair, one tab per pair, at the "
                "size the images were made."
            ),
            inputs=[
                io.Image.Input(
                    "image_a",
                    tooltip=(
                        "The image the divider reveals on the left; IMAGE. A batch is paired "
                        "with image_b frame by frame."
                    ),
                ),
                io.Image.Input(
                    "image_b",
                    tooltip=(
                        "The image the divider reveals on the right; IMAGE. Where the batches "
                        "are different lengths, the shorter one holds its last frame."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="image_a",
                    tooltip="image_a unchanged, so the node can sit in the middle of a chain.",
                ),
                io.Image.Output(
                    display_name="image_b",
                    tooltip="image_b unchanged.",
                ),
                io.Int.Output(
                    display_name="pairs",
                    tooltip=(
                        "How many pairs the comparison holds; INT, the longer of the two "
                        "batches."
                    ),
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, image_a, image_b) -> io.NodeOutput:
        """Write both inputs for the interface and pass them through.

        Raises:
            ValueError: Either input holds no image.
        """
        count_a = int(image_a.shape[0]) if getattr(image_a, "ndim", 0) >= 3 else 0
        count_b = int(image_b.shape[0]) if getattr(image_b, "ndim", 0) >= 3 else 0
        if not count_a or not count_b:
            empty = "image_a" if not count_a else "image_b"
            raise ValueError(
                f"Image Compare (Advanced) was given nothing on {empty}. Connect an image to "
                "both inputs."
            )

        # Each side is written with its own byte budget.
        sides = keep_ui({
            "a_images": preview.write_frames(image_a, PREFIX_A),
            "b_images": preview.write_frames(image_b, PREFIX_B),
        })
        return io.NodeOutput(image_a, image_b, max(count_a, count_b), ui=sides)
