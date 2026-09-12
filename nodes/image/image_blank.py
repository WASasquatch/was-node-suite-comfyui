"""Generate a solid colour image."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.convert.tensors import pil2tensor


class ImageBlank(io.ComfyNode):
    """Emit a batch of images filled with one colour."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Blank",
            display_name="Image Blank",
            search_aliases=["Image Blank", "solid colour", "blank canvas", "fill"],
            category="WAS Suite/Image/Generate",
            description=(
                "Make a new image filled with a single colour, for use as a background, a "
                "matte, or a base to composite onto. The colour is set from the wheel on the "
                "node or by typing the three levels. Both sides are rounded down to a "
                "multiple of divisible_by, which saves a sampler rounding the size itself, "
                "so 513 becomes 512 at the default of 8. Use 16, 32 "
                "or 64 for a model that asks for a coarser step, and 1 for a matte that has "
                "to line up with something else exactly. A side shorter than divisible_by is "
                "taken up to one whole step rather than down to nothing. batch_size repeats "
                "the fill, for matching a batch of frames."
            ),
            inputs=[
                io.Int.Input(
                    "width",
                    default=512,
                    min=8,
                    max=4096,
                    step=1,
                    tooltip=(
                        "Width in pixels, rounded down to a multiple of divisible_by: 500 "
                        "gives 496 at the default of 8, and 500 at a divisible_by of 1."
                    ),
                ),
                io.Int.Input(
                    "height",
                    default=512,
                    min=8,
                    max=4096,
                    step=1,
                    tooltip=(
                        "Height in pixels, rounded down to a multiple of divisible_by: 500 "
                        "gives 496 at the default of 8, and 500 at a divisible_by of 1."
                    ),
                ),
                io.Int.Input(
                    "red",
                    default=255,
                    min=0,
                    max=255,
                    step=1,
                    tooltip="Red level of the fill colour. 0 is none, 255 is full.",
                ),
                io.Int.Input(
                    "green",
                    default=255,
                    min=0,
                    max=255,
                    step=1,
                    tooltip="Green level of the fill colour. 0 is none, 255 is full.",
                ),
                io.Int.Input(
                    "blue",
                    default=255,
                    min=0,
                    max=255,
                    step=1,
                    tooltip=(
                        "Blue level of the fill colour. 0 is none, 255 is full. All three "
                        "at 255 gives white, all three at 0 gives black."
                    ),
                ),
                io.Int.Input(
                    "divisible_by",
                    default=8,
                    max=64,
                    min=1,
                    step=1,
                    tooltip=(
                        "Rounds width and height down to a multiple of this. 8 suits most "
                        "latent models; set it to 1 to get the exact canvas asked for."
                    ),
                ),
                io.Int.Input(
                    "batch_size",
                    default=1,
                    min=1,
                    max=4096,
                    step=1,
                    tooltip=(
                        "How many copies the batch holds. 1 = a single image; 16 = sixteen "
                        "identical fills, for matching a batch of frames a sampler or a "
                        "video node is working on."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    tooltip=(
                        "A batch of batch_size images, each filled edge to edge with the "
                        "chosen colour, at the requested size rounded down to a multiple of "
                        "divisible_by, with a side shorter than that taken up to one whole "
                        "step instead."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls, width, height, red, green, blue, divisible_by=8, batch_size=1
    ) -> io.NodeOutput:
        from PIL import Image

        # Floored at one whole step, since a side of zero pixels is not an image.
        width = max(divisible_by, (width // divisible_by) * divisible_by)
        height = max(divisible_by, (height // divisible_by) * divisible_by)

        blank = Image.new(mode="RGB", size=(width, height), color=(red, green, blue))

        frames = pil2tensor(blank)
        count = max(1, int(batch_size))
        return io.NodeOutput(frames if count == 1 else frames.repeat(count, 1, 1, 1))
