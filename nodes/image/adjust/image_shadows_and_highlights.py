"""Independent shadow and highlight adjustment."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.convert.tensors import image_planes, pil2tensor, tensor2pil
from ....modules.image import dynamic


class ImageShadowsAndHighlights(io.ComfyNode):
    """Brighten or darken the dark and bright ends of an image separately."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Shadows and Highlights",
            display_name="Image Shadows and Highlights",
            search_aliases=[
                "Image Shadows and Highlights",
                "shadows",
                "highlights",
                "dodge and burn",
                "recover detail",
            ],
            category="WAS Suite/Image/Adjustment",
            description=(
                "Lift or crush the dark parts of an image and the bright parts "
                "independently, the way a photo editor's shadows-and-highlights control "
                "does. The two regions it worked on come out as maps on the second and "
                "third outputs."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="The image to adjust. A batch is handled one image at a time.",
                ),
                io.Float.Input(
                    "shadow_threshold",
                    default=75,
                    min=0.0,
                    max=255.0,
                    step=0.1,
                    tooltip=(
                        "How dark a pixel has to be, on a 0-255 brightness scale, to count as "
                        "shadow. 75 takes in the darker quarter of the range; raising it to 150 "
                        "pulls in the midtones as well, and 0 selects nothing."
                    ),
                ),
                io.Float.Input(
                    "shadow_factor",
                    default=1.5,
                    min=-12.0,
                    max=12.0,
                    step=0.1,
                    tooltip=(
                        "What the shadow area is multiplied by. 1.0 leaves it alone, 1.5 lifts "
                        "it by half again to reveal detail, 0.5 darkens it further, and 0 makes "
                        "it solid black."
                    ),
                ),
                io.Float.Input(
                    "shadow_smoothing",
                    default=0.25,
                    min=-255.0,
                    max=255.0,
                    step=0.1,
                    tooltip=(
                        "Blur radius in pixels applied to the shadow selection, which feathers "
                        "the edge of the adjustment so it does not show a hard outline. The "
                        "blur runs twice, so the softening is wider than the number suggests: "
                        "0.25 is a hairline, 8 is a broad fade."
                    ),
                ),
                io.Float.Input(
                    "highlight_threshold",
                    default=175,
                    min=0.0,
                    max=255.0,
                    step=0.1,
                    tooltip=(
                        "How bright a pixel has to be, on a 0-255 brightness scale, to count as "
                        "highlight. 175 takes in the brighter third of the range; 255 selects "
                        "nothing."
                    ),
                ),
                io.Float.Input(
                    "highlight_factor",
                    default=0.5,
                    min=-12.0,
                    max=12.0,
                    step=0.1,
                    tooltip=(
                        "What the highlight area is multiplied by. 1.0 leaves it alone, 0.5 "
                        "halves it to pull back blown-out brights, and values above 1.0 push "
                        "the highlights further towards white."
                    ),
                ),
                io.Float.Input(
                    "highlight_smoothing",
                    default=0.25,
                    min=-255.0,
                    max=255.0,
                    step=0.1,
                    tooltip=(
                        "Blur radius in pixels applied to the highlight selection, feathering "
                        "the edge of that adjustment. As with shadow_smoothing the blur runs "
                        "twice, so the effective fade is wider than the radius."
                    ),
                ),
                io.Float.Input(
                    "simplify_isolation",
                    default=0,
                    min=-255.0,
                    max=255.0,
                    step=0.1,
                    tooltip=(
                        "Blur radius in pixels applied to the brightness reading before either "
                        "region is cut, which merges scattered specks into solid areas. 0 keeps "
                        "the selections pixel-exact; 4 or more gives broader, smoother regions."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="image",
                    tooltip="The adjusted image, with the original colours restored over it.",
                ),
                io.Image.Output(
                    display_name="shadow_map",
                    tooltip=(
                        "Greyscale map of the area treated as shadow: white where the "
                        "adjustment applied at full strength, black where it did not."
                    ),
                ),
                io.Image.Output(
                    display_name="highlight_map",
                    tooltip=(
                        "Greyscale map of the area treated as highlight, on the same "
                        "white-is-adjusted reading as shadow_map."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        shadow_threshold,
        shadow_factor,
        shadow_smoothing,
        highlight_threshold,
        highlight_factor,
        highlight_smoothing,
        simplify_isolation,
    ) -> io.NodeOutput:
        from ....modules.image import scratch
        from ....modules.image.filters import shadows_and_highlights

        folded = dynamic.fold(image)
        planes = image_planes(folded.images)
        batches = None
        for index, plane in enumerate(planes):
            result, shadows, highlights = shadows_and_highlights(
                tensor2pil(plane),
                shadow_threshold,
                highlight_threshold,
                shadow_factor,
                highlight_factor,
                shadow_smoothing,
                highlight_smoothing,
                simplify_isolation,
            )
            frames = (pil2tensor(result), pil2tensor(shadows), pil2tensor(highlights))
            if batches is None:
                batches = tuple(
                    scratch.allocate(
                        (len(planes),) + tuple(frame.shape[1:]),
                        node="Image Shadows and Highlights",
                        advice="Passing fewer or smaller frames also fits it.",
                    )
                    for frame in frames
                )
            batches[0][index] = dynamic.unfold(frames[0][0], folded)
            for batch, frame in zip(batches[1:], frames[1:]):
                batch[index] = frame[0]

        if batches is None:
            raise ValueError(
                "Image Shadows and Highlights was handed an empty image batch. Wire in at "
                "least one image."
            )
        return io.NodeOutput(*batches)
