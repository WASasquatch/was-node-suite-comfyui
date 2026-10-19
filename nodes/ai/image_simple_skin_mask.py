"""Masking the bare skin of the people in a picture from clothing labels and skin colour."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ...modules import log
from ...modules.image import skin

REQUIRES = "preprocessors"

logger = log.get_logger("nodes.ai.image_simple_skin_mask")

#: The pose model whose face points place the cut-outs.
POSE_MODEL = "ViTPose Wholebody"


class ImageSimpleSkinMask(io.ComfyNode):
    """Mask the bare skin of every person in a picture with a small model and skin colour."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageSimpleSkinMask",
            display_name="Image Simple Skin Mask",
            search_aliases=[
                "WASImageSimpleSkinMask",
                "Image Simple Skin Mask",
                "skin mask",
                "simple skin mask",
                "skin segmentation",
                "face skin",
                "skin retouching",
            ],
            category="WAS Suite/Image/Masking",
            description=(
                "Mask the bare skin of every person in a picture, leaving out clothes, hair, "
                "background, eyes, brows and the inside of the mouth. A small clothing model and "
                "each person's own skin colour decide it, so it is quick and light. Image Skin "
                "Mask reads each body part instead and holds bare torsos better."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="The pictures to find skin in; IMAGE. Each picture of a batch is "
                    "masked on its own.",
                ),
                io.Combo.Input(
                    "areas",
                    options=list(skin.AREAS),
                    default=skin.AREAS[0],
                    tooltip="Which skin the mask keeps. 'all skin' = face and body; 'face' = "
                    "from the jaw to the forehead; 'body' = everything below the jaw.",
                ),
                io.Combo.Input(
                    "eyes",
                    options=list(skin.EYES),
                    default=skin.EYES[0],
                    tooltip="'remove with brows' = cut out the eyes and the eyebrows; 'remove' "
                    "= the eyes alone; 'keep' = leave both in the mask.",
                ),
                io.Combo.Input(
                    "mouth",
                    options=list(skin.MOUTHS),
                    default=skin.MOUTHS[0],
                    tooltip="'remove inside' = cut out the teeth and tongue between the lips; "
                    "'remove lips' = the lips as well; 'keep' = leave the mouth in the mask.",
                ),
                io.Float.Input(
                    "tolerance",
                    default=1.0,
                    min=0.5,
                    max=2.0,
                    step=0.05,
                    tooltip="How far a colour may sit from each person's own skin and still "
                    "count. 1.0 = as measured; 1.5 = takes in deep shadow and flushed skin; "
                    "0.7 = stricter, for beige or tan clothing being caught.",
                ),
                io.Float.Input(
                    "person_threshold",
                    default=skin.PERSON_THRESHOLD,
                    min=0.05,
                    max=0.95,
                    step=0.05,
                    advanced=True,
                    tooltip="How sure the person detector has to be. 0.4 = most people in a "
                    "crowd; 0.7 = only clear, large figures. Skin outside every person found "
                    "is left out.",
                ),
            ],
            outputs=[
                io.Mask.Output(
                    display_name="mask",
                    tooltip="The skin; MASK at the picture's size with a soft edge. 1 = skin, "
                    "0 = everything else.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        areas=skin.AREAS[0],
        eyes=skin.EYES[0],
        mouth=skin.MOUTHS[0],
        tolerance=1.0,
        person_threshold=skin.PERSON_THRESHOLD,
    ) -> io.NodeOutput:
        """Mask the skin of every picture in the batch.

        Raises:
            ValueError: An option names nothing the node offers.
            DependencyError: transformers is not installed.
            ModelUnavailable: A model is not on disk and ``features.network`` is off.
        """
        import comfy.utils

        from ...modules.model import body_parts, pose

        for name, value, offered in (
            ("areas", areas, skin.AREAS), ("eyes", eyes, skin.EYES), ("mouth", mouth, skin.MOUTHS)
        ):
            if value not in offered:
                raise ValueError(
                    f"Image Simple Skin Mask has no {name} option {value!r}. "
                    f"Choose one of: {', '.join(offered)}."
                )
        tolerance = min(2.0, max(0.5, float(tolerance)))
        person_threshold = min(0.95, max(0.05, float(person_threshold)))

        parts = body_parts.load()
        pair = pose.load(POSE_MODEL)
        progress = comfy.utils.ProgressBar(len(image))
        masks, people, faces = [], 0, 0
        for frame in image:
            found = skin.find_by_colour(
                frame, parts, pair, areas, eyes, mouth, tolerance, person_threshold
            )
            masks.append(found.mask.cpu())
            people = max(people, found.people)
            faces = max(faces, found.faces)
            progress.update(1)
        result = torch.stack(masks).float().contiguous()
        summary = skin.publish(
            image, result, people, faces, {"areas": areas, "eyes": eyes, "mouth": mouth}
        )
        logger.info("Image Simple Skin Mask: %s", summary)
        return io.NodeOutput(result)
