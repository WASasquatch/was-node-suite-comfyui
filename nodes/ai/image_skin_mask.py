"""Masking the bare skin of the people in a picture from a body-part network."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ...modules import log
from ...modules.image import skin
from ...modules.io import picker
from ...modules.model import sapiens2

REQUIRES = "preprocessors"

logger = log.get_logger("nodes.ai.image_skin_mask")

#: The pose model whose face points place the eye and brow cut-outs.
POSE_MODEL = "ViTPose Wholebody"

#: Shown in the model list when the folder holds no usable checkpoint.
NO_CHECKPOINT = "put a sapiens2 seg checkpoint in models/sapiens"


class ImageSkinMask(io.ComfyNode):
    """Mask the bare skin of every person in a picture with a body-part network."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        found = sapiens2.offered()
        return io.Schema(
            node_id="WASImageSkinMask",
            display_name="Image Skin Mask",
            search_aliases=[
                "WASImageSkinMask",
                "Image Skin Mask",
                "skin mask",
                "skin segmentation",
                "body parts",
                "body part segmentation",
                "sapiens",
                "face skin",
                "skin retouching",
            ],
            category="WAS Suite/Image/Masking",
            description=(
                "Mask the bare skin of every person in a picture, leaving out clothes, hair, "
                "background, eyes, brows and the inside of the mouth. A body-part network "
                "labels each person's face, torso and limbs, and their own skin colour settles "
                "what it takes for clothing. Good for retouching, grading or inpainting skin."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="The pictures to find skin in; IMAGE. Each picture of a batch is "
                    "masked on its own.",
                ),
                picker.path_input(io.Combo.Input(
                    "model",
                    options=found or [NO_CHECKPOINT],
                    default=sapiens2.DEFAULT if sapiens2.DEFAULT in found else None,
                    tooltip="Body-part network in models/sapiens: `sapiens2_1b_seg` = sure of "
                    "unusual poses, 3 GB of VRAM; `0.4b` and `0.8b` = lighter; `5b` = about "
                    "10 GB. Fetched on first use.",
                ), model=True),
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
                    tooltip="'remove inside' = cut out the teeth and tongue; 'remove lips' = "
                    "the lips as well; 'keep' = leave the whole mouth in the mask.",
                ),
                io.Float.Input(
                    "tolerance",
                    default=1.0,
                    min=0.5,
                    max=2.0,
                    step=0.05,
                    tooltip="How far a colour labelled as clothing may sit from each person's "
                    "own skin and still count as skin. 1.0 = as measured; 1.5 = takes in deep "
                    "shadow; 0.7 = stricter, for beige or tan clothing being caught.",
                ),
                io.Float.Input(
                    "person_threshold",
                    default=skin.PERSON_THRESHOLD,
                    min=0.05,
                    max=0.95,
                    step=0.05,
                    advanced=True,
                    tooltip="How sure the person detector has to be. 0.4 = most people in a "
                    "crowd; 0.7 = only clear, large figures. Each person found is read up "
                    "close; with nobody found the whole picture is read.",
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
    def validate_inputs(cls, model=None) -> bool | str:
        """Accept a listed checkpoint, or a wired one, which is checked when the node runs.

        Args:
            model: The stored combo value, or ``None`` where a link carries it.

        Returns:
            True, or the message naming what to pick.
        """
        if model is None or model in sapiens2.offered():
            return True
        if model == NO_CHECKPOINT:
            return (
                "no body-part checkpoint was found. Put a sapiens2_*_seg.safetensors file in "
                "ComfyUI/models/sapiens, or set features.network: true in config.yaml, then "
                "press R to refresh the list"
            )
        return (
            f"{model} is not in models/sapiens any more, or is not a Sapiens2 body-part "
            "checkpoint. Pick one from the model list"
        )

    @classmethod
    def execute(
        cls,
        image,
        model=sapiens2.DEFAULT,
        areas=skin.AREAS[0],
        eyes=skin.EYES[0],
        mouth=skin.MOUTHS[0],
        tolerance=1.0,
        person_threshold=skin.PERSON_THRESHOLD,
    ) -> io.NodeOutput:
        """Mask the skin of every picture in the batch.

        Raises:
            ValueError: An option names nothing the node offers, or the file is not a
                body-part checkpoint.
            DependencyError: transformers is not installed.
            ModelUnavailable: A model is not on disk and ``features.network`` is off.
        """
        import comfy.utils

        from ...modules.model import pose

        for name, value, offered in (
            ("areas", areas, skin.AREAS), ("eyes", eyes, skin.EYES), ("mouth", mouth, skin.MOUTHS)
        ):
            if value not in offered:
                raise ValueError(
                    f"Image Skin Mask has no {name} option {value!r}. "
                    f"Choose one of: {', '.join(offered)}."
                )
        tolerance = min(2.0, max(0.5, float(tolerance)))
        person_threshold = min(0.95, max(0.05, float(person_threshold)))

        model = picker.model_name(model, (sapiens2.FOLDER,)).replace("\\", "/")
        network = sapiens2.load(model)
        pair = pose.load(POSE_MODEL)
        progress = comfy.utils.ProgressBar(len(image))
        masks, people, faces = [], 0, 0
        for frame in image:
            found = skin.find_by_parts(
                frame, network, pair, areas, eyes, mouth, tolerance, person_threshold
            )
            masks.append(found.mask.cpu())
            people = max(people, found.people)
            faces = max(faces, found.faces)
            progress.update(1)
        result = torch.stack(masks).float().contiguous()
        summary = skin.publish(
            image, result, people, faces,
            {"model": model, "areas": areas, "eyes": eyes, "mouth": mouth},
        )
        logger.info("Image Skin Mask: %s", summary)
        return io.NodeOutput(result)
