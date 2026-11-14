"""Put edited crops back into the picture they were cut from, through their regions alone."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.compat.types import EDIT_CROP
from ....modules.image import edit_crop
from ....modules.interface import preview, run_result

NODE = "Image Edit Restore"

logger = log.get_logger("nodes.image.edit")


def _one(value, default=None):
    """The single value a list input holds."""
    if isinstance(value, list):
        return value[0] if value else default
    return value


class ImageEditRestore(io.ComfyNode):
    """Paste each edited crop back through its own regions, leaving every other pixel as it was."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageEditRestore",
            display_name="Image Edit Restore",
            search_aliases=[
                "WASImageEditRestore",
                "Image Edit Restore",
                "crop and stitch",
                "inpaint stitch",
                "paste back",
                "regional edit",
            ],
            category="WAS Suite/Image/Edit",
            description=(
                "Put the crops an edit model changed back into the picture Image Edit Crop cut them "
                "from, each through its own regions with a soft edge and the edit's colour drift "
                "removed. Every pixel outside the regions stays exactly as it was."
            ),
            is_input_list=True,
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="The picture Image Edit Crop cut the crops from. A batch is read from its first picture.",
                ),
                io.Image.Input(
                    "edited",
                    tooltip="The edited crops, in the order Image Edit Crop handed them out, at any size.",
                ),
                EDIT_CROP.Input(
                    "edits",
                    tooltip="Image Edit Crop's edits output.",
                ),
                io.Mask.Input(
                    "edited_masks", optional=True,
                    tooltip=(
                        "What each edited crop holds where its regions were, as WeVisOne-VL Detect finds it: "
                        "an area lying over a region is pasted whole, one beside it is left out. Unwired = the "
                        "original regions alone."
                    ),
                ),
                io.Int.Input(
                    "grow", default=4, min=0, max=256,
                    tooltip="Pixels each region is grown by before it is pasted: 0 = its matte exactly, 4 = a little past it.",
                ),
                io.Int.Input(
                    "feather", default=6, min=0, max=256,
                    tooltip="Pixels each region's edge fades over: 0 = a hard edge, 6 = a soft seam, 24 = a wide blend.",
                ),
                io.Boolean.Input(
                    "correct_drift", default=True,
                    tooltip=(
                        "`true` = remove the colour and light shift the edit made, measured on the crop "
                        "outside the regions; `false` = paste the edit as it came."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="image",
                    tooltip="The picture with every edit pasted back; outside the regions it is the original exactly.",
                ),
                io.Mask.Output(
                    display_name="changed",
                    tooltip="How much of the edit each pixel took: 0 = untouched, 1 = fully replaced.",
                ),
            ],
        )

    @classmethod
    def execute(cls, image, edited, edits, grow, feather, correct_drift, edited_masks=None) -> io.NodeOutput:
        """Paste every edited crop back into the picture.

        Args:
            image: The picture, as a one-item list.
            edited: The edited crops, one per record.
            edits: Image Edit Crop's records.
            grow: Pixels each region is grown by, as a one-item list.
            feather: Pixels each region's edge fades over, as a one-item list.
            correct_drift: Remove the edit's colour drift, as a one-item list.
            edited_masks: Mattes of the edited crops, one per record, or ``None``.

        Returns:
            The picture with the edits pasted back, and the share of the edit each pixel took.

        Raises:
            ValueError: No picture, or the edited crops and the records do not pair up.
        """
        import torch

        source = _one(image)
        if source is None:
            raise ValueError(f"{NODE} was given no picture. Wire the picture Image Edit Crop cut the crops from.")
        crops = [crop for crop in (edits or []) if isinstance(crop, edit_crop.EditCrop)]
        pictures = list(edited or [])
        if not crops:
            raise ValueError(f"{NODE} was given no edits. Wire Image Edit Crop's edits output into edits.")
        if len(pictures) != len(crops):
            raise ValueError(
                f"{NODE} was given {len(pictures)} edited crop(s) for {len(crops)} edit(s). Wire the edit "
                "model's result for every crop Image Edit Crop handed out, in the same order."
            )
        picture = source[0]
        if (picture.shape[1], picture.shape[0]) != tuple(crops[0].source):
            raise ValueError(
                f"{NODE}: the picture is {picture.shape[1]}x{picture.shape[0]} but the crops were cut from a "
                f"{crops[0].source[0]}x{crops[0].source[1]} one. Wire the same picture Image Edit Crop was given."
            )
        afters = list(edited_masks or [])
        grow_px, feather_px = int(_one(grow, 4)), int(_one(feather, 6))
        correct = bool(_one(correct_drift, True))
        result = picture.float()
        changed = torch.zeros(picture.shape[:2])
        for at, (crop, edit) in enumerate(zip(crops, pictures)):
            after = None
            if at < len(afters) and afters[at] is not None:
                plane = afters[at]
                after = plane.reshape(-1, *plane.shape[-2:]).float().amax(dim=0)
            result, alpha = edit_crop.paste(result, edit[0], crop, after, grow_px, feather_px, correct)
            changed = torch.maximum(changed, alpha)
        answer = result.to(picture.dtype)[None]
        preview.publish(source[:1], slot="image")
        preview.publish(pictures[0][:1], slot="edited")
        preview.publish_output(answer, slot="image")
        share = float((changed > 0).float().mean())
        logger.info("%s pasted %d crop(s), changing %.2f%% of the picture", NODE, len(crops), share * 100)
        try:
            run_result.publish(
                status=run_result.OK,
                summary=f"{len(crops)} crop(s) pasted, {share * 100:.2f}% of the picture changed",
                counts={"crops": len(crops), "changed pixels": int((changed > 0).sum())},
                facts={"drift": "removed" if correct else "kept", "edge": f"grow {grow_px}, feather {feather_px}"},
            )
        except Exception as error:
            logger.debug("%s published no report (%s)", NODE, error)
        return io.NodeOutput(answer, changed[None])
