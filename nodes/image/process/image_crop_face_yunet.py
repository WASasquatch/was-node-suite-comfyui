"""Find a face with YuNet and crop a square around it."""

from __future__ import annotations

import numpy as np
from comfy_api.latest import io

from ....modules import log
from ....modules.compat.sockets import require_input
from ....modules.compat.types import CROP_DATA, YUNET_MODEL
from ....modules.convert.tensors import image_planes, stack_images, tensor2pil
from ....modules.interface import preview, size_report
from ....modules.model import yunet

REQUIRES = "yunet"

logger = log.get_logger("nodes.image.process")

#: Shortest side the crop is enlarged to, so a small face still gives a workable image.
MIN_FACE_SIZE = 64

#: Size of the black frame returned when no face is found, matching Image Crop Face.
EMPTY_SIZE = 512


class ImageCropFaceYuNet(io.ComfyNode):
    """Crop a square around the face YuNet finds, with the window to paste it back."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageCropFaceYuNet",
            display_name="Image Crop Face (YuNet)",
            search_aliases=[
                "WASImageCropFaceYuNet",
                "Image Crop Face (YuNet)",
                "face detect",
                "crop face",
                "yunet",
            ],
            category="WAS Suite/Image/Process",
            description=(
                "Find a face with YuNet and crop a square around it, together with the "
                "crop window Image Paste Face needs to put it back. The detector ships with "
                "the pack and runs in torch on whatever device ComfyUI is using, so there "
                "is nothing to install. Set features.yunet to false to leave this node out."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The image to search for a face in. A batch is searched on its first "
                        "image and every image is then cut to that same window."
                    ),
                ),
                YUNET_MODEL.Input(
                    "yunet_model",
                    tooltip="The detector, from YuNet Model Loader.",
                ),
                io.Float.Input(
                    "crop_padding_factor",
                    default=0.25,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    tooltip=(
                        "How much room to leave around the detected face, as a fraction of "
                        "its size. 0.0 crops tight to the detection, 0.25 leaves a quarter "
                        "of the face size as margin, and 2.0 pulls back far enough to "
                        "include the shoulders."
                    ),
                ),
                io.Float.Input(
                    "confidence",
                    default=0.6,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How sure the detector must be that a region is a face. Lower finds "
                        "more faces and more false positives: 0.3 = catches small, blurred or "
                        "stylised faces; 0.9 = rejects false detections on a busy background."
                    ),
                ),
                io.Combo.Input(
                    "select",
                    options=["largest", "highest confidence", "leftmost", "rightmost"],
                    tooltip=(
                        "Which face to crop when several are found. `largest` = the most "
                        "pixels, usually the subject; `highest confidence` = the one the "
                        "detector is surest of, for a crowd; `leftmost`, `rightmost` = by "
                        "position."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    tooltip=(
                        "The square face crop, at least 64 pixels on a side. A black 512x512 "
                        "image when no face was found."
                    ),
                ),
                CROP_DATA.Output(
                    tooltip=(
                        "The crop window, for Image Paste Face to put the reworked face back "
                        "in the right place. One window covers a whole batch. False when no "
                        "face was found."
                    ),
                ),
                io.Int.Output(
                    display_name="faces_found",
                    tooltip=(
                        "How many faces the detector reported before one was chosen. 0 means "
                        "the crop is the black placeholder."
                    ),
                ),
                io.Float.Output(
                    display_name="confidence_score",
                    tooltip=(
                        "How sure the detector was about the face it cropped, 0.0 to 1.0. "
                        "Wire it into a condition node to route a doubtful detection "
                        "somewhere else. 0.0 when no face was found."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        yunet_model,
        crop_padding_factor=0.25,
        confidence=0.6,
        select="largest",
    ) -> io.NodeOutput:
        from PIL import Image

        require_input(
            yunet_model,
            "Image Crop Face (YuNet)",
            "yunet_model",
            "detector",
            "YuNet Model Loader",
        )

        frames = [np.array(tensor2pil(plane).convert("RGB")) for plane in image_planes(image)]
        faces = yunet.detect(yunet_model, frames[0], score_threshold=confidence)

        if not len(faces):
            logger.info("YuNet found no face above a confidence of %.2f", confidence)
            blank = [Image.new("RGB", (EMPTY_SIZE, EMPTY_SIZE), (0, 0, 0))] * len(frames)
            size_report.publish(
                image,
                (EMPTY_SIZE, EMPTY_SIZE),
                action="cropped",
                refused=(
                    f"no face reached a confidence of {confidence:.2f}, so a black "
                    f"{EMPTY_SIZE}x{EMPTY_SIZE} frame stands in"
                ),
            )
            empty = stack_images(blank)
            preview.publish_output(empty)
            return io.NodeOutput(empty, False, 0, 0.0)

        chosen = faces[cls.pick(faces, select)]
        window = cls.window(frames[0].shape, chosen, crop_padding_factor)
        crops = [cls.crop_window(frame, window) for frame in frames]
        size_report.publish(
            image, crops[0].size, action="cropped", facts={"faces": str(len(faces))}
        )
        cropped = stack_images(crops)
        preview.publish_output(cropped)
        return io.NodeOutput(
            cropped, (crops[0].size, window), len(faces), float(chosen[4])
        )

    @staticmethod
    def pick(faces: np.ndarray, select: str) -> int:
        """Index of the face ``select`` names.

        Args:
            faces: ``(x, y, width, height, score)`` rows.
            select: ``largest``, ``highest confidence``, ``leftmost`` or ``rightmost``.

        Returns:
            The row to crop. Detections arrive strongest first, so an unknown mode takes
            the first, which is the highest confidence.
        """
        if select == "largest":
            return int(np.argmax(faces[:, 2] * faces[:, 3]))
        if select == "leftmost":
            return int(np.argmin(faces[:, 0]))
        if select == "rightmost":
            return int(np.argmax(faces[:, 0] + faces[:, 2]))
        return int(np.argmax(faces[:, 4]))

    @staticmethod
    def window(shape, face, padding: float) -> tuple[int, int, int, int]:
        """The padded square to cut around one detection.

        Args:
            shape: ``(height, width, channels)`` of the source image.
            face: One ``(x, y, width, height, score)`` row.
            padding: Margin to leave, as a fraction of the face's size.

        Returns:
            ``(left, top, right, bottom)``, clamped to the image and square wherever the
            image is large enough to allow it.
        """
        height, width = shape[0], shape[1]
        centre_x = int(face[0] + face[2] / 2)
        centre_y = int(face[1] + face[3] / 2)
        half = int(max(face[2], face[3]) * (1.0 + padding) / 2)

        # Clamp the half-width to what the image can give on every side, so the window
        # stays square instead of being cut short on one edge.
        half = max(1, min(half, centre_x, centre_y, width - centre_x, height - centre_y))
        return (centre_x - half, centre_y - half, centre_x + half, centre_y + half)

    @staticmethod
    def crop_window(img: np.ndarray, window):
        """Cut one square out of one image and enlarge it if it is small.

        Args:
            img: Source pixels as an ``RGB`` array shaped ``(height, width, 3)``.
            window: ``(left, top, right, bottom)`` from :meth:`window`.

        Returns:
            An ``RGB`` image of the crop, at least :data:`MIN_FACE_SIZE` on a side.
        """
        from PIL import Image

        left, top, right, bottom = window
        crop = Image.fromarray(img[top:bottom, left:right, :]).convert("RGB")
        if min(crop.size) < MIN_FACE_SIZE:
            crop = crop.resize((MIN_FACE_SIZE, MIN_FACE_SIZE), Image.LANCZOS)
        return crop
