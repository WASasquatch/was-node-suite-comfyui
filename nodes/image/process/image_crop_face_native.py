"""Find a face with a packed classifier cascade and crop a square around it."""

from __future__ import annotations

import numpy as np
from comfy_api.latest import io

from ....modules import log
from ....modules.compat.types import CROP_DATA
from ....modules.convert.tensors import image_planes, stack_images, tensor2pil
from ....modules.image import cascade as cascades
from ....modules.image import face
from ....modules.interface import preview, size_report

logger = log.get_logger("nodes.image.process")


class ImageCropFaceNative(io.ComfyNode):
    """Detect a face with a packed cascade and return a square crop of it."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageCropFaceNative",
            display_name="Image Crop Face",
            search_aliases=[
                'WASImageCropFaceNative',
                "Image Crop Face",
                "Image Crop Face Native",
                "face detect",
                "crop face",
                "cascade",
            ],
            category="WAS Suite/Image/Process",
            description=(
                "Find a face in the image and crop a square around it, together with the "
                "crop window Image Paste Face needs to put it back. Where several faces are "
                "found, the largest is cropped. Needs nothing extra installed."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The image to search for a face in. A batch is searched on its first "
                        "image and every image is then cut to that same window."
                    ),
                ),
                io.Float.Input(
                    "crop_padding_factor",
                    default=0.25,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    tooltip=(
                        "How much room to leave around the detected face, as a fraction of its "
                        "size. 0.0 crops tight to the detection, 0.25 leaves a quarter of the "
                        "face size as margin, and 2.0 pulls back far enough to include the "
                        "shoulders."
                    ),
                ),
                io.Combo.Input(
                    "cascade",
                    options=list(face.CASCADES),
                    tooltip=(
                        "Which classifier to try first; if it finds nothing, the others follow "
                        "in menu order. 'lbpcascade_animeface' = drawn faces; 'frontalface' = "
                        "photographs, 'default' fastest, 'alt2' and 'alt_tree' stricter; "
                        "'profileface' = turned heads; 'upperbody' = head and shoulders; "
                        "'eye' = one eye, never a fallback."
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
                        "The crop window, for Image Paste Face to put the reworked face back in "
                        "the right place. One window covers a whole batch. False when no face "
                        "was found."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, image, cascade=None, crop_padding_factor=0.25) -> io.NodeOutput:
        """Cut a square around the face the chosen classifier finds.

        Raises:
            FileNotFoundError: A classifier file is missing from the installation.
        """
        from PIL import Image

        frames = [np.array(tensor2pil(plane).convert("RGB")) for plane in image_planes(image)]

        # One window at one size covers the whole batch, which suits variations of a single
        # framing; unrelated pictures are all cut to the first one's framing.
        box = cls.detect_face(frames[0], cascade, crop_padding_factor)
        if box is None:
            size_report.publish(
                image,
                (face.EMPTY_SIZE, face.EMPTY_SIZE),
                action="cropped",
                refused=(
                    "no face was found, so a black "
                    f"{face.EMPTY_SIZE}x{face.EMPTY_SIZE} frame stands in"
                ),
            )
            blank = stack_images(
                [Image.new("RGB", (face.EMPTY_SIZE, face.EMPTY_SIZE), (0, 0, 0))] * len(frames)
            )
            preview.publish_output(blank)
            return io.NodeOutput(blank, False)

        faces = [face.crop_window(frame, box) for frame in frames]
        cropped = stack_images(faces)
        size_report.publish(image, faces[0].size, action="cropped")
        preview.publish_output(cropped)
        return io.NodeOutput(cropped, (faces[0].size, box))

    @classmethod
    def detect_face(cls, img, cascade_name=None, padding=0.25):
        """Find a face in torch and work out the padded square to cut around it.

        Args:
            img: Source pixels as an ``RGB`` array shaped ``(height, width, 3)``.
            cascade_name: File name of the classifier to try first, one of
                ``face.CASCADES``. Anything else leaves the default order in place.
            padding: Margin around the detection as a fraction of the face size.

        Returns:
            ``(left, top, right, bottom)``, or None when no classifier found a face.

        Raises:
            FileNotFoundError: A classifier file is missing from the installation.
            ValueError: ``cascade_name`` is not an asset bundled with the pack.
        """
        found, faces = cascades.detect_first(
            img, face.try_order(cascade_name), min_neighbors=5
        )
        if not faces:
            logger.warning("no faces found in the image!")
            return None
        # The detector lists its hits in no dependable order, so the largest is taken rather
        # than the first: two runs on one picture otherwise crop different things.
        chosen = face.largest(faces)
        logger.info(
            "face found with: %s, %d detection(s), cropping the largest at %dx%d",
            found, len(faces), int(chosen[2]), int(chosen[3]),
        )
        return face.window(img, chosen, padding)
