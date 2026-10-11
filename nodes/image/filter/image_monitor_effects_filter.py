"""Analogue and digital display artefacts."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.convert.tensors import filtered_planes


class ImageMonitorEffectsFilter(io.ComfyNode):
    """Tear, shear and interfere with an image the way a failing display does."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Monitor Effects Filter",
            display_name="Image Monitor Effects Filter",
            search_aliases=[
                "Image Monitor Effects Filter",
                "glitch",
                "vhs",
                "scan lines",
                "crt",
                "signal noise",
            ],
            category="WAS Suite/Image/Filter",
            description=(
                "Break an image up the way a bad screen or a worn tape does: torn rows, "
                "scan lines and interference patterns. Each run draws new random values, so "
                "the damage is different every time."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The image to damage. A batch is handled one image at a time, and each "
                        "image draws its own random damage."
                    ),
                ),
                io.Combo.Input(
                    "mode",
                    options=["Digital Distortion", "Signal Distortion", "TV Distortion"],
                    tooltip=(
                        "`Signal Distortion` = each row torn sideways by a random amount; "
                        "`Digital Distortion` = columns sheared along a wave, shuffled scan "
                        "lines punched through; `TV Distortion` = torn rows, crossed "
                        "interference and noise over a desaturated copy, a worn-videotape look."
                    ),
                ),
                io.Int.Input(
                    "amplitude",
                    default=5,
                    min=1,
                    max=255,
                    step=1,
                    tooltip=(
                        "Pixels of displacement: 1 = barely visible; 50 = severe. `TV "
                        "Distortion` divides the image height by it instead, so larger values "
                        "give smaller tears, and a value above the height fails."
                    ),
                ),
                io.Int.Input(
                    "offset",
                    default=10,
                    min=1,
                    max=255,
                    step=1,
                    tooltip=(
                        "Spacing of the scan lines, in rows: 10 replaces every tenth row, 1 "
                        "replaces all of them. Only Digital Distortion uses this; the other two "
                        "modes ignore it."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(display_name="image", tooltip="The distorted image."),
            ],
        )

    @classmethod
    def execute(cls, image, mode, amplitude, offset) -> io.NodeOutput:
        from ....modules.image.distortion import digital_distortion, signal_distortion, tv_vhs_distortion

        def damage(distorted):
            if mode == 'Digital Distortion':
                return digital_distortion(distorted, amplitude, offset)
            if mode == 'Signal Distortion':
                return signal_distortion(distorted, amplitude)
            if mode == 'TV Distortion':
                return tv_vhs_distortion(distorted, amplitude)
            return distorted

        return io.NodeOutput(filtered_planes(image, damage))
