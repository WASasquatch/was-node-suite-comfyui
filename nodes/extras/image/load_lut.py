"""Loading a colour lookup table from a file, a built-in look, or custom settings."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat.types import LUT
from ....modules.image import lut as tables
from ....modules.interface import lut_report

REQUIRES = "extras"

#: Menu entry that builds a table from the custom_* widgets instead of loading one.
CUSTOM = "Custom"

#: Prefix the menu puts in front of a file name, so a file cannot be mistaken for a preset.
FILE_PREFIX = "LUT: "


def lut_choices() -> list[str]:
    """The look menu: the custom entry, the built-in presets, then every cube file found.

    Returns:
        Menu entries in display order.
    """
    names = [CUSTOM]
    names += [name for name, _preset in tables.BUILTIN_PRESETS]
    # ``define_schema`` runs again for every ``/object_info`` request, so an unmemoized
    # scan here would be paid on every browser refresh.
    names += [f"{FILE_PREFIX}{path.name}" for path in tables.cube_files()]
    return names


class LoadLUT(io.ComfyNode):
    """Produce a colour lookup table from a file, a named look, or grading settings."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASNSLoadLUT",
            display_name="Load LUT",
            search_aliases=[
                "WASNSLoadLUT", "WAS Load LUT", "lut", "cube", "grade", "color grading", "look",
            ],
            category="WAS Suite/Image/LUT",
            description=(
                "Produce a colour lookup table to grade images with. Pick a .cube file from "
                "a models/LUT directory, one of the built-in looks, or 'Custom' to build a "
                "table from the exposure, contrast, saturation and white-balance controls "
                "below. Feed the result to Apply LUT, or to LUT Blender to mix two looks."
            ),
            inputs=[
                io.Combo.Input(
                    "look", options=lut_choices(),
                    tooltip=(
                        "Which table to produce. 'Custom' builds one from the controls below "
                        "and ignores any file; the named looks are built in and need no "
                        "file; entries starting 'LUT: ' are .cube files found in a models/LUT "
                        "directory or in the pack's own luts directory."
                    ),
                ),
                io.Int.Input(
                    "builtin_size", default=33, min=17, max=65, step=2,
                    tooltip=(
                        "Edge length of the cube built for a named look or for 'Custom', in "
                        "samples. 33 is the industry-standard size and is plenty; 65 is "
                        "finer and eight times the memory. Ignored when a .cube file is "
                        "chosen, since the file sets its own size."
                    ),
                ),
                io.Float.Input(
                    "custom_ev", default=0.0, min=-4.0, max=4.0, step=0.01,
                    tooltip=(
                        "Exposure in photographic stops, for the 'Custom' look. +1.0 doubles "
                        "the brightness, -1.0 halves it, 0.0 changes nothing."
                    ),
                ),
                io.Float.Input(
                    "custom_contrast", default=1.0, min=0.1, max=3.0, step=0.01,
                    tooltip=(
                        "Contrast for the 'Custom' look, around mid grey. Above 1.0 deepens "
                        "shadows and brightens highlights, below 1.0 flattens the picture "
                        "towards grey, 1.0 changes nothing."
                    ),
                ),
                io.Float.Input(
                    "custom_saturation", default=1.0, min=0.0, max=3.0, step=0.01,
                    tooltip=(
                        "Colour intensity for the 'Custom' look. 0.0 gives black and white, "
                        "1.0 changes nothing, 2.0 doubles the distance of every colour from "
                        "grey."
                    ),
                ),
                io.Float.Input(
                    "custom_vibrance", default=0.0, min=-1.0, max=1.0, step=0.01,
                    tooltip=(
                        "Colour intensity for the 'Custom' look, weighted towards the muted "
                        "colours. Positive lifts pale colour without pushing already-strong "
                        "colour further, which is the gentler way to add life to skin tones; "
                        "0.0 changes nothing."
                    ),
                ),
                io.Float.Input(
                    "custom_gamma", default=1.0, min=0.1, max=3.0, step=0.01,
                    tooltip=(
                        "Midtone brightness for the 'Custom' look, leaving black and white "
                        "where they are. Above 1.0 opens up shadow detail, below 1.0 deepens "
                        "it, 1.0 changes nothing."
                    ),
                ),
                io.Float.Input(
                    "custom_temperature", default=0.0, min=-1.0, max=1.0, step=0.01,
                    tooltip=(
                        "Warmth for the 'Custom' look. Positive shifts towards orange, as "
                        "though shot under tungsten light; negative shifts towards blue, as "
                        "though shot in shade; 0.0 changes nothing."
                    ),
                ),
                io.Float.Input(
                    "custom_tint", default=0.0, min=-1.0, max=1.0, step=0.01,
                    tooltip=(
                        "Green-magenta balance for the 'Custom' look, the second half of "
                        "white balance. Positive adds green, negative adds magenta, which is "
                        "what corrects a fluorescent cast. 0.0 changes nothing."
                    ),
                ),
                io.Float.Input(
                    "custom_red_balance", default=0.0, min=-1.0, max=1.0, step=0.01,
                    tooltip=(
                        "Red channel gain for the 'Custom' look, applied on its own. +0.1 "
                        "raises red by a tenth, -0.1 lowers it, 0.0 changes nothing."
                    ),
                ),
                io.Float.Input(
                    "custom_green_balance", default=0.0, min=-1.0, max=1.0, step=0.01,
                    tooltip=(
                        "Green channel gain for the 'Custom' look, applied on its own. Use "
                        "the three balance controls together to match a reference render "
                        "channel by channel. 0.0 changes nothing."
                    ),
                ),
                io.Float.Input(
                    "custom_blue_balance", default=0.0, min=-1.0, max=1.0, step=0.01,
                    tooltip=(
                        "Blue channel gain for the 'Custom' look, applied on its own. +0.1 "
                        "raises blue by a tenth, -0.1 lowers it, 0.0 changes nothing."
                    ),
                ),
            ],
            outputs=[
                LUT.Output(
                    display_name="lut",
                    tooltip=(
                        "The colour lookup table, for Apply LUT, LUT Blender or Save LUT."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        look,
        builtin_size,
        custom_ev,
        custom_contrast,
        custom_saturation,
        custom_vibrance,
        custom_gamma,
        custom_temperature,
        custom_tint,
        custom_red_balance,
        custom_green_balance,
        custom_blue_balance,
    ) -> io.NodeOutput:
        """Build or read the table.

        Raises:
            ValueError: The chosen file is no longer in any search directory, or the file
                is not a readable ``.cube``.
        """
        import numpy as np

        if look == CUSTOM:
            cube = tables.identity_cube(builtin_size)
            cube = tables.apply_exposure(cube, custom_ev)
            cube = tables.apply_contrast(cube, custom_contrast)
            cube = tables.apply_saturation(cube, custom_saturation)
            cube = tables.apply_vibrance(cube, custom_vibrance)
            cube = tables.apply_white_balance(cube, custom_temperature, custom_tint)
            cube = tables.apply_color_balance(
                cube, custom_red_balance, custom_green_balance, custom_blue_balance
            )
            cube = tables.apply_gamma(cube, custom_gamma)
            table = cube.squeeze(0).clamp(0, 1).cpu().numpy().astype(np.float32)
            built = tables.LUT(CUSTOM, (0, 0, 0), (1, 1, 1), None, table)
            lut_report.publish(built)
            return io.NodeOutput(built)

        if look.startswith(FILE_PREFIX):
            loaded = tables.load_cube(tables.find_cube(look[len(FILE_PREFIX):].strip()))
            lut_report.publish(loaded)
            return io.NodeOutput(loaded)

        preset = tables.synthesize_builtin_lut(look, builtin_size)
        lut_report.publish(preset)
        return io.NodeOutput(preset)
