"""An identity colour lattice to bake a chain of colour nodes into a LUT."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.image import lattice

logger = log.get_logger("nodes.image.lut")

#: The size menu, as option text to Hald level.
SIZES = {f"{lattice.steps(level)} steps": level for level in lattice.LEVELS}

DEFAULT_SIZE = f"{lattice.steps(8)} steps"


class LUTLattice(io.ComfyNode):
    """Answer the identity Hald lattice a colour chain is baked from."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLUTLattice",
            display_name="LUT Lattice",
            search_aliases=[
                "WASLUTLattice", "LUT Lattice", "hald", "hald clut", "identity lut",
                "bake lut", "make lut", "color lattice",
            ],
            category="WAS Suite/Image/LUT",
            description=(
                "Make an identity colour lattice for baking a chain of colour nodes into a LUT. "
                "Wire it through the chain, then into LUT from Lattice. The square is a "
                "standard Hald CLUT, and the probe band under it lets LUT from Lattice refuse "
                "a chain holding a spatial effect."
            ),
            inputs=[
                io.Combo.Input(
                    "size",
                    options=list(SIZES),
                    default=DEFAULT_SIZE,
                    tooltip=(
                        "Steps per channel. `25 steps` = Hald level 5, 125 px; `36` = level 6, "
                        "216 px; `49` = level 7, 343 px; `64` = level 8, 512 px. More steps "
                        "follow a strong grade more closely."
                    ),
                ),
                io.Boolean.Input(
                    "probes",
                    default=True,
                    tooltip=(
                        "`true` = a band of probe patches under the square, so LUT from Lattice "
                        "refuses grain, clarity, a vignette or a blur. `false` = the bare Hald "
                        "square, 512x512 at 64 steps, for an app that grades Hald CLUT pictures."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="lattice",
                    tooltip=(
                        "The identity lattice in picture codes, 0 to 1. Grade it as the "
                        "pictures the LUT is for, then wire it into LUT from Lattice."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, size=DEFAULT_SIZE, probes=True) -> io.NodeOutput:
        """Build the lattice.

        Raises:
            ValueError: ``size`` is not one of :data:`SIZES`.
        """
        level = SIZES.get(str(size).strip())
        if level is None:
            raise ValueError(
                f"LUT Lattice was given a size of {size!r}, and it builds {', '.join(SIZES)}. "
                f"Set size to one of those; 64 steps is the usual Hald level 8."
            )
        picture = lattice.build(level, bool(probes))
        logger.info(
            "LUT Lattice built a %d-step lattice, %dx%d",
            lattice.steps(level), int(picture.shape[2]), int(picture.shape[1]),
        )
        return io.NodeOutput(picture)
