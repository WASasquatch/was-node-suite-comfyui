"""Read a graded colour lattice back into a 3D lookup table."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.compat.types import LUT
from ....modules.image import lattice as lattices
from ....modules.image import lut as tables
from ....modules.interface import lut_report

logger = log.get_logger("nodes.image.lut")

#: Title written into the table when none is given.
DEFAULT_TITLE = "LUT from Lattice"


def _sizes() -> str:
    """The picture sizes LUT Lattice builds, with probes and without, for an error message."""
    banded = ", ".join(
        f"{lattices.side(level)}x{lattices.side(level) + lattices.band_height(level)}"
        for level in lattices.LEVELS
    )
    bare = ", ".join(f"{lattices.side(level)}x{lattices.side(level)}" for level in lattices.LEVELS)
    return f"{banded} with probes, or {bare} without"


def _spatial_message(agreement, tolerance: float) -> str:
    """The refusal for a chain whose probes disagree, naming what kind of effect did it."""
    found = []
    if agreement.noise > tolerance:
        found.append(
            f"flat patches of one colour came out uneven by {agreement.noise:.2f}, which is "
            f"grain, noise or dither"
        )
    if agreement.neighbourhood > tolerance:
        found.append(
            f"a single pixel among other colours came out {agreement.neighbourhood:.2f} from "
            f"the same colour in a flat patch, which is clarity, texture, dehaze, sharpening, "
            f"a blur, halation or a glow"
        )
    if agreement.position > tolerance:
        found.append(
            f"one colour came out {agreement.position:.2f} apart in different parts of the "
            f"frame, which is a vignette, a light leak, a gradient or a mask"
        )
    if not found:
        found.append("the probe band no longer matches the lattice square above it")
    return (
        f"LUT from Lattice found a spatial effect in the chain: the '{agreement.probe}' probe "
        f"came out up to {agreement.worst:.2f} apart (CIEDE2000) against a tolerance of "
        f"{tolerance:.2f}. In detail, {'; '.join(found)}. A LUT maps a colour to one colour "
        f"wherever it sits, so this chain cannot be baked. Set that effect to 0 or bypass its "
        f"node between LUT Lattice and LUT from Lattice, bake the rest, and apply the effect "
        f"after Apply LUT. A picture that is not a lattice from LUT Lattice is refused the same "
        f"way."
    )


class LUTFromLattice(io.ComfyNode):
    """Turn a lattice graded by a colour chain into a 3D lookup table."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLUTFromLattice",
            display_name="LUT from Lattice",
            search_aliases=[
                "WASLUTFromLattice", "LUT from Lattice", "bake lut", "hald", "hald clut",
                "chain to lut", "export look", "cube",
            ],
            category="WAS Suite/Image/LUT",
            description=(
                "Read a lattice from LUT Lattice, graded by a chain of colour nodes, back into "
                "a LUT for Apply LUT or Save LUT (.cube). It refuses a lattice whose size "
                "changed, and a chain with grain, clarity, a vignette or any other spatial "
                "effect, which no LUT can hold."
            ),
            inputs=[
                io.Image.Input(
                    "lattice",
                    tooltip=(
                        "The lattice after the colour chain. A bare Hald square of any level, "
                        "such as one graded in another app, is read too, with no spatial check."
                    ),
                ),
                io.Float.Input(
                    "tolerance", default=0.5, min=0.0, max=10.0, step=0.05,
                    tooltip=(
                        "Largest CIEDE2000 difference allowed between copies of one probe "
                        "colour. 0 = identical; 0.5 = half a just-noticeable difference, which "
                        "refuses clarity at 5, grain at 0.2 and a vignette at 0.1."
                    ),
                ),
                io.String.Input(
                    "title", default="", optional=True,
                    tooltip=(
                        "Name written into the LUT's TITLE line, which Save LUT (.cube) puts in "
                        "the file. Empty = 'LUT from Lattice'."
                    ),
                ),
            ],
            outputs=[
                LUT.Output(
                    display_name="lut",
                    tooltip=(
                        "The chain as a 3D table at the lattice's own steps, for inputs 0 to 1, "
                        "with values below 0 or above 1 kept as the chain answered them. For "
                        "Apply LUT, LUT Blender or Save LUT (.cube)."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, lattice, tolerance=0.5, title="") -> io.NodeOutput:
        """Check the lattice and read it into a table.

        Raises:
            ValueError: The picture has no lattice's size, holds a value that is not finite,
                or its probes disagree past ``tolerance``.
        """
        import numpy as np
        import torch

        images = lattice
        if images.ndim == 3:
            images = images.unsqueeze(0)
        frames, height, width = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
        layout = lattices.layout_of(height, width)
        if layout is None:
            raise ValueError(
                f"LUT from Lattice was handed a {width}x{height} picture, and no lattice has "
                f"that size. LUT Lattice builds {_sizes()}. A node in the chain resized, "
                f"cropped, padded or rotated the lattice: take that node out, or move it after "
                f"Apply LUT."
            )
        level, banded = layout

        picture = images[0].detach().to("cpu", torch.float32)
        if picture.shape[-1] == 1:
            picture = picture.expand(height, width, 3)
        elif picture.shape[-1] == 2:
            picture = picture[..., :1].expand(height, width, 3)
        picture = picture[..., :3].contiguous()
        if not bool(torch.isfinite(picture).all()):
            raise ValueError(
                "LUT from Lattice found values that are not numbers or are infinite in the "
                "lattice, so a node in the chain divided by zero or overflowed. Preview the "
                "lattice after each node to find it."
            )

        tolerance = max(0.0, float(tolerance))
        notes = []
        if banded:
            agreement = lattices.check_probes(picture, level)
            if agreement.worst > tolerance:
                raise ValueError(_spatial_message(agreement, tolerance))
            notes.append(f"probes agree within {agreement.worst:.2f}")
        else:
            notes.append("no probe band, so spatial effects were not checked")

        cube = lattices.read_table(picture, level).numpy().astype(np.float32)
        low, high = float(cube.min()), float(cube.max())
        if low < 0.0 or high > 1.0:
            notes.append(f"values {low:.3f} to {high:.3f} kept")
        if frames > 1:
            notes.append(f"frame 1 of {frames} read")

        built = tables.LUT(
            str(title or "").strip() or DEFAULT_TITLE,
            (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), None, np.ascontiguousarray(cube),
        )
        lut_report.publish(built, detail=", ".join(notes))
        logger.info("LUT from Lattice read a %d-step cube", lattices.steps(level))
        return io.NodeOutput(built)
