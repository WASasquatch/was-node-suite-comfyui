"""Writing a colour lookup table out as a ``.cube`` file."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.compat.types import LUT
from ....modules.image import lut as tables
from ....modules.interface import lut_report
from ....modules.util import sandbox

REQUIRES = "extras"

logger = log.get_logger("nodes.extras.image")

#: Extension every saved table carries.
CUBE_SUFFIX = ".cube"


class SaveLUT(io.ComfyNode):
    """Write a colour lookup table to a ``.cube`` file and pass it on."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASNSSaveLUT",
            display_name="Save LUT (.cube)",
            search_aliases=[
                "WASNSSaveLUT", "WAS Save LUT (.cube)", "lut", "cube", "save", "export",
            ],
            category="WAS Suite/Image/LUT",
            description=(
                "Write a colour lookup table to a .cube file, the format DaVinci Resolve, "
                "Premiere and most grading tools read. Files land in the pack's own luts "
                "directory under ComfyUI's user folder, which Load LUT also reads, so a look "
                "built once here can be reused everywhere. The table is passed straight "
                "through as well, so the node can sit mid-chain."
            ),
            inputs=[
                LUT.Input(
                    "lut",
                    tooltip=(
                        "The table to write, from Load LUT or LUT Blender. A table stored as "
                        "curves is converted to a cube first, since .cube files hold cubes."
                    ),
                ),
                io.String.Input(
                    "filename", default="CustomLUT",
                    tooltip=(
                        "Name of the file to write, such as 'WarmFilm'. The .cube extension "
                        "is added when it is missing. This is a name, not a path: it may "
                        "name a subfolder, but it cannot step outside the luts directory."
                    ),
                ),
                io.Int.Input(
                    "output_size", default=33, min=17, max=65, step=2,
                    tooltip=(
                        "Edge length of the cube written to the file, in samples. 33 is the "
                        "industry-standard size and is what most grading tools expect; 65 is "
                        "finer and produces a file eight times the size."
                    ),
                ),
                io.Boolean.Input(
                    "overwrite", default=True,
                    tooltip=(
                        "Whether an existing file of the same name may be replaced. Turn it "
                        "off to have the node stop rather than overwrite a look already "
                        "saved under that name."
                    ),
                ),
            ],
            outputs=[
                LUT.Output(
                    display_name="lut",
                    tooltip=(
                        "The table as it was written, resampled to output_size, so the "
                        "downstream grade matches the file exactly."
                    ),
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, lut, filename, output_size, overwrite) -> io.NodeOutput:
        """Resample the table, write it, and pass it on.

        Raises:
            ValueError: ``filename`` is empty, or the table holds nothing to write.
            FileExistsError: The file is already there and ``overwrite`` is off.
            PathNotAllowed: ``filename`` names somewhere outside the luts directory.
        """
        name = str(filename).strip()
        if not name:
            raise ValueError("give the LUT a file name to save it under, such as 'WarmFilm'")
        if not name.lower().endswith(CUBE_SUFFIX):
            name = name + CUBE_SUFFIX

        target = sandbox.resolve_write_file(tables.save_directory(), name)
        if target.exists() and not overwrite:
            raise FileExistsError(
                f"{target} is already there. Turn overwrite on to replace it, or save it "
                f"under another name."
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        resampled = tables.convert_to_3d(lut, output_size)
        tables.save_cube(target, resampled)
        logger.info("wrote the LUT to %s", target)

        lut_report.publish(resampled, strip=False, detail=f"written to {target.name}")
        return io.NodeOutput(resampled)
