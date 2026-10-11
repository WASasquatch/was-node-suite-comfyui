"""One named export taken out of a script module."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import (
    THREE_GEOMETRY,
    THREE_MATERIAL,
    THREE_MODULE,
    THREE_OBJECT,
)
from ...modules.threejs.spec import create_spec, require_spec

REQUIRES = "threejs"

MODULE_TOOLTIP = "The named resources to pick from, from Three Script Module."


class ThreeModuleMaterial(io.ComfyNode):
    """Take a material out of a script module."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeImportMaterial",
            display_name="Three Import Material",
            search_aliases=[
                "WASThreeImportMaterial",
                "Three Import Material",
                "import material",
                "module export",
            ],
            category="WAS Suite/Three",
            description=(
                "Take one material out of a Three Script Module by name and put it on a normal "
                "material wire, so a mesh can use it without knowing it came from JavaScript. "
                "The name has to match a key the module returned, and the value it holds has to "
                "be a material or the viewer reports it by name."
            ),
            inputs=[
                THREE_MODULE.Input("module", tooltip=MODULE_TOOLTIP),
                io.String.Input(
                    "export_name",
                    default="gold",
                    multiline=False,
                    tooltip="Which key to take, as `gold`. It must match a key the module returned.",
                ),
            ],
            outputs=[
                THREE_MATERIAL.Output(
                    display_name="material",
                    tooltip="The named material, for the material socket on Three Mesh.",
                ),
            ],
        )

    @classmethod
    def execute(cls, module, export_name) -> io.NodeOutput:
        """Name the export to take.

        Raises:
            ValueError: The input is not a module descriptor.
        """
        require_spec(module, "module")
        return io.NodeOutput(
            create_spec(
                "material",
                "ModuleExport",
                params={"exportName": export_name, "expected": "material"},
                deps={"module": module},
            )
        )


class ThreeModuleGeometry(io.ComfyNode):
    """Take a geometry out of a script module."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeImportGeometry",
            display_name="Three Import Geometry",
            search_aliases=[
                "WASThreeImportGeometry",
                "Three Import Geometry",
                "import geometry",
                "module export",
            ],
            category="WAS Suite/Three",
            description=(
                "Take one geometry out of a Three Script Module by name and put it on a normal "
                "geometry wire, so a mesh can use it without knowing it came from JavaScript. "
                "The name has to match a key the module returned, and the value it holds has to "
                "be a geometry or the viewer reports it by name."
            ),
            inputs=[
                THREE_MODULE.Input("module", tooltip=MODULE_TOOLTIP),
                io.String.Input(
                    "export_name",
                    default="ring",
                    multiline=False,
                    tooltip="Which key to take, as `ring`. It must match a key the module returned.",
                ),
            ],
            outputs=[
                THREE_GEOMETRY.Output(
                    display_name="geometry",
                    tooltip="The named shape, for the geometry socket on Three Mesh.",
                ),
            ],
        )

    @classmethod
    def execute(cls, module, export_name) -> io.NodeOutput:
        """Name the export to take.

        Raises:
            ValueError: The input is not a module descriptor.
        """
        require_spec(module, "module")
        return io.NodeOutput(
            create_spec(
                "geometry",
                "ModuleExport",
                params={"exportName": export_name, "expected": "geometry"},
                deps={"module": module},
            )
        )


class ThreeModuleObject(io.ComfyNode):
    """Take an object out of a script module."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeImportObject",
            display_name="Three Import Object",
            search_aliases=[
                "WASThreeImportObject",
                "Three Import Object",
                "import object",
                "module export",
            ],
            category="WAS Suite/Three",
            description=(
                "Take one object out of a Three Script Module by name and put it on a normal "
                "object wire, for Three Group or Three Scene. The name has to match a key the "
                "module returned, and a value that is not an object is reported by name in the "
                "viewer."
            ),
            inputs=[
                THREE_MODULE.Input("module", tooltip=MODULE_TOOLTIP),
                io.String.Input(
                    "export_name",
                    default="rig",
                    multiline=False,
                    tooltip="Which key to take, as `rig`. It must match a key the module returned.",
                ),
            ],
            outputs=[
                THREE_OBJECT.Output(
                    display_name="object",
                    tooltip="The named object, for Three Group or the root socket on Three Scene.",
                ),
            ],
        )

    @classmethod
    def execute(cls, module, export_name) -> io.NodeOutput:
        """Name the export to take.

        Raises:
            ValueError: The input is not a module descriptor.
        """
        require_spec(module, "module")
        return io.NodeOutput(
            create_spec(
                "object",
                "ModuleExport",
                params={"exportName": export_name, "expected": "object"},
                deps={"module": module},
            )
        )
