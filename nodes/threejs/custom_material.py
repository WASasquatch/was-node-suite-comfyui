"""A surface built by a Three.js module file."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import THREE_MATERIAL, THREE_TEXTURE
from ...modules.threejs import module_file
from ...modules.threejs.spec import compact_deps, create_spec

REQUIRES = "threejs"

TEXTURE_TOOLTIP = (
    "A texture the module receives under the name its header declares. The first declared "
    "texture arrives here, the second in the slot below, and so on."
)

#: Textures a material module may declare, which is how many slots this node draws.
SLOTS = 4


def options() -> list[str]:
    """The module menu's entries."""
    return module_file.options()


class ThreeCustomMaterial(io.ComfyNode):
    """Build a material from a module file."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeCustomMaterial",
            display_name="Three Custom Material",
            search_aliases=[
                "WASThreeCustomMaterial",
                "Three Custom Material",
                "custom material",
                "javascript",
                "module",
                "toon",
            ],
            category="WAS Suite/Three",
            description=(
                "Reach any Three.js material class the pack has no node for, from a module "
                "file you place in ComfyUI's input folder or a folder under "
                "paths.allow_read. The module's header names the textures it wants and the "
                "node's slots take those names. Code is never typed on the node and never "
                "travels inside a workflow, so a graph from someone else cannot bring "
                "javascript with it. A module runs in your browser when the viewer loads, "
                "with the same reach as any frontend extension, and only while "
                "threejs.allow_scripts is on."
            ),
            inputs=[
                io.Combo.Input(
                    "module",
                    options=options(),
                    tooltip=(
                        "Which module builds the material. The menu lists every `.js` and "
                        "`.txt` file opening with `// was-threejs-module 1` in ComfyUI's "
                        "input, output and temp folders and in any folder under "
                        "paths.allow_read."
                    ),
                ),
                THREE_TEXTURE.Input("texture1", optional=True, tooltip=TEXTURE_TOOLTIP),
                THREE_TEXTURE.Input("texture2", optional=True, tooltip=TEXTURE_TOOLTIP),
                THREE_TEXTURE.Input("texture3", optional=True, tooltip=TEXTURE_TOOLTIP),
                THREE_TEXTURE.Input("texture4", optional=True, tooltip=TEXTURE_TOOLTIP),
            ],
            outputs=[
                THREE_MATERIAL.Output(
                    display_name="material",
                    tooltip="The surface the module returned, for the material socket on Three Mesh.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls, module, texture1=None, texture2=None, texture3=None, texture4=None
    ) -> io.NodeOutput:
        """Carry the module's body and its textures to the browser.

        Raises:
            PermissionError: ``threejs.allow_scripts`` is off.
            ValueError: No module is chosen, the module builds something other than a
                material, or it declares more textures than this node has slots.
        """
        if not module_file.chosen(module):
            raise ValueError(
                f"no module was chosen. Put a `.js` or `.txt` file opening with "
                f"`// {module_file.MARKER} {module_file.VERSION}` in ComfyUI's input folder, "
                f"or in a folder under paths.allow_read, and pick it from the menu"
            )
        declared = module_file.load(module)
        if declared.kind != "material":
            raise ValueError(
                f"`{module}` builds a {declared.kind}, and this node needs a material. Use "
                f"the Three Custom {declared.kind.title()} node, or change the module's "
                f"`@kind` line"
            )
        wanted = [one for one in declared.wires if one.kind != "texture"]
        if wanted:
            raise ValueError(
                f"`{module}` declares {wanted[0].kind} `{wanted[0].name}`, and a material "
                f"module takes textures only. Drop that line from its header"
            )
        if len(declared.wires) > SLOTS:
            raise ValueError(
                f"`{module}` declares {len(declared.wires)} textures and this node has "
                f"{SLOTS} slots. Drop one, or read it from a Three Script Module"
            )
        return io.NodeOutput(
            create_spec(
                "material",
                "CustomMaterial",
                params={
                    "javascript": declared.body,
                    "names": [one.name for one in declared.wires],
                    "values": {one.name: one.default for one in declared.values},
                },
                deps=compact_deps(
                    texture1=texture1,
                    texture2=texture2,
                    texture3=texture3,
                    texture4=texture4,
                ),
            )
        )
