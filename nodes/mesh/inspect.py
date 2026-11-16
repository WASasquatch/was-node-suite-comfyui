"""Measure a 3D model the way a modeller checks it before export, and pass it through unchanged."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import FILE_3D, FILE_3D_FBX, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_OBJ, MODEL_3D
from ...modules.mesh import quality
from ...modules.mesh.device import with_cpu_fallback
from ...modules.model3d import edit, inputs, report, selection


class MeshInspect(io.ComfyNode):
    """Report a model's counts, topology, triangle quality, texture coordinates and skin."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASMeshInspect",
            display_name="Mesh Inspect",
            search_aliases=[
                "WASMeshInspect", "Mesh Inspect", "mesh stats", "polycount", "watertight check",
                "non-manifold check", "model info", "uv check", "topology report",
            ],
            category="WAS Suite/3D/Mesh",
            description=(
                "Check a model before export: counts, watertightness, non-manifold edges, holes, shells, "
                "self-intersections, triangle quality, texture coverage and overlaps, and skin weights. "
                "The model passes through unchanged, with a readable report and the key figures as outputs."
            ),
            inputs=[
                io.MultiType.Input("model_3d", types=[MODEL_3D, io.Mesh, FILE_3D, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_FBX, FILE_3D_OBJ], tooltip="The model to work on: a 3D model from another 3D node, a mesh from core's mesh nodes, or a GLB, glTF, VRM, FBX or OBJ file from Load 3D or a 3D generator such as Tripo."),
                io.Boolean.Input(
                    "self_intersections", default=True,
                    tooltip="`true` = count faces crossing another face of their part; about half a minute "
                            "on two million triangles without a GPU. `false` = skip it.",
                ),
                io.Int.Input(
                    "uv_resolution", default=1024, min=64, max=4096, step=64, advanced=True,
                    tooltip="Pixels per side of the grid texture coverage and overlap are measured on. "
                            "1024 = most textures; 4096 = finer, slower. Lowered on its own for very dense models.",
                ),
                io.String.Input("parts", default="", advanced=True, tooltip="Parts to work on: comma-separated names or labels, wildcards allowed, such as `head, hair*`. Empty works on every part."),
            ],
            outputs=[
                MODEL_3D.Output("model_3d", tooltip="The whole model: parts with their materials, and the skeleton, skin, springs, expressions and animations it carries."),
                io.Mesh.Output("mesh", tooltip="The same model as one mesh for core's mesh nodes and Save 3D Model. Parts are merged, their textures packed into one atlas, and the skeleton left out."),
                io.String.Output(
                    "report",
                    tooltip="Every figure as readable text: counts, size, topology, triangle quality, texture "
                            "coordinates and rig.",
                ),
                io.Int.Output("triangles", tooltip="Triangles across the chosen parts."),
                io.Int.Output("vertices", tooltip="Vertices across the chosen parts, counting split texture seams."),
                io.Float.Output("height", tooltip="Height of the chosen parts along Y, in metres."),
                io.Boolean.Output(
                    "watertight",
                    tooltip="`true` when the chosen parts have triangles, no open border and no non-manifold "
                            "edge or vertex.",
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model_3d, self_intersections=True, uv_resolution=1024, parts="") -> io.NodeOutput:
        """Measure the chosen parts and pass the model on.

        Raises:
            ValueError: Nothing is wired, or no part matches ``parts``.
        """
        model = inputs.to_model(model_3d)
        chosen = selection.selected(model, parts)
        if not chosen and model.parts:
            raise ValueError(f"no part matches '{str(parts)[:80]}'; the model's parts are {edit.part_names(model)}. "
                             "Leave parts empty to inspect every part.")
        figures = with_cpu_fallback(
            lambda device: quality.inspect(model, chosen, bool(self_intersections), int(uv_resolution), device),
            label="mesh inspect",
        )
        text = quality.report_text(figures, [model.parts[index].name for index in chosen])
        crossing = figures["intersecting_faces"]
        report.publish(
            model, "Watertight" if figures["watertight"] else "Not watertight",
            counts={
                "border edges": figures["boundary_edges"],
                "non-manifold edges": figures["non_manifold_edges"],
                "non-manifold vertices": figures["non_manifold_vertices"],
                "shells": figures["shells"],
                "genus": figures["genus"],
                "crossing faces": crossing,
                "poor triangles": figures["poor_triangles"],
                "unweighted vertices": figures["unweighted_vertices"],
            },
            facts={
                "watertight": "yes" if figures["watertight"] else "no",
                "size": " x ".join(f"{value:.3f}" for value in figures["size"]),
                "uv coverage": f"{figures['uv_coverage'] * 100:.1f}%",
                "uv overlap": f"{figures['uv_overlap'] * 100:.1f}%",
                "texel density": f"{figures['texel_density']:.0f} px per metre",
                "smallest angle": f"mean {figures['min_angle_mean']:.1f}, lowest {figures['min_angle_min']:.2f} degrees",
                "valence": f"mean {figures['valence_mean']:.2f}",
                "influences": f"up to {figures['max_influences']} per vertex",
            },
        )
        hidden = cls.hidden
        wanted = inputs.mesh_wanted(getattr(hidden, "prompt", None), getattr(hidden, "unique_id", None))
        return io.NodeOutput(
            *inputs.node_output(model, mesh=wanted), text, int(figures["triangles"]), int(figures["vertices"]),
            float(figures["height"]), bool(figures["watertight"]),
        )
