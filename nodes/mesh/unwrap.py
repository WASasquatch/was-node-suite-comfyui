"""Give a 3D model's parts a fresh texture layout: charts cut, flattened and packed into one atlas."""

from __future__ import annotations

from dataclasses import replace

import torch
from comfy_api.latest import io

from ...modules.compat.types import FILE_3D, FILE_3D_FBX, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_OBJ, MODEL_3D
from ...modules.mesh import bake as mesh_bake
from ...modules.mesh import raster
from ...modules.mesh import unwrap as mesh_unwrap
from ...modules.model3d import edit, inputs, report, selection

#: Widest layout picture, in pixels.
LAYOUT_SIDE = 1024

#: Grey of the atlas where no chart lies.
LAYOUT_BACKGROUND = 0.12


def _layout(result: mesh_unwrap.Unwrapped, side: int) -> torch.Tensor:
    """The packed charts drawn in colours, one colour per chart, as an ``IMAGE``."""
    picture = torch.full((side * side, 3), LAYOUT_BACKGROUND)
    generator = torch.Generator().manual_seed(7)
    colours = 0.35 + 0.6 * torch.rand(max(result.charts, 1), 3, generator=generator)
    for part, charts in zip(result.parts, result.face_charts):
        corners = part.uvs[part.faces.long()] * side
        shade = 0.85 + 0.15 * torch.rand(part.triangle_count, generator=generator)
        for piece in raster.coverage_chunks(corners, side, side):
            picture[piece.pixel] = colours[charts[piece.face]] * shade[piece.face, None]
    return picture.reshape(1, side, side, 3).clamp(0.0, 1.0)


def _without_textures(material):
    """A copy of a material with its texture maps removed and its factors kept."""
    return replace(material, base_color_texture=None, metallic_roughness_texture=None, normal_texture=None,
                   occlusion_texture=None, emissive_texture=None,
                   extras={key: value for key, value in material.extras.items()
                           if key not in ("texture_transform", "mtoon", "vrm0_material")})


class MeshUnwrap(io.ComfyNode):
    """Cut, flatten and pack a model's parts into one new texture atlas."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASMeshUnwrap",
            display_name="Mesh UV Unwrap",
            search_aliases=[
                "WASMeshUnwrap", "Mesh UV Unwrap", "uv unwrap", "unwrap", "uv map", "texture coordinates",
                "uv layout", "atlas", "lightmap uv", "uv pack", "seams",
            ],
            category="WAS Suite/3D/Mesh",
            description=(
                "Give a model new texture coordinates: its surface is cut into flat charts along hidden "
                "seams and packed into one square texture with even detail everywhere. Use it after "
                "decimating or retopologising, then bake the old look onto the new layout."
            ),
            inputs=[
                io.MultiType.Input("model_3d", types=[MODEL_3D, io.Mesh, FILE_3D, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_FBX, FILE_3D_OBJ], tooltip="The model to work on: a 3D model from another 3D node, a mesh from core's mesh nodes, or a GLB, glTF, VRM, FBX or OBJ file from Load 3D or a 3D generator such as Tripo."),
                io.Float.Input(
                    "chart_angle", default=60.0, min=10.0, max=mesh_unwrap.MAX_CHART_ANGLE, step=1.0,
                    tooltip="Degrees a face may turn from its chart's main direction. 30 = many small, nearly "
                            "flat charts with almost no stretch; 60 = balanced; 85 = few large charts, more "
                            "stretch.",
                ),
                io.Int.Input(
                    "resolution", default=2048, min=64, max=mesh_bake.MAX_RESOLUTION, step=64,
                    tooltip="Size of the texture the layout is packed for, in pixels. Padding is measured "
                            "at this size; 2048 suits a game character, 4096 a hero asset.",
                ),
                io.Int.Input(
                    "padding", default=4, min=0, max=64, step=1,
                    tooltip="Pixels kept empty between charts at that resolution, so colours do not bleed "
                            "across seams when the texture is shrunk. 4 = usual; 8 = safer for small mipmaps.",
                ),
                io.Combo.Input(
                    "quality", options=list(mesh_unwrap.QUALITIES), default="balanced",
                    tooltip="`fast` = quick flattening; `balanced` = tighter solve; `best` = also relaxes "
                            "each chart towards true triangle shapes, lowest stretch, slowest.",
                ),
                io.Boolean.Input(
                    "keep_textures", default=True,
                    tooltip="`true` = the textures the parts had are redrawn onto the new layout, so the model "
                            "keeps its look; `false` = textures are removed and the colours left for Mesh Bake "
                            "Textures to fill.",
                ),
                io.String.Input("parts", default="", advanced=True, tooltip="Parts to work on: comma-separated names or labels, wildcards allowed, such as `head, hair*`. Empty works on every part."),
            ],
            outputs=[
                MODEL_3D.Output("model_3d", tooltip="The whole model: parts with their materials, and the skeleton, skin, springs, expressions and animations it carries."),
                io.Mesh.Output("mesh", tooltip="The same model as one mesh for core's mesh nodes and Save 3D Model. Parts are merged, their textures packed into one atlas, and the skeleton left out."),
                io.Image.Output(
                    "layout",
                    tooltip="The new atlas drawn as a picture, one colour per chart, at most 1024 pixels wide. "
                            "Grey is unused texture space.",
                ),
                io.String.Output(
                    "report",
                    tooltip="Charts made, share of the atlas used, how much the layout stretches the surface "
                            "and how many vertices were split along the new seams.",
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, model_3d, chart_angle=60.0, resolution=2048, padding=4, quality="balanced",
                keep_textures=True, parts="") -> io.NodeOutput:
        """Unwrap the chosen parts together into one atlas.

        Raises:
            ValueError: Nothing is wired, no part matches ``parts``, the chosen parts have no
                triangles, or they are larger than an unwrap takes.
        """
        model = inputs.to_model(model_3d)
        before = report.figures(model)
        chosen = selection.selected(model, parts)
        if not chosen:
            raise ValueError(f"no part matches '{str(parts)[:80]}'; the model's parts are {edit.part_names(model)}. "
                             "Leave parts empty to unwrap every part.")
        chosen = [index for index in chosen if model.parts[index].triangle_count]
        if not chosen:
            raise ValueError("the chosen parts have no triangles to unwrap; wire a model with faces or choose "
                             "other parts")
        total = sum(model.parts[index].triangle_count for index in chosen)
        if total > mesh_unwrap.MAX_TRIANGLES:
            raise ValueError(f"the chosen parts hold {total:,} triangles, more than the "
                             f"{mesh_unwrap.MAX_TRIANGLES:,} one unwrap takes; decimate them first")
        quality = quality if quality in mesh_unwrap.QUALITIES else "balanced"
        settings = mesh_unwrap.UnwrapSettings(chart_angle=float(chart_angle), resolution=int(resolution),
                                              padding=int(padding), quality=quality)
        vertices_before = sum(model.parts[index].vertex_count for index in chosen)
        result = mesh_unwrap.unwrap_parts([model.parts[index] for index in chosen], settings)
        for index, part in zip(chosen, result.parts):
            model.parts[index] = part

        new_parts = [model.parts[index] for index in chosen]
        slots = []
        for part in new_parts:
            earlier = part.extras.get(mesh_bake.PREVIOUS_MATERIAL) if mesh_bake.has_previous(part) else None
            if earlier is not None:
                slots += [name for name in earlier.textures() if name not in slots]
        carried = ""
        if keep_textures and slots:
            maps = tuple(name for name in mesh_bake.MAPS if name in slots)
            baked = mesh_bake.bake(new_parts, None, mesh_bake.BakeSettings(
                resolution=int(resolution), maps=maps, samples=1, padding=max(2, int(padding)), ao_rays=0))
            for index in chosen:
                model.parts[index].material = baked.material
            carried = ", ".join(maps)
        else:
            for index in chosen:
                model.parts[index].material = _without_textures(model.parts[index].material)

        stats = result.stats
        vertices_after = sum(model.parts[index].vertex_count for index in chosen)
        lines = [
            f"Mesh UV Unwrap: {result.charts:,} charts across {len(chosen)} part{'s' if len(chosen) != 1 else ''}, "
            f"{100.0 * stats['efficiency']:.1f}% of a {int(resolution)} pixel atlas used",
            f"  area stretch {stats['area_distortion']:.4f} (0 = none), angle stretch "
            f"{stats['angle_distortion']:.4f} (1 = none), {stats['flipped']} mirrored triangles, "
            f"{stats['overlapping_texels']} texels claimed twice",
            f"  vertices {vertices_before:,} -> {vertices_after:,} after splitting along the new seams",
        ]
        if carried:
            lines.append(f"  textures redrawn onto the new layout: {carried}")
        elif slots:
            lines.append("  textures removed; wire this model into Mesh Bake Textures to restore its look")
        if stats.get("folded_charts") or stats.get("overlapping_charts"):
            lines.append(f"  {stats['folded_charts']} chart(s) still fold and {stats['overlapping_charts']} still "
                         "overlap themselves; lower chart_angle or pick quality best")
        text = "\n".join(lines)
        report.publish(
            model, f"{result.charts:,} charts, {100.0 * stats['efficiency']:.1f}% of the atlas used", before=before,
            counts={"charts": result.charts, "atlas used %": round(100.0 * stats["efficiency"], 2),
                    "area stretch": round(stats["area_distortion"], 4),
                    "angle stretch": round(stats["angle_distortion"], 4)},
            facts={"quality": quality, "textures": carried or ("removed" if slots else "none"),
                   "parts": edit.part_names(model, 8)},
        )
        picture = _layout(result, min(int(resolution), LAYOUT_SIDE))
        hidden = cls.hidden
        wanted_mesh = inputs.mesh_wanted(getattr(hidden, "prompt", None), getattr(hidden, "unique_id", None))
        return io.NodeOutput(*inputs.node_output(model, mesh=wanted_mesh), picture, text)
