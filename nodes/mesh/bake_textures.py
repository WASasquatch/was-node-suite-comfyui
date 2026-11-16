"""Bake a detailed model's colours, normals and materials into another model's texture layout."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ...modules.compat.types import FILE_3D, FILE_3D_FBX, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_OBJ, MODEL_3D
from ...modules.interface.progress import progress_bar
from ...modules.mesh import bake as mesh_bake
from ...modules.mesh import tangents
from ...modules.model3d import edit, inputs, report, selection
from ...modules.model3d.scene import Model3D

#: Steps the progress bar is divided into.
PROGRESS_STEPS = 100


def _image(texture: torch.Tensor) -> torch.Tensor:
    """A baked texture as a one-picture ``IMAGE`` batch, colour channels only."""
    return texture[:, :, :3].unsqueeze(0).contiguous()


class MeshBakeTextures(io.ComfyNode):
    """Bake a source model's look into a target model's texture layout."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASMeshBakeTextures",
            display_name="Mesh Bake Textures",
            search_aliases=[
                "WASMeshBakeTextures", "Mesh Bake Textures", "bake", "texture bake", "normal map bake",
                "bake normals", "ambient occlusion", "high to low", "transfer textures", "bake maps",
            ],
            category="WAS Suite/3D/Mesh",
            description=(
                "Bake a detailed model's colour, surface detail, metal, roughness, glow and shadowed creases "
                "into the texture layout of a lighter version, so a decimated or retopologised model keeps the "
                "original's look. Without a source, it redraws the textures a model had before Mesh UV Unwrap."
            ),
            inputs=[
                io.MultiType.Input("model_3d", types=[MODEL_3D, io.Mesh, FILE_3D, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_FBX, FILE_3D_OBJ], tooltip="The model to work on: a 3D model from another 3D node, a mesh from core's mesh nodes, or a GLB, glTF, VRM, FBX or OBJ file from Load 3D or a 3D generator such as Tripo."),
                io.MultiType.Input(
                    "source", types=[MODEL_3D, io.Mesh, FILE_3D, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_FBX, FILE_3D_OBJ],
                    optional=True,
                    tooltip="The detailed model the look comes from, such as the high poly before decimation, "
                            "lying where the target lies. Unwired = the target's own textures from before its "
                            "Mesh UV Unwrap.",
                ),
                io.Int.Input(
                    "resolution", default=2048, min=64, max=mesh_bake.MAX_RESOLUTION, step=64,
                    tooltip="Width and height of every baked texture, in pixels. 2048 suits a game character; "
                            "4096 a close-up hero asset.",
                ),
                io.Boolean.Input("base_color", default=True,
                                 tooltip="`true` = bake the colour map, with transparency where the source has it."),
                io.Boolean.Input("normal", default=True,
                                 tooltip="`true` = bake a normal map holding the source's surface detail, in the "
                                         "target's tangent space, +Y up as glTF and Blender expect."),
                io.Boolean.Input("metallic_roughness", default=True,
                                 tooltip="`true` = bake roughness into green and metalness into blue; red holds "
                                         "occlusion when that is baked too."),
                io.Boolean.Input("emissive", default=True,
                                 tooltip="`true` = bake the glow map; a source with no glow leaves the material "
                                         "unlit by it."),
                io.Boolean.Input("occlusion", default=False,
                                 tooltip="`true` = bake ambient occlusion: darkening in creases, traced with "
                                         "ao_rays rays per texel. The slowest map; minutes on a CPU at 2048."),
                io.Float.Input(
                    "cage", default=0.02, min=0.0005, max=0.5, step=0.0005,
                    tooltip="How far from the target's surface the source is searched along each normal, as a "
                            "share of the model's size. 0.02 = 2 cm on a 1 m model; raise it for gaps, lower it "
                            "where thin parts pick up the wrong side.",
                ),
                io.Int.Input(
                    "samples", default=1, min=1, max=mesh_bake.MAX_SAMPLES, step=1,
                    tooltip="Samples along each side of a texel: 1 = one per texel; 2 = four, smoother edges "
                            "and fine detail at four times the time; 4 = sixteen.",
                ),
                io.Int.Input(
                    "padding", default=8, min=0, max=64, step=1,
                    tooltip="Pixels each chart's colours are grown into the empty space around it, hiding "
                            "seams when the texture is filtered or shrunk. 8 = usual; 16 for small mipmaps.",
                ),
                io.Int.Input(
                    "ao_rays", default=16, min=1, max=mesh_bake.MAX_AO_RAYS, step=1,
                    tooltip="Occlusion rays per texel when occlusion is on. 16 = quick and slightly grainy; "
                            "64 = smooth; time grows in step.",
                ),
                io.Float.Input(
                    "ao_distance", default=0.1, min=0.001, max=1.0, step=0.001, advanced=True,
                    tooltip="Farthest surface that darkens a texel, as a share of the model's size. 0.1 = "
                            "nearby folds and limbs; 0.02 = only tight creases.",
                ),
                io.String.Input("parts", default="", advanced=True, tooltip="Parts to work on: comma-separated names or labels, wildcards allowed, such as `head, hair*`. Empty works on every part."),
            ],
            outputs=[
                MODEL_3D.Output("model_3d", tooltip="The whole model: parts with their materials, and the skeleton, skin, springs, expressions and animations it carries."),
                io.Mesh.Output("mesh", tooltip="The same model as one mesh for core's mesh nodes and Save 3D Model. Parts are merged, their textures packed into one atlas, and the skeleton left out."),
                io.Image.Output("base_color", tooltip="The baked colour map, as the picture an image node can edit or save."),
                io.Image.Output("normal", tooltip="The baked normal map, flat blue (0.5, 0.5, 1) where the surfaces match."),
                io.Image.Output("metallic_roughness", tooltip="The baked map with occlusion in red, roughness in green and metalness in blue."),
                io.Image.Output("emissive", tooltip="The baked glow map, black where nothing glows."),
                io.Image.Output("occlusion", tooltip="The baked ambient occlusion, white where open and dark in creases."),
                io.String.Output(
                    "report",
                    tooltip="Maps baked, texels filled, the share of samples that found the source and the time "
                            "each stage took.",
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, model_3d, source=None, resolution=2048, base_color=True, normal=True,
                metallic_roughness=True, emissive=True, occlusion=False, cage=0.02, samples=1, padding=8,
                ao_rays=16, ao_distance=0.1, parts="") -> io.NodeOutput:
        """Bake into the chosen parts, which share one texture layout.

        Raises:
            ValueError: Nothing is wired, no part matches ``parts``, a chosen part has no texture
                coordinates, there is nothing to bake from, or every map is switched off.
        """
        from comfy_execution.graph_utils import ExecutionBlocker

        model = inputs.to_model(model_3d)
        before = report.figures(model)
        chosen = selection.selected(model, parts)
        if not chosen:
            raise ValueError(f"no part matches '{str(parts)[:80]}'; the model's parts are {edit.part_names(model)}. "
                             "Leave parts empty to bake into every part.")
        chosen = [index for index in chosen if model.parts[index].triangle_count]
        if not chosen:
            raise ValueError("the chosen parts have no triangles to bake onto; wire a model with faces or choose "
                             "other parts")
        bare = [model.parts[index].name for index in chosen if model.parts[index].uvs is None]
        if bare:
            raise ValueError(f"part(s) {', '.join(bare[:6])} have no texture coordinates to bake into; run Mesh UV "
                             "Unwrap on the model first")
        switches = {"base_color": base_color, "normal": normal, "metallic_roughness": metallic_roughness,
                    "emissive": emissive, "occlusion": occlusion}
        maps = tuple(name for name in mesh_bake.MAPS if switches[name])
        if not maps:
            raise ValueError("every map is switched off; turn on base_color, normal, metallic_roughness, emissive "
                             "or occlusion")
        donor = None
        if source is not None:
            donor = source if isinstance(source, Model3D) else inputs.to_model(source)
        settings = mesh_bake.BakeSettings(
            resolution=int(resolution), maps=maps, cage=float(cage), samples=int(samples), padding=int(padding),
            ao_rays=int(ao_rays), ao_distance=float(ao_distance))

        bar = progress_bar(PROGRESS_STEPS)
        shown = [0]
        try:
            from comfy.model_management import throw_exception_if_processing_interrupted
        except Exception:
            throw_exception_if_processing_interrupted = None

        def advance(fraction: float) -> None:
            if throw_exception_if_processing_interrupted is not None:
                throw_exception_if_processing_interrupted()
            step = int(round(min(max(fraction, 0.0), 1.0) * PROGRESS_STEPS))
            if step > shown[0]:
                bar.update(step - shown[0])
                shown[0] = step

        baked = mesh_bake.bake([model.parts[index] for index in chosen], donor, settings, progress=advance)
        for index in chosen:
            part = model.parts[index]
            part.material = baked.material
            if baked.material.normal_texture is not None:
                model.parts[index] = tangents.with_tangents(part)

        stats = baked.stats
        lines = [
            f"Mesh Bake Textures: {', '.join(maps)} at {int(resolution)} pixels from "
            f"{'the source' if donor is not None else 'the textures before the unwrap'}",
            f"  {stats['texels']:,} texels filled ({100.0 * stats['coverage']:.1f}% of the texture), "
            f"{stats['samples']:,} samples, {100.0 * stats['hit_share']:.2f}% found the source",
            f"  {stats['prepare_seconds']:.1f} s preparing, {stats['sample_seconds']:.1f} s sampling, "
            f"{stats['seconds']:.1f} s in all",
        ]
        if stats["missed"]:
            lines.append(f"  {stats['missed']:,} samples found nothing within the cage and were filled from "
                         "their neighbours; raise cage if the textures show gaps")
        text = "\n".join(lines)
        report.publish(
            model, f"baked {len(maps)} map{'s' if len(maps) != 1 else ''} at {int(resolution)} pixels", before=before,
            counts={"texels": stats["texels"], "found source %": round(100.0 * stats["hit_share"], 2),
                    "seconds": round(stats["seconds"], 2)},
            facts={"maps": ", ".join(maps), "from": "source" if donor is not None else "before unwrap",
                   "parts": edit.part_names(model, 8)},
        )
        images = []
        for name in mesh_bake.MAPS:
            if name in baked.images:
                images.append(_image(baked.images[name]))
            else:
                images.append(ExecutionBlocker(f"{name} was not baked; switch {name} on in Mesh Bake Textures"))
        hidden = cls.hidden
        wanted_mesh = inputs.mesh_wanted(getattr(hidden, "prompt", None), getattr(hidden, "unique_id", None))
        return io.NodeOutput(*inputs.node_output(model, mesh=wanted_mesh), *images, text)
