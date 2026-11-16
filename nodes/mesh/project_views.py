"""Project edited pictures of a model back into one of its texture maps, through the cameras they were rendered from."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ...modules.compat.types import (
    CAMERA_VIEWS, FILE_3D, FILE_3D_FBX, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_OBJ, MODEL_3D,
)
from ...modules.interface import preview
from ...modules.interface.progress import progress_bar
from ...modules.mesh import project as mesh_project
from ...modules.mesh.views import Views
from ...modules.model3d import edit, inputs, report, selection

#: Steps the progress bar is divided into.
PROGRESS_STEPS = 100

#: Map -> where it is written, as the report names it.
WRITTEN_TO = {
    "base colour": "base colour map",
    "roughness": "green of the metallic-roughness map",
    "metallic": "blue of the metallic-roughness map",
    "normal": "tangent-space normal map",
    "occlusion": "red of the occlusion map",
}


class MeshProjectViews(io.ComfyNode):
    """Project pictures back into one of a model's texture maps."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASMeshProjectViews",
            display_name="Mesh Project Views",
            search_aliases=[
                "WASMeshProjectViews", "Mesh Project Views", "project views", "texture projection",
                "project texture", "reproject", "upscale texture", "enhance texture", "repaint texture",
                "multiview texture", "project normal map", "project roughness", "pbr textures",
            ],
            category="WAS Suite/3D/Mesh",
            description=(
                "Project pictures rendered by Mesh Render Views, after any image model has enhanced, upscaled "
                "or repainted them, back into one of the model's maps: base colour, roughness, metallic, normal "
                "or occlusion. Each texel takes most from the pictures facing it squarely and keeps its old "
                "value where no picture sees it."
            ),
            inputs=[
                io.MultiType.Input("model_3d", types=[MODEL_3D, io.Mesh, FILE_3D, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_FBX, FILE_3D_OBJ], tooltip="The model to work on: a 3D model from another 3D node, a mesh from core's mesh nodes, or a GLB, glTF, VRM, FBX or OBJ file from Load 3D or a 3D generator such as Tripo."),
                CAMERA_VIEWS.Input("views", tooltip="The cameras from the Mesh Render Views that rendered the pictures."),
                io.Image.Input(
                    "images",
                    tooltip="The edited pictures, one per camera in the order they were rendered. Any size of the "
                            "same shape: upscaled pictures give a sharper texture.",
                ),
                io.Mask.Input(
                    "masks", optional=True,
                    tooltip="How much of each picture to use: 1 = use, 0 = ignore. One per picture, or one for all. "
                            "Unwired = all of every picture.",
                ),
                io.Int.Input(
                    "texture_size", default=0, min=0, max=mesh_project.MAX_TEXTURE, step=64,
                    tooltip="Width and height of the new map, in pixels. 0 = keep each map's size, or the colour "
                            "map's for a new one; 4096 with upscaled pictures for a sharper texture.",
                ),
                io.Float.Input(
                    "falloff", default=4.0, min=0.5, max=32.0, step=0.5,
                    tooltip="How strongly each texel favours the picture facing it most squarely. 4 = soft blends "
                            "between views; 16 = nearly one picture per texel, sharper where views disagree.",
                ),
                io.Float.Input(
                    "strength", default=1.0, min=0.0, max=1.0, step=0.05,
                    tooltip="How fully the pictures replace the old values where they see well. 1.0 = replace; "
                            "0.5 = half and half; 0.0 = no change.",
                ),
                io.Float.Input(
                    "edge_fade", default=0.05, min=0.0, max=0.5, step=0.01, advanced=True,
                    tooltip="Share of each picture's size at its borders over which it fades out. 0.05 = a thin "
                            "band; 0 = none.",
                ),
                io.Float.Input(
                    "outline_fade", default=0.01, min=0.0, max=0.1, step=0.005, advanced=True,
                    tooltip="Fade in from the model's outline and from every edge where one surface passes in "
                            "front of another, as a share of each picture's size. 0.01 = 10 pixels at 1024; "
                            "0 = none.",
                ),
                io.Float.Input(
                    "depth_tolerance", default=0.004, min=0.0, max=0.1, step=0.001, advanced=True,
                    tooltip="How far behind the nearest surface a point still counts as seen, as a share of the "
                            "model's size. 0.004 = 4 mm on a 1 m model; raise it where thin surfaces miss, lower "
                            "it where colour bleeds through.",
                ),
                io.Int.Input(
                    "padding", default=8, min=0, max=64, step=1, advanced=True,
                    tooltip="Pixels each chart's colours are grown into the empty space around it, hiding seams "
                            "when the texture is filtered or shrunk. 8 = usual.",
                ),
                io.String.Input("parts", default="", advanced=True, tooltip="Parts to work on: comma-separated names or labels, wildcards allowed, such as `head, hair*`. Empty works on every part."),
                io.Combo.Input(
                    "map", options=list(mesh_project.MAPS), default="base colour",
                    tooltip="Which map the pictures are written into. `base colour` = colour pictures; "
                            "`roughness`, `metallic`, `occlusion` = greyscale pictures; `normal` = camera-space "
                            "normal pictures as Mesh Render Views and Power Preprocessor draw them.",
                ),
                io.Combo.Input(
                    "encoding", options=list(mesh_project.ENCODINGS), default="gamma 2.2", advanced=True,
                    tooltip="How greyscale pictures hold their values. `gamma 2.2` = raised to 1/2.2 for "
                            "showing, as Power Preprocessor's roughness and metallicity are; `linear` = the "
                            "value itself. Read by `roughness`, `metallic` and `occlusion` only.",
                ),
            ],
            outputs=[
                MODEL_3D.Output("model_3d", tooltip="The whole model: parts with their materials, and the skeleton, skin, springs, expressions and animations it carries."),
                io.Mesh.Output("mesh", tooltip="The same model as one mesh for core's mesh nodes and Save 3D Model. Parts are merged, their textures packed into one atlas, and the skeleton left out."),
                io.Image.Output(
                    "texture",
                    tooltip="The first map written, as stored: colours for `base colour`; roughness in green and "
                            "metallic in blue for those two; the tangent-space normal map; occlusion in red.",
                ),
                io.Mask.Output("coverage", tooltip="How fully each texel of that map took the pictures' values: 1 = fully, 0 = kept its own."),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, model_3d, views, images, masks=None, texture_size=0, falloff=4.0, strength=1.0,
                edge_fade=0.05, outline_fade=0.01, depth_tolerance=0.004, padding=8, parts="",
                map="base colour", encoding="gamma 2.2") -> io.NodeOutput:
        """Project the pictures into the chosen parts and publish the first new map.

        Raises:
            ValueError: Nothing is wired, no part matches ``parts``, a chosen part has no texture
                coordinates, the pictures do not match the views, or the map is unknown.
        """
        if not isinstance(views, Views):
            raise ValueError("views must come from Mesh Render Views; wire its views output here")
        if map not in mesh_project.MAPS:
            raise ValueError(f"map '{str(map)[:40]}' is not one of {', '.join(mesh_project.MAPS)}")
        if encoding not in mesh_project.ENCODINGS:
            encoding = "gamma 2.2"
        model = inputs.to_model(model_3d)
        before = report.figures(model)
        chosen = selection.selected(model, parts)
        if not chosen:
            raise ValueError(f"no part matches '{str(parts)[:80]}'; the model's parts are {edit.part_names(model)}. "
                             "Leave parts empty to project onto every part.")
        chosen = [index for index in chosen if model.parts[index].triangle_count]
        bare = [model.parts[index].name for index in chosen if model.parts[index].uvs is None]
        if bare:
            raise ValueError(f"part(s) {', '.join(bare[:6])} have no texture coordinates to project into; run Mesh "
                             "UV Unwrap on the model first")
        if not chosen:
            raise ValueError("the chosen parts have no triangles to project onto; wire a model with faces")
        settings = mesh_project.ProjectSettings(
            texture_size=int(texture_size), falloff=float(falloff), edge_fade=float(edge_fade),
            outline_fade=float(outline_fade), strength=float(strength), depth_tolerance=float(depth_tolerance),
            padding=int(padding), map=map, encoding=encoding)
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

        mask_batch = None
        if masks is not None:
            mask_batch = masks if masks.dim() == 3 else masks.reshape(-1, masks.shape[-2], masks.shape[-1])
        done = mesh_project.project(model, chosen, views, images.float().cpu(),
                                    None if mask_batch is None else mask_batch.float().cpu(), settings,
                                    progress=advance)
        texture = done.textures[0][..., :3][None].contiguous()
        preview.publish_output(texture)
        stats = done.stats
        shares = stats["view shares"]
        facts = {"map": f"{map}, written to the {WRITTEN_TO[map]}",
                 "views": ", ".join(f"{name} {100.0 * share:.0f}%" for name, share in shares.items()),
                 "parts": edit.part_names(model, 8)}
        if map in ("roughness", "metallic", "occlusion"):
            facts["encoding"] = encoding
        report.publish(
            model, f"projected {len(views)} views into {stats['maps']} {map} "
                   f"map{'s' if stats['maps'] != 1 else ''}",
            before=before,
            counts={"texels": stats["texels"], "seen %": round(100.0 * stats["seen share"], 1),
                    "seconds": round(stats["seconds"], 2)},
            facts=facts,
        )
        coverage = done.coverage[None].contiguous() if done.coverage is not None else torch.zeros(1, 64, 64)
        hidden = cls.hidden
        wanted_mesh = inputs.mesh_wanted(getattr(hidden, "prompt", None), getattr(hidden, "unique_id", None))
        return io.NodeOutput(*inputs.node_output(model, mesh=wanted_mesh), texture, coverage)
