"""Render a model from cameras set round it, for an image model to work on and project back."""

from __future__ import annotations

import math

import torch
from comfy_api.latest import io

from ...modules.compat.types import (
    CAMERA_VIEWS, FILE_3D, FILE_3D_FBX, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_OBJ, MODEL_3D,
)
from ...modules.interface import preview
from ...modules.interface.progress import progress_bar
from ...modules.mesh import views as mesh_views
from ...modules.model3d import edit, inputs, report

#: Steps the progress bar is divided into.
PROGRESS_STEPS = 100

#: Widest contact sheet drawn on the node, in pixels.
SHEET_WIDTH = 1536


def _sheet(images: torch.Tensor) -> torch.Tensor:
    """The pictures tiled into one ``(1, H, W, 3)`` contact sheet."""
    count = images.shape[0]
    columns = math.ceil(math.sqrt(count))
    rows = math.ceil(count / columns)
    side = max(16, min(int(images.shape[2]), SHEET_WIDTH // columns))
    tiles = torch.nn.functional.interpolate(images.permute(0, 3, 1, 2), size=(side, side), mode="bilinear",
                                            align_corners=False, antialias=True)
    sheet = torch.full((3, rows * side, columns * side), 0.12)
    for index in range(count):
        row, column = divmod(index, columns)
        sheet[:, row * side:(row + 1) * side, column * side:(column + 1) * side] = tiles[index]
    return sheet.permute(1, 2, 0)[None].clamp(0, 1)


class MeshRenderViews(io.ComfyNode):
    """Render a model from a set of cameras."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASMeshRenderViews",
            display_name="Mesh Render Views",
            search_aliases=[
                "WASMeshRenderViews", "Mesh Render Views", "render views", "multiview", "turntable render",
                "texture projection", "render model", "views for texture", "depth render", "normal render",
            ],
            category="WAS Suite/3D/Mesh",
            description=(
                "Render a model's colour, depth and normals from cameras set round it, as pictures any image "
                "model can enhance, upscale or repaint. Mesh Project Views takes the edited pictures and these "
                "views and projects them back into the model's texture."
            ),
            inputs=[
                io.MultiType.Input("model_3d", types=[MODEL_3D, io.Mesh, FILE_3D, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_FBX, FILE_3D_OBJ], tooltip="The model to work on: a 3D model from another 3D node, a mesh from core's mesh nodes, or a GLB, glTF, VRM, FBX or OBJ file from Load 3D or a 3D generator such as Tripo."),
                io.Combo.Input(
                    "views", options=list(mesh_views.VIEW_SETS), default="six sides",
                    tooltip="`six sides` = front, left, back, right, top and bottom; `around` = `around` cameras "
                            "circling the model; `around, top and bottom` adds those two; `front and back`; "
                            "`front` alone.",
                ),
                io.Int.Input(
                    "resolution", default=1024, min=64, max=mesh_views.MAX_RESOLUTION, step=64,
                    tooltip="Width and height of every picture, in pixels. 1024 suits most image models; render "
                            "at 1024 and upscale the pictures for a sharper texture.",
                ),
                io.Combo.Input(
                    "projection", options=list(mesh_views.PROJECTIONS), default="orthographic",
                    tooltip="`orthographic` = flat views with no perspective, each fitted to the model's outline "
                            "from its side, which project back most evenly; `perspective` = views as a camera "
                            "sees them, through `fov`, all framing the whole model.",
                ),
                io.Int.Input(
                    "around", default=8, min=1, max=mesh_views.MAX_AROUND, step=1,
                    tooltip="Cameras in an `around` set, evenly spaced from the front. 8 = every 45 degrees.",
                ),
                io.Float.Input(
                    "elevation", default=15.0, min=-89.0, max=89.0, step=1.0,
                    tooltip="Degrees an `around` set looks down on the model from. 0 = level; 30 = from above.",
                ),
                io.Float.Input(
                    "lighting", default=0.0, min=0.0, max=1.0, step=0.05,
                    tooltip="0 = the colour map as it is, the right input for projecting back; 1 = shaded by a "
                            "soft light from the upper left, which some image models read shape from.",
                ),
                io.Float.Input(
                    "background", default=1.0, min=0.0, max=1.0, step=0.05,
                    tooltip="Grey behind the model. 1.0 = white; 0.5 = mid grey; 0.0 = black.",
                ),
                io.Float.Input(
                    "fov", default=35.0, min=5.0, max=120.0, step=1.0, advanced=True,
                    tooltip="Vertical field of view of a `perspective` camera, in degrees. 35 = a portrait "
                            "lens; 60 = wide.",
                ),
                io.Float.Input(
                    "margin", default=0.05, min=0.0, max=1.0, step=0.01, advanced=True,
                    tooltip="Room left round the model in every picture, as a share of what the picture "
                            "frames. 0.05 = a close frame.",
                ),
                io.Boolean.Input(
                    "antialias", default=True, advanced=True,
                    tooltip="`true` = render at twice the size and average down, softening jagged edges.",
                ),
            ],
            outputs=[
                io.Image.Output("images", tooltip="One colour picture per camera, in the order Mesh Project Views expects them back."),
                io.Image.Output("depth", tooltip="One depth picture per camera: white nearest, black farthest and behind the model."),
                io.Image.Output("normals", tooltip="One normal map per camera, facing the camera as (0.5, 0.5, 1), +X right and +Y up."),
                io.Mask.Output("masks", tooltip="One mask per camera: 1 where the model covers the picture, 0 on the background."),
                CAMERA_VIEWS.Output("views", tooltip="The cameras, for Mesh Project Views to project the edited pictures back through."),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, model_3d, views="six sides", resolution=1024, projection="orthographic", around=8,
                elevation=15.0, lighting=0.0, background=1.0, fov=35.0, margin=0.05,
                antialias=True) -> io.NodeOutput:
        """Set the cameras, render and publish a contact sheet.

        Raises:
            ValueError: Nothing is wired, the model has no faces, or a setting is out of range.
        """
        model = inputs.to_model(model_3d)
        if not model.triangle_count:
            raise ValueError("the model has no faces to render; wire a model with triangles")
        cameras = mesh_views.rig(
            model, views if views in mesh_views.VIEW_SETS else "six sides", around=int(around),
            elevation=float(elevation), projection=projection if projection in mesh_views.PROJECTIONS
            else "orthographic", fov=float(fov), resolution=int(resolution), margin=float(margin))
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

        rendered = mesh_views.render(model, cameras, lighting=float(lighting), background=float(background),
                                     antialias=bool(antialias), progress=advance)
        preview.publish_output(_sheet(rendered.colour))
        coverage = float(rendered.mask.mean())
        report.publish(
            model, f"{len(cameras)} views at {int(resolution)} pixels, {cameras.cameras[0].projection}",
            counts={"views": len(cameras), "resolution": int(resolution),
                    "model covers %": round(100.0 * coverage, 1)},
            facts={"cameras": ", ".join(camera.name for camera in cameras.cameras),
                   "parts": edit.part_names(model, 8)},
        )
        depth = rendered.depth[..., None].expand(-1, -1, -1, 3).contiguous()
        return io.NodeOutput(rendered.colour.contiguous(), depth, rendered.normals.contiguous(),
                             rendered.mask.contiguous(), cameras)
