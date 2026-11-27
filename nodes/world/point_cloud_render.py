"""Draw a point cloud from each camera of a path as colour, depth and the holes left to fill."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ...modules.compat import limits
from ...modules.compat.types import LOAD3D_CAMERA, SPLAT
from ...modules.image import sizing
from ...modules.interface.progress import progress_bar
from ...modules.mesh.device import compute_device
from ...modules.world import cameras, render, report


class PointCloudRender(io.ComfyNode):
    """Render a point cloud along a camera path."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASPointCloudRender",
            display_name="Point Cloud Render",
            search_aliases=[
                "WASPointCloudRender", "Point Cloud Render", "render point cloud", "novel view", "warp",
                "render splat depth", "holes mask", "camera path render", "reproject",
            ],
            category="WAS Suite/3D/Point Cloud",
            description=(
                "Draw a point cloud from every camera of a path, giving colour, depth and a mask of the holes "
                "nothing covers. Feed the depth to H3 Structure Video as near depth, or the colour and holes "
                "to an inpainting model, to repaint each view whole."
            ),
            inputs=[
                SPLAT.Input(
                    "point_cloud",
                    tooltip="The cloud, from Point Cloud From Depth, Point Cloud Refine or any splat node.",
                ),
                LOAD3D_CAMERA.Input(
                    "camera_info",
                    tooltip="One camera or a path, from Point Cloud Camera Path, Point Cloud From Depth or core's "
                            "camera nodes. One picture per camera; a cloud holding a video's moving people "
                            "plays them across the path.",
                ),
                io.Int.Input(
                    "width", default=1344, min=16, max=limits.max_resolution(), step=8,
                    tooltip="Picture width in pixels. 1344 x 768 = H3 Structure Video's trained landscape size.",
                ),
                io.Int.Input(
                    "height", default=768, min=16, max=limits.max_resolution(), step=8,
                    tooltip="Picture height in pixels. 768 = H3's landscape height; 1344 = portrait. The field of "
                            "view spans the shorter side.",
                ),
                io.Float.Input(
                    "point_size", default=1.0, min=0.1, max=8.0, step=0.05,
                    tooltip="Size of every dot or splat. 1.0 = a lifted cloud's dots just close the gaps between "
                            "neighbours, a fitted splat as fitted; higher fills cracks seen close up but thickens "
                            "edges.",
                ),
                io.Int.Input(
                    "hole_grow", default=2, min=0, max=64, step=1,
                    tooltip="Pixels each hole is widened by in the holes mask, taking in the ragged edges round "
                            "it. 0 = exactly where nothing landed.",
                ),
                io.Combo.Input(
                    "depth_style", options=list(render.DEPTH_STYLES), default="inverse",
                    tooltip="`inverse` = most of the grey range spent near the camera, as depth estimators draw "
                            "it; `linear` = evenly by distance. White is nearest, black is empty.",
                ),
                io.Float.Input(
                    "background", default=0.0, min=0.0, max=1.0, step=0.05, advanced=True,
                    tooltip="Grey of the holes in the colour pictures. 0 = black; 0.5 = mid grey.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="images", tooltip="The colour of each view, holes in the background grey."),
                io.Image.Output(
                    display_name="depth",
                    tooltip="Each view's depth on one scale for the whole path: white nearest, black where empty.",
                ),
                io.Mask.Output(display_name="holes", tooltip="1 where nothing covers the view and a model has to paint, 0 elsewhere."),
            ],
        )

    @classmethod
    def execute(cls, point_cloud, camera_info, width=1344, height=768, point_size=1.0, hole_grow=2,
                depth_style="inverse", background=0.0) -> io.NodeOutput:
        """Render every camera and publish the first, middle and last views with the hole figures.

        Raises:
            ValueError: No camera is wired, or the pictures would pass the batch size limit.
        """
        path = cameras.as_list(camera_info)
        if not path:
            raise ValueError("camera_info holds no camera; wire Point Cloud Camera Path or Point Cloud From Depth")
        width, height = int(width), int(height)
        sizing.check_canvas(width, height, "Each view", "Lower width or height.")
        most = max(1, render.MAX_PATH_PIXELS // (width * height))
        if len(path) > most:
            raise ValueError(f"{len(path)} views at {width}x{height} pass the {most} this size allows in one "
                             "batch; lower width and height or use a shorter path")
        device = compute_device()
        scene = render.Scene(point_cloud, device)
        try:
            from comfy.model_management import throw_exception_if_processing_interrupted
        except Exception:
            throw_exception_if_processing_interrupted = None
        bar = progress_bar(len(path))
        colours, depths = [], []
        for index, camera in enumerate(path):
            if throw_exception_if_processing_interrupted is not None:
                throw_exception_if_processing_interrupted()
            frame = scene.draw(camera, width, height, float(point_size), float(background),
                               scene.frame_for(index, len(path)))
            colours.append(frame.colour.cpu())
            depths.append(frame.depth.cpu())
            bar.update(1)
        greys, near, far = render.depth_pictures(depths, depth_style if depth_style in render.DEPTH_STYLES else "inverse")
        empty = torch.stack([~torch.isfinite(d) for d in depths]).float()
        holes = render.grow_holes(empty, int(hole_grow))
        images = torch.stack(colours)
        depth = greys[..., None].expand(-1, -1, -1, 3).contiguous()
        picks = sorted({0, len(path) // 2, len(path) - 1})
        share = holes.mean(dim=(1, 2))
        report.publish([images[i] for i in picks], f"{len(path)} views at {width}x{height}", {
            "views": len(path),
            "splats" if not scene.dots else "points": scene.count,
            "holes % mean": round(100.0 * float(share.mean()), 2),
            "holes % most": round(100.0 * float(share.max()), 2),
            "nearest": round(near, 4),
            "farthest": round(far, 4),
        }, {"views shown": "first, middle and last",
            **({"moving": f"{len(scene.moving)} frames of moving points spread over the path"} if scene.moving else {})})
        return io.NodeOutput(images, depth, holes)
