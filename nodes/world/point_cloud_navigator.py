"""Fly through a point cloud on the node and keep the flight, or placed keys, as a camera path."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ...modules.compat.types import LOAD3D_CAMERA, SPLAT
from ...modules.interface import three_asset
from ...modules.world import cameras, cloud as clouds, gaussians, navigator


class PointCloudNavigator(io.ComfyNode):
    """Show a point cloud to fly through and answer the path flown."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASPointCloudNavigator",
            display_name="Point Cloud Navigator",
            search_aliases=[
                "WASPointCloudNavigator", "Point Cloud Navigator", "fly through", "record camera", "camera keys",
                "camera rig", "nerfstudio camera path", "chan camera", "walk through point cloud",
            ],
            category="WAS Suite/3D/Point Cloud",
            description=(
                "Fly a camera through a point cloud on the node, recording the flight or placing keys, and "
                "import or export camera rigs. Hands on one camera per frame for Point Cloud Render. Run "
                "once to see the cloud."
            ),
            inputs=[
                SPLAT.Input("point_cloud", tooltip="The cloud to fly through, from Point Cloud From Depth, Point Cloud Refine or a splat node. A fitted splat is drawn as splats."),
                LOAD3D_CAMERA.Input(
                    "camera_info", optional=True,
                    tooltip="Where the navigator opens and the camera held while no path is stored, such as Point "
                            "Cloud From Depth's camera. Unwired = at the origin looking down -Z.",
                ),
                io.Int.Input(
                    "frames", default=158, min=0, max=2000, step=1,
                    tooltip="Cameras to make along the path. 158 = 6.6 s at 24 fps; 0 = the path's own length at "
                            "fps, as flown.",
                ),
                io.Combo.Input(
                    "easing", options=list(cameras.EASINGS), default="linear",
                    tooltip="Speed between placed keys. `linear` = one smooth glide; `ease in and out` = settles on "
                            "each key. A recording always plays as flown.",
                ),
                io.Float.Input(
                    "fps", default=24.0, min=1.0, max=240.0, step=0.001, advanced=True,
                    tooltip="Frames a second when frames is 0. 24 = H3's rate.",
                ),
                io.String.Input(
                    "path", default="", advanced=True,
                    tooltip="The flight or keys, written by the navigator as JSON in world space. Empty = hold "
                            "camera_info.",
                ),
            ],
            outputs=[
                LOAD3D_CAMERA.Output(
                    display_name="camera_info",
                    tooltip="One camera per frame, for Point Cloud Render or core's Render Splat.",
                ),
                io.Int.Output(display_name="frames", tooltip="How many cameras the path holds."),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, point_cloud, camera_info=None, frames=158, easing="linear", fps=24.0,
                path="") -> io.NodeOutput:
        """Read the stored path, hand on its cameras and send the cloud to the panel.

        Raises:
            ValueError: The stored path cannot be read.
        """
        given = cameras.as_list(camera_info)
        start = given[-1] if given else cameras.source(cameras.DEFAULT_FOV)
        stored = navigator.read_path(path)
        cloud = clouds.shown_cloud(point_cloud)
        out = navigator.cameras_for(stored, start, int(frames), float(fps), easing)
        dots = gaussians.drawn_as_dots(point_cloud)
        splats = None if dots else gaussians.from_splat(point_cloud)
        body, shown = navigator.pack(cloud) if dots else navigator.pack_splats(splats)
        spread = float(cloud.spreads.median()) if cloud.count else 0.01
        depth = clouds.scene_depth(cloud, start)
        sample = cloud.positions[:: max(1, cloud.count // 100_000)].float()
        centre = (sample.median(dim=0).values * torch.tensor([1.0, -1.0, -1.0])).tolist() if cloud.count else [0, 0, -1]
        panel = {
            "asset": "%s?key=%s" % (three_asset.ROUTE, three_asset.keep(body, "application/octet-stream")),
            "points": cloud.count if dots else splats.count,
            "style": "dots" if dots else "splats",
            "shown": shown,
            "spread": spread,
            "depth": depth,
            "centre": centre,
            "start": start,
            "frames": len(out),
            "keys": len(stored.keys) if stored is not None else 0,
            "kind": stored.kind if stored is not None else "",
        }
        return io.NodeOutput(out, len(out), ui={"was_point_cloud": [panel]})
