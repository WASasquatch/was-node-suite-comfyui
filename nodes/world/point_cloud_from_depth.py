"""Lift pictures and their estimated depth into a coloured point cloud, or grow one with them."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import DA3_GEOMETRY, LOAD3D_CAMERA, MOGE_GEOMETRY, SPLAT
from ...modules.interface.progress import progress_bar
from ...modules.mesh.device import compute_device
from ...modules.world import cameras, cloud as clouds, render, report

#: Ways new points join a cloud, the first being the default.
ADD_MODES = ("holes", "everything")

#: Degrees each preview view is turned from the first camera.
PREVIEW_TURN = 12.0

#: Longest side of a preview view, in pixels.
PREVIEW_SIDE = 640


class PointCloudFromDepth(io.ComfyNode):
    """Turn depth estimates into a point cloud."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASPointCloudFromDepth",
            display_name="Point Cloud From Depth",
            search_aliases=[
                "WASPointCloudFromDepth", "Point Cloud From Depth", "depth to point cloud", "image to 3d",
                "lift depth", "picture to world", "moge point cloud", "depth anything point cloud",
                "grow point cloud", "splat from depth",
            ],
            category="WAS Suite/3D/Point Cloud",
            description=(
                "Lift a picture, or a video's frames placed by DA3 Inference in multi-view mode, into a "
                "coloured point cloud to fly a camera through. Each later picture, and each picture added to "
                "a wired point cloud, is scaled to meet it and fills only what it lacked. A video's moving "
                "people can be held frame by frame."
            ),
            inputs=[
                io.MultiType.Input(
                    "geometry", types=[MOGE_GEOMETRY, DA3_GEOMETRY],
                    tooltip="Depth for each picture, from MoGe Inference or DA3 Inference. DA3 in multi-view mode "
                            "also places each picture's camera.",
                ),
                io.Image.Input(
                    "image", optional=True,
                    tooltip="Colours for the points, one picture per depth map. Unwired = the picture the depth "
                            "was estimated from.",
                ),
                LOAD3D_CAMERA.Input(
                    "camera_info", optional=True,
                    tooltip="Where each picture was taken, such as the path a render followed. Unwired = the "
                            "first picture's camera at the origin, or DA3's own cameras.",
                ),
                SPLAT.Input(
                    "point_cloud", optional=True,
                    tooltip="A cloud to grow. Each picture's depth is scaled to meet it before its points are "
                            "added. Unwired = a new cloud.",
                ),
                io.Int.Input(
                    "stride", default=1, min=1, max=16, step=1,
                    tooltip="1 = a point per pixel; 2 = every second pixel each way, a quarter of the points. "
                            "Raise it for large or many pictures. Holes filled in a wired point_cloud take "
                            "every pixel.",
                ),
                io.Float.Input(
                    "edge_cut", default=0.04, min=0.0, max=1.0, step=0.005,
                    tooltip="Drops pixels whose depth jumps by more than this share to a neighbour, the smeared "
                            "points between a subject and what is behind it. 0 = keep all; 0.04 = typical.",
                ),
                io.Combo.Input(
                    "add", options=list(ADD_MODES), default="holes", advanced=True,
                    tooltip="With point_cloud wired: `holes` = add points only where the cloud showed nothing "
                            "from that camera; `everything` = add every point.",
                ),
                io.Combo.Input(
                    "motion", options=list(clouds.MOTIONS), default=clouds.MOTIONS[0],
                    tooltip="For a video: `still scene` = every frame joins one cloud; `find what moves` = whatever "
                            "crosses the view is held per frame; `each frame apart` = every frame keeps only its "
                            "own points, the steadiest choice for reshooting a video.",
                ),
                io.Mask.Input(
                    "moving", optional=True,
                    tooltip="What moves in each picture, 1 = moving, such as a tracked person mask per frame. "
                            "Held per frame whatever motion says. Unwired = found from the depth when motion asks.",
                ),
            ],
            outputs=[
                SPLAT.Output(
                    display_name="point_cloud",
                    tooltip="The coloured points as a splat, for Point Cloud Render, Point Cloud Camera Path or "
                            "core's splat nodes.",
                ),
                LOAD3D_CAMERA.Output(
                    display_name="camera_info",
                    tooltip="The camera each picture was lifted from; one camera for one picture, a list for "
                            "several. Start a camera path here.",
                ),
            ],
        )

    @classmethod
    def execute(cls, geometry, image=None, camera_info=None, point_cloud=None, stride=1, edge_cut=0.04,
                add="holes", motion=clouds.MOTIONS[0], moving=None) -> io.NodeOutput:
        """Lift every picture, publish two turned views and the figures.

        Raises:
            ValueError: The geometry carries no depth or the cloud would be too large.
        """
        if not isinstance(geometry, dict):
            raise ValueError("geometry must come from MoGe Inference or DA3 Inference")
        device = compute_device()
        grow = clouds.from_splat(point_cloud) if point_cloud is not None else None
        poses = cameras.as_list(camera_info)
        bar = progress_bar(100)
        done = [0]

        def advance(share: float) -> None:
            step = int(share * 100)
            if step > done[0]:
                bar.update(step - done[0])
                done[0] = step

        def depth_of(cloud, camera, width, height):
            return render.render(cloud, camera, width, height, device=device).depth

        lifted = clouds.lift(geometry, image, poses, grow, int(stride), float(edge_cut),
                             add if add in ADD_MODES else "holes", device, depth_of, advance,
                             motion if motion in clouds.MOTIONS else clouds.MOTIONS[0], moving)
        cloud = lifted.cloud
        held = lifted.moving or (clouds.moving_from_splat(point_cloud) if point_cloud is not None else [])
        first = lifted.cameras[0]
        source = geometry.get("depth") if geometry.get("depth") is not None else geometry.get("points")
        height, width = int(source.shape[1]), int(source.shape[2])
        scale = min(1.0, PREVIEW_SIDE / max(width, height))
        view_w, view_h = max(16, round(width * scale)), max(16, round(height * scale))
        focus = clouds.scene_depth(cloud if cloud.count or not held else held[0], first)
        views = [cameras.move(first, name, 2, 0.0, PREVIEW_TURN, focus)[-1] for name in ("orbit left", "orbit right")]
        shown = cloud.joined(held[len(held) // 2].to(device)) if held else cloud
        pictures = [render.render(shown, view, view_w, view_h, device=device).colour for view in views]
        counts = {"points": cloud.count, "pictures": len(lifted.cameras),
                  "added": sum(lifted.added), "scene depth": round(focus, 3)}
        facts = {"field of view": f"{first.get('fov', 0):.1f} degrees"}
        if sum(lifted.unsure):
            facts["unsure depth, dropped"] = f"{sum(lifted.unsure):,} pixels"
        if sum(lifted.seen_through):
            facts["seen through by other pictures, dropped"] = f"{sum(lifted.seen_through):,} pixels"
        if held:
            counts["moving frames"] = len(held)
            counts["moving points"] = sum(part.count for part in held)
            facts["moving"] = "held per frame; the views show the middle frame's"
        elif motion == clouds.MOTIONS[1] or moving is not None:
            facts["moving"] = "one picture has nothing to compare; lift a video's frames to find what moves"
        if grow is not None:
            counts["points before"] = grow.count
        if grow is not None or len(lifted.cameras) > 1:
            scales = [s for s, ok in zip(lifted.scales, lifted.matched) if ok]
            unmatched = len(lifted.scales) - len(scales) - (1 if grow is None or not grow.count else 0)
            facts["depth scale"] = (f"{min(scales):.3f} to {max(scales):.3f}" if scales else "no picture met the cloud")
            if unmatched:
                facts["unscaled"] = f"{unmatched} picture(s) shared too little of the cloud and kept their own depth"
        report.publish(pictures, f"{cloud.count:,} points from {len(lifted.cameras)} picture(s), seen "
                                 f"{PREVIEW_TURN:g} degrees either side", counts, facts)
        cameras_out = lifted.cameras[0] if len(lifted.cameras) == 1 else lifted.cameras
        return io.NodeOutput(clouds.to_splat(cloud.to("cpu"), [part.to("cpu") for part in held]), cameras_out)
