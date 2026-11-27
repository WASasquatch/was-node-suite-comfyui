"""A preset camera move, one camera per frame, scaled to the point cloud it moves through."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import LOAD3D_CAMERA, SPLAT
from ...modules.mesh.device import compute_device
from ...modules.world import cameras, cloud as clouds, render, report

#: Width of each preview view, in pixels.
PREVIEW_WIDTH = 448


class PointCloudCameraPath(io.ComfyNode):
    """Make a camera move as a list of cameras."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASPointCloudCameraPath",
            display_name="Point Cloud Camera Path",
            search_aliases=[
                "WASPointCloudCameraPath", "Point Cloud Camera Path", "camera move", "dolly", "push in",
                "orbit camera path", "truck", "crane", "pan", "camera animation",
            ],
            category="WAS Suite/3D/Point Cloud",
            description=(
                "Make a camera move, such as a push in, truck, crane, pan or orbit, as one camera per frame. "
                "Distances follow the point cloud's depth, so one setting suits any scene. Feed Point Cloud "
                "Render or core's Render Splat."
            ),
            inputs=[
                io.Combo.Input(
                    "move", options=[*cameras.MOVES, cameras.FOLLOW, cameras.CONTINUE], default="push in",
                    tooltip="`push in`/`pull out` = forward or back; `truck` = sideways; `crane` = up or down; "
                            "`orbit` = circle the subject; `follow the wired cameras` = a steady path through "
                            "camera_info; `carry on the wired turn` = orbit on past a video's end, its way.",
                ),
                io.Int.Input(
                    "frames", default=158, min=1, max=2000, step=1,
                    tooltip="Cameras to make, one per frame. 158 = 6.6 s at 24 fps, H3 Structure Video's trained "
                            "length.",
                ),
                io.Float.Input(
                    "travel", default=0.25, min=0.0, max=10.0, step=0.01,
                    tooltip="Distance moved, as a share of how far the scene sits from the camera. 0.25 = a quarter "
                            "of the way in; 1.0 = all the way.",
                ),
                io.Float.Input(
                    "angle", default=20.0, min=0.0, max=360.0, step=0.5,
                    tooltip="Degrees a pan, tilt or orbit turns through. 20 = a gentle move; 90 = a quarter turn.",
                ),
                io.Combo.Input(
                    "easing", options=list(cameras.EASINGS), default="ease in and out",
                    tooltip="How the move speeds up and slows down. `ease in and out` = starts and stops gently; "
                            "`linear` = constant speed.",
                ),
                SPLAT.Input(
                    "point_cloud", optional=True,
                    tooltip="The cloud the camera moves through, which sets the scale of travel and the centre "
                            "of an orbit. Unwired = the scene is taken as 1 unit away.",
                ),
                LOAD3D_CAMERA.Input(
                    "camera_info", optional=True,
                    tooltip="Where the move starts: a camera, or a path it carries on from or rides along, as "
                            "wired_path sets. Unwired = at the origin looking down -Z.",
                ),
                io.Int.Input(
                    "smooth", default=2, min=0, max=32, step=1, advanced=True,
                    tooltip="For a followed or ridden path: neighbours each side every wired camera is averaged "
                            "with. 0 = exactly through them; 2 = steadies a shaky estimate; 8 = a slow glide. "
                            "Always 0 for a cloud whose frames are kept apart.",
                ),
                io.Combo.Input(
                    "wired_path", options=list(cameras.WIRED), default=cameras.WIRED[0],
                    tooltip="With a list of cameras wired: `carry on from its end` = the move starts at the last "
                            "camera; `ride along it` = the move is added to each camera in turn, reshooting a "
                            "video from an angle that drifts away from its own.",
                ),
            ],
            outputs=[
                LOAD3D_CAMERA.Output(
                    display_name="camera_info",
                    tooltip="One camera per frame, for Point Cloud Render, core's Render Splat or a next "
                            "Point Cloud Camera Path.",
                ),
                io.Int.Output(display_name="frames", tooltip="How many cameras the path holds."),
            ],
        )

    @classmethod
    def execute(cls, move, frames=158, travel=0.25, angle=20.0, easing="ease in and out", point_cloud=None,
                camera_info=None, smooth=2, wired_path=cameras.WIRED[0]) -> io.NodeOutput:
        """Build the path and publish its first, middle and last views when a cloud is wired.

        Raises:
            ValueError: The move is not a preset, or there are too few cameras to follow.
        """
        given = cameras.as_list(camera_info)
        if move == cameras.CONTINUE:
            move = cameras.turn_direction(given)
            wired_path = cameras.WIRED[0]
        ridden = wired_path == cameras.WIRED[1] and len(given) > 1 and move != cameras.FOLLOW
        start = given[0 if ridden else -1] if given else cameras.source(cameras.DEFAULT_FOV)
        cloud = clouds.shown_cloud(point_cloud) if point_cloud is not None else None
        depth = clouds.scene_depth(cloud, start) if cloud is not None and cloud.count else 1.0
        if point_cloud is not None and clouds.moving_counts(point_cloud) and not clouds.from_splat(point_cloud).count:
            smooth = 0
        if move == cameras.FOLLOW:
            path = cameras.follow(given, int(frames), int(smooth))
        elif ridden:
            path = cameras.along(given, move, int(frames), float(travel) * depth, float(angle), depth, easing,
                                 int(smooth))
        else:
            path = cameras.move(start, move, int(frames), float(travel) * depth, float(angle), depth, easing,
                                cameras.upright(given))
        pictures = []
        if cloud is not None and cloud.count:
            device = compute_device()
            scene = render.Scene(point_cloud, device)
            height = round(PREVIEW_WIDTH * 9 / 16)
            picks = sorted({0, len(path) // 2, len(path) - 1})
            pictures = [scene.draw(path[i], PREVIEW_WIDTH, height, frame=scene.frame_for(i, len(path))).colour
                        for i in picks]
        first, last = cameras.pose(path[0])[0], cameras.pose(path[-1])[0]
        report.publish(pictures, f"{move}{', riding the wired path' if ridden else ''}, {len(path)} cameras", {
            "frames": len(path),
            "distance": round(float((last - first).norm()), 4),
            "scene depth": round(depth, 4),
        }, {"views": "first, middle and last" if pictures else "wire point_cloud to see the views"})
        return io.NodeOutput(path, len(path))
