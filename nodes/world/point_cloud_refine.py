"""Fit a point cloud's splats to the pictures it was made from, turning it into a Gaussian splat."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ...modules.compat import limits
from ...modules.compat.types import LOAD3D_CAMERA, SPLAT
from ...modules.interface.progress import progress_bar
from ...modules.mesh import device as devices
from ...modules.world import cameras, fit as fitting, gaussians, render, report

#: Bytes each splat costs while fitting: its fields, their gradients and the optimiser's two moments.
SPLAT_BYTES = 320

#: Working memory the blending passes take besides the splats.
PASS_BYTES = 3 * 1024 ** 3


def _interrupt():
    """ComfyUI's cancel check, or None outside ComfyUI."""
    try:
        from comfy.model_management import throw_exception_if_processing_interrupted
    except Exception:
        return None
    return throw_exception_if_processing_interrupted


#: Pixels the moving points' cover is widened by before those pixels are left out of the fit.
MOVING_MARGIN = 3


class PointCloudRefine(io.ComfyNode):
    """Fit Gaussian splats to pictures from known cameras."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASPointCloudRefine",
            display_name="Point Cloud Refine",
            search_aliases=[
                "WASPointCloudRefine", "Point Cloud Refine", "gaussian splat fit", "train gaussian splat",
                "optimise splat", "optimize splat", "3dgs", "refine splat", "fit point cloud",
            ],
            category="WAS Suite/3D/Point Cloud",
            description=(
                "Turn a point cloud into a sharp Gaussian splat by fitting it to the pictures it was made from. "
                "Add repainted views from a camera path, with their cameras, to fill what the source never saw. "
                "A video's moving people are left out of the fit and kept frame by frame."
            ),
            inputs=[
                SPLAT.Input(
                    "point_cloud",
                    tooltip="The cloud to start from, from Point Cloud From Depth or an earlier Point Cloud Refine.",
                ),
                io.Image.Input(
                    "images",
                    tooltip="The pictures to match, one per camera: the source frames, repainted views, or both "
                            "batched together.",
                ),
                LOAD3D_CAMERA.Input(
                    "camera_info",
                    tooltip="Where each picture was taken, in the same order: Point Cloud From Depth's cameras for "
                            "its own pictures, the path's cameras for rendered or repainted views.",
                ),
                io.Int.Input(
                    "steps", default=2000, min=0, max=30000, step=50,
                    tooltip="Fitting steps, one picture each. 0 = the cloud unchanged; 2000 = sharp from a few "
                            "dozen pictures; 5000 = finer detail and slower.",
                ),
                io.Int.Input(
                    "size", default=768, min=64, max=limits.max_resolution(), step=16,
                    tooltip="Longest side, in pixels, the pictures are fitted at. 768 = quick; 1344 = H3's full "
                            "width and about three times slower. Never larger than the pictures.",
                ),
                io.Boolean.Input(
                    "refine_cameras", default=True,
                    tooltip="`true` = every camera after the first may turn and shift slightly to agree with the "
                            "others, sharpening estimated cameras. `false` = cameras held exactly as wired.",
                ),
                io.Int.Input(
                    "view_colour", default=1, min=0, max=3, step=1, advanced=True,
                    tooltip="How much a splat's colour may change with the viewing angle. 0 = one colour; 1 = "
                            "gentle sheen; 3 = reflections, which need many pictures from many sides.",
                ),
                io.Int.Input(
                    "add_splats", default=100, min=0, max=400, step=10,
                    tooltip="Most splats added where the match is worst, as a percent of those wired. 0 = none; "
                            "100 = up to twice as many, for finer detail from a thin cloud. Stops at four "
                            "splats per pixel of a picture.",
                ),
                io.Mask.Input(
                    "weight", optional=True,
                    tooltip="How much each pixel counts, one mask per picture or one for all: 1 = fully, 0.5 = "
                            "half, as for repainted views beside real frames; 0 = ignored. Unwired = fully.",
                ),
                io.Image.Input(
                    "repainted", optional=True,
                    tooltip="Views a model painted from new cameras, such as H3's frames along a camera path, "
                            "fitted beside the pictures to fill what they never saw.",
                ),
                LOAD3D_CAMERA.Input(
                    "repainted_cameras", optional=True,
                    tooltip="The camera of each repainted view, in order: the path the guide render followed. "
                            "Held exactly; only the pictures' cameras are refined.",
                ),
                io.Float.Input(
                    "repainted_weight", default=0.5, min=0.0, max=1.0, step=0.05,
                    tooltip="How much a repainted view counts beside a real picture. 1 = the same; 0.5 = half, "
                            "so the source wins where both saw the scene; 0 = ignored.",
                ),
            ],
            outputs=[
                SPLAT.Output(
                    display_name="point_cloud",
                    tooltip="The fitted Gaussian splat, for Point Cloud Render, the Navigator or core's splat nodes.",
                ),
                LOAD3D_CAMERA.Output(
                    display_name="camera_info",
                    tooltip="Each picture's camera, as refined. The wired cameras unchanged when refine_cameras is "
                            "off.",
                ),
                io.Image.Output(
                    display_name="images",
                    tooltip="Each picture's view drawn from the fitted splat at the fitted size, to compare with "
                            "the pictures.",
                ),
            ],
        )

    @classmethod
    def execute(cls, point_cloud, images, camera_info, steps=2000, size=768, refine_cameras=True,
                view_colour=1, add_splats=100, weight=None, repainted=None, repainted_cameras=None,
                repainted_weight=0.5) -> io.NodeOutput:
        """Fit the splats and publish a picture beside its view before and after.

        Raises:
            ValueError: The pictures, cameras and weights do not pair up, or the cloud is empty.
        """
        poses = cameras.as_list(camera_info)
        count = int(images.shape[0])
        if not poses:
            raise ValueError("camera_info holds no camera; wire Point Cloud From Depth's or a path's cameras")
        if len(poses) != count:
            raise ValueError(f"{count} pictures but {len(poses)} cameras; wire one camera per picture, in the "
                             "same order")
        extra_poses = cameras.as_list(repainted_cameras) if repainted is not None else []
        extra = int(repainted.shape[0]) if repainted is not None else 0
        if extra and len(extra_poses) != extra:
            raise ValueError(f"{extra} repainted views but {len(extra_poses)} repainted cameras; wire the camera "
                             "path the guide render followed, one camera per view")
        if extra and float(repainted_weight) <= 0.0:
            extra, extra_poses = 0, []
        start = gaussians.from_splat(point_cloud)
        if not start.count:
            raise ValueError("the point cloud is empty; check Point Cloud From Depth's output")
        pictures = fitting.shrink(images[..., :3].float(), int(size))
        height, width = int(pictures.shape[1]), int(pictures.shape[2])
        device = devices.compute_device()
        grow = max(0, int(add_splats)) / 100.0
        devices.prepare(int(start.count * (1.0 + grow)) * SPLAT_BYTES + PASS_BYTES, device)
        steps = int(steps)
        before_scene = render.Scene(point_cloud, device)
        moving = before_scene.moving
        weights = None
        if weight is not None:
            if int(weight.shape[0]) not in (1, count):
                raise ValueError(f"{int(weight.shape[0])} weight masks but {count} pictures; wire one mask for "
                                 "all or one per picture")
            weights = torch.nn.functional.interpolate(weight.float()[:, None], size=(height, width), mode="bilinear",
                                                      align_corners=False)[:, 0].clamp(0, 1)
            weights = weights.expand(count, height, width).clone()
        hidden = 0
        if moving:
            weights = torch.ones(count, height, width) if weights is None else weights
            for index in range(min(count, len(moving))):
                cover = render.grow_holes(before_scene.moving_cover(poses[index], width, height, index)[None],
                                          MOVING_MARGIN)[0] > 0.5
                weights[index][cover.cpu()] = 0.0
                hidden += int(cover.sum())
        all_pictures, all_poses, all_weights = pictures, poses, weights
        if extra:
            painted = repainted[..., :3].float()
            if painted.shape[1] != height or painted.shape[2] != width:
                painted = torch.nn.functional.interpolate(painted.permute(0, 3, 1, 2), size=(height, width),
                                                          mode="bilinear", align_corners=False,
                                                          antialias=True).permute(0, 2, 3, 1)
            all_pictures = torch.cat([pictures, painted.clamp(0, 1)])
            all_poses = list(poses) + list(extra_poses)
            base = weights if weights is not None else torch.ones(count, height, width)
            all_weights = torch.cat([base, torch.full((extra, height, width), float(repainted_weight))])
        bar = progress_bar(max(1, steps))
        done = [0]

        def advance(share: float) -> None:
            reached = round(share * steps)
            if reached > done[0]:
                bar.update(reached - done[0])
                done[0] = reached

        if steps > 0:
            result = fitting.fit(start, all_pictures, all_poses, steps, int(view_colour), bool(refine_cameras),
                                 device, progress=advance, interrupt=_interrupt(), weights=all_weights, grow=grow,
                                 movable=count)
            result.cameras = result.cameras[:count]
            fitted = gaussians.to_splat(result.gaussians, moving)
        else:
            result = fitting.Fitted(start, [dict(p) for p in poses])
            fitted = point_cloud
            bar.update(1)
        after_scene = render.Scene(fitted, device)
        before, after, drawn = [], [], []
        middle = count // 2
        shown = [pictures[middle]]
        for index in range(count):
            target = pictures[index].to(device)
            frame = index if index < len(moving) else None
            view = after_scene.draw(result.cameras[index], width, height, frame=frame).colour
            first = before_scene.draw(poses[index], width, height, frame=frame).colour
            before.append(fitting.psnr(first, target))
            after.append(fitting.psnr(view, target))
            drawn.append(view.cpu())
            if index == middle:
                shown += [first.cpu(), drawn[-1]]
        facts = {
            "views shown": "picture, before, after",
            "faint splats removed": result.removed,
            "splats added": result.added,
            "cameras moved % of depth": round(100.0 * max(result.moved or [0.0]), 3),
        }
        if extra:
            facts["repainted views"] = f"{extra} fitted at weight {float(repainted_weight):g}, cameras held"
        if moving:
            facts["moving"] = (f"{len(moving)} frames of moving points kept as they were; "
                               f"{100.0 * hidden / max(1, count * height * width):.1f}% of the pixels left out")
        report.publish(shown, f"{count} pictures, {result.steps} steps at {width}x{height}", {
            "pictures": count,
            "steps": result.steps,
            "splats in": start.count,
            "splats out": result.gaussians.count,
            "match before dB": round(sum(before) / count, 2),
            "match after dB": round(sum(after) / count, 2),
            "seconds": round(result.seconds, 1),
        }, facts)
        del before_scene, after_scene
        devices.release(device)
        return io.NodeOutput(fitted, result.cameras, torch.stack(drawn))
