"""Turn unposed pictures into a Gaussian splat and a camera per picture in one pass."""

from __future__ import annotations

import time

import torch
from comfy_api.latest import io

from ...modules.compat.types import LOAD3D_CAMERA, SPLAT
from ...modules.io import picker
from ...modules.mesh.device import compute_device
from ...modules.model import anysplat, model_directories
from ...modules.world import fit as fitting, gaussians, report


class PointCloudFromPictures(io.ComfyNode):
    """Reconstruct a scene from photos or video frames with no camera information."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASPointCloudFromPictures",
            display_name="Point Cloud From Pictures",
            search_aliases=[
                "WASPointCloudFromPictures", "Point Cloud From Pictures", "anysplat", "pictures to splat",
                "video to gaussian splat", "feed forward splat", "photos to 3d", "unposed reconstruction",
            ],
            category="WAS Suite/3D/Point Cloud",
            description=(
                "Turn a few photos or a video's frames, with no camera information, into a Gaussian splat and "
                "the camera each picture was taken from, in one pass. Point Cloud Refine sharpens the result "
                "against the same pictures."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="Pictures of one scene, such as a video's frames or an H3 tour of a picture, in "
                            "any order. Longer batches are thinned to `pictures`.",
                ),
                picker.path_input(io.Combo.Input(
                    "model",
                    options=anysplat.offered(),
                    tooltip="The AnySplat checkpoint, from ComfyUI/models/geometry_estimation; "
                            "`anysplat.safetensors` is fetched on first use with features.network on.",
                ), model=True),
                io.Int.Input(
                    "size", default=448, min=224, max=1022, step=14,
                    tooltip="Shorter side, in pixels, the pictures are read at. 448 = the trained size; "
                            "672 = finer detail with about twice the splats and memory.",
                ),
                io.Combo.Input(
                    "framing", options=list(anysplat.FRAMINGS), default=anysplat.FRAMINGS[0],
                    tooltip="`whole picture` = every picture kept whole; `square crop` = a centred square cut "
                            "from each, as the model was trained.",
                ),
                io.Int.Input(
                    "pictures", default=24, min=2, max=anysplat.MAX_PICTURES, step=1,
                    tooltip=f"Most pictures read, evenly spaced through the batch, first and last kept. 7 = one a "
                            f"second of a 6.6 s H3 clip; 24 = every seventh frame; {anysplat.MAX_PICTURES} = one "
                            f"pass's most.",
                ),
            ],
            outputs=[
                SPLAT.Output(
                    display_name="point_cloud",
                    tooltip="The Gaussian splat, for Point Cloud Refine, Render, the Navigator or core's splat nodes.",
                ),
                LOAD3D_CAMERA.Output(
                    display_name="camera_info",
                    tooltip="The camera of each picture, the first at the origin.",
                ),
                io.Image.Output(
                    display_name="images",
                    tooltip="The pictures as the model read them, sized and framed to match camera_info; wire "
                            "both into Point Cloud Refine.",
                ),
            ],
        )

    @classmethod
    def execute(cls, images, model, size=448, framing=anysplat.FRAMINGS[0], pictures=24) -> io.NodeOutput:
        """Reconstruct the scene and publish a picture beside its view drawn from the splat.

        Raises:
            ValueError: Fewer than two pictures.
            ModelUnavailable: The checkpoint is not on disk and could not be fetched.
        """
        if int(images.shape[0]) > int(pictures):
            picks = torch.linspace(0, int(images.shape[0]) - 1, int(pictures)).round().long().unique()
            images = images[picks]
        count = int(images.shape[0])
        if count < 2:
            raise ValueError("Point Cloud From Pictures needs at least two pictures of the scene; for one "
                             "picture use MoGe Inference into Point Cloud From Depth")
        model_directories(anysplat.FOLDER)
        model = picker.model_name(model, (anysplat.FOLDER,))
        began = time.perf_counter()
        result = anysplat.reconstruct(model, images, int(size), framing)
        seconds = time.perf_counter() - began
        height, width = int(result.pictures.shape[1]), int(result.pictures.shape[2])
        device = compute_device()
        splats = result.gaussians.to(device)
        scores, shown = [], []
        middle = count // 2
        with torch.no_grad():
            for index in range(count):
                drawn = gaussians.rasterize(splats, result.cameras[index], width, height).colour.clamp(0, 1).cpu()
                scores.append(fitting.psnr(drawn, result.pictures[index]))
                if index == middle:
                    shown = [result.pictures[index], drawn]
        report.publish(shown, f"{count} pictures read at {width}x{height}", {
            "pictures": count,
            "splats": result.gaussians.count,
            "match dB": round(sum(scores) / count, 2),
            "seconds": round(seconds, 1),
        }, {
            "views shown": "picture, splat",
            "pixels fused": result.fused,
        })
        del splats
        return io.NodeOutput(gaussians.to_splat(result.gaussians), result.cameras, result.pictures)
