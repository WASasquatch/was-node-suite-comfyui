"""Upscaling video with Kandinsky 6 video super-resolution."""

from __future__ import annotations

import threading
import time

from comfy_api.latest import io

from ...modules.interface import size_report
from ...modules.io import picker
from ...modules.model.kandinsky6 import sr, upscale

NODE_NAME = "Kandinsky 6 Upscale Video"

#: Upscaler choice that picks the Kandinsky 6 latent upscaler file matching the scale.
AUTO = "auto"
#: Upscaler choice that enlarges each tile in pixels and encodes it.
PIXELS = "pixels"

_HELD = {"key": None, "upscaler": None}
_LOCK = threading.Lock()


def _upscalers() -> list[str]:
    import folder_paths

    return [AUTO, *folder_paths.get_filename_list(sr.UPSCALER_FOLDER), PIXELS]


def _auto_name(factor: int) -> str | None:
    import folder_paths

    names = [name for name in folder_paths.get_filename_list(sr.UPSCALER_FOLDER) if "kandinsky" in name.lower()]
    tagged = [name for name in names if f"x{factor}" in name.lower()]
    return (tagged or names or [None])[0]


def _upscaler(choice: str, factor: int):
    import os

    import folder_paths

    if choice == PIXELS:
        return None
    name = _auto_name(factor) if choice == AUTO else choice
    if not name:
        raise ValueError(
            f"{NODE_NAME} found no Kandinsky 6 latent upscaler. Put kandinsky6_vsr_latent_upscaler_x{factor}"
            ".safetensors in ComfyUI/models/latent_upscale_models, or set upscaler to pixels."
        )
    path = folder_paths.get_full_path_or_raise(sr.UPSCALER_FOLDER, name)
    key = (path, os.path.getmtime(path), factor)
    with _LOCK:
        if _HELD["key"] != key:
            _HELD["key"], _HELD["upscaler"] = None, None
            _HELD["upscaler"] = sr.load_upscaler(path, factor)
            _HELD["key"] = key
        return _HELD["upscaler"]


class Kandinsky6UpscaleVideo(io.ComfyNode):
    """Enlarge video frames two, 2.25 or four times with Kandinsky 6 video super-resolution."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASKandinsky6UpscaleVideo",
            display_name=NODE_NAME,
            search_aliases=[
                "WASKandinsky6UpscaleVideo", NODE_NAME,
                "kandinsky vsr", "video super resolution", "upscale video", "k6 upscale", "vsr 2.0",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Enlarge a video two, 2.25 or four times with Kandinsky 6 video super-resolution, which "
                "redraws fine detail while keeping every frame of the motion. Any video works, generated or "
                "filmed, of any length. The panel shows the size before and after."
            ),
            inputs=[
                io.Model.Input("model", tooltip="The upscaler from Kandinsky 6 Upscale Model Loader."),
                io.Vae.Input("vae", tooltip="The K-VAE from Kandinsky 6 Upscale VAE Loader."),
                io.MultiType.Input(
                    "images",
                    [io.Image, io.Video],
                    tooltip=(
                        "The clip: IMAGE frames, or a VIDEO, which keeps its frame rate and sound and is "
                        "read without holding a long clip in memory. Every frame is kept."
                    ),
                ),
                picker.path_input(io.Combo.Input(
                    "upscaler",
                    options=_upscalers(),
                    default=AUTO,
                    tooltip=(
                        "The latent upscaler, from ComfyUI/models/latent_upscale_models: `auto` = the "
                        "Kandinsky 6 x2 or x4 file the scale needs; `pixels` = enlarge each tile in pixels "
                        "first, softer."
                    ),
                ), model=True),
                io.Combo.Input(
                    "scale", options=list(upscale.SCALES), default="2x",
                    tooltip="`2x` = 960x544 to 1920x1088; `2.25x` = 864x480 to 1944x1080; `4x` = 480x272 to 1920x1088.",
                ),
                io.Int.Input(
                    "seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True,
                    tooltip="Seed of the noise each tile starts from: `0` and `1` give two different redraws of the same detail.",
                ),
                io.Int.Input(
                    "steps", default=0, min=0, max=50, advanced=True,
                    tooltip="Model passes per tile: `0` = the model's own, 2 for the distilled release.",
                ),
                io.Float.Input(
                    "noise", default=0.7, min=0.05, max=1.0, step=0.05, advanced=True,
                    tooltip="Noise mixed into the source before redrawing: `0.7` = as trained; lower keeps more of the source.",
                ),
                io.Float.Input(
                    "overlap", default=0.2, min=0.0, max=0.75, step=0.05, advanced=True,
                    tooltip="Least share of each tile its neighbours also cover: `0.2` = as released; higher hides seams, slower.",
                ),
                io.Int.Input(
                    "tiles_per_pass", default=1, min=1, max=16, advanced=True,
                    tooltip="Tiles refined at once: `1` = least memory; more is faster on a large card.",
                ),
                io.Int.Input(
                    "window_frames", default=upscale.WINDOW_FRAMES, min=5, max=361, step=4, advanced=True,
                    tooltip=(
                        "Frames upscaled together: `121` = five seconds at 24 fps, as trained; a longer "
                        "clip is cut into windows of this length. Fewer = less memory."
                    ),
                ),
                io.Int.Input(
                    "window_overlap", default=upscale.WINDOW_OVERLAP, min=0, max=64, advanced=True,
                    tooltip=(
                        "Frames two windows share, crossfaded, at most half a window: `8` = a third of a second "
                        "at 24 fps; `0` = a cut."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip="The enlarged frames, as many as came in. Built only when wired; a long clip is kept on disk.",
                ),
                io.Video.Output(
                    display_name="video",
                    tooltip="The enlarged clip with the source's sound and frame rate, read from disk by Save Video.",
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, model, vae, images, upscaler=AUTO, scale="2x", seed=0, steps=0, noise=0.7,
                overlap=0.2, tiles_per_pass=1, window_frames=upscale.WINDOW_FRAMES,
                window_overlap=upscale.WINDOW_OVERLAP) -> io.NodeOutput:
        """Enlarge ``images`` window by window into a frame cache.

        Raises:
            ValueError: The model is not the Kandinsky 6 upscaler, the VAE is not the K-VAE, the
                frames are smaller than one tile, or no latent upscaler is found.
            PathNotAllowed: A wired upscaler path lies outside the latent upscaler folders.
            MemoryError: The clip, or the wired images output, fits neither in memory nor on a scratch drive.
        """
        import torch

        from ...modules.image import scratch
        from ...modules.io import rooted
        from ...modules.lora import stack
        from ...modules.media import clip as clips
        from ...modules.media import frame_cache

        if not sr.is_upscaler_model(model):
            raise ValueError(
                f"{NODE_NAME} needs the Kandinsky 6 upscaler. Load it with Kandinsky 6 Upscale Model Loader "
                "and wire it into model."
            )
        if getattr(vae, "latent_channels", None) != sr.CHANNELS or getattr(vae, "latent_dim", None) != 3:
            raise ValueError(
                f"{NODE_NAME} needs the K-VAE. Load it with Kandinsky 6 Upscale VAE Loader and wire it into vae."
            )
        if images is None:
            raise ValueError(f"{NODE_NAME} was given no frames.")
        if scale not in upscale.SCALES:
            raise ValueError(f"No scale {scale!r}. Choose one of: {', '.join(upscale.SCALES)}.")
        if upscaler not in (AUTO, PIXELS):
            upscaler = picker.model_name(upscaler, (sr.UPSCALER_FOLDER,))
        latent_upscaler = _upscaler(upscaler, upscale.SCALES[scale][0])

        import comfy.utils

        began = time.perf_counter()
        bars = {}

        def progress(done, total):
            if "bar" not in bars:
                bars["bar"] = comfy.utils.ProgressBar(total)
            bars["bar"].update_absolute(done, total)

        clip = clips.open_clip(images, NODE_NAME, compact=True)
        source = clip.frames
        count, height, width = (int(side) for side in source.shape[:3])
        factor, pre = upscale.SCALES[scale]
        size = (round(height * factor * pre), round(width * factor * pre))
        wanted = stack.consumed(cls.hidden.prompt, cls.hidden.unique_id, 0)
        batch = scratch.allocate(
            (count, size[0], size[1], 3), node=NODE_NAME,
            advice="Save the video output instead, which needs no batch.",
        ) if wanted else None
        cache = frame_cache.FrameCache.create(rooted.destination(rooted.TEMP, "frame_cache"), "k6_upscale", clip.rate,
                                              color_space=clip.color_space)
        written = {"frames": 0}

        def emit(done):
            cache.append(done)
            if batch is not None:
                batch[written["frames"]:written["frames"] + int(done.shape[0])] = done
                scratch.trim(batch[written["frames"]:written["frames"] + int(done.shape[0])])
            written["frames"] += int(done.shape[0])

        try:
            result = upscale.upscale_clip(
                model, vae, latent_upscaler, lambda start, stop: source[start:stop], count, emit, scale=scale,
                seed=int(seed), steps=int(steps), noise=float(noise), overlap=float(overlap),
                tiles_per_pass=int(tiles_per_pass), window=int(window_frames),
                window_overlap=int(window_overlap), progress=progress,
            )
        except BaseException:
            cache.delete()
            raise
        sound = clips.slice_audio(clip.audio, 0, count, clip.rate)
        if sound is not None:
            cache.add_audio(sound)
        layout, spent = result["plan"], result["seconds"]
        seconds = time.perf_counter() - began
        size_report.publish(
            (width, height), (size[1], size[0]), action="upscaled",
            facts={
                "tiles": f"{len(layout.tiles)} of {layout.refined[1]}x{layout.refined[0]} in "
                         f"{len(result['windows'])} window(s) of up to {int(window_frames)} frames",
                "took": f"{seconds:.0f} s: encode {spent['encode']:.0f}, refine {spent['refine']:.0f}, "
                        f"decode {spent['decode']:.0f}, {latent_upscaler.name if latent_upscaler else 'pixel'} enlarging",
            },
        )
        video = frame_cache.CachedVideo(cache.folder, 0, cache.frames)
        frames = batch if batch is not None else torch.zeros((1, 64, 64, 3))
        return io.NodeOutput(frames, video)
