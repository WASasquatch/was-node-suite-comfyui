"""A Three.js scene rendered to an image the graph can carry on."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import THREE_APP
from ...modules.interface import three_render
from ...modules.threejs.spec import refuse_script, require_spec

REQUIRES = "threejs"


class ThreeRender(io.ComfyNode):
    """Render a scene in the browser and answer the frame."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeRender",
            display_name="Three Render",
            search_aliases=[
                "WASThreeRender",
                "Three Render",
                "render 3d",
                "screenshot",
                "viewport export",
            ],
            category="WAS Suite/Three",
            description=(
                "Draw the scene at a size you choose and hand it on as an IMAGE, so a Three.js "
                "scene can be saved, composited or fed to a sampler. Every frame comes back "
                "three ways, picture, depth and normals, so one render feeds a preview and a "
                "ControlNet at once. Three App's loop_seconds is how long the "
                "animation runs before it repeats, and fps and num_frames only sample it, so "
                "fps times loop_seconds is one whole loop and changing fps alone changes "
                "smoothness rather than speed. Each batch is in time order, and the fps output "
                "feeds a video saver's own fps. The scene is wound forward through the run, so "
                "motion that adds up frame by frame lands where it would. The drawing happens "
                "in an open ComfyUI tab, so one has to be open and the graph queued from it; a "
                "headless run says so rather than hanging."
            ),
            inputs=[
                THREE_APP.Input(
                    "app",
                    tooltip="The scene, camera and renderer settings, from Three App.",
                ),
                io.Int.Input(
                    "width",
                    default=1024,
                    min=16,
                    max=8192,
                    step=8,
                    tooltip="Frame width in pixels. 1024 is a working size, 4096 needs a capable GPU.",
                ),
                io.Int.Input(
                    "height",
                    default=1024,
                    min=16,
                    max=8192,
                    step=8,
                    tooltip="Frame height in pixels. 1024 is square; the camera fits its view to this shape.",
                ),
                io.Boolean.Input(
                    "transparent",
                    default=False,
                    tooltip=(
                        "`true` leaves the background clear and returns alpha; `false` fills it "
                        "with the scene's background colour."
                    ),
                ),
                io.Int.Input(
                    "num_frames",
                    default=96,
                    min=1,
                    max=three_render.MAX_FRAMES,
                    tooltip=(
                        "How many frames to draw. `fps` times Three App's loop_seconds is one "
                        "whole loop, so 96 at 24 a second covers a 4 second loop exactly. 1 "
                        "captures `start` alone as a still."
                    ),
                ),
                io.Float.Input(
                    "start",
                    default=0.0,
                    min=0.0,
                    max=3600.0,
                    step=0.01,
                    tooltip=(
                        "Seconds into the animation the first frame is taken at. 0.0 is the "
                        "pose the scene starts in."
                    ),
                ),
                io.Float.Input(
                    "fps",
                    default=24.0,
                    min=0.01,
                    max=1000.0,
                    step=1.0,
                    tooltip=(
                        "Frames a second. It sets how densely the animation is sampled, never "
                        "how fast it moves. 24.0 over 96 frames is four seconds. Give the same "
                        "number to a video saver, or wire the fps output straight into it."
                    ),
                ),
                io.Float.Input(
                    "timeout",
                    default=180.0,
                    min=1.0,
                    max=86400.0,
                    step=10.0,
                    tooltip=(
                        "Seconds to wait for the whole run. 180.0 covers a 96 frame loop, 30.0 "
                        "a single frame, and a long run at a large size wants thousands."
                    ),
                ),
                io.Int.Input(
                    "supersample",
                    default=2,
                    min=1,
                    max=4,
                    tooltip=(
                        "Draws the frame this many times oversize and scales it back down, "
                        "which is what smooths a stepped edge. 1 is fastest, 2 is the usual "
                        "choice, 4 costs sixteen times the pixels."
                    ),
                ),
                io.Float.Input(
                    "depth_near",
                    default=0.0,
                    min=0.0,
                    max=100000.0,
                    step=0.1,
                    tooltip=(
                        "Distance from the camera the depth pass calls white. 0.0 fits the "
                        "range to whatever is in shot, which a wide floor stretches; set it "
                        "and depth_far around the subject to spend the whole range on it."
                    ),
                ),
                io.Float.Input(
                    "depth_far",
                    default=0.0,
                    min=0.0,
                    max=100000.0,
                    step=0.1,
                    tooltip=(
                        "Distance the depth pass calls black. 0.0 fits it to what is in shot. "
                        "For a figure 8 units away, 6.0 and 10.0 give it the whole range."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip=(
                        "The frames, as one batch in time order. RGBA where transparent was "
                        "on, RGB otherwise."
                    ),
                ),
                io.Image.Output(
                    display_name="depth",
                    tooltip=(
                        "The same frames as distance from the camera, white for near, spread "
                        "across what is actually in shot rather than across near and far. "
                        "Feeds a depth ControlNet."
                    ),
                ),
                io.Image.Output(
                    display_name="normal",
                    tooltip=(
                        "The same frames as the direction each surface faces, in the "
                        "tangent-space layout a normal ControlNet reads."
                    ),
                ),
                io.Int.Output(
                    display_name="frame_count",
                    tooltip="How many frames each batch holds, which is num_frames.",
                ),
                io.Float.Output(
                    display_name="fps",
                    tooltip=(
                        "The frame rate the frames were taken at, for a video saver's own fps "
                        "so the two cannot disagree."
                    ),
                ),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(
        cls, app, width, height, transparent, num_frames, start, fps, timeout,
        supersample=2,
        depth_near=0.0,
        depth_far=0.0,
    ) -> io.NodeOutput:
        """Ask the browser for the frames and wait for them.

        Raises:
            ValueError: ``app`` is not an app descriptor, no browser answered in time, or the
                browser reported that it could not draw the scene.
            InterruptProcessingException: The run was cancelled while waiting.
        """
        require_spec(app, "app")
        refuse_script(app, "Three Render")
        token = three_render.file_job(
            app, int(width), int(height), bool(transparent),
            cls.moments(int(num_frames), float(start), float(fps)),
            int(supersample),
            float(depth_near),
            float(depth_far),
        )
        outcome, bodies, message = three_render.wait_for_frames(
            token, float(timeout), str(cls.hidden.unique_id or "")
        )

        if outcome == three_render.TIMED_OUT:
            raise ValueError(
                f"Three Render waited {float(timeout):.0f}s and the browser did not finish the "
                f"frames. The drawing happens in an open ComfyUI tab, so keep one open on this "
                f"server and queue from it. A long run of frames may just need a longer "
                f"timeout. A headless run cannot use this node."
            )
        if outcome != three_render.DELIVERED or not bodies.get("png"):
            raise ValueError(
                f"The browser could not draw the scene: {message or 'it gave no reason'}. The "
                f"viewer's own error line usually says more."
            )
        return io.NodeOutput(
            cls.as_batch(bodies["png"]),
            cls.as_batch(bodies["depth"]),
            cls.as_batch(bodies["normal"]),
            len(bodies["png"]),
            float(fps),
        )

    @staticmethod
    def moments(num_frames: int, start: float, fps: float) -> list[float]:
        """When each frame is taken, in seconds.

        Args:
            num_frames: How many frames to draw.
            start: Seconds the first frame is taken at.
            fps: Frames a second, which sets the gap between them.

        Returns:
            One moment per frame, ascending. A single frame is taken at ``start``.
        """
        if num_frames <= 1:
            return [start]
        step = 1.0 / max(fps, 1e-6)
        return [start + step * index for index in range(num_frames)]

    @staticmethod
    def as_batch(bodies: list[bytes]):
        """Frames as one ``IMAGE`` batch.

        Args:
            bodies: The PNG bytes the browser posted back, in time order.

        Returns:
            A float tensor shaped ``(frames, height, width, channels)`` in ``[0, 1]``.
        """
        import io as _io

        import numpy as np
        import torch
        from PIL import Image

        frames = []
        for body in bodies:
            picture = Image.open(_io.BytesIO(body))
            picture = picture.convert("RGBA" if "A" in picture.getbands() else "RGB")
            frames.append(np.asarray(picture).astype(np.float32) / 255.0)
        return torch.from_numpy(np.stack(frames, axis=0))
