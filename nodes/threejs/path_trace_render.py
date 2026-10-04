"""A Three.js scene path traced to an image the graph can carry on."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import THREE_APP
from ...modules.interface import three_render
from ...modules.threejs.spec import require_spec
from .render import ThreeRender

REQUIRES = "threejs"


class ThreePathTraceRender(io.ComfyNode):
    """Path trace a scene in the browser and answer the frames."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreePathTraceRender",
            display_name="Three Path Trace Render",
            search_aliases=[
                "WASThreePathTraceRender",
                "Three Path Trace Render",
                "path tracing",
                "ray tracing",
                "global illumination",
                "caustics",
                "raytrace",
            ],
            category="WAS Suite/Three",
            description=(
                "Render the scene by following light as it bounces, so soft shadows, colour "
                "bleeding between surfaces, mirror reflections and refracting glass come out of "
                "the geometry itself rather than being approximated. Every pixel averages "
                "`samples` traced paths and noise falls as that number rises, so this trades "
                "time for cleanliness where Three Render trades nothing. Light comes from "
                "Three Environment, from an emissive material and from directional, point and "
                "spot lights; an ambient or hemisphere light is not traced, and a scene lit "
                "only by one comes out black. The effect chain, antialias and supersample "
                "settings are not used. The "
                "drawing happens in an open ComfyUI tab, so a tab has to be open and the "
                "graph queued from it; a headless run says so rather than hanging."
            ),
            inputs=[
                THREE_APP.Input(
                    "app",
                    tooltip="The scene, camera and renderer settings, from Three App.",
                ),
                io.Int.Input(
                    "width",
                    default=512,
                    min=16,
                    max=8192,
                    tooltip=(
                        "Frame width in pixels. Every pixel is traced `samples` times, so 512 "
                        "costs a quarter of what 1024 does."
                    ),
                ),
                io.Int.Input(
                    "height",
                    default=512,
                    min=16,
                    max=8192,
                    tooltip=(
                        "Frame height in pixels. 512 with a 512 width is square; the camera "
                        "fits its view to this shape."
                    ),
                ),
                io.Boolean.Input(
                    "transparent",
                    default=False,
                    tooltip=(
                        "`true` leaves the background clear and returns alpha; `false` fills "
                        "it with the scene's background colour."
                    ),
                ),
                io.Int.Input(
                    "samples",
                    default=64,
                    min=1,
                    max=10000,
                    tooltip=(
                        "Traced paths averaged per pixel. 16 is a rough look, 64 a clean "
                        "still-life, 512 clean through glass and caustics. Noise halves for "
                        "every four times this number."
                    ),
                ),
                io.Int.Input(
                    "bounces",
                    default=5,
                    min=1,
                    max=30,
                    tooltip=(
                        "How many surfaces one path may hit. 1 is direct light only, 5 suits "
                        "most scenes, 12 for a room lit through a doorway."
                    ),
                ),
                io.Int.Input(
                    "num_frames",
                    default=1,
                    min=1,
                    max=three_render.MAX_FRAMES,
                    tooltip=(
                        "How many frames to trace. 1 captures `start` alone as a still. `fps` "
                        "times Three App's loop_seconds is one whole loop, so 96 at 24 a second "
                        "covers a 4 second loop, at 96 times the cost of a still."
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
                    default=300.0,
                    min=1.0,
                    max=86400.0,
                    step=10.0,
                    tooltip=(
                        "Seconds to wait for the whole run before giving up. A traced frame "
                        "takes far longer than a drawn one, so 300.0 suits a still and a long "
                        "run of frames wants thousands."
                    ),
                ),
                io.Int.Input(
                    "transmissive_bounces",
                    default=10,
                    min=0,
                    max=60,
                    tooltip=(
                        "Extra bounces allowed inside glass, on top of `bounces`. 10 carries a "
                        "path through a few panes; 0 turns glass black inside."
                    ),
                ),
                io.Float.Input(
                    "filter_glossy",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Roughens sharp reflections to settle the bright speckle they leave. "
                        "0.0 is exact, 0.05 clears most speckle, 0.5 visibly blurs highlights."
                    ),
                ),
                io.Int.Input(
                    "tiles",
                    default=3,
                    min=1,
                    max=16,
                    tooltip=(
                        "Splits each pass into this many tiles across and down, so one piece of "
                        "work is short enough not to stall the browser. 3 gives nine tiles. "
                        "Raise it where a frame is large enough to time the tab out."
                    ),
                ),
                io.Int.Input(
                    "texture_size",
                    default=1024,
                    min=16,
                    max=8192,
                    tooltip=(
                        "Size every texture in the scene is fitted to for tracing. 1024 suits "
                        "most work; 2048 keeps fine detail in a close-up, at more memory."
                    ),
                ),
                io.Float.Input(
                    "depth_near",
                    default=0.0,
                    min=0.0,
                    max=100000.0,
                    step=0.1,
                    tooltip=(
                        "Distance the depth pass calls white. 0.0 fits it to what is in shot."
                    ),
                ),
                io.Float.Input(
                    "depth_far",
                    default=0.0,
                    min=0.0,
                    max=100000.0,
                    step=0.1,
                    tooltip=(
                        "Distance the depth pass calls black. 0.0 fits it to what is in shot."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip=(
                        "The traced frames, as one batch in time order. RGBA where transparent "
                        "was on, RGB otherwise."
                    ),
                ),
                io.Image.Output(
                    display_name="depth",
                    tooltip=(
                        "The same frames as distance from the camera, white for near. Drawn "
                        "rather than traced, and feeds a depth ControlNet."
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
        cls, app, width, height, transparent, samples, bounces, num_frames, start, fps,
        timeout,
        transmissive_bounces=10,
        filter_glossy=0.0,
        tiles=3,
        texture_size=1024,
        depth_near=0.0,
        depth_far=0.0,
    ) -> io.NodeOutput:
        """Ask the browser to trace the frames and wait for them.

        Raises:
            ValueError: ``app`` is not an app descriptor, no browser answered in time, or the
                browser reported that it could not trace the scene.
            InterruptProcessingException: The run was cancelled while waiting.
        """
        require_spec(app, "app")
        token = three_render.file_job(
            app, int(width), int(height), bool(transparent),
            ThreeRender.moments(int(num_frames), float(start), float(fps)),
            1,
            float(depth_near),
            float(depth_far),
            trace={
                "samples": int(samples),
                "bounces": int(bounces),
                "transmissiveBounces": int(transmissive_bounces),
                "filterGlossy": float(filter_glossy),
                "tiles": int(tiles),
                "textureSize": int(texture_size),
                # The browser gives up on a frame no later than the node gives up on the run.
                "patience": float(timeout) * 1000.0,
            },
            progress_total=max(1, int(num_frames)) * max(1, int(samples)),
        )
        outcome, bodies, message = three_render.wait_for_frames(
            token, float(timeout), str(cls.hidden.unique_id or "")
        )

        if outcome == three_render.TIMED_OUT:
            raise ValueError(
                f"Three Path Trace Render waited {float(timeout):.0f}s and the browser did not "
                f"finish the frames. Tracing is slow: lower samples, bounces, width and height, "
                f"or raise timeout. The drawing happens in an open ComfyUI tab, so keep one open "
                f"on this server and queue from it. A headless run cannot use this node."
            )
        if outcome != three_render.DELIVERED or not bodies.get("png"):
            raise ValueError(
                f"The browser could not trace the scene: {message or 'it gave no reason'}. The "
                f"viewer's own error line usually says more."
            )
        return io.NodeOutput(
            ThreeRender.as_batch(bodies["png"], "Three Path Trace Render"),
            ThreeRender.as_batch(bodies["depth"], "Three Path Trace Render"),
            ThreeRender.as_batch(bodies["normal"], "Three Path Trace Render"),
            len(bodies["png"]),
            float(fps),
        )
