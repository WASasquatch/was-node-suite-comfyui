"""Turning a still picture into a video by moving a virtual camera over it."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.interface import batch_report

REQUIRES = "extras"

#: The move a freshly dropped node performs: a slow push in with a full rotation, ending
#: off-centre with a mild tilt and dolly.
DEFAULT_TRAJECTORY_SPEC = """{
  "loop": false,
  "default_ease": "linear",
  "keyframes": [
    {
      "frame": 0,
      "zoom": 1.0,
      "center": [0.5, 0.5],
      "angle": 0.0,
      "pan": [0.0, 0.0],
      "tilt": [0.0, 0.0],
      "dolly_strength": 0.0,
      "dolly_radius": [0.3, 0.3],
      "dolly_feather": 0.5,
      "dolly_mode": "radial",
      "sphereize_strength": 0.0,
      "sphereize_radius": [0.4, 0.4],
      "sphereize_feather": 0.5,
      "depth_strength": 0.0,
      "ease": "ease_in_out"
    },
    {
      "frame": 30,
      "zoom": 1.5,
      "center": [0.5, 0.5],
      "angle": 90.0,
      "pan": [0.0, 0.0],
      "tilt": [0.0, 0.15],
      "dolly_strength": 0.3,
      "dolly_radius": [0.35, 0.35],
      "dolly_feather": 0.5,
      "dolly_mode": "radial",
      "sphereize_strength": 0.0,
      "sphereize_radius": [0.4, 0.4],
      "sphereize_feather": 0.5,
      "depth_strength": 0.0,
      "ease": "ease_in_out"
    },
    {
      "frame": 59,
      "zoom": 2.0,
      "center": [0.6, 0.4],
      "angle": 360.0,
      "pan": [0.1, 0.0],
      "tilt": [0.0, 0.3],
      "dolly_strength": 0.6,
      "dolly_radius": [0.4, 0.4],
      "dolly_feather": 0.5,
      "dolly_mode": "radial",
      "sphereize_strength": 0.0,
      "sphereize_radius": [0.4, 0.4],
      "sphereize_feather": 0.5,
      "depth_strength": 0.0,
      "ease": "linear"
    }
  ]
}"""

#: How each edge_mode option is sampled outside the picture.
PADDING_MODES = {"mirror": "reflection", "border": "border", "wrap": "border"}

#: Largest shake seed, the top of the signed 32-bit range.
MAX_SHAKE_SEED = 2147483647


class CameraMotionTrajectory(io.ComfyNode):
    """Render a keyframed camera move over a still picture as a frame sequence."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASNSCameraMotionTrajectory",
            display_name="Camera Motion Trajectory from Images",
            search_aliases=[
                "WASNSCameraMotionTrajectory", "camera", "ken burns", "pan", "zoom", "parallax",
            ],
            category="WAS Suite/Animation",
            description=(
                "Move a virtual camera over a still picture and emit the result as a frame "
                "sequence: zoom, rotate, pan, tilt, dolly and fisheye, keyframed in JSON "
                "with easing between keys. Feed it a depth map and near parts of the scene "
                "move more than far ones, which turns a single image into a parallax shot "
                "ready for a video encoder or an image-to-video model. A keyframe takes any "
                "of zoom, center, angle, pan, tilt, dolly_strength, sphereize_strength and "
                "depth_strength, 'ease' on it shapes the run to the next keyframe, and "
                "'loop': true wraps the last keyframe back round to the first. A property a "
                "later keyframe leaves out keeps moving at the '<name>_speed' the earlier one "
                "gave it."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The picture the camera moves over. A batch is read as source "
                        "frames: when it holds exactly num_frames images each output frame "
                        "uses its own, otherwise the batch is cycled."
                    ),
                ),
                io.Int.Input(
                    "num_frames", default=60, min=1, max=2048, step=1,
                    tooltip=(
                        "How many frames to render. At 24 frames per second, 60 frames is "
                        "two and a half seconds. Keyframe numbers in the spec are held "
                        "inside this range."
                    ),
                ),
                io.String.Input(
                    "trajectory_spec", default=DEFAULT_TRAJECTORY_SPEC, multiline=True,
                    tooltip=(
                        "The move, as JSON: a 'keyframes' list, each entry carrying a "
                        "'frame' number and the camera properties it sets there. Empty text "
                        "holds the picture still."
                    ),
                ),
                io.Combo.Input(
                    "edge_mode", options=["border", "mirror", "wrap"], default="mirror",
                    tooltip=(
                        "What fills the frame when the camera looks past the edge of the "
                        "picture. `mirror` reflects the picture back, which is the least "
                        "visible; `border` smears the edge pixels; `wrap` brings the "
                        "opposite edge round, which suits a seamless texture."
                    ),
                ),
                io.Boolean.Input(
                    "enable_camera_shake", default=False,
                    tooltip=(
                        "Whether to add a handheld wobble on top of the keyframed move. Off "
                        "gives a locked-off, tripod-steady result; on makes the shot feel "
                        "operated by a person."
                    ),
                ),
                io.Float.Input(
                    "shake_position_amplitude", default=0.03, min=0.0, max=0.5, step=0.001,
                    tooltip=(
                        "How far the wobble drifts, as a share of the frame. 0.01 is a "
                        "barely visible breath, 0.03 a natural handheld hold, 0.2 a running "
                        "shot. Ignored while enable_camera_shake is off."
                    ),
                ),
                io.Float.Input(
                    "shake_rotation_amplitude", default=1.5, min=0.0, max=45.0, step=0.1,
                    tooltip=(
                        "How far the wobble rolls, in degrees. 1.5 reads as a steady hand, "
                        "10 as an unsteady one. Set to 0 for drift without any roll. Ignored "
                        "while enable_camera_shake is off."
                    ),
                ),
                io.Int.Input(
                    "shake_seed", default=0, min=0, max=MAX_SHAKE_SEED, step=1,
                    tooltip=(
                        "Seed for the wobble. The same seed always produces the same wobble, "
                        "so a shot can be re-rendered identically; change it to try another "
                        "take."
                    ),
                ),
                io.Image.Input(
                    "depth_map", optional=True,
                    tooltip=(
                        "Optional depth map, white near and black far, at any size. With one "
                        "connected, 'depth_strength' in the spec holds the far parts of the "
                        "scene back while the near parts move fully, which is what makes the "
                        "shot read as parallax. Leave it unconnected for a flat move."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="video",
                    tooltip=(
                        "The rendered frames in order, all at the input's size, ready for a "
                        "video writer or an image-to-video model."
                    ),
                ),
                io.Int.Output(
                    display_name="frame_count",
                    tooltip=(
                        "How many frames were rendered, for wiring straight into a video "
                        "writer's frame count or a duration calculation."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        num_frames,
        trajectory_spec,
        edge_mode,
        enable_camera_shake,
        shake_position_amplitude,
        shake_rotation_amplitude,
        shake_seed,
        depth_map=None,
    ) -> io.NodeOutput:
        """Render the move.

        Raises:
            ValueError: The trajectory spec is not valid JSON, or an input tensor is not
                shaped ``(batch, height, width, channels)``.
        """
        import torch
        import torch.nn.functional as functional

        from ....modules.image.camera_path import (
            build_base_grid,
            build_camera_shake,
            create_frame_grid,
            interpolate_camera_paths,
            load_trajectory_config,
            normalize_keyframes,
        )

        if image.ndim != 4:
            raise ValueError(
                f"image must be shaped (batch, height, width, channels) and is "
                f"{tuple(image.shape)}"
            )

        batch_size, height, width, _channels = image.shape
        config = load_trajectory_config(trajectory_spec)

        default_mode = str(config.get("dolly_mode", "radial")).lower()
        if default_mode not in ("radial", "aspect", "box"):
            default_mode = "radial"

        keyframes, loop = normalize_keyframes(
            config=config,
            num_frames=num_frames,
            default_dolly_mode=default_mode,
            default_depth_strength=float(config.get("depth_strength", 0.0)),
        )
        tracks = interpolate_camera_paths(keyframes, num_frames, loop)
        shake_x, shake_y, shake_angle = build_camera_shake(
            num_frames=num_frames,
            enable=enable_camera_shake,
            position_amplitude=shake_position_amplitude,
            rotation_amplitude=shake_rotation_amplitude,
            seed=shake_seed,
        )

        source = image.movedim(-1, 1).contiguous()
        base_grid = build_base_grid(height, width, device=image.device, dtype=image.dtype)

        depth = None
        if depth_map is not None:
            if depth_map.ndim != 4:
                raise ValueError(
                    f"depth_map must be shaped (batch, height, width, channels) and is "
                    f"{tuple(depth_map.shape)}"
                )
            planes = depth_map.movedim(-1, 1).to(image.dtype)
            if depth_map.shape[1] != height or depth_map.shape[2] != width:
                planes = functional.interpolate(
                    planes, size=(height, width), mode="bilinear", align_corners=False
                )
            if depth_map.shape[3] == 3:
                depth = (
                    0.299 * planes[:, 0:1] + 0.587 * planes[:, 1:2] + 0.114 * planes[:, 2:3]
                )
            else:
                depth = planes[:, 0:1]

        padding_mode = PADDING_MODES.get(edge_mode, "border")
        frames = []

        for index in range(num_frames):
            src_index = index if (batch_size > 1 and num_frames == batch_size) else index % batch_size

            depth_frame = None
            if depth is not None:
                if depth.shape[0] == 1:
                    depth_frame = depth[0, 0:1]
                else:
                    depth_frame = depth[src_index % depth.shape[0], 0:1]

            grid = create_frame_grid(
                base_grid=base_grid,
                center_x_norm=float(tracks["center_x"][index]),
                center_y_norm=float(tracks["center_y"][index]),
                zoom=float(tracks["zoom"][index]),
                angle_deg=float(tracks["angle"][index]) + float(shake_angle[index]),
                pan_x_norm=float(tracks["pan_x"][index]),
                pan_y_norm=float(tracks["pan_y"][index]),
                tilt_x=float(tracks["tilt_x"][index]),
                tilt_y=float(tracks["tilt_y"][index]),
                dolly_strength=float(tracks["dolly_strength"][index]),
                dolly_radius_x=float(tracks["dolly_radius_x"][index]),
                dolly_radius_y=float(tracks["dolly_radius_y"][index]),
                dolly_feather=float(tracks["dolly_feather"][index]),
                sphereize_strength=float(tracks["sphereize_strength"][index]),
                sphereize_radius_x=float(tracks["sphereize_radius_x"][index]),
                sphereize_radius_y=float(tracks["sphereize_radius_y"][index]),
                sphereize_feather=float(tracks["sphereize_feather"][index]),
                dolly_mode=str(tracks["dolly_mode"][index]),
                depth_grid=depth_frame,
                depth_strength=float(tracks["depth_strength"][index]),
                shake_x_norm=float(shake_x[index]),
                shake_y_norm=float(shake_y[index]),
            )

            if edge_mode == "wrap":
                grid = (grid + 1.0) % 2.0 - 1.0

            frames.append(
                functional.grid_sample(
                    source[src_index:src_index + 1],
                    grid,
                    mode="bilinear",
                    padding_mode=padding_mode,
                    align_corners=True,
                )
            )

        video = torch.cat(frames, dim=0).movedim(1, 3).contiguous()
        size, mode = batch_report.describe_images(video)
        batch_report.publish(
            frames=int(video.shape[0]),
            slots=1,
            size=size,
            mode=mode,
            memory=batch_report.memory_of(video),
        )
        return io.NodeOutput(video, num_frames)
