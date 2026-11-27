"""A whole blockout scene, its cameras and its H3 prompt, read from JSON."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import BEATS, THREE_CAMERA, THREE_OBJECT
from ...modules.interface import run_result
from ...modules.threejs import blockout_scene

REQUIRES = "threejs"

NODE_NAME = "Three Blockout Scene"

EXAMPLE = """{
  "setting": "A small kitchen in the evening.",
  "props": [
    {"id": "table", "type": "table"},
    {"id": "chair_a", "type": "chair", "at": "table.left"},
    {"id": "chair_b", "type": "chair", "at": "table.right"},
    {"id": "plate", "type": "plate", "on": "table", "count": 2},
    {"id": "back_wall", "type": "wall_with_window", "position": [0, -2.5]}
  ],
  "people": [
    {"id": "anna", "sit_on": "chair_a", "look_at": "ben", "description": "a woman in a green sweater"},
    {"id": "ben", "sit_on": "chair_b", "look_at": "anna", "description": "a man in a grey shirt"}
  ],
  "shots": [
    {"seconds": 3, "rig": "two shot", "subjects": ["anna", "ben"], "move": "push in"},
    {"seconds": 2, "rig": "over the shoulder", "from": "ben", "onto": "anna", "action": "Anna talks and smiles."},
    {"seconds": 2, "rig": "over the shoulder", "from": "anna", "onto": "ben", "action": "Ben laughs."}
  ]
}"""


class ThreeBlockoutScene(io.ComfyNode):
    """Build a blockout scene, its shots and its H3 prompt from JSON."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeBlockoutScene",
            display_name=NODE_NAME,
            search_aliases=["WASThreeBlockoutScene", NODE_NAME, "blockout", "scene json", "shot list", "storyboard"],
            category="WAS Suite/Three",
            description=(
                "Build a grey-box scene from a JSON description: ready-made props, props made of shapes, posed "
                "people and a list of shots the camera cuts between. It also writes the matching MiniMax H3 "
                "prompt, one line per shot, so a language model can block out a whole sequence."
            ),
            inputs=[
                io.String.Input(
                    "scene",
                    multiline=True,
                    default=EXAMPLE,
                    tooltip=(
                        "The scene as JSON: `props`, `people`, `shots`, and optionally `kinds`, `setting`, `look`, "
                        "`sound`, `lighting`, `floor`. Up to 64 KB, 150 props, 12 people and 24 shots."
                    ),
                ),
                io.Int.Input(
                    "fps",
                    default=24,
                    min=1,
                    max=60,
                    tooltip="Frames per second the shots are timed at. 24 matches MiniMax H3.",
                ),
                BEATS.Input(
                    "beats",
                    optional=True,
                    tooltip=(
                        "A song's beats from Audio Beats: a shot's `beats` count on them and every cut lands on the "
                        "nearest beat. Unwired = `tempo` from the JSON, cuts where the seconds fall."
                    ),
                ),
                io.String.Input(
                    "blueprint",
                    optional=True,
                    force_input=True,
                    tooltip=(
                        "Video Blueprint's blueprint: every shot written `\"rig\": \"measured\"` takes the "
                        "camera keys, seconds and handheld shake measured for the shot of the same number. "
                        "Unwired = no shot may be measured."
                    ),
                ),
            ],
            outputs=[
                THREE_OBJECT.Output(
                    display_name="objects",
                    tooltip="The set, people and lights, for the root socket on Three Scene.",
                ),
                THREE_CAMERA.Output(
                    display_name="camera",
                    tooltip="One camera that cuts between the shots, for the camera socket on Three App.",
                ),
                io.String.Output(
                    display_name="summary",
                    tooltip="One line saying what the scene is, for summary on H3 Structure Video.",
                ),
                io.String.Output(
                    display_name="description",
                    tooltip="The H3 prompt with one [Shot N] line per shot, for description on H3 Structure Video.",
                ),
                io.String.Output(
                    display_name="soundscape",
                    tooltip="The scene's sound, for soundscape on H3 Structure Video. Empty when none is given.",
                ),
                io.Int.Output(
                    display_name="frames",
                    tooltip=(
                        "Frames all the shots take at fps, for frame_count on Three Render and length on H3. "
                        "The last shot is lengthened to the next count H3 accepts: 158, 175, 192 and so on."
                    ),
                ),
                io.String.Output(
                    display_name="prompt",
                    tooltip=(
                        "The whole MiniMax H3 reference prompt, the block-out a loose guide to placement, camera and "
                        "cuts, for prompt on MiniMax H3 Reference to Video with the render as its first video."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, scene, fps=24, beats=None, blueprint=None) -> io.NodeOutput:
        """Build the scene.

        Raises:
            ValueError: The JSON is not a scene this builds; every problem found is listed.
        """
        try:
            built = blockout_scene.build(scene, float(fps), beats, blueprint)
        except blockout_scene.BlockoutError as error:
            run_result.publish(status=run_result.ERROR, summary="the scene was refused",
                               items=str(error).splitlines()[1:])
            raise ValueError(f"{NODE_NAME}: {error}") from None
        frames = max(1, int(round(built.seconds * int(fps))))
        run_result.publish(
            status=run_result.OK,
            summary=f"{built.counts['shots']} shot{'s' if built.counts['shots'] != 1 else ''}, "
                    f"{built.seconds:.1f} s, {frames} frames",
            counts=built.counts,
            bodies=run_result.body("shots", "\n".join(built.shots)),
        )
        return io.NodeOutput(built.objects, built.camera, built.summary, built.description, built.soundscape, frames,
                             built.prompt)
