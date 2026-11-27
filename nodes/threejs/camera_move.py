"""A preset camera move, such as a push in or an orbit, laid on a Three.js camera."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import THREE_CAMERA
from ...modules.threejs.spec import create_spec, require_spec

REQUIRES = "threejs"

#: Moves in the order the menu lists them, with the default amount each starts from.
MOVES = {
    "push in": 2.0,
    "orbit": 45.0,
    "truck": 2.0,
    "crane": 2.0,
    "pedestal": 1.5,
    "pan": 30.0,
    "tilt": 15.0,
    "zoom": 85.0,
    "dolly zoom": 3.0,
    "dutch": 15.0,
}

#: Speed curves in the order the menu lists them.
EASES = ("ease in and out", "ease in", "ease out", "linear")

#: How the start and end are read, in the order the menu lists them.
UNITS = ("per capture", "per timeline")


class ThreeCameraMove(io.ComfyNode):
    """Lay one preset move on a camera."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeCameraMove",
            display_name="Three Camera Move",
            search_aliases=[
                "WASThreeCameraMove",
                "Three Camera Move",
                "camera move",
                "push in",
                "dolly",
                "orbit",
                "crane",
                "truck",
                "dolly zoom",
                "blockout",
            ],
            category="WAS Suite/Three",
            description=(
                "Move a camera through a shot with a preset move: push in, orbit, truck, crane, "
                "pedestal, pan, tilt, zoom, dolly zoom or dutch. Chain several to cut one move "
                "into the next, each over its own part of the shot."
            ),
            inputs=[
                THREE_CAMERA.Input(
                    "camera",
                    tooltip="The camera to move, from Three Perspective Camera or another Three Camera Move.",
                ),
                io.Combo.Input(
                    "move",
                    options=list(MOVES),
                    default="push in",
                    tooltip=(
                        "`push in` toward the target; `orbit` round it; `truck` sideways; `crane` up "
                        "with the target held; `pedestal` up with the target; `pan` and `tilt` turn on "
                        "the spot; `zoom` changes the lens; `dolly zoom` pushes in and widens; `dutch` "
                        "rolls the horizon."
                    ),
                ),
                io.Float.Input(
                    "amount",
                    default=2.0,
                    min=-100000.0,
                    max=100000.0,
                    step=0.1,
                    tooltip=(
                        "Scene units for push in, truck, crane, pedestal and dolly zoom; degrees for "
                        "orbit, pan, tilt and dutch; the lens to end on in mm for zoom. `2.0` pushes in "
                        "two units, `-2.0` pulls out, `90` orbits a quarter turn right."
                    ),
                ),
                io.Float.Input(
                    "start",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Where the move begins, as a fraction of the shot. 0.0 = the first frame, 0.5 = halfway.",
                ),
                io.Float.Input(
                    "end",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Where the move is complete, as a fraction of the shot. 1.0 = the last frame.",
                ),
                io.Combo.Input(
                    "ease",
                    options=list(EASES),
                    default="ease in and out",
                    tooltip=(
                        "`ease in and out` = starts and settles gently; `ease in` = starts gently; "
                        "`ease out` = settles gently; `linear` = one speed throughout."
                    ),
                ),
                io.Float.Input(
                    "rise",
                    default=0.0,
                    min=-100000.0,
                    max=100000.0,
                    step=0.1,
                    tooltip="Scene units the camera climbs during an orbit. 0.0 = level; other moves leave it unused.",
                    advanced=True,
                ),
                io.Combo.Input(
                    "units",
                    options=list(UNITS),
                    default="per capture",
                    tooltip=(
                        "`per capture` = start and end are fractions of the frames Three Render "
                        "takes; `per timeline` = fractions of Three App's loop_seconds."
                    ),
                    advanced=True,
                ),
            ],
            outputs=[
                THREE_CAMERA.Output(
                    display_name="camera",
                    tooltip="The camera with the move added, for Three App or another Three Camera Move.",
                ),
            ],
        )

    @classmethod
    def execute(cls, camera, move, amount, start, end, ease, rise=0.0, units=UNITS[0]) -> io.NodeOutput:
        """Add the move to the camera's list.

        Raises:
            ValueError: The input is not a camera, the move is unknown, or end is before start.
        """
        require_spec(camera, "camera")
        if move not in MOVES:
            raise ValueError(f"Three Camera Move has no move {move!r}. Choose one of: {', '.join(MOVES)}.")
        if float(end) < float(start):
            raise ValueError(
                f"Three Camera Move was given start {start} and end {end}. Set end at or after start, "
                "such as start 0.0 and end 1.0."
            )
        if move == "zoom" and float(amount) <= 0:
            raise ValueError(f"Three Camera Move was given a {amount} mm lens to zoom to. Try 85.0.")
        params = dict(camera.get("params") or {})
        moves = list(params.get("moves") or [])
        moves.append(
            {
                "move": move,
                "amount": float(amount),
                "start": float(start),
                "end": float(end),
                "ease": ease if ease in EASES else EASES[0],
                "rise": float(rise),
                "units": units if units in UNITS else UNITS[0],
            }
        )
        params["moves"] = moves
        return io.NodeOutput(
            create_spec(
                "camera",
                camera["type"],
                params=params,
                deps=dict(camera.get("deps") or {}),
                meta=camera.get("meta"),
            )
        )
