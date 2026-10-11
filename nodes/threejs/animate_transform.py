"""Spin, bob and pulse an object every frame."""

from __future__ import annotations

import math

from comfy_api.latest import io

from ...modules.compat.types import THREE_OBJECT
from ...modules.threejs.spec import create_spec, require_spec

REQUIRES = "threejs"

#: How the amounts are read, in the order the menu lists them.
UNITS = ("per second", "per capture", "per timeline")


class ThreeAnimateTransform(io.ComfyNode):
    """Give an object a looping motion."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeAnimateTransform",
            display_name="Three Animate Transform",
            search_aliases=[
                "WASThreeAnimateTransform",
                "Three Animate Transform",
                "animate",
                "spin",
                "bob",
            ],
            category="WAS Suite/Three",
            description=(
                "Keep an object moving: rotate turns it on each axis, bob slides it up and "
                "down, and pulse breathes its scale. Amounts are rates, totals over the "
                "captured frames, or totals over Three App's loop_seconds, as `units` "
                "chooses. Every amount at 0.0 leaves the object still."
            ),
            inputs=[
                THREE_OBJECT.Input(
                    "object",
                    tooltip="The object to animate. It is wrapped rather than changed.",
                ),
                io.Combo.Input(
                    "units",
                    options=list(UNITS),
                    default="per timeline",
                    tooltip=(
                        "`per second` = rates, so `90` is a quarter turn a second; `per "
                        "capture` = totals across the frames taken, whatever their count; `per "
                        "timeline` = totals across Three App's loop_seconds, so `180` is 180 "
                        "degrees per loop."
                    ),
                ),
                io.Float.Input(
                    "rotate_x",
                    default=0.0,
                    min=-36000.0,
                    max=36000.0,
                    step=1.0,
                    tooltip=(
                        "Degrees around X: per second, per capture or per loop, as `units` "
                        "reads it. 0.0 is still, 90.0 a quarter turn."
                    ),
                ),
                io.Float.Input(
                    "rotate_y",
                    default=30.0,
                    min=-36000.0,
                    max=36000.0,
                    step=1.0,
                    tooltip=(
                        "Degrees around Y: per second, per capture or per loop, as `units` "
                        "reads it. 360.0 is one full turn, which loops cleanly on `per "
                        "timeline`; negative reverses it."
                    ),
                ),
                io.Float.Input(
                    "rotate_z",
                    default=0.0,
                    min=-36000.0,
                    max=36000.0,
                    step=1.0,
                    tooltip=(
                        "Degrees around Z: per second, per capture or per loop, as `units` "
                        "reads it. 0.0 is still, 90.0 a quarter turn."
                    ),
                ),
                io.Float.Input(
                    "bob_amplitude",
                    default=0.0,
                    min=0.0,
                    max=100000.0,
                    step=0.01,
                    tooltip="How far it rises and falls, in scene units. 0.0 is off, 0.25 is a gentle float.",
                ),
                io.Float.Input(
                    "bob_frequency",
                    default=1.0,
                    min=0.0,
                    max=1000.0,
                    step=0.01,
                    tooltip=(
                        "Bobs per second, per capture or per loop, as `units` reads it. 1.0 is "
                        "one rise and fall, 0.25 is languid."
                    ),
                ),
                io.Float.Input(
                    "pulse_amplitude",
                    default=0.0,
                    min=0.0,
                    max=1000.0,
                    step=0.01,
                    tooltip="How much the scale breathes, as a fraction. 0.0 is off, 0.1 swells it a tenth.",
                ),
                io.Float.Input(
                    "pulse_frequency",
                    default=1.0,
                    min=0.0,
                    max=1000.0,
                    step=0.01,
                    tooltip=(
                        "Pulses per second, per capture or per loop, as `units` reads it. 1.0 "
                        "is one breath, 2.0 is twice as quick."
                    ),
                ),
                io.Float.Input(
                    "phase",
                    default=0.0,
                    min=-100000.0,
                    max=100000.0,
                    step=0.01,
                    tooltip=(
                        "Offset into the bob and pulse cycles, in radians: 3.14 is half a "
                        "cycle. Give copies 0.0, 0.5 and 1.0 so they do not move in step."
                    ),
                ),
            ],
            outputs=[
                THREE_OBJECT.Output(
                    display_name="animated",
                    tooltip="The moving object, for Three Group or the root socket on Three Scene.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        object,
        units,
        rotate_x,
        rotate_y,
        rotate_z,
        bob_amplitude,
        bob_frequency,
        pulse_amplitude,
        pulse_frequency,
        phase,
    ) -> io.NodeOutput:
        """Describe the motion.

        Raises:
            ValueError: The input is not an object descriptor.
        """
        require_spec(object, "object")
        return io.NodeOutput(
            create_spec(
                "object",
                "AnimatedGroup",
                params={
                    "units": units,
                    "rotate": [
                        math.radians(float(rotate_x)),
                        math.radians(float(rotate_y)),
                        math.radians(float(rotate_z)),
                    ],
                    "bobAmplitude": float(bob_amplitude),
                    "bobFrequency": float(bob_frequency),
                    "pulseAmplitude": float(pulse_amplitude),
                    "pulseFrequency": float(pulse_frequency),
                    "phase": float(phase),
                },
                children=[object],
            )
        )
