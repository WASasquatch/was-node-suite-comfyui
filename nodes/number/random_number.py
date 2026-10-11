"""A seeded pseudo-random number."""

from __future__ import annotations

import random

from comfy_api.latest import io

from ...modules.compat.types import NUMBER


class RandomNumber(io.ComfyNode):
    """Draw a pseudo-random value between ``minimum`` and ``maximum``.

    The draw comes from a private ``random.Random`` seeded with ``seed``.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Random Number",
            display_name="Random Number",
            search_aliases=["Random Number", "random", "rng", "seed"],
            category="WAS Suite/Number",
            description=(
                "Draw a random number from a seed, for a value that should vary from run to "
                "run such as a strength, a step count or a seed of its own. The same seed "
                "and the same bounds always draw the same number, so a result can be "
                "reproduced."
            ),
            inputs=[
                io.Combo.Input(
                    "number_type",
                    options=["integer", "float", "bool"],
                    tooltip=(
                        "`integer` = a whole number, both bounds included, fractions cut off "
                        "the bounds first; `float` = a decimal between them; `bool` = bounds "
                        "ignored, a decimal from 0 to 1 that the INT output rounds to 0 or 1, "
                        "a coin flip."
                    ),
                ),
                io.Float.Input(
                    "minimum",
                    default=0,
                    min=-18446744073709551615,
                    max=18446744073709551615,
                    tooltip=(
                        "The lowest value that can come out, itself included. Ignored in "
                        "`bool` mode. Both bounds default to 0, which draws 0 every time "
                        "until they are changed."
                    ),
                ),
                io.Float.Input(
                    "maximum",
                    default=0,
                    min=-18446744073709551615,
                    max=18446744073709551615,
                    tooltip=(
                        "The highest value that can come out, itself included. Ignored in "
                        "`bool` mode, and in `integer` mode a maximum below minimum stops "
                        "with an error."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=18446744073709551615,
                    tooltip=(
                        "Seed for the draw. The same seed always gives the same number; change "
                        "it for a different one. Only this node's draw is affected, sampling "
                        "noise elsewhere in the prompt is left alone. Any whole number; `0` is "
                        "as good a seed as any."
                    ),
                ),
            ],
            outputs=[
                NUMBER.Output(
                    tooltip=(
                        "The drawn value: whole in `integer` mode, decimal in `float` and "
                        "`bool` mode."
                    ),
                ),
                io.Float.Output(tooltip="The same value as a decimal."),
                io.Int.Output(
                    tooltip=(
                        "The same value rounded to the nearest whole number, which is where "
                        "the 0 or 1 of `bool` mode comes from."
                    ),
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, number_type, minimum, maximum, seed):
        """The seed, which together with the other widgets decides the whole output."""
        return seed

    @classmethod
    def execute(cls, number_type, minimum, maximum, seed) -> io.NodeOutput:
        rng = random.Random(seed)

        if number_type == "integer":
            # The widgets are FLOAT sockets, and randint refuses a non-integral bound.
            number = rng.randint(int(minimum), int(maximum))
        elif number_type == "float":
            number = rng.uniform(minimum, maximum)
        else:
            number = rng.random()

        return io.NodeOutput(number, float(number), round(number))
