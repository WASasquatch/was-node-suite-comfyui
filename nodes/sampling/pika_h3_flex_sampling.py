"""The Pika H3 Flex schedule: sigmas from knots, and core's re-noise sampler."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.interface import run_result
from ...modules.model import pika_h3

NODE_NAME = "Pika H3 Flex Sampling"

#: The released knots, as the widget writes them.
RELEASE_KNOTS = "1000, 750, 500, 250, 100, 40, 0"


class PikaH3FlexSampling(io.ComfyNode):
    """The sampler and sigmas Pika H3 Flex was released with."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASPikaH3FlexSampling",
            display_name=NODE_NAME,
            search_aliases=[
                "WASPikaH3FlexSampling", NODE_NAME,
                "pika",
                "pika h3 flex",
                "h3 flex",
                "knots",
                "six step",
                "manual sigmas",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Sample Pika H3 Flex as released: six model calls on its knots, each step's clean "
                "prediction re-noised to the next. Knots starting low, as 100, 40, 0, refine a "
                "latent from a first pass."
            ),
            inputs=[
                io.String.Input(
                    "knots",
                    default=RELEASE_KNOTS,
                    tooltip=(
                        f"Steps on the 0 to 1000 grid, ending in 0: `{RELEASE_KNOTS}` = "
                        "the release; `100, 40, 0` = a refining pass; a value that climbs back, as "
                        "`..., 40, 100, 40, 0`, samples the tail again."
                    ),
                ),
            ],
            outputs=[
                io.Sampler.Output(
                    display_name="sampler",
                    tooltip="Core's lcm sampler: the clean prediction, then fresh noise at the next sigma.",
                ),
                io.Sigmas.Output(
                    display_name="sigmas",
                    tooltip="Video sigmas at shift 12. Released: `1, 0.973, 0.923, 0.8, 0.571, 0.333, 0`.",
                ),
            ],
        )

    @classmethod
    def execute(cls, knots=None) -> io.NodeOutput:
        """Build the sampler and sigmas.

        Raises:
            ValueError: The knots are not numbers in (0, 1000] ending in one 0.
        """
        import comfy.samplers

        values = pika_h3.parse_knots(RELEASE_KNOTS if knots is None else knots)
        sigmas = pika_h3.sigmas_from_knots(values)
        video = [float(s) for s in sigmas]
        run_result.publish(
            summary=f"{len(video) - 1} model calls",
            facts={
                "video": ", ".join(f"{s:.3f}" for s in video),
                "audio": ", ".join(f"{pika_h3.audio_sigma(s):.3f}" for s in video),
            },
        )
        return io.NodeOutput(comfy.samplers.sampler_object("lcm"), sigmas)
