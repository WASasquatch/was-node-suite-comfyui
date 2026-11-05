"""Sample MiniMax H3's picture and sound with separate step rules."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.sampling import h3_streams

NODE_NAME = "H3 Stream Sampler"


class H3StreamSampler(io.ComfyNode):
    """A sampler stepping MiniMax H3's picture and sound by different rules."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASH3StreamSampler",
            display_name=NODE_NAME,
            search_aliases=[
                "WASH3StreamSampler", NODE_NAME,
                "per stream sampler",
                "audio sampler",
                "sound sampler",
                "lcm euler ancestral",
                "minimax h3",
                "pika dmad",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Sample MiniMax H3's picture and sound in one pass with different step rules, by "
                "default the picture as lcm and the sound as Euler ancestral. The sound steps on its "
                "own sigmas from the model's audio shift."
            ),
            inputs=[
                io.Combo.Input(
                    "video",
                    options=list(h3_streams.RULES),
                    default="lcm",
                    tooltip=(
                        "Picture step: `lcm` = clean prediction re-noised to the next sigma, as Pika "
                        "H3 Flex; `euler` = no fresh noise; `euler_ancestral` = fresh noise by eta."
                    ),
                ),
                io.Combo.Input(
                    "audio",
                    options=list(h3_streams.RULES),
                    default="euler_ancestral",
                    tooltip=(
                        "Sound step, on the sound's own sigmas: `euler_ancestral` = fresh noise by "
                        "eta; `lcm` = re-noised clean prediction; `euler` = no fresh noise."
                    ),
                ),
                io.Float.Input(
                    "eta",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Fresh noise in an `euler_ancestral` step: `1.0` = full; `0` = euler. Other rules ignore it.",
                ),
            ],
            outputs=[
                io.Sampler.Output(
                    display_name="sampler",
                    tooltip=(
                        "For Sampler Custom Advanced on MiniMax H3, with picture sigmas such as Pika "
                        "H3 Flex Sampling's. `lcm` on both = core's lcm."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, video="lcm", audio="euler_ancestral", eta=1.0) -> io.NodeOutput:
        """Build the sampler.

        Raises:
            ValueError: A step rule is unknown.
        """
        import comfy.samplers

        for rule in (video, audio):
            if rule not in h3_streams.RULES:
                raise ValueError(f"{NODE_NAME} has no step rule {rule!r}. Choose one of: {', '.join(h3_streams.RULES)}.")
        return io.NodeOutput(comfy.samplers.KSAMPLER(
            h3_streams.sample, {"video": video, "audio": audio, "eta": float(eta)}))
