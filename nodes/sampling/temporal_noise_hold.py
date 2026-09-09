"""Starting noise that carries from one video frame to the next."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.sampling.temporal_noise import MAX_HOLD, TemporalNoiseHold


class TemporalNoiseHoldNode(io.ComfyNode):
    """Starting noise for a video sampler, correlated along the latent's time axis."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASTemporalNoiseHold",
            display_name="Temporal Noise Hold",
            search_aliases=[
                "WASTemporalNoiseHold",
                "Temporal Noise Hold",
                "noise hold",
                "noise flow",
                "correlated noise",
                "frame hold noise",
                "video noise",
                "scene complexity",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Starting noise whose pattern carries over from one video frame to the "
                "next instead of being redrawn each time. The further it carries, the "
                "denser and more detailed the scene comes back. Wire it into the noise "
                "socket of any custom sampler in place of Random Noise. Only the first "
                "draw is shaped, so a sampler that draws fresh noise at every step "
                "overwrites it: the ancestral and SDE families, and anything run with "
                "eta above 0."
            ),
            inputs=[
                io.Int.Input(
                    "noise_seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip=(
                        "The seed the noise field is drawn from, as `0` or `12345`. The "
                        "same seed and hold reproduce a clip exactly."
                    ),
                ),
                io.Float.Input(
                    "hold",
                    default=2.0,
                    min=0.0,
                    max=MAX_HOLD,
                    step=0.1,
                    round=0.01,
                    tooltip=(
                        "How many latent frames the noise pattern carries over. 0.0 draws "
                        "ordinary noise and matches Random Noise exactly. Higher values "
                        "cut the frame-to-frame change: 2.0 to 0.63 of ordinary noise, "
                        "8.0 to 0.34, 25.0 to 0.20, 64.0 to 0.12, 128.0 to 0.09. One latent "
                        "frame is about 3.4 output frames on MiniMax H3, 4 on Wan and 8 on LTX."
                    ),
                ),
            ],
            outputs=[
                io.Noise.Output(
                    display_name="NOISE",
                    tooltip=(
                        "The starting noise, for the noise socket of Sampler Custom "
                        "Advanced or Custom Sampler Affine Advanced."
                    ),
                ),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, noise_seed, hold) -> io.NodeOutput:
        return io.NodeOutput(
            TemporalNoiseHold(int(noise_seed), float(hold), node_id=cls.hidden.unique_id)
        )
