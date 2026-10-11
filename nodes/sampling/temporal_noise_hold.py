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
                "next, so the scene comes back denser and more detailed. Wire it into the "
                "noise socket of any custom sampler in place of RandomNoise. Samplers that "
                "redraw noise each step (ancestral, SDE, or eta above 0) overwrite it; euler, "
                "heun, dpm_2 and dpmpp_2m keep it."
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
                        "Latent frames the noise carries over: 0.0 = ordinary noise, as "
                        "RandomNoise; 2.0 = 0.63 of ordinary frame-to-frame change; 8.0 = 0.34; "
                        "25.0 = 0.20; 128.0 = 0.09. A latent frame is about 3.4 output frames on "
                        "MiniMax H3, 4 on Wan, 8 on LTX."
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
