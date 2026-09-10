"""Starting noise shaped by an affine mask pattern."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import DICT
from ...modules.latent import affine
from ...modules.sampling.pattern_noise import NOISE_PATTERNS, PatternNoise
from ..latent.latent_affine import OPTIONS_HINT, STREAMS_HINT, TEMPORAL_HINT


class AffinePatternNoise(io.ComfyNode):
    """Starting noise multiplied and offset through a generated mask."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAffinePatternNoise",
            display_name="Affine Pattern Noise",
            search_aliases=[
                "WASAffinePatternNoise",
                "Affine Pattern Noise",
                "pattern noise",
                "structured noise",
                "masked noise",
                "affine noise",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Starting noise with an affine mask laid over it: the draw is multiplied by "
                "max_scale and offset by max_bias wherever the pattern is white, so the "
                "noise carries the pattern's structure instead of being flat. Wire it into "
                "the noise socket of any custom sampler in place of RandomNoise. The four "
                "patterns read off a picture are not offered, since a starting draw has no "
                "picture to read."
            ),
            inputs=[
                io.Combo.Input(
                    "pattern",
                    options=list(NOISE_PATTERNS),
                    default="white_noise",
                    tooltip=(
                        "Which mask shapes the draw. `white_noise` is one value per latent "
                        "element; `pink_noise`, `brown_noise` and `perlin` spread over "
                        "larger areas; `checker`, `bayer` and `tile_lines` repeat. Affine "
                        "Options carries each pattern's own settings."
                    ),
                ),
                io.Int.Input(
                    "noise_seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip=(
                        "The seed the draw and the mask are both taken from, as `0` or "
                        "`12345`. One seed reproduces a clip exactly."
                    ),
                ),
                io.Float.Input(
                    "max_scale",
                    default=1.1,
                    min=0.0,
                    max=10.0,
                    step=0.01,
                    round=0.001,
                    tooltip=(
                        "What the draw is multiplied by where the mask is white. 1.0 leaves "
                        "the draw alone, 1.1 lifts it a tenth, 0.5 halves it. Away from the "
                        "mask the draw is untouched."
                    ),
                ),
                io.Float.Input(
                    "max_bias",
                    default=0.0,
                    min=-2.0,
                    max=2.0,
                    step=0.01,
                    round=0.001,
                    tooltip=(
                        "What is added where the mask is white. 0.0 adds nothing, 0.05 is a "
                        "faint lift, 0.5 prints the pattern into the draw."
                    ),
                ),
                io.Combo.Input(
                    "temporal_mode",
                    options=list(affine.TEMPORAL_MODES),
                    default="static",
                    optional=True,
                    tooltip=TEMPORAL_HINT,
                ),
                io.Combo.Input(
                    "streams",
                    options=list(affine.STREAM_MODES),
                    default="video",
                    optional=True,
                    tooltip=STREAMS_HINT,
                ),
                io.Boolean.Input(
                    "normalize",
                    default=True,
                    optional=True,
                    tooltip=(
                        "`true` returns the draw at a spread of 1.0, which is the magnitude "
                        "a sampler's schedule is built for. `false` returns whatever "
                        "max_scale and max_bias worked out to."
                    ),
                ),
                io.Float.Input(
                    "clamp_sigma",
                    default=0.0,
                    min=0.0,
                    max=8.0,
                    step=0.1,
                    round=0.01,
                    tooltip=(
                        "Where the draw is cut off, in standard deviations. 0.0 leaves it "
                        "uncut. 3.0 cuts the furthest 0.3 percent of values."
                    ),
                ),
                DICT.Input(
                    "affine_options",
                    optional=True,
                    tooltip=OPTIONS_HINT,
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
    def execute(
        cls,
        pattern="white_noise",
        noise_seed=0,
        max_scale=1.1,
        max_bias=0.0,
        temporal_mode="static",
        streams="video",
        normalize=True,
        clamp_sigma=0.0,
        affine_options=None,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            PatternNoise(
                seed=int(noise_seed),
                pattern=str(pattern),
                max_scale=float(max_scale),
                max_bias=float(max_bias),
                temporal_mode=str(temporal_mode),
                streams=str(streams),
                normalize=bool(normalize),
                clamp_sigma=float(clamp_sigma),
                options=affine_options,
                node_id=cls.hidden.unique_id,
            )
        )
