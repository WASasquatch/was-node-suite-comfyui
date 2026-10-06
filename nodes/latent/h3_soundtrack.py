"""One long soundtrack laid under a multi-scene MiniMax H3 video, a window at a time."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.latent import h3_extend, h3_soundtrack

logger = log.get_logger("nodes.h3_soundtrack")

NODE_NAME = "H3 Soundtrack"

WINDOW_HINT = (
    "The window to sample, from H3 Extend Window's window output, or an Empty MiniMax H3 AV "
    "Latent for a single shot. Passed on to the sampler's latent input."
)

AUDIO_HINT = (
    "The whole track, as Load Audio's output: a music bed or a recorded dialogue. Any sample "
    "rate; a mono track plays on both sides. The same track goes into every segment."
)

AUDIO_VAE_HINT = (
    "The H3 audio VAE, as Load VAE set to `minimax_h3_audio_vae_fp32`. It encodes the track "
    "once and every later segment reuses that."
)

OFFSET_HINT = (
    "Where the track starts on the finished video, in seconds: `0` = with the first frame; "
    "`2.5` = 2.5 s in, with the model's own sound before it; `-10` = skips the track's first "
    "10 s."
)

STRENGTH_HINT = (
    "How firmly the track is held under the new picture: `0` = ignored, the window passes "
    "through; `0.5` = guides the sound, which may drift from it; `1.0` = the track exactly."
)

RELEASE_HINT = (
    "Audio steps the hold eases over where the track starts, ends or loops inside the video, "
    "at 40 a second: `0` = a hard edge; `8` = 0.2 s; `20` = half a second for the model to "
    "blend in and out."
)

PAST_END_HINT = (
    "What plays once the track runs out before the video does: `model sound` = whatever the "
    "model makes; `silence` = held quiet; `loop` = the track starts again, the seam eased over "
    "`release`."
)


class H3Soundtrack(io.ComfyNode):
    """Write each window's slice of one long track into its sound, held there by the mask."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASH3Soundtrack",
            display_name=NODE_NAME,
            search_aliases=[
                "WASH3Soundtrack",
                NODE_NAME,
                "minimax h3 soundtrack",
                "music bed",
                "dialogue track",
                "audio under video",
                "h3 audio",
            ],
            category="WAS Suite/Latent/Video",
            description=(
                "Lay one long soundtrack, a music bed or a recorded dialogue, under a whole "
                "MiniMax H3 video built a segment at a time. Place it between H3 Extend "
                "Window's window and the sampler: each segment samples its picture under its "
                "own slice of the same track, so the sound runs on unbroken across carries and "
                "cuts. `offset_seconds` sets where the track starts on the video, `strength` "
                "how firmly it is held, and `past_end` what plays once it runs out. Sound the "
                "window already carries from the clip so far stays as it was. The track is "
                "encoded once and every later segment reuses it."
            ),
            inputs=[
                io.Latent.Input("window", tooltip=WINDOW_HINT),
                io.Audio.Input("audio", tooltip=AUDIO_HINT),
                io.Vae.Input("audio_vae", tooltip=AUDIO_VAE_HINT),
                io.Float.Input(
                    "offset_seconds", default=0.0, min=-36000.0, max=36000.0, step=0.025,
                    tooltip=OFFSET_HINT,
                ),
                io.Float.Input(
                    "strength", default=1.0, min=0.0, max=1.0, step=0.05,
                    tooltip=STRENGTH_HINT,
                ),
                io.Int.Input(
                    "release", default=h3_extend.AUDIO_RELEASE, min=0, max=64,
                    tooltip=RELEASE_HINT,
                ),
                io.Combo.Input(
                    "past_end", options=list(h3_soundtrack.PAST_END),
                    default=h3_soundtrack.MODEL_SOUND, tooltip=PAST_END_HINT,
                ),
            ],
            outputs=[
                io.Latent.Output(
                    display_name="window",
                    tooltip="The window holding its slice of the track, for the sampler's latent input.",
                ),
                io.String.Output(
                    display_name="report",
                    tooltip=(
                        "Which stretch of the track the window holds, how firmly, and under "
                        "which frames of the finished video."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, window, audio, audio_vae, offset_seconds=0.0, strength=1.0,
                release=h3_extend.AUDIO_RELEASE,
                past_end=h3_soundtrack.MODEL_SOUND) -> io.NodeOutput:
        """Encode the track once, write this window's slice into its sound and mask it.

        Raises:
            ValueError: The window is not an H3 joint latent, the audio holds no sound, or
                audio_vae is not the H3 audio VAE.
        """
        h3_extend.split(window)
        if float(strength) <= 0.0:
            return io.NodeOutput(window, "strength is 0, so the window passes on without the track")
        encoded, reused = h3_soundtrack.track_for(audio_vae, audio)
        laid, report = h3_soundtrack.lay(window, encoded, offset_seconds, strength, release,
                                         past_end)
        source = ("reusing the encoded track" if reused else
                  f"encoded the {encoded.seconds:.2f} s track once in {encoded.pieces} "
                  f"piece{'s' if encoded.pieces != 1 else ''}")
        report = f"{report}; {source}"
        logger.info("%s: %s", NODE_NAME, report)
        return io.NodeOutput(laid, report)
