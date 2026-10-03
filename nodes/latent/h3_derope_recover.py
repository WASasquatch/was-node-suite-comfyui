"""Taking the held frames back out of a refined MiniMax H3 clip."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import H3_DEROPE
from ...modules.latent import h3_derope


class H3DeRopeRecover(io.ComfyNode):
    """Return a refined stretched clip to its source timing."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASH3DeRopeRecover",
            display_name="H3 De-RoPE Recover",
            search_aliases=[
                "WASH3DeRopeRecover",
                "H3 De-RoPE Recover",
                "de-rope",
                "derope",
                "minimax h3",
            ],
            category="WAS Suite/Latent/Video",
            description=(
                "Take the frames H3 De-RoPE Stretch held back out of the decoded, refined "
                "clip, so it plays at its source length and speed, with the source audio "
                "beside it for Create Video. Audio decoded from the pass can be wired instead; "
                "it is sped back up with its pitch kept and comes back rough."
            ),
            inputs=[
                io.Image.Input(
                    "images", tooltip="The refined clip, decoded from the sampler's output.",
                ),
                H3_DEROPE.Input("derope", tooltip="The derope output of H3 De-RoPE Stretch."),
                io.Audio.Input(
                    "audio", optional=True,
                    tooltip="Leave empty to pass out the source audio, which matches the "
                            "recovered frames. Wired, audio decoded from the refining pass is sped "
                            "back up to match and comes back rough.",
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip="One frame per source frame, at the source timing.",
                ),
                io.Audio.Output(
                    display_name="audio",
                    tooltip="The pass's audio at the source timing, or the source audio when "
                            "none is wired. Empty when neither exists.",
                ),
                io.String.Output(
                    display_name="report", tooltip="Frames in and frames out.",
                ),
            ],
        )

    @classmethod
    def execute(cls, images, derope, audio=None) -> io.NodeOutput:
        """Keep the first showing of every source frame and retime the pass's audio.

        Raises:
            ValueError: derope is not wired, or the clip is shorter than the stretch made it.
        """
        if getattr(derope, "holds", None) is None:
            raise ValueError(
                "H3 De-RoPE Recover needs the plan H3 De-RoPE Stretch answers on its derope "
                "output. Wire that output into derope."
            )
        frames = h3_derope.recover(images, derope.holds)[:derope.source]
        sound = derope.audio
        if audio is not None:
            sound = h3_derope.squeeze_audio(audio, derope.holds, derope.source, derope.fps)
        report = (f"{images.shape[0]} frames -> {frames.shape[0]} at {derope.fps:g} fps"
                  + ("; pass audio retimed" if audio is not None else ""))
        return io.NodeOutput(frames, sound, report)
