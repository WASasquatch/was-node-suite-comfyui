"""Taking the held frames back out of a refined MiniMax H3 clip."""

from __future__ import annotations

import numpy as np
from comfy_api.latest import io

from ...modules.compat.types import H3_DEROPE
from ...modules.latent import h3_derope, h3_extend
from ...modules.model import release_models

#: Which showing of each held frame Recover keeps.
PICKS = ("first", "nearest")


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
                "Put a clip refined after H3 De-RoPE Stretch back on its source timing, with the "
                "source audio for Create Video. With the video VAE wired it also hands back an "
                "H3 latent, ready for H3 Extend Append in a loop."
            ),
            inputs=[
                io.Image.Input(
                    "images", optional=True,
                    tooltip="The refined clip, decoded from the sampler's output. Leave empty "
                            "to decode it from latent.",
                ),
                H3_DEROPE.Input("derope", optional=True,
                                tooltip="The derope output of H3 De-RoPE Stretch."),
                io.Audio.Input(
                    "audio", optional=True,
                    tooltip="Leave empty to pass out the source audio, which matches the "
                            "recovered frames. Wired, audio decoded from the refining pass is sped "
                            "back up to match and comes back rough.",
                ),
                io.Latent.Input(
                    "latent", optional=True,
                    tooltip="The sampler's output for the stretched latent. Decoded scene by "
                            "scene with vae where images is empty.",
                ),
                io.Vae.Input(
                    "vae", optional=True,
                    tooltip="The H3 video VAE. Decodes latent where images is empty, and "
                            "encodes the recovered frames for the latent output.",
                ),
                io.Combo.Input(
                    "pick", options=list(PICKS), default=PICKS[0], optional=True,
                    tooltip="Which showing of each held frame is kept: `first` = the first "
                            "showing; `nearest` = the one that looks most like the source frame, "
                            "for a refine that drifted through its holds.",
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
                io.Latent.Output(
                    display_name="latent",
                    tooltip="The recovered frames as an H3 latent with the source's audio, "
                            "for H3 Extend Append's sampled input or a decode. Empty without vae.",
                ),
            ],
        )

    @classmethod
    def execute(cls, derope=None, images=None, audio=None, latent=None,
                vae=None, pick=PICKS[0]) -> io.NodeOutput:
        """Keep the first showing of every source frame, retime the pass's audio and re-encode.

        Raises:
            ValueError: derope is not wired, neither images nor latent is wired, latent is
                wired without vae, or the clip is shorter than the stretch made it.
        """
        if getattr(derope, "holds", None) is None:
            raise ValueError(
                "H3 De-RoPE Recover needs the plan H3 De-RoPE Stretch answers on its derope "
                "output. Wire that output into derope."
            )
        decoded = images is None
        if decoded:
            if latent is None:
                raise ValueError(
                    "H3 De-RoPE Recover has no clip. Wire the refined clip's frames into "
                    "images, or the sampler's output into latent with the H3 video VAE into vae"
                )
            if vae is None:
                raise ValueError(
                    "H3 De-RoPE Recover was given a latent and no vae to decode it. Wire the H3 "
                    "video VAE into vae, or decode the latent and wire the frames into images"
                )
            release_models()
            images = h3_derope.decode_scenes(vae, h3_derope.video_of(latent),
                                             h3_extend.scene_starts(latent))
        picks, moved = None, ""
        guide = getattr(derope, "guide", None)
        if pick == "nearest" and guide is not None and int(guide.shape[0]) == len(derope.holds) \
                and int(images.shape[0]) >= int(sum(derope.holds)):
            picks = h3_derope.nearest_showings(guide, h3_derope.guide_frames(images), derope.holds)
            firsts = np.concatenate([[0], np.cumsum(derope.holds)[:-1]]).astype(int)
            shift = np.asarray(picks) - firsts
            changed = int(np.count_nonzero(shift))
            moved = (f"; {changed} of {len(picks)} frames taken from a nearer showing, the furthest "
                     f"{int(np.abs(shift).max())} away" if changed else "; every first showing nearest")
        recovered = h3_derope.recover(images, derope.holds, picks)
        frames = recovered[:derope.source]
        rebuilt = None
        if vae is not None:
            if not decoded:
                release_models()
            rebuilt = h3_derope.recovered_latent(vae, recovered, derope)
            release_models()
        sound = derope.audio
        if audio is not None:
            sound = h3_derope.squeeze_audio(audio, derope.holds, derope.source, derope.fps)
        report = (f"{images.shape[0]} frames -> {frames.shape[0]} at {derope.fps:g} fps"
                  + moved
                  + ("; decoded from the latent" if decoded else "")
                  + (f"; encoded again as {h3_derope.video_of(rebuilt).shape[2]} latent rows"
                     + (" with the source audio" if derope.sound is not None
                        else " with silence" if derope.joint else "")
                     if rebuilt is not None else "")
                  + ("; pass audio retimed" if audio is not None else ""))
        return io.NodeOutput(frames, sound, report, rebuilt)
