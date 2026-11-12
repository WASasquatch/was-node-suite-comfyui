"""Pictures, a clip and a voice encoded once and saved as a MiniMax H3 RefMod."""

from __future__ import annotations

import os

from comfy_api.latest import io

from ...modules import log
from ...modules.interface import preview, run_result
from ...modules.latent import h3_assets, h3_extend, h3_references, h3_refmods
from .h3_asset import at_model_rate

logger = log.get_logger("nodes.h3_save_refmod")

NODE_NAME = "MiniMax H3 Save RefMod"

#: Longest side a picture is encoded at, in pixels.
SIZES = ("512", "768", "1024", "1536")


class MiniMaxH3SaveRefMod(io.ComfyNode):
    """Encode pictures, a clip and a voice once and save them as a RefMod in models/refmods."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASMiniMaxH3SaveRefMod",
            display_name=NODE_NAME,
            search_aliases=[
                "WASMiniMaxH3SaveRefMod",
                NODE_NAME,
                "save refmod",
                "create refmod",
                "refmod",
                "minimax h3",
                "h3 reference",
                "saved reference",
            ],
            category="WAS Suite/Latent/Video",
            description=(
                "Encode a character, place or voice once and save it in models/refmods as a "
                "RefMod, a reference any MiniMax H3 prompt can name again without encoding it. "
                "Pictures become one <Picture N> or <Video N>, a clip a <Video N>, a voice an "
                "<Audio N>. MiniMax H3 RefMod reads it back."
            ),
            is_output_node=True,
            inputs=[
                io.Vae.Input("vae", tooltip="The H3 video VAE, which encodes the pictures and the clip."),
                io.String.Input(
                    "name", default="refmod",
                    tooltip=(
                        "The file in models/refmods, as `alice` or `cast/alice`. A RefMod this "
                        "node saved under the name is replaced; any other file there is left alone."
                    ),
                ),
                io.Combo.Input(
                    "size", options=list(SIZES), default="768",
                    tooltip=(
                        "Longest side a picture is encoded at, in pixels: `512` = 256 tokens for "
                        "a square picture; `768` = 576; `1024` = 1024. More tokens keep more detail "
                        "and slow every step."
                    ),
                ),
                io.Float.Input(
                    "voice_seconds", default=10.0, min=1.0, max=60.0, step=0.5,
                    tooltip="Seconds of audio kept from its start, as `10`. Each second costs 80 tokens.",
                ),
                io.Image.Input(
                    "image", optional=True,
                    tooltip=(
                        "One picture, a `<Picture N>`; or several pictures of one subject at one "
                        "size, together a `<Video N>`."
                    ),
                ),
                io.Video.Input(
                    "video", optional=True,
                    tooltip="A clip, a `<Video N>`, kept from its start for up to 362 frames at 24 fps.",
                ),
                io.Audio.Input("audio", optional=True, tooltip="A voice or sound, an `<Audio N>`."),
                io.Vae.Input(
                    "audio_vae", optional=True,
                    tooltip="The H3 audio VAE, which encodes audio. Needed only with audio.",
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="refmod",
                    tooltip="The saved RefMod's name, for refmod on MiniMax H3 RefMod.",
                ),
                io.String.Output(display_name="report", tooltip="What was encoded and where it was saved."),
            ],
        )

    @classmethod
    def execute(cls, vae, name="refmod", size="768", voice_seconds=10.0, image=None, video=None,
                audio=None, audio_vae=None) -> io.NodeOutput:
        """Encode what is wired and write it as one RefMod.

        Raises:
            ValueError: Nothing is wired, audio arrives with no audio VAE, a clip is under five
                frames, or another file already holds the name.
            PathNotAllowed: The name leaves models/refmods.
        """
        import torch

        if image is None and video is None and audio is None:
            raise ValueError(
                f"{NODE_NAME} has nothing to save. Wire pictures into image, a clip into video "
                f"or a voice into audio"
            )
        if audio is not None and audio_vae is None:
            raise ValueError(
                f"{NODE_NAME} has audio and no audio_vae to encode it with. Wire the H3 audio VAE "
                f"into audio_vae, or unwire the audio"
            )
        if str(size) not in SIZES:
            raise ValueError(f"size `{size}` is not one of {', '.join(SIZES)}")
        path, menu = h3_refmods.target(name)
        if h3_refmods.foreign(path):
            raise ValueError(
                f"models/refmods already holds `{menu}`, written by something other than {NODE_NAME}, "
                f"and it is left as it is. Type another name"
            )
        stem = os.path.splitext(os.path.basename(path))[0]

        members, parts, shown = [], [], None
        if image is not None:
            count = int(image.shape[0])
            if count > h3_refmods.MOST_FRAMES:
                raise ValueError(
                    f"image holds {count} pictures, and a RefMod holds at most "
                    f"{h3_refmods.MOST_FRAMES}. Pick fewer"
                )
            wide, high = int(image.shape[2]), int(image.shape[1])
            pictures = [h3_references.encode_picture(vae, image[place:place + 1], wide, high, str(size))
                        for place in range(count)]
            frames = torch.cat([picture.item["data"] for picture in pictures])
            latent = torch.cat([picture.block["latent"] for picture in pictures], dim=2)
            kind = "image" if count == 1 else "video"
            members.append({"kind": kind, "latent": latent, "source": "image" if count == 1 else "stack",
                            "shown": frames, "times": [float(place) for place in range(count)]})
            parts.append(f"{'a picture' if count == 1 else f'{count} pictures'} at {pictures[0].size}")
            shown = frames[:1]
        if video is not None:
            components = video.get_components()
            frames = at_model_rate(components.images, float(components.frame_rate or h3_extend.FPS))
            clip = h3_references.encode_video(vae, None, frames, None, h3_assets.LONGEST_CLIP, "video")
            item = clip.items[-1]
            members.append({"kind": "video", "latent": clip.block["latent"], "source": "video",
                            "shown": item["data"], "times": list(item["timestamps"])})
            parts.append(f"a clip of {clip.text}")
            shown = item["data"][:1] if shown is None else shown
        if audio is not None:
            rate = int(audio["sample_rate"])
            kept = {"waveform": audio["waveform"][..., :max(1, int(round(float(voice_seconds) * rate)))],
                    "sample_rate": rate}
            sound = h3_references.encode_sound(audio_vae, kept, "audio")
            members.append({"kind": "audio", "latent": sound.block["audio_latent"], "source": "audio",
                            "shown": None, "times": []})
            parts.append(f"{sound.seconds:.1f}s of sound")

        h3_refmods.save(path, stem, members)
        mods = h3_refmods.load(menu)
        total = sum(h3_refmods.tokens(mod) for mod in mods)
        held = ", ".join(f"<{h3_refmods.TAG_WORDS[mod.kind]} N>" for mod in mods)
        summary = f"saved {menu}: {', '.join(parts)}; {held}, {total} tokens a step"
        logger.info("%s: %s", NODE_NAME, summary)
        if shown is not None:
            preview.publish_output(shown, slot="refmod")
        run_result.publish(
            summary=summary,
            counts={"references": len(mods), "tokens": total},
            facts={"file": menu, "holds": held},
        )
        return io.NodeOutput(menu, summary, ui={"was_refmod_saved": [menu]})
