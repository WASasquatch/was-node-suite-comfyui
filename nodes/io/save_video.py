"""Write frames, sound or a whole video to a video file."""

from __future__ import annotations

import os

from comfy_api.latest import io, ui

from ...modules.io import rooted
from ...modules import log
from ...modules.io.naming import next_name
from ...modules.util import sandbox
from .image_save import preview_place

logger = log.get_logger("nodes.io")


def containers() -> list[str]:
    """Container names the installed comfy_api offers."""
    from comfy_api.latest._util.video_types import VideoContainer

    return list(VideoContainer.as_input())


#: Codec options written through ComfyUI's own writer, mapped to the name it takes. A video
#: passed straight through on one of these is stream-copied rather than re-encoded.
COMFY_CODECS = {"ComfyUI Auto": "auto", "ComfyUI H264": "h264"}


def codecs() -> list[str]:
    """Codec names the widget offers: ComfyUI's writer first, then the pack's own."""
    from ...modules.media import video

    return [*COMFY_CODECS, *video.codec_options()]


def color_spaces() -> list[str]:
    """Colour options the widget offers, the default first."""
    from ...modules.media import video

    return list(video.COLOR_SPACES)


class SaveVideo(io.ComfyNode):
    """Write a video from frames and sound, or from a video, under a numbered name."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASSaveVideo",
            display_name="Save Video (Advanced)",
            search_aliases=[
                "WASSaveVideo",
                "Save Video",
                "write video",
                "export video",
                "save mp4",
                "render out",
            ],
            category="WAS Suite/IO",
            is_output_node=True,
            description=(
                "Write a video file from an image batch or a video, with an optional sound "
                "track. The name follows the same prefix, delimiter and numbering as Image "
                "Save, and the pack's tokens resolve in it."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    optional=True,
                    tooltip=(
                        "Frames to encode, in order, at the rate below. Leave unconnected "
                        "when a video is wired instead."
                    ),
                ),
                io.Video.Input(
                    "video",
                    optional=True,
                    tooltip=(
                        "A whole video to write out. Used in preference to images when both "
                        "are wired."
                    ),
                ),
                io.Audio.Input(
                    "audio",
                    optional=True,
                    tooltip=(
                        "Sound laid under the frames. Read only when images are encoded, "
                        "since a video already carries its own."
                    ),
                ),
                io.Float.Input(
                    "fps",
                    default=24.0,
                    min=0.01,
                    max=240.0,
                    step=0.01,
                    tooltip="Rate the frames play at. Read only when images are encoded.",
                ),
                io.Combo.Input(
                    "root",
                    options=rooted.options(),
                    tooltip=(
                        "Which folder the files land in: ComfyUI's own 'output' or 'temp', "
                        "or any folder added under paths.allow_write in config.yaml, listed "
                        "by its own name. The name below it says the rest."
                    ),
                ),
                io.String.Input(
                    "filename_prefix",
                    default="ComfyUI",
                    tooltip=(
                        "Text before the number. Tokens resolve here, so [time(%Y-%m-%d)] "
                        "becomes the date and [hostname] the machine name."
                    ),
                ),
                io.String.Input(
                    "filename_delimiter",
                    default="_",
                    tooltip="What separates the prefix from the number: clip_0001.mp4.",
                ),
                io.Int.Input(
                    "filename_number_padding",
                    default=4,
                    min=1,
                    max=9,
                    tooltip="Digits the number is padded to. 4 gives 0001.",
                ),
                io.Combo.Input(
                    "container",
                    options=containers(),
                    default="mp4",
                    tooltip=(
                        "File container, read only on the two ComfyUI codecs: `mp4`, or "
                        "`auto`, which also writes mp4. The other codecs each bring their own: "
                        "mkv for FFV1 and H264, mov for PRORES, webm for VP90, mp4 for the rest."
                    ),
                ),
                io.Combo.Input(
                    "codec",
                    options=codecs(),
                    default="ComfyUI H264",
                    tooltip=(
                        "`ComfyUI H264` = H.264, copying H.264 video; `ComfyUI Auto` = any "
                        "wired video copied as is; `AVC1`/`H264` = H.264 in mp4/mkv; `MP4V` = "
                        "MPEG-4; `FFV1` = lossless; `PRORES` = 10-bit; `AV01`/`VP90` = "
                        "smallest; `H265`/`HEVC` = small, rare in browsers. Firefox shows no "
                        "H.264 above 4096x2304."
                    ),
                ),
                io.Float.Input(
                    "crf",
                    default=0.0,
                    min=0.0,
                    max=51.0,
                    step=0.5,
                    optional=True,
                    tooltip=(
                        "Quality for H.264, H.265, AV1 and VP9, lower is better and larger. 0 "
                        "leaves the encoder's default; 18 is near-lossless, 23 typical, 28 small."
                    ),
                ),
                io.Boolean.Input(
                    "overwrite",
                    default=False,
                    optional=True,
                    tooltip=(
                        "Write prefix.mp4 every run instead of a new number, so the file is "
                        "replaced rather than added to."
                    ),
                ),
                io.Combo.Input(
                    "color_space",
                    options=color_spaces(),
                    default="SDR sRGB",
                    optional=True,
                    tooltip=(
                        "Tags only, no conversion. SDR sRGB = ordinary pictures; Rec.709 "
                        "(BT.1886) = gamma 2.4 codes; Display P3 = P3, sRGB curve; PQ (HDR10) "
                        "and HLG = Rec.2020 codes from Image Color Space, written 10-bit by "
                        "H265, PRORES or AV01."
                    ),
                ),
                io.Boolean.Input(
                    "dither",
                    default=False,
                    optional=True,
                    tooltip=(
                        "false = plain rounding; true = an ordered dither, so slow gradients "
                        "such as skies band less while pure black and white stay exact. "
                        "Strongest on FFV1, weakest on H.264 at crf 18 to 23. Read by the "
                        "pack's 8-bit codecs only."
                    ),
                ),
            ],
            outputs=[
                io.Video.Output(
                    display_name="video",
                    tooltip="What was written, so the node can sit in the middle of a chain.",
                ),
                io.String.Output(
                    display_name="file_path",
                    tooltip="Full path of the file written.",
                ),
                io.String.Output(
                    display_name="filename",
                    tooltip="Its name alone, without the folder.",
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
        )

    @classmethod
    def execute(
        cls, images=None, video=None, audio=None, fps=24.0, root=rooted.DEFAULT,
        filename_prefix="ComfyUI", filename_delimiter="_", filename_number_padding=4,
        container="mp4", codec="ComfyUI H264", crf=0.0, overwrite=False,
        color_space="SDR sRGB", dither=False,
    ) -> io.NodeOutput:
        """Write the file and answer where it went.

        Args:
            images: Frames to encode.
            video: A video to write out instead.
            audio: Sound laid under the frames.
            fps: Rate the frames play at.
            root: Which folder to write into, as :func:`modules.io.rooted.options` names them.
            filename_prefix: Text before the number.
            filename_delimiter: What separates the prefix from the number.
            filename_number_padding: Digits the number is padded to.
            container: File container.
            codec: Video codec.
            crf: Quality, 0 for the encoder's default.
            overwrite: Reuse one name rather than numbering.
            color_space: What the frames are, written as the stream's colour tags.
            dither: Dither the 8-bit quantisation.

        Returns:
            The video, the full path and the file name.

        Raises:
            ValueError: Neither images nor a video was connected, the colour option is
                unknown, or it or the dither needs one of the pack's codecs.
        """
        from comfy_api.latest._util.video_types import VideoCodec, VideoContainer

        from ...modules.media import reader
        from ...modules.media import video as encoding

        if video is None and images is None:
            raise ValueError(
                "Save Video has nothing to write. Connect images, or a video, or both."
            )
        color_space = color_space or encoding.SDR_SRGB
        if color_space not in encoding.COLOR_SPACES:
            raise ValueError(
                f"Save Video color_space must be one of {', '.join(encoding.COLOR_SPACES)}, "
                f"not {color_space!r}."
            )
        through_comfy = codec in COMFY_CODECS
        if through_comfy and (color_space != encoding.SDR_SRGB or dither):
            wanted = "dither" if color_space == encoding.SDR_SRGB else f"color_space {color_space!r}"
            deep = "H265, HEVC, PRORES or AV01" if encoding.COLOR_TAGS[color_space].bits >= 10 \
                else "AVC1, H264 or H265"
            raise ValueError(
                f"Save Video's {wanted} is written by the pack's own codecs, and {codec} hands "
                f"the file to ComfyUI's writer, which always writes SDR sRGB without dither. "
                f"Pick {deep} as the codec."
            )

        written = video if video is not None else reader.to_video(images, float(fps), audio)

        below, _, leaf = (filename_prefix or "").replace("\\", "/").rpartition("/")
        destination = rooted.destination(root, below)
        filename_prefix = leaf
        os.makedirs(destination, exist_ok=True)

        if through_comfy:
            extension = VideoContainer.get_extension(container) or "mp4"
        else:
            extension = encoding.container_extensions()[codec.lower()].lstrip(".")

        name = next_name(
            str(destination), filename_prefix.strip() or "ComfyUI", filename_delimiter,
            int(filename_number_padding), extension, overwrite=bool(overwrite),
        )
        target = sandbox.resolve_write_file(destination, name)

        if through_comfy:
            written.save_to(
                str(target),
                format=VideoContainer(container),
                codec=VideoCodec(COMFY_CODECS[codec]),
                metadata=cls.embedded(),
                crf=float(crf) if crf and crf > 0 else None,
            )
        else:
            frames, sound, rate = images, audio, float(fps)
            if frames is None:
                rate = float(written.get_frame_rate())
                lazy = getattr(written, "lazy_frames", None)
                if lazy is not None:
                    frames = lazy()
                    if sound is None:
                        sound = written.cache().audio(written.start, written.stop)
                else:
                    parts = written.get_components()
                    frames, sound = parts.images, (sound if sound is not None else parts.audio)
            encoding.write_frames(
                str(target), frames, rate, codec, audio=sound, color=color_space,
                crf=float(crf or 0.0), dither=bool(dither),
            )
            if images is None and hasattr(written, "saved"):
                written.saved()
        logger.info("wrote %s", target)

        # A file outside ComfyUI's output, input and temp folders is saved with no preview.
        place = preview_place(str(destination))
        preview = None
        if place is not None:
            preview = ui.PreviewVideo([ui.SavedResult(name, *place)])
        return io.NodeOutput(written, str(target), name, ui=preview)

    @classmethod
    def embedded(cls) -> dict | None:
        """The workflow to write into the file, or None when there is nothing to write."""
        metadata = {}
        try:
            if cls.hidden.extra_pnginfo is not None:
                metadata.update(cls.hidden.extra_pnginfo)
            if cls.hidden.prompt is not None:
                metadata["prompt"] = cls.hidden.prompt
        except Exception:
            return None
        return metadata or None
