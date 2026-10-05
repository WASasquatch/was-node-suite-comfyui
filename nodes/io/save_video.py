"""Write frames, sound or a whole video to a video file."""

from __future__ import annotations

import os

from comfy_api.latest import io, ui

from ...modules.io import rooted
from ...modules import log
from ...modules.io.naming import next_name
from ...modules.util import sandbox
from .image_save import subfolder_of

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
                "Write a video file from an image batch, from a video, or from either with a "
                "sound track alongside. The name follows the same prefix, delimiter and "
                "numbering as Image Save, and the pack's tokens resolve in it. Core's Save "
                "Video takes a video only and has no audio input."
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
                    default="auto",
                    tooltip=(
                        "File container, read only on the two ComfyUI codecs. auto writes "
                        "mp4. The other codecs each bring their own: mkv for FFV1 and H264, "
                        "mov for PRORES, webm for VP90, mp4 for the rest."
                    ),
                ),
                io.Combo.Input(
                    "codec",
                    options=codecs(),
                    default="ComfyUI Auto",
                    tooltip=(
                        "How the video is encoded. 'ComfyUI Auto' copies a wired video "
                        "through without re-encoding it, so nothing is lost; 'ComfyUI H264' "
                        "re-encodes to mp4. The rest are the pack's own: FFV1 and PRORES are "
                        "lossless, AV01 and VP90 make the smallest files, AVC1 and H265 play "
                        "everywhere."
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
                        "Quality, lower is better and larger. 0 leaves the encoder's default; "
                        "18 is near-lossless, 23 typical, 28 small."
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
        container="auto", codec="auto", crf=0.0, overwrite=False,
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

        Returns:
            The video, the full path and the file name.

        Raises:
            ValueError: Neither images nor a video was connected.
        """
        import folder_paths
        from comfy_api.latest._util.video_types import VideoCodec, VideoContainer

        from ...modules.media import reader

        if video is None and images is None:
            raise ValueError(
                "Save Video has nothing to write. Connect images, or a video, or both."
            )

        written = video if video is not None else reader.to_video(images, float(fps), audio)
        through_comfy = codec in COMFY_CODECS

        below, _, leaf = (filename_prefix or "").replace("\\", "/").rpartition("/")
        destination = rooted.destination(root, below)
        filename_prefix = leaf
        os.makedirs(destination, exist_ok=True)

        if through_comfy:
            extension = VideoContainer.get_extension(container) or "mp4"
        else:
            from ...modules.media import video as encoding

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
            encoding.write_frames(str(target), frames, rate, codec, audio=sound)
            if images is None and hasattr(written, "saved"):
                written.saved()
        logger.info("wrote %s", target)

        # A preview is addressed as a name and a subfolder of one of ComfyUI's own
        # directories, so a file written anywhere else is saved without one.
        subfolder = subfolder_of(str(destination), folder_paths.get_output_directory())
        preview = None
        if subfolder is not None:
            preview = ui.PreviewVideo([ui.SavedResult(name, subfolder, io.FolderType.output)])
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
