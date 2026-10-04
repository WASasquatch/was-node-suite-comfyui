"""Writing video files with PyAV.

Codecs are named by four-character code. Frames are ``bgr24`` numpy arrays; pixel formats
are 4:2:0 for lossy codecs and packed BGRA for lossless ffv1.
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image

from .. import config, deps, log
from ..constants import ALLOWED_EXT
from ..util import sandbox

__all__ = [
    "AUDIO_ENCODER",
    "AUDIO_ENCODERS",
    "CODECS",
    "blend_frames",
    "ENCODERS",
    "Encoder",
    "EXTENSIONS",
    "PIXEL_FORMAT",
    "PIXEL_FORMATS",
    "VideoWriter",
    "codec_options",
    "container_extensions",
    "pil_to_bgr",
    "progress_bar",
    "read_frame",
    "require_codec",
    "resize_frame",
    "write_frames",
]

logger = log.get_logger("media.video")


def resize_frame(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Scale one frame onto a given size.

    Args:
        image: Source frame, three 8-bit channels.
        width: Width in pixels of the result.
        height: Height in pixels of the result.

    Returns:
        The scaled frame, or the source when it is already that size.
    """
    if image.shape[1] == width and image.shape[0] == height:
        return image
    source = Image.fromarray(np.ascontiguousarray(image), mode="RGB")
    return np.asarray(source.resize((max(1, width), max(1, height)), Image.BILINEAR))


def read_frame(path: str) -> np.ndarray:
    """Read an image file as a ``bgr24`` frame.

    Args:
        path: File to read.

    Returns:
        The pixels, three 8-bit channels in blue, green, red order.

    Raises:
        ValueError: The file is not an image this can read.
    """
    try:
        with Image.open(path) as source:
            return pil_to_bgr(source)
    except OSError as error:
        raise ValueError(f"{path} could not be read as an image: {error}") from error


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    """Convert a PIL image to the ``bgr24`` array layout the encoder takes."""
    return np.ascontiguousarray(np.asarray(img.convert("RGB"))[:, :, ::-1])


def blend_frames(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
    """Mix two frames of the same shape, at ``alpha`` towards the second."""
    mixed = first.astype(np.float32) * (1.0 - alpha) + second.astype(np.float32) * alpha
    return np.clip(mixed + 0.5, 0, 255).astype(np.uint8)

#: Four-character codes mapped onto the libavcodec encoder that produces them. A code
#: absent from this table is looked up as an encoder name, which is what makes a
#: ``video.extra_codecs`` entry such as ``{"libx265": ".mkv"}`` work. ``h264``, ``vp9`` and
#: ``av1`` resolve to the same encoders ComfyUI's own video helpers use.
ENCODERS = {
    "av01": "libsvtav1",
    "av1": "libsvtav1",
    "avc1": "libx264",
    "divx": "mpeg4",
    "ffv1": "ffv1",
    "h264": "libx264",
    "h265": "libx265",
    "hevc": "libx265",
    "hvc1": "libx265",
    "mjpg": "mjpeg",
    "mp4v": "mpeg4",
    "prores": "prores",
    "vp80": "libvpx",
    "vp90": "libvpx-vp9",
    "vp9": "libvpx-vp9",
    "x264": "libx264",
    "xvid": "mpeg4",
}

#: Container extension appended to the path a code is written to. The extension decides
#: the muxer, so each one here is a container that carries its codec: Matroska for ffv1's
#: BGRA and for h264, MP4 for the rest, WebM for VP9, QuickTime for ProRes.
EXTENSIONS = {
    "avc1": ".mp4",
    "ffv1": ".mkv",
    "h264": ".mkv",
    "mp4v": ".mp4",
    "av01": ".mp4",
    "h265": ".mp4",
    "hevc": ".mp4",
    "prores": ".mov",
    "vp90": ".webm",
}

#: The ``codec`` widget's options, lowercase, in the order the widget lists them. Frozen
#: at the front: v2 offered ``avc1``, ``ffv1``, ``h264`` and ``mp4v`` in that order, and a
#: combo's first option is the value a workflow saved without touching the widget stores.
#: Everything after them is an addition, which is the only change a combo's options accept.
CODECS = ("avc1", "ffv1", "h264", "mp4v", "av01", "h265", "hevc", "prores", "vp90")

#: Audio encoder laid alongside the video where the container carries sound.
AUDIO_ENCODER = "aac"

#: Audio encoder for a codec whose container or losslessness calls for another. WebM takes
#: Opus rather than AAC, and the two lossless video codecs take uncompressed audio.
AUDIO_ENCODERS = {
    "ffv1": "pcm_s16le",
    "vp90": "libopus",
    "prores": "pcm_s16le",
}

#: Pixel format requested for an encoder that :data:`PIXEL_FORMATS` does not name.
PIXEL_FORMAT = "yuv420p"

#: Matrix coefficients number of BT.709.
BT709 = 1

#: Pixel formats to try for one encoder, best first, where :data:`PIXEL_FORMAT` is wrong
#: for it. ffv1 is a lossless codec and 4:2:0 chroma subsampling is not lossless, so it
#: takes packed BGRA, which is what a ``bgr24`` frame reaches without resampling and what
#: v2's OpenCV writer produced. ProRes has no 8-bit format at all.
PIXEL_FORMATS = {
    "ffv1": ("bgra", "bgr0", "gbrp"),
    "prores": ("yuv422p10le", "yuv444p10le"),
    "prores_aw": ("yuv422p10le", "yuv444p10le"),
    "prores_ks": ("yuv422p10le", "yuv444p10le"),
}


def codec_options() -> list[str]:
    """The ``codec`` widget's options: :data:`CODECS`, uppercased.

    Returns:
        The four-character codes, uppercased, in widget order.
    """
    # A saved workflow carries the value its combo held and validation rejects one the
    # options do not list, so a per-machine menu would refuse workflows saved elsewhere.
    return [code.upper() for code in CODECS]


def container_extensions() -> dict[str, str]:
    """The container extension for every codec, with ``video.extra_codecs`` folded in.

    Returns:
        Lowercase code to extension, including the dot.
    """
    extensions = dict(EXTENSIONS)
    for code, extension in (config.load_config()["video"]["extra_codecs"] or {}).items():
        text = str(extension).strip()
        if text:
            extensions[str(code).lower()] = text if text.startswith(".") else f".{text}"
    return extensions


def write_frames(path: str, frames, fps: float, codec: str, audio=None) -> str:
    """Encode an image batch to a video file with one of the pack's codecs.

    Args:
        path: File to write, without an extension.
        frames: ``(batch, height, width, channels)`` in ``[0, 1]``.
        fps: Rate the frames play at.
        codec: Four-character code from :data:`CODECS`, in any case.
        audio: ``{"waveform", "sample_rate"}`` laid under the frames, or ``None``.

    Returns:
        The path written, carrying the codec's container extension.

    Raises:
        DependencyError: av is not installed.
        ValueError: The codec has no encoder in this build of av, or the batch is empty.
    """
    from fractions import Fraction

    code = codec.lower()
    extensions = container_extensions()
    if code not in extensions:
        raise ValueError(f"Video codec must be one of {', '.join(codec_options())}, not {codec!r}")
    if len(frames) == 0:
        raise ValueError("Save Video was given an empty image batch, so there is nothing to encode.")

    target = str(path)
    if not target.lower().endswith(extensions[code]):
        target += extensions[code]

    height, width = int(frames.shape[1]), int(frames.shape[2])
    rate = Fraction(float(fps)).limit_denominator(65535)

    with Encoder(target, code, width, height, rate, audio=audio, color_space="sRGB") as encoder:
        for frame in frames:
            picture = frame.detach().cpu().numpy() if hasattr(frame, "detach") else np.asarray(frame)
            picture = np.round(np.clip(picture[..., :3], 0.0, 1.0) * 255.0).astype(np.uint8)
            encoder.write(np.ascontiguousarray(picture[..., ::-1]))
    return target


def require_codec(codec: str):
    """Resolve a code to the encoder that produces it, or say what is missing.

    Args:
        codec: Four-character code, or an encoder name for a configured extra codec.

    Returns:
        The av ``Codec`` to add a stream with.

    Raises:
        DependencyError: av is not installed.
        ValueError: This build of av has no encoder for ``codec``.
    """
    av = deps.require("av")

    name = ENCODERS.get(codec.lower(), codec.lower())
    try:
        return av.codec.Codec(name, "w")
    except Exception as error:
        installed = ", ".join(_installed_codecs(av)) or "none of them"
        raise ValueError(
            f"the {codec.upper()} codec needs the {name} encoder, which this build of av "
            f"({av.__version__}) does not have: {error}. The codecs this build can write "
            f"are: {installed}"
        ) from error


def _installed_codecs(av) -> list[str]:
    """The :data:`CODECS` entries this build of av has an encoder for, uppercased."""
    installed = []
    for code in CODECS:
        try:
            av.codec.Codec(ENCODERS.get(code, code), "w")
        except Exception:
            continue
        installed.append(code.upper())
    return installed


class VideoWriter:
    """Renders images into a video file.

    Args:
        transition_frames: Blended frames rendered between one image and the next.
        fps: Frame rate of a newly created video. Appending to an existing one keeps that
            file's rate instead.
        still_image_delay_sec: Seconds each image is held for, rounded to whole frames at
            ``fps`` and never fewer than one, so a delay shorter than a single frame still
            leaves a file behind rather than an empty container.
        max_size: Longest edge, in pixels, images are scaled to before encoding.
        codec: Four-character code from :data:`CODECS`, in either case, or one registered
            by ``video.extra_codecs``. It decides the container extension the writer
            appends. A code that is neither falls back to ``mp4v``. Whether this build of
            av can encode it is settled by :func:`require_codec` on the first write.
    """

    def __init__(
        self,
        transition_frames: int = 30,
        fps: int = 25,
        still_image_delay_sec: float = 2,
        max_size: int = 512,
        codec: str = "mp4v",
    ):
        self.transition_frames = transition_frames
        self.fps = fps
        self.still_image_delay_frames = max(1, round(still_image_delay_sec * fps))
        self.max_size = int(max_size)
        self.valid_codecs = list(CODECS)
        self.extensions = dict(EXTENSIONS)
        self.add_codecs(config.load_config()["video"]["extra_codecs"])
        code = codec.lower()
        if code not in self.valid_codecs:
            logger.warning("unknown codec %r; writing mp4v instead", codec)
            code = "mp4v"
        self.codec = code

    def write(self, image: Image.Image, video_path: str) -> str:
        """Append one image to a video, creating it if absent.

        Args:
            image: The image to append.
            video_path: Destination path without an extension; the codec's container
                extension is appended and the result returned.

        Returns:
            The path written, extension included.

        Raises:
            DependencyError: av is not installed.
            ValueError: This build of av cannot encode the codec, the image or the existing
                video has no usable dimensions, or the codec cannot encode at that size.
        """
        deps.require("av")

        import av

        # Settled before the destination is touched, so a missing encoder is reported
        # instead of a half-written file.
        require_codec(self.codec)

        video_path = str(sandbox.resolve_write(video_path + self.extensions[self.codec]))
        end_image = self.rescale(self.pil2cv(image), self.max_size)

        if not os.path.isfile(video_path):
            height, width = end_image.shape[:2]
            if width <= 0 or height <= 0:
                raise ValueError("Invalid image dimensions")

            progress = progress_bar(self.still_image_delay_frames)
            with Encoder(video_path, self.codec, width, height, self.fps, color_space="sRGB") as encoder:
                for _ in range(self.still_image_delay_frames):
                    encoder.write(end_image)
                    progress.update()

            logger.info("created new video at: %s", video_path)
            return video_path

        # The name is rebuilt from its own stem, leaving any directory along the way that
        # carries the same extension untouched. The sibling is resolved again.
        stem, extension = os.path.splitext(os.path.basename(video_path))
        temp_file_path = str(
            sandbox.resolve_write_file(os.path.dirname(video_path), stem + "_temp" + extension)
        )

        with av.open(video_path) as source:
            stream = source.streams.video[0]
            stream.thread_type = "AUTO"
            width = stream.codec_context.width
            height = stream.codec_context.height
            if width <= 0 or height <= 0:
                raise ValueError("Invalid video dimensions")

            rate = stream.average_rate or self.fps
            copied = stream.frames or 0
            progress = progress_bar(copied + self.transition_frames + self.still_image_delay_frames)
            last_frame = None

            with Encoder(temp_file_path, self.codec, width, height, rate, color_space="sRGB") as encoder:
                for frame in source.decode(stream):
                    last_frame = frame.to_ndarray(format="bgr24")
                    encoder.write(last_frame)
                    progress.update()

                if self.transition_frames > 0 and last_frame is not None:
                    transition = self.generate_transition_frames(last_frame, end_image, self.transition_frames)
                    for frame in transition:
                        encoder.write(resize_frame(frame, width, height))
                        progress.update()

                for _ in range(self.still_image_delay_frames):
                    encoder.write(end_image)
                    progress.update()

        os.remove(video_path)
        os.rename(temp_file_path, video_path)

        logger.info("edited video at: %s", video_path)

        return video_path

    def create_video(self, image_folder: str, video_path: str) -> str:
        """Render every image in a directory into one video.

        Args:
            image_folder: Directory to read. Only its own files are read, in sorted order,
                and only those whose extension is in ``ALLOWED_EXT``.
            video_path: Destination path without an extension.

        Returns:
            The path written, extension included, or ``""`` when the directory held no
            readable image or the file did not appear.

        Raises:
            DependencyError: av is not installed.
            ValueError: This build of av cannot encode the codec, or the codec cannot
                encode at the first image's scaled size.
        """
        deps.require("av")

        # Settled before the destination is touched, so a missing encoder is reported
        # instead of a half-written file.
        require_codec(self.codec)

        folder = sandbox.resolve_read(image_folder)
        image_paths = []
        for name in sorted(os.listdir(folder)):
            if not name.lower().endswith(ALLOWED_EXT):
                continue
            try:
                entry = sandbox.resolve_read(folder / name)
            except sandbox.PathNotAllowed:
                # A symlink in a permitted folder can still point outside it, and the
                # containment check reads the link's target rather than its name.
                logger.warning("skipping %s, which resolves outside the permitted roots", name)
                continue
            if entry.is_file():
                image_paths.append(str(entry))

        if len(image_paths) == 0:
            logger.error("no valid image files found in `%s` directory", image_folder)
            logger.error("the valid formats are: %s", ", ".join(sorted(ALLOWED_EXT)))
            return ""

        output_file = str(sandbox.resolve_write(video_path + self.extensions[self.codec]))
        image = self.rescale(read_frame(image_paths[0]), self.max_size)
        height, width = image.shape[:2]

        progress = progress_bar(len(image_paths))
        with Encoder(output_file, self.codec, width, height, self.fps, color_space="sRGB") as encoder:
            encoder.write(image)
            for _ in range(self.still_image_delay_frames - 1):
                encoder.write(image)

            for i in range(len(image_paths)):
                start_frame = read_frame(image_paths[i])
                end_frame = None
                if i + 1 <= len(image_paths) - 1:
                    end_frame = self.rescale(read_frame(image_paths[i + 1]), self.max_size)

                if isinstance(end_frame, np.ndarray):
                    transition_frames = self.generate_transition_frames(start_frame, end_frame, self.transition_frames)
                    transition_frames = [resize_frame(frame, width, height) for frame in transition_frames]
                    for frame in transition_frames:
                        encoder.write(frame)

                    for _ in range(self.still_image_delay_frames - self.transition_frames):
                        encoder.write(end_frame)

                else:
                    encoder.write(start_frame)
                    for _ in range(self.still_image_delay_frames - 1):
                        encoder.write(start_frame)

                progress.update()

        if os.path.exists(output_file):
            logger.info("created video at: %s", output_file)
            return output_file

        logger.error("unable to create video at: %s", output_file)
        return ""

    def rescale(self, image: np.ndarray, max_size: int) -> np.ndarray:
        """Scale an image so neither edge exceeds ``max_size``, keeping its aspect ratio.

        Args:
            image: Source frame, three 8-bit channels.
            max_size: Longest edge in pixels. An image smaller than this is scaled up to
                it, so the result always touches ``max_size`` on one edge.

        Returns:
            The scaled array. Each edge is truncated to a whole pixel, so the result can
            be one pixel short of the requested size.
        """
        f1 = max_size / image.shape[1]
        f2 = max_size / image.shape[0]
        f = min(f1, f2)
        return resize_frame(image, int(image.shape[1] * f), int(image.shape[0] * f))

    def generate_transition_frames(
        self,
        img1: np.ndarray | None,
        img2: np.ndarray | None,
        num_frames: int,
    ) -> list[np.ndarray]:
        """Blend from one frame to another.

        Args:
            img1: Frame the transition starts from. ``None`` starts from black at
                ``img2``'s size.
            img2: Frame the transition runs towards, scaled onto ``img1``'s size when the
                two differ. ``None`` runs towards black at ``img1``'s size.
            num_frames: Frames to render.

        Returns:
            ``num_frames`` blended arrays at evenly spaced weights, starting on a copy of
            ``img1`` and stopping one step short of ``img2``. Both being ``None`` returns
            no frames.
        """
        if img1 is None and img2 is None:
            return []

        if img1 is not None and img2 is not None:
            if img1.shape != img2.shape:
                img2 = resize_frame(img2, img1.shape[1], img1.shape[0])
        elif img1 is not None:
            img2 = np.zeros_like(img1)
        else:
            img1 = np.zeros_like(img2)

        frame_sequence = []
        for i in range(num_frames):
            alpha = i / float(num_frames)
            frame_sequence.append(blend_frames(img1, img2, alpha))

        return frame_sequence

    def pil2cv(self, img: Image.Image) -> np.ndarray:
        """Convert a PIL image to the ``bgr24`` array layout the encoder takes."""
        return pil_to_bgr(img)

    def add_codecs(self, codecs) -> None:
        """Register extra codecs, or re-container ones already registered.

        Args:
            codecs: ``{four-character code: container extension}``. Anything that is not a
                mapping is ignored, which is what a misconfigured ``video.extra_codecs``
                block reduces to.
        """
        if isinstance(codecs, dict):
            self.valid_codecs.extend(code for code in codecs if code not in self.valid_codecs)
            self.extensions.update(codecs)

    def get_codecs(self) -> list[str]:
        """Every code this writer accepts, lowercase, in registration order."""
        return self.valid_codecs


class Encoder:
    """An open output container, its video stream, and an audio stream where audio is given.

    Args:
        path: File to write. Its extension selects the container format.
        codec: Four-character code, resolved through :data:`ENCODERS`.
        width: Frame width in pixels.
        height: Frame height in pixels.
        rate: Frame rate, as an int or a ``Fraction``.
        audio: ``{"waveform", "sample_rate"}`` laid under the frames, or ``None``.
        options: Encoder options, such as ``{"crf": "16"}``, or ``None``.
        color_space: ``"sRGB"`` converts and tags a YUV stream as BT.709 in limited range;
            ``None`` leaves the converter's default untagged.
        bit_depth: 10 or more writes a 10-bit 4:2:0 stream where the encoder takes one.

    Raises:
        DependencyError: av is not installed.
        ValueError: The codec has no encoder in this build of av, or the encoder refuses
            the frame size.
    """

    def __init__(
        self, path: str, codec: str, width: int, height: int, rate, audio=None, options=None,
        color_space: str | None = None, bit_depth: int = 8,
    ):
        av = deps.require("av")

        self.av = av
        self.width = int(width)
        self.height = int(height)
        self.audio = audio
        self.sound = None
        # Resolved before the container is opened, so a codec this build of av cannot write
        # leaves no empty file behind.
        encoder, pixel_format = _encoder_for(codec)
        if int(bit_depth) >= 10 and pixel_format == PIXEL_FORMAT:
            offered = {entry.name for entry in encoder.video_formats or ()}
            if "yuv420p10le" in offered:
                pixel_format = "yuv420p10le"
        self.pixel_format = pixel_format
        self.color_space = color_space if pixel_format.startswith("yuv") else None
        self.container = av.open(path, mode="w")
        try:
            self.stream = self.container.add_stream(encoder.name, rate=rate)
            self.stream.width = self.width
            self.stream.height = self.height
            self.stream.pix_fmt = pixel_format
            if options:
                self.stream.options = {str(key): str(value) for key, value in options.items()}
            if self.color_space == "sRGB":
                _tag_bt709(self.stream.codec_context)
            try:
                self.stream.codec_context.open()
            except Exception as error:
                raise ValueError(
                    f"the {codec.upper()} codec's {encoder.name} encoder would not open for "
                    f"{self.width}x{self.height} video: {error}. Some encoders reject odd frame "
                    f"sizes; a max_size that scales to even dimensions, or the mp4v codec, will "
                    f"encode this."
                ) from error
            if audio is not None:
                self.sound = self.container.add_stream(
                    AUDIO_ENCODERS.get(codec, AUDIO_ENCODER), rate=int(audio["sample_rate"])
                )
        except Exception:
            self.container.close()
            raise

    def __enter__(self) -> "Encoder":
        return self

    def __exit__(self, kind, value, traceback) -> bool:
        try:
            if kind is None:
                for packet in self.stream.encode():
                    self.container.mux(packet)
                if self.sound is not None:
                    self._lay_sound()
        finally:
            # On Windows a container left open cannot be deleted or renamed, so the handle
            # is closed on every path out.
            self.container.close()
        return False

    def _lay_sound(self) -> None:
        """Encode the whole waveform onto the audio stream and flush it."""
        waveform = self.audio["waveform"]
        packed = waveform[0].detach().cpu().numpy().astype(np.float32)
        if packed.ndim == 1:
            packed = packed[None, :]
        layout = "mono" if packed.shape[0] == 1 else "stereo"
        packed = np.ascontiguousarray(packed[:2])

        frame = self.av.AudioFrame.from_ndarray(packed, format="fltp", layout=layout)
        frame.sample_rate = int(self.audio["sample_rate"])
        resampler = self.av.audio.resampler.AudioResampler(
            format=self.sound.format, layout=self.sound.layout, rate=self.sound.rate
        )
        for resampled in resampler.resample(frame):
            for packet in self.sound.encode(resampled):
                self.container.mux(packet)
        for packet in self.sound.encode():
            self.container.mux(packet)

    def write(self, array: np.ndarray) -> None:
        """Encode one ``bgr24`` array, scaling it onto the stream's size if it differs."""
        # An encoder is opened once for a fixed frame size.
        if array.shape[0] != self.height or array.shape[1] != self.width:
            array = resize_frame(array, self.width, self.height)
        self._encode(self.av.VideoFrame.from_ndarray(array, format="bgr24"))

    def write_rgb(self, array: np.ndarray) -> None:
        """Encode one ``(height, width, 3)`` array of the stream's size, uint8 or uint16 RGB."""
        packed = "rgb48le" if array.dtype == np.uint16 else "rgb24"
        self._encode(self.av.VideoFrame.from_ndarray(np.ascontiguousarray(array), format=packed))

    def _encode(self, frame) -> None:
        """Convert one frame to the stream's pixel format and encode it."""
        if self.color_space == "sRGB":
            frame = frame.reformat(format=self.pixel_format, dst_colorspace=BT709)
            _tag_bt709(frame)
        for packet in self.stream.encode(frame):
            self.container.mux(packet)


def _tag_bt709(target) -> None:
    """Mark a codec context or a frame as BT.709 primaries and matrix, sRGB transfer, limited range."""
    from av.video.reformatter import ColorPrimaries, ColorRange, ColorTrc

    target.color_primaries = ColorPrimaries.BT709
    target.color_trc = ColorTrc.IEC61966_2_1
    target.colorspace = BT709
    target.color_range = ColorRange.MPEG


def _encoder_for(codec: str):
    """Resolve a four-character code to an encoder and the pixel format to feed it.

    Args:
        codec: Four-character code, or an encoder name for a configured extra codec.

    Returns:
        ``(codec, pixel format)``: the av ``Codec`` to add a stream with, and the format
        from :func:`_pixel_format`.

    Raises:
        DependencyError: av is not installed.
        ValueError: This build of av has no encoder under that name.
    """
    encoder = require_codec(codec)
    return encoder, _pixel_format(encoder)


def _pixel_format(encoder) -> str:
    """The pixel format to hand ``encoder``.

    Args:
        encoder: An av ``Codec`` opened for writing.

    Returns:
        The first format :data:`PIXEL_FORMATS` prefers for this encoder that it accepts,
        or :data:`PIXEL_FORMAT`, or the first format it lists. An encoder that lists no
        formats takes whatever it is given, so it gets :data:`PIXEL_FORMAT`.
    """
    formats = [pixel_format.name for pixel_format in encoder.video_formats or ()]
    if not formats:
        return PIXEL_FORMAT
    for wanted in PIXEL_FORMATS.get(encoder.name, ()):
        if wanted in formats:
            return wanted
    if PIXEL_FORMAT in formats:
        return PIXEL_FORMAT
    return formats[0]


def progress_bar(total: int) -> "_Progress":
    """A progress bar for a loop of ``total`` frames.

    Reports to ComfyUI's front end when running under it, and does nothing at all
    otherwise.

    Args:
        total: Frames the loop will produce.

    Returns:
        The bar. Its ``update`` is safe to call outside ComfyUI.
    """
    return _Progress(total)


class _Progress:
    """ComfyUI's progress bar where there is one, and a no-op where there is not.

    Args:
        total: Frames the loop will produce.
    """

    def __init__(self, total: int):
        self.bar = None
        try:
            from comfy.utils import ProgressBar

            self.bar = ProgressBar(total)
        except Exception as error:
            logger.debug("no progress bar available: %s", error)

    def update(self, count: int = 1) -> None:
        """Advance the bar by ``count`` frames."""
        if self.bar is not None:
            self.bar.update(count)
