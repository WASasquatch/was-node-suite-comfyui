"""Clips kept on disk as raw frames beside a manifest, written and read a batch at a time.

A cache folder holds :data:`DATA`, frames as 8-bit codes or half floats, :data:`SOUND`, and
the JSON manifest :data:`MANIFEST`.
"""

from __future__ import annotations

import bisect
import contextlib
import io as _io
import json
import os
import re
import shutil
from fractions import Fraction

import numpy as np
import torch
from comfy_api.latest import Input, InputImpl, Types

from .. import log
from ..image import scratch

__all__ = [
    "DATA",
    "DEPTHS",
    "EXTENSION",
    "MANIFEST",
    "SOUND",
    "CachedVideo",
    "FrameCache",
    "cached",
]

logger = log.get_logger("media.frame_cache")

#: Name of the manifest in every cache folder.
MANIFEST = "cache.wasframes"

#: Suffix a loader lists caches by.
EXTENSION = ".wasframes"

#: Name of the file holding the frames.
DATA = "frames.bin"

#: Name of the file holding the sound, a float32 ``(channels, samples)`` array.
SOUND = "audio.npy"

#: Manifest layout written and read.
VERSION = 1

#: How frames are kept: `auto` = 8-bit codes, half floats for a batch outside 0 to 1;
#: `16 bit` = half floats always.
DEPTHS = ("auto", "16 bit")

#: Bytes of frames converted together on the compute device.
CONVERT_BYTES = 256 * 1024 ** 2

#: Folder names a cache takes, the name then a five digit number.
FOLDER = re.compile(r"^(?P<name>.+)_(?P<number>\d{5})$")


def _next_folder(parent: str, name: str) -> str:
    """The next unused ``name_NNNNN`` folder below ``parent``."""
    highest = 0
    with contextlib.suppress(OSError):
        for entry in os.listdir(parent):
            found = FOLDER.match(entry)
            if found and found.group("name") == name:
                highest = max(highest, int(found.group("number")))
    return os.path.join(parent, f"{name}_{highest + 1:05d}")


class FrameCache:
    """One cache folder and its manifest, appended to and read a frame at a time.

    Args:
        folder: The cache's folder.
        manifest: Its manifest.
    """

    def __init__(self, folder: str, manifest: dict):
        self.folder = str(folder)
        self.manifest = manifest

    @classmethod
    def create(cls, parent: str, name: str, rate, depth: str = "auto",
               color_space: str = "sRGB") -> "FrameCache":
        """A new, empty cache in a fresh numbered folder.

        Args:
            parent: The folder the cache's own folder is made in.
            name: The cache's name, numbered after.
            rate: Frames per second.
            depth: An entry of :data:`DEPTHS`.
            color_space: The colour space the frames are in.

        Returns:
            The :class:`FrameCache`.
        """
        os.makedirs(parent, exist_ok=True)
        folder = _next_folder(parent, name)
        os.makedirs(folder)
        rate = Fraction(rate).limit_denominator(1001)
        manifest = {
            "version": VERSION,
            "width": 0,
            "height": 0,
            "channels": 3,
            "rate": [rate.numerator, rate.denominator],
            "depth": depth if depth in DEPTHS else DEPTHS[0],
            "color_space": str(color_space),
            "frames": 0,
            "runs": [],
            "scenes": [],
            "audio": None,
        }
        open(os.path.join(folder, DATA), "wb").close()
        cache = cls(folder, manifest)
        cache._save()
        return cache

    @classmethod
    def open(cls, folder: str) -> "FrameCache":
        """An existing cache.

        Args:
            folder: The cache's folder.

        Returns:
            The :class:`FrameCache`.

        Raises:
            FileNotFoundError: The folder holds no cache, as after one was deleted.
            ValueError: The manifest was written by a newer layout.
        """
        path = os.path.join(str(folder), MANIFEST)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"no frame cache is left at {folder}. A cache set to delete after saving is "
                f"gone once saved; run the node that wrote it again."
            )
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        if int(manifest.get("version", 0)) > VERSION:
            raise ValueError(f"the frame cache at {folder} was written by a newer version of the pack")
        return cls(folder, manifest)

    @property
    def frames(self) -> int:
        """Frames held."""
        return int(self.manifest["frames"])

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)`` of every frame, ``(0, 0)`` before the first is written."""
        return int(self.manifest["width"]), int(self.manifest["height"])

    @property
    def channels(self) -> int:
        """Channels of every frame."""
        return int(self.manifest["channels"])

    @property
    def rate(self) -> Fraction:
        """Frames per second."""
        numerator, denominator = self.manifest["rate"]
        return Fraction(int(numerator), int(denominator))

    @property
    def color_space(self) -> str:
        """The colour space the frames are in."""
        return str(self.manifest.get("color_space") or "sRGB")

    @property
    def scenes(self) -> list[int]:
        """First frame of every scene marked as one."""
        return [int(start) for start in self.manifest.get("scenes", [])]

    @property
    def halves(self) -> bool:
        """Whether any frame is kept as half floats."""
        return any(run["dtype"] == "float16" for run in self.manifest["runs"])

    def _save(self) -> None:
        """Write the manifest, replacing the last one whole."""
        path = os.path.join(self.folder, MANIFEST)
        draft = path + ".part"
        with open(draft, "w", encoding="utf-8") as handle:
            json.dump(self.manifest, handle, indent=1)
        os.replace(draft, path)

    def _frame_bytes(self, dtype: str) -> int:
        """Bytes one frame takes in ``dtype``."""
        width, height = self.size
        return width * height * self.channels * (1 if dtype == "uint8" else 2)

    def append(self, images, scene: bool = False, device=None) -> int:
        """Write a batch of frames after the last.

        Args:
            images: ``(frames, height, width, channels)``, on any device.
            scene: Mark the batch's first frame as the start of a scene.
            device: Where frames are converted; ComfyUI's compute device when None.

        Returns:
            The number of the batch's first frame.

        Raises:
            ValueError: The batch's size or channel count differs from the cache's.
        """
        count, height, width, channels = (int(side) for side in images.shape)
        first = self.frames
        if count == 0:
            return first
        if first == 0:
            self.manifest.update(width=width, height=height, channels=channels)
        elif (width, height, channels) != (*self.size, self.channels):
            raise ValueError(
                f"the frame cache at {self.folder} holds {self.size[0]}x{self.size[1]} frames "
                f"with {self.channels} channel(s), and this batch is {width}x{height} with "
                f"{channels}. Resize the frames to the cache's size first."
            )
        if device is None:
            import comfy.model_management

            device = comfy.model_management.get_torch_device()
        halves = self.manifest["depth"] == "16 bit" or not bool(
            images.min() >= 0.0 and images.max() <= 1.0
        )
        dtype = "float16" if halves else "uint8"
        group = max(1, CONVERT_BYTES // max(1, height * width * channels * 4))
        path = os.path.join(self.folder, DATA)
        offset = os.path.getsize(path)
        with open(path, "ab", buffering=0) as handle:
            for start in range(0, count, group):
                part = images[start:start + group].to(device=device)
                if halves:
                    part = part.to(torch.float16)
                else:
                    part = part.float().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)
                handle.write(memoryview(part.contiguous().cpu().numpy()).cast("B"))
                scratch.trim(images[start:start + group])
        runs = self.manifest["runs"]
        last = runs[-1] if runs else None
        if last is not None and last["dtype"] == dtype and \
                last["offset"] + last["count"] * self._frame_bytes(dtype) == offset:
            last["count"] += count
        else:
            runs.append({"start": first, "count": count, "dtype": dtype, "offset": offset})
        self.manifest["frames"] = first + count
        if scene and first not in self.manifest["scenes"]:
            self.manifest["scenes"].append(first)
        self._save()
        return first

    def add_audio(self, audio) -> None:
        """Lay sound after any already kept.

        Args:
            audio: An ``AUDIO`` dict; its first batch item is kept.

        Raises:
            ValueError: The sample rate or channel count differs from the sound kept.
        """
        if not isinstance(audio, dict) or audio.get("waveform") is None:
            return
        wave = audio["waveform"][0].detach().float().cpu().numpy()
        rate = int(audio.get("sample_rate", 44100))
        held = self.manifest.get("audio")
        path = os.path.join(self.folder, SOUND)
        if held is not None:
            if int(held["sample_rate"]) != rate or int(held["channels"]) != wave.shape[0]:
                raise ValueError(
                    f"the frame cache holds {held['channels']} channel(s) at {held['sample_rate']} "
                    f"Hz and this sound is {wave.shape[0]} at {rate} Hz. Resample it to match."
                )
            wave = np.concatenate([np.load(path), wave], axis=-1)
        np.save(path, np.ascontiguousarray(wave, dtype=np.float32))
        self.manifest["audio"] = {"sample_rate": rate, "channels": int(wave.shape[0]),
                                  "samples": int(wave.shape[-1])}
        self._save()

    def _run(self, index: int) -> dict:
        """The run frame ``index`` sits in."""
        runs = self.manifest["runs"]
        place = bisect.bisect_right([run["start"] for run in runs], int(index)) - 1
        if place < 0 or not 0 <= index - runs[place]["start"] < runs[place]["count"]:
            raise IndexError(f"frame {index} is outside the cache's {self.frames}")
        return runs[place]

    def _read(self, handle, index: int):
        """Frame ``index`` as it is kept: uint8 codes or half floats, ``(height, width, channels)``."""
        run = self._run(index)
        width, height = self.size
        size = self._frame_bytes(run["dtype"])
        buffer = np.empty(size, dtype=np.uint8)
        handle.seek(run["offset"] + (int(index) - run["start"]) * size)
        view = memoryview(buffer)
        done = 0
        while done < size:
            got = handle.readinto(view[done:])
            if not got:
                raise OSError(f"the frame cache at {self.folder} ends inside frame {index}")
            done += got
        kept = buffer.view(np.float16 if run["dtype"] == "float16" else np.uint8)
        return torch.from_numpy(kept.reshape(height, width, self.channels))

    @contextlib.contextmanager
    def reader(self):
        """A callable reading frames as float32 in ``[0, 1]`` for codes, kept open while in use.

        Yields:
            A callable taking a frame number and answering ``(height, width, channels)``.
        """
        with open(os.path.join(self.folder, DATA), "rb", buffering=0) as handle:
            def read(index):
                frame = self._read(handle, index)
                return frame.float() / 255.0 if frame.dtype == torch.uint8 else frame.float()

            yield read

    def codes(self, index: int):
        """Frame ``index`` as uint8 codes, or ``None`` where it is kept as half floats."""
        with open(os.path.join(self.folder, DATA), "rb", buffering=0) as handle:
            frame = self._read(handle, index)
        return frame if frame.dtype == torch.uint8 else None

    def audio(self, start: int = 0, stop: int | None = None) -> dict | None:
        """The sound under frames ``start`` to ``stop``.

        Args:
            start: First frame.
            stop: Frame after the last; the cache's end when None.

        Returns:
            An ``AUDIO`` dict, or ``None`` where the cache holds no sound.
        """
        held = self.manifest.get("audio")
        if not held:
            return None
        wave = np.load(os.path.join(self.folder, SOUND), mmap_mode="r")
        per_frame = int(held["sample_rate"]) / float(self.rate)
        stop = self.frames if stop is None else int(stop)
        first = max(0, int(round(int(start) * per_frame)))
        last = max(first, min(wave.shape[-1], int(round(stop * per_frame))))
        piece = torch.from_numpy(np.array(wave[:, first:last], dtype=np.float32))
        return {"waveform": piece.unsqueeze(0), "sample_rate": int(held["sample_rate"])}

    def delete(self) -> None:
        """Remove the cache's folder and everything in it."""
        shutil.rmtree(self.folder, ignore_errors=True)
        logger.info("deleted the frame cache at %s", self.folder)


class _Frames:
    """A cached clip's frames read from disk one at a time, shaped like an ``IMAGE`` batch."""

    def __init__(self, cache: FrameCache, start: int, stop: int):
        self.cache = cache
        self.start = int(start)
        self.stop = int(stop)
        width, height = cache.size
        self.shape = torch.Size((self.stop - self.start, height, width, cache.channels))
        self.ndim = 4
        self.dtype = torch.float32
        self.device = torch.device("cpu")

    def __len__(self) -> int:
        return self.stop - self.start

    def __iter__(self):
        with self.cache.reader() as read:
            for index in range(self.start, self.stop):
                yield read(index)

    def __getitem__(self, key):
        if isinstance(key, slice):
            numbers = range(self.start, self.stop)[key]
            with self.cache.reader() as read:
                return torch.stack([read(index) for index in numbers]) if len(numbers) else \
                    torch.empty((0,) + tuple(self.shape[1:]))
        index = int(key) + (len(self) if int(key) < 0 else 0)
        if not 0 <= index < len(self):
            raise IndexError(f"frame {key} is outside the clip's {len(self)}")
        with self.cache.reader() as read:
            return read(self.start + index)


class CachedVideo(Input.Video):
    """A ``VIDEO`` whose frames stay in a frame cache on disk.

    Args:
        folder: The cache's folder.
        start: First frame of the clip.
        stop: Frame after its last; the cache's end when None.
        delete_after_save: Remove the cache once a save has written all of it.
    """

    def __init__(self, folder: str, start: int = 0, stop: int | None = None,
                 delete_after_save: bool = False):
        self.folder = str(folder)
        cache = FrameCache.open(self.folder)
        self.start = max(0, int(start))
        self.stop = cache.frames if stop is None else min(int(stop), cache.frames)
        self.delete_after_save = bool(delete_after_save)
        self._keep = 0

    def cache(self) -> FrameCache:
        """The cache, read afresh."""
        return FrameCache.open(self.folder)

    def lazy_frames(self) -> _Frames:
        """The frames, read from disk as they are iterated."""
        return _Frames(self.cache(), self.start, self.stop)

    @contextlib.contextmanager
    def keeping(self):
        """Saves made inside this leave the cache in place."""
        self._keep += 1
        try:
            yield self
        finally:
            self._keep -= 1

    def get_components(self):
        """Every frame loaded, with the sound and the rate."""
        cache = self.cache()
        width, height = cache.size
        count = self.stop - self.start
        images = scratch.allocate((count, height, width, cache.channels), torch.float32,
                                  node="Video Cache")
        with cache.reader() as read:
            for row, index in enumerate(range(self.start, self.stop)):
                images[row].copy_(read(index))
                scratch.trim(images[row])
        return Types.VideoComponents(
            images=images, audio=cache.audio(self.start, self.stop), frame_rate=cache.rate,
            metadata=None,
        )

    def save_to(self, path, format=Types.VideoContainer.AUTO, codec=Types.VideoCodec.AUTO,
                metadata=None, bit_depth=None, crf=None, color_space=None, preset=None):
        """Encode the clip to ``path`` frame by frame, deleting the cache afterwards if asked."""
        cache = self.cache()
        components = Types.VideoComponents(
            images=_Frames(cache, self.start, self.stop),
            audio=cache.audio(self.start, self.stop),
            frame_rate=cache.rate,
            metadata=None,
        )
        video = InputImpl.VideoFromComponents(
            components, bit_depth=self.get_bit_depth(), color_space=cache.color_space,
        )
        options = {
            name: value for name, value in (
                ("metadata", metadata), ("bit_depth", bit_depth), ("crf", crf),
                ("color_space", color_space), ("preset", preset),
            ) if value is not None
        }
        video.save_to(path, format=format, codec=codec, **options)
        if isinstance(path, (str, os.PathLike)):
            self.saved()

    def saved(self) -> None:
        """Note that a save has written the clip, deleting the cache if asked to and the clip is all of it."""
        if not self.delete_after_save or self._keep:
            return
        cache = self.cache()
        if self.start == 0 and self.stop == cache.frames:
            cache.delete()

    def as_trimmed(self, start_time=None, duration=None, strict_duration=False):
        """The part of the clip from ``start_time`` for ``duration`` seconds."""
        rate = float(self.cache().rate)
        first = self.start + max(0, int(round(float(start_time or 0.0) * rate)))
        last = self.stop if not duration else min(self.stop, first + int(round(float(duration) * rate)))
        if last <= first:
            return None
        if strict_duration and duration and (last - first) / rate < float(duration) - 1.0 / rate:
            return None
        return CachedVideo(self.folder, first, last, self.delete_after_save)

    def get_stream_source(self):
        """The clip encoded to an in-memory mp4, the cache left in place."""
        buffer = _io.BytesIO()
        with self.keeping():
            self.save_to(buffer, format=Types.VideoContainer.MP4)
        buffer.seek(0)
        return buffer

    def get_dimensions(self) -> tuple[int, int]:
        """``(width, height)`` of the frames."""
        return self.cache().size

    def get_frame_count(self) -> int:
        """Frames in the clip."""
        return self.stop - self.start

    def get_frame_rate(self) -> Fraction:
        """Frames per second."""
        return self.cache().rate

    def get_duration(self) -> float:
        """Seconds the clip plays for."""
        return (self.stop - self.start) / float(self.cache().rate)

    def get_bit_depth(self) -> int:
        """10 where any frame is kept as half floats, otherwise 8."""
        return 10 if self.cache().halves else 8

    def get_color_space(self) -> str:
        """The colour space the frames are in."""
        return self.cache().color_space

    def get_container_format(self) -> str:
        """``wasframes``, the cache's own layout."""
        return "wasframes"


def cached(video) -> bool:
    """Whether ``video`` is a clip kept in a frame cache."""
    return isinstance(video, CachedVideo)
