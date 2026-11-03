"""A training set's health: picture quality flags, near-duplicates, aspect balance, faces and captions.

Each picture is measured once at :data:`LOOK_EDGE` pixels and held in memory by path, size and
modification time. Scores run 0 to 1, 1 being healthy.
"""

from __future__ import annotations

import hashlib
import os
import re
import statistics
import threading
from collections import Counter, OrderedDict
from pathlib import Path
from typing import NamedTuple

from .. import log
from ..constants import ALLOWED_EXT
from ..util import sandbox

__all__ = [
    "FACE_FLAGS",
    "FACE_ISSUES",
    "FLAG_ORDER",
    "MAX_PICTURES",
    "SCORE_ORDER",
    "Picture",
    "caption_suffix",
    "faces",
    "faces_pixels",
    "fingerprint",
    "fingerprint_pictures",
    "from_folder",
    "jpeg_quality",
    "look",
    "look_pixels",
    "measure",
    "pairs",
    "read_caption",
    "scan",
]

logger = log.get_logger("dataset.health")

#: Most pictures one measurement reads. The rest of a larger folder is left out and counted.
MAX_PICTURES = 2000

#: Long side, in pixels, a picture's look is measured at.
LOOK_EDGE = 256

#: Looks and face counts held in memory, each keyed by path, modification time and size.
MAX_CACHED = 10000

#: Bytes of one caption file read. The rest of a longer file is left out.
MAX_CAPTION_BYTES = 64 * 1024

#: Brightness, 0 to 1, under which a picture is flagged dark.
DARK = 0.15

#: Brightness over which a picture is flagged blown out.
BLOWN_OUT = 0.85

#: Standard deviation of brightness under which a picture is flagged flat.
FLAT = 0.06

#: Share of the set's median sharpness under which a picture is flagged blurry.
BLURRY = 0.25

#: Share of a picture taken by bars along its edges over which it is flagged.
BARS = 0.08

#: Mean saturation under which a picture is flagged greyscale, while the set's median is over
#: :data:`GREYSCALE_SET`.
GREYSCALE = 0.03
GREYSCALE_SET = 0.08

#: Share of a picture's pixels see-through for it to be flagged transparent.
TRANSPARENT = 0.01

#: Estimated JPEG quality, 1 to 100, under which a picture is flagged compressed.
JPEG_LOW = 60

#: Two pictures are near-duplicates when their 256-bit difference hashes differ in at most
#: :data:`NEAR_BITS` bits, their 16x16 grey shapes correlate at :data:`NEAR_SHAPE` or more and
#: their mean colours differ by at most :data:`NEAR_COLOUR` levels of 255. Mirrored copies count.
NEAR_BITS = 24
NEAR_SHAPE = 0.95
NEAR_COLOUR = 15.0

#: Least detector confidence a face is counted at.
FACE_SCORE = 0.6

#: Height of the main face as a share of the picture's under which it is flagged small.
SMALL_FACE = 0.05

#: A second face at least this share of the main face's height makes a picture several faces.
SECOND_FACE = 0.5

#: Every flag a picture can carry, in the order a report lists them.
FLAG_ORDER = (
    "unreadable", "dark", "blown out", "flat", "blurry", "bars", "greyscale", "transparent",
    "compressed", "small", "near-duplicate", "no face", "several faces", "small face",
)

#: The flags a face check adds.
FACE_FLAGS = frozenset({"no face", "several faces", "small face"})

#: The face flags the faces score counts against.
FACE_ISSUES = frozenset({"no face", "several faces"})

#: The scores, in the order a radar draws them.
SCORE_ORDER = ("captions", "consistency", "quality", "uniqueness", "balance", "faces")

#: Flags the quality score does not count.
_NOT_QUALITY = FACE_FLAGS | {"near-duplicate"}

#: The luminance weights brightness is measured with.
_LUMA = (0.299, 0.587, 0.114)

#: The standard JPEG luminance quantisation table, at quality 50.
_JPEG_LUMA = (
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
)

#: An extension a caption file may carry: a dot and up to fifteen letters, digits, `_` or `-`.
_SUFFIX = re.compile(r"\.[A-Za-z0-9_-]{1,15}")

#: EXIF orientation to the transpose that stands the picture upright, by method name.
_ORIENTATIONS = {
    2: "FLIP_LEFT_RIGHT", 3: "ROTATE_180", 4: "FLIP_TOP_BOTTOM", 5: "TRANSPOSE",
    6: "ROTATE_270", 7: "TRANSVERSE", 8: "ROTATE_90",
}

_lock = threading.Lock()
_looks: OrderedDict[tuple, dict | None] = OrderedDict()
_faces: OrderedDict[tuple, dict | None] = OrderedDict()


class Picture(NamedTuple):
    """One picture to measure, read from its file or from pixels standing in for it.

    Attributes:
        name: What the figures call it.
        path: The picture file, already contained, or ``""`` where ``pixels`` stand in.
        caption: Its caption, or None where it has none.
        pixels: A ``(1, height, width, channels)`` float tensor in ``[0, 1]`` measured in place
            of the file, or None.
    """

    name: str
    path: str = ""
    caption: str | None = None
    pixels: object = None


def from_folder(paths, root, suffix: str) -> list[Picture]:
    """The pictures found in a folder, each named below it and carrying the caption saved beside it.

    Args:
        paths: The pictures, already contained.
        root: The folder they were found in.
        suffix: The caption extension, from :func:`caption_suffix`.

    Returns:
        One :class:`Picture` per path, in order.
    """
    base = Path(os.path.realpath(root))
    return [Picture(_relative(path, base), str(path), read_caption(path, suffix)) for path in paths]


def caption_suffix(extension) -> str:
    """The caption file extension as typed, with its leading dot.

    Args:
        extension: Such as ``.txt``, ``txt`` or ``.caption``.

    Returns:
        The extension starting with a dot.

    Raises:
        ValueError: The extension is empty, names a folder, holds anything but letters, digits,
            ``_`` and ``-``, or is a picture's own extension.
    """
    text = str(extension or "").strip()
    if not text:
        raise ValueError(
            "caption_extension is empty. Type the extension the captions are saved with, "
            "such as `.txt`"
        )
    suffix = text if text.startswith(".") else f".{text}"
    if not _SUFFIX.fullmatch(suffix):
        raise ValueError(
            f"caption_extension `{sandbox._shown(text)}` is not a file extension. Type the "
            f"ending alone, such as `.txt` or `.caption`"
        )
    if suffix.lower() in ALLOWED_EXT:
        raise ValueError(
            f"caption_extension `{suffix}` is a picture's own extension, so every picture "
            f"would be read as its own caption. Type the captions' extension, such as `.txt`"
        )
    return suffix


def scan(directory, pattern: str) -> list[str]:
    """Every picture under a folder that a glob pattern matches, sorted by its path below it.

    Args:
        directory: The folder, already resolved inside a permitted read root.
        pattern: Glob pattern matched under ``directory``; ``**`` crosses folders.

    Returns:
        Absolute paths of the matching pictures. A match a link or a junction leads outside
        the folder is left out.

    Raises:
        PathNotAllowed: The pattern is empty, absolute or climbs out with ``..``.
    """
    base = Path(os.path.realpath(directory))
    found = []
    for name in sandbox.glob_read(base, pattern, recursive=True):
        if not name.lower().endswith(ALLOWED_EXT):
            continue
        real = Path(os.path.realpath(name))
        if not sandbox.contains(base, real) or not real.is_file():
            logger.debug("a match outside the folder or not a file was skipped: %s", name)
            continue
        found.append(str(sandbox.resolve_read(real)))
    unique = sorted(set(found), key=lambda path: _relative(path, base).casefold())
    return unique


def read_caption(image, suffix: str) -> str | None:
    """The caption saved beside a picture under the same name, or None where there is none.

    Args:
        image: The picture's path, already contained.
        suffix: The caption extension, from :func:`caption_suffix`.

    Returns:
        The caption's text, cut to :data:`MAX_CAPTION_BYTES`, or None when no caption file is
        there or it cannot be read.
    """
    path = _caption_file(image, suffix)
    if path is None:
        return None
    try:
        with open(path, "rb") as handle:
            data = handle.read(MAX_CAPTION_BYTES + 1)
    except OSError as error:
        logger.debug("a caption could not be read (%s)", error.strerror)
        return None
    if len(data) > MAX_CAPTION_BYTES:
        logger.debug("a caption past %d bytes was cut", MAX_CAPTION_BYTES)
        data = data[:MAX_CAPTION_BYTES]
    return data.decode("utf-8-sig", errors="replace")


def fingerprint(paths, suffix: str) -> str:
    """A digest that changes whenever a picture or its caption is added, removed or edited.

    Args:
        paths: The pictures, already contained.
        suffix: The caption extension, from :func:`caption_suffix`.

    Returns:
        A hexadecimal digest.
    """
    digest = hashlib.sha256()
    for image in paths:
        for item in (Path(image), _caption_file(image, suffix)):
            if item is None:
                digest.update(b"-\n")
                continue
            try:
                info = os.stat(item)
                digest.update(f"{item}|{info.st_mtime_ns}|{info.st_size}\n".encode("utf-8", "replace"))
            except OSError:
                digest.update(f"{item}|-\n".encode("utf-8", "replace"))
    return digest.hexdigest()


def fingerprint_pictures(pictures) -> str:
    """A digest that changes whenever a picture's name, caption, file or stand-in pixels change.

    Args:
        pictures: :class:`Picture` entries.

    Returns:
        A hexadecimal digest.
    """
    digest = hashlib.sha256()
    for picture in pictures:
        digest.update(f"{picture.name!r}|{picture.caption!r}|".encode("utf-8", "replace"))
        if picture.pixels is not None:
            digest.update(f"pixels {id(picture.pixels)}\n".encode("utf-8"))
            continue
        try:
            info = os.stat(picture.path)
            stamp = f"{picture.path}|{info.st_mtime_ns}|{info.st_size}\n"
        except (OSError, ValueError):
            stamp = f"{picture.path}|-\n"
        digest.update(stamp.encode("utf-8", "replace"))
    return digest.hexdigest()


def jpeg_quality(picture) -> int | None:
    """A JPEG's quality, 1 to 100, estimated from its luminance quantisation table.

    Args:
        picture: An opened PIL image.

    Returns:
        The estimate on the IJG scale, or None for anything but a JPEG.
    """
    tables = getattr(picture, "quantization", None)
    if not tables or 0 not in tables:
        return None
    try:
        held = sum(int(value) for value in tables[0])
    except (TypeError, ValueError):
        return None
    scale = 100.0 * held / sum(_JPEG_LUMA)
    quality = (200.0 - scale) / 2.0 if scale <= 100.0 else 5000.0 / scale
    return max(1, min(100, round(quality)))


def look(path) -> dict | None:
    """One picture's size, exposure, sharpness, bars, transparency, JPEG quality and signature.

    Args:
        path: The picture, already contained.

    Returns:
        ``width`` and ``height`` upright, ``bright``, ``contrast``, ``sat`` and ``alpha`` from 0
        to 1, ``sharp``, ``border``, ``jpeg``, and the signature :func:`pairs` compares; None
        for a file that cannot be read as a picture.
    """
    key = _key(path)
    if key is None:
        return None
    with _lock:
        if key in _looks:
            _looks.move_to_end(key)
            return _looks[key]
    measured = _measure(str(path))
    _remember(_looks, key, measured)
    return measured


def look_pixels(pixels) -> dict | None:
    """One picture held as pixels, measured as :func:`look` measures a file.

    Args:
        pixels: A ``(1, height, width, channels)`` or ``(height, width, channels)`` float tensor
            in ``[0, 1]``.

    Returns:
        The fields :func:`look` describes, ``jpeg`` None; None for a tensor that is not a picture.
    """
    try:
        picture, width, height = _from_pixels(pixels, LOOK_EDGE)
    except Exception as error:
        logger.debug("pixels could not be measured (%s)", type(error).__name__)
        return None
    return _measured(picture, width, height, None)


def faces(path, detector) -> dict | None:
    """How many faces one picture holds and how much of its height the largest two fill.

    Args:
        path: The picture, already contained.
        detector: A detector from the pack's YuNet loader.

    Returns:
        ``count``, ``share`` as the main face's height over the picture's, and ``second`` as
        the next face's; None for a file that cannot be read as a picture.
    """
    key = _key(path)
    if key is None:
        return None
    with _lock:
        if key in _faces:
            _faces.move_to_end(key)
            return _faces[key]
    counted = _count_faces(str(path), detector)
    _remember(_faces, key, counted)
    return counted


def faces_pixels(pixels, detector) -> dict | None:
    """How many faces a picture held as pixels holds, as :func:`faces` counts them in a file.

    Args:
        pixels: A ``(1, height, width, channels)`` or ``(height, width, channels)`` float tensor
            in ``[0, 1]``.
        detector: A detector from the pack's YuNet loader.

    Returns:
        The fields :func:`faces` describes; None for a tensor that is not a picture.
    """
    from ..model import yunet

    try:
        picture, _width, _height = _from_pixels(pixels, yunet.DEFAULT_SIZE)
    except Exception as error:
        logger.debug("pixels could not be read for faces (%s)", type(error).__name__)
        return None
    return _faces_in(picture, detector)


def pairs(names, looks) -> list[dict]:
    """Every pair of pictures that are near-duplicates of each other, mirrored or not.

    Args:
        names: One name per picture.
        looks: One :func:`look` answer per picture, None where it could not be read.

    Returns:
        ``{"a", "b", "bits", "shape", "colour", "mirrored"}`` per pair, closest first.
    """
    import numpy as np

    kept = [index for index, seen in enumerate(looks) if seen]
    if len(kept) < 2:
        return []
    shapes = np.stack([looks[index]["shape"] for index in kept]).astype(np.float32)
    mirrors = shapes.reshape(-1, 16, 16)[:, :, ::-1].reshape(len(kept), -1)
    hashes = np.stack([np.frombuffer(looks[index]["hash"], np.uint8) for index in kept])
    flipped = np.stack([np.frombuffer(looks[index]["hash_m"], np.uint8) for index in kept])
    colours = np.array([looks[index]["colour"] for index in kept], np.float32)

    found: dict[tuple[int, int], dict] = {}
    for start in range(0, len(kept), 512):
        block = shapes[start:start + 512]
        for mirrored, against, other in ((False, shapes, hashes), (True, mirrors, flipped)):
            agreement = block @ against.T
            for row, column in zip(*np.nonzero(agreement >= NEAR_SHAPE)):
                first, second = start + int(row), int(column)
                if second <= first:
                    continue
                bits = int(np.unpackbits(hashes[first] ^ other[second]).sum())
                if bits > NEAR_BITS:
                    continue
                colour = float(np.abs(colours[first] - colours[second]).mean())
                if colour > NEAR_COLOUR:
                    continue
                held = found.get((first, second))
                if held is not None and held["bits"] <= bits:
                    continue
                found[(first, second)] = {
                    "a": names[kept[first]],
                    "b": names[kept[second]],
                    "bits": bits,
                    "shape": round(float(agreement[row, column]), 3),
                    "colour": round(colour, 1),
                    "mirrored": mirrored,
                }
    return sorted(found.values(), key=lambda pair: (pair["bits"], -pair["shape"], pair["a"], pair["b"]))


def measure(
    pictures,
    subject: str = "[trigger]",
    target_side: int = 768,
    check_faces: bool = False,
    min_words: int = 5,
    step=None,
) -> dict:
    """Measure a set of pictures and their captions, and score its health.

    Args:
        pictures: :class:`Picture` entries, at most :data:`MAX_PICTURES`; :func:`from_folder`
            builds them for a folder.
        subject: What every caption should name. Empty checks nothing.
        target_side: Short side under which a picture is flagged small. 0 checks nothing.
        check_faces: Count faces and score them.
        min_words: A caption under this many words counts as short. 0 counts none.
        step: Called with no arguments after each picture is measured, and after each face count.

    Returns:
        The figures: counts, caption measurements, exposure, aspect ratios, flag counts,
        near-duplicate ``pairs``, per-picture ``rows`` and the 0 to 1 ``scores``.

    Raises:
        FileNotFoundError: ``check_faces`` is on and the face detector's weights are missing.
    """
    from ..prompt import caption_stats

    pictures = list(pictures)
    names = [picture.name for picture in pictures]
    total = len(names)

    looks = []
    for picture in pictures:
        looks.append(look(picture.path) if picture.pixels is None else look_pixels(picture.pixels))
        if step is not None:
            step()

    counted = None
    if check_faces:
        from ..model import yunet

        detector = yunet.load()
        counted = []
        for picture in pictures:
            counted.append(
                faces(picture.path, detector) if picture.pixels is None
                else faces_pixels(picture.pixels, detector)
            )
            if step is not None:
                step()

    texts = [picture.caption for picture in pictures]
    captioned = sum(1 for text in texts if text and text.strip())
    subject = str(subject or "").strip()
    consistency = caption_stats.consistency([text or "" for text in texts], min_words)
    phrases = caption_stats.phrases([text or "" for text in texts], subject)
    coverage = caption_stats.coverage([text or "" for text in texts], subject) if subject else 0

    seen = [entry for entry in looks if entry]
    sharp_median = statistics.median(entry["sharp"] for entry in seen) if seen else 0.0
    sat_median = statistics.median(entry["sat"] for entry in seen) if seen else 0.0
    found = pairs(names, looks)
    paired = {name for pair in found for name in (pair["a"], pair["b"])}

    rows = []
    for index, name in enumerate(names):
        entry = looks[index]
        face = counted[index] if counted is not None else None
        flags = _flags(entry, face, sharp_median, sat_median, target_side, name in paired,
                       check_faces)
        rows.append({
            "name": name,
            "width": entry["width"] if entry else None,
            "height": entry["height"] if entry else None,
            "flags": flags,
            "brightness": round(entry["bright"], 3) if entry else None,
            "sharpness": round(entry["sharp"] / sharp_median, 2) if entry and sharp_median else None,
            "jpeg_quality": entry["jpeg"] if entry else None,
            "faces": face["count"] if face else (0 if check_faces else None),
            "face_share": round(face["share"], 3) if face else None,
        })

    sides = [min(entry["width"], entry["height"]) for entry in seen]
    aspects = Counter(round(entry["width"] / entry["height"], 1) for entry in seen)
    top_share = max(aspects.values()) / len(seen) if seen else 0.0
    bright = sorted(entry["bright"] for entry in seen)
    low, high = _percentiles(bright)
    flag_counts = Counter(flag for row in rows for flag in row["flags"])
    issues = sum(1 for row in rows if set(row["flags"]) - _NOT_QUALITY)

    scores = {
        "captions": _clamp(captioned / total) if total else 0.0,
        "consistency": _clamp(consistency["score"]) if captioned else 0.0,
        "quality": _clamp(1.0 - issues / total) if total else 0.0,
        "uniqueness": _clamp(1.0 - len(paired) / total) if total else 0.0,
        "balance": 1.0 if len(aspects) <= 1 else _clamp(1.2 - top_share),
    }
    face_figures = None
    if counted is not None:
        clean = sum(1 for row in rows if not set(row["flags"]) & FACE_ISSUES)
        face_figures = {
            "checked": total,
            "one": clean,
            "none": flag_counts.get("no face", 0),
            "several": flag_counts.get("several faces", 0),
            "small": flag_counts.get("small face", 0),
            "with_face": sum(1 for face in counted if face and face["count"] > 0),
        }
        scores["faces"] = _clamp(clean / total) if total else 0.0

    return {
        "pictures": total,
        "captioned": captioned,
        "subject": subject,
        "coverage": coverage,
        "consistency": consistency,
        "phrases": phrases,
        "short_side_median": int(statistics.median(sides)) if sides else 0,
        "below_target": sum(1 for side in sides if target_side > 0 and side < target_side),
        "target_side": int(target_side),
        "brightness_median": round(statistics.median(bright), 3) if bright else 0.0,
        "brightness_spread": round(high - low, 3),
        "aspects": [{"ratio": ratio, "count": count} for ratio, count in aspects.most_common()],
        "aspect_top_share": round(top_share, 3),
        "flag_counts": {flag: flag_counts[flag] for flag in FLAG_ORDER if flag_counts.get(flag)},
        "image_issues": issues,
        "near_duplicates": len(paired),
        "pairs": found,
        "faces": face_figures,
        "rows": rows,
        "scores": {name: round(value, 3) for name, value in scores.items()},
    }


def _flags(entry, face, sharp_median, sat_median, target_side, paired, check_faces) -> list[str]:
    """The flags one picture earns, in :data:`FLAG_ORDER`.

    Args:
        entry: Its :func:`look`, or None where it could not be read.
        face: Its :func:`faces` answer, or None.
        sharp_median: The set's median sharpness.
        sat_median: The set's median saturation.
        target_side: Short side under which it is small. 0 checks nothing.
        paired: Whether it has a near-duplicate.
        check_faces: Whether faces were counted.

    Returns:
        The flag names.
    """
    flags = []
    if entry is None:
        flags.append("unreadable")
    else:
        if entry["bright"] < DARK:
            flags.append("dark")
        if entry["bright"] > BLOWN_OUT:
            flags.append("blown out")
        if entry["contrast"] < FLAT:
            flags.append("flat")
        if sharp_median > 0 and entry["sharp"] < BLURRY * sharp_median:
            flags.append("blurry")
        if entry["border"] > BARS:
            flags.append("bars")
        if entry["sat"] < GREYSCALE and sat_median > GREYSCALE_SET:
            flags.append("greyscale")
        if entry["alpha"] > TRANSPARENT:
            flags.append("transparent")
        if entry["jpeg"] is not None and entry["jpeg"] < JPEG_LOW:
            flags.append("compressed")
        if target_side > 0 and min(entry["width"], entry["height"]) < target_side:
            flags.append("small")
    if paired:
        flags.append("near-duplicate")
    if check_faces:
        if not face or face["count"] == 0:
            flags.append("no face")
        else:
            if face["second"] >= SECOND_FACE * face["share"]:
                flags.append("several faces")
            if face["share"] < SMALL_FACE:
                flags.append("small face")
    return flags


def _key(path) -> tuple | None:
    """The cache key of one file: its path, modification time and size, or None when it is gone."""
    try:
        info = os.stat(path)
    except OSError:
        return None
    return (str(path), info.st_mtime_ns, info.st_size)


def _remember(cache: OrderedDict, key: tuple, value) -> None:
    """Hold one answer, dropping the least recently used past :data:`MAX_CACHED`."""
    with _lock:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > MAX_CACHED:
            cache.popitem(last=False)


def _caption_file(image, suffix: str) -> Path | None:
    """The caption file beside a picture, resolved inside the read roots, or None where it is not one."""
    try:
        path = Path(sandbox.resolve_read(Path(image).with_suffix(suffix)))
    except (sandbox.PathNotAllowed, ValueError) as error:
        logger.debug("a caption outside the readable folders was skipped (%s)", error)
        return None
    return path if path.is_file() else None


def _relative(path, base: Path) -> str:
    """A picture's path below the folder it was found in, with forward slashes."""
    return os.path.relpath(str(path), str(base)).replace("\\", "/")


def _clamp(value: float) -> float:
    """A score held to 0 to 1."""
    return max(0.0, min(1.0, float(value)))


def _percentiles(values) -> tuple[float, float]:
    """The 10th and 90th percentile of sorted values, both 0 below three values."""
    if len(values) < 3:
        return 0.0, 0.0
    return values[int(len(values) * 0.1)], values[min(len(values) - 1, int(len(values) * 0.9))]


def _decoded(path: str, edge: int):
    """A picture fitted inside ``edge`` pixels a side, stood upright.

    Args:
        path: The picture, already contained.
        edge: The longest side it is fitted to.

    Returns:
        ``(picture, width, height, quality)``: the picture in ``RGB`` or ``RGBA``, its size
        upright before fitting, and its JPEG quality or None.

    Raises:
        OSError: The file is not a picture PIL reads.
        ValueError: The picture holds more pixels than the pack decodes.
    """
    import numpy as np
    from PIL import Image

    from ..image.sizing import MAX_SOURCE_PIXELS

    with Image.open(path) as opened:
        width, height = opened.size
        if width < 1 or height < 1 or width * height > MAX_SOURCE_PIXELS:
            raise ValueError(f"{width}x{height} is past the {MAX_SOURCE_PIXELS} pixels decoded")
        quality = jpeg_quality(opened)
        try:
            orientation = int(opened.getexif().get(0x0112, 1))
        except Exception:
            orientation = 1
        opened.draft("RGB", (edge * 2, edge * 2))
        picture = opened
        if picture.mode.startswith("I;16"):
            picture = picture.convert("I")
        if picture.mode in ("I", "F"):
            picture = picture.copy()
            picture.thumbnail((edge, edge))
            values = np.asarray(picture, dtype=np.float32)
            top = 65535.0 if values.size and float(values.max()) > 255.0 else 255.0
            picture = Image.fromarray(np.clip(values * (255.0 / top), 0, 255).astype(np.uint8), "L")
        clear = picture.mode in ("RGBA", "LA", "PA") or (
            picture.mode == "P" and "transparency" in picture.info
        )
        picture = picture.convert("RGBA" if clear else "RGB")
        picture.thumbnail((edge, edge))
    method = _ORIENTATIONS.get(orientation)
    if method is not None:
        picture = picture.transpose(getattr(Image.Transpose, method))
        if orientation >= 5:
            width, height = height, width
    return picture, width, height, quality


def _from_pixels(pixels, edge: int):
    """A tensor picture as a PIL image fitted inside ``edge`` pixels a side.

    Args:
        pixels: A ``(1, height, width, channels)`` or ``(height, width, channels)`` float tensor
            in ``[0, 1]``; a fourth channel is read as alpha.
        edge: The longest side it is fitted to.

    Returns:
        ``(picture, width, height)``: the picture in ``RGB`` or ``RGBA`` and its size before fitting.

    Raises:
        ValueError: The tensor is not one picture of 1 to 4 channels.
    """
    import numpy as np
    from PIL import Image

    frame = pixels[0] if len(pixels.shape) == 4 else pixels
    if len(frame.shape) != 3 or not 1 <= int(frame.shape[-1]) <= 4 or min(frame.shape[:2]) < 1:
        raise ValueError(f"pixels shaped {tuple(pixels.shape)} are not one picture")
    values = np.ascontiguousarray(
        frame.detach().float().clamp(0.0, 1.0).mul(255.0).round().byte().cpu().numpy()
    )
    height, width, channels = values.shape
    if channels == 1:
        picture = Image.fromarray(values[..., 0], "L").convert("RGB")
    elif channels == 2:
        picture = Image.fromarray(values, "LA").convert("RGBA")
    else:
        picture = Image.fromarray(values, "RGB" if channels == 3 else "RGBA")
    picture.thumbnail((edge, edge))
    return picture, int(width), int(height)


def _measure(path: str) -> dict | None:
    """Measure one picture file's look, or None where it cannot be read.

    Args:
        path: The picture, already contained.

    Returns:
        The fields :func:`look` describes.
    """
    try:
        picture, width, height, quality = _decoded(path, LOOK_EDGE)
    except Exception as error:
        logger.debug("a picture could not be measured (%s)", type(error).__name__)
        return None
    return _measured(picture, width, height, quality)


def _measured(picture, width: int, height: int, quality: int | None) -> dict:
    """Measure one picture already fitted inside :data:`LOOK_EDGE` pixels a side.

    Args:
        picture: The fitted picture in ``RGB`` or ``RGBA``.
        width: Its width upright before fitting.
        height: Its height upright before fitting.
        quality: Its JPEG quality, or None.

    Returns:
        The fields :func:`look` describes.
    """
    import numpy as np

    pixels = np.asarray(picture, dtype=np.float32) / 255.0
    rgb = pixels[..., :3]
    seen = pixels[..., 3] if pixels.shape[-1] == 4 else None
    luma = rgb @ np.array(_LUMA, np.float32)
    brightest, darkest = rgb.max(-1), rgb.min(-1)
    saturation = np.where(brightest > 0, (brightest - darkest) / np.maximum(brightest, 1e-6), 0.0)
    alpha = float((seen < 250.0 / 255.0).mean()) if seen is not None else 0.0

    if alpha > TRANSPARENT and float(seen.sum()) > 0.01 * seen.size:
        # Weighted by how much each pixel shows, sharpness where all five of a cross show.
        laplacian = _laplacian(luma)
        bright = _weighted(luma, seen)
        contrast = _weighted((luma - bright) ** 2, seen) ** 0.5
        sat = _weighted(saturation, seen)
        inner = np.minimum.reduce(
            [seen[1:-1, 1:-1], seen[:-2, 1:-1], seen[2:, 1:-1], seen[1:-1, :-2], seen[1:-1, 2:]]
        )
        centre = _weighted(laplacian, inner) if float(inner.sum()) > 0 else 0.0
        sharp = _weighted((laplacian - centre) ** 2, inner) if float(inner.sum()) > 0 else 0.0
        border = 0.0
    else:
        top, bottom, left, right = _bars(luma)
        height_px, width_px = luma.shape
        kept = max(0, height_px - top - bottom) * max(0, width_px - left - right)
        border = 1.0 - kept / max(1, height_px * width_px)
        # Exposure and sharpness are read on what lies inside the bars.
        if border > 0 and height_px - top - bottom >= 3 and width_px - left - right >= 3:
            luma = luma[top:height_px - bottom, left:width_px - right]
            saturation = saturation[top:height_px - bottom, left:width_px - right]
        laplacian = _laplacian(luma)
        bright = float(luma.mean())
        contrast = float(luma.std())
        sat = float(saturation.mean())
        sharp = float(laplacian.var()) if laplacian.size else 0.0

    return {
        "width": int(width),
        "height": int(height),
        "bright": round(bright, 4),
        "contrast": round(contrast, 4),
        "sharp": round(sharp * 1000.0, 4),
        "sat": round(sat, 4),
        "border": round(border, 3),
        "alpha": round(alpha, 4),
        "jpeg": quality,
        **_signature(picture.convert("RGB")),
    }


def _weighted(values, weights) -> float:
    """The mean of ``values`` with each counted as much as its weight."""
    total = float(weights.sum())
    return float((values * weights).sum() / total) if total > 0 else 0.0


def _laplacian(luma):
    """The four-neighbour Laplacian of a brightness plane, one pixel smaller on every side."""
    return luma[:-2, 1:-1] + luma[2:, 1:-1] + luma[1:-1, :-2] + luma[1:-1, 2:] - 4.0 * luma[1:-1, 1:-1]


def _bars(luma) -> tuple[int, int, int, int]:
    """How deep the flat bars along each edge of a picture run, such as letterboxing.

    Args:
        luma: Brightness, 0 to 1, shaped ``(height, width)``.

    Returns:
        ``(top, bottom, left, right)`` in pixels, 0 where an edge has no bar.
    """
    import numpy as np

    def depth(lines) -> int:
        """Lines in from one edge that are one flat tone, ending where the picture starts."""
        limit = len(lines) // 3
        tone = float(lines[0].mean())
        count = 0
        while count < limit:
            line = lines[count]
            if float(line.std()) >= 0.02 or abs(float(line.mean()) - tone) >= 0.03:
                break
            count += 1
        if count == 0 or count >= limit:
            return 0
        return count if float(np.abs(lines[count] - tone).mean()) > 0.08 else 0

    return depth(luma), depth(luma[::-1]), depth(luma.T), depth(luma.T[::-1])


def _signature(picture) -> dict:
    """What :func:`pairs` compares: two difference hashes, a grey shape and a mean colour.

    Args:
        picture: The measured picture in ``RGB``.

    Returns:
        ``hash`` and ``hash_m``, 32 bytes each for the picture and its mirror, ``shape`` as
        256 floats of zero mean and unit length, and ``colour`` as three means from 0 to 255.
    """
    import numpy as np
    from PIL import Image

    grey = picture.convert("L")
    wide = np.asarray(grey.resize((17, 16), Image.Resampling.BILINEAR), np.int16)
    mirror = wide[:, ::-1]
    shape = np.asarray(grey.resize((16, 16), Image.Resampling.BILINEAR), np.float32).reshape(-1)
    shape -= shape.mean()
    shape /= float(np.linalg.norm(shape)) + 1e-6
    colour = np.asarray(picture.resize((8, 8), Image.Resampling.BILINEAR), np.float32).mean((0, 1))
    return {
        "hash": np.packbits(wide[:, 1:] > wide[:, :-1]).tobytes(),
        "hash_m": np.packbits(mirror[:, 1:] > mirror[:, :-1]).tobytes(),
        "shape": shape,
        "colour": tuple(round(float(value), 2) for value in colour),
    }


def _count_faces(path: str, detector) -> dict | None:
    """Count the faces in one picture, or None where it cannot be read.

    Args:
        path: The picture, already contained.
        detector: A detector from the pack's YuNet loader.

    Returns:
        The fields :func:`faces` describes.
    """
    from ..model import yunet

    try:
        picture, _width, _height, _quality = _decoded(path, yunet.DEFAULT_SIZE)
    except Exception as error:
        logger.debug("a picture could not be read for faces (%s)", type(error).__name__)
        return None
    return _faces_in(picture, detector)


def _faces_in(picture, detector) -> dict:
    """Count the faces in one picture already fitted to the detector's size.

    Args:
        picture: The fitted picture.
        detector: A detector from the pack's YuNet loader.

    Returns:
        The fields :func:`faces` describes.
    """
    import numpy as np

    from ..model import yunet

    pixels = np.ascontiguousarray(np.asarray(picture.convert("RGB"), dtype=np.uint8))
    found = yunet.detect(detector, pixels, score_threshold=FACE_SCORE)
    tall = max(1, pixels.shape[0])
    heights = sorted((max(0.0, float(row[3])) / tall for row in found), reverse=True)
    return {
        "count": len(heights),
        "share": min(1.0, heights[0]) if heights else 0.0,
        "second": min(1.0, heights[1]) if len(heights) > 1 else 0.0,
    }
