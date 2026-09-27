"""The colour profile a file carries, applied to the pixels it holds.

Every node in the pack reads an image as sRGB, so a file tagged with another space is
converted on the way in and its tag dropped.
"""

from __future__ import annotations

import io
from typing import NamedTuple

from .. import log

__all__ = [
    "ASSIGN",
    "Carried",
    "CONVERT",
    "KEEP",
    "MODES",
    "SPACES",
    "carried",
    "describe",
    "from_srgb",
    "from_srgb_array",
    "interpret",
    "profile_for",
    "to_srgb",
]

logger = log.get_logger("image.colour_profile")

#: Modes carrying colour a transform can be built for. Anything else is left alone.
COLOUR_MODES = frozenset({"RGB", "RGBA"})

#: What a profile description starts with when it already names sRGB.
SRGB_PREFIX = "srgb"

#: Rendering intent used for the conversion, which holds the relationship between colours
#: rather than their exact values.
INTENT = 0

#: Widget value that leaves a tagged file in the space it was written in.
KEEP = "the file's own"

#: What to do with the space the widget names: change the numbers so the colour is kept, or
#: keep the numbers and change what they mean.
CONVERT = "convert"
ASSIGN = "assign"
MODES = (CONVERT, ASSIGN)


class Carried(NamedTuple):
    """The colour profile a file was tagged with, and what became of its pixels.

    Attributes:
        name: What the profile calls itself, such as ``Adobe RGB (1998)``.
        data: The profile itself, as the bytes a file carries.
        converted: True when the pixels were brought to sRGB and this is where they came
            from, False when the pixels are still in this profile's own space.
    """

    name: str
    data: bytes
    converted: bool = True


def carried(image, converted: bool = True):
    """The profile an image arrived tagged with.

    Args:
        image: A decoded PIL image.
        converted: Whether the pixels beside it were brought to sRGB.

    Returns:
        A :class:`Carried`, or None for an image carrying no profile.
    """
    profile = (getattr(image, "info", None) or {}).get("icc_profile")
    if not profile:
        return None
    return Carried(
        name=describe(profile) or "unnamed", data=bytes(profile), converted=bool(converted)
    )


def spaces() -> list[str]:
    """Every colour space a widget offers, the file's own first."""
    from . import icc

    return [KEEP, *icc.SPACES]

def profile_for(space: str):
    """One named colour space, as a profile a file can carry.

    Args:
        space: A key of :data:`modules.image.icc.SPACES`.

    Returns:
        A :class:`Carried` holding the profile, or None where it cannot be built.
    """
    from . import icc

    try:
        from PIL import ImageCms

        if space == "sRGB":
            # littleCMS builds the piecewise curve sRGB really has, where a plain exponent
            # is five codes out in the darks.
            data = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        else:
            data = icc.build(space)
    except Exception as error:
        logger.warning("the %s profile could not be built (%s)", space, error)
        return None
    return Carried(name=describe(data) or space, data=data, converted=False)

def interpret(image, space: str, mode: str, name: str = ""):
    """Put an image into the colour space a widget names.

    Args:
        image: A decoded PIL image, tagged or not.
        space: :data:`KEEP`, or a key of :data:`modules.image.icc.SPACES`.
        mode: :data:`CONVERT` to change the numbers and keep the colour, :data:`ASSIGN` to
            keep the numbers and change what they mean.
        name: The file it came from, named in the log line.

    Returns:
        ``(image, profile)``. The profile says what the pixels are now, and carries
        ``converted`` set where they are sRGB and the profile is where they came from.
    """
    tagged = carried(image, converted=False)
    if space == KEEP or mode not in MODES:
        return image, tagged

    target = profile_for(space)
    if target is None:
        return image, tagged
    if mode == ASSIGN:
        logger.info(
            "%s is read as %s, with its numbers left alone.", name or "an image", space
        )
        return image, target

    # Nothing on an untagged file says otherwise, so it is read as sRGB before converting.
    source = tagged or Carried(name="sRGB", data=target.data, converted=False)
    if tagged is None and space == "sRGB":
        return image, None
    moved = _through(image, source, target, name)
    if moved is None:
        return image, tagged
    # Converting into sRGB is the working case: the file's own space is answered so a save
    # can put the picture back the way it arrived.
    if space == "sRGB" and tagged is not None:
        return moved, Carried(name=tagged.name, data=tagged.data, converted=True)
    return moved, target

def interpret_deep(tagged, space: str, mode: str, name: str = ""):
    """The profile a 16-bit picture is read as, its numbers left as they are.

    Args:
        tagged: The profile the file carries, as :func:`carried` answers it, or None.
        space: :data:`KEEP`, or a key of :data:`modules.image.icc.SPACES`.
        mode: :data:`CONVERT` or :data:`ASSIGN`.
        name: The file it came from, named in the log line.

    Returns:
        The profile the pixels are in. A conversion is declined rather than run at 8 bits.
    """
    if tagged is not None:
        tagged = Carried(name=tagged.name, data=tagged.data, converted=False)
    if space == KEEP or mode not in MODES:
        return tagged
    target = profile_for(space)
    if target is None:
        return tagged
    if mode == ASSIGN:
        logger.info("%s is read as %s, with its numbers left alone.", name or "an image", space)
        return target
    # An untagged file is read as sRGB, so bringing either to sRGB changes nothing.
    already_srgb = tagged is None or tagged.name.lower().startswith(SRGB_PREFIX)
    if already_srgb and space == "sRGB":
        return None
    logger.warning(
        "%s is 16 bits a channel and a colour transform runs at 8, so its numbers are kept in "
        "%s rather than converted to %s. Set the mode to assign to relabel it without a "
        "transform.",
        name or "an image", tagged.name if tagged is not None else "sRGB", space,
    )
    return tagged


def _through(image, source, target, name: str):
    """One image sent from one profile into another, or None where it cannot be."""
    if image.mode not in COLOUR_MODES:
        return None
    try:
        from PIL import ImageCms

        moved = ImageCms.profileToProfile(
            image,
            ImageCms.ImageCmsProfile(io.BytesIO(source.data)),
            ImageCms.ImageCmsProfile(io.BytesIO(target.data)),
            renderingIntent=INTENT,
            outputMode=image.mode,
        )
    except Exception as error:
        logger.warning(
            "%s could not be converted from %s to %s (%s), so its numbers are left alone.",
            name or "an image", source.name, target.name, error,
        )
        return None
    if moved is not None:
        moved.info.pop("icc_profile", None)
        logger.info(
            "%s converted from %s to %s.", name or "an image", source.name, target.name
        )
    return moved


def from_srgb(image, profile):
    """Convert an sRGB image into the space one profile describes.

    Args:
        image: A decoded PIL image holding sRGB colour.
        profile: The profile to convert into, as :func:`carried` answers it.

    Returns:
        ``(image, profile bytes)`` for saving. The image comes back unchanged and the bytes
        are empty when the profile cannot be applied, so a save is never blocked by it.
    """
    if profile is None or not getattr(profile, "data", b"") or image.mode not in COLOUR_MODES:
        return image, b""
    # Pixels already in the profile's own space are tagged and nothing else.
    if not profile.converted or profile.name.lower().startswith(SRGB_PREFIX):
        return image, profile.data

    try:
        from PIL import ImageCms

        converted = ImageCms.profileToProfile(
            image,
            ImageCms.createProfile("sRGB"),
            ImageCms.ImageCmsProfile(io.BytesIO(profile.data)),
            renderingIntent=INTENT,
            outputMode=image.mode,
        )
    except Exception as error:
        logger.warning(
            "the %s profile could not be applied on the way out (%s), so the file is written "
            "in sRGB and carries no profile.",
            profile.name, error,
        )
        return image, b""
    if converted is None:
        return image, b""
    return converted, profile.data


def from_srgb_array(frame, profile, deep: bool = False):
    """Convert an sRGB pixel array into the space one profile describes.

    Args:
        frame: ``(height, width, channels)`` float array in ``[0, 1]`` holding sRGB colour.
        profile: The profile to convert into, as :func:`carried` answers it.
        deep: The file is written at more than 8 bits a channel, which no CMS transform
            here can carry, so a conversion is declined rather than run at 8 bits.

    Returns:
        ``(frame, profile bytes)`` for saving. The frame comes back unchanged and the bytes
        are empty when the profile cannot be applied, so a save is never blocked by it.
    """
    import numpy as np

    data = getattr(profile, "data", b"")
    if profile is None or not data or np.asarray(frame).ndim != 3:
        return frame, b""
    array = np.asarray(frame)
    if array.shape[2] not in (3, 4):
        return frame, b""
    # Pixels already in the profile's own space are tagged and nothing else.
    if not profile.converted or profile.name.lower().startswith(SRGB_PREFIX):
        return frame, data
    if deep:
        logger.warning(
            "the %s profile needs a colour transform, which is only available at 8 bits a "
            "channel, so the file is written in sRGB and carries no profile. Set bit_depth "
            "to 8-bit to convert into it.",
            profile.name,
        )
        return frame, b""

    from PIL import Image

    mode = "RGB" if array.shape[2] == 3 else "RGBA"
    quantised = np.rint(np.clip(array, 0.0, 1.0) * 255).astype(np.uint8)
    converted, tag = from_srgb(Image.fromarray(quantised, mode=mode), profile)
    if not tag:
        return frame, b""
    return np.asarray(converted).astype(np.float32) / 255.0, tag


def describe(profile: bytes) -> str:
    """The name a profile gives itself.

    Args:
        profile: The ``icc_profile`` bytes out of an image's ``info``.

    Returns:
        The profile's description, or an empty string when it cannot be read.
    """
    try:
        from PIL import ImageCms

        return ImageCms.getProfileDescription(
            ImageCms.ImageCmsProfile(io.BytesIO(profile))
        ).strip()
    except Exception as error:
        logger.debug("a colour profile could not be described (%s)", error)
        return ""


def to_srgb(image, name: str = ""):
    """Convert an image to sRGB through the profile it carries.

    Args:
        image: A decoded PIL image, which may carry an ``icc_profile`` in its ``info``.
        name: The file it came from, named in the log line.

    Returns:
        The image in sRGB, with its profile dropped. An image with no profile, one already
        in sRGB, and one whose profile cannot be applied all come back unchanged.
    """
    profile = (getattr(image, "info", None) or {}).get("icc_profile")
    if not profile or image.mode not in COLOUR_MODES:
        return image

    described = describe(profile)
    if described.lower().startswith(SRGB_PREFIX):
        image.info.pop("icc_profile", None)
        return image

    try:
        from PIL import ImageCms

        converted = ImageCms.profileToProfile(
            image,
            ImageCms.ImageCmsProfile(io.BytesIO(profile)),
            ImageCms.createProfile("sRGB"),
            renderingIntent=INTENT,
            outputMode=image.mode,
        )
    except Exception as error:
        logger.warning(
            "%s carries a %s profile that could not be applied (%s), so its colours are read "
            "as sRGB and will look flatter than the file does elsewhere.",
            name or "an image", described or "colour", error,
        )
        return image
    if converted is None:
        return image

    converted.info.pop("icc_profile", None)
    logger.info(
        "%s carries a %s profile, converted to sRGB.", name or "an image", described or "colour"
    )
    return converted
