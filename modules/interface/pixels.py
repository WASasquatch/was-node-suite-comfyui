"""Both sides of a pixels node's picture, filed for its interface.

:func:`apply` wraps ``execute`` on a :data:`FAMILY` class, filing the first ``IMAGE`` input
as the before and output 0 as the after. Only the first frame is filed.
"""

from __future__ import annotations

import functools
import inspect

from .. import log
from . import preview

__all__ = ["FAMILY", "MARKER", "apply", "sides"]

logger = log.get_logger("interface.pixels")

#: ``io_type`` of the sockets this pairs. Nothing else on a node is a picture to compare.
IMAGE_TYPE = "IMAGE"

#: Set on a class once :func:`apply` has wrapped it, so a class collected twice is wrapped
#: once. A second frame would file every picture twice under the same key.
MARKER = "_was_pixels_preview"

#: The node ids that file a before and an after. ``web/was_pixels_before_after.js`` holds the
#: same list read from the other end, and draws the band for exactly these.
#:
#: `Image Seamless Texture` is deliberately absent although the edge blend it does is a
#: member's job. Its ``tiled`` combo repeats the answer ``tiles`` times along each side, so the
#: pair is two frames of different sizes, which the measurement band refuses and the fidelity
#: glyph can then claim nothing about. ``web/was_geometry_readout.js`` draws that multiplier as
#: a size instead, and one node carrying two panels leaves neither of them room.
#:
#: `Image Gradient Map` is absent for the same
#: reason. `web/was_image_generate_gradient.js` draws the ramp they read and the stops they are
#: edited through, and one node carrying two panels leaves neither of them room.
#:
#: `Image Flip` is here rather than among the sizes. A mirror moves every pixel and changes no
#: size, so a size band could only ever report the frame as unchanged, while the pair names the
#: axis it was mirrored on.
FAMILY = frozenset({
    "CLIPSEG2",
    "Image Blend",
    "Image Blend by Mask",
    "Image Blending Mode",
    "Image Bloom Filter",
    "Image Canny Filter",
    "Image Chromatic Aberration",
    "Image Color Match",
    "Image Displacement Warp",
    "Image Dragan Photography Filter",
    "Image Edge Detection Filter",
    "Image Film Grain",
    "Image Flip",
    "Image High Pass Filter",
    "Image Levels Adjustment",
    "Image Lucy Sharpen",
    "Image Median Filter",
    "Image Mix RGB Channels",
    "Image Monitor Effects Filter",
    "Image Pixelate",
    "Image Rembg (Remove Background)",
    "Image Remove Background (Alpha)",
    "Image Remove Color",
    "Image Rotate Hue",
    "Image SSAO (Ambient Occlusion)",
    "Image SSDO (Direct Occlusion)",
    "Image Select Channel",
    "Image Select Color",
    "Image Threshold",
    "Image fDOF Filter",
    "Image to Noise",
    "Images to Linear",
    "Images to RGB",
    "MiDaS Depth Approximation",
    "MiDaS Mask Image",
    "WASVividSharpen",
    "WASVividSharpenV2",
    "WASNSApplyLUT",
    "WASDrawImageBounds",
    "WASImageAutoLevels",
    "WASImageColorBalance",
    "WASImageCompositeMasked",
    "WASImageDequantise",
    "WASImageDirectionalBlur",
    "WASImageFrequencyBlend",
    "WASImageGuidedFilter",
    "WASImageLensDistortion",
    "WASImageTemporalEqualize",
    "WASImageToneMap",
    "WASImageVignette",
    "WASImageWhiteBalance",
})


def sides(schema) -> tuple[str, tuple[str, ...]]:
    """The socket a node's answer is compared with, and the sockets that only steer it.

    Args:
        schema: The node's ``io.Schema``.

    Returns:
        ``(before, controls)``. The before is the first ``IMAGE`` input, whose name the
        answer is also filed under; the controls are every later ``IMAGE`` input, in the
        order the sockets are drawn. ``("", ())`` when the node takes no picture.
    """
    names = _image_inputs(schema)
    if not names:
        return "", ()
    return names[0], names[1:]


def apply(node_cls, node_id=None) -> None:
    """Wrap ``node_cls.execute`` so both sides of its picture reach the interface.

    Args:
        node_cls: A node class the loader has collected.
        node_id: The id the loader already read off the schema. Left out, the schema is
            built to read it, which is the only reason a class outside :data:`FAMILY` costs
            anything at all.
    """
    if getattr(node_cls, MARKER, False):
        return
    try:
        _wrap(node_cls, node_id)
    except Exception as error:
        logger.debug("%s was left unwrapped (%s)", getattr(node_cls, "__name__", node_cls), error)


def _wrap(node_cls, node_id) -> None:
    """Do the wrapping. Split out so :func:`apply` owns the one guard around all of it."""
    if node_id is None:
        node_id = node_cls.GET_SCHEMA().node_id
    if node_id not in FAMILY:
        return
    setattr(node_cls, MARKER, True)
    schema = node_cls.GET_SCHEMA()
    before, controls = sides(schema)
    if not before:
        logger.debug("%s declares no IMAGE input, so neither side is filed", node_id)
        return
    if not _answers_image(schema):
        logger.debug("%s does not answer an IMAGE first, so neither side is filed", node_id)
        return
    declared = node_cls.__dict__.get("execute")
    if not isinstance(declared, classmethod):
        # Inherited from a shared base, or absent. Wrapping the inherited function here
        # would file one node's pictures under every sibling that shares the base.
        logger.debug("%s does not declare execute(), so neither side is filed", node_id)
        return
    function = declared.__func__

    if inspect.iscoroutinefunction(function):

        @functools.wraps(function)
        async def execute(cls, *args, **kwargs):
            _before(kwargs, before, controls)
            answer = await function(cls, *args, **kwargs)
            _after(answer, before)
            return answer

    else:

        @functools.wraps(function)
        def execute(cls, *args, **kwargs):
            _before(kwargs, before, controls)
            answer = function(cls, *args, **kwargs)
            _after(answer, before)
            return answer

    node_cls.execute = classmethod(execute)


def _before(values, before, controls) -> None:
    """File the pictures a node was handed, under the names of the sockets they arrived on.

    Args:
        values: The keyword arguments bound for one ``execute`` call.
        before: The socket the answer is compared with.
        controls: The other picture sockets, each filed under its own name.
    """
    _send(preview.publish, values.get(before), before)
    for name in controls:
        _send(preview.publish, values.get(name), name)


def _after(answer, before) -> None:
    """File the picture a node answered with, under the before's name.

    Args:
        answer: Whatever ``execute`` returned.
        before: The socket the answer is compared with, which names both sides.
    """
    _send(preview.publish_output, _first(answer), before)


def _send(publisher, value, name) -> None:
    """Hand one tensor to one publisher, at the cost of a picture and nothing more.

    Args:
        publisher: A :mod:`.preview` entry point taking a tensor and a ``slot``.
        value: What the socket carried. A value with no ``shape`` is skipped, which is what
            an optional picture input left unconnected carries.
        name: The socket's own id, which is the slot the picture is filed under.
    """
    if getattr(value, "shape", None) is None:
        return
    try:
        publisher(value, slot=name)
    except Exception as error:
        logger.debug("slot %r was not filed by %s (%s)", name, publisher.__name__, error)


def _first(answer):
    """The tensor on a node's first output.

    Args:
        answer: An ``io.NodeOutput``, a tuple, or anything else.

    Returns:
        The first returned value, or ``None`` when the answer carries none, which is what a
        blocked execution and an expanded graph both return.
    """
    values = getattr(answer, "args", None)
    if values is None and isinstance(answer, (tuple, list)):
        values = answer
    return values[0] if values else None


def _image_inputs(schema) -> tuple[str, ...]:
    """The picture sockets on one schema, required first then optional.

    Args:
        schema: The node's ``io.Schema``.

    Returns:
        Input ids, in the order the frontend draws the sockets, so the two ends of the
        family derive the same names from the same node.
    """
    specs = tuple(schema.inputs or ())
    picture = [spec for spec in specs if getattr(spec, "io_type", None) == IMAGE_TYPE]
    return tuple(
        [spec.id for spec in picture if not getattr(spec, "optional", False)]
        + [spec.id for spec in picture if getattr(spec, "optional", False)]
    )


def _answers_image(schema) -> bool:
    """Whether the node's first output is a picture.

    Args:
        schema: The node's ``io.Schema``.

    Returns:
        True when output 0 is an ``IMAGE``, which is the only output the after is read from.
    """
    outputs = tuple(schema.outputs or ())
    return bool(outputs) and getattr(outputs[0], "io_type", None) == IMAGE_TYPE
