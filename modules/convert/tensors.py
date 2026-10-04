"""Conversions between ComfyUI's tensors, PIL images and numpy arrays.

An ``IMAGE`` is float32 ``(batch, height, width, channels)`` in ``[0, 1]``. A ``MASK`` is
``(batch, height, width)``, ``(batch, 1, height, width)``, ``(batch, height, width, 1)`` or
``(height, width)``.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from PIL import Image

from ..image import scratch

__all__ = [
    "array_shape",
    "broadcast_image_planes",
    "filtered_planes",
    "image_planes",
    "mask2pil",
    "mask_images",
    "mask_planes",
    "pil2hex",
    "pil2mask",
    "pil2tensor",
    "plane2pil",
    "plane_shape",
    "sam2tensor",
    "stack_images",
    "tensor2pil",
    "tensor2sam",
]

#: Trailing axis lengths an ``IMAGE`` plane's channel axis can have: greyscale, RGB and RGBA.
CHANNEL_COUNTS = (1, 3, 4)

#: Raised by the single-image conversions when the argument is a whole batch, with the
#: batch size and the name of the split that fixes the call.
BATCH_GIVEN = (
    "{name}() converts one image and was given a batch of {size}. Split the batch with "
    "image_planes() from modules.convert.tensors and convert one plane at a time."
)


def _reject_batch(name: str, image: torch.Tensor) -> None:
    """Refuse a batched image tensor on behalf of a single-image conversion.

    Args:
        name: Conversion the message names, without parentheses.
        image: Tensor about to be converted. Four axes with a leading axis longer than 1
            is a batch; every other shape passes.

    Raises:
        ValueError: The tensor holds more than one image.
    """
    if image.ndim == 4 and image.shape[0] > 1:
        raise ValueError(BATCH_GIVEN.format(name=name, size=image.shape[0]))


def tensor2pil(image: torch.Tensor) -> Image.Image:
    """Convert a single image tensor to a PIL image.

    Args:
        image: Float tensor scaled to ``[0, 1]`` holding one image. Every length-1 axis is
            squeezed away, so a batch of one becomes ``(height, width, channels)`` and a
            single-channel image becomes a 2D greyscale array.

    Returns:
        An 8-bit PIL image. Values outside ``[0, 1]`` are clipped rather than scaled.

    Raises:
        ValueError: The tensor holds a batch of more than one image, which
            :func:`image_planes` splits into one argument per call.
        TypeError: PIL cannot build an image from the squeezed array.
    """
    _reject_batch("tensor2pil", image)
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def image_planes(images: torch.Tensor) -> list[torch.Tensor]:
    """Split an image tensor into one image per item of its batch.

    Args:
        images: Image tensor in the layout an ``IMAGE`` socket carries. Four axes are read
            as ``(batch, height, width, channels)`` and iterated. Three axes are read as
            ``(height, width, channels)`` when the last of them is one of
            :data:`CHANNEL_COUNTS`, and as a batch of greyscale frames otherwise, which is
            the layout a node converting to mode ``L`` emits. Two or fewer are read as one
            unbatched image. A ``MASK`` is split
            by :func:`mask_planes` instead.

    Returns:
        One view per image, in batch order, each carrying the axes it had inside the
        batch. A tensor of two or fewer axes gives a list holding it unchanged.
    """
    if images.ndim >= 4:
        return list(images)
    if images.ndim == 3 and int(images.shape[-1]) not in CHANNEL_COUNTS:
        return list(images)
    return [images]


def mask_planes(masks: torch.Tensor) -> list[torch.Tensor]:
    """Split a mask tensor into one 2D mask per item of its batch.

    Args:
        masks: Mask tensor in any of the layouts a ``MASK`` socket carries. Three axes or
            more are read as a batch and iterated; two are read as a single unbatched
            mask; fewer are read as one row.

    Returns:
        One ``(height, width)`` view per mask, in batch order, and none for a batch holding
        no mask.
    """
    if masks.ndim < 2:
        masks = masks.reshape(1, -1)
    if masks.ndim == 2:
        return [masks]
    return [_mask_plane(mask) for mask in masks]


def _mask_plane(mask: torch.Tensor) -> torch.Tensor:
    """Reduce one mask of a batch to two axes by dropping length-1 leading and trailing axes.

    Args:
        mask: One item of a mask batch, of any rank.

    Returns:
        A view of ``mask`` with two axes, or with however many are left when neither the
        first nor the last is length 1.
    """
    while mask.ndim > 2 and mask.shape[0] == 1:
        mask = mask[0]
    while mask.ndim > 2 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    return mask


def mask_images(masks: torch.Tensor) -> list[Image.Image]:
    """Convert every mask of a batch to an 8-bit greyscale image.

    Args:
        masks: Mask tensor in any of the layouts a ``MASK`` socket carries, holding values
            in ``[0, 1]``. Values outside that range are clipped rather than scaled.

    Returns:
        One mode ``L`` image per mask, in batch order.
    """
    return [mask2pil(plane) for plane in mask_planes(masks)]


def broadcast_image_planes(*images: torch.Tensor) -> list[tuple[torch.Tensor, ...]]:
    """Align several image tensors frame by frame, cycling shorter batches to the longest length.

    Args:
        *images: One image tensor per socket, in the order the node reads them. Each is
            split by :func:`image_planes`.

    Returns:
        One tuple per output frame, holding one image from each input in argument order.

    Raises:
        ValueError: No tensor was given, or one of them holds an empty batch.
    """
    if not images:
        raise ValueError("At least one image must be provided.")
    planes = [image_planes(image) for image in images]
    if not all(planes):
        raise ValueError("An image tensor holds an empty batch and has no frame to align.")
    length = max(len(group) for group in planes)
    return [tuple(group[index % len(group)] for group in planes) for index in range(length)]


def array_shape(image) -> tuple[int, ...]:
    """The shape :func:`numpy.array` gives an image, read without decoding its pixels.

    Args:
        image: A PIL image, or anything :func:`numpy.shape` accepts.

    Returns:
        ``(height, width)`` for a single-band image and ``(height, width, bands)`` otherwise.
    """
    if isinstance(image, Image.Image):
        bands = len(image.getbands())
        return (image.height, image.width) + ((bands,) if bands != 1 else ())
    return tuple(int(side) for side in np.shape(image))


def _place(slot: torch.Tensor, plane: torch.Tensor) -> None:
    """Copy one image into its slot of a batch, centred on black where it is smaller.

    Args:
        slot: ``(height, width, ...)`` view of the batch, overwritten.
        plane: The image, no larger than ``slot`` on either side.
    """
    high, wide = int(plane.shape[0]), int(plane.shape[1])
    if (high, wide) == (int(slot.shape[0]), int(slot.shape[1])):
        slot.copy_(plane)
        return
    slot.zero_()
    top = (int(slot.shape[0]) - high) // 2
    left = (int(slot.shape[1]) - wide) // 2
    slot[top:top + high, left:left + wide] = plane


def stack_images(
    images: list[Image.Image], node: str = "", advice: str = "Fewer or smaller images also fit it.",
) -> torch.Tensor:
    """Assemble PIL images into one ``IMAGE`` batch, centred and black-padded to the largest size.

    Args:
        images: One PIL image per item of the batch, in batch order. All of them must
            carry the same number of channels, since one batch carries one channel count.
        node: Display name of the calling node, named where the batch does not fit.
        advice: Closing sentence of that refusal, naming what to change on the node.

    Returns:
        A float32 tensor scaled to ``[0, 1]``, shaped
        ``(len(images), height, width, channels)`` for a colour batch and
        ``(len(images), height, width)`` for a greyscale one.

    Raises:
        ValueError: No image was given, or the images differ in channel count.
        TypeError: A tensor was given where a PIL image was expected; a batch of tensors
            is assembled with :func:`torch.cat` instead.
        MemoryError: Neither free memory nor any scratch drive has room for the batch.
    """
    if not images:
        raise ValueError("At least one image must be provided.")
    for image in images:
        if isinstance(image, torch.Tensor):
            raise TypeError(
                "stack_images() assembles PIL images. Concatenate image tensors with "
                "torch.cat(planes, dim=0) instead."
            )
    shapes = [array_shape(image) for image in images]
    channels = {shape[2:] for shape in shapes}
    if len(channels) > 1:
        raise ValueError(f"All images must have the same channel count, got {sorted(channels)}.")
    height = max(shape[0] for shape in shapes)
    width = max(shape[1] for shape in shapes)
    batch = scratch.allocate(
        (len(images), height, width) + shapes[0][2:], node=node, advice=advice,
    )
    for index, image in enumerate(images):
        plane = pil2tensor(image)[0]
        if tuple(plane.shape) != shapes[index]:
            raise ValueError(
                f"Image {index} decoded to the shape {tuple(plane.shape)}, not the "
                f"{shapes[index]} its size and mode give."
            )
        _place(batch[index], plane)
    return batch


def _progress(total: int):
    """A callable advancing the running node's progress bar by one frame.

    Args:
        total: Frames in the batch.

    Returns:
        A callable that also stops a cancelled run; it does nothing outside ComfyUI or for a
        single frame.
    """
    if total < 2:
        return lambda: None
    try:
        import comfy.model_management
        import comfy.utils
    except ImportError:
        return lambda: None
    bar = comfy.utils.ProgressBar(total)

    def step():
        comfy.model_management.throw_exception_if_processing_interrupted()
        bar.update(1)

    return step


def filtered_planes(images: torch.Tensor, filter_fn) -> torch.Tensor:
    """Run a PIL filter over every image of a batch, keeping light outside 0 to 1.

    Args:
        images: The batch, in the layout :func:`image_planes` reads.
        filter_fn: Called with one PIL image, answering the filtered PIL image.

    Returns:
        The filtered batch, assembled by :func:`stack_images`, on the scale it arrived on
        and on the device and floating dtype it arrived on. A batch already inside 0 to 1
        takes the same path it would without this.
    """
    from ..image import dynamic

    scale = dynamic.peak(images) if dynamic.carries(images) else 1.0
    planes = image_planes(images)
    step = _progress(len(planes))
    filtered = []
    for plane in planes:
        folded = dynamic.fold(plane, scale)
        filtered.append(filter_fn(tensor2pil(folded.images)))
        step()
    result = stack_images(filtered)
    del filtered
    if folded.scale != 1.0:
        result.mul_(folded.scale)
    if images.is_floating_point():
        return result.to(device=images.device, dtype=images.dtype)
    return result.to(images.device)


def pil2tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to an image tensor.

    Args:
        image: Any PIL image, or anything :func:`numpy.array` accepts.

    Returns:
        A float32 tensor scaled to ``[0, 1]`` carrying a leading batch axis of one. The
        remaining axes are the image's own, so an ``RGB`` image gives
        ``(1, height, width, 3)`` and an ``L`` image gives ``(1, height, width)``.
    """
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def pil2hex(image: torch.Tensor) -> str:
    """Digest an image tensor.

    Args:
        image: An image tensor, converted with :func:`tensor2pil` first, so the argument
            is a tensor rather than the PIL image the name suggests.

    Returns:
        The SHA-256 hex digest of the pixel buffer, widened from uint8 to uint16 before
        hashing, which pads every sample with a zero byte.

    Raises:
        ValueError: The tensor holds a batch of more than one image, since one digest
            describes one image.
    """
    return hashlib.sha256(np.array(tensor2pil(image)).astype(np.uint16).tobytes()).hexdigest()


def pil2mask(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to an inverted mask tensor.

    Args:
        image: Any PIL image; it is converted to greyscale first.

    Returns:
        A float32 tensor shaped ``(height, width)`` with no batch axis, holding
        ``1.0 - luminance``: black pixels arrive as 1.0 and white pixels as 0.0.
    """
    image_np = np.array(image.convert("L")).astype(np.float32) / 255.0
    mask = torch.from_numpy(image_np)
    return 1.0 - mask


def mask2pil(mask: torch.Tensor) -> Image.Image:
    """Convert a mask tensor to a greyscale PIL image.

    The inversion :func:`pil2mask` applies is not undone.

    Args:
        mask: Mask tensor holding values in ``[0, 1]``, which are scaled to ``[0, 255]``
            and clipped. The leading axis is squeezed off when the tensor has more than
            two dimensions.

    Returns:
        A PIL image in mode ``L``.
    """
    if mask.ndim > 2:
        mask = mask.squeeze(0)
    mask_np = np.clip(255. * mask.cpu().numpy(), 0, 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_np, mode="L")
    return mask_pil


def plane_shape(plane: torch.Tensor) -> tuple[int, int, int]:
    """The height, width and channel count one image plane carries.

    Args:
        plane: Image plane, as :func:`image_planes` or :func:`mask_planes` answers it.
            Leading length-1 axes are dropped, so ``(1, height, width, 3)`` reads the same
            as ``(height, width, 3)``.

    Returns:
        ``(height, width, channels)``, the channel count being 1 for a plane with no
        channel axis.

    Raises:
        ValueError: The plane has neither two nor three axes once leading length-1 axes
            are dropped, so it is not one image.
    """
    shape = tuple(int(size) for size in plane.shape)
    while len(shape) > 3 and shape[0] == 1:
        shape = shape[1:]
    if len(shape) == 3:
        return shape[0], shape[1], shape[2]
    if len(shape) == 2:
        return shape[0], shape[1], 1
    raise ValueError(
        f"An image plane has two or three axes and this one is shaped {tuple(plane.shape)}. "
        f"Split a batch with image_planes() from modules.convert.tensors first."
    )


def plane2pil(plane: torch.Tensor) -> Image.Image:
    """Convert one image plane to a PIL image, reading its axes by rank.

    Args:
        plane: Float tensor scaled to ``[0, 1]`` holding one image, as
            :func:`image_planes` answers: ``(height, width, channels)``, ``(height,
            width)``, or either of those behind leading length-1 axes.

    Returns:
        An 8-bit PIL image the size the plane declares. Values outside ``[0, 1]`` are
        clipped rather than scaled.

    Raises:
        ValueError: The plane has neither two nor three axes once leading length-1 axes
            are dropped, so it is not one image.
    """
    height, width, channels = plane_shape(plane)
    array = np.clip(
        255. * plane.detach().cpu().numpy().reshape(height, width, channels), 0, 255,
    ).astype(np.uint8)
    # Mode L is built from two axes, so a lone channel axis is dropped.
    if channels == 1:
        array = array[..., 0]
    return Image.fromarray(array)


def tensor2sam(image: torch.Tensor) -> np.ndarray:
    """Convert a single image tensor to the HWC uint8 array Segment Anything expects.

    Args:
        image: Float tensor scaled to ``[0, 1]`` holding one image. Every length-1 axis is
            squeezed away before the shape is inspected.

    Returns:
        A uint8 array in HWC order, clipped to ``[0, 255]``. An array whose first axis is
        length 3 is transposed as though it were CHW.

    Raises:
        ValueError: The tensor holds a batch of more than one image. Nothing downstream
            of here reads a batch axis, so a batch would reach the segmenter as a
            four-dimensional array rather than as an image.
    """
    _reject_batch("tensor2sam", image)
    sam_image = np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    if sam_image.shape[0] == 3:
        sam_image = np.transpose(sam_image, (1, 2, 0))
    return sam_image


def sam2tensor(image: np.ndarray) -> torch.Tensor:
    """Convert an HWC array from Segment Anything to a CHW float tensor.

    Args:
        image: HWC array of 8-bit pixel values.

    Returns:
        A float32 tensor in CHW order scaled to ``[0, 1]``, with no batch axis.
    """
    float_image = image.astype(np.float32) / 255.0
    chw_image = np.transpose(float_image, (2, 0, 1))
    tensor_image = torch.from_numpy(chw_image)
    return tensor_image
