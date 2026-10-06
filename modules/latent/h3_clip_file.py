"""A finished MiniMax H3 clip and the records its extend loop wrote, as one file.

The file is safetensors: every tensor the latent holds under its own name, and the latent's
layout as JSON in the header metadata under :data:`LATENT_KEY`.
"""

from __future__ import annotations

import json
import os

from . import h3_extend

__all__ = [
    "EXTENSION",
    "FORMAT",
    "VERSION",
    "decode",
    "encode",
    "read",
    "restored",
    "scene_lines",
    "scene_spans",
    "summary",
    "write",
]

#: Extension every saved clip carries.
EXTENSION = ".h3clip"

#: What the ``format`` metadata entry holds, and the layout version written.
FORMAT = "was_h3_clip"
VERSION = 1

#: Header metadata entries.
FORMAT_KEY = "format"
VERSION_KEY = "version"
LATENT_KEY = "latent"
SUMMARY_KEY = "summary"

#: Tags of the JSON layout, one per kind of value a latent carries.
TENSOR = "tensor"
NESTED = "nested"
DICT = "dict"
TUPLE = "tuple"
LIST = "list"

#: Types written into the layout as they are.
PLAIN = (str, int, float, bool, type(None))


def encode(latent: dict) -> tuple[dict, dict, list[str]]:
    """The tensors and the JSON layout one latent is written as.

    Args:
        latent: A latent dict.

    Returns:
        ``(tensors, layout, skipped)``: CPU tensors by name, the layout naming them, and
        ``"key (type)"`` for each entry holding something neither can carry.
    """
    tensors: dict = {}
    entries, skipped = [], []
    for key, value in latent.items():
        own: dict = {}
        try:
            entries.append([str(key), _described(value, [str(key)], own, tensors)])
        except TypeError:
            skipped.append(f"{key} ({type(value).__name__})")
            continue
        tensors.update(own)
    return tensors, {"keys": entries}, skipped


def decode(tensors: dict, layout: dict) -> dict:
    """The latent a layout and its tensors describe.

    Args:
        tensors: Tensors by name, as :func:`encode` named them.
        layout: The JSON layout :func:`encode` answered.

    Returns:
        The latent dict.

    Raises:
        ValueError: The layout names a tensor the file does not hold, or a tag it does not know.
    """
    return {str(key): _rebuilt(value, tensors) for key, value in layout.get("keys", [])}


def write(latent: dict, path: str) -> list[str]:
    """Write a latent to a clip file, replacing the file only once it is complete.

    Args:
        latent: An H3 joint latent.
        path: The file to write.

    Returns:
        The entries left out, as :func:`encode` reports them.
    """
    from safetensors.torch import save_file

    tensors, layout, skipped = encode(latent)
    metadata = {
        FORMAT_KEY: FORMAT,
        VERSION_KEY: str(VERSION),
        LATENT_KEY: json.dumps(layout),
        SUMMARY_KEY: json.dumps(summary(latent)),
    }
    partial = f"{path}.part"
    try:
        save_file(tensors, partial, metadata=metadata)
        os.replace(partial, path)
    finally:
        if os.path.exists(partial):
            os.remove(partial)
    return skipped


def read(path: str) -> dict:
    """The latent a clip file holds.

    Args:
        path: A file :func:`write` wrote.

    Returns:
        The latent, every tensor on the CPU.

    Raises:
        ValueError: The file is not a clip file, or a newer layout than this one reads.
    """
    from safetensors.torch import load

    with open(path, "rb") as handle:
        data = handle.read()
    metadata = _metadata(data, path)
    if metadata.get(FORMAT_KEY) != FORMAT or LATENT_KEY not in metadata:
        raise ValueError(
            f"{os.path.basename(path)} is not a clip H3 Save Clip wrote. Pick a "
            f"{EXTENSION} file H3 Save Clip saved"
        )
    try:
        version = int(metadata.get(VERSION_KEY, "0"))
    except ValueError:
        version = VERSION + 1
    if version > VERSION:
        raise ValueError(
            f"{os.path.basename(path)} was saved in clip layout {metadata.get(VERSION_KEY)}, and "
            f"this install reads up to {VERSION}. Update WAS Node Suite to read it"
        )
    return decode(load(data), json.loads(metadata[LATENT_KEY]))


def summary(latent: dict) -> dict:
    """What a clip holds, for a report.

    Args:
        latent: An H3 joint latent.

    Returns:
        ``scenes``, ``frames``, ``seconds``, ``tokens``, ``audio_steps`` and ``scene_ends``,
        the frame count at the end of each scene.

    Raises:
        ValueError: The latent is not an H3 joint latent.
    """
    video, audio = h3_extend.split(latent)
    frames = h3_extend.frames_for(int(video.shape[2]))
    ends = h3_extend.segment_ends(latent) or [frames]
    return {
        "scenes": len(ends),
        "frames": frames,
        "seconds": round(frames / h3_extend.FPS, 3),
        "tokens": int(video.shape[2]),
        "audio_steps": int(audio.shape[-1]),
        "scene_ends": ends,
    }


def scene_spans(ends: list[int]) -> list[tuple[int, int]]:
    """The first and last frame of each scene.

    Args:
        ends: The frame count at the end of each scene.

    Returns:
        ``(first, last)`` frame indices, from 0.
    """
    spans, start = [], 0
    for end in ends:
        spans.append((start, max(start, int(end) - 1)))
        start = int(end)
    return spans


def scene_lines(ends: list[int]) -> list[str]:
    """One report line per scene.

    Args:
        ends: The frame count at the end of each scene.

    Returns:
        Lines such as ``scene 1: frames 0 to 101, ending at 4.25s``.
    """
    return [
        f"scene {number}: frames {first} to {last}, ending at {(last + 1) / h3_extend.FPS:.2f}s"
        for number, (first, last) in enumerate(scene_spans(ends), 1)
    ]


def restored(latent: dict, scene: int) -> tuple[dict, int]:
    """A clip as it stood when one of its scenes finished, with the records it held then.

    Args:
        latent: An H3 joint latent from H3 Extend Append.
        scene: The scene, from 1, or 0 for the whole clip.

    Returns:
        ``(clip, next_index)``: the clip, and the loop index of the first scene it lacks.

    Raises:
        ValueError: ``scene`` is past the clip's last scene, or the latent is not an H3 joint latent.
    """
    h3_extend.split(latent)
    ends = h3_extend.segment_ends(latent)
    count = len(ends) if ends else 1
    scene = int(scene)
    if scene < 0 or scene > count:
        raise ValueError(
            f"scene {scene} was asked for and the clip holds {count}. Ask for 1 to {count}, "
            f"or 0 for the whole clip"
        )
    if scene in (0, count):
        return dict(latent), count

    out = dict(h3_extend.until_segment(latent, scene - 1))
    tails = latent.get(h3_extend.TAILS_KEY)
    if tails is not None:
        # Keeps the trims of the scenes before the last one kept.
        out[h3_extend.TAILS_KEY] = {
            int(index): kept for index, kept in tails.items() if int(index) < scene - 1
        }
    starts = h3_extend.scene_starts(latent)
    if starts is not None:
        reach = h3_extend.seen_rows(ends[scene - 1])
        out = h3_extend.with_scenes(out, [start for start in starts if start < reach])
    return out, scene


def _described(value, path: list[str], own: dict, taken: dict):
    """One value's JSON layout, its tensors added to ``own`` under names not in ``taken``."""
    import torch

    if isinstance(value, torch.Tensor):
        name = _free_name("/".join(path), own, taken)
        own[name] = _storable(value, own, taken)
        return {TENSOR: name}
    if _is_nested(value):
        return {NESTED: [_described(part, [*path, str(index)], own, taken)
                         for index, part in enumerate(value.tensors)]}
    if isinstance(value, dict):
        pairs = []
        for key, item in value.items():
            if not isinstance(key, PLAIN):
                raise TypeError(f"a {type(key).__name__} key cannot be written")
            pairs.append([key, _described(item, [*path, str(key)], own, taken)])
        return {DICT: pairs}
    if isinstance(value, (tuple, list)):
        tag = TUPLE if isinstance(value, tuple) else LIST
        return {tag: [_described(item, [*path, str(index)], own, taken)
                      for index, item in enumerate(value)]}
    if isinstance(value, PLAIN):
        return value
    raise TypeError(f"a {type(value).__name__} cannot be written")


def _rebuilt(layout, tensors: dict):
    """The value one JSON layout entry describes."""
    if not isinstance(layout, dict):
        return layout
    if len(layout) != 1:
        raise ValueError("the clip file's layout is damaged. Save the clip again")
    tag, body = next(iter(layout.items()))
    if tag == TENSOR:
        if body not in tensors:
            raise ValueError(f"the clip file is missing its `{body}` tensor. Save the clip again")
        return tensors[body]
    if tag == NESTED:
        import comfy.nested_tensor

        return comfy.nested_tensor.NestedTensor([_rebuilt(part, tensors) for part in body])
    if tag == DICT:
        return {key: _rebuilt(item, tensors) for key, item in body}
    if tag == TUPLE:
        return tuple(_rebuilt(item, tensors) for item in body)
    if tag == LIST:
        return [_rebuilt(item, tensors) for item in body]
    raise ValueError(f"the clip file holds a `{tag}` entry this install cannot read")


def _is_nested(value) -> bool:
    """Whether a value is ComfyUI's pair of video and audio tensors."""
    return type(value).__name__ == "NestedTensor" and isinstance(getattr(value, "tensors", None), list)


def _free_name(wanted: str, own: dict, taken: dict) -> str:
    """``wanted``, or ``wanted#2`` onward where a tensor already carries it."""
    name, number = wanted or "tensor", 1
    while name in own or name in taken:
        number += 1
        name = f"{wanted}#{number}"
    return name


def _storable(tensor, own: dict, taken: dict):
    """A contiguous CPU tensor owning its memory apart from every tensor already named."""
    held = tensor.detach().to("cpu").contiguous()
    if held.numel() == 0:
        return held.clone()
    pointer = held.untyped_storage().data_ptr()
    for other in (*own.values(), *taken.values()):
        if other.numel() and other.untyped_storage().data_ptr() == pointer:
            return held.clone()
    return held


def _metadata(data: bytes, path: str) -> dict:
    """The ``__metadata__`` block of a safetensors file read into memory."""
    header = None
    if len(data) >= 8:
        try:
            header = json.loads(data[8:8 + int.from_bytes(data[:8], "little")])
        except (ValueError, UnicodeDecodeError):
            header = None
    if not isinstance(header, dict):
        raise ValueError(
            f"{os.path.basename(path)} is not a safetensors file. Pick a {EXTENSION} file "
            f"H3 Save Clip saved"
        )
    return header.get("__metadata__") or {}
