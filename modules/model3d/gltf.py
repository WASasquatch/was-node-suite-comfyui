"""glTF 2.0 reading and writing for `scene.Model3D`, as ``.glb`` or ``.gltf`` with its files.

Static parts are baked into model space; skinned parts keep their bind-space vertices. Meshopt
compressed buffer views are decoded in place.
"""

from __future__ import annotations

__all__ = [
    "GltfBuild", "GltfDocument", "GltfError", "MAX_IMAGES", "MAX_TRIANGLES", "MAX_VERTICES",
    "NodeMap", "build", "parse", "read", "to_model", "write",
]

import base64
import binascii
import functools
import hashlib
import json
import math
import struct
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from ..log import get_logger
from . import textures
from .scene import (
    Animation, Channel, Material, Model3D, MorphTarget, Part, Skeleton, Texture,
)

logger = get_logger("model3d.gltf")

#: Most vertices one model may hold across every part.
MAX_VERTICES = 60_000_000

#: Most triangles one model may hold across every part.
MAX_TRIANGLES = 120_000_000

#: Most images one file may list.
MAX_IMAGES = 512

#: Largest JSON document read, in bytes.
MAX_JSON_BYTES = 512 * 1024 * 1024

#: Most entries any top-level JSON array may hold.
MAX_ITEMS = 2_000_000

#: Most elements one accessor may hold.
MAX_ACCESSOR_COUNT = 360_000_000

#: Largest buffer view a compressed view may decode to, in bytes.
MAX_DECODED_VIEW = 1024 * 1024 * 1024

#: Most bytes all compressed views of one file may decode to.
MAX_DECODED_TOTAL = 4 * 1024 * 1024 * 1024

#: Most bytes all files one model names beside itself may hold together.
MAX_EXTERNAL_BYTES = 2 * 1024 * 1024 * 1024

#: Most texture pixels one model may decode across every image.
MAX_TEXTURE_TOTAL_PIXELS = 4 * 8192 * 8192

#: Most shape key vertices one model may hold across every part and target.
MAX_MORPH_VERTICES = 100_000_000

#: Most keyframe numbers one model may hold across every animation channel.
MAX_ANIMATION_VALUES = 200_000_000

#: Most parts one model may hold.
MAX_PARTS = 100_000

#: Most instances one GPU-instanced node may place.
MAX_INSTANCES = 65_536

#: Most influences kept per vertex.
MAX_INFLUENCES = 8

_GLB_MAGIC = b"glTF"
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942

_DTYPES = {
    5120: np.dtype("<i1"), 5121: np.dtype("<u1"), 5122: np.dtype("<i2"),
    5123: np.dtype("<u2"), 5125: np.dtype("<u4"), 5126: np.dtype("<f4"),
}
_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
_MATRIX_ROWS = {"MAT2": 2, "MAT3": 3, "MAT4": 4}
_NORMALIZE = {5120: 127.0, 5121: 255.0, 5122: 32767.0, 5123: 65535.0, 5125: 4294967295.0}

_WRAP_READ = {33071: "clamp", 33648: "mirror", 10497: "repeat"}
_WRAP_WRITE = {"clamp": 33071, "mirror": 33648, "repeat": 10497}

_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963

_MESHOPT = ("EXT_meshopt_compression", "KHR_meshopt_compression")

#: Extensions read or written, beyond core glTF.
_KNOWN_EXTENSIONS = {
    "KHR_mesh_quantization", "KHR_texture_transform", "KHR_materials_unlit",
    "KHR_materials_emissive_strength", "EXT_texture_webp", "KHR_texture_basisu",
    "EXT_meshopt_compression", "KHR_meshopt_compression", "EXT_mesh_gpu_instancing",
    "VRM", "VRMC_vrm", "VRMC_springBone", "VRMC_materials_mtoon", "VRMC_node_constraint",
    "KHR_materials_variants", "KHR_lights_punctual",
}

_IDENTITY = np.eye(4)


class GltfError(ValueError):
    """A glTF file that cannot be read, or a model that cannot be written as one."""


@dataclass
class NodeMap:
    """Where each glTF node, mesh and material landed in the model.

    Attributes:
        joint_of_node: glTF node index against the skeleton joint it became.
        parts_of_mesh: glTF mesh index against the parts made from it.
        parts_of_node: glTF node index against the parts it placed.
        material_of_part: glTF material index of each part, ``-1`` for the default material.
    """

    joint_of_node: dict[int, int] = field(default_factory=dict)
    parts_of_mesh: dict[int, list[int]] = field(default_factory=dict)
    parts_of_node: dict[int, list[int]] = field(default_factory=dict)
    material_of_part: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------------------------
# Matrix helpers
# ---------------------------------------------------------------------------------------------


def _quat_matrix(quat) -> np.ndarray:
    """A 3x3 rotation matrix from an ``xyzw`` quaternion.

    Args:
        quat: Four numbers, normalised here.

    Returns:
        ``(3, 3)`` float64.
    """
    x, y, z, w = (float(value) for value in quat)
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm < 1e-12:
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _matrix_quat(rotation: np.ndarray) -> np.ndarray:
    """An ``xyzw`` quaternion from a 3x3 rotation matrix.

    Args:
        rotation: ``(3, 3)`` orthonormal matrix with determinant 1.

    Returns:
        ``(4,)`` float64 unit quaternion.
    """
    m = rotation
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        quat = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-12)) * 2
        quat = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-12)) * 2
        quat = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = math.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-12)) * 2
        quat = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    quat = np.array(quat, dtype=np.float64)
    quat /= max(np.linalg.norm(quat), 1e-12)
    return quat


def _trs_matrix(translation, rotation, scale) -> np.ndarray:
    """A 4x4 matrix composed as translation, then rotation, then scale.

    Args:
        translation: ``(3,)``.
        rotation: ``(4,)`` ``xyzw`` quaternion.
        scale: ``(3,)``.

    Returns:
        ``(4, 4)`` float64.
    """
    matrix = np.eye(4)
    matrix[:3, :3] = _quat_matrix(rotation) * np.asarray(scale, dtype=np.float64)[None, :]
    matrix[:3, 3] = translation
    return matrix


def _decompose(matrix: np.ndarray, tolerance: float = 1e-4):
    """Translation, rotation and scale of an affine matrix with no shear.

    Args:
        matrix: ``(4, 4)``.
        tolerance: Largest deviation from orthonormal accepted after scale is removed.

    Returns:
        ``(translation, quaternion, scale)`` as float64 arrays, or None when the matrix is not
        affine or carries shear.
    """
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
        return None
    linear = matrix[:3, :3].copy()
    scale = np.linalg.norm(linear, axis=0)
    if np.any(scale < 1e-12):
        return None
    if np.linalg.det(linear) < 0:
        scale[0] = -scale[0]
    rotation = linear / scale[None, :]
    if np.abs(rotation.T @ rotation - np.eye(3)).max() > tolerance:
        return None
    return matrix[:3, 3].copy(), _matrix_quat(rotation), scale


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton products of ``xyzw`` quaternions, broadcast over rows.

    Args:
        left: ``(..., 4)``.
        right: ``(..., 4)``.

    Returns:
        ``(..., 4)`` products ``left * right``.
    """
    x1, y1, z1, w1 = np.moveaxis(left, -1, 0)
    x2, y2, z2, w2 = np.moveaxis(right, -1, 0)
    return np.stack([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], axis=-1)


def _damaged(function):
    """Wrap a reader so malformed JSON values raise `GltfError` instead of a bare exception."""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except GltfError:
            raise
        except (TypeError, KeyError, IndexError, AttributeError, OverflowError, ValueError,
                RecursionError, struct.error, np.linalg.LinAlgError) as error:
            raise GltfError(f"the glTF file holds a value that cannot be read "
                            f"({error.__class__.__name__}: {str(error)[:120]}). The file is damaged; "
                            f"re-export it.") from error

    return wrapper


def _is_identity(matrix: np.ndarray | None, tolerance: float = 1e-7) -> bool:
    return matrix is None or bool(np.abs(matrix - _IDENTITY).max() <= tolerance)


def _finite_list(value, length: int):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return None
    try:
        numbers = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return numbers if all(math.isfinite(item) for item in numbers) else None


# ---------------------------------------------------------------------------------------------
# Meshopt decoding
# ---------------------------------------------------------------------------------------------


def _popcount(value: int) -> int:
    return bin(value).count("1")


def _unzigzag(values: np.ndarray) -> np.ndarray:
    one = values.dtype.type(1)
    return (values >> one) ^ (np.zeros_like(values) - (values & one))


def _meshopt_attributes(source: bytes, count: int, stride: int) -> bytes:
    """Decode a meshopt attribute stream.

    Args:
        source: The compressed bytes.
        count: Number of elements.
        stride: Bytes per element, a multiple of 4 up to 256.

    Returns:
        ``count * stride`` decoded bytes.

    Raises:
        GltfError: The stream is malformed or truncated.
    """
    if stride <= 0 or stride > 256 or stride % 4:
        raise GltfError(f"a meshopt attribute stream has a stride of {stride}, which is not a "
                        f"multiple of 4 up to 256. Re-export the file with gltfpack.")
    size = len(source)
    if size < 1 or (source[0] & 0xF0) != 0xA0:
        raise GltfError("a meshopt attribute stream has no valid header. Re-export the file.")
    version = source[0] & 0x0F
    if version > 1:
        raise GltfError(f"a meshopt attribute stream uses codec version {version}, which this "
                        f"reader does not decode. Re-export with an older gltfpack.")
    if count == 0:
        return b""
    tail_size = stride + (0 if version == 0 else stride // 4)
    tail_pad = max(tail_size, 32 if version == 0 else 24)
    if size - 1 < tail_pad:
        raise GltfError("a meshopt attribute stream is truncated. Re-export the file.")
    data = np.frombuffer(source, dtype=np.uint8)
    last = data[size - tail_size: size - tail_size + stride].copy()
    channels = data[size - tail_size + stride: size] if version else None
    block_limit = min((8192 // stride) & ~15, 256)
    end = size

    blocks = []
    offset = 0
    stream_total = 0
    while offset < count:
        block_count = min(block_limit, count - offset)
        aligned = (block_count + 15) & ~15
        blocks.append((block_count, aligned, stream_total))
        stream_total += aligned * stride
        offset += block_count
    streams = np.zeros(stream_total, dtype=np.uint8)
    groups = {1: ([], []), 2: ([], []), 4: ([], []), 8: ([], [])}
    bit_tables = ((0, 2, 4, 8),) if version == 0 else ((0, 1, 2, 4), (1, 2, 4, 8))

    position = 1
    for block_count, aligned, base in blocks:
        group_count = aligned // 16
        header_size = (group_count + 3) // 4
        control = b""
        if version:
            if end - position < stride // 4:
                raise GltfError("a meshopt attribute stream is truncated. Re-export the file.")
            control = source[position:position + stride // 4]
            position += stride // 4
        for byte in range(stride):
            mode = (control[byte >> 2] >> ((byte & 3) * 2)) & 3 if version else 0
            stream_base = base + byte * aligned
            if mode == 3:
                if end - position < block_count:
                    raise GltfError("a meshopt attribute stream is truncated. Re-export the file.")
                streams[stream_base:stream_base + block_count] = data[position:position + block_count]
                position += block_count
                continue
            if mode == 2:
                continue
            bits_for = bit_tables[mode]
            if end - position < header_size:
                raise GltfError("a meshopt attribute stream is truncated. Re-export the file.")
            header = source[position:position + header_size]
            position += header_size
            for group in range(group_count):
                if end - position < 24:
                    raise GltfError("a meshopt attribute stream is truncated. Re-export the file.")
                bits = bits_for[(header[group >> 2] >> ((group & 3) * 2)) & 3]
                if bits == 0:
                    continue
                outs, starts = groups[bits]
                outs.append(stream_base + group * 16)
                starts.append(position)
                if bits == 8:
                    position += 16
                elif bits == 2:
                    word = int.from_bytes(source[position:position + 4], "big")
                    position += 4 + _popcount(word & (word >> 1) & 0x55555555)
                elif bits == 4:
                    word = int.from_bytes(source[position:position + 8], "big")
                    nibbles = word & (word >> 1) & (word >> 2) & (word >> 3) & 0x1111111111111111
                    position += 8 + _popcount(nibbles)
                else:
                    position += 2 + _popcount(source[position] | (source[position + 1] << 8))
    if end - position != tail_pad:
        raise GltfError("a meshopt attribute stream has trailing or missing bytes. Re-export "
                        "the file.")

    lanes = np.arange(16)
    for bits, (outs, starts) in groups.items():
        if not outs:
            continue
        out = np.asarray(outs, dtype=np.int64)[:, None] + lanes
        start = np.asarray(starts, dtype=np.int64)[:, None]
        if bits == 8:
            streams[out] = data[start + lanes]
            continue
        packed_bytes = 16 * bits // 8
        packed = data[start + np.arange(packed_bytes)].astype(np.uint8)
        if bits == 1:
            values = (packed[:, :, None] >> np.arange(8, dtype=np.uint8)) & 1
        elif bits == 2:
            values = (packed[:, :, None] >> np.array([6, 4, 2, 0], dtype=np.uint8)) & 3
        else:
            values = (packed[:, :, None] >> np.array([4, 0], dtype=np.uint8)) & 15
        values = values.reshape(len(outs), 16)
        sentinel = values == (1 << bits) - 1
        rank = np.cumsum(sentinel, axis=1) - sentinel
        extra_index = np.minimum(start + packed_bytes + rank, size - 1)
        values = np.where(sentinel, data[extra_index], values)
        streams[out] = values.astype(np.uint8)

    deltas = np.concatenate([
        streams[base:base + stride * aligned].reshape(stride, aligned)[:, :block_count].T
        for block_count, aligned, base in blocks
    ], axis=0)
    result = np.empty((count, stride), dtype=np.uint8)
    for lane in range(0, stride, 4):
        channel = int(channels[lane // 4]) if version else 0
        kind = channel & 3
        piece = np.ascontiguousarray(deltas[:, lane:lane + 4])
        seed = last[lane:lane + 4]
        if kind == 0:
            values = np.cumsum(_unzigzag(piece), axis=0, dtype=np.uint8) + seed
        elif kind == 1:
            wide = piece.view("<u2")
            values = (np.cumsum(_unzigzag(wide), axis=0, dtype=np.uint16) + seed.view("<u2"))
            values = values.astype("<u2").view(np.uint8)
        elif kind == 2:
            wide = piece.view("<u4").astype(np.uint64)
            turn = (32 - (channel >> 4)) & 31
            turned = ((wide << np.uint64(turn)) | (wide >> np.uint64((32 - turn) & 31))) & np.uint64(0xFFFFFFFF)
            values = np.bitwise_xor.accumulate(turned.astype(np.uint32), axis=0) ^ seed.view("<u4")
            values = values.astype("<u4").view(np.uint8)
        else:
            raise GltfError("a meshopt attribute stream names an unknown channel mode. Re-export "
                            "the file.")
        result[:, lane:lane + 4] = values.reshape(count, 4)
    return result.tobytes()


def _vbyte(source: bytes, position: int) -> tuple[int, int]:
    lead = source[position]
    position += 1
    if lead < 128:
        return lead, position
    result = lead & 127
    shift = 7
    for _ in range(4):
        group = source[position]
        position += 1
        result |= (group & 127) << shift
        shift += 7
        if group < 128:
            break
    return result, position


def _meshopt_triangles(source: bytes, count: int) -> np.ndarray:
    """Decode a meshopt triangle index stream.

    Args:
        source: The compressed bytes.
        count: Number of indices, a multiple of 3.

    Returns:
        ``(count,)`` uint32 indices.

    Raises:
        GltfError: The stream is malformed or truncated.
    """
    size = len(source)
    triangles = count // 3
    if count % 3 or size < 1 + triangles + 16 or (source[0] & 0xF0) != 0xE0:
        raise GltfError("a meshopt triangle stream is malformed. Re-export the file.")
    version = source[0] & 0x0F
    if version > 1:
        raise GltfError(f"a meshopt triangle stream uses codec version {version}, which this "
                        f"reader does not decode. Re-export with an older gltfpack.")
    mask = 0xFFFFFFFF
    edge_a = [mask] * 16
    edge_b = [mask] * 16
    fifo = [mask] * 16
    edge_at = vertex_at = 0
    following = last = 0
    fec_max = 13 if version >= 1 else 15
    data = 1 + triangles
    safe_end = size - 16
    table = source[safe_end:size]
    out = [0] * count
    write = 0
    try:
        for code_index in range(1, 1 + triangles):
            code = source[code_index]
            if code < 0xF0:
                slot = (edge_at - 1 - (code >> 4)) & 15
                a = edge_a[slot]
                b = edge_b[slot]
                fec = code & 15
                if fec < fec_max:
                    if fec == 0:
                        c = following
                        following += 1
                        fifo[vertex_at] = c
                        vertex_at = (vertex_at + 1) & 15
                    else:
                        c = fifo[(vertex_at - 1 - fec) & 15]
                        fifo[vertex_at] = c
                else:
                    if data > safe_end:
                        raise GltfError("a meshopt triangle stream is truncated. Re-export the file.")
                    if fec != 15:
                        c = (last + fec * 2 - 27) & mask
                    else:
                        value, data = _vbyte(source, data)
                        c = (last + ((value >> 1) ^ -(value & 1))) & mask
                    last = c
                    fifo[vertex_at] = c
                    vertex_at = (vertex_at + 1) & 15
                edge_a[edge_at] = c
                edge_b[edge_at] = b
                edge_at = (edge_at + 1) & 15
                edge_a[edge_at] = a
                edge_b[edge_at] = c
                edge_at = (edge_at + 1) & 15
            else:
                if code < 0xFE:
                    aux = table[code & 15]
                    feb = aux >> 4
                    fec = aux & 15
                    a = following
                    following += 1
                    b_fifo = fifo[(vertex_at - feb) & 15]
                    b = following if feb == 0 else b_fifo
                    following += feb == 0
                    c_fifo = fifo[(vertex_at - fec) & 15]
                    c = following if fec == 0 else c_fifo
                    following += fec == 0
                    push_b = feb == 0
                    push_c = fec == 0
                else:
                    if data > safe_end:
                        raise GltfError("a meshopt triangle stream is truncated. Re-export the file.")
                    aux = source[data]
                    data += 1
                    fea = 0 if code == 0xFE else 15
                    feb = aux >> 4
                    fec = aux & 15
                    if aux == 0:
                        following = 0
                    if fea == 0:
                        a = following
                        following += 1
                    else:
                        a = 0
                    if feb == 0:
                        b = following
                        following += 1
                    else:
                        b = fifo[(vertex_at - feb) & 15]
                    if fec == 0:
                        c = following
                        following += 1
                    else:
                        c = fifo[(vertex_at - fec) & 15]
                    if fea == 15:
                        value, data = _vbyte(source, data)
                        last = a = (last + ((value >> 1) ^ -(value & 1))) & mask
                    if feb == 15:
                        value, data = _vbyte(source, data)
                        last = b = (last + ((value >> 1) ^ -(value & 1))) & mask
                    if fec == 15:
                        value, data = _vbyte(source, data)
                        last = c = (last + ((value >> 1) ^ -(value & 1))) & mask
                    push_b = feb == 0 or feb == 15
                    push_c = fec == 0 or fec == 15
                fifo[vertex_at] = a
                vertex_at = (vertex_at + 1) & 15
                fifo[vertex_at] = b
                vertex_at = (vertex_at + push_b) & 15
                fifo[vertex_at] = c
                vertex_at = (vertex_at + push_c) & 15
                for first, second in ((b, a), (c, b), (a, c)):
                    edge_a[edge_at] = first
                    edge_b[edge_at] = second
                    edge_at = (edge_at + 1) & 15
            out[write] = a & mask
            out[write + 1] = b & mask
            out[write + 2] = c & mask
            write += 3
    except IndexError as error:
        raise GltfError("a meshopt triangle stream is truncated. Re-export the file.") from error
    if data != safe_end:
        raise GltfError("a meshopt triangle stream has trailing or missing bytes. Re-export the file.")
    return np.asarray(out, dtype=np.uint32)


def _meshopt_indices(source: bytes, count: int) -> np.ndarray:
    """Decode a meshopt index sequence stream.

    Args:
        source: The compressed bytes.
        count: Number of indices.

    Returns:
        ``(count,)`` uint32 indices.

    Raises:
        GltfError: The stream is malformed or truncated.
    """
    size = len(source)
    if size < 1 + count + 4 or (source[0] & 0xF0) != 0xD0 or (source[0] & 0x0F) > 1:
        raise GltfError("a meshopt index stream is malformed. Re-export the file.")
    if count == 0:
        return np.zeros(0, dtype=np.uint32)
    body = np.frombuffer(source, dtype=np.uint8)[1:size - 4].astype(np.uint64)
    ends = np.flatnonzero(body < 128)
    if ends.size < count or int(ends[count - 1]) + 1 != body.size:
        raise GltfError("a meshopt index stream has trailing or missing bytes. Re-export the file.")
    owner = np.concatenate([[0], np.cumsum(body < 128)[:-1]]).astype(np.int64)
    starts = np.concatenate([[0], ends[:-1] + 1])
    shift = (np.arange(body.size) - starts[owner]) * 7
    if shift.max(initial=0) > 28:
        raise GltfError("a meshopt index stream holds an over-long value. Re-export the file.")
    pieces = (body & np.uint64(127)) << shift.astype(np.uint64)
    values = np.zeros(count, dtype=np.uint64)
    np.add.at(values, owner, pieces)
    values &= np.uint64(0xFFFFFFFF)
    baseline = (values & np.uint64(1)).astype(bool)
    values >>= np.uint64(1)
    delta = ((values >> np.uint64(1)) ^ (np.uint64(0) - (values & np.uint64(1)))) & np.uint64(0xFFFFFFFF)
    result = np.zeros(count, dtype=np.uint64)
    for which in (False, True):
        chosen = baseline == which
        result[chosen] = np.cumsum(delta[chosen]) & np.uint64(0xFFFFFFFF)
    return result.astype(np.uint32)


def _meshopt_filter(decoded: bytes, count: int, stride: int, name: str) -> bytes:
    """Apply a meshopt decode filter to decoded attribute bytes.

    Args:
        decoded: ``count * stride`` bytes.
        count: Number of elements.
        stride: Bytes per element.
        name: ``"NONE"``, ``"OCTAHEDRAL"``, ``"QUATERNION"``, ``"EXPONENTIAL"`` or ``"COLOR"``.

    Returns:
        The filtered bytes.

    Raises:
        GltfError: The filter is unknown or does not fit the stride.
    """
    if name in ("", "NONE") or count == 0:
        return decoded
    if name == "OCTAHEDRAL" and stride in (4, 8):
        kind = np.int8 if stride == 4 else np.int16
        top = float((1 << (8 * (stride // 4) - 1)) - 1)
        values = np.frombuffer(decoded, dtype=kind).reshape(count, 4).copy()
        x = values[:, 0].astype(np.float32)
        y = values[:, 1].astype(np.float32)
        z = values[:, 2].astype(np.float32) - np.abs(x) - np.abs(y)
        t = np.minimum(z, 0.0)
        x = x + np.where(x >= 0, t, -t)
        y = y + np.where(y >= 0, t, -t)
        length = np.sqrt(x * x + y * y + z * z)
        scale = np.float32(top) / np.maximum(length, np.float32(1e-30))
        for column, component in enumerate((x, y, z)):
            scaled = component * scale
            values[:, column] = np.trunc(scaled + np.where(scaled >= 0, 0.5, -0.5)).astype(kind)
        return values.tobytes()
    if name == "QUATERNION" and stride == 8:
        values = np.frombuffer(decoded, dtype=np.int16).reshape(count, 4).copy()
        sf = (values[:, 3].astype(np.int32) | 3).astype(np.float32)
        x, y, z = (values[:, index].astype(np.float32) for index in range(3))
        ww = sf * sf * 2.0 - x * x - y * y - z * z
        w = np.sqrt(np.maximum(ww, 0.0))
        ss = np.float32(32767.0 / math.sqrt(2.0)) / sf

        def rounded(component):
            scaled = component * ss
            return np.trunc(scaled + np.where(scaled >= 0, 0.5, -0.5)).astype(np.int16)

        order = values[:, 3].astype(np.int32) & 3
        rows = np.arange(count)
        out = np.empty_like(values)
        out[rows, (order + 1) & 3] = rounded(x)
        out[rows, (order + 2) & 3] = rounded(y)
        out[rows, (order + 3) & 3] = rounded(z)
        out[rows, order] = np.trunc(w * ss + 0.5).astype(np.int16)
        return out.tobytes()
    if name == "EXPONENTIAL" and stride % 4 == 0:
        raw = np.frombuffer(decoded, dtype="<u4").astype(np.int64)
        mantissa = ((raw << 8) & 0xFFFFFFFF).astype(np.uint32).view(np.int32) >> 8
        exponent = raw.astype(np.uint32).view(np.int32) >> 24
        values = np.ldexp(mantissa.astype(np.float32), exponent.astype(np.int32)).astype("<f4")
        return values.tobytes()
    if name == "COLOR" and stride in (4, 8):
        kind, signed = (np.uint8, np.int8) if stride == 4 else (np.uint16, np.int16)
        top = float((1 << (8 * (stride // 4))) - 1)
        values = np.frombuffer(decoded, dtype=kind).reshape(count, 4)
        alpha = values[:, 3].astype(np.int32)
        spread = alpha.copy()
        for step in (1, 2, 4, 8):
            spread |= spread >> step
        luma = values[:, 0].astype(np.int32)
        co = values[:, 1].view(signed).astype(np.int32)
        cg = values[:, 2].view(signed).astype(np.int32)
        alpha = ((alpha << 1) & spread) | (alpha & 1)
        scale = np.float32(top) / np.maximum(spread, 1).astype(np.float32)
        channels = (luma + co - cg, luma + cg, luma - co - cg, alpha)
        out = np.stack([np.trunc(c.astype(np.float32) * scale + 0.5) for c in channels], axis=1)
        return out.astype(np.int64).astype(kind).tobytes()
    raise GltfError(f"a meshopt buffer view uses the {name} filter with a stride of {stride}, "
                    f"which this reader does not decode. Re-export the file with gltfpack.")


# ---------------------------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------------------------


class GltfDocument:
    """A parsed glTF file: its JSON, resolved buffers, and cached views, accessors and textures.

    Attributes:
        json: The glTF JSON document.
        buffers: Bytes of each buffer, None where a buffer has no data.
        base_dir: Folder external files were read from, or None.
        binary: Whether the file was a ``.glb``.
    """

    def __init__(self, document: dict, buffers: list, base_dir: str | None, binary: bool):
        self.json = document
        self.buffers = buffers
        self.base_dir = base_dir
        self.binary = binary
        self._views: dict[int, tuple[memoryview, int | None]] = {}
        self._images: dict[int, Any] = {}
        self._decoded: dict[Any, Texture | None] = {}
        self._textures: dict[tuple[int, int], Texture | None] = {}
        self._warned: set[str] = set()
        self._sidecars: dict[str, bytes | None] = {}
        self._counted: set[str] = set()
        self._external_bytes = 0
        self._decoded_bytes = 0
        self._texture_pixels = 0

    def sidecar(self, name: str, keep: bool = True) -> bytes | None:
        """The bytes of a file the model names beside itself, read once per name.

        Args:
            name: The URI-decoded reference.
            keep: Hold the bytes for later calls with the same name.

        Returns:
            The bytes, or None when the file cannot be read or the model's files exceed
            `MAX_EXTERNAL_BYTES` together, with a warning logged.
        """
        if name in self._sidecars:
            return self._sidecars[name]
        found = textures.read_sidecar(self.base_dir, name)
        if found is not None and name not in self._counted:
            self._counted.add(name)
            self._external_bytes += len(found)
            if self._external_bytes > MAX_EXTERNAL_BYTES:
                self.warn_once("external", "the files this model names beside itself hold more than "
                               "%d MB together; the rest were skipped. Use a .glb instead.",
                               MAX_EXTERNAL_BYTES // (1024 * 1024))
                found = None
        if keep or found is None:
            self._sidecars[name] = found
        return found

    def _data_bytes(self) -> int:
        return sum(len(buffer) for buffer in self.buffers if buffer is not None)

    @property
    def extensions_used(self) -> set[str]:
        """Names listed in ``extensionsUsed``."""
        return {str(name) for name in self.json.get("extensionsUsed", []) or []}

    def warn_once(self, key: str, message: str, *args) -> None:
        """Log a warning the first time ``key`` is seen.

        Args:
            key: Identity of the warning.
            message: Log format string.
            *args: Format arguments.
        """
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(message, *args)

    def item(self, kind: str, index) -> dict:
        """One entry of a top-level array, such as an accessor.

        Args:
            kind: Array name, such as ``"accessors"``.
            index: Entry index.

        Returns:
            The entry.

        Raises:
            GltfError: The index is not an integer naming an entry.
        """
        entries = self.json.get(kind)
        if (not isinstance(index, int) or isinstance(index, bool) or not isinstance(entries, list)
                or not 0 <= index < len(entries) or not isinstance(entries[index], dict)):
            raise GltfError(f"the file refers to {kind} {index!r}, which it does not hold. The "
                            f"file is damaged; re-export it.")
        return entries[index]

    @_damaged
    def view(self, index: int) -> tuple[memoryview, int | None]:
        """The bytes of a buffer view, decompressed when it is meshopt compressed.

        Args:
            index: Buffer view index.

        Returns:
            ``(bytes, byte_stride)``; the stride is None when the view declares none.

        Raises:
            GltfError: The view lies outside its buffer, or its buffer has no data.
        """
        if index in self._views:
            return self._views[index]
        entry = self.item("bufferViews", index)
        stride = entry.get("byteStride")
        if stride is not None and (not isinstance(stride, int) or not 4 <= stride <= 252):
            raise GltfError(f"buffer view {index} has a byte stride of {stride!r}, outside 4 to "
                            f"252. The file is damaged; re-export it.")
        compressed = None
        extensions = entry.get("extensions") or {}
        for name in _MESHOPT:
            if isinstance(extensions.get(name), dict):
                compressed = extensions[name]
                break
        if compressed is not None:
            result = (memoryview(self._decode_meshopt(index, compressed)), stride)
        else:
            length = int(entry.get("byteLength", 0))
            offset = int(entry.get("byteOffset", 0))
            data = self._buffer(entry.get("buffer"), f"buffer view {index}")
            if offset < 0 or length < 0 or offset + length > len(data):
                raise GltfError(f"buffer view {index} runs past the end of its buffer. The file is "
                                f"truncated; re-export it.")
            result = (data[offset:offset + length], stride)
        self._views[index] = result
        return result

    def _buffer(self, index, owner: str) -> memoryview:
        buffers = self.buffers
        if not isinstance(index, int) or not 0 <= index < len(buffers):
            raise GltfError(f"{owner} names buffer {index!r}, which the file does not hold. The "
                            f"file is damaged; re-export it.")
        data = buffers[index]
        if data is None:
            uri = (self.json.get("buffers") or [{}])[index].get("uri", "")
            label = _unquote(str(uri)).replace("\\", "/").rsplit("/", 1)[-1] if uri else ""
            if label:
                raise GltfError(f"{owner} needs the file {label}, which could not be read beside "
                                f"the model. Keep the .bin file next to the .gltf, or use a .glb.")
            raise GltfError(f"{owner} needs buffer {index}, which holds no data. Re-export the file "
                            f"as a .glb.")
        return data

    def _decode_meshopt(self, index: int, spec: dict) -> bytes:
        try:
            count = int(spec["count"])
            stride = int(spec["byteStride"])
            length = int(spec["byteLength"])
            offset = int(spec.get("byteOffset", 0))
        except (KeyError, TypeError, ValueError) as error:
            raise GltfError(f"buffer view {index} has an incomplete meshopt description. Re-export "
                            f"the file.") from error
        mode = str(spec.get("mode", ""))
        if count < 0 or stride <= 0 or count * stride > MAX_DECODED_VIEW:
            raise GltfError(f"buffer view {index} claims {count} elements of {stride} bytes, more "
                            f"than one view may decode to. The file is damaged or too large.")
        data = self._buffer(spec.get("buffer"), f"buffer view {index}")
        if offset < 0 or length < 0 or offset + length > len(data):
            raise GltfError(f"buffer view {index} runs past the end of its buffer. The file is "
                            f"truncated; re-export it.")
        self._decoded_bytes += count * stride
        if self._decoded_bytes > MAX_DECODED_TOTAL:
            raise GltfError(f"the file's compressed buffer views decode to more than "
                            f"{MAX_DECODED_TOTAL // (1024 * 1024)} MB together. Decimate the model "
                            f"before exporting it.")
        source = bytes(data[offset:offset + length])
        if mode == "ATTRIBUTES":
            decoded = _meshopt_attributes(source, count, stride)
            return _meshopt_filter(decoded, count, stride, str(spec.get("filter", "NONE")))
        if mode in ("TRIANGLES", "INDICES") and stride in (2, 4):
            if mode == "TRIANGLES":
                indices = _meshopt_triangles(source, count)
            else:
                indices = _meshopt_indices(source, count)
            return indices.astype("<u2" if stride == 2 else "<u4").tobytes()
        raise GltfError(f"buffer view {index} uses meshopt mode {mode!r} with a stride of {stride}, "
                        f"which is not valid. Re-export the file.")

    @_damaged
    def accessor(self, index: int, normalize: bool | None = None) -> np.ndarray:
        """The elements of an accessor, sparse substitutions applied.

        Args:
            index: Accessor index.
            normalize: True scales integer components to ``[0, 1]`` or ``[-1, 1]``, False keeps
                raw integers, None follows the accessor's ``normalized`` flag.

        Returns:
            ``(count,)`` for scalars, ``(count, n)`` for vectors, ``(count, rows, cols)``
            row-major for matrices. Float32 when scaled or stored as float.

        Raises:
            GltfError: The accessor is malformed, too large or reaches past its data.
        """
        entry = self.item("accessors", index)
        component = entry.get("componentType")
        kind = entry.get("type")
        count = entry.get("count")
        if component not in _DTYPES or kind not in _COMPONENTS or not isinstance(count, int):
            raise GltfError(f"accessor {index} has an unknown layout. The file is damaged; "
                            f"re-export it.")
        if not 0 <= count <= MAX_ACCESSOR_COUNT:
            raise GltfError(f"accessor {index} claims {count} elements, more than one model may "
                            f"hold. Decimate the model before exporting it.")
        dtype = _DTYPES[component]
        width = dtype.itemsize
        rows = _MATRIX_ROWS.get(kind, 0)
        column_bytes = rows * width
        padded = rows in (2, 3) and column_bytes % 4 != 0
        element = (rows * ((column_bytes + 3) & ~3)) if padded else _COMPONENTS[kind] * width
        if "bufferView" in entry:
            raw = self._elements(entry["bufferView"], int(entry.get("byteOffset", 0)), count,
                                 element, use_stride=True, owner=f"accessor {index}")
        else:
            if count * element > max(1 << 24, 8 * self._data_bytes()):
                raise GltfError(f"accessor {index} claims {count} elements with no data behind "
                                f"them. The file is damaged; re-export it.")
            raw = np.zeros((count, element), dtype=np.uint8)
        values = self._unpack(raw, dtype, kind, padded, rows, width)
        sparse = entry.get("sparse")
        if isinstance(sparse, dict):
            values = self._apply_sparse(index, sparse, values, dtype, kind, padded, rows, width, element, count)
        scale = entry.get("normalized", False) if normalize is None else normalize
        if scale and dtype.kind in "iu":
            values = values.astype(np.float32) / np.float32(_NORMALIZE[component])
            if dtype.kind == "i":
                values = np.maximum(values, np.float32(-1.0))
        if rows:
            values = values.reshape(count, rows, rows).transpose(0, 2, 1).copy()
        elif _COMPONENTS[kind] == 1:
            values = values.reshape(count)
        else:
            values = values.reshape(count, _COMPONENTS[kind])
        return values

    def _elements(self, view_index, offset: int, count: int, element: int, use_stride: bool,
                  owner: str) -> np.ndarray:
        data, stride = self.view(view_index)
        step = (stride if use_stride and stride else element)
        if offset < 0:
            raise GltfError(f"{owner} has a negative offset. The file is damaged; re-export it.")
        if step < element:
            raise GltfError(f"{owner} has elements of {element} bytes in a buffer view whose byte "
                            f"stride is {step}, so they overlap. The file is damaged; re-export it.")
        if count == 0:
            return np.zeros((0, element), dtype=np.uint8)
        need = offset + step * (count - 1) + element
        if need > len(data):
            raise GltfError(f"{owner} reaches past the end of its buffer view. The file is "
                            f"truncated; re-export it.")
        base = np.frombuffer(data, dtype=np.uint8, count=need - offset, offset=offset)
        if step == element:
            return base.reshape(count, element).copy()
        return np.lib.stride_tricks.as_strided(base, shape=(count, element), strides=(step, 1)).copy()

    @staticmethod
    def _unpack(raw: np.ndarray, dtype, kind: str, padded: bool, rows: int, width: int) -> np.ndarray:
        count = raw.shape[0]
        if count == 0:
            return np.zeros((0, _COMPONENTS[kind]), dtype=dtype)
        if padded:
            column_padded = raw.shape[1] // rows
            raw = raw.reshape(count, rows, column_padded)[:, :, :rows * width].reshape(count, -1)
            raw = np.ascontiguousarray(raw)
        return raw.view(dtype).reshape(count, -1)

    def _apply_sparse(self, index, sparse, values, dtype, kind, padded, rows, width, element, count):
        try:
            sparse_count = int(sparse["count"])
            index_spec = sparse["indices"]
            value_spec = sparse["values"]
        except (KeyError, TypeError, ValueError) as error:
            raise GltfError(f"accessor {index} has an incomplete sparse section. Re-export the "
                            f"file.") from error
        if not 0 <= sparse_count <= count:
            raise GltfError(f"accessor {index} claims more sparse entries than elements. The file "
                            f"is damaged; re-export it.")
        index_type = _DTYPES.get(index_spec.get("componentType"))
        if index_type is None or index_type.kind != "u":
            raise GltfError(f"accessor {index} has sparse indices of an unknown type. Re-export "
                            f"the file.")
        positions = self._elements(index_spec.get("bufferView"), int(index_spec.get("byteOffset", 0)),
                                   sparse_count, index_type.itemsize, False, f"accessor {index}")
        positions = positions.view(index_type).reshape(-1).astype(np.int64)
        if positions.size and int(positions.max()) >= count:
            raise GltfError(f"accessor {index} has a sparse index past its last element. The file "
                            f"is damaged; re-export it.")
        replacement = self._elements(value_spec.get("bufferView"), int(value_spec.get("byteOffset", 0)),
                                     sparse_count, element, False, f"accessor {index}")
        values = values.copy()
        values[positions] = self._unpack(replacement, dtype, kind, padded, rows, width)
        return values

    def image_bytes(self, index: int) -> bytes | None:
        """The encoded bytes of an image.

        Args:
            index: Image index.

        Returns:
            The bytes, or None when they cannot be found, with a warning logged.
        """
        entry = self.item("images", index)
        if "bufferView" in entry:
            data, _ = self.view(entry["bufferView"])
            return bytes(data)
        uri = entry.get("uri")
        if not isinstance(uri, str) or not uri:
            self.warn_once(f"image{index}", "image %d has no data and was skipped", index)
            return None
        if uri.startswith("data:"):
            try:
                return _data_uri(uri, f"image {index}")
            except GltfError as error:
                self.warn_once(f"image{index}", "image %d was skipped: %s", index, error)
                return None
        found = self.sidecar(_unquote(uri), keep=False)
        if found is None and self.base_dir is None:
            self.warn_once(f"image{index}", "image %d names a file beside the model, and the model "
                           "came from memory, so it was skipped", index)
        return found

    @_damaged
    def texture(self, index) -> Texture | None:
        """A texture with its sampler's wrap modes, decoded once per image and sampler.

        Args:
            index: Texture index.

        Returns:
            The texture, or None when its image is missing or in an unsupported format.
        """
        entry = self.item("textures", index)
        extensions = entry.get("extensions") or {}
        sources = []
        webp = extensions.get("EXT_texture_webp")
        if isinstance(webp, dict) and isinstance(webp.get("source"), int):
            sources.append(webp["source"])
        if isinstance(entry.get("source"), int):
            sources.append(entry["source"])
        if not sources:
            if "KHR_texture_basisu" in extensions:
                self.warn_once("basisu", "a texture is stored only as KTX2 Basis Universal, which "
                               "is not decoded, so it was skipped. Export textures as PNG or JPEG.")
            else:
                self.warn_once(f"texture{index}", "texture %d has no image and was skipped", index)
            return None
        sampler = entry.get("sampler", -1)
        sampler = sampler if isinstance(sampler, int) else -1
        for source in sources:
            key = (source, sampler)
            if key in self._textures:
                if self._textures[key] is not None:
                    return self._textures[key]
                continue
            image = self._image(source)
            if image is None:
                self._textures[key] = None
                continue
            texture = Texture(image=image.image, name=image.name, mime=image.mime)
            if sampler >= 0:
                spec = self.item("samplers", sampler)
                texture.wrap_s = _WRAP_READ.get(spec.get("wrapS", 10497), "repeat")
                texture.wrap_t = _WRAP_READ.get(spec.get("wrapT", 10497), "repeat")
            self._textures[key] = texture
            return texture
        return None

    def _image(self, index: int) -> Texture | None:
        if index in self._images:
            return self._images[index]
        entry = self.item("images", index)
        mime = str(entry.get("mimeType", ""))
        uri = str(entry.get("uri", ""))
        name = str(entry.get("name") or "")
        if not name and uri and not uri.startswith("data:"):
            name = _unquote(uri.replace("\\", "/").rsplit("/", 1)[-1])
        if mime == "image/ktx2" or uri.lower().endswith(".ktx2"):
            self.warn_once("ktx2", "a KTX2 texture was skipped: KTX2 images are not decoded. Export "
                           "textures as PNG or JPEG.")
            self._images[index] = None
            return None
        external = uri and not uri.startswith("data:") and "bufferView" not in entry
        key = ("uri", _unquote(uri)) if external else None
        if key is None or key not in self._decoded:
            data = self.image_bytes(index)
            if key is None and data:
                key = ("bytes", hashlib.sha1(data).hexdigest())
            if key is not None and key in self._decoded:
                shared = self._decoded[key]
            else:
                shared = self._decode_budgeted(data, name or f"image_{index}") if data else None
                if key is not None:
                    self._decoded[key] = shared
        else:
            shared = self._decoded[key]
        decoded = None
        if shared is not None:
            decoded = Texture(image=shared.image, name=name or shared.name, mime=shared.mime)
        self._images[index] = decoded
        return decoded

    def _decode_budgeted(self, data: bytes, name: str) -> Texture | None:
        decoded = textures.decode(data, name=name)
        if decoded is None:
            return None
        self._texture_pixels += int(decoded.image.shape[0]) * int(decoded.image.shape[1])
        if self._texture_pixels > MAX_TEXTURE_TOTAL_PIXELS:
            self.warn_once("pixels", "the model's textures hold more than %d megapixels together; "
                           "the rest were skipped. Downscale the textures before loading.",
                           MAX_TEXTURE_TOTAL_PIXELS // 1_000_000)
            return None
        if decoded.mime not in ("image/png", "image/jpeg"):
            decoded.mime = "image/png"
        return decoded


_UNRESERVED = frozenset(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~/")


def _unquote(text: str) -> str:
    """Percent-decode a URI reference as UTF-8.

    Args:
        text: The reference, such as ``"my%20texture.png"``.

    Returns:
        The decoded text; malformed escapes are kept as written.
    """
    if "%" not in text:
        return text
    raw = text.encode("utf-8")
    out = bytearray()
    index = 0
    while index < len(raw):
        if raw[index] == 0x25 and index + 2 < len(raw):
            try:
                out.append(int(raw[index + 1:index + 3], 16))
                index += 3
                continue
            except ValueError:
                pass
        out.append(raw[index])
        index += 1
    return out.decode("utf-8", errors="replace")


def _quote(text: str) -> str:
    """Percent-encode a file name for use as a URI reference.

    Args:
        text: The file name.

    Returns:
        The encoded reference.
    """
    return "".join(chr(byte) if byte in _UNRESERVED else f"%{byte:02X}" for byte in text.encode("utf-8"))


def _number(value, default: float) -> float:
    """A finite float from a JSON value, or ``default``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _data_uri(uri: str, owner: str) -> bytes:
    """The bytes a base64 ``data:`` URI holds.

    Args:
        uri: The URI.
        owner: What the URI belongs to, for messages.

    Returns:
        The decoded bytes.

    Raises:
        GltfError: The URI is not base64 or does not decode.
    """
    head, _, payload = uri.partition(",")
    if not head.endswith(";base64"):
        raise GltfError(f"{owner} is a data URI that is not base64. Re-export the file.")
    try:
        return base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError) as error:
        raise GltfError(f"{owner} holds a data URI that does not decode. Re-export the "
                        f"file.") from error


def _split_glb(data: bytes) -> tuple[dict, memoryview | None]:
    """The JSON document and binary chunk of a ``.glb``.

    Args:
        data: The whole file.

    Returns:
        ``(json, binary chunk or None)``.

    Raises:
        GltfError: The container is malformed.
    """
    if len(data) < 20:
        raise GltfError("the .glb file is too short to hold a model. It is truncated; re-export it.")
    _, version, total = struct.unpack_from("<4sII", data, 0)
    if version != 2:
        raise GltfError(f"the .glb file is glTF version {version}; only glTF 2.0 is read. "
                        f"Re-export it as glTF 2.0.")
    total = min(total, len(data))
    view = memoryview(data)
    offset = 12
    document = None
    binary = None
    while offset + 8 <= total:
        length, kind = struct.unpack_from("<II", data, offset)
        offset += 8
        if offset + length > total:
            raise GltfError("a chunk of the .glb file runs past its end. The file is truncated; "
                            "re-export it.")
        if kind == _CHUNK_JSON and document is None:
            if length > MAX_JSON_BYTES:
                raise GltfError("the .glb file's JSON chunk is larger than a model file may hold.")
            document = _load_json(bytes(view[offset:offset + length]))
        elif kind == _CHUNK_BIN and binary is None:
            binary = view[offset:offset + length]
        offset += (length + 3) & ~3
    if document is None:
        raise GltfError("the .glb file holds no JSON chunk. It is damaged; re-export it.")
    return document, binary


def _load_json(text: bytes) -> dict:
    try:
        document = json.loads(text.decode("utf-8-sig"))
    except (UnicodeDecodeError, ValueError, RecursionError) as error:
        raise GltfError(f"the glTF JSON could not be read ({error.__class__.__name__}). The file is "
                        f"damaged; re-export it.") from error
    if not isinstance(document, dict):
        raise GltfError("the glTF JSON is not an object. The file is damaged; re-export it.")
    return document


@_damaged
def parse(data: bytes, base_dir: str | None = None) -> GltfDocument:
    """Parse a ``.glb`` or ``.gltf`` file and resolve its buffers.

    Args:
        data: The file's bytes.
        base_dir: Folder holding the file, for buffers and images it names; None reads none.

    Returns:
        The parsed document.

    Raises:
        GltfError: The file is not glTF 2.0, is damaged, or needs Draco decoding.
    """
    data = bytes(data) if not isinstance(data, bytes) else data
    binary = data[:4] == _GLB_MAGIC
    if binary:
        document, chunk = _split_glb(data)
    else:
        if len(data) > MAX_JSON_BYTES:
            raise GltfError("the .gltf file is larger than a model file may hold.")
        document, chunk = _load_json(data), None
    asset = document.get("asset")
    version = str(asset.get("version", "")) if isinstance(asset, dict) else ""
    if not version.startswith("2"):
        if version:
            raise GltfError(f"the file is glTF version {version}; only glTF 2.0 is read. "
                            f"Re-export it as glTF 2.0.")
        logger.warning("the glTF file declares no asset version; reading it as glTF 2.0")
    for name, value in document.items():
        if isinstance(value, list) and len(value) > MAX_ITEMS:
            raise GltfError(f"the file lists {len(value)} {name}, more than one model may hold.")
    images = document.get("images") or []
    if len(images) > MAX_IMAGES:
        raise GltfError(f"the file lists {len(images)} images, more than the {MAX_IMAGES} one "
                        f"model may hold. Merge textures into an atlas before exporting.")
    required = {str(name) for name in document.get("extensionsRequired", []) or []}
    if "KHR_draco_mesh_compression" in required:
        raise GltfError("the file is compressed with Draco, which is not decoded. Export it again "
                        "without Draco compression, or with meshopt compression (gltfpack -cc).")
    unknown = sorted(required - _KNOWN_EXTENSIONS)
    if unknown:
        logger.warning("the glTF file requires %s, which is not read; the model may load "
                       "incomplete", ", ".join(unknown))
    doc = GltfDocument(document, [], base_dir, binary)
    for index, entry in enumerate(document.get("buffers") or []):
        entry = entry if isinstance(entry, dict) else {}
        uri = entry.get("uri")
        if uri is None:
            doc.buffers.append(chunk if index == 0 and chunk is not None else None)
        elif isinstance(uri, str) and uri.startswith("data:"):
            doc.buffers.append(memoryview(_data_uri(uri, f"buffer {index}")))
        elif isinstance(uri, str):
            found = doc.sidecar(_unquote(uri))
            doc.buffers.append(memoryview(found) if found is not None else None)
        else:
            doc.buffers.append(None)
    return doc


# ---------------------------------------------------------------------------------------------
# Conversion to a model
# ---------------------------------------------------------------------------------------------


def _node_local(node: dict, index: int) -> np.ndarray:
    matrix = node.get("matrix")
    if matrix is not None:
        numbers = _finite_list(matrix, 16)
        if numbers is None:
            logger.warning("node %d has an unreadable matrix; identity was used", index)
            return np.eye(4)
        return np.array(numbers, dtype=np.float64).reshape(4, 4).T
    translation = _finite_list(node.get("translation", [0, 0, 0]), 3) or [0.0, 0.0, 0.0]
    rotation = _finite_list(node.get("rotation", [0, 0, 0, 1]), 4) or [0.0, 0.0, 0.0, 1.0]
    scale = _finite_list(node.get("scale", [1, 1, 1]), 3) or [1.0, 1.0, 1.0]
    return _trs_matrix(translation, rotation, scale)


class _Graph:
    """The node hierarchy with local and world matrices."""

    def __init__(self, document: dict):
        nodes = document.get("nodes") or []
        self.nodes = [node if isinstance(node, dict) else {} for node in nodes]
        count = len(self.nodes)
        self.local = [_node_local(node, index) for index, node in enumerate(self.nodes)]
        self.parent = [-1] * count
        self.children: list[list[int]] = [[] for _ in range(count)]
        for index, node in enumerate(self.nodes):
            for child in node.get("children", []) or []:
                if not isinstance(child, int) or not 0 <= child < count or child == index:
                    raise GltfError(f"node {index} lists an invalid child. The file is damaged; "
                                    f"re-export it.")
                if self.parent[child] != -1:
                    raise GltfError(f"node {child} has two parents. The file is damaged; "
                                    f"re-export it.")
                self.parent[child] = index
                self.children[index].append(child)
        self.world: list[np.ndarray | None] = [None] * count
        order = []
        stack = [index for index in range(count) if self.parent[index] == -1][::-1]
        while stack:
            index = stack.pop()
            parent = self.parent[index]
            self.world[index] = self.local[index] if parent < 0 else self.world[parent] @ self.local[index]
            order.append(index)
            stack.extend(self.children[index][::-1])
        if len(order) != count:
            raise GltfError("the node hierarchy contains a cycle. The file is damaged; re-export it.")
        self.order = order

    def active(self, document: dict) -> list[int]:
        """Nodes of the default scene in depth-first order."""
        scenes = document.get("scenes")
        roots = None
        if isinstance(scenes, list) and scenes:
            chosen = document.get("scene", 0)
            if not isinstance(chosen, int) or not 0 <= chosen < len(scenes):
                chosen = 0
            scene = scenes[chosen] if isinstance(scenes[chosen], dict) else {}
            roots = [node for node in scene.get("nodes", []) or []
                     if isinstance(node, int) and 0 <= node < len(self.nodes) and self.parent[node] == -1]
        if roots is None:
            roots = [index for index in range(len(self.nodes)) if self.parent[index] == -1]
        result = []
        stack = roots[::-1]
        while stack:
            index = stack.pop()
            result.append(index)
            stack.extend(self.children[index][::-1])
        return result

    def ancestors(self, index: int):
        """Each ancestor of a node, nearest first."""
        parent = self.parent[index]
        while parent >= 0:
            yield parent
            parent = self.parent[parent]


def _uv_matrix(transform: dict) -> np.ndarray:
    offset = _finite_list(transform.get("offset", [0, 0]), 2) or [0.0, 0.0]
    scale = _finite_list(transform.get("scale", [1, 1]), 2) or [1.0, 1.0]
    try:
        rotation = float(transform.get("rotation", 0.0))
    except (TypeError, ValueError):
        rotation = 0.0
    cos, sin = math.cos(rotation), math.sin(rotation)
    translate = np.array([[1, 0, offset[0]], [0, 1, offset[1]], [0, 0, 1]], dtype=np.float64)
    rotate = np.array([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]], dtype=np.float64)
    return translate @ rotate @ np.diag([scale[0], scale[1], 1.0])


def _uv_transform_dict(matrix: np.ndarray) -> dict:
    scale_x = math.hypot(matrix[0, 0], matrix[1, 0])
    scale_y = math.hypot(matrix[0, 1], matrix[1, 1])
    rotation = math.atan2(-matrix[1, 0], matrix[0, 0]) if scale_x > 1e-12 else 0.0
    return {"offset": [float(matrix[0, 2]), float(matrix[1, 2])], "rotation": float(rotation),
            "scale": [float(scale_x), float(scale_y)]}


class _Reader:
    """State shared while one document becomes a model."""

    def __init__(self, doc: GltfDocument):
        self.doc = doc
        self.document = doc.json
        self.graph = _Graph(doc.json)
        self.vertices = 0
        self.triangles = 0
        self.morph_vertices = 0
        self.materials: dict[int, tuple[Material, np.ndarray | None]] = {}
        self.default_material: Material | None = None

    # Materials -------------------------------------------------------------------------------

    def _texture_slot(self, info, slot: str, material_index: int):
        if not isinstance(info, dict):
            return None, None
        coord = info.get("texCoord", 0)
        extensions = info.get("extensions") or {}
        transform = extensions.get("KHR_texture_transform")
        transform = transform if isinstance(transform, dict) else None
        if transform is not None and "texCoord" in transform:
            coord = transform["texCoord"]
        if coord != 0:
            self.doc.warn_once(f"texcoord{material_index}{slot}", "material %d samples its %s "
                               "texture with a second UV set, which is not read, so that texture "
                               "was dropped", material_index, slot)
            return None, None
        try:
            texture = self.doc.texture(info.get("index"))
        except GltfError as error:
            logger.warning("material %d: its %s texture was skipped: %s", material_index, slot, error)
            return None, None
        return texture, transform

    def material(self, index) -> tuple[Material, np.ndarray | None]:
        """The material at ``index`` and the UV matrix baked into parts drawn with it."""
        if not isinstance(index, int):
            if self.default_material is None:
                self.default_material = Material(name="default")
            return self.default_material, None
        if index in self.materials:
            return self.materials[index]
        spec = self.doc.item("materials", index)
        pbr = spec.get("pbrMetallicRoughness") or {}
        extensions = spec.get("extensions") or {}
        material = Material(name=str(spec.get("name") or f"material_{index}"))
        transforms = {}
        base = _finite_list(pbr.get("baseColorFactor", [1, 1, 1, 1]), 4)
        material.base_color = tuple(base) if base else (1.0, 1.0, 1.0, 1.0)
        material.metallic = _number(pbr.get("metallicFactor", 1.0), 1.0)
        material.roughness = _number(pbr.get("roughnessFactor", 1.0), 1.0)
        slots = (
            ("base_color", pbr.get("baseColorTexture"), "base_color_texture"),
            ("metallic_roughness", pbr.get("metallicRoughnessTexture"), "metallic_roughness_texture"),
            ("normal", spec.get("normalTexture"), "normal_texture"),
            ("occlusion", spec.get("occlusionTexture"), "occlusion_texture"),
            ("emissive", spec.get("emissiveTexture"), "emissive_texture"),
        )
        for slot, info, attribute in slots:
            texture, transform = self._texture_slot(info, slot, index)
            setattr(material, attribute, texture)
            if texture is not None and transform is not None:
                transforms[slot] = _uv_matrix(transform)
        normal = spec.get("normalTexture")
        if isinstance(normal, dict):
            material.normal_scale = _number(normal.get("scale", 1.0), 1.0)
        occlusion = spec.get("occlusionTexture")
        if isinstance(occlusion, dict):
            material.occlusion_strength = _number(occlusion.get("strength", 1.0), 1.0)
        emissive = _finite_list(spec.get("emissiveFactor", [0, 0, 0]), 3)
        material.emissive = tuple(emissive) if emissive else (0.0, 0.0, 0.0)
        mode = str(spec.get("alphaMode", "OPAQUE")).upper()
        material.alpha_mode = mode if mode in ("OPAQUE", "MASK", "BLEND") else "OPAQUE"
        material.alpha_cutoff = _number(spec.get("alphaCutoff", 0.5), 0.5)
        material.double_sided = bool(spec.get("doubleSided", False))
        material.unlit = "KHR_materials_unlit" in extensions
        strength = extensions.get("KHR_materials_emissive_strength")
        if isinstance(strength, dict) and "emissiveStrength" in strength:
            material.extras["emissive_strength"] = _number(strength["emissiveStrength"], 1.0)
        baked = transforms.get("base_color")
        if baked is None and transforms and len(transforms) == len(material.textures()):
            first = next(iter(transforms.values()))
            if all(np.allclose(matrix, first) for matrix in transforms.values()):
                baked = first
        # A transform that collapses an axis stays a transform instead of being baked into UVs.
        if baked is not None and abs(np.linalg.det(baked[:2, :2])) < 1e-12:
            baked = None
        remaining = {}
        inverse = np.linalg.inv(baked) if baked is not None else None
        for slot, matrix in transforms.items():
            relative = matrix @ inverse if inverse is not None else matrix
            if not np.allclose(relative, np.eye(3), atol=1e-7):
                remaining[slot] = _uv_transform_dict(relative)
        if inverse is not None:
            for slot in material.textures():
                if slot not in transforms:
                    remaining[slot] = _uv_transform_dict(inverse)
        if remaining:
            material.extras["texture_transform"] = remaining
        others = {name: value for name, value in extensions.items() if name not in (
            "KHR_materials_unlit", "KHR_materials_emissive_strength")}
        if others:
            material.extras["extensions"] = others
        if isinstance(spec.get("extras"), dict) and spec["extras"]:
            material.extras["extras"] = spec["extras"]
        self.materials[index] = (material, baked)
        return material, baked

    # Geometry --------------------------------------------------------------------------------

    def _count(self, vertices: int, triangles: int) -> None:
        self.vertices += vertices
        self.triangles += triangles
        if self.vertices > MAX_VERTICES:
            raise GltfError(f"the model holds more than {MAX_VERTICES:,} vertices. Decimate it "
                            f"before loading.")
        if self.triangles > MAX_TRIANGLES:
            raise GltfError(f"the model holds more than {MAX_TRIANGLES:,} triangles. Decimate it "
                            f"before loading.")

    def _attribute(self, attributes: dict, name: str, normalize: bool | None = None,
                   widths: tuple[int, ...] | None = None, mesh_index: int = -1):
        index = attributes.get(name)
        if index is None:
            return None
        array = self.doc.accessor(index, normalize=normalize)
        if widths is not None and (array.ndim != 2 or array.shape[1] not in widths):
            self.doc.warn_once(f"shape{mesh_index}{name}", "mesh %d has a %s attribute of the wrong "
                               "type; it was skipped", mesh_index, name)
            return None
        return array

    def primitive(self, mesh_index: int, prim_index: int, prim: dict, matrix: np.ndarray | None,
                  skin_map: np.ndarray | None, rigid_joint: int, target_names: list[str],
                  name: str) -> tuple[Part | None, int]:
        """One primitive as a part.

        Returns:
            ``(part or None, glTF material index or -1)``.
        """
        attributes = prim.get("attributes") or {}
        extensions = prim.get("extensions") or {}
        if "KHR_draco_mesh_compression" in extensions and (
                "POSITION" not in attributes
                or "bufferView" not in self.doc.item("accessors", attributes["POSITION"])):
            raise GltfError("the file is compressed with Draco, which is not decoded. Export it "
                            "again without Draco compression, or with meshopt compression.")
        if "POSITION" not in attributes:
            self.doc.warn_once(f"nopos{mesh_index}", "mesh %d has a primitive with no positions; it "
                               "was skipped", mesh_index)
            return None, -1
        mode = prim.get("mode", 4)
        if mode not in (4, 5, 6):
            self.doc.warn_once(f"mode{mode}", "points and lines are not read; primitives drawn "
                               "with mode %s were skipped", mode)
            return None, -1
        positions = self.doc.accessor(attributes["POSITION"]).astype(np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise GltfError(f"mesh {mesh_index} has positions that are not 3D. Re-export the file.")
        count = positions.shape[0]
        if "indices" in prim:
            indices = self.doc.accessor(prim["indices"], normalize=False).astype(np.int64).reshape(-1)
        else:
            indices = np.arange(count, dtype=np.int64)
        faces = _triangulate(indices, mode)
        if faces.size and (faces.max() >= count or faces.min() < 0):
            valid = (faces < count).all(axis=1) & (faces >= 0).all(axis=1)
            self.doc.warn_once(f"range{mesh_index}", "mesh %d has triangles naming vertices it does "
                               "not hold; %d were dropped", mesh_index, int((~valid).sum()))
            faces = faces[valid]
        keep = None
        if faces.size and count:
            used = np.zeros(count, dtype=bool)
            used[faces.reshape(-1)] = True
            if not used.all():
                keep = np.flatnonzero(used)
                remap = np.full(count, -1, dtype=np.int64)
                remap[keep] = np.arange(keep.size)
                faces = remap[faces]
                positions = positions[keep]
        if faces.shape[0] == 0:
            self.doc.warn_once(f"empty{mesh_index}", "mesh %d has a primitive with no triangles; "
                               "it was skipped", mesh_index)
            return None, -1
        self._count(positions.shape[0], faces.shape[0])

        def take(array):
            if array is None:
                return None
            if array.shape[0] != count:
                raise GltfError(f"mesh {mesh_index} has attributes of unequal length. Re-export the "
                                f"file.")
            return array[keep] if keep is not None else array

        normals = take(self._attribute(attributes, "NORMAL", widths=(3,), mesh_index=mesh_index))
        tangents = take(self._attribute(attributes, "TANGENT", widths=(3, 4), mesh_index=mesh_index))
        uvs = take(self._attribute(attributes, "TEXCOORD_0", widths=(2,), mesh_index=mesh_index))
        colors = take(self._attribute(attributes, "COLOR_0", normalize=True, widths=(3, 4),
                                      mesh_index=mesh_index))
        if "TEXCOORD_1" in attributes:
            self.doc.warn_once("uv1", "second UV sets are not read")
        joints = weights = None
        if skin_map is not None and "JOINTS_0" in attributes and "WEIGHTS_0" in attributes:
            joints, weights = self._skin(attributes, skin_map, take)
        elif rigid_joint >= 0:
            joints = np.zeros((positions.shape[0], 4), dtype=np.int64)
            joints[:, 0] = rigid_joint
            weights = np.zeros((positions.shape[0], 4), dtype=np.float32)
            weights[:, 0] = 1.0
        morphs = {}
        targets = prim.get("targets") or []
        for target_index, target in enumerate(targets):
            if not isinstance(target, dict):
                continue
            self.morph_vertices += positions.shape[0]
            if self.morph_vertices > MAX_MORPH_VERTICES:
                raise GltfError(f"the model's shape keys hold more than {MAX_MORPH_VERTICES:,} "
                                f"vertices together. Remove shape keys before exporting it.")
            delta = take(self._attribute(target, "POSITION", widths=(3,), mesh_index=mesh_index))
            if delta is None:
                delta = np.zeros_like(positions)
            delta_normals = take(self._attribute(target, "NORMAL", widths=(3,), mesh_index=mesh_index))
            label = target_names[target_index] if target_index < len(target_names) else f"morph_{target_index}"
            morphs[label] = [delta.astype(np.float32), None if delta_normals is None else delta_normals.astype(np.float32)]

        if normals is not None:
            normals = normals.astype(np.float32)
        if tangents is not None:
            tangents = tangents.astype(np.float32)
            if tangents.shape[1] == 3:
                tangents = np.concatenate([tangents, np.ones((tangents.shape[0], 1), np.float32)], 1)
        if colors is not None:
            colors = colors.astype(np.float32)
            if colors.shape[1] == 3:
                colors = np.concatenate([colors, np.ones((colors.shape[0], 1), np.float32)], 1)
        if uvs is not None:
            uvs = uvs.astype(np.float32)

        if not _is_identity(matrix):
            linear = matrix[:3, :3]
            positions = (positions.astype(np.float64) @ linear.T + matrix[:3, 3]).astype(np.float32)
            normal_matrix = np.linalg.inv(linear).T
            if normals is not None:
                normals = _unit(normals.astype(np.float64) @ normal_matrix.T)
            if tangents is not None:
                tangents[:, :3] = _unit(tangents[:, :3].astype(np.float64) @ linear.T)
                if np.linalg.det(linear) < 0:
                    tangents[:, 3] = -tangents[:, 3]
            for pair in morphs.values():
                pair[0] = (pair[0].astype(np.float64) @ linear.T).astype(np.float32)
                if pair[1] is not None:
                    pair[1] = (pair[1].astype(np.float64) @ normal_matrix.T).astype(np.float32)
            if np.linalg.det(linear) < 0:
                faces = faces[:, [0, 2, 1]]

        material_index = prim.get("material")
        material, baked = self.material(material_index)
        if uvs is not None and baked is not None:
            uvs = (uvs.astype(np.float64) @ baked[:2, :2].T + baked[:2, 2]).astype(np.float32)

        part = Part(
            name=name,
            vertices=_tensor(positions, torch.float32),
            faces=_tensor(faces, torch.int64).reshape(-1, 3),
            material=material,
            uvs=_tensor(uvs, torch.float32),
            normals=_tensor(normals, torch.float32),
            tangents=_tensor(tangents, torch.float32),
            colors=_tensor(colors, torch.float32),
            joints=_tensor(joints, torch.int64),
            weights=_tensor(weights, torch.float32),
            morphs={label: MorphTarget(_tensor(pair[0], torch.float32), _tensor(pair[1], torch.float32))
                    for label, pair in morphs.items()},
        )
        return part, material_index if isinstance(material_index, int) else -1

    def _skin(self, attributes: dict, skin_map: np.ndarray, take):
        joint_sets = []
        weight_sets = []
        for set_index in range(4):
            joints = self._attribute(attributes, f"JOINTS_{set_index}", normalize=False)
            weights = self._attribute(attributes, f"WEIGHTS_{set_index}", normalize=True)
            if joints is None or weights is None:
                break
            joints = take(joints).astype(np.int64)
            weights = take(weights).astype(np.float32)
            if joints.ndim != 2 or weights.shape != joints.shape:
                raise GltfError("a skinned mesh has joint and weight sets of different shapes. "
                                "Re-export the file.")
            joint_sets.append(joints)
            weight_sets.append(weights)
        joints = np.concatenate(joint_sets, axis=1)
        weights = np.concatenate(weight_sets, axis=1)
        valid = (joints >= 0) & (joints < len(skin_map))
        weights = np.where(valid, np.maximum(weights, 0.0), 0.0).astype(np.float32)
        joints = np.where(valid, skin_map[np.clip(joints, 0, len(skin_map) - 1)], 0)
        if joints.shape[1] > 4:
            order = np.argsort(-weights, axis=1, kind="stable")
            weights = np.take_along_axis(weights, order, axis=1)
            joints = np.take_along_axis(joints, order, axis=1)
            used = int((weights > 0).any(axis=0).sum())
            width = 4 if used <= 4 else min(MAX_INFLUENCES, used)
            weights = weights[:, :width]
            joints = joints[:, :width]
        total = weights.sum(axis=1, keepdims=True)
        empty = total[:, 0] <= 0
        weights = np.where(total > 0, weights / np.maximum(total, 1e-12), 0.0).astype(np.float32)
        weights[empty, 0] = 1.0
        joints = np.where(weights > 0, joints, 0)
        return joints.astype(np.int64), weights


def _unit(vectors: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.maximum(length, 1e-12)).astype(np.float32)


def _tensor(array, dtype) -> torch.Tensor | None:
    if array is None:
        return None
    return torch.from_numpy(np.ascontiguousarray(array)).to(dtype)


def _triangulate(indices: np.ndarray, mode: int) -> np.ndarray:
    """Triangles from a list, strip or fan index stream.

    Args:
        indices: ``(K,)`` int64.
        mode: 4 list, 5 strip, 6 fan.

    Returns:
        ``(M, 3)`` int64, degenerate strip and fan triangles removed.
    """
    if mode == 4:
        usable = indices.size - indices.size % 3
        return indices[:usable].reshape(-1, 3)
    if indices.size < 3:
        return np.zeros((0, 3), dtype=np.int64)
    step = np.arange(indices.size - 2)
    if mode == 5:
        odd = step % 2
        faces = np.stack([indices[step], indices[step + 1 + odd], indices[step + 2 - odd]], axis=1)
    else:
        faces = np.stack([indices[step + 1], indices[step + 2], np.full(step.size, indices[0])], axis=1)
    keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    return faces[keep]


def _unique_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    result = []
    for name in names:
        if name in seen:
            seen[name] += 1
            candidate = f"{name}_{seen[name]}"
            while candidate in seen:
                seen[name] += 1
                candidate = f"{name}_{seen[name]}"
            seen[candidate] = 0
            result.append(candidate)
        else:
            seen[name] = 0
            result.append(name)
    return result


def _channel_correction(pre: np.ndarray, path: str, values: np.ndarray, cubic: bool) -> np.ndarray:
    """Animation values carried from a node's own parent space into its joint's parent space.

    Args:
        pre: ``(4, 4)`` matrix from the node's parent space into the joint's parent space.
        path: ``"translation"``, ``"rotation"`` or ``"scale"``.
        values: ``(T, k)`` values, or ``(3T, k)`` in-tangent, value, out-tangent triples.
        cubic: Whether ``values`` are cubic spline triples.

    Returns:
        The corrected values, float32.
    """
    parts = _decompose(pre, tolerance=1e-3)
    linear = pre[:3, :3]
    values = values.astype(np.float64)
    role = np.zeros(values.shape[0], dtype=bool)
    if cubic:
        role[1::3] = True
    else:
        role[:] = True
    if path == "translation":
        out = values @ linear.T
        out[role] += pre[:3, 3]
    elif path == "rotation":
        rotation = parts[1] if parts is not None else _matrix_quat(_orthonormal(linear))
        out = _quat_multiply(np.broadcast_to(rotation, values.shape), values)
        if not cubic:
            out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    else:
        scale = np.abs(parts[2]) if parts is not None else np.linalg.norm(linear, axis=0)
        out = values * scale
    return out.astype(np.float32)


def _orthonormal(linear: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(linear)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] = -u[:, -1]
        rotation = u @ vt
    return rotation


@_damaged
def to_model(doc: GltfDocument, name: str = "") -> tuple[Model3D, NodeMap]:
    """Convert a parsed document into a model.

    Args:
        doc: The parsed document.
        name: Model name stored in ``meta["name"]`` when given.

    Returns:
        ``(model, node map)``.

    Raises:
        GltfError: The file is damaged or exceeds the model limits.
    """
    reader = _Reader(doc)
    document = doc.json
    graph = reader.graph
    nodes = graph.nodes
    active = graph.active(document)
    node_map = NodeMap()

    skins = [skin if isinstance(skin, dict) else {} for skin in document.get("skins") or []]
    joint_nodes: list[int] = []
    inverse_binds: dict[int, np.ndarray] = {}
    skin_has_ibm: dict[int, bool] = {}
    bind_arrays: dict[Any, np.ndarray] = {}
    for skin_index, skin in enumerate(skins):
        listed = [joint for joint in skin.get("joints", []) or []
                  if isinstance(joint, int) and not isinstance(joint, bool) and 0 <= joint < len(nodes)]
        if len(listed) != len(skin.get("joints", []) or []):
            raise GltfError(f"skin {skin_index} names joints the file does not hold. Re-export it.")
        matrices = None
        if "inverseBindMatrices" in skin:
            source = skin["inverseBindMatrices"]
            key = source if isinstance(source, int) else id(source)
            if key not in bind_arrays:
                bind_arrays[key] = doc.accessor(source).astype(np.float64)
            matrices = bind_arrays[key]
            if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
                raise GltfError(f"skin {skin_index} has inverse bind matrices that are not 4x4. "
                                f"Re-export it.")
            if matrices.shape[0] < len(listed):
                raise GltfError(f"skin {skin_index} has fewer bind matrices than joints. Re-export it.")
        for position, joint in enumerate(listed):
            if joint not in skin_has_ibm:
                joint_nodes.append(joint)
                skin_has_ibm[joint] = matrices is not None
                if matrices is not None:
                    inverse_binds[joint] = matrices[position].copy()
    bind_arrays.clear()

    animations = [entry if isinstance(entry, dict) else {} for entry in document.get("animations") or []]
    animated = set()
    for animation in animations:
        for channel in animation.get("channels", []) or []:
            target = channel.get("target") if isinstance(channel, dict) else None
            if (isinstance(target, dict) and target.get("path") in ("translation", "rotation", "scale")
                    and isinstance(target.get("node"), int) and 0 <= target["node"] < len(nodes)):
                animated.add(target["node"])
    joint_set = set(joint_nodes)
    promote = set()
    if animated:
        # Walks up from each start, visiting every node at most once.
        visited: set[int] = set()

        def climb(start: int) -> None:
            current = start
            while current >= 0 and current not in visited:
                visited.add(current)
                if current in animated:
                    promote.add(current)
                current = graph.parent[current]

        for index in active:
            node = nodes[index]
            if "mesh" in node and "skin" not in node:
                climb(index)
        for joint in joint_nodes:
            climb(graph.parent[joint])
    for index in sorted(promote - joint_set):
        joint_nodes.append(index)
        skin_has_ibm[index] = False
    joint_set = set(joint_nodes)

    # Nearest joint at or above each node, filled parents first.
    nearest = [-1] * len(nodes)
    for index in graph.order:
        parent = graph.parent[index]
        nearest[index] = index if index in joint_set else (nearest[parent] if parent >= 0 else -1)

    def joint_parent(index: int) -> int:
        parent = graph.parent[index]
        return nearest[parent] if parent >= 0 else -1

    ordered: list[int] = []
    placed = set()
    for joint in joint_nodes:
        chain = []
        current = joint
        while current >= 0 and current not in placed:
            chain.append(current)
            current = joint_parent(current)
        for item in reversed(chain):
            placed.add(item)
            ordered.append(item)
    joint_index = {node: position for position, node in enumerate(ordered)}
    node_map.joint_of_node = dict(joint_index)

    space = None
    skeleton = None
    rest = inverse = np.zeros((0, 4, 4))
    if ordered:
        # Each joint's map from the file's node space into its skin's bind space.
        corrections = [np.linalg.inv(inverse_binds[joint]) @ np.linalg.inv(graph.world[joint])
                       for joint in ordered if joint in inverse_binds]
        consistent = False
        if corrections:
            first = corrections[0]
            scale = max(1.0, float(np.abs(first).max()))
            consistent = all(np.abs(item - first).max() <= 1e-4 * scale for item in corrections)
            if consistent and not _is_identity(first, 1e-6):
                space = np.linalg.inv(first)
            if not consistent:
                logger.info("the file's pose differs from its bind pose; joints use the bind pose")
        rest = np.zeros((len(ordered), 4, 4))
        inverse = np.zeros((len(ordered), 4, 4))
        for position, joint in enumerate(ordered):
            if joint in inverse_binds:
                bind = np.linalg.inv(inverse_binds[joint])
                rest[position] = bind if space is None else space @ bind
                inverse[position] = (inverse_binds[joint] if space is None
                                     else inverse_binds[joint] @ np.linalg.inv(space))
            elif joint not in promote:
                rest[position] = graph.world[joint]
                inverse[position] = np.eye(4)
            else:
                rest[position] = graph.world[joint]
                inverse[position] = np.linalg.inv(graph.world[joint])
        names = _unique_names([str(nodes[joint].get("name") or f"joint_{joint}") for joint in ordered])
        parents = [joint_index.get(joint_parent(joint), -1) for joint in ordered]
        skeleton = Skeleton(
            names=names,
            parents=torch.tensor(parents, dtype=torch.int64),
            rest=torch.from_numpy(rest.astype(np.float32)),
            inverse_bind=torch.from_numpy(inverse.astype(np.float32)),
        )

    skin_maps = []
    for skin in skins:
        listed = [joint for joint in skin.get("joints", []) or []]
        skin_maps.append(np.asarray([joint_index[joint] for joint in listed], dtype=np.int64)
                         if listed else None)

    meshes = document.get("meshes") or []
    parts: list[Part] = []
    for index in active:
        node = nodes[index]
        mesh_index = node.get("mesh")
        if not isinstance(mesh_index, int):
            continue
        mesh = doc.item("meshes", mesh_index)
        extras = mesh.get("extras") if isinstance(mesh.get("extras"), dict) else {}
        target_names = _unique_names([str(item) for item in extras.get("targetNames", []) or []])
        skin_index = node.get("skin")
        skin_map = None
        if isinstance(skin_index, int) and 0 <= skin_index < len(skin_maps):
            skin_map = skin_maps[skin_index]
        base_name = str(node.get("name") or mesh.get("name") or f"mesh_{mesh_index}")
        primitives = [prim for prim in mesh.get("primitives", []) or [] if isinstance(prim, dict)]
        placements = [graph.world[index]]
        instancing = (node.get("extensions") or {}).get("EXT_mesh_gpu_instancing")
        if skin_map is None and isinstance(instancing, dict):
            placements = _instances(doc, instancing, graph.world[index]) or placements
        rigid = -1
        carry = None
        if skin_map is None and ordered and nearest[index] >= 0:
            rigid = joint_index[nearest[index]]
            # Moves the mesh with its joint from the file's pose to the joint's rest, in bind space.
            carry = (np.linalg.inv(rest[rigid] @ inverse[rigid]) @ rest[rigid]
                     @ np.linalg.inv(graph.world[nearest[index]]))
            carry = None if _is_identity(carry, 1e-9) else carry
        for placement_index, placement in enumerate(placements):
            for prim_index, prim in enumerate(primitives):
                label = base_name if len(primitives) == 1 else f"{base_name}_{prim_index}"
                if len(placements) > 1:
                    label = f"{label}_{placement_index}"
                skinned = skin_map is not None and "JOINTS_0" in (prim.get("attributes") or {})
                matrix = space if skinned else (placement if carry is None else carry @ placement)
                part, material_index = reader.primitive(
                    mesh_index, prim_index, prim, matrix, skin_map if skinned else None,
                    -1 if skinned else rigid, target_names, label)
                if part is None:
                    continue
                if len(parts) >= MAX_PARTS:
                    raise GltfError(f"the model holds more than {MAX_PARTS:,} parts. Merge meshes "
                                    f"before exporting it.")
                if isinstance(mesh.get("weights"), list) and part.morphs:
                    part.extras["morph_weights"] = [_number(value, 0.0) for value in mesh["weights"]]
                if isinstance(extras.get("label"), str):
                    part.label = extras["label"]
                position = len(parts)
                parts.append(part)
                node_map.parts_of_mesh.setdefault(mesh_index, []).append(position)
                node_map.parts_of_node.setdefault(index, []).append(position)
                node_map.material_of_part.append(material_index)

    model = Model3D(parts=parts, skeleton=skeleton, source_format="glb" if doc.binary else "gltf")
    model.animations, morph_tracks = _animations(doc, graph, animations, ordered, joint_index,
                                                 skeleton, node_map)
    if morph_tracks:
        model.extras["morph_animations"] = morph_tracks
    asset = document.get("asset") if isinstance(document.get("asset"), dict) else {}
    if isinstance(asset.get("extras"), dict):
        model.meta.update({str(key): value for key, value in asset["extras"].items()})
    for key in ("copyright", "generator"):
        if isinstance(asset.get(key), str):
            model.meta[key] = asset[key]
    if name:
        model.meta.setdefault("name", name)
    used = sorted(doc.extensions_used)
    if used:
        model.extras["gltf_extensions"] = used
    return model, node_map


def _instances(doc: GltfDocument, spec: dict, world: np.ndarray) -> list[np.ndarray]:
    attributes = spec.get("attributes") or {}
    arrays = {}
    for key in ("TRANSLATION", "ROTATION", "SCALE"):
        if key in attributes:
            arrays[key] = doc.accessor(attributes[key]).astype(np.float64)
    if not arrays:
        return []
    count = min(array.shape[0] for array in arrays.values())
    if count > MAX_INSTANCES:
        raise GltfError(f"a node places {count:,} instances, more than the {MAX_INSTANCES:,} one "
                        f"node may place. Export fewer instances.")
    widths = {"TRANSLATION": 3, "ROTATION": 4, "SCALE": 3}
    if any(array.ndim != 2 or array.shape[1] != widths[key] for key, array in arrays.items()):
        raise GltfError("a node's instancing attributes have the wrong type. Re-export the file.")
    result = []
    for item in range(count):
        translation = arrays["TRANSLATION"][item] if "TRANSLATION" in arrays else (0.0, 0.0, 0.0)
        rotation = arrays["ROTATION"][item] if "ROTATION" in arrays else (0.0, 0.0, 0.0, 1.0)
        scale = arrays["SCALE"][item] if "SCALE" in arrays else (1.0, 1.0, 1.0)
        result.append(world @ _trs_matrix(translation, rotation, scale))
    return result


def _animations(doc, graph, animations, ordered, joint_index, skeleton, node_map):
    """Joint animation clips and the morph weight tracks recorded beside them."""
    result = []
    morph_tracks = []
    if not animations:
        return result, morph_tracks
    kept_values = 0
    pre: dict[int, np.ndarray | None] = {}
    rest = skeleton.rest.double().numpy() if skeleton is not None else None
    for joint in ordered:
        node_parent = graph.parent[joint]
        parent_world = graph.world[node_parent] if node_parent >= 0 else np.eye(4)
        position = joint_index[joint]
        skeleton_parent = int(skeleton.parents[position])
        if skeleton_parent >= 0:
            matrix = np.linalg.inv(graph.world[ordered[skeleton_parent]]) @ parent_world
        else:
            matrix = rest[position] @ np.linalg.inv(graph.world[joint]) @ parent_world
        pre[joint] = None if _is_identity(matrix, 1e-6) else matrix
    for animation_index, animation in enumerate(animations):
        clip = Animation(name=str(animation.get("name") or f"animation_{animation_index}"))
        samplers = animation.get("samplers", []) or []
        for channel in animation.get("channels", []) or []:
            if not isinstance(channel, dict) or not isinstance(channel.get("target"), dict):
                continue
            target = channel["target"]
            node = target.get("node")
            path = target.get("path")
            sampler_index = channel.get("sampler")
            if not isinstance(sampler_index, int) or not 0 <= sampler_index < len(samplers):
                continue
            if path == "weights":
                if not isinstance(node, int) or not node_map.parts_of_node.get(node):
                    continue
            elif path not in ("translation", "rotation", "scale") or node not in joint_index:
                continue
            sampler = samplers[sampler_index] if isinstance(samplers[sampler_index], dict) else {}
            interpolation = str(sampler.get("interpolation", "LINEAR")).upper()
            if interpolation not in ("LINEAR", "STEP", "CUBICSPLINE"):
                interpolation = "LINEAR"
            try:
                times = doc.accessor(sampler.get("input")).astype(np.float32).reshape(-1)
                values = doc.accessor(sampler.get("output")).astype(np.float32)
            except GltfError as error:
                logger.warning("animation %s: a channel was skipped: %s", clip.name, error)
                continue
            kept_values += times.size + values.size
            if kept_values > MAX_ANIMATION_VALUES:
                raise GltfError(f"the file's animations hold more than {MAX_ANIMATION_VALUES:,} "
                                f"keyframe values together. Bake fewer keys before exporting it.")
            if not (np.isfinite(times).all() and np.isfinite(values).all()):
                logger.warning("animation %s: a %s channel holds numbers that are not finite and was "
                               "skipped", clip.name, path)
                continue
            cubic = interpolation == "CUBICSPLINE"
            expected = times.size * (3 if cubic else 1)
            if path == "weights":
                if times.size == 0 or values.size % expected:
                    continue
                parts_here = node_map.parts_of_node.get(node, [])
                morph_tracks.append({
                    "animation": clip.name, "node": node, "parts": list(parts_here),
                    "times": torch.from_numpy(times.copy()),
                    "values": torch.from_numpy(values.reshape(expected, -1).copy()),
                    "interpolation": interpolation,
                })
                continue
            width = 4 if path == "rotation" else 3
            if values.ndim == 1 or values.shape[0] != expected or values.shape[1] != width:
                logger.warning("animation %s: a %s channel has the wrong number of values and was "
                               "skipped", clip.name, path)
                continue
            correction = pre.get(node)
            if correction is not None:
                values = _channel_correction(correction, path, values, cubic)
            elif path == "rotation" and not cubic:
                values = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
            clip.channels.append(Channel(
                joint=joint_index[node], path=path, times=torch.from_numpy(times.copy()),
                values=torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32)),
                interpolation=interpolation,
            ))
        if clip.channels:
            result.append(clip)
    return result, morph_tracks


def read(data: bytes, base_dir: str | None = None, name: str = "") -> Model3D:
    """Read a ``.glb`` or ``.gltf`` file into a model.

    Args:
        data: The file's bytes.
        base_dir: Folder holding the file, for buffers and images it names.
        name: Model name stored in ``meta["name"]``.

    Returns:
        The model.

    Raises:
        GltfError: The file is not glTF 2.0, is damaged, or exceeds the model limits.
    """
    return to_model(parse(data, base_dir), name)[0]


# ---------------------------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------------------------


def _component_type(dtype: np.dtype) -> int:
    wanted = (np.dtype(dtype).kind, np.dtype(dtype).itemsize)
    for code, candidate in _DTYPES.items():
        if (candidate.kind, candidate.itemsize) == wanted:
            return code
    raise GltfError(f"an array of type {dtype} cannot be written to glTF.")


def _json_safe(value):
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value if not isinstance(value, float) or math.isfinite(value) else None
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            safe = _json_safe(item)
            if safe is not None or item is None:
                out[str(key)] = safe
        return out
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, torch.Tensor) and value.numel() <= 64:
        return _json_safe(value.tolist())
    return None


class GltfBuild:
    """A glTF document being assembled: JSON, one binary buffer, and where each part landed.

    Attributes:
        json: The JSON document.
        buffer: The binary buffer.
        joint_nodes: Node index of each skeleton joint.
        part_meshes: Mesh index of each part, ``-1`` when the part was empty.
        part_nodes: Node index of each part, ``-1`` when the part was empty.
        material_of_part: Material index of each part, ``-1`` when the part was empty.
    """

    def __init__(self):
        self.json: dict = {"asset": {"version": "2.0", "generator": "WAS Node Suite"}}
        self.buffer = bytearray()
        self.joint_nodes: list[int] = []
        self.part_meshes: list[int] = []
        self.part_nodes: list[int] = []
        self.material_of_part: list[int] = []
        self._external: dict[int, tuple[bytes, str]] = {}
        self._materials: dict[int, int] = {}
        self._textures: dict[tuple[int, int], int] = {}
        self._images: dict[int, int] = {}
        self._samplers: dict[tuple[str, str], int] = {}
        self._embed = True
        self._keep: list = []

    def entries(self, kind: str) -> list:
        """The top-level JSON array ``kind``, created when absent."""
        return self.json.setdefault(kind, [])

    def add_extension_used(self, name: str, required: bool = False) -> None:
        """List an extension in ``extensionsUsed`` and optionally ``extensionsRequired``.

        Args:
            name: Extension name.
            required: Also list it as required.
        """
        used = self.entries("extensionsUsed")
        if name not in used:
            used.append(name)
        if required:
            needed = self.entries("extensionsRequired")
            if name not in needed:
                needed.append(name)

    def add_view(self, data: bytes, target: int | None = None, stride: int | None = None) -> int:
        """Append bytes to the buffer as a new buffer view, 4-byte aligned.

        Args:
            data: The bytes.
            target: ``34962`` for vertex data, ``34963`` for indices, or None.
            stride: Byte stride, or None for tightly packed data.

        Returns:
            The buffer view index.
        """
        padding = (-len(self.buffer)) % 4
        self.buffer.extend(b"\x00" * padding)
        view = {"buffer": 0, "byteOffset": len(self.buffer), "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        if stride is not None:
            view["byteStride"] = stride
        self.buffer.extend(data)
        views = self.entries("bufferViews")
        views.append(view)
        return len(views) - 1

    def add_accessor(self, array, kind: str | None = None, normalized: bool = False,
                     bounds: bool = False, target: int | None = None) -> int:
        """Write an array as a buffer view and an accessor over it.

        Args:
            array: numpy array or tensor: ``(N,)`` scalars, ``(N, k)`` vectors or ``(N, 4, 4)``
                row-major matrices. Its dtype selects the component type.
            kind: Accessor type such as ``"VEC3"``; None infers it from the shape.
            normalized: Mark integer components as normalized.
            bounds: Record ``min`` and ``max``.
            target: Buffer view target.

        Returns:
            The accessor index.
        """
        if isinstance(array, torch.Tensor):
            array = array.detach().cpu().numpy()
        array = np.asarray(array)
        if array.dtype == np.float64:
            array = array.astype(np.float32)
        elif array.dtype == np.int64:
            array = array.astype(np.uint32)
        count = int(array.shape[0])
        if array.ndim == 3:
            kind = kind or "MAT4"
            flat = np.ascontiguousarray(array.transpose(0, 2, 1)).reshape(count, -1)
        elif array.ndim == 2:
            kind = kind or f"VEC{array.shape[1]}"
            flat = np.ascontiguousarray(array)
        else:
            kind = kind or "SCALAR"
            flat = np.ascontiguousarray(array).reshape(count, 1)
        flat = flat.astype(flat.dtype.newbyteorder("<"), copy=False)
        accessor = {
            "bufferView": self.add_view(flat.tobytes(), target=target),
            "componentType": _component_type(flat.dtype),
            "count": count,
            "type": kind,
        }
        if normalized:
            accessor["normalized"] = True
        if bounds and count:
            low = flat.min(axis=0)
            high = flat.max(axis=0)
            cast = float if flat.dtype.kind == "f" else int
            accessor["min"] = [cast(value) for value in low]
            accessor["max"] = [cast(value) for value in high]
        accessors = self.entries("accessors")
        accessors.append(accessor)
        return len(accessors) - 1

    def add_sampler(self, wrap_s: str, wrap_t: str) -> int:
        """A sampler index for a pair of wrap modes, shared between textures.

        Args:
            wrap_s: ``"repeat"``, ``"clamp"`` or ``"mirror"``.
            wrap_t: The same along ``v``.

        Returns:
            The sampler index.
        """
        key = (wrap_s, wrap_t)
        if key not in self._samplers:
            samplers = self.entries("samplers")
            samplers.append({"magFilter": 9729, "minFilter": 9987,
                             "wrapS": _WRAP_WRITE.get(wrap_s, 10497),
                             "wrapT": _WRAP_WRITE.get(wrap_t, 10497)})
            self._samplers[key] = len(samplers) - 1
        return self._samplers[key]

    def add_image(self, texture: Texture) -> int:
        """An image index for a texture's pixels, encoded once per tensor.

        Args:
            texture: The texture.

        Returns:
            The image index.
        """
        key = id(texture.image)
        if key in self._images:
            return self._images[key]
        self._keep.append(texture.image)
        data, mime = textures.encode(texture)
        images = self.entries("images")
        entry: dict = {"mimeType": mime}
        if texture.name:
            entry["name"] = texture.name
        index = len(images)
        if self._embed:
            entry["bufferView"] = self.add_view(data)
        else:
            self._external[index] = (data, mime)
            entry["uri"] = f"image_{index}{'.jpg' if mime == 'image/jpeg' else '.png'}"
        images.append(entry)
        self._images[key] = index
        return index

    def add_texture(self, texture: Texture) -> int:
        """A texture index for a texture, shared between materials.

        Args:
            texture: The texture.

        Returns:
            The texture index.
        """
        image = self.add_image(texture)
        sampler = self.add_sampler(texture.wrap_s, texture.wrap_t)
        key = (image, sampler)
        if key not in self._textures:
            entries = self.entries("textures")
            entries.append({"source": image, "sampler": sampler})
            self._textures[key] = len(entries) - 1
        return self._textures[key]

    def _texture_info(self, material: Material, slot: str, texture: Texture, **fields) -> dict:
        info = {"index": self.add_texture(texture)}
        info.update(fields)
        transform = (material.extras.get("texture_transform") or {}).get(slot)
        if isinstance(transform, dict):
            safe = _json_safe(transform)
            if safe:
                info["extensions"] = {"KHR_texture_transform": safe}
                self.add_extension_used("KHR_texture_transform")
        return info

    def add_material(self, material: Material) -> int:
        """A material index for a material object, written once per object.

        Args:
            material: The material.

        Returns:
            The material index.
        """
        key = id(material)
        if key in self._materials:
            return self._materials[key]
        self._keep.append(material)
        def unit(value, default: float) -> float:
            return min(max(_number(value, default), 0.0), 1.0)

        pbr: dict = {}
        base = [unit(value, 1.0) for value in material.base_color]
        if base != [1.0, 1.0, 1.0, 1.0]:
            pbr["baseColorFactor"] = base
        if material.base_color_texture is not None:
            pbr["baseColorTexture"] = self._texture_info(material, "base_color", material.base_color_texture)
        pbr["metallicFactor"] = unit(material.metallic, 1.0)
        pbr["roughnessFactor"] = unit(material.roughness, 1.0)
        if material.metallic_roughness_texture is not None:
            pbr["metallicRoughnessTexture"] = self._texture_info(
                material, "metallic_roughness", material.metallic_roughness_texture)
        entry: dict = {"name": material.name or "material", "pbrMetallicRoughness": pbr}
        if material.normal_texture is not None:
            scale = _number(material.normal_scale, 1.0)
            fields = {"scale": scale} if scale != 1.0 else {}
            entry["normalTexture"] = self._texture_info(material, "normal", material.normal_texture, **fields)
        if material.occlusion_texture is not None:
            occlusion = unit(material.occlusion_strength, 1.0)
            fields = {"strength": occlusion} if occlusion != 1.0 else {}
            entry["occlusionTexture"] = self._texture_info(material, "occlusion", material.occlusion_texture, **fields)
        emissive = [max(_number(value, 0.0), 0.0) for value in material.emissive]
        strength = max(_number(material.extras.get("emissive_strength", 1.0), 1.0), 0.0)
        peak = max(emissive) if emissive else 0.0
        if peak > 1.0:
            emissive = [value / peak for value in emissive]
            strength *= peak
        if any(emissive):
            entry["emissiveFactor"] = emissive
        if material.emissive_texture is not None:
            entry["emissiveTexture"] = self._texture_info(material, "emissive", material.emissive_texture)
            if not any(emissive):
                entry["emissiveFactor"] = [1.0, 1.0, 1.0]
        extensions = {}
        if strength != 1.0:
            extensions["KHR_materials_emissive_strength"] = {"emissiveStrength": strength}
            self.add_extension_used("KHR_materials_emissive_strength")
        if material.unlit:
            extensions["KHR_materials_unlit"] = {}
            self.add_extension_used("KHR_materials_unlit")
        if extensions:
            entry["extensions"] = extensions
        if material.alpha_mode in ("MASK", "BLEND"):
            entry["alphaMode"] = material.alpha_mode
            if material.alpha_mode == "MASK":
                entry["alphaCutoff"] = max(_number(material.alpha_cutoff, 0.5), 0.0)
        if material.double_sided:
            entry["doubleSided"] = True
        materials = self.entries("materials")
        materials.append(entry)
        self._materials[key] = len(materials) - 1
        return self._materials[key]

    def _copy_json(self) -> dict:
        try:
            return json.loads(json.dumps(self.json, allow_nan=False))
        except ValueError as error:
            raise GltfError("the model holds a number that is not finite (NaN or infinity), such as "
                            "a vertex, a joint pose or a material value, so it cannot be written. "
                            "Repair the model before saving.") from error

    def _glb_json(self) -> tuple[dict, bytearray]:
        document = self._copy_json()
        buffer = bytearray(self.buffer)
        for index, (data, mime) in self._external.items():
            padding = (-len(buffer)) % 4
            buffer.extend(b"\x00" * padding)
            views = document.setdefault("bufferViews", [])
            views.append({"buffer": 0, "byteOffset": len(buffer), "byteLength": len(data)})
            buffer.extend(data)
            image = document["images"][index]
            image.pop("uri", None)
            image["bufferView"] = len(views) - 1
            image["mimeType"] = mime
        if buffer:
            document["buffers"] = [{"byteLength": len(buffer)}]
        else:
            document.pop("buffers", None)
        return document, buffer

    def to_glb(self) -> bytes:
        """The document as a ``.glb`` file.

        Returns:
            The file's bytes.
        """
        document, buffer = self._glb_json()
        text = json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        text += b" " * ((-len(text)) % 4)
        buffer += b"\x00" * ((-len(buffer)) % 4)
        total = 12 + 8 + len(text) + (8 + len(buffer) if buffer else 0)
        if total >= 1 << 32:
            raise GltfError("the model is larger than the 4 GB a .glb file may hold. Decimate it or "
                            "downscale its textures before saving.")
        out = bytearray(struct.pack("<4sII", _GLB_MAGIC, 2, total))
        out += struct.pack("<II", len(text), _CHUNK_JSON) + text
        if buffer:
            out += struct.pack("<II", len(buffer), _CHUNK_BIN) + buffer
        return bytes(out)

    def to_gltf(self, stem: str) -> dict[str, bytes]:
        """The document as a ``.gltf`` file with its ``.bin`` buffer and image files.

        Args:
            stem: File name without extension, such as ``"avatar"``.

        Returns:
            File name against bytes.
        """
        document = self._copy_json()
        files: dict[str, bytes] = {}
        if self.buffer:
            document["buffers"] = [{"byteLength": len(self.buffer), "uri": _quote(f"{stem}.bin")}]
            files[f"{stem}.bin"] = bytes(self.buffer)
        else:
            document.pop("buffers", None)
        for index, (data, mime) in self._external.items():
            file_name = f"{stem}_{index}{'.jpg' if mime == 'image/jpeg' else '.png'}"
            document["images"][index]["uri"] = _quote(file_name)
            files[file_name] = data
        text = json.dumps(document, indent=1, ensure_ascii=False).encode("utf-8")
        return {f"{stem}.gltf": text, **files}

    def to_embedded_gltf(self) -> bytes:
        """The document as one self-contained ``.gltf`` with its buffer in a data URI.

        Returns:
            The file's bytes.
        """
        document, buffer = self._glb_json()
        if buffer:
            document["buffers"] = [{
                "byteLength": len(buffer),
                "uri": "data:application/octet-stream;base64," + base64.b64encode(bytes(buffer)).decode("ascii"),
            }]
        return json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _smooth_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Area-weighted vertex normals.

    Args:
        vertices: ``(N, 3)``.
        faces: ``(M, 3)``.

    Returns:
        ``(N, 3)`` float32 unit normals, ``+Y`` where a vertex has no area.
    """
    vertices = vertices.double()
    normals = torch.zeros_like(vertices)
    if faces.numel():
        corners = vertices[faces]
        face_normals = torch.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0], dim=1)
        for column in range(3):
            normals.index_add_(0, faces[:, column], face_normals)
    length = normals.norm(dim=1, keepdim=True)
    up = torch.tensor([0.0, 1.0, 0.0], dtype=normals.dtype).expand_as(normals)
    normals = torch.where(length > 1e-20, normals / length.clamp_min(1e-30), up)
    return normals.float()


def _write_skeleton(out: GltfBuild, skeleton: Skeleton, scene_nodes: list[int]) -> None:
    local = skeleton.local_rest().double().numpy()
    nodes = out.entries("nodes")
    start = len(nodes)
    parents = [int(value) for value in skeleton.parents]
    for joint in range(skeleton.count):
        node: dict = {"name": skeleton.names[joint] or f"joint_{joint}"}
        parts = _decompose(local[joint])
        if parts is None:
            node["matrix"] = [float(value) for value in local[joint].T.reshape(-1)]
        else:
            translation, rotation, scale = parts
            if np.abs(translation).max() > 0:
                node["translation"] = [float(value) for value in translation]
            if np.abs(rotation - (0, 0, 0, 1)).max() > 1e-9:
                node["rotation"] = [float(value) for value in rotation]
            if np.abs(scale - 1.0).max() > 1e-9:
                node["scale"] = [float(value) for value in scale]
        nodes.append(node)
        out.joint_nodes.append(start + joint)
    for joint, parent in enumerate(parents):
        if parent >= 0:
            nodes[start + parent].setdefault("children", []).append(start + joint)
        else:
            scene_nodes.append(start + joint)


def _write_part(out: GltfBuild, part: Part, joint_count: int, skin_index: int | None) -> int:
    vertices = part.vertices.detach().float().cpu()
    faces = part.faces.detach().long().cpu().reshape(-1, 3)
    count = vertices.shape[0]
    if faces.numel() and (int(faces.max()) >= count or int(faces.min()) < 0):
        raise GltfError(f"part {part.name!r} has triangles naming vertices it does not hold, so it "
                        f"cannot be written. Rebuild the part before saving.")
    checks = [("normals", part.normals, (3,)), ("tangents", part.tangents, (4,)),
              ("UVs", part.uvs, (2,)), ("colours", part.colors, (3, 4)),
              ("joints", part.joints, None), ("weights", part.weights, None)]
    checks += [(f"shape key {name!r}", morph.positions, (3,)) for name, morph in part.morphs.items()]
    for label, tensor, widths in checks:
        if tensor is None or (label == "tangents" and tensor.shape[-1] != 4):
            continue
        if tensor.ndim != 2 or tensor.shape[0] != count or (widths and tensor.shape[1] not in widths):
            raise GltfError(f"part {part.name!r} has {label} of shape {tuple(tensor.shape)} for "
                            f"{count} vertices, so it cannot be written. Rebuild the part before "
                            f"saving.")
    if part.joints is not None and part.weights is not None and part.joints.shape != part.weights.shape:
        raise GltfError(f"part {part.name!r} has joint and weight arrays of different shapes, so it "
                        f"cannot be written. Rebind the part before saving.")
    attributes = {"POSITION": out.add_accessor(vertices.numpy(), bounds=True, target=_ARRAY_BUFFER)}
    normals = part.normals if part.normals is not None else _smooth_normals(vertices, faces)
    normals = torch.nn.functional.normalize(normals.detach().float().cpu(), dim=1)
    attributes["NORMAL"] = out.add_accessor(normals.numpy(), target=_ARRAY_BUFFER)
    if part.tangents is not None and part.tangents.shape[-1] == 4:
        tangents = part.tangents.detach().float().cpu().clone()
        tangents[:, :3] = torch.nn.functional.normalize(tangents[:, :3], dim=1)
        tangents[:, 3] = torch.where(tangents[:, 3] < 0, -1.0, 1.0)
        attributes["TANGENT"] = out.add_accessor(tangents.numpy(), target=_ARRAY_BUFFER)
    if part.uvs is not None:
        attributes["TEXCOORD_0"] = out.add_accessor(part.uvs.detach().float().cpu().numpy(), target=_ARRAY_BUFFER)
    if part.colors is not None:
        colors = part.colors.detach().float().cpu().clamp(0, 1)
        if colors.shape[1] == 3:
            colors = torch.cat([colors, torch.ones(count, 1)], dim=1)
        attributes["COLOR_0"] = out.add_accessor(colors.numpy(), target=_ARRAY_BUFFER)
    if skin_index is not None and part.joints is not None and part.weights is not None:
        joints = part.joints.detach().long().cpu()
        weights = part.weights.detach().float().cpu().clamp_min(0)
        if joints.numel() and (int(joints.max()) >= joint_count or int(joints.min()) < 0):
            raise GltfError(f"part {part.name!r} is skinned to joints the skeleton does not hold. "
                            f"Rebind the part before saving.")
        width = joints.shape[1]
        sets = max(1, (width + 3) // 4)
        pad = sets * 4 - width
        if pad:
            joints = torch.cat([joints, joints.new_zeros(count, pad)], dim=1)
            weights = torch.cat([weights, weights.new_zeros(count, pad)], dim=1)
        total = weights.sum(dim=1, keepdim=True)
        weights = torch.where(total > 0, weights / total.clamp_min(1e-12), torch.zeros_like(weights))
        weights[:, 0] = torch.where(total[:, 0] > 0, weights[:, 0], torch.ones_like(weights[:, 0]))
        joints = torch.where(weights > 0, joints, torch.zeros_like(joints))
        if joint_count > 65536:
            raise GltfError(f"the skeleton has {joint_count:,} joints, more than the 65,536 glTF "
                            f"can address. Remove joints before saving.")
        joint_type = np.uint8 if joint_count <= 256 else np.uint16
        for set_index in range(sets):
            columns = slice(set_index * 4, set_index * 4 + 4)
            attributes[f"JOINTS_{set_index}"] = out.add_accessor(
                joints[:, columns].numpy().astype(joint_type), target=_ARRAY_BUFFER)
            attributes[f"WEIGHTS_{set_index}"] = out.add_accessor(
                weights[:, columns].numpy().astype(np.float32), target=_ARRAY_BUFFER)
    primitive: dict = {"attributes": attributes}
    if faces.numel():
        index_type = np.uint16 if count < 65535 else np.uint32
        primitive["indices"] = out.add_accessor(
            faces.reshape(-1).numpy().astype(index_type), target=_ELEMENT_ARRAY_BUFFER)
        primitive["mode"] = 4
    else:
        primitive["mode"] = 0
    primitive["material"] = out.add_material(part.material)
    mesh: dict = {"name": part.name or "part", "primitives": [primitive]}
    extras: dict = {}
    if part.morphs:
        targets = []
        for morph in part.morphs.values():
            target = {"POSITION": out.add_accessor(
                morph.positions.detach().float().cpu().numpy(), bounds=True, target=_ARRAY_BUFFER)}
            if morph.normals is not None and morph.normals.shape == morph.positions.shape:
                target["NORMAL"] = out.add_accessor(
                    morph.normals.detach().float().cpu().numpy(), target=_ARRAY_BUFFER)
            targets.append(target)
        primitive["targets"] = targets
        extras["targetNames"] = list(part.morphs.keys())
        defaults = part.extras.get("morph_weights")
        if isinstance(defaults, list) and len(defaults) == len(targets):
            mesh["weights"] = [_number(value, 0.0) for value in defaults]
        else:
            mesh["weights"] = [0.0] * len(targets)
    if part.label:
        extras["label"] = part.label
    if extras:
        mesh["extras"] = extras
    meshes = out.entries("meshes")
    meshes.append(mesh)
    return len(meshes) - 1


def _write_animations(out: GltfBuild, model: Model3D) -> None:
    animations = out.entries("animations")
    for clip in model.animations:
        samplers = []
        channels = []
        for channel in clip.channels:
            if (not 0 <= channel.joint < len(out.joint_nodes) or channel.times.numel() == 0
                    or channel.path not in ("translation", "rotation", "scale")
                    or channel.interpolation not in ("LINEAR", "STEP", "CUBICSPLINE")):
                continue
            times = channel.times.detach().float().cpu().reshape(-1).numpy()
            values = channel.values.detach().float().cpu().numpy()
            expected = times.size * (3 if channel.interpolation == "CUBICSPLINE" else 1)
            width = 4 if channel.path == "rotation" else 3
            if values.shape != (expected, width):
                logger.warning("animation %s: a %s channel has the wrong number of values and was "
                               "not written", clip.name, channel.path)
                continue
            samplers.append({
                "input": out.add_accessor(times, bounds=True),
                "output": out.add_accessor(values),
                "interpolation": channel.interpolation,
            })
            channels.append({"sampler": len(samplers) - 1,
                             "target": {"node": out.joint_nodes[channel.joint], "path": channel.path}})
        for track in model.extras.get("morph_animations", []) or []:
            if not isinstance(track, dict) or track.get("animation") != clip.name:
                continue
            _morph_track(out, model, track, samplers, channels)
        if channels:
            animations.append({"name": clip.name, "samplers": samplers, "channels": channels})
    named = {clip.name for clip in model.animations}
    leftovers: dict[str, list] = {}
    for track in model.extras.get("morph_animations", []) or []:
        if isinstance(track, dict) and track.get("animation") not in named:
            leftovers.setdefault(str(track.get("animation", "morphs")), []).append(track)
    for clip_name, tracks in leftovers.items():
        samplers, channels = [], []
        for track in tracks:
            _morph_track(out, model, track, samplers, channels)
        if channels:
            animations.append({"name": clip_name, "samplers": samplers, "channels": channels})
    if not animations:
        out.json.pop("animations", None)


def _morph_track(out: GltfBuild, model: Model3D, track: dict, samplers: list, channels: list) -> None:
    times = track.get("times")
    values = track.get("values")
    if not isinstance(times, torch.Tensor) or not isinstance(values, torch.Tensor) or times.numel() == 0:
        return
    interpolation = str(track.get("interpolation", "LINEAR"))
    if interpolation not in ("LINEAR", "STEP", "CUBICSPLINE"):
        return
    keys = times.numel() * (3 if interpolation == "CUBICSPLINE" else 1)
    sampler_index = None
    for part_index in track.get("parts", []) or []:
        if not isinstance(part_index, int) or not 0 <= part_index < len(out.part_nodes):
            continue
        node = out.part_nodes[part_index]
        targets = len(model.parts[part_index].morphs)
        if node < 0 or targets != values.shape[-1] or values.numel() != keys * targets:
            continue
        if sampler_index is None:
            samplers.append({
                "input": out.add_accessor(times.detach().float().cpu().reshape(-1).numpy(), bounds=True),
                "output": out.add_accessor(values.detach().float().cpu().reshape(-1).numpy()),
                "interpolation": interpolation,
            })
            sampler_index = len(samplers) - 1
        channels.append({"sampler": sampler_index, "target": {"node": node, "path": "weights"}})


def build(model: Model3D, embed_images: bool = True) -> GltfBuild:
    """Assemble a glTF document for a model.

    Args:
        model: The model.
        embed_images: Store images in the binary buffer; False keeps them as separate files.

    Returns:
        The assembled document.

    Raises:
        GltfError: A part's triangles or skin name data the part or skeleton does not hold.
    """
    out = GltfBuild()
    out._embed = embed_images
    scene_nodes: list[int] = []
    skeleton = model.skeleton if model.skeleton is not None and model.skeleton.count else None
    skin_index = None
    if skeleton is not None:
        _write_skeleton(out, skeleton, scene_nodes)
        if any(part.weights is not None and part.joints is not None for part in model.parts):
            skin = {
                "joints": list(out.joint_nodes),
                "inverseBindMatrices": out.add_accessor(
                    skeleton.inverse_bind.detach().float().cpu().numpy(), kind="MAT4"),
            }
            roots = [out.joint_nodes[joint] for joint in range(skeleton.count) if int(skeleton.parents[joint]) < 0]
            if len(roots) == 1:
                skin["skeleton"] = roots[0]
            out.entries("skins").append(skin)
            skin_index = 0
    nodes = out.entries("nodes")
    for part in model.parts:
        if part.vertex_count == 0:
            out.part_meshes.append(-1)
            out.part_nodes.append(-1)
            out.material_of_part.append(-1)
            continue
        skinned = skin_index is not None and part.joints is not None and part.weights is not None
        mesh = _write_part(out, part, skeleton.count if skeleton is not None else 0,
                           skin_index if skinned else None)
        node: dict = {"name": part.name or "part", "mesh": mesh}
        if skinned:
            node["skin"] = skin_index
        nodes.append(node)
        out.part_meshes.append(mesh)
        out.part_nodes.append(len(nodes) - 1)
        out.material_of_part.append(out.json["meshes"][mesh]["primitives"][0]["material"])
        scene_nodes.append(len(nodes) - 1)
    _write_animations(out, model)
    out.json["scene"] = 0
    out.json["scenes"] = [{"nodes": scene_nodes}]
    meta = _json_safe({key: value for key, value in model.meta.items() if key != "generator"})
    if isinstance(meta.get("copyright"), str):
        out.json["asset"]["copyright"] = meta.pop("copyright")
    if meta:
        out.json["asset"]["extras"] = meta
    for kind in ("nodes", "skins", "meshes", "materials"):
        if not out.json.get(kind):
            out.json.pop(kind, None)
    return out


def write(model: Model3D, binary: bool = True) -> bytes:
    """Write a model as one glTF file.

    Args:
        model: The model.
        binary: True writes a ``.glb``; False writes a self-contained ``.gltf``.

    Returns:
        The file's bytes.

    Raises:
        GltfError: The model holds data that cannot be written.
    """
    assembled = build(model, embed_images=True)
    return assembled.to_glb() if binary else assembled.to_embedded_gltf()
