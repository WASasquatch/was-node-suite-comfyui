"""Moving a whole model: turning, uniform scaling and placing it, skeleton and springs included.

A transform is a rotation, one scale factor and a move. Joint frames keep unit scale; joint
positions, collider sizes and spring radii are scaled instead.
"""

from __future__ import annotations

__all__ = [
    "FACES_FORWARD_KEY", "facing_angle", "face_forward", "place", "rotation_matrix", "scale_to_height",
    "similarity", "transform",
]

import math

import torch

from ..log import get_logger
from .scene import Model3D

logger = get_logger("model3d.orient")

#: Where `place` puts the model's origin.
ORIGINS = ("base", "centre", "as is")

#: Model meta key set True on a figure known to face +Z; a turn that moves +Z removes it.
FACES_FORWARD_KEY = "faces_forward"

#: Humanoid bone pairs whose left-minus-right direction is the character's left, best first.
_SIDE_PAIRS = (
    ("leftUpperLeg", "rightUpperLeg"),
    ("leftUpperArm", "rightUpperArm"),
    ("leftShoulder", "rightShoulder"),
    ("leftFoot", "rightFoot"),
    ("leftHand", "rightHand"),
)


def rotation_matrix(x_deg: float = 0.0, y_deg: float = 0.0, z_deg: float = 0.0) -> torch.Tensor:
    """A rotation turning about X, then Y, then Z, in degrees.

    Returns:
        ``(3, 3)`` float64 matrix.
    """
    def axis(index: int, degrees: float) -> torch.Tensor:
        angle = math.radians(degrees)
        c, s = math.cos(angle), math.sin(angle)
        a, b = [(1, 2), (2, 0), (0, 1)][index]
        matrix = torch.eye(3, dtype=torch.float64)
        matrix[a, a], matrix[a, b], matrix[b, a], matrix[b, b] = c, -s, s, c
        return matrix

    return axis(2, z_deg) @ axis(1, y_deg) @ axis(0, x_deg)


def similarity(rotation: torch.Tensor | None = None, scale: float = 1.0,
               translation: tuple[float, float, float] | torch.Tensor = (0.0, 0.0, 0.0)) -> torch.Tensor:
    """The ``(4, 4)`` float64 matrix of ``x -> scale * rotation @ x + translation``."""
    matrix = torch.eye(4, dtype=torch.float64)
    turn = torch.eye(3, dtype=torch.float64) if rotation is None else rotation.double()
    matrix[:3, :3] = float(scale) * turn
    matrix[:3, 3] = torch.as_tensor(translation, dtype=torch.float64)
    return matrix


def _split(matrix: torch.Tensor) -> tuple[torch.Tensor, float, torch.Tensor]:
    """Rotation, scale and translation of a similarity matrix."""
    linear = matrix[:3, :3].double()
    scale = float(torch.linalg.det(linear)) ** (1.0 / 3.0) if float(torch.linalg.det(linear)) > 0 else 0.0
    if scale <= 0 or not math.isfinite(scale):
        raise ValueError("the transform must keep the model's handedness and have a scale above 0")
    rotation = linear / scale
    if float((rotation @ rotation.T - torch.eye(3, dtype=torch.float64)).abs().max()) > 1e-4:
        raise ValueError("the transform must be a rotation, one scale and a move; it stretches "
                         "or shears the model")
    return rotation, scale, matrix[:3, 3].double()


def _quaternion_of(rotation: torch.Tensor) -> torch.Tensor:
    """``(4,)`` ``xyzw`` quaternion of a ``(3, 3)`` rotation."""
    m = rotation.double()
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    return torch.tensor([float(v) for v in q], dtype=torch.float64)


def _quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of ``xyzw`` quaternions, ``a`` applied after ``b``; ``b`` may be ``(T, 4)``."""
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dim=-1)


def transform(model: Model3D, matrix: torch.Tensor) -> Model3D:
    """Apply a similarity to a model in place: geometry, skeleton, root animation and springs.

    Args:
        model: The model to move.
        matrix: ``(4, 4)`` similarity, as `similarity` builds.

    Returns:
        The same model.

    Raises:
        ValueError: The matrix mirrors, stretches or shears.
    """
    rotation, scale, translation = _split(matrix)
    if float(rotation[2, 2]) < 1.0 - 1e-6:
        model.meta.pop(FACES_FORWARD_KEY, None)
    turn32 = rotation.float()
    for part in model.parts:
        if part.vertex_count:
            part.vertices = ((part.vertices.double() @ rotation.T) * scale + translation).float()
        if part.normals is not None:
            part.normals = torch.nn.functional.normalize(part.normals.float() @ turn32.T, dim=-1)
        if part.tangents is not None:
            turned = part.tangents[:, :3].float() @ turn32.T
            part.tangents = torch.cat([turned, part.tangents[:, 3:]], dim=1)
        for morph in part.morphs.values():
            morph.positions = (morph.positions.double() @ rotation.T * scale).float()
            if morph.normals is not None:
                morph.normals = morph.normals.float() @ turn32.T
    skeleton = model.skeleton
    if skeleton is not None and skeleton.count:
        full = matrix.double()
        unscale = torch.diag(torch.tensor([1.0 / scale] * 3 + [1.0], dtype=torch.float64))
        rescale = torch.diag(torch.tensor([scale] * 3 + [1.0], dtype=torch.float64))
        inverse = torch.linalg.inv(full)
        skeleton.rest = (full @ skeleton.rest.double() @ unscale).float()
        skeleton.inverse_bind = (rescale @ skeleton.inverse_bind.double() @ inverse).float()
        roots = {joint for joint in range(skeleton.count) if int(skeleton.parents[joint]) < 0}
        for animation in model.animations:
            for channel in animation.channels:
                values = channel.values.double()
                if channel.path == "translation":
                    if channel.joint in roots:
                        if channel.interpolation == "CUBICSPLINE":
                            keys = values.shape[0] // 3
                            offset = torch.zeros_like(values)
                            offset[keys:2 * keys] = translation
                            values = values @ rotation.T * scale + offset
                        else:
                            values = values @ rotation.T * scale + translation
                    else:
                        values = values * scale
                elif channel.path == "rotation" and channel.joint in roots:
                    values = _quaternion_multiply(_quaternion_of(rotation), values)
                channel.values = values.float()
    springs = model.springs
    if springs is not None:
        for collider in springs.colliders:
            collider.offset = collider.offset.float() * scale
            collider.tail = collider.tail.float() * scale
            collider.radius = float(collider.radius) * scale
        for chain in springs.chains:
            chain.hit_radius = float(chain.hit_radius) * scale
    look_at = model.extras.get("vrm_look_at")
    if isinstance(look_at, dict) and isinstance(look_at.get("offsetFromHeadBone"), (list, tuple)):
        look_at["offsetFromHeadBone"] = [float(value) * scale for value in look_at["offsetFromHeadBone"]]
    return model


def facing_angle(model: Model3D) -> float | None:
    """Turn about Y, in degrees, that puts a humanoid model's left side along +X.

    Returns:
        Degrees to turn, or None when the model has no skeleton to read a facing from.
    """
    skeleton = model.skeleton
    if skeleton is None or not skeleton.count:
        return None
    humanoid = skeleton.humanoid
    if not humanoid:
        from ..character.bones import guess_humanoid
        try:
            humanoid = guess_humanoid(skeleton.names, skeleton.parents, skeleton.heads())
        except ValueError:
            return None
    heads = skeleton.heads().double()
    for left, right in _SIDE_PAIRS:
        if left in humanoid and right in humanoid:
            side = heads[humanoid[left]] - heads[humanoid[right]]
            if float(side[0] ** 2 + side[2] ** 2) > 1e-12:
                return -math.degrees(math.atan2(float(-side[2]), float(side[0])))
    return None


def face_forward(model: Model3D) -> float:
    """Turn a humanoid model about Y, around its hips, until it faces +Z.

    Returns:
        Degrees turned, 0 when the model already faced +Z or has no skeleton to read.
    """
    angle = facing_angle(model)
    if angle is None:
        return 0.0
    if abs(angle) >= 1e-3:
        low, high = model.bounds()
        pivot = ((low + high) / 2).double()
        pivot[1] = 0.0
        turn = rotation_matrix(0.0, angle, 0.0)
        matrix = similarity(translation=pivot) @ similarity(turn) @ similarity(translation=-pivot)
        transform(model, matrix)
    model.meta[FACES_FORWARD_KEY] = True
    return angle if abs(angle) >= 1e-3 else 0.0


def place(model: Model3D, origin: str = "base") -> Model3D:
    """Move a model so its origin sits where ``origin`` says.

    Args:
        model: The model to move in place.
        origin: ``"base"`` centres X and Z and stands the model on Y = 0, ``"centre"`` puts
            the middle of its box at the origin, ``"as is"`` leaves it.

    Returns:
        The same model.
    """
    if origin == "as is" or not model.vertex_count:
        return model
    low, high = model.bounds()
    centre = ((low + high) / 2).double()
    if origin == "base":
        centre[1] = float(low[1])
    return transform(model, similarity(translation=-centre))


def scale_to_height(model: Model3D, height: float) -> float:
    """Scale a model uniformly about its feet until it stands ``height`` metres tall.

    Returns:
        The scale factor applied, 1 when the model has no height or ``height`` is 0 or less.
    """
    current = model.height()
    if height <= 0 or current <= 1e-9:
        return 1.0
    factor = float(height) / current
    low, high = model.bounds()
    foot = ((low + high) / 2).double()
    foot[1] = float(low[1])
    matrix = similarity(translation=foot) @ similarity(scale=factor) @ similarity(translation=-foot)
    transform(model, matrix)
    return factor
