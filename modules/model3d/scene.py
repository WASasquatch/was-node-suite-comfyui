"""A 3D model in memory: parts, materials, a skeleton, springs, animations and metadata.

Metres, right-handed, Y up, facing +Z; ``v`` runs down the image. Tensors are on the CPU,
with vertices split where an attribute differs, one row per vertex.
"""

from __future__ import annotations

__all__ = [
    "Animation", "Channel", "Collider", "Expression", "Material", "Model3D", "MorphTarget",
    "Part", "Polygons", "Skeleton", "SpringChain", "Springs", "Texture",
]

import copy
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class Texture:
    """One image a material samples.

    Attributes:
        image: ``(H, W, C)`` float32 in ``[0, 1]``, ``C`` 3 or 4. Colour maps hold sRGB codes,
            data maps (normal, metallic-roughness, occlusion) hold linear values.
        name: Name carried from the file, or empty.
        wrap_s: ``"repeat"``, ``"clamp"`` or ``"mirror"`` along ``u``.
        wrap_t: The same along ``v``.
        mime: Encoding preferred when written, ``"image/png"`` or ``"image/jpeg"``.
    """

    image: torch.Tensor
    name: str = ""
    wrap_s: str = "repeat"
    wrap_t: str = "repeat"
    mime: str = "image/png"

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)`` in pixels."""
        return int(self.image.shape[1]), int(self.image.shape[0])


@dataclass
class Material:
    """A metallic-roughness material, as glTF defines one.

    Attributes:
        name: Material name.
        base_color: RGBA factor, multiplied into ``base_color_texture``.
        base_color_texture: Colour map, or None.
        metallic: Metalness factor.
        roughness: Roughness factor.
        metallic_roughness_texture: Roughness in G and metalness in B, or None.
        normal_texture: Tangent-space normal map, OpenGL convention (+Y up), or None.
        normal_scale: Strength of the normal map.
        occlusion_texture: Ambient occlusion in R, or None.
        occlusion_strength: Strength of the occlusion map.
        emissive: RGB emissive factor.
        emissive_texture: Emissive map, or None.
        alpha_mode: ``"OPAQUE"``, ``"MASK"`` or ``"BLEND"``.
        alpha_cutoff: Threshold used by ``"MASK"``.
        double_sided: Whether back faces are drawn.
        unlit: Whether the material ignores lighting.
        extras: Format-specific settings carried through untouched, such as MToon parameters
            under ``"mtoon"``.
    """

    name: str = "material"
    base_color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    base_color_texture: Texture | None = None
    metallic: float = 1.0
    roughness: float = 1.0
    metallic_roughness_texture: Texture | None = None
    normal_texture: Texture | None = None
    normal_scale: float = 1.0
    occlusion_texture: Texture | None = None
    occlusion_strength: float = 1.0
    emissive: tuple[float, float, float] = (0.0, 0.0, 0.0)
    emissive_texture: Texture | None = None
    alpha_mode: str = "OPAQUE"
    alpha_cutoff: float = 0.5
    double_sided: bool = False
    unlit: bool = False
    extras: dict[str, Any] = field(default_factory=dict)

    def textures(self) -> dict[str, Texture]:
        """Every texture the material holds, by slot name."""
        slots = {
            "base_color": self.base_color_texture,
            "metallic_roughness": self.metallic_roughness_texture,
            "normal": self.normal_texture,
            "occlusion": self.occlusion_texture,
            "emissive": self.emissive_texture,
        }
        return {name: texture for name, texture in slots.items() if texture is not None}


@dataclass
class Polygons:
    """Faces with more than three corners, such as the quads of a retopologised mesh.

    Attributes:
        sizes: ``(P,)`` int64, the corner count of each polygon.
        indices: ``(sizes.sum(),)`` int64 vertex indices, polygon after polygon, counter-clockwise
            seen from the front.
        triangle_polygon: ``(M,)`` int64, the polygon each row of the part's ``faces`` came from.
    """

    sizes: torch.Tensor
    indices: torch.Tensor
    triangle_polygon: torch.Tensor

    @property
    def count(self) -> int:
        """Number of polygons."""
        return int(self.sizes.shape[0])

    def quad_fraction(self) -> float:
        """Share of polygons that have four corners."""
        return float((self.sizes == 4).float().mean()) if self.count else 0.0


@dataclass
class MorphTarget:
    """A shape key: offsets added to a part's vertices at full weight.

    Attributes:
        positions: ``(N, 3)`` float32 position offsets.
        normals: ``(N, 3)`` float32 normal offsets, or None.
    """

    positions: torch.Tensor
    normals: torch.Tensor | None = None


@dataclass
class Part:
    """One mesh of a model, drawn with one material.

    Attributes:
        name: Part name.
        vertices: ``(N, 3)`` float32 positions in model space.
        faces: ``(M, 3)`` int64 triangles, counter-clockwise seen from the front.
        material: The material it is drawn with.
        label: Role in a character, such as ``"head"``, ``"hair"``, ``"body"`` or ``""``.
        polygons: Quads or n-gons the triangles came from, or None.
        uvs: ``(N, 2)`` float32 texture coordinates, or None.
        normals: ``(N, 3)`` float32 unit normals, or None.
        tangents: ``(N, 4)`` float32 tangents with handedness in ``w``, or None.
        colors: ``(N, 4)`` float32 linear RGBA vertex colours, or None.
        joints: ``(N, K)`` int64 joint indices into the model's skeleton, or None.
        weights: ``(N, K)`` float32 skin weights, rows summing to 1, or None.
        morphs: Shape keys by name, in file order.
        extras: Format-specific data carried through untouched.
    """

    name: str
    vertices: torch.Tensor
    faces: torch.Tensor
    material: Material = field(default_factory=Material)
    label: str = ""
    polygons: Polygons | None = None
    uvs: torch.Tensor | None = None
    normals: torch.Tensor | None = None
    tangents: torch.Tensor | None = None
    colors: torch.Tensor | None = None
    joints: torch.Tensor | None = None
    weights: torch.Tensor | None = None
    morphs: dict[str, MorphTarget] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    #: Per-vertex attribute names, in the order they are listed.
    VERTEX_ATTRIBUTES = ("uvs", "normals", "tangents", "colors", "joints", "weights")

    @property
    def vertex_count(self) -> int:
        """Number of vertices."""
        return int(self.vertices.shape[0])

    @property
    def triangle_count(self) -> int:
        """Number of triangles."""
        return int(self.faces.shape[0])

    def vertex_attributes(self) -> dict[str, torch.Tensor]:
        """Every per-vertex tensor the part holds besides positions, by attribute name."""
        held = {name: getattr(self, name) for name in self.VERTEX_ATTRIBUTES}
        return {name: tensor for name, tensor in held.items() if tensor is not None}

    def bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(min, max)`` corners of the part's box, each ``(3,)``."""
        if self.vertex_count == 0:
            zero = torch.zeros(3)
            return zero, zero.clone()
        return self.vertices.min(dim=0).values, self.vertices.max(dim=0).values

    def copy(self) -> Part:
        """A copy sharing no tensor with this part."""
        return copy.deepcopy(self)


@dataclass
class Skeleton:
    """Joints, their hierarchy and the pose the mesh was bound in.

    Attributes:
        names: Joint names.
        parents: ``(J,)`` int64 parent index of each joint, ``-1`` for a root. A parent always
            comes before its children.
        rest: ``(J, 4, 4)`` float32 joint transforms in model space at the bind pose.
        inverse_bind: ``(J, 4, 4)`` float32 inverse bind matrices.
        humanoid: VRM humanoid bone name, such as ``"hips"`` or ``"leftUpperArm"``, against the
            joint index it names.
    """

    names: list[str]
    parents: torch.Tensor
    rest: torch.Tensor
    inverse_bind: torch.Tensor
    humanoid: dict[str, int] = field(default_factory=dict)

    @property
    def count(self) -> int:
        """Number of joints."""
        return len(self.names)

    def index(self, name: str) -> int:
        """Index of the joint called ``name``, or ``-1``."""
        try:
            return self.names.index(name)
        except ValueError:
            return -1

    def local_rest(self) -> torch.Tensor:
        """``(J, 4, 4)`` rest transforms relative to each joint's parent."""
        local = self.rest.clone()
        for joint in range(self.count):
            parent = int(self.parents[joint])
            if parent >= 0:
                local[joint] = torch.linalg.inv(self.rest[parent]) @ self.rest[joint]
        return local

    def heads(self) -> torch.Tensor:
        """``(J, 3)`` joint positions in model space at the bind pose."""
        return self.rest[:, :3, 3].clone()


@dataclass
class Collider:
    """A sphere or capsule a spring chain collides with.

    Attributes:
        joint: Joint the collider follows.
        shape: ``"sphere"`` or ``"capsule"``.
        offset: ``(3,)`` centre, or capsule start, in the joint's space.
        tail: ``(3,)`` capsule end in the joint's space, ignored for a sphere.
        radius: Radius in metres.
    """

    joint: int
    shape: str
    offset: torch.Tensor
    tail: torch.Tensor
    radius: float


@dataclass
class SpringChain:
    """Joints simulated as one swinging chain, as VRM spring bones define one.

    Attributes:
        name: Chain name, such as ``"hair_front"``.
        joints: Joint indices from the root of the chain to its tip.
        stiffness: Pull back towards the rest pose.
        drag: Loss of speed each step, ``0`` to ``1``.
        gravity_power: Strength of gravity on the chain.
        gravity_dir: ``(3,)`` gravity direction in model space.
        hit_radius: Radius of each joint against colliders, in metres.
        collider_groups: Indices into `Springs.collider_groups` the chain collides with.
    """

    name: str
    joints: list[int]
    stiffness: float = 1.0
    drag: float = 0.4
    gravity_power: float = 0.0
    gravity_dir: torch.Tensor = field(default_factory=lambda: torch.tensor([0.0, -1.0, 0.0]))
    hit_radius: float = 0.02
    collider_groups: list[int] = field(default_factory=list)


@dataclass
class Springs:
    """Spring chains, the colliders they hit and how colliders are grouped.

    Attributes:
        chains: The simulated chains.
        colliders: Every collider.
        collider_groups: Named groups, each a list of indices into ``colliders``.
        group_names: Name of each collider group.
    """

    chains: list[SpringChain] = field(default_factory=list)
    colliders: list[Collider] = field(default_factory=list)
    collider_groups: list[list[int]] = field(default_factory=list)
    group_names: list[str] = field(default_factory=list)


@dataclass
class Channel:
    """One animated property of one joint.

    Attributes:
        joint: Joint index.
        path: ``"translation"``, ``"rotation"`` (quaternion ``xyzw``) or ``"scale"``.
        times: ``(T,)`` float32 key times in seconds.
        values: ``(T, 3)`` or ``(T, 4)`` float32 key values in the joint's parent space;
            ``(3T, K)`` in-tangent, value, out-tangent rows for ``"CUBICSPLINE"``.
        interpolation: ``"LINEAR"``, ``"STEP"`` or ``"CUBICSPLINE"``.
    """

    joint: int
    path: str
    times: torch.Tensor
    values: torch.Tensor
    interpolation: str = "LINEAR"


@dataclass
class Animation:
    """A named clip of joint channels.

    Attributes:
        name: Clip name.
        channels: Animated joint properties.
    """

    name: str
    channels: list[Channel] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """Last key time across every channel, in seconds."""
        ends = [float(channel.times[-1]) for channel in self.channels if channel.times.numel()]
        return max(ends, default=0.0)


@dataclass
class Expression:
    """A facial expression driving shape keys, as VRM defines one.

    Attributes:
        name: Preset name such as ``"happy"`` or ``"blinkLeft"``, or a custom name.
        binds: ``(part index, morph name, weight)`` triples.
        is_binary: Whether the expression snaps to 0 or 1.
        override_blink: ``"none"``, ``"block"`` or ``"blend"``.
        override_look_at: The same for gaze.
        override_mouth: The same for the mouth.
    """

    name: str
    binds: list[tuple[int, str, float]] = field(default_factory=list)
    is_binary: bool = False
    override_blink: str = "none"
    override_look_at: str = "none"
    override_mouth: str = "none"


@dataclass
class Model3D:
    """A whole model: what a 3D file holds once read.

    Attributes:
        parts: The meshes, each with its own material.
        skeleton: Joints and bind pose, or None for a static model.
        springs: Spring bone setup, or None.
        animations: Joint animation clips.
        expressions: Facial expressions by name.
        meta: Title, authors, licence and the rest of the model's metadata. VRM meta fields
            keep their VRM 1.0 names, such as ``"name"``, ``"authors"`` and
            ``"commercialUsage"``.
        source_format: Format the model was read from, such as ``"glb"`` or ``"fbx"``, or empty.
        extras: Format-specific data carried through untouched.
    """

    parts: list[Part] = field(default_factory=list)
    skeleton: Skeleton | None = None
    springs: Springs | None = None
    animations: list[Animation] = field(default_factory=list)
    expressions: dict[str, Expression] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    source_format: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def triangle_count(self) -> int:
        """Triangles across every part."""
        return sum(part.triangle_count for part in self.parts)

    @property
    def vertex_count(self) -> int:
        """Vertices across every part."""
        return sum(part.vertex_count for part in self.parts)

    @property
    def is_rigged(self) -> bool:
        """Whether the model has a skeleton and at least one skinned part."""
        return self.skeleton is not None and any(part.weights is not None for part in self.parts)

    def bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(min, max)`` corners of the box around every part, each ``(3,)``."""
        filled = [part.bounds() for part in self.parts if part.vertex_count]
        if not filled:
            zero = torch.zeros(3)
            return zero, zero.clone()
        lows = torch.stack([low for low, _ in filled])
        highs = torch.stack([high for _, high in filled])
        return lows.min(dim=0).values, highs.max(dim=0).values

    def height(self) -> float:
        """Extent along Y, in metres."""
        low, high = self.bounds()
        return float(high[1] - low[1])

    def parts_labelled(self, label: str) -> list[int]:
        """Indices of the parts carrying ``label``."""
        return [index for index, part in enumerate(self.parts) if part.label == label]

    def copy(self) -> Model3D:
        """A copy sharing no tensor with this model."""
        return copy.deepcopy(self)
