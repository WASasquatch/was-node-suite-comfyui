"""Reading the model input of the 3D nodes and building their two outputs.

The input takes a `scene.Model3D`, a core ``MESH`` or any core 3D file. Every 3D node outputs
the model and the same model as a ``MESH``.
"""

from __future__ import annotations

__all__ = ["mesh_wanted", "node_output", "to_model"]

from .scene import Model3D


def to_model(value) -> Model3D:
    """A wired value as a model the node may change freely.

    Args:
        value: A `scene.Model3D`, a core ``MESH`` or a core ``File3D``.

    Returns:
        A new model sharing no tensor with the value.

    Raises:
        ValueError: Nothing is wired, the value is none of the accepted kinds, or the file it
            names cannot be read.
    """
    if value is None:
        raise ValueError("no model is wired; connect a 3D model, a mesh or a 3D file")
    if isinstance(value, Model3D):
        return value.copy()
    if hasattr(value, "get_bytes") and hasattr(value, "is_disk_backed"):
        from . import files
        return files.load_file3d(value)
    if hasattr(value, "vertices") and hasattr(value, "faces"):
        from .core_mesh import from_core_mesh
        return from_core_mesh(value)
    raise ValueError(f"a value of type {type(value).__name__} was wired where a model belongs; connect a 3D "
                     "model, a mesh or a GLB, glTF, VRM, FBX or OBJ file")


def mesh_wanted(prompt, unique_id, output: int = 1) -> bool:
    """Whether any node in the queued graph reads one of this node's outputs.

    Args:
        prompt: The queued graph, as the node's hidden ``prompt`` input gives it, or None.
        unique_id: The node's hidden ``unique_id``.
        output: Index of the output asked about, 1 for ``mesh``.

    Returns:
        True when a link reads it, or when the graph cannot be read.
    """
    if not isinstance(prompt, dict) or unique_id is None:
        return True
    source = str(unique_id)
    for entry in prompt.values():
        node_inputs = entry.get("inputs") if isinstance(entry, dict) else None
        if not isinstance(node_inputs, dict):
            continue
        for value in node_inputs.values():
            if (isinstance(value, (list, tuple)) and len(value) == 2 and str(value[0]) == source
                    and value[1] == output):
                return True
    return False


def node_output(model: Model3D, mesh: bool = True) -> tuple:
    """The values of a 3D node's ``model_3d`` and ``mesh`` outputs.

    Args:
        model: The model the node produced.
        mesh: Build the core mesh; False returns None in its place, for an unwired output.

    Returns:
        ``(model, mesh)``.

    Raises:
        ValueError: The model has no triangles and a mesh was asked for.
    """
    from .core_mesh import to_core_mesh
    return model, (to_core_mesh(model) if mesh else None)
