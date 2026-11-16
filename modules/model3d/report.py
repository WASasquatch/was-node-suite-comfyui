"""What a 3D node publishes about the model it produced, for its interface to draw.

Model figures go out as counts; the node's own figures and the earlier model's as tallies.
"""

from __future__ import annotations

__all__ = ["figures", "publish"]

from ..interface import run_result
from ..log import get_logger
from .scene import Model3D

logger = get_logger("model3d.report")


def figures(model: Model3D) -> dict[str, int | float]:
    """The headline numbers of a model.

    Args:
        model: The model to count.

    Returns:
        Parts, vertices, triangles, polygons, quads, joints, morphs and height in metres.
    """
    polygons = quads = 0
    for part in model.parts:
        if part.polygons is not None:
            polygons += part.polygons.count
            quads += int((part.polygons.sizes == 4).sum())
        else:
            polygons += part.triangle_count
    return {
        "parts": len(model.parts),
        "vertices": model.vertex_count,
        "triangles": model.triangle_count,
        "polygons": polygons,
        "quads": quads,
        "joints": model.skeleton.count if model.skeleton is not None else 0,
        "morphs": sum(len(part.morphs) for part in model.parts),
        "height_m": round(model.height(), 4),
    }


#: Figures of the model a node was given, shown beside the ones it produced.
BEFORE_FIGURES = ("vertices", "triangles", "polygons", "quads")


def publish(model: Model3D, summary: str, before: Model3D | dict | None = None,
            counts: dict | None = None, facts: dict | None = None,
            status: str = run_result.OK) -> bool:
    """Publish a run's figures for the node's interface.

    Args:
        model: The model the node produced.
        summary: One line saying what the run did.
        before: The model the node was given, or its `figures`, when the node changed it.
        counts: The node's own named numbers, drawn first among the tallies.
        facts: Named strings.
        status: One of the run result statuses.

    Returns:
        True when a result was stored.
    """
    tallies = [{"name": name, "value": value} for name, value in (counts or {}).items()]
    if before is not None:
        earlier = before if isinstance(before, dict) else figures(before)
        tallies += [{"name": f"{name} before", "value": earlier[name]} for name in BEFORE_FIGURES
                    if name in earlier]
    return run_result.publish(
        status=status, summary=summary, counts=figures(model), tallies=tallies,
        facts=dict(facts or {}),
    )
