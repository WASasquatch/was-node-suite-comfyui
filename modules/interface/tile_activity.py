"""The tiles a tiled sampler is working on, announced as they change.

:data:`EVENT` carries ``node_id``, ``prompt_id``, ``phase``, ``tiles``, ``step``, ``steps`` and
``boxes``, each box ``[top, bottom, left, right]`` in fractions of the frame.
"""

from __future__ import annotations

from .. import log
from .channel import executing_node_id, executing_prompt_id

__all__ = ["CAPTION", "DONE", "EVENT", "OVERVIEW", "PREPARE", "SAMPLE", "Announcer"]

logger = log.get_logger("interface.tile_activity")

#: The message the active tiles are announced by.
EVENT = "was-tile-activity"

#: Phases a run announces, in order.
PREPARE = "prepare"
OVERVIEW = "overview"
CAPTION = "caption"
SAMPLE = "sample"
DONE = "done"


class Announcer:
    """Sends the active tiles of one node's run, skipping a message that repeats the last."""

    def __init__(self, boxes: list, node_id=None, prompt_id=None):
        """Bind the announcer to the executing node.

        Args:
            boxes: ``[top, bottom, left, right]`` of every tile as fractions of the frame.
            node_id: The node's id, read from the execution context when left out.
            prompt_id: The prompt's id, read from the execution context when left out.
        """
        self.boxes = [[round(float(v), 5) for v in box] for box in boxes]
        self.node_id = node_id if node_id is not None else executing_node_id()
        self.prompt_id = prompt_id if prompt_id is not None else executing_prompt_id()
        self.last = None

    def __call__(self, phase: str, tiles=(), step: int = 0, steps: int = 0) -> None:
        """Announce the tiles being worked on.

        Args:
            phase: :data:`PREPARE`, :data:`OVERVIEW`, :data:`CAPTION`, :data:`SAMPLE` or
                :data:`DONE`.
            tiles: Indices of the tiles, row by row.
            step: The step being taken, from 0.
            steps: Steps in the run.
        """
        if self.node_id is None:
            return
        active = sorted({int(tile) for tile in tiles})
        key = (phase, tuple(active), int(step))
        if key == self.last:
            return
        self.last = key
        fields = {
            "node_id": str(self.node_id), "prompt_id": self.prompt_id or "", "phase": phase,
            "tiles": active, "step": int(step), "steps": int(steps), "boxes": self.boxes,
        }
        try:
            from server import PromptServer

            PromptServer.instance.send_sync(EVENT, fields)
        except Exception as error:
            logger.debug("the active tiles could not be announced (%s)", error)
