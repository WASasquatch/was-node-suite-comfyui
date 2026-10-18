"""The tiles and captions a tiled sampler is working on, announced as they change.

:data:`EVENT` carries ``node_id``, ``prompt_id``, ``phase``, ``tiles``, ``step``, ``steps``,
``visited``, ``stage``, ``stages`` and ``boxes`` as ``[top, bottom, left, right]`` fractions; a
caption adds ``written`` and ``line``.
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

    def __init__(self, boxes: list, node_id=None, prompt_id=None, stage=(1, 1)):
        """Bind the announcer to the executing node.

        Args:
            boxes: ``[top, bottom, left, right]`` of every tile as fractions of the frame.
            node_id: The node's id, read from the execution context when left out.
            prompt_id: The prompt's id, read from the execution context when left out.
            stage: ``(pass, passes)`` of a staged run, sent with every message.
        """
        self.stage = (int(stage[0]), int(stage[1]))
        self.boxes = [[round(float(v), 5) for v in box] for box in boxes]
        self.node_id = node_id if node_id is not None else executing_node_id()
        self.prompt_id = prompt_id if prompt_id is not None else executing_prompt_id()
        self.last = None
        self.step = 0
        self.steps = 0
        self.visited = set()

    def begin(self, steps: int) -> None:
        """Start counting a sampling run.

        Args:
            steps: Steps in the run.
        """
        self.steps, self.step, self.visited = int(steps), 0, set()

    def finished(self, step: int) -> None:
        """Note a step the sampler finished, which opens the next.

        Args:
            step: The finished step, from 0.
        """
        self.step, self.visited = min(int(step) + 1, max(self.steps - 1, 0)), set()

    def __call__(self, phase: str, tiles=(), step=None, steps=None) -> None:
        """Announce the tiles being worked on.

        Args:
            phase: :data:`PREPARE`, :data:`OVERVIEW`, :data:`CAPTION`, :data:`SAMPLE` or
                :data:`DONE`.
            tiles: Indices of the tiles, row by row.
            step: The step being taken from 0, the counted one when left out.
            steps: Steps in the run, the counted number when left out.
        """
        if self.node_id is None:
            return
        active = sorted({int(tile) for tile in tiles})
        step = self.step if step is None else int(step)
        steps = self.steps if steps is None else int(steps)
        if phase == SAMPLE:
            self.visited.update(active)
        key = (phase, tuple(active), step)
        if key == self.last:
            return
        self.last = key
        self._send(phase, active, step, steps)

    def wrote(self, tile: int, line: str) -> None:
        """Announce a caption the moment it is written, under the phase last announced.

        Args:
            tile: The tile it describes, from 0, or ``-1`` for the overview.
            line: The caption as the captions output writes it.
        """
        if self.node_id is None:
            return
        phase, active, step = self.last or (CAPTION, (), self.step)
        self._send(phase, list(active), step, self.steps, {"written": int(tile), "line": str(line)})

    def _send(self, phase: str, active: list, step: int, steps: int, extra=None) -> None:
        """Send one message to the node's panel."""
        fields = {
            "node_id": str(self.node_id), "prompt_id": self.prompt_id or "", "phase": phase,
            "tiles": active, "step": step, "steps": steps, "boxes": self.boxes,
            "visited": len(self.visited) if phase == SAMPLE else 0,
            "stage": self.stage[0], "stages": self.stage[1],
            **(extra or {}),
        }
        try:
            from server import PromptServer

            PromptServer.instance.send_sync(EVENT, fields)
        except Exception as error:
            logger.debug("the active tiles could not be announced (%s)", error)
