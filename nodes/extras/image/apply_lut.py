"""Grading an image batch through a colour lookup table."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat.sockets import require_input
from ....modules.compat.types import LUT
from ....modules.image import lut as tables

REQUIRES = "extras"

#: Cube size a table with no usable size of its own is resampled to.
FALLBACK_SIZE = 33

#: Most worker threads the parallel path will start, whatever was asked for.
MAX_WORKERS = 64


class ApplyLUT(io.ComfyNode):
    """Grade images through a colour lookup table."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASNSApplyLUT",
            display_name="Apply LUT",
            search_aliases=["WASNSApplyLUT", "WAS Apply LUT", "lut", "grade", "cube", "color"],
            category="WAS Suite/Image/LUT",
            description=(
                "Grade pictures through a colour lookup table from Load LUT or LUT Blender. "
                "Each pixel's colour is looked up in the table and blended between the eight "
                "nearest entries, which is how a film look, a camera profile or a corrective "
                "grade is applied. Blend the result back over the original to use the look "
                "at partial strength."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The pictures to grade. The whole batch is graded at once, so a "
                        "video sequence gets exactly the same treatment frame to frame."
                    ),
                ),
                LUT.Input(
                    "lut",
                    tooltip=(
                        "The colour lookup table to apply, from Load LUT or LUT Blender. A "
                        "table stored as curves rather than a cube is converted first."
                    ),
                ),
                io.Float.Input(
                    "strength", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip=(
                        "How far the graded result is mixed over the original. 0.0 returns "
                        "the pictures untouched, 1.0 applies the look outright, 0.5 applies "
                        "it at half strength."
                    ),
                ),
                io.Boolean.Input(
                    "use_threads", default=False,
                    tooltip=(
                        "Whether to grade the frames of a batch on several CPU threads at "
                        "once. Leave it off on a GPU, where one pass over the whole batch is "
                        "already fastest; turn it on for a long CPU-bound sequence."
                    ),
                ),
                io.Int.Input(
                    "threads", default=0, min=0, max=64, step=1,
                    tooltip=(
                        "How many worker threads to use. 0 picks one per CPU core, up to the "
                        "number of frames. Ignored while use_threads is off."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="image",
                    tooltip="The graded pictures, same size and batch order as the input.",
                ),
            ],
        )

    @classmethod
    def execute(cls, image, lut, strength, use_threads=False, threads=0) -> io.NodeOutput:
        """Grade the batch.

        Raises:
            ValueError: Nothing is connected to the lut input, or the LUT holds no table to
                apply.
        """
        import os
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        import torch
        from comfy.utils import ProgressBar

        require_input(lut, "Apply LUT", "lut", "table", "Load LUT or LUT Blender")

        size = lut.size() if lut.size() > 1 else FALLBACK_SIZE
        cube = tables.convert_to_3d(lut, size)
        frames = int(image.shape[0])
        progress = ProgressBar(frames)

        def grade(source):
            graded = tables.apply_lut_3d(
                source, cube.table_3d, cube.domain_min, cube.domain_max
            ).clamp(0, 1)
            if strength < 1.0:
                graded = source * (1.0 - strength) + graded * strength
            return graded.clamp(0, 1)

        if not use_threads or frames <= 1:
            result = grade(image)
            progress.update(frames)
            return io.NodeOutput(result)

        workers = int(threads) if int(threads) > 0 else min(os.cpu_count() or 1, frames)
        workers = max(1, min(workers, MAX_WORKERS))

        results = [None] * frames
        lock = threading.Lock()

        def work(index):
            graded = grade(image[index:index + 1])
            with lock:
                progress.update(1)
            return index, graded

        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = [pool.submit(work, index) for index in range(frames)]
            for finished in as_completed(pending):
                index, graded = finished.result()
                results[index] = graded

        return io.NodeOutput(torch.cat(results, dim=0))
