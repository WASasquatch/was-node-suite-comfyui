"""Handing memory back to the compute device partway through a run.

Every memory figure in this module is gigabytes of the device ComfyUI computes on.
"""

from __future__ import annotations

from comfy_api.latest import io, ui

from ...modules import log

logger = log.get_logger("nodes.utility")

#: Bytes in one gigabyte.
GIGABYTE = 1024 ** 3


def gigabytes(value) -> float:
    """A byte count as gigabytes.

    Args:
        value: A count of bytes.

    Returns:
        Gigabytes rounded to three decimals, and 0.0 for a count that cannot be read.
    """
    try:
        return round(float(value) / GIGABYTE, 3)
    except (TypeError, ValueError):
        return 0.0


def usage(management, device) -> tuple[float, float, float]:
    """How much of a device's memory is taken.

    Args:
        management: ComfyUI's ``comfy.model_management``.
        device: The device to measure.

    Returns:
        Gigabytes in use, gigabytes free and gigabytes the device holds, each 0.0 where
        the device reports no figures.
    """
    try:
        free = gigabytes(management.get_free_memory(device))
        total = gigabytes(management.get_total_memory(device))
    except Exception as error:
        logger.warning("%s reports no memory figures: %s", device, error)
        return 0.0, 0.0, 0.0
    return round(max(total - free, 0.0), 3), free, total


def reading(label: str, measured: tuple[float, float, float]) -> str:
    """One memory reading written out as a line.

    Args:
        label: Which reading it is, as ``before``.
        measured: What :func:`usage` answered.

    Returns:
        The line, naming the figures or saying they are unavailable.
    """
    used, free, total = measured
    if total <= 0.0:
        return f"{label}: no memory figures on this device"
    return f"{label}: {used:.2f} GB used, {free:.2f} GB free of {total:.2f} GB"


def summary(device, steps: list[str], before, after, freed: float) -> str:
    """The whole run written out for the node and for a text socket.

    Args:
        device: The device that was measured.
        steps: What was done, in the order it happened.
        before: What :func:`usage` answered first.
        after: What :func:`usage` answered afterwards.
        freed: Gigabytes the run handed back.

    Returns:
        Five lines: the device, the reading before, what was done, the reading after and
        the amount freed.
    """
    return "\n".join(
        [
            f"Free Memory on {device}",
            reading("before", before),
            "did: " + ("; ".join(steps) if steps else "nothing, every switch is false"),
            reading("after", after),
            f"freed: {freed:.2f} GB",
        ]
    )


class FreeMemory(io.ComfyNode):
    """Unload models, empty the cache and collect garbage in the middle of a graph."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        template = io.MatchType.Template("free_memory_passthrough")
        return io.Schema(
            node_id="WASFreeMemory",
            display_name="Free Memory",
            search_aliases=[
                "WASFreeMemory",
                "Free Memory",
                "free vram",
                "clear vram",
                "unload models",
                "empty cache",
                "out of memory",
                "oom",
                "garbage collect",
            ],
            category="WAS Suite/Utilities",
            description=(
                "Hand memory back to the graphics card partway through a run. ComfyUI can "
                "only be asked to free memory from its own menu, which a running graph "
                "cannot reach, so a chain that loads, upscales and then encodes video can "
                "run out on the last stage while the first two are still resident. Wire "
                "the stage that has finished into passthrough and the stage that needs the "
                "room after it, and the freeing happens between the two. Reports what the "
                "device held before and after, so the effect is a number rather than a "
                "guess. It frees where something upstream ran, leaving a graph that has "
                "not changed on the cache; always_run frees on every queue instead. "
                "Harmless on a machine with no graphics card."
            ),
            inputs=[
                io.MatchType.Input(
                    "passthrough",
                    template=template,
                    optional=True,
                    tooltip=(
                        "Anything at all: an image, a model, a latent, text. It comes back "
                        "out unchanged once the freeing is done, which is what pins the "
                        "free to a point in the chain instead of leaving it to happen "
                        "whenever. Unwired, the node frees on its own and wants always_run."
                    ),
                ),
                io.Boolean.Input(
                    "unload_models",
                    default=True,
                    tooltip=(
                        "true hands every loaded checkpoint, VAE, CLIP and ControlNet "
                        "back; false leaves them where they are. This is what frees the "
                        "most. They load again by themselves when a node next asks for "
                        "one, which costs the seconds that load took."
                    ),
                ),
                io.Boolean.Input(
                    "empty_cache",
                    default=True,
                    tooltip=(
                        "true gives the driver back the blocks torch has reserved and is "
                        "not using. Torch reuses those blocks itself, so this seldom "
                        "changes what the next sampler can fit; reach for it when another "
                        "program, or a library such as OpenCV, needs room on the card."
                    ),
                ),
                io.Boolean.Input(
                    "collect_garbage",
                    default=True,
                    tooltip=(
                        "true runs Python's collector before the cache is emptied, so "
                        "anything the graph has finished with is actually handed back "
                        "rather than only marked unused. It costs a few milliseconds and "
                        "makes unload_models worth more."
                    ),
                ),
                io.Float.Input(
                    "release_fraction",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    optional=True,
                    tooltip=(
                        "Move weights to system memory when less than this share of the "
                        "card is free, as `0.5` before a VAE decode. `0` does nothing, and "
                        "so does any share already free. The weights come back from memory "
                        "rather than from disk, unlike unload_models."
                    ),
                ),
                io.Boolean.Input(
                    "always_run",
                    default=False,
                    optional=True,
                    tooltip=(
                        "`false` = free only where something upstream ran, leaving an "
                        "unchanged graph served from cache; `true` = free on every prompt, "
                        "which also runs everything wired after passthrough again. A node "
                        "with nothing wired into passthrough needs `true`, or it frees once "
                        "and is served from cache from then on."
                    ),
                ),
            ],
            outputs=[
                io.MatchType.Output(
                    template=template,
                    display_name="passthrough",
                    tooltip=(
                        "The value that came in, unchanged, on a socket carrying its type. "
                        "Nothing wired to it starts until the freeing is over. Empty when "
                        "nothing was wired into passthrough."
                    ),
                ),
                io.Float.Output(
                    display_name="vram_before",
                    tooltip=(
                        "Gigabytes in use on the device ComfyUI computes on when the node "
                        "started, as 18.42. On a machine with no graphics card that device "
                        "is the processor and the figure is system RAM."
                    ),
                ),
                io.Float.Output(
                    display_name="vram_after",
                    tooltip=(
                        "The same figure once the freeing has finished, as 2.10. Wire it "
                        "into Compare to stop a run that still has too little room, or "
                        "into Text Concatenate to record it."
                    ),
                ),
                io.Float.Output(
                    display_name="freed",
                    tooltip=(
                        "vram_before minus vram_after, in gigabytes, as 16.32. 0.00 means "
                        "nothing was handed back. It reads negative when another program "
                        "took memory on the same device while this ran."
                    ),
                ),
                io.String.Output(
                    display_name="report",
                    tooltip=(
                        "The device, the used, free and total figures on both sides, what "
                        "was done and how much came back, on five lines. Drawn on the node "
                        "and wireable to Display Any or Text Save."
                    ),
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def fingerprint_inputs(cls, always_run=False, **kwargs) -> float | bool:
        """Whether the freeing is held out of the cache.

        Args:
            always_run: Whether every prompt frees again.
            **kwargs: Every other input, none of which decides this.

        Returns:
            ``NaN``, which never equals itself, or ``False`` to follow what is upstream.
        """
        return float("NaN") if always_run else False

    @classmethod
    def execute(
        cls,
        passthrough=None,
        unload_models=True,
        empty_cache=True,
        collect_garbage=True,
        release_fraction=0.0,
        always_run=False,
    ) -> io.NodeOutput:
        """Free what was asked for and answer the figures either side of it.

        Args:
            passthrough: Any value, handed back unchanged.
            unload_models: Whether every loaded model is handed back.
            empty_cache: Whether torch's reserved blocks go back to the driver.
            collect_garbage: Whether Python's collector runs first.
            release_fraction: Share of the device to make free by moving weights to
                system memory.
            always_run: Read by :meth:`fingerprint_inputs`, not by the freeing.

        Returns:
            The value that came in, the gigabytes in use before and after, the gigabytes
            freed, and the five-line report.
        """
        import gc

        import comfy.model_management as management

        from ...modules.model import compute_device, unpin_staged

        device = compute_device()
        before = usage(management, device)

        steps: list[str] = []
        share = min(1.0, max(0.0, float(release_fraction or 0.0)))
        if share > 0.0:
            # Staged pages are handed back before the release is asked for.
            try:
                unpinned = unpin_staged()
                wanted = share * management.get_total_memory(device)
                moved = management.free_memory(wanted, device, for_dynamic=False)
                staged = "unpinned the staged pages and " if unpinned else ""
                steps.append(
                    f"{staged}released {len(moved)} model(s) to make "
                    f"{gigabytes(wanted):.2f} GB free"
                )
            except Exception as error:
                logger.warning("the models could not be released: %s", error)
                steps.append(f"could not release the models ({error})")
        if unload_models:
            try:
                management.unload_all_models()
                steps.append("unloaded every model")
            except Exception as error:
                logger.warning("the loaded models could not be unloaded: %s", error)
                steps.append(f"could not unload the models ({error})")
        if collect_garbage:
            gc.collect()
            steps.append("collected garbage")
        if empty_cache:
            try:
                management.soft_empty_cache(force=True)
                steps.append("emptied the cache")
            except Exception as error:
                logger.warning("the cache could not be emptied: %s", error)
                steps.append(f"could not empty the cache ({error})")

        after = usage(management, device)
        freed = round(before[0] - after[0], 3)
        report = summary(device, steps, before, after, freed)
        logger.info("%s", report)
        return io.NodeOutput(
            passthrough,
            before[0],
            after[0],
            freed,
            report,
            ui=ui.PreviewText(report),
        )
