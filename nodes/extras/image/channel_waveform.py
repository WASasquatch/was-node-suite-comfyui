"""Waveform scopes and an RGB parade for an image batch."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.convert.tensors import image_planes, plane_shape
from ....modules.interface import channel, preview, run_result

logger = log.get_logger("nodes.extras.image")

REQUIRES = "extras"

#: The order the statistics rows are drawn in, which is also the order the channels are
#: plotted in.
CHANNEL_LABELS = ("R", "G", "B")

#: Slot the scope drawn for the node's own panel is published under.
#: ``web/was_channel_waveform.js`` names the same.
PARADE_SLOT = "parade_scope"

#: The fact the report carries the scope's own resampling under.
SCOPE_FACT = "scope"


def _out_of_range(summaries) -> tuple[list[str], list[int]]:
    """Which channels leave 0 to 1 anywhere in the batch, and on which frames.

    Args:
        summaries: One list of three ``waveform.stats_tensor`` summaries per frame, in batch
            order.

    Returns:
        ``(channels, frames)``, the channels in :data:`CHANNEL_LABELS` order and the frame
        numbers counting from 1, each list empty when every frame stays inside the range.
    """
    channels, frames = [], set()
    for number, stats in enumerate(summaries, start=1):
        for label, channel in zip(CHANNEL_LABELS, stats):
            if channel[0] < 0.0 or channel[1] > 1.0:
                if label not in channels:
                    channels.append(label)
                frames.add(number)
    return [label for label in CHANNEL_LABELS if label in channels], sorted(frames)


def _summary_line(frames: int, measured: int, channels: list[str], clipped: list[int]) -> str:
    """The one line the readout puts above the numbers.

    Args:
        frames: Frames in the batch.
        measured: Frames the report carries figures for.
        channels: Channels leaving 0 to 1, from :func:`_out_of_range`.
        clipped: Frame numbers those channels leave it on, from :func:`_out_of_range`.

    Returns:
        The line, naming how much of the batch was measured and then the channels and where
        they leave the range.
    """
    if frames == 1:
        line = "one frame measured"
    elif measured < frames:
        line = f"{measured} of {frames} frames measured"
    else:
        line = f"{frames} frames measured"
    if not channels:
        return line
    names = " and ".join(channels)
    if frames == 1:
        return f"{line}, {names} outside 0 to 1"
    where = f"frame {clipped[0]}" if len(clipped) == 1 else f"{len(clipped)} frames"
    return f"{line}, {names} outside 0 to 1 on {where}"


def _publish_report(summaries, rails, frames: int, scope: str = "") -> None:
    """Report the measured channels to the node's own interface.

    Never raises, and never changes what the node returns.

    Args:
        summaries: One list of three ``waveform.stats_tensor`` summaries per frame of the
            batch, in batch order.
        rails: One list of three ``waveform.rail_fractions`` pairs per frame the report
            carries, in batch order, which is the head of ``summaries``.
        frames: How many frames were plotted.
        scope: How the scope published for the panel was resampled, from
            :func:`_scope_note`. Empty where no scope was drawn.
    """
    try:
        if not run_result.watching():
            return
        from ....modules.image import waveform

        # A channel outside 0 to 1 is not a scope reading, it is data the picture cannot hold:
        # everything below 0 and above 1 is gone the moment the frame is written out. Judged
        # over the whole batch, so a clipped frame past the ones the report carries still
        # reaches the readout.
        channels, clipped = _out_of_range(summaries)
        items = []
        for stats, pairs in zip(summaries, rails):
            black = max(pair[0] for pair in pairs) * 100.0
            white = max(pair[1] for pair in pairs) * 100.0
            items.append({
                "text": "\n".join(
                    waveform.stats_row(label, channel)
                    for label, channel in zip(CHANNEL_LABELS, stats)
                ),
                "note": f"black {black:.2f}%, white {white:.2f}%",
            })
        counts = {"frames": frames}
        if len(items) < frames:
            counts["measured"] = len(items)
        facts = {"columns": waveform.stats_header()}
        if scope:
            facts[SCOPE_FACT] = scope
        run_result.publish(
            status=run_result.WARNING if channels else run_result.OK,
            summary=_summary_line(frames, len(items), channels, clipped),
            counts=counts,
            facts=facts,
            items=items,
            items_total=frames,
        )
    except Exception as error:
        logger.debug("no waveform report was published (%s)", error)


def _axis_note(shown: int, held: int, name: str) -> str:
    """How much of one axis of a plot the scope drawn for the panel keeps.

    Args:
        shown: Samples the scope holds on that axis.
        held: Samples the plot held on it.
        name: What the axis is called, such as ``"columns"``.

    Returns:
        ``"240 of 1872 columns"`` where the axis was reduced, ``"all 128 columns"`` where the
        scope is at least as long as the plot and keeps every sample.
    """
    return f"{shown} of {held} {name}" if shown < held else f"all {held} {name}"


def _scope_note(plot_shape) -> str:
    """How the scope drawn for the panel was resampled from the plots the node made.

    Args:
        plot_shape: The ``(levels, columns)`` shape of one plot, from
            ``waveform.make_waveform_gray``.

    Returns:
        A phrase naming both axes, for the panel to show on hover, such as ``"240 of 1872
        columns and 256 of 512 levels per channel"``.
    """
    from ....modules.image import waveform

    levels, columns = int(plot_shape[0]), int(plot_shape[1])
    return (
        f"{_axis_note(waveform.PANEL_PLOT_WIDTH, columns, 'columns')} and "
        f"{_axis_note(waveform.PANEL_PLOT_HEIGHT, levels, 'levels')} per channel"
    )


class ChannelWaveform(io.ComfyNode):
    """Plot each colour channel as a broadcast waveform, and the three side by side."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASNSChannelWaveform",
            display_name="Image Waveform",
            search_aliases=[
                "WASNSChannelWaveform", "WAS Channel Waveform (Parade)",
                "Channel Waveform (Parade)", "Image Waveform", "waveform", "parade",
                "rgb parade", "scope", "histogram", "levels",
            ],
            category="WAS Suite/Image/Analyze",
            description=(
                "Plot the red, green and blue channels of each picture as broadcast waveform "
                "scopes, and the three together as an RGB parade. Each column of the plot is "
                "a column of the picture, so it shows where in the frame the brightness sits "
                "and whether the channels agree: a colour cast reads as three traces at "
                "different heights, clipping as a trace pinned to the top of the grid. Min, "
                "max, mean, deviation and median are printed underneath."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The pictures to measure. Each frame of a batch gets its own set of "
                        "plots, so a sequence can be checked for drift frame by frame."
                    ),
                ),
                io.Int.Input(
                    "waveform_height", default=512, min=128, max=2048, step=1,
                    tooltip=(
                        "Height of the plots in pixels, which is how finely the brightness "
                        "scale is divided. 512 separates levels that a 256-step scale would "
                        "merge; raise it to 1024 to see fine banding, lower it for a compact "
                        "on-screen scope."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="red_waveform",
                    tooltip=(
                        "The red channel's scope, with its own IRE grid and statistics line."
                    ),
                ),
                io.Image.Output(
                    display_name="green_waveform",
                    tooltip=(
                        "The green channel's scope, with its own IRE grid and statistics line."
                    ),
                ),
                io.Image.Output(
                    display_name="blue_waveform",
                    tooltip=(
                        "The blue channel's scope, with its own IRE grid and statistics line."
                    ),
                ),
                io.Image.Output(
                    display_name="rgb_parade",
                    tooltip=(
                        "All three scopes side by side under one grid, which is the view a "
                        "colour cast or a channel clipping early shows up in. This is also "
                        "what the node previews."
                    ),
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, image, waveform_height) -> io.NodeOutput:
        import numpy as np
        import torch

        from ....modules.image import waveform

        def as_image(panel: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(
                np.ascontiguousarray(panel.astype(np.float32) / 255.0)
            ).unsqueeze(0)

        reds, greens, blues, parades = [], [], [], []
        summaries, rails = [], []
        # The scope is only for the node's own panel, so it is composed under the gate the
        # picture channel encodes behind, and only for the frames that channel will hold.
        drawn = channel.watching() and channel.wanted(channel.executing_node_id())
        scopes = [] if drawn else None
        scope_note = ""

        for index, frame in enumerate(image_planes(image)):
            height, width, channels = plane_shape(frame)
            plane = frame.reshape(height, width, channels)
            # A greyscale frame carries one plane and is plotted three times, so the parade
            # reads as three matching scopes rather than raising on a channel it has not got.
            picks = [plane[..., min(channel, channels - 1)] for channel in range(3)]
            stats = [waveform.stats_tensor(pick) for pick in picks]
            summaries.append(stats)
            # One report carries a bounded sample of the batch, so the rails are measured for
            # the frames it can carry rather than for every frame of a long sequence.
            if index < run_result.MAX_ITEMS:
                rails.append([waveform.rail_fractions(pick) for pick in picks])
            plots = [
                waveform.make_waveform_gray(
                    pick.detach().cpu().numpy().astype(np.float32), waveform_height,
                )
                for pick in picks
            ]

            for panels, plot, colour, channel_stats in zip(
                (reds, greens, blues), plots, ("red", "green", "blue"), stats
            ):
                panels.append(
                    as_image(
                        waveform.compose_waveform_panel(
                            plot, colour, waveform.stats_line(channel_stats)
                        )
                    )
                )
            parades.append(as_image(waveform.compose_parade(*plots, *stats)))
            if scopes is not None and index < preview.MAX_FRAMES:
                scopes.append(as_image(waveform.compose_panel_parade(*plots)))
                scope_note = scope_note or _scope_note(plots[0].shape)

        parade_batch = torch.cat(parades, dim=0)
        if scopes:
            preview.publish_output_frames(torch.cat(scopes, dim=0), slot=PARADE_SLOT)
        _publish_report(summaries, rails, len(parades), scope_note)
        return io.NodeOutput(
            torch.cat(reds, dim=0),
            torch.cat(greens, dim=0),
            torch.cat(blues, dim=0),
            parade_batch,
        )
