"""SMPTE timecode or frame numbers burned into a frame batch, in a box at a corner."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.image import burn_in, draw
from ...modules.media import subtitles, timecode

logger = log.get_logger("nodes.video_timecode")

NODE_NAME = "Video Timecode"

#: What the window shows.
DISPLAYS = ("timecode", "frame number", "timecode and frame number")

#: Where the window sits, inside the safe area.
CORNERS = ("top left", "top center", "top right", "bottom left", "bottom center", "bottom right")

#: The monospaced face the window is drawn in.
FACE = "DejaVu Sans Mono"


def _text(frame: int, rate, drop_frame: bool, display: str, digits: int, label: str) -> str:
    """What the window says on one frame."""
    parts = [label] if label else []
    if display != "frame number":
        parts.append(timecode.label(frame, rate, drop_frame))
    if display != "timecode":
        parts.append(str(frame).rjust(digits, "0" if display == "frame number" else " "))
    return "  ".join(parts)


class VideoTimecode(io.ComfyNode):
    """Burn a running SMPTE timecode or frame number into a frame batch."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASVideoTimecode",
            display_name=NODE_NAME,
            search_aliases=[
                "WASVideoTimecode", NODE_NAME,
                "smpte",
                "window burn",
                "burn in timecode",
                "frame counter",
                "drop frame",
            ],
            category="WAS Suite/Animation",
            description=(
                "Burn a running SMPTE timecode, a frame number or both into every frame, in a "
                "monospaced box at a corner of the safe area. Drop-frame timecode is written at "
                "29.97 and 59.94. For review copies and edit references."
            ),
            inputs=[
                io.Image.Input("images", tooltip="The frames to stamp, in playback order."),
                io.Float.Input(
                    "fps",
                    default=0.0,
                    min=0.0,
                    max=1000.0,
                    step=0.001,
                    tooltip=(
                        "The clip's frame rate: 23.976, 25, 29.97, 59.94. Wire Video Metadata's "
                        "fps; 0 = not set, refused."
                    ),
                ),
                io.String.Input(
                    "start",
                    default="00:00:00:00",
                    tooltip=(
                        "The first frame's timecode, 01:00:00:00, or its frame number, 1001. "
                        "Drop-frame labels such as 00:01:00;00 that do not exist are refused."
                    ),
                ),
                io.Combo.Input(
                    "display",
                    options=list(DISPLAYS),
                    default="timecode",
                    tooltip=(
                        "'timecode' = 01:00:00:00; 'frame number' = 86400; 'timecode and frame "
                        "number' = both. Frame numbers count from 00:00:00:00."
                    ),
                ),
                io.Boolean.Input(
                    "drop_frame",
                    default=True,
                    tooltip=(
                        "At 29.97 and 59.94, true = drop-frame 00:01:00;02 that keeps pace with "
                        "the clock; false = 00:01:00:00, which drifts 3.6 s an hour. Ignored at other rates."
                    ),
                ),
                io.Combo.Input(
                    "corner",
                    options=list(CORNERS),
                    default="bottom center",
                    tooltip="Where the box sits inside the safe area: 'bottom center' = the usual window burn; 'top right' = clear of subtitles.",
                ),
                io.Float.Input(
                    "size",
                    default=3.5,
                    min=1.0,
                    max=20.0,
                    step=0.1,
                    tooltip="Text size as percent of the frame height: 3.5 = 38 px at 1080p; 2.5 = discreet; 6 = large.",
                ),
                io.String.Input(
                    "label",
                    default="",
                    tooltip="Text before the timecode, such as `SRC` or `CAM A`. Empty draws none.",
                ),
                io.String.Input(
                    "text_color",
                    default="#FFFFFF",
                    advanced=True,
                    tooltip="Digit colour: #FFFFFF, #FFD000 or a name such as 'white'.",
                ),
                io.String.Input(
                    "box_color",
                    default="#000000C0",
                    advanced=True,
                    tooltip="Box colour, #RRGGBBAA for its strength: #000000C0 = black at 75%; #00000000 = no box.",
                ),
                io.Float.Input(
                    "safe_margin",
                    default=5.0,
                    min=0.0,
                    max=25.0,
                    step=0.5,
                    advanced=True,
                    tooltip="Space kept clear at every edge, as percent of the frame: 5 = EBU R95 graphics safe; 0 = in the corner.",
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip="The frames with the timecode burned in; same size, count and rate.",
                ),
                io.String.Output(
                    display_name="end_timecode",
                    tooltip="The last frame's timecode, as 00:00:14;29, for a log or a filename.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        images,
        fps=0.0,
        start="00:00:00:00",
        display="timecode",
        drop_frame=True,
        corner="bottom center",
        size=3.5,
        label="",
        text_color="#FFFFFF",
        box_color="#000000C0",
        safe_margin=5.0,
    ) -> io.NodeOutput:
        """Stamp every frame.

        Raises:
            ValueError: fps is not set, the batch is empty, or start is not a timecode.
        """
        from ...modules.image import scratch
        from ...modules.interface import preview, run_result
        from ...modules.media import clip as clips

        rate = subtitles.clip_rate(fps, NODE_NAME)
        if images is None or int(images.shape[0]) == 0:
            raise ValueError(f"{NODE_NAME} was handed an empty frame batch. Wire in the clip's frames.")
        drop = bool(drop_frame)
        try:
            first = timecode.parse(start, rate, drop)
        except ValueError as error:
            raise ValueError(f"{NODE_NAME}'s start {error}") from None
        display = display if display in DISPLAYS else "timecode"
        corner = corner if corner in CORNERS else "bottom center"
        label = " ".join(str(label or "").split())[:40]

        count, height, width = (int(side) for side in images.shape[:3])
        last = first + count - 1
        digits = max(len(str(last)), 1)
        template = "".join("8" if character.isdigit() else character for character in _text(last, rate, drop, display, digits, label))

        setter = burn_in.Typesetter(burn_in.Fonts(FACE), max(burn_in.MIN_SIZE, round(float(size) / 100.0 * height)))
        fill = draw.parse_color(text_color)
        box = draw.parse_color(box_color, (0, 0, 0, 192))
        margin_x = round(width * float(safe_margin) / 100.0)
        margin_y = round(height * float(safe_margin) / 100.0)

        out = scratch.allocate(
            tuple(images.shape), dtype=images.dtype, node=NODE_NAME,
            advice="Passing fewer or smaller frames also fits it.",
        )
        out.copy_(images)
        step = clips.progress(count)
        vertical, horizontal = corner.split(" ")
        for index in range(count):
            text = _text(first + index, rate, drop, display, digits, label)
            tile = burn_in.timecode_tile(text, template, setter, fill, box)
            if horizontal == "left":
                x = margin_x
            elif horizontal == "right":
                x = width - margin_x - tile.width
            else:
                x = round((width - tile.width) / 2.0)
            y = margin_y if vertical == "top" else height - margin_y - tile.height
            burn_in.lay(out[index], tile, x, y)
            step(1)

        dropping = timecode.dropped(rate, drop) > 0
        opening = timecode.label(first, rate, drop)
        closing = timecode.label(last, rate, drop)
        preview.publish_output(out[:1])
        rate_text = f"{float(rate):.6g} fps" + (", drop frame" if dropping else ", non-drop")
        run_result.publish(
            summary=f"{opening} to {closing} over {count} frame(s)",
            counts={"frames": count, "first frame": first, "last frame": last},
            facts={
                "frame rate": rate_text,
                "labels a second": timecode.nominal(rate),
                "first": opening,
                "last": closing,
            },
        )
        logger.info("stamped %d frame(s), %s to %s at %s", count, opening, closing, rate_text)
        return io.NodeOutput(out, closing)
