"""Subtitles from a SubRip or WebVTT file, or from typed text, burned into a frame batch."""

from __future__ import annotations

import os

from comfy_api.latest import io

from ...modules import log
from ...modules.data import paths
from ...modules.image import burn_in, draw
from ...modules.io import picker
from ...modules.media import subtitles

logger = log.get_logger("nodes.video_subtitles")

NODE_NAME = "Video Subtitles"

#: The subtitle menu entry that reads subtitle_text instead of a file.
TEXT_SOURCE = "none (use subtitle_text)"

def subtitle_options() -> list[str]:
    """The subtitle menu: reading the text box first, then every .srt and .vtt file."""
    return [TEXT_SOURCE, *picker.labels(subtitles.EXTENSIONS)]


def _read(subtitle_file: str, subtitle_text: str):
    """The cues to burn in, and the name of where they came from.

    Raises:
        FileNotFoundError: The menu names a file that is no longer there.
        ValueError: Neither a file nor text was given, the source is too large, or it holds
            no timing line at all.
        PathNotAllowed: A wired path lies outside every folder this pack may read.
    """
    from ...modules.util import file_listing

    chosen = str(subtitle_file or "").strip()
    if chosen and chosen != TEXT_SOURCE:
        path = picker.resolve(chosen, subtitles.EXTENSIONS)
        if path is None:
            if file_listing.is_label(chosen):
                raise file_listing.missing_label(chosen)
            raise FileNotFoundError(
                f"{NODE_NAME} found no .srt or .vtt file named `{chosen[:120]}`. Pick the file "
                f"again from the subtitle_file menu, or wire its path in as text."
            )
        try:
            track = subtitles.read_file(path)
        except OSError as error:
            raise OSError(
                f"{NODE_NAME} could not read `{chosen[:120]}`: {error.strerror or 'read failed'}. "
                f"Check the file is not open in another program, or pick it again."
            ) from None
        source = file_listing.label_of(path, subtitles.EXTENSIONS) if file_listing.path_text(chosen) else chosen
    else:
        text = str(subtitle_text or "")
        if not text.strip():
            raise ValueError(
                f"{NODE_NAME} has no subtitles to burn in. Pick an .srt or .vtt file in "
                f"subtitle_file, or write or wire SubRip text into subtitle_text."
            )
        if len(text) > subtitles.MAX_FILE_BYTES:
            raise ValueError(
                f"subtitle_text holds {len(text)} characters, and {subtitles.MAX_FILE_BYTES} are "
                f"read. Load the subtitles from a file, split into reels or episodes."
            )
        track = subtitles.parse(text)
        track.encoding = "text"
        source = "subtitle_text"
    if not track.cues and not track.skipped and not track.unread:
        raise ValueError(
            f"`{source}` holds no cue. A cue opens with a timing line such as "
            f"00:00:01,000 --> 00:00:03,500, then its text, then a blank line."
        )
    return track, source


def _clock(value) -> str:
    """Seconds written as ``h:mm:ss.mmm``."""
    millis = int(round(float(value) * 1000))
    sign = "-" if millis < 0 else ""
    millis = abs(millis)
    hours, rest = divmod(millis, 3600000)
    minutes, rest = divmod(rest, 60000)
    return f"{sign}{hours}:{minutes:02d}:{rest / 1000:06.3f}"


class VideoSubtitles(io.ComfyNode):
    """Burn SubRip or WebVTT subtitles into a frame batch, timed to the frame."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASVideoSubtitles",
            display_name=NODE_NAME,
            search_aliases=[
                "WASVideoSubtitles", NODE_NAME,
                "srt",
                "webvtt",
                "captions",
                "burn in subtitles",
                "hardsub",
                "closed captions",
            ],
            category="WAS Suite/Animation",
            description=(
                "Burn subtitles from an .srt or .vtt file, or from typed SubRip text, into a "
                "frame batch, each cue on exactly the frames its times cover. Captions wrap to "
                "two balanced lines inside the safe area, with an outline, shadow or box, and "
                "right-to-left text joined."
            ),
            inputs=[
                io.Image.Input("images", tooltip="The frames to caption, in playback order, frame 0 at 0 seconds."),
                io.Float.Input(
                    "fps",
                    default=0.0,
                    min=0.0,
                    max=1000.0,
                    step=0.001,
                    tooltip=(
                        "The clip's frame rate: 23.976, 25, 29.97. A cue shows on frame n when "
                        "start <= n / fps < end. Wire Video Metadata's fps; 0 = not set, refused."
                    ),
                ),
                picker.path_input(io.Combo.Input(
                    "subtitle_file",
                    options=subtitle_options(),
                    default=TEXT_SOURCE,
                    tooltip=(
                        "Which .srt or .vtt file to burn in, from input, output, temp or "
                        "paths.allow_read. 'none (use subtitle_text)' reads the text box instead. "
                        "Files over 4 MB are refused."
                    ),
                )),
                io.String.Input(
                    "subtitle_text",
                    default="",
                    multiline=True,
                    tooltip=(
                        "SubRip or WebVTT text read when subtitle_file is 'none': "
                        "`1`, `00:00:01,000 --> 00:00:03,500`, `Hello.`, then a blank line per cue. "
                        "<i>, <b>, <u> and <font color> are drawn."
                    ),
                ),
                io.Float.Input(
                    "offset",
                    default=0.0,
                    min=-86400.0,
                    max=86400.0,
                    step=0.001,
                    tooltip=(
                        "Seconds added to every cue time: 2 = cues 2 s later; -600 = a batch cut "
                        "from 10 minutes into the film. 0 = frame 0 is 00:00:00,000."
                    ),
                ),
                picker.path_input(io.Combo.Input(
                    "font",
                    options=list(paths.font_names()),
                    default=draw.DEFAULT_FONT,
                    tooltip=(
                        "Typeface, from the same menu as Image Draw Text. Bold uses the face's "
                        "Bold file where one exists; italic is slanted. Arabic and Hebrew fall "
                        "back to DejaVu Sans."
                    ),
                )),
                io.Float.Input(
                    "size",
                    default=5.0,
                    min=1.0,
                    max=25.0,
                    step=0.1,
                    tooltip=(
                        "Text size as percent of the frame height: 5 = 54 px at 1080p; "
                        "4 = smaller, for vertical video; 7 = large print."
                    ),
                ),
                io.String.Input(
                    "text_color",
                    default="#FFFFFF",
                    tooltip="Glyph colour where a cue sets none: #FFFFFF, #FFFF00 or a name such as 'white'.",
                ),
                io.Float.Input(
                    "outline",
                    default=6.0,
                    min=0.0,
                    max=30.0,
                    step=0.5,
                    tooltip="Outline width as percent of the text size: 0 = none; 6 = thin edge; 12 = heavy.",
                ),
                io.Combo.Input(
                    "box",
                    options=list(burn_in.BOX_MODES),
                    default="none",
                    tooltip=(
                        "Background box in box_color: 'none'; 'lines' = one box behind each line, "
                        "the broadcast style; 'block' = one box round the whole caption."
                    ),
                ),
                io.Combo.Input(
                    "position",
                    options=["bottom", "top"],
                    default="bottom",
                    tooltip=(
                        "Where captions sit inside the safe area. A cue carrying {\\an8} or a "
                        "WebVTT line: setting keeps its own place."
                    ),
                ),
                io.Float.Input(
                    "fade",
                    default=0.0,
                    min=0.0,
                    max=5.0,
                    step=0.01,
                    tooltip=(
                        "Seconds each cue fades in and out, inside its own time: 0 = cut, as "
                        "broadcast does; 0.2 = soft. The first and last frames are faint, never empty."
                    ),
                ),
                io.String.Input(
                    "outline_color",
                    default="#000000",
                    advanced=True,
                    tooltip="Outline colour: #000000, or #000000AA for a half-transparent edge.",
                ),
                io.Float.Input(
                    "shadow",
                    default=4.0,
                    min=0.0,
                    max=30.0,
                    step=0.5,
                    advanced=True,
                    tooltip="Drop shadow offset as percent of the text size, blurred by half of it: 0 = none; 4 = soft; 10 = deep.",
                ),
                io.String.Input(
                    "shadow_color",
                    default="#000000B4",
                    advanced=True,
                    tooltip="Shadow colour, #RRGGBBAA for its strength: #000000B4 = black at 70%.",
                ),
                io.String.Input(
                    "box_color",
                    default="#000000A0",
                    advanced=True,
                    tooltip="Box colour, #RRGGBBAA for its strength: #000000A0 = black at 63%; #000000 = solid.",
                ),
                io.Float.Input(
                    "safe_margin",
                    default=5.0,
                    min=0.0,
                    max=25.0,
                    step=0.5,
                    advanced=True,
                    tooltip=(
                        "Space kept clear at every edge, as percent of the frame: 5 = EBU R95 "
                        "graphics safe; 10 = the older title safe; 0 = to the edge."
                    ),
                ),
                io.Float.Input(
                    "max_width",
                    default=80.0,
                    min=20.0,
                    max=100.0,
                    step=1.0,
                    advanced=True,
                    tooltip=(
                        "Widest a caption may be, as percent of the frame width, never past the "
                        "safe area: 80 = about 45 characters at size 5 in 16:9."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip="The frames with the subtitles burned in; same size, count and rate.",
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, subtitle_file=TEXT_SOURCE, **_inputs):
        """When the chosen subtitle file was last written, so an edited file is read again."""
        chosen = str(subtitle_file or "").strip()
        if not chosen or chosen == TEXT_SOURCE:
            return ""
        try:
            path = picker.resolve(chosen, subtitles.EXTENSIONS)
            if path is None:
                return float("NaN")
            info = os.stat(path)
            return f"{info.st_mtime_ns}:{info.st_size}"
        except Exception:
            return float("NaN")

    @classmethod
    def execute(
        cls,
        images,
        fps=0.0,
        subtitle_file=TEXT_SOURCE,
        subtitle_text="",
        offset=0.0,
        font=draw.DEFAULT_FONT,
        size=5.0,
        text_color="#FFFFFF",
        outline=6.0,
        box="none",
        position="bottom",
        fade=0.0,
        outline_color="#000000",
        shadow=4.0,
        shadow_color="#000000B4",
        box_color="#000000A0",
        safe_margin=5.0,
        max_width=80.0,
    ) -> io.NodeOutput:
        """Burn the cues into the frames.

        Raises:
            ValueError: fps is not set, the batch is empty, or there are no subtitles.
            FileNotFoundError: The chosen file is gone.
        """
        from ...modules.image import scratch
        from ...modules.interface import preview, run_result
        from ...modules.media import clip as clips

        rate = subtitles.clip_rate(fps, NODE_NAME)
        if images is None or int(images.shape[0]) == 0:
            raise ValueError(f"{NODE_NAME} was handed an empty frame batch. Wire in the clip's frames.")
        track, source = _read(subtitle_file, subtitle_text)

        count, height, width = (int(side) for side in images.shape[:3])
        look = burn_in.Look(
            fill=draw.parse_color(text_color),
            outline=float(outline) / 100.0,
            outline_color=draw.parse_color(outline_color, (0, 0, 0, 255)),
            shadow=float(shadow) / 100.0,
            shadow_color=draw.parse_color(shadow_color, (0, 0, 0, 180)),
            box=box if box in burn_in.BOX_MODES else "none",
            box_color=draw.parse_color(box_color, (0, 0, 0, 160)),
        )
        typefaces = burn_in.Fonts(font)
        size_px = max(burn_in.MIN_SIZE, round(float(size) / 100.0 * height))
        margin_x = round(width * float(safe_margin) / 100.0)
        margin_y = round(height * float(safe_margin) / 100.0)
        limit = min(width * float(max_width) / 100.0, width - 2 * margin_x)

        def width_at(pixels: int) -> float:
            pad = round(burn_in.BOX_PAD_X * pixels) + round(look.outline * pixels)
            return max(1.0, limit - 2 * pad)

        def fitted(cue) -> tuple:
            pixels, smaller = size_px, False
            for _attempt in range(6):
                lines, setter, shrank, longer = burn_in.set_cue(cue.lines, typefaces, pixels, width_at)
                tile = burn_in.paint(lines, setter, look)
                smaller = smaller or shrank
                if tile.height <= height - 2 * margin_y or setter.size <= burn_in.MIN_SIZE:
                    break
                pixels = max(burn_in.MIN_SIZE, int(setter.size * (height - 2 * margin_y) / tile.height))
                smaller = True
            return tile, smaller, longer

        shift = subtitles.seconds(offset)
        fade_length = subtitles.fade_frames(fade, rate)
        spans, missed = [], 0
        active: list[list[int]] = [[] for _ in range(count)]
        for index, cue in enumerate(track.cues):
            first, stop = subtitles.frame_span(cue.start, cue.end, rate, shift)
            if stop <= 0 or first >= count:
                spans.append(None)
                continue
            if stop <= first:
                missed += 1
                spans.append(None)
                continue
            spans.append((first, stop))
            for frame in range(max(0, first), min(count, stop)):
                active[frame].append(index)

        out = scratch.allocate(
            tuple(images.shape), dtype=images.dtype, node=NODE_NAME,
            advice="Passing fewer or smaller frames also fits it.",
        )
        out.copy_(images)
        tiles: dict = {}
        shrunk = over = 0
        gap = round(burn_in.STACK_GAP * size_px)
        step = clips.progress(count)
        best_frame, best_alpha = None, -1.0
        for frame in range(count):
            bottom, top = height - margin_y, margin_y
            for index in active[frame]:
                cue = track.cues[index]
                if index not in tiles:
                    tiles[index], smaller, longer = fitted(cue)
                    shrunk += int(smaller)
                    over += int(longer)
                tile = tiles[index]
                first, stop = spans[index]
                alpha = subtitles.fade_alpha(frame, first, stop, fade_length)
                x = round(width / 2.0 - (tile.left + tile.right) / 2.0)
                if (cue.place or position) == "top":
                    y = top - tile.top
                    top += tile.height + gap
                else:
                    y = bottom - tile.bottom
                    bottom -= tile.height + gap
                burn_in.lay(out[frame], tile, x, y, alpha)
                if alpha > best_alpha:
                    best_frame, best_alpha = frame, alpha
            step(1)

        shown = len(tiles)
        captioned = sum(1 for frame in active if frame)
        pick = best_frame or 0
        preview.publish_output(out[pick:pick + 1])
        on_screen = [span for span in spans if span is not None]
        rate_text = f"{float(rate):.6g} fps" + (f" ({rate.numerator}/{rate.denominator})" if rate.denominator == 1001 else "")
        facts = {"source": source, "format": f"{track.kind}, {track.encoding}", "frame rate": rate_text}
        if on_screen:
            facts["first cue"] = f"frame {max(0, on_screen[0][0])}"
            facts["last cue ends"] = f"frame {min(count, max(span[1] for span in on_screen)) - 1}"
        facts["shrunk to fit"] = shrunk
        facts["over two lines"] = over
        facts["between frames"] = missed
        counts = {
            "cues read": len(track.cues),
            "on screen": shown,
            "frames captioned": captioned,
            "skipped": len(track.skipped),
        }
        if track.unread:
            counts["not read"] = track.unread
        warning = bool(track.skipped or track.unread or not shown)
        run_result.publish(
            status=run_result.WARNING if warning else run_result.OK,
            summary=(
                f"{shown} of {len(track.cues)} cue(s) on {captioned} of {count} frame(s)"
                + (f"; {len(track.skipped)} skipped" if track.skipped else "")
                + ("" if shown else f"; none falls inside the clip's {_clock(count / rate)}, check offset")
            ),
            counts=counts,
            facts=facts,
            items=[f"line {number}: {subtitles.REASONS[reason]}" for number, reason in track.skipped[:8]],
            items_total=len(track.skipped),
        )
        logger.info(
            "%d of %d cue(s) burned into %d of %d frame(s) from %s; %d skipped, %d shrunk to fit",
            shown, len(track.cues), captioned, count, source, len(track.skipped), shrunk,
        )
        return io.NodeOutput(out)
