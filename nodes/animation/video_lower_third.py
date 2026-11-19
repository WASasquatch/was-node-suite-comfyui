"""A name and title graphic that animates in and out over a frame batch."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.data import paths
from ...modules.image import burn_in, draw
from ...modules.io import picker
from ...modules.media import subtitles
from ...modules.util import easing

logger = log.get_logger("nodes.video_lower_third")

NODE_NAME = "Video Lower Third"

#: How the graphic enters and leaves.
MOTIONS = ("slide", "rise", "fade")

#: Where the graphic sits: the frame edge it keeps to, then its side.
POSITIONS = ("bottom left", "bottom center", "bottom right", "top left", "top center", "top right")

#: Title size as a share of the name's.
TITLE_SCALE = 0.62

#: Title strength as a share of the name colour's.
TITLE_ALPHA = 0.82


class VideoLowerThird(io.ComfyNode):
    """Draw a name and title graphic that animates in and out at set times."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASVideoLowerThird",
            display_name=NODE_NAME,
            search_aliases=[
                "WASVideoLowerThird", NODE_NAME,
                "upper third",
                "name strap",
                "chyron",
                "name super",
                "caption bar",
                "speaker title",
                "title graphic",
            ],
            category="WAS Suite/Animation",
            description=(
                "Introduce a speaker with a name and title, drawn as a bar, boxes, a banner, a card "
                "or plain text, at the top or bottom of the frame. It slides, rises or fades in and "
                "out at the times given and stays inside the safe area."
            ),
            inputs=[
                io.Image.Input("images", tooltip="The frames to draw on, in playback order, frame 0 at 0 seconds."),
                io.Float.Input(
                    "fps",
                    default=0.0,
                    min=0.0,
                    max=1000.0,
                    step=0.001,
                    tooltip=(
                        "The clip's frame rate: 23.976, 25, 29.97. The graphic shows on frame n when "
                        "start <= n / fps < start + duration. Wire Video Metadata's fps; 0 = not set, refused."
                    ),
                ),
                io.String.Input(
                    "name",
                    default="",
                    tooltip="The first line, drawn bold: `Dr Amira Haddad`. Arabic and Hebrew are joined right to left.",
                ),
                io.String.Input(
                    "title",
                    default="",
                    tooltip="The second line, smaller: `Climate Scientist, Oxford`. Empty draws the name alone.",
                ),
                io.Float.Input(
                    "start",
                    default=1.0,
                    min=0.0,
                    max=86400.0,
                    step=0.01,
                    tooltip="Seconds into the batch the graphic starts to enter: 1 = frame 24 at 24 fps.",
                ),
                io.Float.Input(
                    "duration",
                    default=5.0,
                    min=0.0,
                    max=86400.0,
                    step=0.01,
                    tooltip="Seconds from start until the graphic has left, transitions included: 5 = a usual hold; 0 = never shown.",
                ),
                io.Combo.Input(
                    "style",
                    options=list(burn_in.GRAPHIC_STYLES),
                    default="stripe",
                    tooltip=(
                        "'stripe' = bar with an accent edge; 'split' = name on an accent box, title on "
                        "a dark box; 'banner' = band across the frame, accent name block; "
                        "'card' = rounded card, accent title; 'underline' = accent rule, no bar; "
                        "'minimal' = text only."
                    ),
                ),
                io.Combo.Input(
                    "position",
                    options=list(POSITIONS),
                    default="bottom left",
                    tooltip=(
                        "Frame edge and side the graphic keeps to. Text aligns to the side; 'right' "
                        "suits Arabic and Hebrew names. 'top' = an upper third, clear of subtitles."
                    ),
                ),
                io.Combo.Input(
                    "motion",
                    options=list(MOTIONS),
                    default="slide",
                    tooltip=(
                        "'slide' = the graphic wipes open from its side and the text glides in with "
                        "it, from the middle when centred; 'rise' = it moves into place from below, "
                        "or from above at the top, as it fades in; 'fade' = it fades."
                    ),
                ),
                io.Float.Input(
                    "transition",
                    default=0.5,
                    min=0.0,
                    max=5.0,
                    step=0.01,
                    tooltip="Seconds each entrance and exit takes, inside duration: 0.5 = brisk; 1 = gentle; 0 = cut.",
                ),
                io.String.Input(
                    "accent_color",
                    default="#3D8BFD",
                    tooltip="The style's accent: stripe, name box, banner block, rule or card title. #3D8BFD blue, #E8A33D amber, #D93A3A red.",
                ),
                picker.path_input(io.Combo.Input(
                    "font",
                    options=list(paths.font_names()),
                    default=draw.DEFAULT_FONT,
                    tooltip=(
                        "Typeface, from the same menu as Image Draw Text. The name uses the face's "
                        "Bold file where one exists."
                    ),
                )),
                io.Float.Input(
                    "size",
                    default=4.0,
                    min=1.0,
                    max=20.0,
                    step=0.1,
                    tooltip="Name size as percent of the frame height: 4 = 43 px at 1080p. The title is 62% of it.",
                ),
                io.String.Input(
                    "text_color",
                    default="#FFFFFF",
                    advanced=True,
                    tooltip="Name colour; the title is drawn in it at 82%: #FFFFFF, #F2F2F2.",
                ),
                io.String.Input(
                    "bar_color",
                    default="#101318E0",
                    advanced=True,
                    tooltip=(
                        "Bar, box or card colour, #RRGGBBAA for its strength, and the shadow colour "
                        "for 'underline' and 'minimal': #101318E0 = near black at 88%; #FFFFFFE6 = light."
                    ),
                ),
                io.Float.Input(
                    "offset",
                    default=15.0,
                    min=0.0,
                    max=90.0,
                    step=0.5,
                    advanced=True,
                    tooltip=(
                        "Distance from the top or bottom frame edge to the graphic, as percent of the "
                        "frame height: 15 = clear of subtitles; 5 = on the safe line."
                    ),
                ),
                io.Float.Input(
                    "safe_margin",
                    default=5.0,
                    min=0.0,
                    max=25.0,
                    step=0.5,
                    advanced=True,
                    tooltip="Space kept clear at every edge, as percent of the frame: 5 = EBU R95 graphics safe; 10 = title safe.",
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip="The frames with the graphic drawn on; same size, count and rate.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        images,
        fps=0.0,
        name="",
        title="",
        start=1.0,
        duration=5.0,
        style="stripe",
        position="bottom left",
        motion="slide",
        transition=0.5,
        accent_color="#3D8BFD",
        font=draw.DEFAULT_FONT,
        size=4.0,
        text_color="#FFFFFF",
        bar_color="#101318E0",
        offset=15.0,
        safe_margin=5.0,
    ) -> io.NodeOutput:
        """Draw the graphic on the frames its time covers.

        Raises:
            ValueError: fps is not set, the batch is empty, there is no name, or a menu value is
                not one the node offers.
        """
        from ...modules.image import scratch
        from ...modules.interface import preview, run_result
        from ...modules.media import clip as clips

        rate = subtitles.clip_rate(fps, NODE_NAME)
        if images is None or int(images.shape[0]) == 0:
            raise ValueError(f"{NODE_NAME} was handed an empty frame batch. Wire in the clip's frames.")
        for label, value, options in (
            ("style", style, burn_in.GRAPHIC_STYLES), ("position", position, POSITIONS), ("motion", motion, MOTIONS),
        ):
            if value not in options:
                raise ValueError(f"{NODE_NAME} has no {label} {value!r}. Choose one of: {', '.join(options)}.")
        name = " ".join(str(name or "").split())[:200]
        title = " ".join(str(title or "").split())[:200]
        if not name and not title:
            raise ValueError(f"{NODE_NAME} has nothing to show. Type a name, a title or both.")
        if not name:
            name, title = title, ""
        edge, align = position.split(" ")

        count, height, width = (int(side) for side in images.shape[:3])
        margin_x = round(width * float(safe_margin) / 100.0)
        margin_y = round(height * float(safe_margin) / 100.0)
        reach = max(margin_y, round(height * float(offset) / 100.0))
        room = width - 2 * margin_x
        fonts = burn_in.Fonts(font)
        fill = draw.parse_color(text_color)
        title_fill = fill[:3] + (round(fill[3] * TITLE_ALPHA),)
        back = draw.parse_color(bar_color, (16, 19, 24, 224))
        accent = draw.parse_color(accent_color, (61, 139, 253, 255))

        wanted = max(burn_in.MIN_SIZE, round(float(size) / 100.0 * height))
        pixels = wanted
        while True:
            name_set = burn_in.Typesetter(fonts, pixels)
            title_set = burn_in.Typesetter(fonts, max(burn_in.MIN_SIZE, round(pixels * TITLE_SCALE)))
            bar, text, lead = burn_in.name_graphic(
                style, name, title, name_set, title_set, align, fill, title_fill, back, accent,
                span=room if style == "banner" else 0,
            )
            if bar.width <= room or pixels <= burn_in.MIN_SIZE:
                break
            pixels = max(burn_in.MIN_SIZE, int(pixels * room / bar.width))
        x = margin_x if align == "left" else width - margin_x - bar.width if align == "right" else (width - bar.width) // 2
        y = max(0, height - reach - bar.height) if edge == "bottom" else min(reach, max(0, height - bar.height))

        first, stop = subtitles.frame_span(
            subtitles.seconds(start), subtitles.seconds(start) + subtitles.seconds(duration), rate
        )
        ramp = subtitles.fade_frames(transition, rate)
        out = scratch.allocate(
            tuple(images.shape), dtype=images.dtype, node=NODE_NAME,
            advice="Passing fewer or smaller frames also fits it.",
        )
        out.copy_(images)
        step = clips.progress(count)
        shown = 0
        held = None
        curve = "ease_in_out_sine" if motion == "fade" else "ease_out_cubic"
        for index in range(count):
            amount = subtitles.fade_alpha(index, first, stop, ramp)
            if amount > 0.0:
                burn_in.lay_bar(
                    out[index], bar, text, lead, x, y, easing.ease(curve, amount), align, motion,
                    1 if edge == "bottom" else -1,
                )
                shown += 1
                if held is None and amount >= 1.0:
                    held = index
            step(1)

        preview.publish_output(out[held:held + 1] if held is not None else out[:1])
        run_result.publish(
            status=run_result.OK if shown else run_result.WARNING,
            summary=(
                f"shown on {shown} of {count} frame(s)" if shown
                else "never shown: duration is shorter than one frame" if stop <= first
                else f"never shown: frames {first} to {stop - 1} lie outside the batch's {count}"
            ),
            counts={"frames shown": shown, "first frame": max(0, first), "transition frames": ramp},
            facts={
                "frame rate": f"{float(rate):.6g} fps",
                "graphic": f"{style}, {bar.width}x{bar.height} px at x {x}, y {y}",
                "name size": f"{name_set.size} px" + (", shrunk to fit" if name_set.size < wanted else ""),
            },
        )
        logger.info("lower third on %d of %d frame(s), %s, %s, %s", shown, count, style, position, motion)
        return io.NodeOutput(out)
