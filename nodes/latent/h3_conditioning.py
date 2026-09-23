"""One MiniMax H3 segment per row, each with its own prompt and its own length."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import H3_PROMPTS
from ...modules.latent import h3_conditioning, h3_extend, h3_references

MODE_HINT = (
    "The task to condition for. `t2va` = prompt only; `i2va` = opens on first_frame; "
    "`fl2va` = opens on first_frame, closes on last_frame; `fl2va_batched` = every "
    "segment between a neighbouring pair of images; `ref2va` = every segment built on the "
    "ref inputs, named `<Picture 1>`, `<Video 1>`, `<Audio 1>` in the prompt."
)

REF_IMAGE_HINT = (
    "A reference picture for `ref2va`, named `<Picture 1>`, `<Picture 2>` and on in the "
    "prompt, counting the wired ones in order, as `<Picture 1> steps out of the car`. The "
    "report lists every tag."
)

REF_VIDEO_HINT = (
    "A reference clip for `ref2va` at 24 fps, named `<Video 1>` and on in the prompt. Cut "
    "to the longest segment and down to the 17k+5 frame grid; at least 5 frames."
)

REF_VIDEO_AUDIO_HINT = (
    "The soundtrack of the ref_video in the same slot, named `<Audio 1>` and on in the "
    "prompt, ahead of any ref_audio. Needs audio_vae."
)

REF_AUDIO_HINT = (
    "A reference audio for `ref2va`, as a voice or a score, named `<Audio N>` in the "
    "prompt, numbered after the video soundtracks. Needs audio_vae."
)

REF_SIZE_HINT = (
    "How large each reference picture is encoded, for every row. `match` = scaled down to "
    "the clip's area; `256` to `2048` = longest side at most that; `max` = up to a 2048 "
    "pixel short edge, the closest likeness and the heaviest. Always 256 to 5760 pixels a "
    "side. Read by `ref2va` only."
)

#: The inputs `ref2va` reads, drawn only in that mode.
REFERENCE_INPUTS = [
    io.Vae.Input(
        "audio_vae",
        optional=True,
        tooltip=(
            "The H3 audio VAE, which encodes the reference soundtracks and audio. Needed "
            "by `ref2va` when any audio is wired."
        ),
    ),
    io.Combo.Input(
        "ref_image_size",
        options=list(h3_references.IMAGE_SIZES),
        default=h3_references.IMAGE_SIZES[0],
        optional=True,
        tooltip=REF_SIZE_HINT,
    ),
    io.Image.Input("ref_image_1", optional=True, tooltip=REF_IMAGE_HINT),
    io.Image.Input("ref_image_2", optional=True, tooltip=REF_IMAGE_HINT),
    io.Image.Input("ref_image_3", optional=True, tooltip=REF_IMAGE_HINT),
    io.Image.Input("ref_image_4", optional=True, tooltip=REF_IMAGE_HINT),
    io.Image.Input("ref_image_5", optional=True, tooltip=REF_IMAGE_HINT),
    io.Image.Input("ref_image_6", optional=True, tooltip=REF_IMAGE_HINT),
    io.Image.Input("ref_image_7", optional=True, tooltip=REF_IMAGE_HINT),
    io.Image.Input("ref_image_8", optional=True, tooltip=REF_IMAGE_HINT),
    io.Image.Input("ref_image_9", optional=True, tooltip=REF_IMAGE_HINT),
    io.Image.Input("ref_video_1", optional=True, tooltip=REF_VIDEO_HINT),
    io.Audio.Input("ref_video_audio_1", optional=True, tooltip=REF_VIDEO_AUDIO_HINT),
    io.Image.Input("ref_video_2", optional=True, tooltip=REF_VIDEO_HINT),
    io.Audio.Input("ref_video_audio_2", optional=True, tooltip=REF_VIDEO_AUDIO_HINT),
    io.Image.Input("ref_video_3", optional=True, tooltip=REF_VIDEO_HINT),
    io.Audio.Input("ref_video_audio_3", optional=True, tooltip=REF_VIDEO_AUDIO_HINT),
    io.Audio.Input("ref_audio_1", optional=True, tooltip=REF_AUDIO_HINT),
    io.Audio.Input("ref_audio_2", optional=True, tooltip=REF_AUDIO_HINT),
    io.Audio.Input("ref_audio_3", optional=True, tooltip=REF_AUDIO_HINT),
]

PROMPT_HEADER_HINT = (
    "Text put before every segment's prompt, as `subject_definitions:` and the wardrobe "
    "lines that hold for the whole run. Blank adds nothing, and a blank line separates it "
    "from the row's own prompt."
)

PROMPT_FOOTER_HINT = (
    "Text put after every segment's prompt, as `camera: slow dolly in` or "
    "`audio: wind and breath`. Blank adds nothing."
)

ASPECT_HINT = (
    "Canvas shape, as `16:9` or `9:16`. `custom` takes it from the first picture the mode "
    "reads, and 16:9 where it reads none. A width or height above `0` sets that side "
    "itself, so the shape is whatever those give."
)

MEGAPIXELS_HINT = (
    "Canvas area in millions of pixels, as `0.4` for 832x480 or `1.0` for 1344x736. "
    "Worked against aspect_ratio. Ignored when both width and height are set."
)

BATCH_HINT = (
    "The pictures `fl2va_batched` runs between, in order. Segment 1 runs from picture 1 "
    "to picture 2, segment 2 from 2 to 3, and so on, so 5 pictures cover 4 segments. "
    "Read by `fl2va_batched` only."
)

SEGMENT_PROMPT_HINT = (
    "One segment of the video, as `a lone astronaut walks across a red desert plain`. The "
    "first segment starts the clip and each one after it continues where the last left off. "
    "Blank ends the run."
)

SEGMENT_DURATION_HINT = (
    "How long this segment runs, as `5.2` or `8.5` seconds. Snapped onto the model's "
    "frame grid, and the report states the frames each segment came to."
)

SEGMENT_CONTINUITY_HINT = (
    "Overrides H3 Extend Window's setting for this segment. `as set` = that node's choice; "
    "`carry` = one unbroken shot; `refresh` = the same, detail softened; `handoff` = a cut "
    "opening on the last frame; `reference` = a cut keeping the cast; `cut` = a new scene, "
    "like an overlap of `0`. Ignored on segment 1."
)

SEGMENT_OVERLAP_HINT = (
    "Frames of the previous segment this one continues from, as `22` for a scene carrying "
    "on or `0` for a cut to somewhere new. `39` and `56` hold the scene harder. Ignored on "
    "the first segment."
)


class MiniMaxH3Conditioning(io.ComfyNode):
    """Encode every segment's prompt in one pass, for a loop to sample one at a time."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASMiniMaxH3Conditioning",
            display_name="MiniMax H3 Conditioning",
            search_aliases=[
                "WASMiniMaxH3Conditioning",
                "MiniMax H3 Conditioning",
                "minimax h3",
                "h3 prompt",
                "t2va",
                "i2va",
                "fl2va",
                "ref2va",
                "reference to video",
                "video continuation",
            ],
            category="WAS Suite/Latent/Video",
            description=(
                "Prompt every segment of a MiniMax H3 video on one node. Each row is one "
                "segment with its own prompt and its own frame count, and a new row appears "
                "as the last one is filled. Each row also says how much of the segment "
                "before it to continue from and how, so a scene can carry on, cut "
                "somewhere new, or cut and still keep the cast it had. `ref2va` builds "
                "every segment on reference pictures, clips and audio, so the whole run "
                "keeps the same people, places and voices. "
                "A loop sampling one row per iteration builds the whole video. Every "
                "prompt is encoded together before any sampling starts, so the "
                "text encoder is loaded once. Send segments to a While Loop's count and "
                "prompts to H3 Extend Window."
            ),
            inputs=[
                io.Clip.Input(
                    "clip",
                    tooltip="A minimax CLIP, from Load CLIP with type `minimax`.",
                ),
                io.Vae.Input(
                    "vae",
                    tooltip=(
                        "The H3 video VAE, which encodes first_frame, last_frame, images "
                        "and the reference pictures and clips."
                    ),
                ),
                io.Combo.Input(
                    "mode",
                    options=list(h3_conditioning.MODES),
                    default=h3_conditioning.MODES[0],
                    tooltip=MODE_HINT,
                ),
                io.Combo.Input(
                    "aspect_ratio",
                    options=list(h3_conditioning.ASPECTS),
                    default=h3_conditioning.ASPECTS[0],
                    tooltip=ASPECT_HINT,
                ),
                io.Float.Input(
                    "megapixels",
                    default=0.4,
                    min=0.0,
                    max=16.0,
                    step=0.05,
                    tooltip=MEGAPIXELS_HINT,
                ),
                io.Int.Input(
                    "width",
                    default=0,
                    min=0,
                    max=16384,
                    step=h3_conditioning.CANVAS_MULTIPLE,
                    tooltip=(
                        "Canvas width in pixels, as `1024`. `0` works it out from "
                        "megapixels. Rounded to a multiple of 32."
                    ),
                ),
                io.Int.Input(
                    "height",
                    default=0,
                    min=0,
                    max=16384,
                    step=h3_conditioning.CANVAS_MULTIPLE,
                    tooltip=(
                        "Canvas height in pixels, as `576`. `0` works it out from "
                        "megapixels. Rounded to a multiple of 32."
                    ),
                ),
                io.String.Input(
                    "prompt_header", multiline=True, dynamic_prompts=True,
                    default="", tooltip=PROMPT_HEADER_HINT,
                ),
                io.String.Input(
                    "prompt_footer", multiline=True, dynamic_prompts=True,
                    default="", tooltip=PROMPT_FOOTER_HINT,
                ),
                io.String.Input(
                    "prompt_1", multiline=True, dynamic_prompts=True,
                    default="", tooltip=SEGMENT_PROMPT_HINT,
                ),
                io.Float.Input(
                    "duration_1", default=5.2, min=0.2, max=150.0, step=0.1,
                    tooltip=SEGMENT_DURATION_HINT,
                ),
                io.Int.Input(
                    "overlap_1", default=22, min=0, max=362, step=1,
                    tooltip=SEGMENT_OVERLAP_HINT,
                ),
                io.Combo.Input(
                    "continuity_1", options=list(h3_conditioning.ROW_CONTINUITY),
                    default=h3_conditioning.AS_SET,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.String.Input(
                    "prompt_2", multiline=True, dynamic_prompts=True,
                    default="", optional=True, tooltip=SEGMENT_PROMPT_HINT,
                ),
                io.Float.Input(
                    "duration_2", default=5.2, min=0.2, max=150.0, step=0.1,
                    optional=True, tooltip=SEGMENT_DURATION_HINT,
                ),
                io.Int.Input(
                    "overlap_2", default=22, min=0, max=362, step=1,
                    optional=True, tooltip=SEGMENT_OVERLAP_HINT,
                ),
                io.Combo.Input(
                    "continuity_2", options=list(h3_conditioning.ROW_CONTINUITY),
                    default=h3_conditioning.AS_SET,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.String.Input(
                    "prompt_3", multiline=True, dynamic_prompts=True,
                    default="", optional=True, tooltip=SEGMENT_PROMPT_HINT,
                ),
                io.Float.Input(
                    "duration_3", default=5.2, min=0.2, max=150.0, step=0.1,
                    optional=True, tooltip=SEGMENT_DURATION_HINT,
                ),
                io.Int.Input(
                    "overlap_3", default=22, min=0, max=362, step=1,
                    optional=True, tooltip=SEGMENT_OVERLAP_HINT,
                ),
                io.Combo.Input(
                    "continuity_3", options=list(h3_conditioning.ROW_CONTINUITY),
                    default=h3_conditioning.AS_SET,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.String.Input(
                    "prompt_4", multiline=True, dynamic_prompts=True,
                    default="", optional=True, tooltip=SEGMENT_PROMPT_HINT,
                ),
                io.Float.Input(
                    "duration_4", default=5.2, min=0.2, max=150.0, step=0.1,
                    optional=True, tooltip=SEGMENT_DURATION_HINT,
                ),
                io.Int.Input(
                    "overlap_4", default=22, min=0, max=362, step=1,
                    optional=True, tooltip=SEGMENT_OVERLAP_HINT,
                ),
                io.Combo.Input(
                    "continuity_4", options=list(h3_conditioning.ROW_CONTINUITY),
                    default=h3_conditioning.AS_SET,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.String.Input(
                    "prompt_5", multiline=True, dynamic_prompts=True,
                    default="", optional=True, tooltip=SEGMENT_PROMPT_HINT,
                ),
                io.Float.Input(
                    "duration_5", default=5.2, min=0.2, max=150.0, step=0.1,
                    optional=True, tooltip=SEGMENT_DURATION_HINT,
                ),
                io.Int.Input(
                    "overlap_5", default=22, min=0, max=362, step=1,
                    optional=True, tooltip=SEGMENT_OVERLAP_HINT,
                ),
                io.Combo.Input(
                    "continuity_5", options=list(h3_conditioning.ROW_CONTINUITY),
                    default=h3_conditioning.AS_SET,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.String.Input(
                    "prompt_6", multiline=True, dynamic_prompts=True,
                    default="", optional=True, tooltip=SEGMENT_PROMPT_HINT,
                ),
                io.Float.Input(
                    "duration_6", default=5.2, min=0.2, max=150.0, step=0.1,
                    optional=True, tooltip=SEGMENT_DURATION_HINT,
                ),
                io.Int.Input(
                    "overlap_6", default=22, min=0, max=362, step=1,
                    optional=True, tooltip=SEGMENT_OVERLAP_HINT,
                ),
                io.Combo.Input(
                    "continuity_6", options=list(h3_conditioning.ROW_CONTINUITY),
                    default=h3_conditioning.AS_SET,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.String.Input(
                    "prompt_7", multiline=True, dynamic_prompts=True,
                    default="", optional=True, tooltip=SEGMENT_PROMPT_HINT,
                ),
                io.Float.Input(
                    "duration_7", default=5.2, min=0.2, max=150.0, step=0.1,
                    optional=True, tooltip=SEGMENT_DURATION_HINT,
                ),
                io.Int.Input(
                    "overlap_7", default=22, min=0, max=362, step=1,
                    optional=True, tooltip=SEGMENT_OVERLAP_HINT,
                ),
                io.Combo.Input(
                    "continuity_7", options=list(h3_conditioning.ROW_CONTINUITY),
                    default=h3_conditioning.AS_SET,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.String.Input(
                    "prompt_8", multiline=True, dynamic_prompts=True,
                    default="", optional=True, tooltip=SEGMENT_PROMPT_HINT,
                ),
                io.Float.Input(
                    "duration_8", default=5.2, min=0.2, max=150.0, step=0.1,
                    optional=True, tooltip=SEGMENT_DURATION_HINT,
                ),
                io.Int.Input(
                    "overlap_8", default=22, min=0, max=362, step=1,
                    optional=True, tooltip=SEGMENT_OVERLAP_HINT,
                ),
                io.Combo.Input(
                    "continuity_8", options=list(h3_conditioning.ROW_CONTINUITY),
                    default=h3_conditioning.AS_SET,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.String.Input(
                    "prompt_9", multiline=True, dynamic_prompts=True,
                    default="", optional=True, tooltip=SEGMENT_PROMPT_HINT,
                ),
                io.Float.Input(
                    "duration_9", default=5.2, min=0.2, max=150.0, step=0.1,
                    optional=True, tooltip=SEGMENT_DURATION_HINT,
                ),
                io.Int.Input(
                    "overlap_9", default=22, min=0, max=362, step=1,
                    optional=True, tooltip=SEGMENT_OVERLAP_HINT,
                ),
                io.Combo.Input(
                    "continuity_9", options=list(h3_conditioning.ROW_CONTINUITY),
                    default=h3_conditioning.AS_SET,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.String.Input(
                    "prompt_10", multiline=True, dynamic_prompts=True,
                    default="", optional=True, tooltip=SEGMENT_PROMPT_HINT,
                ),
                io.Float.Input(
                    "duration_10", default=5.2, min=0.2, max=150.0, step=0.1,
                    optional=True, tooltip=SEGMENT_DURATION_HINT,
                ),
                io.Int.Input(
                    "overlap_10", default=22, min=0, max=362, step=1,
                    optional=True, tooltip=SEGMENT_OVERLAP_HINT,
                ),
                io.Combo.Input(
                    "continuity_10", options=list(h3_conditioning.ROW_CONTINUITY),
                    default=h3_conditioning.AS_SET,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.String.Input(
                    "prompt_11", multiline=True, dynamic_prompts=True,
                    default="", optional=True, tooltip=SEGMENT_PROMPT_HINT,
                ),
                io.Float.Input(
                    "duration_11", default=5.2, min=0.2, max=150.0, step=0.1,
                    optional=True, tooltip=SEGMENT_DURATION_HINT,
                ),
                io.Int.Input(
                    "overlap_11", default=22, min=0, max=362, step=1,
                    optional=True, tooltip=SEGMENT_OVERLAP_HINT,
                ),
                io.Combo.Input(
                    "continuity_11", options=list(h3_conditioning.ROW_CONTINUITY),
                    default=h3_conditioning.AS_SET,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.String.Input(
                    "prompt_12", multiline=True, dynamic_prompts=True,
                    default="", optional=True, tooltip=SEGMENT_PROMPT_HINT,
                ),
                io.Float.Input(
                    "duration_12", default=5.2, min=0.2, max=150.0, step=0.1,
                    optional=True, tooltip=SEGMENT_DURATION_HINT,
                ),
                io.Int.Input(
                    "overlap_12", default=22, min=0, max=362, step=1,
                    optional=True, tooltip=SEGMENT_OVERLAP_HINT,
                ),
                io.Combo.Input(
                    "continuity_12", options=list(h3_conditioning.ROW_CONTINUITY),
                    default=h3_conditioning.AS_SET,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.String.Input(
                    "prompt_13", multiline=True, dynamic_prompts=True,
                    default="", optional=True, tooltip=SEGMENT_PROMPT_HINT,
                ),
                io.Float.Input(
                    "duration_13", default=5.2, min=0.2, max=150.0, step=0.1,
                    optional=True, tooltip=SEGMENT_DURATION_HINT,
                ),
                io.Int.Input(
                    "overlap_13", default=22, min=0, max=362, step=1,
                    optional=True, tooltip=SEGMENT_OVERLAP_HINT,
                ),
                io.Combo.Input(
                    "continuity_13", options=list(h3_conditioning.ROW_CONTINUITY),
                    default=h3_conditioning.AS_SET,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.String.Input(
                    "prompt_14", multiline=True, dynamic_prompts=True,
                    default="", optional=True, tooltip=SEGMENT_PROMPT_HINT,
                ),
                io.Float.Input(
                    "duration_14", default=5.2, min=0.2, max=150.0, step=0.1,
                    optional=True, tooltip=SEGMENT_DURATION_HINT,
                ),
                io.Int.Input(
                    "overlap_14", default=22, min=0, max=362, step=1,
                    optional=True, tooltip=SEGMENT_OVERLAP_HINT,
                ),
                io.Combo.Input(
                    "continuity_14", options=list(h3_conditioning.ROW_CONTINUITY),
                    default=h3_conditioning.AS_SET,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.String.Input(
                    "prompt_15", multiline=True, dynamic_prompts=True,
                    default="", optional=True, tooltip=SEGMENT_PROMPT_HINT,
                ),
                io.Float.Input(
                    "duration_15", default=5.2, min=0.2, max=150.0, step=0.1,
                    optional=True, tooltip=SEGMENT_DURATION_HINT,
                ),
                io.Int.Input(
                    "overlap_15", default=22, min=0, max=362, step=1,
                    optional=True, tooltip=SEGMENT_OVERLAP_HINT,
                ),
                io.Combo.Input(
                    "continuity_15", options=list(h3_conditioning.ROW_CONTINUITY),
                    default=h3_conditioning.AS_SET,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.String.Input(
                    "prompt_16", multiline=True, dynamic_prompts=True,
                    default="", optional=True, tooltip=SEGMENT_PROMPT_HINT,
                ),
                io.Float.Input(
                    "duration_16", default=5.2, min=0.2, max=150.0, step=0.1,
                    optional=True, tooltip=SEGMENT_DURATION_HINT,
                ),
                io.Int.Input(
                    "overlap_16", default=22, min=0, max=362, step=1,
                    optional=True, tooltip=SEGMENT_OVERLAP_HINT,
                ),
                io.Combo.Input(
                    "continuity_16", options=list(h3_conditioning.ROW_CONTINUITY),
                    default=h3_conditioning.AS_SET,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.String.Input(
                    "prompt_17", multiline=True, dynamic_prompts=True,
                    default="", optional=True, tooltip=SEGMENT_PROMPT_HINT,
                ),
                io.Float.Input(
                    "duration_17", default=5.2, min=0.2, max=150.0, step=0.1,
                    optional=True, tooltip=SEGMENT_DURATION_HINT,
                ),
                io.Int.Input(
                    "overlap_17", default=22, min=0, max=362, step=1,
                    optional=True, tooltip=SEGMENT_OVERLAP_HINT,
                ),
                io.Combo.Input(
                    "continuity_17", options=list(h3_conditioning.ROW_CONTINUITY),
                    default=h3_conditioning.AS_SET,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.String.Input(
                    "prompt_18", multiline=True, dynamic_prompts=True,
                    default="", optional=True, tooltip=SEGMENT_PROMPT_HINT,
                ),
                io.Float.Input(
                    "duration_18", default=5.2, min=0.2, max=150.0, step=0.1,
                    optional=True, tooltip=SEGMENT_DURATION_HINT,
                ),
                io.Int.Input(
                    "overlap_18", default=22, min=0, max=362, step=1,
                    optional=True, tooltip=SEGMENT_OVERLAP_HINT,
                ),
                io.Combo.Input(
                    "continuity_18", options=list(h3_conditioning.ROW_CONTINUITY),
                    default=h3_conditioning.AS_SET,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.String.Input(
                    "prompt_19", multiline=True, dynamic_prompts=True,
                    default="", optional=True, tooltip=SEGMENT_PROMPT_HINT,
                ),
                io.Float.Input(
                    "duration_19", default=5.2, min=0.2, max=150.0, step=0.1,
                    optional=True, tooltip=SEGMENT_DURATION_HINT,
                ),
                io.Int.Input(
                    "overlap_19", default=22, min=0, max=362, step=1,
                    optional=True, tooltip=SEGMENT_OVERLAP_HINT,
                ),
                io.Combo.Input(
                    "continuity_19", options=list(h3_conditioning.ROW_CONTINUITY),
                    default=h3_conditioning.AS_SET,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.String.Input(
                    "prompt_20", multiline=True, dynamic_prompts=True,
                    default="", optional=True, tooltip=SEGMENT_PROMPT_HINT,
                ),
                io.Float.Input(
                    "duration_20", default=5.2, min=0.2, max=150.0, step=0.1,
                    optional=True, tooltip=SEGMENT_DURATION_HINT,
                ),
                io.Int.Input(
                    "overlap_20", default=22, min=0, max=362, step=1,
                    optional=True, tooltip=SEGMENT_OVERLAP_HINT,
                ),
                io.Combo.Input(
                    "continuity_20", options=list(h3_conditioning.ROW_CONTINUITY),
                    default=h3_conditioning.AS_SET,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.String.Input(
                    "prompt_21", multiline=True, dynamic_prompts=True,
                    default="", optional=True, tooltip=SEGMENT_PROMPT_HINT,
                ),
                io.Float.Input(
                    "duration_21", default=5.2, min=0.2, max=150.0, step=0.1,
                    optional=True, tooltip=SEGMENT_DURATION_HINT,
                ),
                io.Int.Input(
                    "overlap_21", default=22, min=0, max=362, step=1,
                    optional=True, tooltip=SEGMENT_OVERLAP_HINT,
                ),
                io.Combo.Input(
                    "continuity_21", options=list(h3_conditioning.ROW_CONTINUITY),
                    default=h3_conditioning.AS_SET,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.String.Input(
                    "prompt_22", multiline=True, dynamic_prompts=True,
                    default="", optional=True, tooltip=SEGMENT_PROMPT_HINT,
                ),
                io.Float.Input(
                    "duration_22", default=5.2, min=0.2, max=150.0, step=0.1,
                    optional=True, tooltip=SEGMENT_DURATION_HINT,
                ),
                io.Int.Input(
                    "overlap_22", default=22, min=0, max=362, step=1,
                    optional=True, tooltip=SEGMENT_OVERLAP_HINT,
                ),
                io.Combo.Input(
                    "continuity_22", options=list(h3_conditioning.ROW_CONTINUITY),
                    default=h3_conditioning.AS_SET,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.String.Input(
                    "prompt_23", multiline=True, dynamic_prompts=True,
                    default="", optional=True, tooltip=SEGMENT_PROMPT_HINT,
                ),
                io.Float.Input(
                    "duration_23", default=5.2, min=0.2, max=150.0, step=0.1,
                    optional=True, tooltip=SEGMENT_DURATION_HINT,
                ),
                io.Int.Input(
                    "overlap_23", default=22, min=0, max=362, step=1,
                    optional=True, tooltip=SEGMENT_OVERLAP_HINT,
                ),
                io.Combo.Input(
                    "continuity_23", options=list(h3_conditioning.ROW_CONTINUITY),
                    default=h3_conditioning.AS_SET,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.String.Input(
                    "prompt_24", multiline=True, dynamic_prompts=True,
                    default="", optional=True, tooltip=SEGMENT_PROMPT_HINT,
                ),
                io.Float.Input(
                    "duration_24", default=5.2, min=0.2, max=150.0, step=0.1,
                    optional=True, tooltip=SEGMENT_DURATION_HINT,
                ),
                io.Int.Input(
                    "overlap_24", default=22, min=0, max=362, step=1,
                    optional=True, tooltip=SEGMENT_OVERLAP_HINT,
                ),
                io.Combo.Input(
                    "continuity_24", options=list(h3_conditioning.ROW_CONTINUITY),
                    default=h3_conditioning.AS_SET,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.Image.Input(
                    "first_frame",
                    optional=True,
                    tooltip="The frame the clip opens on, stretched to the canvas.",
                ),
                io.Image.Input(
                    "last_frame",
                    optional=True,
                    tooltip="The frame the clip closes on, cropped to cover the canvas.",
                ),
                io.Image.Input(
                    "images",
                    optional=True,
                    tooltip=BATCH_HINT,
                ),
                *REFERENCE_INPUTS,
            ],
            outputs=[
                io.Latent.Output(
                    display_name="latent",
                    tooltip="The empty latent the first segment samples into, for a loop's value slot.",
                ),
                H3_PROMPTS.Output(
                    display_name="prompts",
                    tooltip="Every segment's prompt and frame count, for H3 Extend Window.",
                ),
                io.Int.Output(
                    display_name="segments",
                    tooltip="Rows carrying a prompt, for a While Loop's count.",
                ),
                io.String.Output(
                    display_name="report",
                    tooltip="What each segment snapped to and the frames they come to.",
                ),
                io.Latent.Output(
                    display_name="latents",
                    is_output_list=True,
                    tooltip=(
                        "One empty latent per segment, at that segment's own length, so a "
                        "loop samples every clip from fresh. Wire to MiniMax H3 Clip "
                        "Select, or take one with an index node."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, clip, vae, mode, aspect_ratio, megapixels, width, height,
                prompt_header="", prompt_footer="", first_frame=None, last_frame=None,
                images=None, audio_vae=None, ref_image_size="match",
                **rows) -> io.NodeOutput:
        """Encode every filled row and size the opening segment's latent.

        Raises:
            ValueError: No row carries a prompt, there is no size to work from, a
                batched mode was given too few pictures for the rows written, or
                ``ref2va`` was given no reference or audio without an audio VAE.
        """
        import node_helpers

        filled = h3_conditioning.filled_rows(rows)
        if not filled:
            raise ValueError(
                "MiniMax H3 Conditioning has no prompt. Write what the first segment shows "
                "in the first box"
            )
        filled = [
            (h3_conditioning.composed(prompt_header, text, prompt_footer), count, overlap,
             choice)
            for text, count, overlap, choice in filled
        ]

        pictures = (images, first_frame, last_frame)
        if mode in h3_conditioning.REFERENCE_MODES:
            ref_images, ref_videos, ref_audios = h3_references.collect(rows)
            if not (ref_images or ref_videos or ref_audios):
                raise ValueError(
                    "MiniMax H3 Conditioning is set to `ref2va` and no reference is wired. "
                    "Wire a picture into ref_image_1, a clip into ref_video_1 or a sound "
                    "into ref_audio_1, or set mode to `t2va`"
                )
            pictures = (*ref_images, *(frames for _, frames, _ in ref_videos))

        width, height = h3_conditioning.canvas(
            megapixels,
            width,
            height,
            h3_conditioning.ratio_of(aspect_ratio, *pictures),
        )

        references = None
        if mode in h3_conditioning.REFERENCE_MODES:
            references = h3_references.build(
                vae, audio_vae, ref_images, ref_videos, ref_audios, width, height,
                max(h3_extend.snap_clip(count) for _, count, _, _ in filled),
                ref_image_size,
            )

        if mode in h3_conditioning.BATCHED_MODES:
            segments, drawn = cls.batched(
                clip, vae, images, width, height, filled, node_helpers
            )
            latent, frames = segments[0][3], segments[0][1]
        else:
            latent, frames = h3_conditioning.empty_latent(width, height, filled[0][1])
            wanted = h3_conditioning.MODE_IMAGES[mode]
            drawn, blocks = h3_conditioning.keyframes(
                vae,
                first_frame if "first_frame" in wanted else None,
                last_frame if "last_frame" in wanted else None,
                width,
                height,
                frames,
            )
            opening = h3_conditioning.encode(clip, filled[0][0], drawn, references)
            if blocks:
                opening = node_helpers.conditioning_set_values(
                    opening, {"minimax_keyframes": blocks}
                )
            # Every clip gets an empty latent of its own.
            segments = [(opening, frames, 0, latent, h3_conditioning.AS_SET)]
            for text, count, overlap, choice in filled[1:]:
                length = h3_conditioning.snap_segment(count)
                own, _ = h3_conditioning.empty_latent(width, height, length)
                segments.append((
                    h3_conditioning.encode(clip, text, [], references),
                    length,
                    h3_conditioning.snap_overlap_for(count, overlap),
                    own,
                    choice,
                ))
        images_drawn = drawn
        prompts = h3_conditioning.bundle(segments)

        total = sum(segment[1] for segment in segments)
        shape = ", ".join(
            f"{h3_conditioning.duration_of(segment[1])}s ({segment[1]}f)"
            + ("" if index == 0 else f"/{segment[2] or 'cut'}")
            for index, segment in enumerate(segments)
        )
        report = (
            f"{mode}, {width}x{height} ({width * height / 1e6:.2f} MP), "
            f"{len(segments)} segment(s): {shape}"
            + (f", on {len(images_drawn)} keyframe(s)" if images_drawn else "")
            + f", {h3_conditioning.duration_of(total)}s ({total} frames) in total"
            + (f"; every segment references {', '.join(references.tags)}"
               if references else "")
        )
        return io.NodeOutput(
            latent, prompts, len(segments), report,
            h3_conditioning.latents_of(prompts),
        )

    @classmethod
    def batched(cls, clip, vae, images, width, height, filled, node_helpers):
        """One segment per neighbouring pair of a batch, each pinned at both ends.

        Args:
            clip: The loaded minimax CLIP.
            vae: The H3 video VAE.
            images: The batch of pictures, or None.
            width: Canvas width in pixels.
            height: Canvas height in pixels.
            filled: The rows carrying a prompt.
            node_helpers: ComfyUI's ``node_helpers``.

        Returns:
            ``(segments, pictures)``, one segment per row and the prepared pictures.

        Raises:
            ValueError: No batch arrived, or it holds too few pictures for the rows.
        """
        if images is None or getattr(images, "shape", (0,))[0] < 2:
            raise ValueError(
                "fl2va_batched runs each segment between two pictures, so it needs a "
                "batch of at least 2 on images. Load Image Sequence reads a folder as "
                "one batch, and Image Batch Advanced joins single images"
            )
        pictures = h3_conditioning.prepared(images, width, height, "cover")
        if len(filled) > len(pictures) - 1:
            raise ValueError(
                f"{len(filled)} prompt row(s) need {len(filled) + 1} pictures and "
                f"{len(pictures)} arrived. Add pictures, or clear the rows past segment "
                f"{len(pictures) - 1}"
            )
        # Each picture is encoded once and shared by the segments either side of it.
        pinned = [vae.encode(picture) for picture in pictures]
        segments = []
        for index, (text, count, _, _) in enumerate(filled):
            length = h3_extend.snap_clip(count)
            own, _ = h3_conditioning.empty_latent(width, height, length)
            conditioning = h3_conditioning.encode(
                clip, text, [pictures[index], pictures[index + 1]]
            )
            conditioning = node_helpers.conditioning_set_values(
                conditioning,
                {"minimax_keyframes": [
                    {"resolved_frame_index": 0, "latent": pinned[index]},
                    {"resolved_frame_index": length - 1, "latent": pinned[index + 1]},
                ]},
            )
            segments.append((conditioning, length, 0, own, h3_conditioning.AS_SET))
        return segments, pictures
