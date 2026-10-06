"""One MiniMax H3 segment per row, each with its own prompt, length, assets and model."""

from __future__ import annotations

import logging

from comfy_api.latest import io

from ...modules.compat.types import H3_ASSETS, H3_PROMPTS
from ...modules.interface import segment_preview
from ...modules.latent import h3_assets, h3_conditioning, h3_extend, h3_references

MODE_HINT = (
    "The task to condition for. `t2va` = prompt only; `i2va` = opens on first_frame; "
    "`fl2va` = opens on first_frame, closes on last_frame; `fl2va_batched` = every "
    "segment between a neighbouring pair of images; `ref2va` = every segment built on the "
    "ref inputs, named `<Picture 1>`, `<Video 1>`, `<Audio 1>` in the prompt. Assets "
    "wired into assets add to any mode."
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
    "side. Read by `ref2va` and by reference assets."
)

MODEL_FL2VA_HINT = (
    "The MiniMax H3 fl2va model, which samples a segment from text and pinned frames. "
    "Optional: with it wired, H3 Extend Window and MiniMax H3 Clip Select answer it on "
    "their model output for every row set to `fl2va`, or to `auto` with no references."
)

MODEL_REF2VA_HINT = (
    "The MiniMax H3 ref2va model, which samples a segment on reference pictures, clips "
    "and sounds. Optional: with it wired, H3 Extend Window and MiniMax H3 Clip Select "
    "answer it on their model output for every row set to `ref2va`, or to `auto` with "
    "references."
)

ASSETS_HINT = (
    "The pictures, clips and sounds of a chain of MiniMax H3 Asset nodes. Each names the "
    "segment it belongs to and whether it opens or closes it, is pinned at a frame of it, "
    "or is referenced by its prompt. Wire the last asset of the chain here."
)

#: The inputs `ref2va` reads, drawn only in that mode.
REFERENCE_INPUTS = [
    io.Vae.Input(
        "audio_vae",
        optional=True,
        tooltip=(
            "The H3 audio VAE, which encodes the reference soundtracks and audio. Needed "
            "by `ref2va` when any audio is wired, and by any asset carrying sound."
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
    "from the row's own prompt. A section a row writes itself, as `visual_style:`, "
    "replaces the one here for that row. A `<d>` line here is spoken in every segment."
)

PROMPT_FOOTER_HINT = (
    "Text put after every segment's prompt, as `overall_soundscape:` and "
    "`non_diegetic_music:` for the whole run. Blank adds nothing. A section a row writes "
    "itself, as `overall_soundscape: Server hum.`, replaces the one here for that row. "
    "A `<d>` line here is spoken in every segment."
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
    "A blank row is skipped."
)

SEGMENT_DURATION_HINT = (
    "How long this segment runs, carried frames included, matching its "
    "`duration_seconds` line. The model makes 17k+5 frames at 24 fps, so `8.0` is exact, "
    "`7` snaps to 7.29 and `9` to 8.71, the nearest lengths it makes; the report states "
    "the frames each segment came to."
)

SEGMENT_SOURCE_HINT = (
    "Which segment this one continues from: `-1` = the one before, `-2` = the one before "
    "that, `2` = Segment 2, `0` = the same as `-1`. After a cutaway, `carry` picks the "
    "scene back up where it was left, with fresh sound; the new frames still join the "
    "clip's end. Ignored on segment 1."
)

SEGMENT_WRAP_HINT = (
    "Which shared text this segment's prompt is wrapped in. `both` = prompt_header and "
    "prompt_footer; `header only`; `footer only` = for a cutaway that shares the run's "
    "sound but none of the cast the header defines; `neither` = the row's prompt alone."
)

SEGMENT_CONTINUITY_HINT = (
    "How this segment follows the last. `carry` = one "
    "shot; `refresh` = re-noised; `handoff` = cut on last frame; `reference (video)` = "
    "cut, cast kept; `reference (sample)` = cut, earlier cast; `cut` = new scene; "
    "`carry (audio only)` = cut, sound kept; `carry (audio) + reference (video)` = cut, "
    "sound and cast kept."
)

SEGMENT_OVERLAP_HINT = (
    "Frames of the previous segment this one continues from, as `22` for a scene carrying "
    "on or `0` for a cut to somewhere new. `39` and `56` hold the scene harder; both "
    "sound carries take whole clips of sound, `17` for about 0.7s. Ignored on the first "
    "segment."
)

SEGMENT_STRENGTH_HINT = (
    "How firmly this segment holds its pinned frames and references, the ones a transition "
    "adds included: `1.0` = held as given; lower values add noise to them before sampling, "
    "as `0.7`. Sound references are not changed."
)

SEGMENT_SOUND_HINT = (
    "How this segment's sound follows the last: `auto` = as its transition does; `carry` = "
    "the last segment's sound runs on across the cut, under any cut picture; `fresh` = new "
    "sound, under a carried shot too."
)

SEGMENT_SEED_HINT = (
    "This segment's seed, as `0` for H3 Extend Window's seed plus the segment number, or "
    "`1234` to sample this segment from its own seed whatever the others use."
)

SEGMENT_MODEL_HINT = (
    "Which wired model samples this segment, answered on H3 Extend Window's model output. "
    "`auto` = ref2va where the segment references anything and fl2va otherwise, or "
    "whichever one model is wired; `fl2va` = model_fl2va, which holds pinned frames "
    "exactly; `ref2va` = model_ref2va. Ignored where no model is wired."
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
                "h3 prompt timeline",
                "prompt timeline",
                "storyboard",
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
                "segment with its own prompt, length, overlap and continuity, so a scene "
                "can carry on, cut somewhere new, or cut and keep its cast. `ref2va` builds "
                "every segment on shared reference pictures, clips and audio. A chain of "
                "MiniMax H3 Asset nodes wired into assets gives a segment its own opening "
                "frame, closing frame, keyframe at any frame, or references, and with both "
                "models wired in each row picks the one it samples with. The Prompt Timeline "
                "button opens a window that writes these rows and places the assets. A loop "
                "sampling one row per iteration builds the whole video, with every prompt "
                "encoded before sampling starts. Send segments to a While Loop's count and "
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
                    default=h3_conditioning.DEFAULT_CONTINUITY,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.Int.Input(
                    "source_1", default=h3_conditioning.PREVIOUS_SOURCE,
                    min=-h3_conditioning.MAX_ROWS, max=h3_conditioning.MAX_ROWS,
                    optional=True, tooltip=SEGMENT_SOURCE_HINT,
                ),
                io.Combo.Input(
                    "header_footer_1", options=list(h3_conditioning.WRAPS),
                    default=h3_conditioning.WRAPS[0],
                    optional=True, tooltip=SEGMENT_WRAP_HINT,
                ),
                io.Combo.Input(
                    "model_1", options=list(h3_conditioning.MODEL_CHOICES),
                    default=h3_conditioning.AUTO_MODEL,
                    optional=True, tooltip=SEGMENT_MODEL_HINT,
                ),
                io.Float.Input(
                    "strength_1", default=1.0, min=0.0, max=1.0, step=0.05,
                    optional=True, tooltip=SEGMENT_STRENGTH_HINT,
                ),
                io.Combo.Input(
                    "sound_1", options=list(h3_extend.SOUNDS),
                    default=h3_extend.SOUNDS[0], optional=True, tooltip=SEGMENT_SOUND_HINT,
                ),
                io.Int.Input(
                    "seed_1", default=0, min=0, max=0xFFFFFFFF,
                    optional=True, control_after_generate=False, tooltip=SEGMENT_SEED_HINT,
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
                    default=h3_conditioning.DEFAULT_CONTINUITY,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.Int.Input(
                    "source_2", default=h3_conditioning.PREVIOUS_SOURCE,
                    min=-h3_conditioning.MAX_ROWS, max=h3_conditioning.MAX_ROWS,
                    optional=True, tooltip=SEGMENT_SOURCE_HINT,
                ),
                io.Combo.Input(
                    "header_footer_2", options=list(h3_conditioning.WRAPS),
                    default=h3_conditioning.WRAPS[0],
                    optional=True, tooltip=SEGMENT_WRAP_HINT,
                ),
                io.Combo.Input(
                    "model_2", options=list(h3_conditioning.MODEL_CHOICES),
                    default=h3_conditioning.AUTO_MODEL,
                    optional=True, tooltip=SEGMENT_MODEL_HINT,
                ),
                io.Float.Input(
                    "strength_2", default=1.0, min=0.0, max=1.0, step=0.05,
                    optional=True, tooltip=SEGMENT_STRENGTH_HINT,
                ),
                io.Combo.Input(
                    "sound_2", options=list(h3_extend.SOUNDS),
                    default=h3_extend.SOUNDS[0], optional=True, tooltip=SEGMENT_SOUND_HINT,
                ),
                io.Int.Input(
                    "seed_2", default=0, min=0, max=0xFFFFFFFF,
                    optional=True, control_after_generate=False, tooltip=SEGMENT_SEED_HINT,
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
                    default=h3_conditioning.DEFAULT_CONTINUITY,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.Int.Input(
                    "source_3", default=h3_conditioning.PREVIOUS_SOURCE,
                    min=-h3_conditioning.MAX_ROWS, max=h3_conditioning.MAX_ROWS,
                    optional=True, tooltip=SEGMENT_SOURCE_HINT,
                ),
                io.Combo.Input(
                    "header_footer_3", options=list(h3_conditioning.WRAPS),
                    default=h3_conditioning.WRAPS[0],
                    optional=True, tooltip=SEGMENT_WRAP_HINT,
                ),
                io.Combo.Input(
                    "model_3", options=list(h3_conditioning.MODEL_CHOICES),
                    default=h3_conditioning.AUTO_MODEL,
                    optional=True, tooltip=SEGMENT_MODEL_HINT,
                ),
                io.Float.Input(
                    "strength_3", default=1.0, min=0.0, max=1.0, step=0.05,
                    optional=True, tooltip=SEGMENT_STRENGTH_HINT,
                ),
                io.Combo.Input(
                    "sound_3", options=list(h3_extend.SOUNDS),
                    default=h3_extend.SOUNDS[0], optional=True, tooltip=SEGMENT_SOUND_HINT,
                ),
                io.Int.Input(
                    "seed_3", default=0, min=0, max=0xFFFFFFFF,
                    optional=True, control_after_generate=False, tooltip=SEGMENT_SEED_HINT,
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
                    default=h3_conditioning.DEFAULT_CONTINUITY,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.Int.Input(
                    "source_4", default=h3_conditioning.PREVIOUS_SOURCE,
                    min=-h3_conditioning.MAX_ROWS, max=h3_conditioning.MAX_ROWS,
                    optional=True, tooltip=SEGMENT_SOURCE_HINT,
                ),
                io.Combo.Input(
                    "header_footer_4", options=list(h3_conditioning.WRAPS),
                    default=h3_conditioning.WRAPS[0],
                    optional=True, tooltip=SEGMENT_WRAP_HINT,
                ),
                io.Combo.Input(
                    "model_4", options=list(h3_conditioning.MODEL_CHOICES),
                    default=h3_conditioning.AUTO_MODEL,
                    optional=True, tooltip=SEGMENT_MODEL_HINT,
                ),
                io.Float.Input(
                    "strength_4", default=1.0, min=0.0, max=1.0, step=0.05,
                    optional=True, tooltip=SEGMENT_STRENGTH_HINT,
                ),
                io.Combo.Input(
                    "sound_4", options=list(h3_extend.SOUNDS),
                    default=h3_extend.SOUNDS[0], optional=True, tooltip=SEGMENT_SOUND_HINT,
                ),
                io.Int.Input(
                    "seed_4", default=0, min=0, max=0xFFFFFFFF,
                    optional=True, control_after_generate=False, tooltip=SEGMENT_SEED_HINT,
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
                    default=h3_conditioning.DEFAULT_CONTINUITY,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.Int.Input(
                    "source_5", default=h3_conditioning.PREVIOUS_SOURCE,
                    min=-h3_conditioning.MAX_ROWS, max=h3_conditioning.MAX_ROWS,
                    optional=True, tooltip=SEGMENT_SOURCE_HINT,
                ),
                io.Combo.Input(
                    "header_footer_5", options=list(h3_conditioning.WRAPS),
                    default=h3_conditioning.WRAPS[0],
                    optional=True, tooltip=SEGMENT_WRAP_HINT,
                ),
                io.Combo.Input(
                    "model_5", options=list(h3_conditioning.MODEL_CHOICES),
                    default=h3_conditioning.AUTO_MODEL,
                    optional=True, tooltip=SEGMENT_MODEL_HINT,
                ),
                io.Float.Input(
                    "strength_5", default=1.0, min=0.0, max=1.0, step=0.05,
                    optional=True, tooltip=SEGMENT_STRENGTH_HINT,
                ),
                io.Combo.Input(
                    "sound_5", options=list(h3_extend.SOUNDS),
                    default=h3_extend.SOUNDS[0], optional=True, tooltip=SEGMENT_SOUND_HINT,
                ),
                io.Int.Input(
                    "seed_5", default=0, min=0, max=0xFFFFFFFF,
                    optional=True, control_after_generate=False, tooltip=SEGMENT_SEED_HINT,
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
                    default=h3_conditioning.DEFAULT_CONTINUITY,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.Int.Input(
                    "source_6", default=h3_conditioning.PREVIOUS_SOURCE,
                    min=-h3_conditioning.MAX_ROWS, max=h3_conditioning.MAX_ROWS,
                    optional=True, tooltip=SEGMENT_SOURCE_HINT,
                ),
                io.Combo.Input(
                    "header_footer_6", options=list(h3_conditioning.WRAPS),
                    default=h3_conditioning.WRAPS[0],
                    optional=True, tooltip=SEGMENT_WRAP_HINT,
                ),
                io.Combo.Input(
                    "model_6", options=list(h3_conditioning.MODEL_CHOICES),
                    default=h3_conditioning.AUTO_MODEL,
                    optional=True, tooltip=SEGMENT_MODEL_HINT,
                ),
                io.Float.Input(
                    "strength_6", default=1.0, min=0.0, max=1.0, step=0.05,
                    optional=True, tooltip=SEGMENT_STRENGTH_HINT,
                ),
                io.Combo.Input(
                    "sound_6", options=list(h3_extend.SOUNDS),
                    default=h3_extend.SOUNDS[0], optional=True, tooltip=SEGMENT_SOUND_HINT,
                ),
                io.Int.Input(
                    "seed_6", default=0, min=0, max=0xFFFFFFFF,
                    optional=True, control_after_generate=False, tooltip=SEGMENT_SEED_HINT,
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
                    default=h3_conditioning.DEFAULT_CONTINUITY,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.Int.Input(
                    "source_7", default=h3_conditioning.PREVIOUS_SOURCE,
                    min=-h3_conditioning.MAX_ROWS, max=h3_conditioning.MAX_ROWS,
                    optional=True, tooltip=SEGMENT_SOURCE_HINT,
                ),
                io.Combo.Input(
                    "header_footer_7", options=list(h3_conditioning.WRAPS),
                    default=h3_conditioning.WRAPS[0],
                    optional=True, tooltip=SEGMENT_WRAP_HINT,
                ),
                io.Combo.Input(
                    "model_7", options=list(h3_conditioning.MODEL_CHOICES),
                    default=h3_conditioning.AUTO_MODEL,
                    optional=True, tooltip=SEGMENT_MODEL_HINT,
                ),
                io.Float.Input(
                    "strength_7", default=1.0, min=0.0, max=1.0, step=0.05,
                    optional=True, tooltip=SEGMENT_STRENGTH_HINT,
                ),
                io.Combo.Input(
                    "sound_7", options=list(h3_extend.SOUNDS),
                    default=h3_extend.SOUNDS[0], optional=True, tooltip=SEGMENT_SOUND_HINT,
                ),
                io.Int.Input(
                    "seed_7", default=0, min=0, max=0xFFFFFFFF,
                    optional=True, control_after_generate=False, tooltip=SEGMENT_SEED_HINT,
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
                    default=h3_conditioning.DEFAULT_CONTINUITY,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.Int.Input(
                    "source_8", default=h3_conditioning.PREVIOUS_SOURCE,
                    min=-h3_conditioning.MAX_ROWS, max=h3_conditioning.MAX_ROWS,
                    optional=True, tooltip=SEGMENT_SOURCE_HINT,
                ),
                io.Combo.Input(
                    "header_footer_8", options=list(h3_conditioning.WRAPS),
                    default=h3_conditioning.WRAPS[0],
                    optional=True, tooltip=SEGMENT_WRAP_HINT,
                ),
                io.Combo.Input(
                    "model_8", options=list(h3_conditioning.MODEL_CHOICES),
                    default=h3_conditioning.AUTO_MODEL,
                    optional=True, tooltip=SEGMENT_MODEL_HINT,
                ),
                io.Float.Input(
                    "strength_8", default=1.0, min=0.0, max=1.0, step=0.05,
                    optional=True, tooltip=SEGMENT_STRENGTH_HINT,
                ),
                io.Combo.Input(
                    "sound_8", options=list(h3_extend.SOUNDS),
                    default=h3_extend.SOUNDS[0], optional=True, tooltip=SEGMENT_SOUND_HINT,
                ),
                io.Int.Input(
                    "seed_8", default=0, min=0, max=0xFFFFFFFF,
                    optional=True, control_after_generate=False, tooltip=SEGMENT_SEED_HINT,
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
                    default=h3_conditioning.DEFAULT_CONTINUITY,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.Int.Input(
                    "source_9", default=h3_conditioning.PREVIOUS_SOURCE,
                    min=-h3_conditioning.MAX_ROWS, max=h3_conditioning.MAX_ROWS,
                    optional=True, tooltip=SEGMENT_SOURCE_HINT,
                ),
                io.Combo.Input(
                    "header_footer_9", options=list(h3_conditioning.WRAPS),
                    default=h3_conditioning.WRAPS[0],
                    optional=True, tooltip=SEGMENT_WRAP_HINT,
                ),
                io.Combo.Input(
                    "model_9", options=list(h3_conditioning.MODEL_CHOICES),
                    default=h3_conditioning.AUTO_MODEL,
                    optional=True, tooltip=SEGMENT_MODEL_HINT,
                ),
                io.Float.Input(
                    "strength_9", default=1.0, min=0.0, max=1.0, step=0.05,
                    optional=True, tooltip=SEGMENT_STRENGTH_HINT,
                ),
                io.Combo.Input(
                    "sound_9", options=list(h3_extend.SOUNDS),
                    default=h3_extend.SOUNDS[0], optional=True, tooltip=SEGMENT_SOUND_HINT,
                ),
                io.Int.Input(
                    "seed_9", default=0, min=0, max=0xFFFFFFFF,
                    optional=True, control_after_generate=False, tooltip=SEGMENT_SEED_HINT,
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
                    default=h3_conditioning.DEFAULT_CONTINUITY,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.Int.Input(
                    "source_10", default=h3_conditioning.PREVIOUS_SOURCE,
                    min=-h3_conditioning.MAX_ROWS, max=h3_conditioning.MAX_ROWS,
                    optional=True, tooltip=SEGMENT_SOURCE_HINT,
                ),
                io.Combo.Input(
                    "header_footer_10", options=list(h3_conditioning.WRAPS),
                    default=h3_conditioning.WRAPS[0],
                    optional=True, tooltip=SEGMENT_WRAP_HINT,
                ),
                io.Combo.Input(
                    "model_10", options=list(h3_conditioning.MODEL_CHOICES),
                    default=h3_conditioning.AUTO_MODEL,
                    optional=True, tooltip=SEGMENT_MODEL_HINT,
                ),
                io.Float.Input(
                    "strength_10", default=1.0, min=0.0, max=1.0, step=0.05,
                    optional=True, tooltip=SEGMENT_STRENGTH_HINT,
                ),
                io.Combo.Input(
                    "sound_10", options=list(h3_extend.SOUNDS),
                    default=h3_extend.SOUNDS[0], optional=True, tooltip=SEGMENT_SOUND_HINT,
                ),
                io.Int.Input(
                    "seed_10", default=0, min=0, max=0xFFFFFFFF,
                    optional=True, control_after_generate=False, tooltip=SEGMENT_SEED_HINT,
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
                    default=h3_conditioning.DEFAULT_CONTINUITY,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.Int.Input(
                    "source_11", default=h3_conditioning.PREVIOUS_SOURCE,
                    min=-h3_conditioning.MAX_ROWS, max=h3_conditioning.MAX_ROWS,
                    optional=True, tooltip=SEGMENT_SOURCE_HINT,
                ),
                io.Combo.Input(
                    "header_footer_11", options=list(h3_conditioning.WRAPS),
                    default=h3_conditioning.WRAPS[0],
                    optional=True, tooltip=SEGMENT_WRAP_HINT,
                ),
                io.Combo.Input(
                    "model_11", options=list(h3_conditioning.MODEL_CHOICES),
                    default=h3_conditioning.AUTO_MODEL,
                    optional=True, tooltip=SEGMENT_MODEL_HINT,
                ),
                io.Float.Input(
                    "strength_11", default=1.0, min=0.0, max=1.0, step=0.05,
                    optional=True, tooltip=SEGMENT_STRENGTH_HINT,
                ),
                io.Combo.Input(
                    "sound_11", options=list(h3_extend.SOUNDS),
                    default=h3_extend.SOUNDS[0], optional=True, tooltip=SEGMENT_SOUND_HINT,
                ),
                io.Int.Input(
                    "seed_11", default=0, min=0, max=0xFFFFFFFF,
                    optional=True, control_after_generate=False, tooltip=SEGMENT_SEED_HINT,
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
                    default=h3_conditioning.DEFAULT_CONTINUITY,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.Int.Input(
                    "source_12", default=h3_conditioning.PREVIOUS_SOURCE,
                    min=-h3_conditioning.MAX_ROWS, max=h3_conditioning.MAX_ROWS,
                    optional=True, tooltip=SEGMENT_SOURCE_HINT,
                ),
                io.Combo.Input(
                    "header_footer_12", options=list(h3_conditioning.WRAPS),
                    default=h3_conditioning.WRAPS[0],
                    optional=True, tooltip=SEGMENT_WRAP_HINT,
                ),
                io.Combo.Input(
                    "model_12", options=list(h3_conditioning.MODEL_CHOICES),
                    default=h3_conditioning.AUTO_MODEL,
                    optional=True, tooltip=SEGMENT_MODEL_HINT,
                ),
                io.Float.Input(
                    "strength_12", default=1.0, min=0.0, max=1.0, step=0.05,
                    optional=True, tooltip=SEGMENT_STRENGTH_HINT,
                ),
                io.Combo.Input(
                    "sound_12", options=list(h3_extend.SOUNDS),
                    default=h3_extend.SOUNDS[0], optional=True, tooltip=SEGMENT_SOUND_HINT,
                ),
                io.Int.Input(
                    "seed_12", default=0, min=0, max=0xFFFFFFFF,
                    optional=True, control_after_generate=False, tooltip=SEGMENT_SEED_HINT,
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
                    default=h3_conditioning.DEFAULT_CONTINUITY,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.Int.Input(
                    "source_13", default=h3_conditioning.PREVIOUS_SOURCE,
                    min=-h3_conditioning.MAX_ROWS, max=h3_conditioning.MAX_ROWS,
                    optional=True, tooltip=SEGMENT_SOURCE_HINT,
                ),
                io.Combo.Input(
                    "header_footer_13", options=list(h3_conditioning.WRAPS),
                    default=h3_conditioning.WRAPS[0],
                    optional=True, tooltip=SEGMENT_WRAP_HINT,
                ),
                io.Combo.Input(
                    "model_13", options=list(h3_conditioning.MODEL_CHOICES),
                    default=h3_conditioning.AUTO_MODEL,
                    optional=True, tooltip=SEGMENT_MODEL_HINT,
                ),
                io.Float.Input(
                    "strength_13", default=1.0, min=0.0, max=1.0, step=0.05,
                    optional=True, tooltip=SEGMENT_STRENGTH_HINT,
                ),
                io.Combo.Input(
                    "sound_13", options=list(h3_extend.SOUNDS),
                    default=h3_extend.SOUNDS[0], optional=True, tooltip=SEGMENT_SOUND_HINT,
                ),
                io.Int.Input(
                    "seed_13", default=0, min=0, max=0xFFFFFFFF,
                    optional=True, control_after_generate=False, tooltip=SEGMENT_SEED_HINT,
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
                    default=h3_conditioning.DEFAULT_CONTINUITY,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.Int.Input(
                    "source_14", default=h3_conditioning.PREVIOUS_SOURCE,
                    min=-h3_conditioning.MAX_ROWS, max=h3_conditioning.MAX_ROWS,
                    optional=True, tooltip=SEGMENT_SOURCE_HINT,
                ),
                io.Combo.Input(
                    "header_footer_14", options=list(h3_conditioning.WRAPS),
                    default=h3_conditioning.WRAPS[0],
                    optional=True, tooltip=SEGMENT_WRAP_HINT,
                ),
                io.Combo.Input(
                    "model_14", options=list(h3_conditioning.MODEL_CHOICES),
                    default=h3_conditioning.AUTO_MODEL,
                    optional=True, tooltip=SEGMENT_MODEL_HINT,
                ),
                io.Float.Input(
                    "strength_14", default=1.0, min=0.0, max=1.0, step=0.05,
                    optional=True, tooltip=SEGMENT_STRENGTH_HINT,
                ),
                io.Combo.Input(
                    "sound_14", options=list(h3_extend.SOUNDS),
                    default=h3_extend.SOUNDS[0], optional=True, tooltip=SEGMENT_SOUND_HINT,
                ),
                io.Int.Input(
                    "seed_14", default=0, min=0, max=0xFFFFFFFF,
                    optional=True, control_after_generate=False, tooltip=SEGMENT_SEED_HINT,
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
                    default=h3_conditioning.DEFAULT_CONTINUITY,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.Int.Input(
                    "source_15", default=h3_conditioning.PREVIOUS_SOURCE,
                    min=-h3_conditioning.MAX_ROWS, max=h3_conditioning.MAX_ROWS,
                    optional=True, tooltip=SEGMENT_SOURCE_HINT,
                ),
                io.Combo.Input(
                    "header_footer_15", options=list(h3_conditioning.WRAPS),
                    default=h3_conditioning.WRAPS[0],
                    optional=True, tooltip=SEGMENT_WRAP_HINT,
                ),
                io.Combo.Input(
                    "model_15", options=list(h3_conditioning.MODEL_CHOICES),
                    default=h3_conditioning.AUTO_MODEL,
                    optional=True, tooltip=SEGMENT_MODEL_HINT,
                ),
                io.Float.Input(
                    "strength_15", default=1.0, min=0.0, max=1.0, step=0.05,
                    optional=True, tooltip=SEGMENT_STRENGTH_HINT,
                ),
                io.Combo.Input(
                    "sound_15", options=list(h3_extend.SOUNDS),
                    default=h3_extend.SOUNDS[0], optional=True, tooltip=SEGMENT_SOUND_HINT,
                ),
                io.Int.Input(
                    "seed_15", default=0, min=0, max=0xFFFFFFFF,
                    optional=True, control_after_generate=False, tooltip=SEGMENT_SEED_HINT,
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
                    default=h3_conditioning.DEFAULT_CONTINUITY,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.Int.Input(
                    "source_16", default=h3_conditioning.PREVIOUS_SOURCE,
                    min=-h3_conditioning.MAX_ROWS, max=h3_conditioning.MAX_ROWS,
                    optional=True, tooltip=SEGMENT_SOURCE_HINT,
                ),
                io.Combo.Input(
                    "header_footer_16", options=list(h3_conditioning.WRAPS),
                    default=h3_conditioning.WRAPS[0],
                    optional=True, tooltip=SEGMENT_WRAP_HINT,
                ),
                io.Combo.Input(
                    "model_16", options=list(h3_conditioning.MODEL_CHOICES),
                    default=h3_conditioning.AUTO_MODEL,
                    optional=True, tooltip=SEGMENT_MODEL_HINT,
                ),
                io.Float.Input(
                    "strength_16", default=1.0, min=0.0, max=1.0, step=0.05,
                    optional=True, tooltip=SEGMENT_STRENGTH_HINT,
                ),
                io.Combo.Input(
                    "sound_16", options=list(h3_extend.SOUNDS),
                    default=h3_extend.SOUNDS[0], optional=True, tooltip=SEGMENT_SOUND_HINT,
                ),
                io.Int.Input(
                    "seed_16", default=0, min=0, max=0xFFFFFFFF,
                    optional=True, control_after_generate=False, tooltip=SEGMENT_SEED_HINT,
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
                    default=h3_conditioning.DEFAULT_CONTINUITY,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.Int.Input(
                    "source_17", default=h3_conditioning.PREVIOUS_SOURCE,
                    min=-h3_conditioning.MAX_ROWS, max=h3_conditioning.MAX_ROWS,
                    optional=True, tooltip=SEGMENT_SOURCE_HINT,
                ),
                io.Combo.Input(
                    "header_footer_17", options=list(h3_conditioning.WRAPS),
                    default=h3_conditioning.WRAPS[0],
                    optional=True, tooltip=SEGMENT_WRAP_HINT,
                ),
                io.Combo.Input(
                    "model_17", options=list(h3_conditioning.MODEL_CHOICES),
                    default=h3_conditioning.AUTO_MODEL,
                    optional=True, tooltip=SEGMENT_MODEL_HINT,
                ),
                io.Float.Input(
                    "strength_17", default=1.0, min=0.0, max=1.0, step=0.05,
                    optional=True, tooltip=SEGMENT_STRENGTH_HINT,
                ),
                io.Combo.Input(
                    "sound_17", options=list(h3_extend.SOUNDS),
                    default=h3_extend.SOUNDS[0], optional=True, tooltip=SEGMENT_SOUND_HINT,
                ),
                io.Int.Input(
                    "seed_17", default=0, min=0, max=0xFFFFFFFF,
                    optional=True, control_after_generate=False, tooltip=SEGMENT_SEED_HINT,
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
                    default=h3_conditioning.DEFAULT_CONTINUITY,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.Int.Input(
                    "source_18", default=h3_conditioning.PREVIOUS_SOURCE,
                    min=-h3_conditioning.MAX_ROWS, max=h3_conditioning.MAX_ROWS,
                    optional=True, tooltip=SEGMENT_SOURCE_HINT,
                ),
                io.Combo.Input(
                    "header_footer_18", options=list(h3_conditioning.WRAPS),
                    default=h3_conditioning.WRAPS[0],
                    optional=True, tooltip=SEGMENT_WRAP_HINT,
                ),
                io.Combo.Input(
                    "model_18", options=list(h3_conditioning.MODEL_CHOICES),
                    default=h3_conditioning.AUTO_MODEL,
                    optional=True, tooltip=SEGMENT_MODEL_HINT,
                ),
                io.Float.Input(
                    "strength_18", default=1.0, min=0.0, max=1.0, step=0.05,
                    optional=True, tooltip=SEGMENT_STRENGTH_HINT,
                ),
                io.Combo.Input(
                    "sound_18", options=list(h3_extend.SOUNDS),
                    default=h3_extend.SOUNDS[0], optional=True, tooltip=SEGMENT_SOUND_HINT,
                ),
                io.Int.Input(
                    "seed_18", default=0, min=0, max=0xFFFFFFFF,
                    optional=True, control_after_generate=False, tooltip=SEGMENT_SEED_HINT,
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
                    default=h3_conditioning.DEFAULT_CONTINUITY,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.Int.Input(
                    "source_19", default=h3_conditioning.PREVIOUS_SOURCE,
                    min=-h3_conditioning.MAX_ROWS, max=h3_conditioning.MAX_ROWS,
                    optional=True, tooltip=SEGMENT_SOURCE_HINT,
                ),
                io.Combo.Input(
                    "header_footer_19", options=list(h3_conditioning.WRAPS),
                    default=h3_conditioning.WRAPS[0],
                    optional=True, tooltip=SEGMENT_WRAP_HINT,
                ),
                io.Combo.Input(
                    "model_19", options=list(h3_conditioning.MODEL_CHOICES),
                    default=h3_conditioning.AUTO_MODEL,
                    optional=True, tooltip=SEGMENT_MODEL_HINT,
                ),
                io.Float.Input(
                    "strength_19", default=1.0, min=0.0, max=1.0, step=0.05,
                    optional=True, tooltip=SEGMENT_STRENGTH_HINT,
                ),
                io.Combo.Input(
                    "sound_19", options=list(h3_extend.SOUNDS),
                    default=h3_extend.SOUNDS[0], optional=True, tooltip=SEGMENT_SOUND_HINT,
                ),
                io.Int.Input(
                    "seed_19", default=0, min=0, max=0xFFFFFFFF,
                    optional=True, control_after_generate=False, tooltip=SEGMENT_SEED_HINT,
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
                    default=h3_conditioning.DEFAULT_CONTINUITY,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.Int.Input(
                    "source_20", default=h3_conditioning.PREVIOUS_SOURCE,
                    min=-h3_conditioning.MAX_ROWS, max=h3_conditioning.MAX_ROWS,
                    optional=True, tooltip=SEGMENT_SOURCE_HINT,
                ),
                io.Combo.Input(
                    "header_footer_20", options=list(h3_conditioning.WRAPS),
                    default=h3_conditioning.WRAPS[0],
                    optional=True, tooltip=SEGMENT_WRAP_HINT,
                ),
                io.Combo.Input(
                    "model_20", options=list(h3_conditioning.MODEL_CHOICES),
                    default=h3_conditioning.AUTO_MODEL,
                    optional=True, tooltip=SEGMENT_MODEL_HINT,
                ),
                io.Float.Input(
                    "strength_20", default=1.0, min=0.0, max=1.0, step=0.05,
                    optional=True, tooltip=SEGMENT_STRENGTH_HINT,
                ),
                io.Combo.Input(
                    "sound_20", options=list(h3_extend.SOUNDS),
                    default=h3_extend.SOUNDS[0], optional=True, tooltip=SEGMENT_SOUND_HINT,
                ),
                io.Int.Input(
                    "seed_20", default=0, min=0, max=0xFFFFFFFF,
                    optional=True, control_after_generate=False, tooltip=SEGMENT_SEED_HINT,
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
                    default=h3_conditioning.DEFAULT_CONTINUITY,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.Int.Input(
                    "source_21", default=h3_conditioning.PREVIOUS_SOURCE,
                    min=-h3_conditioning.MAX_ROWS, max=h3_conditioning.MAX_ROWS,
                    optional=True, tooltip=SEGMENT_SOURCE_HINT,
                ),
                io.Combo.Input(
                    "header_footer_21", options=list(h3_conditioning.WRAPS),
                    default=h3_conditioning.WRAPS[0],
                    optional=True, tooltip=SEGMENT_WRAP_HINT,
                ),
                io.Combo.Input(
                    "model_21", options=list(h3_conditioning.MODEL_CHOICES),
                    default=h3_conditioning.AUTO_MODEL,
                    optional=True, tooltip=SEGMENT_MODEL_HINT,
                ),
                io.Float.Input(
                    "strength_21", default=1.0, min=0.0, max=1.0, step=0.05,
                    optional=True, tooltip=SEGMENT_STRENGTH_HINT,
                ),
                io.Combo.Input(
                    "sound_21", options=list(h3_extend.SOUNDS),
                    default=h3_extend.SOUNDS[0], optional=True, tooltip=SEGMENT_SOUND_HINT,
                ),
                io.Int.Input(
                    "seed_21", default=0, min=0, max=0xFFFFFFFF,
                    optional=True, control_after_generate=False, tooltip=SEGMENT_SEED_HINT,
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
                    default=h3_conditioning.DEFAULT_CONTINUITY,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.Int.Input(
                    "source_22", default=h3_conditioning.PREVIOUS_SOURCE,
                    min=-h3_conditioning.MAX_ROWS, max=h3_conditioning.MAX_ROWS,
                    optional=True, tooltip=SEGMENT_SOURCE_HINT,
                ),
                io.Combo.Input(
                    "header_footer_22", options=list(h3_conditioning.WRAPS),
                    default=h3_conditioning.WRAPS[0],
                    optional=True, tooltip=SEGMENT_WRAP_HINT,
                ),
                io.Combo.Input(
                    "model_22", options=list(h3_conditioning.MODEL_CHOICES),
                    default=h3_conditioning.AUTO_MODEL,
                    optional=True, tooltip=SEGMENT_MODEL_HINT,
                ),
                io.Float.Input(
                    "strength_22", default=1.0, min=0.0, max=1.0, step=0.05,
                    optional=True, tooltip=SEGMENT_STRENGTH_HINT,
                ),
                io.Combo.Input(
                    "sound_22", options=list(h3_extend.SOUNDS),
                    default=h3_extend.SOUNDS[0], optional=True, tooltip=SEGMENT_SOUND_HINT,
                ),
                io.Int.Input(
                    "seed_22", default=0, min=0, max=0xFFFFFFFF,
                    optional=True, control_after_generate=False, tooltip=SEGMENT_SEED_HINT,
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
                    default=h3_conditioning.DEFAULT_CONTINUITY,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.Int.Input(
                    "source_23", default=h3_conditioning.PREVIOUS_SOURCE,
                    min=-h3_conditioning.MAX_ROWS, max=h3_conditioning.MAX_ROWS,
                    optional=True, tooltip=SEGMENT_SOURCE_HINT,
                ),
                io.Combo.Input(
                    "header_footer_23", options=list(h3_conditioning.WRAPS),
                    default=h3_conditioning.WRAPS[0],
                    optional=True, tooltip=SEGMENT_WRAP_HINT,
                ),
                io.Combo.Input(
                    "model_23", options=list(h3_conditioning.MODEL_CHOICES),
                    default=h3_conditioning.AUTO_MODEL,
                    optional=True, tooltip=SEGMENT_MODEL_HINT,
                ),
                io.Float.Input(
                    "strength_23", default=1.0, min=0.0, max=1.0, step=0.05,
                    optional=True, tooltip=SEGMENT_STRENGTH_HINT,
                ),
                io.Combo.Input(
                    "sound_23", options=list(h3_extend.SOUNDS),
                    default=h3_extend.SOUNDS[0], optional=True, tooltip=SEGMENT_SOUND_HINT,
                ),
                io.Int.Input(
                    "seed_23", default=0, min=0, max=0xFFFFFFFF,
                    optional=True, control_after_generate=False, tooltip=SEGMENT_SEED_HINT,
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
                    default=h3_conditioning.DEFAULT_CONTINUITY,
                    optional=True, tooltip=SEGMENT_CONTINUITY_HINT,
                ),
                io.Int.Input(
                    "source_24", default=h3_conditioning.PREVIOUS_SOURCE,
                    min=-h3_conditioning.MAX_ROWS, max=h3_conditioning.MAX_ROWS,
                    optional=True, tooltip=SEGMENT_SOURCE_HINT,
                ),
                io.Combo.Input(
                    "header_footer_24", options=list(h3_conditioning.WRAPS),
                    default=h3_conditioning.WRAPS[0],
                    optional=True, tooltip=SEGMENT_WRAP_HINT,
                ),
                io.Combo.Input(
                    "model_24", options=list(h3_conditioning.MODEL_CHOICES),
                    default=h3_conditioning.AUTO_MODEL,
                    optional=True, tooltip=SEGMENT_MODEL_HINT,
                ),
                io.Float.Input(
                    "strength_24", default=1.0, min=0.0, max=1.0, step=0.05,
                    optional=True, tooltip=SEGMENT_STRENGTH_HINT,
                ),
                io.Combo.Input(
                    "sound_24", options=list(h3_extend.SOUNDS),
                    default=h3_extend.SOUNDS[0], optional=True, tooltip=SEGMENT_SOUND_HINT,
                ),
                io.Int.Input(
                    "seed_24", default=0, min=0, max=0xFFFFFFFF,
                    optional=True, control_after_generate=False, tooltip=SEGMENT_SEED_HINT,
                ),
                io.Model.Input("model_fl2va", optional=True, tooltip=MODEL_FL2VA_HINT),
                io.Model.Input("model_ref2va", optional=True, tooltip=MODEL_REF2VA_HINT),
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
                H3_ASSETS.Input("assets", optional=True, tooltip=ASSETS_HINT),
                io.Boolean.Input(
                    "loop", default=False, optional=True,
                    tooltip=(
                        "`true` = the last segment closes on the video's first frame, so the video "
                        "plays round as a loop; `false` = it ends where its prompt takes it."
                    ),
                ),
                io.Clip.Input(
                    "vlm_clip", optional=True, lazy=True,
                    tooltip=(
                        "A language model from Load CLIP, as `qwen3vl_8b_fp8_scaled.safetensors`, "
                        "for the Prompt Timeline's Write scenes. Never loaded by a render of this "
                        "node."
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output(
                    display_name="latent",
                    tooltip="The empty latent the first segment samples into, for a loop's value slot.",
                ),
                H3_PROMPTS.Output(
                    display_name="prompts",
                    tooltip=(
                        "Every segment's prompt, frame count, assets and model, for H3 "
                        "Extend Window."
                    ),
                ),
                io.Int.Output(
                    display_name="segments",
                    tooltip="Rows carrying a prompt, for a While Loop's count.",
                ),
                io.String.Output(
                    display_name="report",
                    tooltip=(
                        "What each segment snapped to, the frames they come to, and the "
                        "assets and model each one was given."
                    ),
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
            hidden=[io.Hidden.unique_id, io.Hidden.extra_pnginfo],
        )

    @classmethod
    def check_lazy_status(cls, **inputs) -> list[str]:
        """The lazy inputs a run needs: none, ``vlm_clip`` being for Write scenes alone."""
        return []

    @classmethod
    def execute(cls, clip, vae, mode, aspect_ratio, megapixels, width, height,
                prompt_header="", prompt_footer="", model_fl2va=None, model_ref2va=None,
                first_frame=None, last_frame=None, images=None, audio_vae=None,
                ref_image_size="match", assets=None, loop=False, vlm_clip=None,
                **rows) -> io.NodeOutput:
        """Encode every filled row and size the opening segment's latent.

        Raises:
            ValueError: No row carries a prompt, there is no size to work from, a
                batched mode was given too few pictures for the rows written, ``ref2va``
                was given no reference, audio arrived without an audio VAE, or an asset
                names a segment the run does not have or lands outside its window, or
                assets carries something other than a chain of assets.
        """
        import comfy.utils
        import node_helpers

        filled = h3_conditioning.filled_rows(rows)
        if not filled:
            raise ValueError(
                "MiniMax H3 Conditioning has no prompt. Write what the first segment shows "
                "in the first box"
            )
        wraps = h3_conditioning.wraps_of(rows)
        choices = h3_conditioning.models_of(rows)
        holds = h3_conditioning.strengths_of(rows)
        sounds = h3_conditioning.sounds_of(rows)
        seeds = h3_conditioning.seeds_of(rows)
        encodings = []
        assets = h3_assets.collect(assets, "assets on MiniMax H3 Conditioning")
        stray = h3_assets.past_segment(assets, len(filled))
        if stray:
            named = ", ".join(f"{asset.name} (segment {asset.segment})" for asset in stray)
            raise ValueError(
                f"{len(stray)} asset(s) name a segment past the {len(filled)} carrying a "
                f"prompt: {named}. Write that segment's prompt, or set the asset's segment "
                f"to one that has one"
            )
        mismatched = [
            f"row {index + 1} states {stated:g}s and its duration is "
            f"{h3_conditioning.duration_of(count):g}s"
            for index, (text, count, _, _) in enumerate(filled)
            if (stated := h3_conditioning.stated_seconds(text)) is not None
            and abs(stated - h3_conditioning.duration_of(count)) > 0.05
        ]
        filled = [
            (h3_conditioning.carried_timeline(text, count, overlap, choice, index == 0),
             count, overlap, choice)
            for index, (text, count, overlap, choice) in enumerate(filled)
        ]
        filled = [
            (h3_conditioning.composed(
                prompt_header if h3_conditioning.takes_header(wrap) else "",
                text,
                prompt_footer if h3_conditioning.takes_footer(wrap) else "",
            ), count, overlap, choice)
            for (text, count, overlap, choice), wrap in zip(filled, wraps)
        ]

        pictures = (images, first_frame, last_frame)
        ref_images, ref_videos, ref_audios = [], [], []
        referencing = [asset for asset in assets if asset.role in h3_assets.REFERENCING]
        if mode in h3_conditioning.REFERENCE_MODES:
            ref_images, ref_videos, ref_audios = h3_references.collect(rows)
            if not (ref_images or ref_videos or ref_audios or referencing):
                raise ValueError(
                    "MiniMax H3 Conditioning is set to `ref2va` and no reference is wired. "
                    "Wire a picture into ref_image_1, a clip into ref_video_1 or a sound "
                    "into ref_audio_1, wire a MiniMax H3 Asset set to a reference role into "
                    "assets, or set mode to `t2va`"
                )
            pictures = (*ref_images, *(frames for _, frames, _ in ref_videos))
        pictures = (*pictures, *(asset.frames for asset in assets if asset.frames is not None))

        width, height = h3_conditioning.canvas(
            megapixels,
            width,
            height,
            h3_conditioning.ratio_of(aspect_ratio, *pictures),
        )
        longest = max(h3_extend.snap_clip(count) for _, count, _, _ in filled)

        if audio_vae is None and (
            ref_audios or any(sound is not None for _, _, sound in ref_videos)
        ):
            raise ValueError(
                "a reference audio or video soundtrack is wired and audio_vae is not, so "
                "there is nothing to encode it with. Wire the H3 audio VAE into audio_vae"
            )
        # Every reference is encoded once, whichever segments carry it.
        shared = (
            [h3_references.encode_picture(vae, picture, width, height, ref_image_size)
             for picture in ref_images],
            [h3_references.encode_video(vae, audio_vae, frames, sound, longest,
                                        h3_references.video_name(slot))
             for slot, frames, sound in ref_videos],
            [h3_references.encode_sound(audio_vae, audio, h3_references.audio_name(number))
             for number, audio in enumerate(ref_audios, start=1)],
        )
        encoded = {}

        def references_for(index):
            own = h3_assets.for_segment(assets, index)
            pictures_, videos_, sounds_ = h3_assets.reference_parts(
                vae, audio_vae, own, width, height, longest, ref_image_size, encoded
            )
            parts = (shared[0] + pictures_, shared[1] + videos_, shared[2] + sounds_)
            if not any(parts):
                return None
            return h3_references.assemble(*parts, named=f"segment {index + 1}")

        notes = []
        pinned_count = 0
        wired = {h3_conditioning.FL2VA: model_fl2va, h3_conditioning.REF2VA: model_ref2va}
        progress = comfy.utils.ProgressBar(len(filled))
        if mode in h3_conditioning.BATCHED_MODES:
            pinned = [asset for asset in assets if asset.role in h3_assets.PINNING]
            if pinned:
                named = ", ".join(asset.name for asset in pinned)
                raise ValueError(
                    f"fl2va_batched takes every keyframe from images, so {named} has nowhere "
                    f"to go. Set mode to `fl2va`, or set that asset's role to a reference"
                )
            segments, drawn = cls.batched(
                clip, vae, images, width, height, filled, node_helpers, references_for,
                choices, model_fl2va, model_ref2va, encodings, progress,
            )
            latent, frames = segments[0][3], segments[0][1]
            pinned_count = len(drawn)
            for index, segment in enumerate(segments):
                notes.append(cls.describe(index, segment, [], references_for(index)))
        else:
            segments = []
            wanted = h3_conditioning.MODE_IMAGES[mode]
            for index, (text, count, overlap, choice) in enumerate(filled):
                opening = index == 0
                closing = index == len(filled) - 1
                shape = h3_extend.geometry_of(choice, sounds[index])
                head, window = h3_conditioning.window_of(count, overlap, shape, opening)
                own = h3_assets.for_segment(assets, index)
                shown, blocks, lines = [], [], []
                if opening or closing:
                    # An asset opening segment 1 or closing the last replaces the socket's picture.
                    taken = {asset.role for asset in own
                             if asset.role in (h3_assets.FIRST_FRAME, h3_assets.LAST_FRAME)}
                    shown, blocks = h3_conditioning.keyframes(
                        vae,
                        first_frame if opening and "first_frame" in wanted
                        and h3_assets.FIRST_FRAME not in taken else None,
                        last_frame if closing and "last_frame" in wanted
                        and h3_assets.LAST_FRAME not in taken else None,
                        width,
                        height,
                        window,
                    )
                more_shown, more_blocks, lines = h3_assets.keyframes(
                    vae, audio_vae, own, width, height, head, window
                )
                shown, blocks = shown + more_shown, blocks + more_blocks
                references = references_for(index)
                conditioning = h3_conditioning.encode(clip, text, shown, references)
                encodings.append(h3_conditioning.encoding_of(clip, text, shown, references))
                if blocks:
                    conditioning = node_helpers.conditioning_set_values(
                        conditioning, {"minimax_keyframes": blocks}
                    )
                model, kind = h3_conditioning.pick_model(
                    choices[index], model_fl2va, model_ref2va, references is not None
                )
                if opening:
                    own_latent, length = h3_conditioning.empty_latent(width, height, count)
                    carried, continuity = 0, h3_conditioning.DEFAULT_CONTINUITY
                else:
                    length = h3_conditioning.snap_segment(count, overlap, shape)
                    # Sampled on its own, a segment runs the whole window its prompt was timed to.
                    own_latent, _ = h3_conditioning.empty_latent(width, height, window)
                    carried, continuity = h3_conditioning.snap_overlap_for(count, overlap), choice
                segment = (conditioning, length, carried, own_latent, continuity, model, kind,
                           choices[index], wired)
                segments.append(segment)
                pinned_count += len(blocks)
                notes.append(cls.describe(index, segment, lines, references))
                progress.update(1)
            latent, frames = segments[0][3], segments[0][1]
        segments = [(h3_conditioning.held(segment[0], hold),) + tuple(segment[1:])
                    for segment, hold in zip(segments, holds)]
        for index, hold in enumerate(holds):
            if hold < 1.0 and index < len(notes):
                notes[index] = (f"{notes[index]}; " if notes[index] else f"segment {index + 1}: ") + f"hold {hold:.2f}"
        prompts = h3_conditioning.bundle(segments)
        for entry, encoding in zip(prompts, encodings):
            entry[h3_conditioning.ENCODING_KEY] = encoding
        for index, (entry, sound, seed) in enumerate(zip(prompts, sounds, seeds)):
            entry["sound"] = sound
            entry["seed"] = seed
            entry[h3_assets.MOMENTS_KEY] = h3_assets.moments_for(assets, index)
            entry[h3_conditioning.LOOP_KEY] = bool(loop) and index == len(prompts) - 1 and index > 0
        owner = segment_preview.owner_of(cls.hidden.unique_id, cls.hidden.extra_pnginfo)
        for entry, source in zip(prompts, h3_conditioning.sources_of(rows)):
            entry["source"] = source
            entry[h3_conditioning.OWNER_KEY] = owner

        total = sum(segment[1] for segment in segments)
        # A cut keeps the clip before it on a whole clip, so those frames leave the earlier scene.
        trims = [
            h3_extend.CLIP_LEAD
            if h3_assets.cuts_into(segment[4], segment[2], source, index) else 0
            for index, (segment, source) in enumerate(zip(segments, h3_conditioning.sources_of(rows)))
        ] + [0]
        on_screen = [segment[1] + trims[index] - trims[index + 1] for index, segment in enumerate(segments)]
        shape = ", ".join(
            f"{h3_conditioning.duration_of(frames)}s ({frames}f)"
            + ("" if index == 0 else f"/{cls.joined(segment)}")
            for index, (segment, frames) in enumerate(zip(segments, on_screen))
        )
        report = (
            f"{mode}, {width}x{height} ({width * height / 1e6:.2f} MP), "
            f"{len(segments)} segment(s): {shape}"
            + (f", on {pinned_count} keyframe(s)" if pinned_count else "")
            + f", {h3_conditioning.duration_of(total)}s ({total} frames) in total"
            + "".join(f"; {note}" for note in notes if note)
        )
        shared_dialogue = {
            name: h3_conditioning.dialogue_blocks(text)
            for name, text in (("prompt_header", prompt_header), ("prompt_footer", prompt_footer))
        }
        for name, count in shared_dialogue.items():
            if not count:
                continue
            warning = (
                f"{name} holds {count} `<d>` dialogue tag(s), and {name} is added to every "
                f"segment, so every segment is given that line to speak. A segment with no "
                f"dialogue of its own fills it with speech. Describe dialogue without `<d>` "
                f"in {name}, and write each `<d>` line in the row that speaks it"
            )
            logging.warning("MiniMax H3 Conditioning: %s.", warning)
            report += f"; WARNING: {warning}"
        if mismatched:
            warning = (
                f"{'; '.join(mismatched)}. The model paces a segment by its duration, so a "
                f"timeline written for another length runs short or long. Set the duration "
                f"to the length the prompt describes"
            )
            logging.warning("MiniMax H3 Conditioning: %s.", warning)
            report += f"; WARNING: {warning}"
        return io.NodeOutput(
            latent, prompts, len(segments), report,
            h3_conditioning.latents_of(prompts),
        )

    @staticmethod
    def joined(segment) -> str:
        """How a segment joins the clip, for the report.

        Args:
            segment: The segment tuple, its carried frames third and its continuity fifth.

        Returns:
            The carried frame count, or ``cut`` where nothing is carried.
        """
        continuity = segment[4] if len(segment) > 4 else h3_conditioning.DEFAULT_CONTINUITY
        if not segment[2] or continuity in h3_conditioning.CUT_LIKE:
            return "cut"
        return str(segment[2])

    @staticmethod
    def describe(index, segment, lines, references) -> str:
        """One segment's assets and model, for the report.

        Args:
            index: Segment number, from 0.
            segment: The segment tuple, its model's kind seventh.
            lines: What the segment's pinning assets contributed.
            references: The segment's references, or ``None``.

        Returns:
            A clause naming the segment, or an empty string where it has nothing to say.
        """
        parts = list(lines)
        if references is not None and references.tags:
            parts.append("references " + ", ".join(references.tags))
        kind = segment[6] if len(segment) > 6 else ""
        if kind:
            wired = len(segment) > 5 and segment[5] is not None
            parts.append(f"model {kind}" + ("" if wired else " (not wired)"))
        if not parts:
            return ""
        return f"segment {index + 1}: " + "; ".join(parts)

    @classmethod
    def batched(cls, clip, vae, images, width, height, filled, node_helpers, references_for,
                choices, model_fl2va, model_ref2va, encodings=None, progress=None):
        """One segment per neighbouring pair of a batch, each pinned at both ends.

        Args:
            clip: The loaded minimax CLIP.
            vae: The H3 video VAE.
            images: The batch of pictures, or None.
            width: Canvas width in pixels.
            height: Canvas height in pixels.
            filled: The rows carrying a prompt.
            node_helpers: ComfyUI's ``node_helpers``.
            references_for: Given a segment number from 0, its references or ``None``.
            choices: Each row's model choice.
            model_fl2va: The wired fl2va model, or ``None``.
            model_ref2va: The wired ref2va model, or ``None``.
            encodings: A list each segment's :func:`h3_conditioning.encoding_of` is added to.
            progress: A ComfyUI ``ProgressBar`` advanced once per segment, or ``None``.

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
            references = references_for(index)
            conditioning = h3_conditioning.encode(
                clip, text, [pictures[index], pictures[index + 1]], references
            )
            if encodings is not None:
                encodings.append(h3_conditioning.encoding_of(
                    clip, text, [pictures[index], pictures[index + 1]], references))
            conditioning = node_helpers.conditioning_set_values(
                conditioning,
                {"minimax_keyframes": [
                    {"resolved_frame_index": 0, "latent": pinned[index]},
                    {"resolved_frame_index": length - 1, "latent": pinned[index + 1],
                     h3_conditioning.CLOSING_KEY: True},
                ]},
            )
            model, kind = h3_conditioning.pick_model(
                choices[index], model_fl2va, model_ref2va, references is not None
            )
            segments.append((conditioning, length, 0, own, "cut", model, kind,
                             choices[index],
                             {h3_conditioning.FL2VA: model_fl2va, h3_conditioning.REF2VA: model_ref2va}))
            if progress is not None:
                progress.update(1)
        return segments, pictures
