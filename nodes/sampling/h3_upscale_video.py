"""Upscaling a long video with MiniMax H3, a chunk of the clip at a time."""

from __future__ import annotations

import comfy.samplers
from comfy_api.latest import io

from ...modules import log
from ...modules.image import color_fix
from ...modules.io import rooted
from ...modules.latent import h3_upscale
from ...modules.model import h3_latent_upscaler

logger = log.get_logger("nodes.h3_upscale_video")

NODE_NAME = "H3 Upscale Video"

#: Prefix of the two clips the panel plays.
PREFIX = "was.h3_upscale"

#: The preset a mode outside the menu falls back to.
DEFAULT_MODE = "8 step (light)"

#: Choices for when tiles are guided by the source.
ANCHOR_MODES = ("auto", "on", "off")

#: The settings `manual` reveals.
MANUAL_INPUTS = [
    io.Int.Input(
        "steps", default=8, min=1, max=10000,
        tooltip="Steps each chunk runs from strength, as `8` for a turbo LoRA or `25` without.",
    ),
    io.Float.Input(
        "cfg", default=1.0, min=0.0, max=100.0, step=0.1, round=0.01,
        tooltip="Guidance, `1` for turbo LoRAs and for H3 itself.",
    ),
    io.Combo.Input(
        "sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="euler",
        tooltip="The solver each step runs, such as `euler` or `res_multistep`.",
    ),
    io.Combo.Input(
        "scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="beta",
        tooltip="How the noise level falls from step to step, such as `simple` or `beta`.",
    ),
    io.Float.Input(
        "strength", default=0.5, min=0.01, max=1.0, step=0.01,
        tooltip=(
            "Noise level each chunk starts from, on any scheduler: `0.3` sharpens; `0.5` adds "
            "detail; `0.7` redraws detail on the clip's own layout."
        ),
    ),
    io.Int.Input(
        "chunk_frames", default=175, min=22, max=362, step=17,
        tooltip=(
            "Frames refined in one pass, shared frames included, brought to 17k+5: `175` = "
            "about 7 seconds; `362` = the longest H3 window."
        ),
    ),
    io.Int.Input(
        "overlap_frames", default=17, min=0, max=340, step=17,
        tooltip=(
            "Frames each chunk repeats from the one before and holds as that chunk refined them, "
            "in whole 17s: `17` = one clip; `0` = none, and a seam can show."
        ),
    ),
    io.Int.Input(
        "max_tokens", default=64000, min=4000, max=1048000, step=1000,
        tooltip=(
            "Most tokens one tile holds; a chunk past it is split across the frame: `64000` on a "
            "24 GB card; `24000` on 12 to 16 GB."
        ),
    ),
    io.Combo.Input(
        "anchor", options=list(ANCHOR_MODES), default="auto",
        tooltip=(
            "Guides each tile by its own part of the enlarged source at the first frame of every "
            "17: `auto` when tiles split the frame; `on` always; `off` never."
        ),
    ),
]


def _mode_options() -> list:
    """The presets, then manual with its settings."""
    presets = [io.DynamicCombo.Option(key=name, inputs=[]) for name in h3_upscale.PRESETS]
    return presets + [io.DynamicCombo.Option(key=h3_upscale.MANUAL, inputs=MANUAL_INPUTS)]


def _upscale_models() -> list[str]:
    """The learned upscaler checkpoints a run can use, then bicubic."""
    return [*h3_latent_upscaler.offered(), h3_upscale.BICUBIC]


class H3UpscaleVideo(io.ComfyNode):
    """Enlarge a clip in the H3 latent and refine it chunk by chunk, writing frames to disk."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASH3UpscaleVideo",
            display_name=NODE_NAME,
            search_aliases=[
                "WASH3UpscaleVideo", NODE_NAME,
                "h3 upscale",
                "upscale video",
                "video upscale",
                "long video upscale",
                "minimax h3",
                "latent upscale",
            ],
            category="WAS Suite/Sampling",
            is_output_node=True,
            description=(
                "Upscale a video of any length with MiniMax H3: the clip is enlarged in the "
                "latent and refined a chunk at a time, each chunk continuing from the frames the "
                "one before refined, so a long clip fits one card without a seam. Frames are "
                "written to disk as each chunk settles, with the clip's own sound under them, "
                "and the panel plays the source beside the result."
            ),
            inputs=[
                io.MultiType.Input(
                    "source",
                    [io.Video, io.Image],
                    tooltip=(
                        "The clip to upscale, as a VIDEO or IMAGE frames. A VIDEO gives the "
                        "result its frame rate and sound; frames play at 24 fps with none. A "
                        "long clip comes from core Load Video, which reads frames as they are used."
                    ),
                ),
                io.Model.Input("model", tooltip="The MiniMax H3 model, after any LoRA."),
                io.Conditioning.Input(
                    "positive",
                    tooltip="What the clip shows, or how to refine it, from CLIP Text Encode.",
                ),
                io.Conditioning.Input(
                    "negative",
                    tooltip="What to avoid, read only when cfg is above 1.",
                ),
                io.Vae.Input("vae", tooltip="The H3 video VAE, for reading and writing the frames."),
                io.Combo.Input(
                    "upscale_model",
                    options=_upscale_models(),
                    tooltip=(
                        "The learned upscaler, from ComfyUI/models/latent_upscale_models; "
                        "`minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors` is fetched on "
                        "first use with features.network on. `bicubic` resizes without one."
                    ),
                ),
                io.Float.Input(
                    "scale",
                    default=2.0, min=1.0, max=4.0, step=0.05,
                    tooltip=(
                        "Size multiplier, sides rounded to 32 pixels: `1` = refine at the same "
                        "size; `2` = 1216x672 to 2432x1344."
                    ),
                ),
                io.DynamicCombo.Input(
                    "mode",
                    options=_mode_options(),
                    display_name="mode",
                    tooltip=(
                        "`light` starts each chunk at sigma 0.4 and sharpens; `creative` starts "
                        "at sigma 0.75 and redraws more. "
                        "`2` to `12 step` expect a turbo LoRA; `full 25 step` none."
                    ),
                ),
                io.Boolean.Input(
                    "color_transfer",
                    default=True,
                    tooltip=(
                        "`true` = each frame's colour and brightness are corrected against its "
                        "source frame, by color_method; `false` = frames are written as H3 "
                        "decoded them."
                    ),
                ),
                io.Combo.Input(
                    "color_method",
                    options=list(color_fix.METHODS),
                    default="wavelet",
                    tooltip=(
                        "`wavelet` = colour and brightness at coarse scales from the source, fine "
                        "detail from H3, fixing local drift too; `reinhard`, `mkl`, `histogram` "
                        "= each frame's colour statistics matched to its source's."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True,
                    tooltip=(
                        "Noise seed of the first chunk, as `0` or `42`; each later chunk adds "
                        "its number."
                    ),
                ),
                io.Combo.Input(
                    "root",
                    options=rooted.options(),
                    default=rooted.TEMP,
                    tooltip=(
                        "Which folder the frames land in: 'temp' = cleared when ComfyUI "
                        "restarts; 'output' = kept until deleted, for Load Video Cache."
                    ),
                ),
                io.String.Input(
                    "name",
                    default="frame_cache/h3_upscale",
                    tooltip=(
                        "The frames' folder below root, numbered on each run, as "
                        "`frame_cache/h3_upscale` for `frame_cache/h3_upscale_00001`."
                    ),
                ),
                io.Boolean.Input(
                    "delete_after_save",
                    default=True,
                    tooltip=(
                        "`true` = the frames are deleted once a save has written all of them, "
                        "and the node runs again on every queue; `false` = they are kept."
                    ),
                ),
                io.Vae.Input(
                    "audio_vae",
                    optional=True,
                    tooltip=(
                        "The H3 audio VAE, so each chunk is refined beside its own sound. Left "
                        "empty, silence is held. The result carries the source's sound either way."
                    ),
                ),
            ],
            outputs=[
                io.Video.Output(
                    display_name="video",
                    tooltip=(
                        "The upscaled clip with the source's sound, read from disk by Save Video "
                        "a frame at a time."
                    ),
                ),
                io.Latent.Output(
                    display_name="latent",
                    tooltip=(
                        "The refined video latent of the whole clip, padded to an H3 length and "
                        "before colour transfer, for H3 Decode Video or another pass."
                    ),
                ),
                io.Int.Output(display_name="frames", tooltip="Frames written."),
                io.String.Output(
                    display_name="report",
                    tooltip="The sizes, the chunks and how long each took.",
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, delete_after_save=False, **inputs):
        """Run on every queue while the frames are deleted after saving."""
        return float("NaN") if delete_after_save else ""

    @classmethod
    def execute(
        cls, source, model, positive, negative, vae, upscale_model=h3_upscale.BICUBIC, scale=2.0,
        mode=None, color_transfer=True, color_method="wavelet", seed=0, root=rooted.TEMP,
        name="frame_cache/h3_upscale", delete_after_save=True, audio_vae=None,
    ) -> io.NodeOutput:
        """Upscale the clip chunk by chunk and answer it from disk.

        Raises:
            ValueError: The model is not MiniMax H3, or the clip holds no frames.
            ModelUnavailable: The learned upscaler is not on disk and could not be fetched.
            PathNotAllowed: root and name settle outside every permitted write folder.
            MemoryError: The source fits neither in memory nor on a scratch drive.
        """
        from ...modules.latent import h3_extend
        from ...modules.media import clip as clips
        from ...modules.media import frame_cache, temp_video

        if not h3_extend.samples_sound(model):
            raise ValueError(
                f"{NODE_NAME} needs a MiniMax H3 model. Wire the H3 diffusion model, after any "
                "LoRA, into model."
            )
        mode = mode or {"mode": DEFAULT_MODE}
        chosen = mode.get("mode", DEFAULT_MODE)
        if chosen == h3_upscale.MANUAL:
            settings = {
                "steps": int(mode.get("steps", 8)), "cfg": float(mode.get("cfg", 1.0)),
                "sampler_name": mode.get("sampler_name", "euler"),
                "scheduler": mode.get("scheduler", "beta"),
                "strength": float(mode.get("strength", 0.5)),
                "chunk_frames": int(mode.get("chunk_frames", 175)),
                "overlap_frames": int(mode.get("overlap_frames", 17)),
                "max_tokens": int(mode.get("max_tokens", 64000)),
                "anchor": str(mode.get("anchor", "auto")),
            }
        else:
            preset = h3_upscale.PRESETS.get(chosen, h3_upscale.PRESETS[DEFAULT_MODE])
            settings = {
                "steps": preset.steps, "cfg": 1.0, "sampler_name": preset.sampler_name,
                "scheduler": preset.scheduler, "strength": preset.strength,
                "chunk_frames": None, "overlap_frames": None, "max_tokens": None, "anchor": "auto",
            }

        clip = clips.open_clip(source, NODE_NAME, compact=True)
        below, _, leaf = (name or "").replace("\\", "/").rpartition("/")
        parent = rooted.destination(root, below)
        cache = frame_cache.FrameCache.create(str(parent), leaf.strip() or "h3_upscale", clip.rate)
        try:
            result = h3_upscale.render(
                clip, model, positive, negative, vae, cache, upscale_model=str(upscale_model),
                scale=float(scale), seed=int(seed), audio_vae=audio_vae, node=NODE_NAME,
                color_method=str(color_method) if color_transfer else None,
                **settings,
            )
        except BaseException:
            written = cache.frames
            if written:
                logger.warning("%s stopped; the %d frame(s) written so far are kept at %s",
                               NODE_NAME, written, cache.folder)
            else:
                cache.delete()
            raise
        sound = clips.slice_audio(clip.audio, 0, result.frames, clip.rate)
        if sound is not None:
            cache.add_audio(sound)

        lines = [
            f"{result.size[0]}x{result.size[1]} from {result.source[0]}x{result.source[1]}, "
            f"{result.frames} frames in {len(result.chunks)} chunk(s) of up to "
            f"{result.chunk_frames} frames, {result.overlap_frames} shared",
            f"{chosen}: {settings['steps']} steps of {settings['sampler_name']} on "
            f"{settings['scheduler']} from sigma {result.start:.3f}, tiles up to "
            f"{result.max_tokens} tokens; {upscale_model}; colour "
            + (f"by {color_method} from the source" if color_transfer else "as decoded"),
        ]
        if result.disagreement is not None:
            lines.append(
                f"chunk upscales differ by {result.disagreement:.2%} of the latent's spread "
                "where they overlap"
            )
        report = "\n".join(lines + result.lines)
        logger.info("%s\n%s", NODE_NAME, report)

        video = frame_cache.CachedVideo(cache.folder, 0, cache.frames, bool(delete_after_save))
        shown = source if hasattr(source, "get_components") else clips.rebuild(
            clip, clip.frames, audio=None)
        before = temp_video.to_temp(shown, f"{PREFIX}.source")
        with video.keeping():
            after = temp_video.to_temp(video, f"{PREFIX}.result")
        ui = {"a_video": [before] if before else [], "b_video": [after] if after else []}
        return io.NodeOutput(video, {"samples": result.latent}, result.frames, report, ui=ui)
