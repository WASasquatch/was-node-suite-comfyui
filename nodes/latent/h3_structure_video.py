"""Condition MiniMax H3 on a structure video and a first frame, as SolarRenderer was trained."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.interface import run_result
from ...modules.latent import h3_structure

logger = log.get_logger("nodes.h3_structure_video")

NODE_NAME = "H3 Structure Video"


class H3StructureVideo(io.ComfyNode):
    """Encode a structure video, a first frame and a six-part prompt for H3 Structure Renderer."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASH3StructureVideo",
            display_name=NODE_NAME,
            search_aliases=[
                "WASH3StructureVideo", NODE_NAME,
                "solarrenderer",
                "solar renderer",
                "structure to video",
                "grey box",
                "depth to video",
                "skeleton to video",
                "minimax h3",
            ],
            category="WAS Suite/Latent/Video",
            description=(
                "Turn a structure video, such as a grey box, depth or skeleton render, plus an optional first "
                "frame and a written brief, into the conditioning and empty latent H3 Structure Renderer "
                "samples. The render follows the structure frame for frame and takes its look from the "
                "first frame and the prompt."
            ),
            inputs=[
                io.Clip.Input(
                    "clip",
                    tooltip="MiniMax H3's text encoder, as Load CLIP with qwen3vl_32b_minimax_h3_int8_convrot.",
                ),
                io.Vae.Input(
                    "vae",
                    tooltip="The H3 video VAE, as Load VAE with minimax_h3_video_vae_fp16.",
                ),
                io.Image.Input(
                    "structure",
                    tooltip=(
                        "The structure video: a grey box, a segmentation, skeletons, depth or a sketch, as Three "
                        "Render's images or Load Video. Cropped to 16:9 or 9:16 at the centre."
                    ),
                ),
                io.Image.Input(
                    "first_frame",
                    optional=True,
                    tooltip=(
                        "The shot's first frame, already in the finished look; its first picture is used. "
                        "Unwired = the prompt alone sets the look."
                    ),
                ),
                io.Combo.Input(
                    "structure_type",
                    options=[*h3_structure.KINDS, h3_structure.CUSTOM, h3_structure.WRITTEN],
                    default="grey box",
                    tooltip=(
                        "What the structure video shows; picks the trained sentences naming it. `described below` = "
                        "looks_like and not_shown; `prompt as written` = description is the whole prompt."
                    ),
                ),
                io.String.Input(
                    "summary",
                    multiline=True,
                    default="",
                    tooltip=(
                        "One paragraph: who does what and how the camera moves, as 'A courier cycles down a rainy "
                        "street at night; the camera tracks beside her.'"
                    ),
                ),
                io.String.Input(
                    "description",
                    multiline=True,
                    default="",
                    tooltip=(
                        "350 to 500 words: the look first (style, light, weather), then [Shot 1] and what is "
                        "seen, in order. Later shots open '[Shot 2] At 00:03.500, ...'."
                    ),
                ),
                io.String.Input(
                    "soundscape",
                    default="N/A",
                    tooltip="The ambience in a sentence, as 'rain on tarmac and distant traffic'; N/A = none.",
                ),
                io.Int.Input(
                    "length",
                    default=h3_structure.TRAINED_FRAMES,
                    min=h3_structure.LEAST_FRAMES,
                    max=h3_structure.MOST_FRAMES,
                    step=1,
                    tooltip=(
                        "Frames to render at 24 fps, at most the structure video's; snapped down to 17n + 5. "
                        "158 = 6.6 s, as trained."
                    ),
                ),
                io.Float.Input(
                    "fps",
                    default=24.0,
                    min=1.0,
                    max=240.0,
                    step=0.001,
                    advanced=True,
                    tooltip="Rate the structure video was made at; frames are dropped or held onto 24 fps. 24 = as is.",
                ),
                io.Int.Input(
                    "width",
                    default=0,
                    min=0,
                    max=2048,
                    step=32,
                    advanced=True,
                    tooltip="0 = 1344 for a wide structure video, 768 for a tall one, as trained; else a multiple of 32.",
                ),
                io.Int.Input(
                    "height",
                    default=0,
                    min=0,
                    max=2048,
                    step=32,
                    advanced=True,
                    tooltip="0 = 768 for a wide structure video, 1344 for a tall one, as trained; else a multiple of 32.",
                ),
                io.String.Input(
                    "looks_like",
                    default="",
                    advanced=True,
                    tooltip=(
                        "For `described below`: what the structure video looks like, as 'a black pencil line "
                        "sketch of the scene on white paper'."
                    ),
                ),
                io.String.Input(
                    "not_shown",
                    default="",
                    advanced=True,
                    tooltip="For `described below`: what of its look must not appear, as 'the pencil lines and white paper'.",
                ),
                io.Combo.Input(
                    "encoding",
                    options=list(h3_structure.ENCODINGS),
                    default=h3_structure.AS_TRAINED,
                    advanced=True,
                    tooltip=(
                        "`as trained` = latents drawn from the VAE's posterior at seed 42, as the LoRAs saw them; "
                        "`posterior mean` = the VAE's mean, as core's encode gives."
                    ),
                ),
                io.Vae.Input(
                    "structure_vae",
                    optional=True,
                    tooltip=(
                        "A VAE that encodes the structure video in place of vae, as taeh3: much faster, a "
                        "little less exact. The first frame keeps vae."
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output(
                    display_name="positive",
                    tooltip="The prompt, the structure video and the first frame, for KSampler's positive.",
                ),
                io.Conditioning.Output(
                    display_name="negative",
                    tooltip="The same with the prompt zeroed, for KSampler's negative; at cfg 1 it is never sampled.",
                ),
                io.Latent.Output(
                    display_name="latent",
                    tooltip="An empty video and sound latent of the render's size and length, for KSampler.",
                ),
                io.String.Output(
                    display_name="prompt",
                    tooltip="The six-part prompt as the text encoder read it.",
                ),
            ],
        )

    @classmethod
    def execute(cls, clip, vae, structure, first_frame=None, structure_type="grey box", summary="",
                description="", soundscape="N/A", length=h3_structure.TRAINED_FRAMES, fps=24.0, width=0,
                height=0, looks_like="", not_shown="", encoding=h3_structure.AS_TRAINED,
                structure_vae=None) -> io.NodeOutput:
        """Encode the structure video, the first frame and the prompt.

        Raises:
            ValueError: A part the structure type needs is empty, the structure video is too
                short, or a side is not a multiple of 32.
        """
        if bool(width) != bool(height):
            raise ValueError(f"{NODE_NAME}: set width and height together, or both to 0 for the trained canvas.")
        text = h3_structure.prompt(structure_type, summary, description, soundscape, looks_like, not_shown,
                                   first_frame is not None)
        positive, latent, facts = h3_structure.build(
            clip, vae, structure, text, int(width), int(height), int(length), float(fps), first_frame,
            structure_vae, encoding)
        words = h3_structure.words_in(description)
        low, high = h3_structure.DESCRIPTION_WORDS
        short = structure_type != h3_structure.WRITTEN and not low <= words <= high
        trained = (facts["canvas"] in ("1344x768", "768x1344")
                   and facts["frames"] == h3_structure.TRAINED_FRAMES)
        run_result.publish(
            status=run_result.WARNING if short or not trained else run_result.OK,
            summary=(f"{facts['frames']} frames at {facts['canvas']}, {structure_type}"
                     + (f"; description is {words} words, trained on {low} to {high}" if short else "")),
            counts={"frames": facts["frames"], "seconds": facts["seconds"], "latents": facts["latents"],
                    "description words": words},
            facts={"first frame": facts["first frame"], "encoding": facts["encoding"],
                   "trained on": "158 frames at 1344x768 or 768x1344"},
        )
        logger.info("%s: %s frames at %s, %s", NODE_NAME, facts["frames"], facts["canvas"], structure_type)
        return io.NodeOutput(positive, h3_structure.negative(positive), latent, text)
