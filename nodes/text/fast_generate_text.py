"""Generate text with a CLIP's language model on ComfyUI's fast decode path."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.interface import run_result
from ...modules.model import text_decode

logger = log.get_logger("nodes.text")

#: The largest seed the sampler takes.
MAX_SEED = 0xFFFFFFFFFFFFFFFF

#: The choices for mtp, as the words on the widget.
MTP_CHOICES = ["auto", "off", "2", "3", "4", "5"]


class FastGenerateText(io.ComfyNode):
    """Generate Text, decoding on the fixed-cache graph path where the model allows it."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        sampling = [
            io.DynamicCombo.Option(
                key="on",
                inputs=[
                    io.Float.Input(
                        "temperature", default=0.7, min=0.01, max=2.0, step=0.01,
                        tooltip="How adventurous each pick is. `0.7` is balanced, `0.2` stays close to the likeliest words, `1.2` wanders.",
                    ),
                    io.Int.Input(
                        "top_k", default=64, min=0, max=1000,
                        tooltip="Pick only among this many likeliest tokens. `0` turns the limit off.",
                    ),
                    io.Float.Input(
                        "top_p", default=0.95, min=0.0, max=1.0, step=0.01,
                        tooltip="Pick only among the likeliest tokens whose chances add up to this. `1.0` turns it off.",
                    ),
                    io.Float.Input(
                        "min_p", default=0.05, min=0.0, max=1.0, step=0.01,
                        tooltip="Drop any token less likely than this fraction of the likeliest one. `0` turns it off.",
                    ),
                    io.Float.Input(
                        "repetition_penalty", default=1.05, min=0.0, max=5.0, step=0.01,
                        tooltip="Above `1.0` makes a token already used less likely again. `1.0` turns it off.",
                    ),
                    io.Int.Input(
                        "seed", default=0, min=0, max=MAX_SEED,
                        tooltip="The same seed with the same settings writes the same text.",
                    ),
                    io.Float.Input(
                        "presence_penalty", default=0.0, min=0.0, max=5.0, step=0.01, optional=True,
                        tooltip="Subtracted from every token already used. `0` turns it off, `0.5` pushes toward new words.",
                    ),
                ],
            ),
            io.DynamicCombo.Option(key="off", inputs=[]),
        ]
        return io.Schema(
            node_id="WASFastGenerateText",
            display_name="Fast Generate Text",
            search_aliases=[
                "WASFastGenerateText",
                "Fast Generate Text",
                "generate text",
                "LLM",
                "prompt enhance",
                "qwen",
                "gemma",
            ],
            category="WAS Suite/Text",
            description=(
                "Write text with the language model inside a loaded CLIP, as core Generate Text "
                "does, several times faster on the models core leaves on its slow decode. It "
                "switches the model onto ComfyUI's own fixed cache and graph captured decode for "
                "the run and back afterwards. The panel shows tokens per second and which decode "
                "ran."
            ),
            inputs=[
                io.Clip.Input(
                    "clip",
                    tooltip="A CLIP holding a language model, such as Qwen3 4B from Load CLIP with type `lumina2`, or Gemma 3 with type `ltxv`.",
                ),
                io.String.Input(
                    "prompt", multiline=True, dynamic_prompts=True, default="",
                    tooltip="What to ask the model, such as `Describe a lighthouse on a stormy night.`",
                ),
                io.Image.Input(
                    "image", optional=True,
                    tooltip="A picture to ask about, for a model that reads images such as Qwen 2.5 VL or Gemma 3.",
                ),
                io.Image.Input(
                    "video", optional=True,
                    tooltip="Video frames as an image batch, read as 24 fps and sampled at 1 fps.",
                ),
                io.Audio.Input(
                    "audio", optional=True,
                    tooltip="Sound to ask about, for a model that hears it.",
                ),
                io.Int.Input(
                    "max_length", default=512, min=1, max=32768,
                    tooltip="The most tokens to write. A token is about three quarters of a word, so `512` is about 380 words.",
                ),
                io.DynamicCombo.Input(
                    "sampling_mode", options=sampling, display_name="Sampling Mode",
                    tooltip="`on` draws each token at random within the limits below. `off` always takes the likeliest token and writes the same text every time.",
                ),
                io.Boolean.Input(
                    "thinking", default=False, optional=True,
                    tooltip="`true` lets a model that reasons, such as Qwen3, think before answering.",
                ),
                io.Boolean.Input(
                    "use_default_template", default=True, optional=True, advanced=True,
                    tooltip="`true` wraps the prompt in the model's own chat template and system prompt.",
                ),
                io.Combo.Input(
                    "mtp", options=MTP_CHOICES, default="auto", optional=True, advanced=True,
                    tooltip="Multi-token prediction for a checkpoint carrying those heads. `auto` picks the draft depth, `2` to `5` fixes it, `off` turns it off.",
                ),
                io.Boolean.Input(
                    "fast_decode", default=True, optional=True,
                    tooltip="`true` decodes on the fixed cache graph path where the model allows it; `false` runs exactly as core Generate Text.",
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="generated_text",
                    tooltip="What the model wrote, special tokens removed.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        clip,
        prompt,
        max_length,
        sampling_mode,
        image=None,
        video=None,
        audio=None,
        thinking=False,
        use_default_template=True,
        mtp="auto",
        fast_decode=True,
    ) -> io.NodeOutput:
        """Tokenize, generate and decode.

        Args:
            clip: A CLIP holding a language model.
            prompt: What to ask.
            max_length: The most tokens to generate.
            sampling_mode: The sampling choice and its settings.
            image: An optional picture.
            video: Optional frames.
            audio: Optional sound.
            thinking: Whether a reasoning model thinks first.
            use_default_template: Whether the model's chat template wraps the prompt.
            mtp: Multi-token prediction depth.
            fast_decode: Whether to take the fast decode path.

        Returns:
            The generated text.
        """
        if clip is None:
            raise ValueError(
                "Fast Generate Text needs a CLIP. Connect Load CLIP with a language model such "
                "as qwen_3_4b.safetensors."
            )
        tokens = clip.tokenize(
            prompt, image=image, skip_template=not use_default_template, min_length=1,
            thinking=thinking, video=video, audio=audio,
        )
        sampling_mode = sampling_mode or {}
        options = {
            "do_sample": sampling_mode.get("sampling_mode") == "on",
            "temperature": sampling_mode.get("temperature", 1.0),
            "top_k": sampling_mode.get("top_k", 50),
            "top_p": sampling_mode.get("top_p", 1.0),
            "min_p": sampling_mode.get("min_p", 0.0),
            "repetition_penalty": sampling_mode.get("repetition_penalty", 1.0),
            "presence_penalty": sampling_mode.get("presence_penalty", 0.0),
            "seed": sampling_mode.get("seed", None),
            "mtp": False if mtp == "off" else (True if mtp == "auto" else int(mtp)),
        }

        ids, report = text_decode.generate(clip, tokens, max_length, fast=fast_decode, **options)
        text = clip.decode(ids)

        rate = report.tokens_per_second
        logger.info(
            "Fast Generate Text wrote %d token(s) in %.2fs, %.1f tokens/s, on %s%s",
            report.new_tokens, report.seconds, rate, report.path,
            f" ({report.reason})" if report.reason else "",
        )
        facts = {"decode": report.path}
        if report.reason:
            facts["why not fast"] = report.reason
        facts["prompt tokens"] = report.prompt_tokens
        if report.extras.get("graph_units") is not None:
            facts["graph blocks"] = report.extras["graph_units"]
        run_result.publish(
            status=run_result.OK if not report.reason or not fast_decode else run_result.WARNING,
            summary=f"{report.new_tokens} tokens at {rate:.1f} tokens/s",
            counts={"tokens": report.new_tokens, "seconds": round(report.seconds, 2)},
            facts=facts,
            bodies=run_result.body("text", text),
        )
        return io.NodeOutput(text)
