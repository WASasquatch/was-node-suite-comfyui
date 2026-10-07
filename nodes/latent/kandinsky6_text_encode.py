"""Encoding a Kandinsky 6 prompt."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.model.kandinsky6 import conditioning

NODE_NAME = "Kandinsky 6 Text Encode"


class Kandinsky6TextEncode(io.ComfyNode):
    """Encode a Kandinsky 6 prompt from what is seen and what is heard."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASKandinsky6TextEncode",
            display_name=NODE_NAME,
            search_aliases=[
                "WASKandinsky6TextEncode", NODE_NAME,
                "kandinsky prompt",
                "k6 prompt",
                "text to video audio",
                "audio caption",
            ],
            category="WAS Suite/Latent/Video",
            description=(
                "Turn a Kandinsky 6 prompt into conditioning for KSampler, with what is seen and "
                "what is heard written separately. Use one for the prompt and one for the negative, "
                "both on the text encoders DualCLIPLoader loads with type kandinsky5."
            ),
            inputs=[
                io.Clip.Input(
                    "clip",
                    tooltip="qwen_2.5_vl_7b and clip_l, from DualCLIPLoader with type 'kandinsky5'.",
                ),
                io.String.Input(
                    "video_prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    tooltip=(
                        "What is seen, as 'a red fox trots through deep snow, low tracking shot'. "
                        "Spoken lines go where they are said, as <S>Look there!<E>."
                    ),
                ),
                io.String.Input(
                    "audio_prompt",
                    multiline=True,
                    dynamic_prompts=True,
                    tooltip=(
                        "What is heard besides speech, as 'rain on a tin roof, distant thunder'. "
                        "Empty = no sound description."
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output(
                    tooltip="Conditioning for KSampler, or for Kandinsky 6 Image To Video.",
                ),
            ],
        )

    @classmethod
    def execute(cls, clip, video_prompt, audio_prompt) -> io.NodeOutput:
        return io.NodeOutput(conditioning.encode(clip, video_prompt, audio_prompt))
