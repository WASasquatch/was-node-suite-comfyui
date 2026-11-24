"""Encode a text input into conditioning with a CLIP model."""

from __future__ import annotations

from comfy_api.latest import io, ui


class TextToConditioning(io.ComfyNode):
    """Encode ``text`` with ``clip`` and emit the conditioning."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Text to Conditioning",
            display_name="Text to Conditioning",
            search_aliases=["Text to Conditioning", "clip text encode", "prompt", "encode"],
            category="WAS Suite/Text/Operations",
            description=(
                "Encode a linked prompt with a CLIP model, for prompts built by the text "
                "nodes rather than typed into a widget."
            ),
            inputs=[
                io.Clip.Input(
                    "clip",
                    tooltip=(
                        "The text encoder that turns the prompt into conditioning. Wire it "
                        "from the CLIP output of the checkpoint being sampled with, or the "
                        "prompt will be encoded for the wrong model."
                    ),
                ),
                io.String.Input(
                    "text",
                    multiline=True,
                    placeholder="Eg: a cat on a mat",
                    tooltip=(
                        "Prompt to encode with the clip input; STRING, as `a tabby cat`. Also shown on "
                        "the node after the run."
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output(
                    tooltip=(
                        "The encoded prompt, for the positive or negative input of a sampler."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, clip, text) -> io.NodeOutput:
        import nodes

        from ...modules.sampling import h3_scene_attention

        encoded = nodes.CLIPTextEncode().encode(clip=clip, text=text)
        conditioning = h3_scene_attention.with_shots(encoded[0], clip, text)
        return io.NodeOutput(conditioning, ui=ui.PreviewText(text))
