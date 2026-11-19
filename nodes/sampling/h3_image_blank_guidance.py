"""A guider for MiniMax H3 Image steered by the prompt read beside a blank picture."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.sampling import h3_blank

NODE_NAME = "H3 Image Blank Guidance"


class H3ImageBlankGuidance(io.ComfyNode):
    """Encode a MiniMax H3 Image prompt alone and beside a blank picture, and guide between them."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASH3ImageBlankGuidance",
            display_name=NODE_NAME,
            search_aliases=[
                "WASH3ImageBlankGuidance", NODE_NAME,
                "minimax h3 image",
                "blank picture",
                "empty reference",
                "photographic detail",
                "texture cleanup",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Make MiniMax H3 Image text to image more photographic, with crisper skin, fur and "
                "stone, by reading the prompt beside a blank picture. Wire its guider and latent into "
                "Sampler Custom Advanced in place of MiniMax H3 Image Conditioning."
            ),
            inputs=[
                io.Model.Input("model", tooltip="The MiniMax H3 Image model."),
                io.Clip.Input("clip", tooltip="The MiniMax H3 Image text encoder."),
                io.Vae.Input("vae", tooltip="The MiniMax H3 Image VAE."),
                io.String.Input("prompt", multiline=True, tooltip="What to make, as for MiniMax H3 Image Conditioning: `A portrait of a woman in a garden`."),
                io.Int.Input("width", default=2048, min=256, max=16384, step=32, tooltip="Width of the picture: `2048`; wire Resolution Selector for 16 MP."),
                io.Int.Input("height", default=2048, min=256, max=16384, step=32, tooltip="Height of the picture: `2048`."),
                io.DynamicCombo.Input(
                    "mode",
                    options=[
                        io.DynamicCombo.Option(key=h3_blank.BLANK, inputs=[
                            io.Float.Input(
                                "strength",
                                default=1.0, min=-2.0, max=4.0, step=0.05,
                                tooltip="`1` = the prompt beside the blank, one run a step; `1.5` = further, two runs a step; `0` = off.",
                            ),
                        ]),
                        io.DynamicCombo.Option(key=h3_blank.SUBTRACTION, inputs=[
                            io.Float.Input(
                                "strength",
                                default=2.0, min=-2.0, max=6.0, step=0.05,
                                tooltip="How far past the blank's change the first steps go: `2` = twice it; `1` = as the blank; `0` = off.",
                            ),
                            io.Int.Input(
                                "steps",
                                default=2, min=1, max=6,
                                tooltip="First steps guided, of six, each run twice: `2` = sigma 1.0 and 0.97; then the prompt alone.",
                            ),
                        ]),
                    ],
                    tooltip=(
                        "`blank` = every step reads the prompt beside the blank, as adding an empty picture "
                        "does; `subtraction` = the first steps pushed past the change the blank makes. Both "
                        "change the composition."
                    ),
                ),
                io.String.Input(
                    "colour",
                    default="#808080",
                    advanced=True,
                    tooltip="Colour of the blank: `#808080` mid grey, `#000000`, `#FFFFFF`.",
                ),
            ],
            outputs=[
                io.Guider.Output(display_name="guider", tooltip="For Sampler Custom Advanced, with H3 Image Sampler as its sampler."),
                io.Latent.Output(display_name="latent", tooltip="An empty latent at width x height, for Sampler Custom Advanced."),
            ],
        )

    @classmethod
    def execute(cls, model, clip, vae, prompt, width, height, mode, colour="#808080") -> io.NodeOutput:
        """Encode both prompts and build the guider.

        Raises:
            ValueError: The mode is unknown or the prompt is empty.
        """
        import comfy.model_management

        from ...modules.image import draw

        chosen = mode.get("mode", h3_blank.BLANK) if isinstance(mode, dict) else mode
        if chosen not in h3_blank.MODES:
            raise ValueError(f"{NODE_NAME} has no mode {chosen!r}. Choose {' or '.join(h3_blank.MODES)}.")
        if not str(prompt or "").strip():
            raise ValueError(f"{NODE_NAME} has no prompt. Type what to make.")
        settings = mode if isinstance(mode, dict) else {}
        strength = float(settings.get("strength", 1.0 if chosen == h3_blank.BLANK else 2.0))
        steps = int(settings.get("steps", 2))
        red, green, blue, _alpha = draw.parse_color(colour, (128, 128, 128, 255))
        plain, blank = h3_blank.encode(clip, vae, str(prompt), (red / 255.0, green / 255.0, blue / 255.0))
        guider = h3_blank.BlankGuider(model, chosen, strength, steps)
        guider.set_conds(plain, blank)
        latent = h3_blank.empty_latent(width, height, comfy.model_management.intermediate_device())
        return io.NodeOutput(guider, latent)
