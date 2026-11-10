"""Sample MiniMax H3 Image with fresh noise fed into its last steps."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.sampling import h3_image

NODE_NAME = "H3 Image Sampler"
STANDARD = "standard"
PROGRESSIVE = "progressive"


class H3ImageSampler(io.ComfyNode):
    """A sampler for MiniMax H3 Image: Euler steps, the last ones taken as lcm steps."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASH3ImageSampler",
            display_name=NODE_NAME,
            search_aliases=[
                "WASH3ImageSampler", NODE_NAME,
                "minimax h3 image",
                "pdd sampler",
                "euler lcm",
                "progressive sampling",
                "spectral progressive",
                "high resolution texture",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Sample MiniMax H3 Image at 4 to 16 megapixels with cleaner fine texture. By "
                "default the first steps run on a smaller latent that grows partway, and the step "
                "into the final one takes fresh noise. Use it with Sampler Custom on six steps of "
                "the simple scheduler."
            ),
            inputs=[
                io.DynamicCombo.Input(
                    "mode",
                    options=[
                        io.DynamicCombo.Option(key=PROGRESSIVE, inputs=[
                            io.Float.Input(
                                "start_scale",
                                default=0.5, min=0.25, max=0.9, step=0.05,
                                tooltip="Share of each side the first steps run at: `0.5` = 16 MP starts at 4 MP; `0.75` = at 9 MP.",
                            ),
                            io.Int.Input(
                                "grow_at",
                                default=3, min=1, max=5,
                                tooltip=(
                                    "Steps run small before growing to full size: `3` of six = through "
                                    "sigma 0.75, then one step repeated at full size; `2` = through 0.89."
                                ),
                            ),
                        ]),
                        io.DynamicCombo.Option(key=STANDARD, inputs=[]),
                    ],
                    tooltip=(
                        "`progressive` = the first steps on a smaller latent, grown to full size "
                        "partway; `standard` = every step at full size, with the composition of the "
                        "model's own Euler sampler."
                    ),
                ),
                io.Int.Input(
                    "renoise_steps",
                    default=1, min=0, max=16,
                    tooltip=(
                        "Last steps that re-noise the prediction with fresh noise, as lcm does. "
                        "`0` = the model's Euler sampler exactly; `1` = only the step into the "
                        "final one; `5` on six steps = lcm."
                    ),
                ),
                io.Float.Input(
                    "noise_scale",
                    default=1.0, min=0.5, max=1.5, step=0.01,
                    tooltip=(
                        "Fresh noise in a renoised step. `1.0` = as lcm; `0.9` = softer texture; "
                        "`1.1` = crisper fine detail; `1.2` = close to Euler's crunch again."
                    ),
                ),
            ],
            outputs=[
                io.Sampler.Output(
                    display_name="sampler",
                    tooltip="For Sampler Custom with MiniMax H3 Image, on six steps of the simple scheduler.",
                ),
            ],
        )

    @classmethod
    def execute(cls, mode, renoise_steps=1, noise_scale=1.0) -> io.NodeOutput:
        """Build the sampler.

        Raises:
            ValueError: The mode is unknown.
        """
        import comfy.samplers

        chosen = mode.get("mode", PROGRESSIVE) if isinstance(mode, dict) else mode
        options = {"renoise_steps": int(renoise_steps), "noise_scale": float(noise_scale)}
        if chosen == PROGRESSIVE:
            options.update(start_scale=float(mode.get("start_scale", 0.5)), grow_at=int(mode.get("grow_at", 3)))
        elif chosen != STANDARD:
            raise ValueError(f"{NODE_NAME} has no mode {chosen!r}. Choose {STANDARD} or {PROGRESSIVE}.")
        return io.NodeOutput(comfy.samplers.KSAMPLER(h3_image.sample, options))
