"""Dust, hairs, scratches and uneven density, as on old or handled film."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.image import film

logger = log.get_logger("nodes.image.film")


class ImageFilmDamage(io.ComfyNode):
    """Lay dust, hairs, scratches and mottle over a batch."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageFilmDamage",
            display_name="Image Film Damage",
            search_aliases=[
                "WASImageFilmDamage", "Image Film Damage", "dust", "scratches", "hair", "old film",
                "vintage", "worn", "mottle",
            ],
            category="WAS Suite/Image/Film",
            description=(
                "Make a picture look printed from old or handled film: dust specks, stray hairs, "
                "running scratches and uneven density. Specks change every frame while scratches "
                "run through a whole batch, as on projected film."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="The pictures to mark. An alpha channel passes through.",
                ),
                io.Float.Input(
                    "dust", default=0.3, min=0.0, max=1.0, step=0.01,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="Dust specks. 0 = none; 0.3 = a lightly dusty print, about 120 specks; 1 = about 400 on the frame, at any resolution.",
                ),
                io.Float.Input(
                    "hairs", default=0.2, min=0.0, max=1.0, step=0.01,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="Stray hairs and fibres. 0 = none; 0.2 = two or three; 1 = about twelve on the frame.",
                ),
                io.Float.Input(
                    "scratches", default=0.0, min=0.0, max=1.0, step=0.01,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="Long running scratches, top to bottom. 0 = none; 0.2 = one faint line; 1 = five strong lines.",
                ),
                io.Float.Input(
                    "mottle", default=0.0, min=0.0, max=1.0, step=0.01,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="Uneven density, as from tired chemistry. 0 = even; 0.3 = faint blotches; 1 = strongly patchy.",
                ),
                io.Combo.Input(
                    "tone",
                    options=list(film.SPECK_TONES),
                    tooltip="`light` = white marks, as dust on a negative prints; `dark` = black marks, as on a slide or a print; `mixed` = some of each.",
                ),
                io.Int.Input(
                    "seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=io.ControlAfterGenerate.fixed,
                    tooltip="Which marks: `0` and `1` give two different sets, and one seed always gives the same. Each frame of a batch draws its own specks.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="images", tooltip="The marked pictures."),
            ],
        )

    @classmethod
    def execute(cls, images, dust=0.3, hairs=0.2, scratches=0.0, mottle=0.0, tone="light",
                seed=0) -> io.NodeOutput:
        result = film.film_damage(images, dust, scratches, hairs, mottle, tone, seed)
        logger.info("Image Film Damage marked %d frame(s)", int(images.shape[0]))
        return io.NodeOutput(result)
