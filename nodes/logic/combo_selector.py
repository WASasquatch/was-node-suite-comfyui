"""Pick a model file, a sampler or a scheduler by name."""

from __future__ import annotations

from comfy_api.latest import io

#: Folder each source lists, against the name shown on the widget.
FOLDERS: tuple[tuple[str, str], ...] = (
    ("checkpoints", "checkpoint"),
    ("loras", "LoRA"),
    ("vae", "VAE"),
    ("clip", "CLIP"),
    ("text_encoders", "text encoder"),
    ("diffusion_models", "diffusion model"),
    ("controlnet", "ControlNet"),
    ("style_models", "style model"),
    ("hypernetworks", "hypernetwork"),
    ("upscale_models", "upscale model"),
    ("embeddings", "embedding"),
    ("gligen", "GLIGEN"),
)

#: What a source with nothing in it offers.
EMPTY = "None"


def folder_options(folder: str) -> list[str]:
    """The files a model folder holds.

    Args:
        folder: A key of ComfyUI's folder table, such as ``"loras"``.

    Returns:
        The file names, or ``["None"]`` where the folder is empty or unknown.
    """
    try:
        import folder_paths

        found = list(folder_paths.get_filename_list(folder))
    except Exception:
        return [EMPTY]
    return found or [EMPTY]


def sampler_options() -> tuple[list[str], list[str]]:
    """The sampler and scheduler names this ComfyUI offers.

    Returns:
        The samplers and the schedulers, each ``["None"]`` where they cannot be read.
    """
    try:
        from comfy.samplers import KSampler

        return list(KSampler.SAMPLERS) or [EMPTY], list(KSampler.SCHEDULERS) or [EMPTY]
    except Exception:
        return [EMPTY], [EMPTY]


def build_options() -> list[io.DynamicCombo.Option]:
    """One option per source, each carrying its own list.

    Returns:
        The options in the order they are listed.
    """
    options = []
    for folder, shown in FOLDERS:
        options.append(
            io.DynamicCombo.Option(
                folder,
                [
                    io.Combo.Input(
                        "value",
                        display_name=shown,
                        options=folder_options(folder),
                        tooltip=(
                            f"Which {shown} to name. A sub-folder is part of the name: "
                            f"Flux/detail_v2.safetensors."
                        ),
                    )
                ],
            )
        )
    samplers, schedulers = sampler_options()
    options.append(
        io.DynamicCombo.Option(
            "samplers",
            [
                io.Combo.Input(
                    "value",
                    display_name="sampler",
                    options=samplers,
                    tooltip="Which sampler to name: euler, dpmpp_2m, ddim.",
                )
            ],
        )
    )
    options.append(
        io.DynamicCombo.Option(
            "schedulers",
            [
                io.Combo.Input(
                    "value",
                    display_name="scheduler",
                    options=schedulers,
                    tooltip="Which scheduler to name: normal, karras, beta.",
                )
            ],
        )
    )
    return options


class ComboSelector(io.ComfyNode):
    """Name a model file, a sampler or a scheduler and send it to another node's dropdown."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASComboSelector",
            display_name="Combo Selector",
            search_aliases=[
                "WASComboSelector",
                "Combo Selector",
                "model picker",
                "pick checkpoint",
                "pick lora",
                "pick sampler",
            ],
            category="WAS Suite/Logic",
            description=(
                "Pick an installed checkpoint, LoRA, VAE, ControlNet, upscale model, sampler or "
                "scheduler and send it to another node's dropdown, and as text. Convert the "
                "target node's dropdown to an input and connect combo to it."
            ),
            inputs=[
                io.DynamicCombo.Input(
                    "source",
                    options=build_options(),
                    tooltip=(
                        "What to pick from: checkpoints, loras, vae, clip, text_encoders, "
                        "diffusion_models, controlnet, style_models, hypernetworks, "
                        "upscale_models, embeddings, gligen, samplers, schedulers."
                    ),
                ),
            ],
            outputs=[
                io.AnyType.Output(
                    display_name="combo",
                    tooltip=(
                        "The choice, on a wire a dropdown accepts. Connect it to a converted "
                        "dropdown such as ckpt_name, lora_name or sampler_name."
                    ),
                ),
                io.String.Output(
                    display_name="name",
                    tooltip="The choice as text: sd_xl_base_1.0.safetensors, euler, karras.",
                ),
                io.String.Output(
                    display_name="source",
                    tooltip="Which list it came from: checkpoints, loras, samplers.",
                ),
            ],
        )

    @classmethod
    def execute(cls, source: dict) -> io.NodeOutput:
        """Answer the chosen name.

        Args:
            source: The chosen list and the name picked from it.

        Returns:
            The name on a dropdown wire, the name as text, and the list it came from.

        Raises:
            ValueError: The chosen list is empty.
        """
        kind = source.get("source") or ""
        chosen = source.get("value") or ""
        if not chosen or chosen == EMPTY:
            raise ValueError(
                f"Combo Selector has nothing to name: {kind or 'that list'} is empty. Put a "
                f"file in the folder and press Refresh Node Definitions, or pick another list"
            )
        return io.NodeOutput(chosen, chosen, kind)
