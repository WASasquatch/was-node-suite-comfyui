"""Loading the Kandinsky 6 video super-resolution transformer."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.interface import run_result
from ...modules.io import picker
from ...modules.model.bluepixel import model as weights
from ...modules.model.kandinsky6 import sr

NODE_NAME = "Kandinsky 6 Upscale Model Loader"

#: Shown in the file list when the diffusion_models folders hold nothing.
NO_FILE = "put a Kandinsky 6 VSR transformer in models/diffusion_models"


def offered() -> tuple[list[str], str | None]:
    """The files in the diffusion_models folders, and the one most likely to be the upscaler."""
    import folder_paths

    files = list(folder_paths.get_filename_list(sr.FOLDER))
    likely = [name for name in files if "vsr" in name.lower() and "kandinsky" in name.lower()]
    preferred = [name for name in likely if "int8" in name.lower()] or likely
    return files, (preferred[0] if preferred else None)


class Kandinsky6UpscaleModelLoader(io.ComfyNode):
    """Load the Kandinsky 6 video super-resolution transformer as a MODEL."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        files, likely = offered()
        return io.Schema(
            node_id="WASKandinsky6UpscaleModelLoader",
            display_name=NODE_NAME,
            search_aliases=[
                "WASKandinsky6UpscaleModelLoader", NODE_NAME,
                "kandinsky vsr", "kandinsky 6 vsr", "video super resolution", "k6 upscale", "vsr 2.0",
            ],
            category="WAS Suite/Loaders",
            description=(
                "Load the Kandinsky 6 video super-resolution model, which redraws a video at two, 2.25 or "
                "four times its size with sharp, natural detail. Wire it into Kandinsky 6 Upscale Video. "
                "The panel shows the file's size, its parameter count and how its weights are stored."
            ),
            inputs=[
                picker.path_input(io.Combo.Input(
                    "unet_name",
                    options=files or [NO_FILE],
                    default=likely,
                    tooltip=(
                        "A Kandinsky 6 VSR transformer in models/diffusion_models: kandinsky6_vsr2_int8_convrot "
                        "(1.8 GB), _w6a8 (1.5 GB), _w4a8 (1.1 GB), or a release's own transformer file."
                    ),
                ), model=True),
                io.Combo.Input(
                    "weight_dtype", options=list(sr.WEIGHT_DTYPES), default="default", advanced=True,
                    tooltip=(
                        "default = as stored; fp8_e4m3fn or fp8_e5m2 = store a bf16 file at 8 bits, half "
                        "the memory; fp8_e4m3fn_fast = also multiply at 8 bits on cards that can."
                    ),
                ),
            ],
            outputs=[
                io.Model.Output(tooltip="The upscaler, for Kandinsky 6 Upscale Video."),
            ],
        )

    @classmethod
    def execute(cls, unet_name, weight_dtype="default") -> io.NodeOutput:
        """Load the transformer in ``unet_name``.

        Raises:
            ValueError: No file is chosen, or the file is not a Kandinsky 6 VSR transformer.
            PathNotAllowed: A wired path lies outside the diffusion_models folders.
        """
        import os

        import folder_paths

        unet_name = picker.model_name(unet_name, (sr.FOLDER,))
        if not unet_name or unet_name == NO_FILE:
            raise ValueError(
                f"{NODE_NAME} has no file to load. Put kandinsky6_vsr2_int8_convrot.safetensors in "
                "ComfyUI/models/diffusion_models, then refresh the node list."
            )
        path = folder_paths.get_full_path_or_raise(sr.FOLDER, unet_name)
        patcher = sr.load_model(path, weight_dtype)
        formats = weights.weight_formats(path)
        stored = ", ".join(f"{count} {name}" for name, count in sorted(formats.items(), key=lambda item: -item[1])) or "not read"
        steps = 2 if int(getattr(patcher.model, "n_grid", 1)) > 1 else 4
        run_result.publish(
            status=run_result.OK,
            summary=f"{os.path.basename(unet_name)} is ready to upscale",
            counts={
                "parameters (B)": round(weights.parameters(patcher) / 1e9, 2),
                "file (GB)": round(os.path.getsize(path) / 1e9, 2),
                "steps per tile": steps,
            },
            facts={"weights": stored, "storage": weight_dtype},
        )
        return io.NodeOutput(patcher)
