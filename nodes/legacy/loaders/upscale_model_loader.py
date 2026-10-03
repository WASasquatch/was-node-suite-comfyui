"""Upscale model loading, plus the model's file name."""

from __future__ import annotations

from comfy_api.latest import io

REQUIRES = "loaders"


def upscale_model_names() -> list[str]:
    """The upscale models this install offers."""
    import folder_paths

    return folder_paths.get_filename_list("upscale_models")


class UpscaleModelLoader(io.ComfyNode):
    """Load an ESRGAN-family upscale model and report the file name it came from."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Upscale Model Loader",
            display_name="Upscale Model Loader (Advanced)",
            search_aliases=["Upscale Model Loader", "upscale", "esrgan", "load upscaler"],
            category="WAS Suite/Loaders",
            description=(
                "Deprecated: use ComfyUI's Load Upscale Model instead. Loads an upscale "
                "model and returns it, plus the file name it came from as a string. That "
                "name is already on the model_name widget."
            ),
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=upscale_model_names(),
                    tooltip=(
                        "The upscale model in models/upscale_models to load. Its own scale "
                        "factor, usually 2x or 4x, decides how much larger the result is."
                    ),
                ),
            ],
            outputs=[
                io.UpscaleModel.Output(
                    display_name="UPSCALE_MODEL",
                    tooltip="The loaded model, for Upscale Image (using Model).",
                ),
                io.String.Output(
                    display_name="MODEL_NAME_TEXT",
                    tooltip=(
                        "The file name as it appears in the widget, extension included, "
                        "for captions, file names and log lines."
                    ),
                ),
            ],
            is_deprecated=True,
        )

    @classmethod
    def execute(cls, model_name) -> io.NodeOutput:
        import comfy.utils
        import folder_paths
        from spandrel import ModelLoader

        model_path = folder_paths.get_full_path_or_raise("upscale_models", model_name)
        state_dict = comfy.utils.load_torch_file(model_path, safe_load=True)
        return io.NodeOutput(ModelLoader().load_from_state_dict(state_dict).eval(), model_name)
