"""Loading a language model as a CLIP that writes text."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.interface import run_result
from ...modules.model import language_model

NODE_NAME = "Language Model Loader"

#: Shown in the file list when the text_encoders folders hold nothing.
NO_FILE = "put a language model in models/text_encoders"


class WASLanguageModelLoader(io.ComfyNode):
    """Load a language model from the text_encoders folders as a CLIP that writes text."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        files, likely = language_model.offered()
        return io.Schema(
            node_id="WASLanguageModelLoader",
            display_name=NODE_NAME,
            search_aliases=[
                "WASLanguageModelLoader", NODE_NAME,
                "load language model",
                "load llm",
                "qwen3 32b",
                "qwen3 14b",
                "text generation model",
            ],
            category="WAS Suite/Loaders",
            description=(
                "Load a language model from models/text_encoders as a CLIP that writes text, "
                "for Generate Text, Fast Generate Text and LLM From CLIP. Reads dense Qwen3 "
                "models up to 32B as well as every language model ComfyUI reads. The panel "
                "shows the model, its size and how its weights are stored."
            ),
            inputs=[
                io.Combo.Input(
                    "model_name",
                    options=files or [NO_FILE],
                    default=likely,
                    tooltip=(
                        "A language model in models/text_encoders, such as "
                        "qwen3_32b_w4a8.safetensors or qwen_3_4b.safetensors. Qwen3 14B and 32B "
                        "load in their Hugging Face layout, plain or quantized."
                    ),
                ),
            ],
            outputs=[
                io.Clip.Output(
                    tooltip=(
                        "The model, for Generate Text, Fast Generate Text or LLM From CLIP. A "
                        "Qwen3 answers in its chat format, thinking only when asked."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, model_name) -> io.NodeOutput:
        """Build the model in ``model_name``.

        Raises:
            ValueError: No file is chosen, or the file holds no model that writes text.
        """
        if not model_name or model_name == NO_FILE:
            raise ValueError(
                f"{NODE_NAME} has no file to load. Put a language model, such as "
                "qwen3_32b_w4a8.safetensors from huggingface.co/WAS/LLM, in "
                "ComfyUI/models/text_encoders, then refresh the node list."
            )
        loaded = language_model.load(model_name)
        facts = {
            "model": loaded.model,
            "built by": loaded.builder,
            "weights": loaded.weights,
            "output layer": "its own" if loaded.head else "shared with the embedding",
        }
        counts = {
            "parameters (B)": round(loaded.parameters / 1e9, 2),
            "file (GB)": round(loaded.file_bytes / 1e9, 2),
        }
        if loaded.config:
            facts["shape"] = (
                f"{loaded.config['num_hidden_layers']} layers, {loaded.config['hidden_size']} wide, "
                f"{loaded.config['num_attention_heads']} heads over "
                f"{loaded.config['num_key_value_heads']} key/value heads"
            )
            counts["context tokens"] = loaded.config["max_position_embeddings"]
        run_result.publish(
            status=run_result.OK,
            summary=f"{loaded.model} is ready to write text",
            counts=counts,
            facts=facts,
        )
        return io.NodeOutput(loaded.clip)
