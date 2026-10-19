"""Quantizing a loaded diffusion model's weights into ComfyUI's own formats."""

from __future__ import annotations

import os

from comfy_api.latest import io

from ...modules import log
from ...modules.io import naming, rooted
from ...modules.model import quantize
from ...modules.util import sandbox

logger = log.get_logger("nodes.utility.quantize_model")

#: Bytes in the gigabyte the report counts in.
GIGABYTE = 1e9

#: How a saved file's counter is joined and padded, and its extension.
DELIMITER = "_"
PADDING = 5
EXTENSION = ".safetensors"


class QuantizeModel(io.ComfyNode):
    """Quantize a diffusion model's linear layers to int8, w6a8, w4a8, fp8, mxfp8 or nvfp4."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASQuantizeModel",
            display_name="Quantize Model",
            search_aliases=[
                "WASQuantizeModel",
                "Quantize Model",
                "quantize",
                "int8 convrot",
                "w6a8",
                "w4a8",
                "fp8",
                "nvfp4",
                "compress model",
            ],
            category="WAS Suite/Utilities",
            description=(
                "Quantize a diffusion model's weights to int8 convrot, w6a8, w4a8, fp8, mxfp8 or "
                "nvfp4, the formats ComfyUI runs natively, with any LoRAs applied before it merged "
                "in. Set like Comfy-Org, it makes the same file Comfy-Org publishes for that model. "
                "It writes the file itself when given a name and hands the model on to a sampler."
            ),
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="The diffusion model to quantize; MODEL, such as from Load Diffusion "
                    "Model. LoRAs applied before this node are merged in at full precision first.",
                ),
                io.Combo.Input(
                    "format",
                    options=list(quantize.FORMATS),
                    default="int8 convrot",
                    tooltip="'int8 convrot' = half of bf16, close to lossless, fast on RTX 20 "
                    "series and newer; 'w6a8' = about 40%; 'w4a8' = about 30%, the most loss; "
                    "'fp8 e4m3fn' = RTX 40 and newer; 'mxfp8', 'nvfp4' = fast on RTX 50 only.",
                ),
                io.Combo.Input(
                    "layers",
                    options=list(quantize.LAYERS),
                    default=quantize.LAYERS[0],
                    tooltip="'like Comfy-Org' = the layers Comfy-Org's quantizer picks, or the "
                    "layers and precision of Comfy-Org's own file for this model and format; "
                    "'transformer blocks' = every layer of the repeated blocks; 'every linear "
                    "layer' = embeddings and final layer too, smaller, less faithful.",
                ),
                io.Combo.Input(
                    "qkv",
                    options=list(quantize.QKV),
                    default=quantize.QKV[0],
                    tooltip="'as loaded' = the model's own form; 'fused' = query, key and value in "
                    "one layer, one matmul, as Comfy-Org ships its files; 'separate' = three "
                    "layers, as LoRAs for split projections expect. Flux 3 and MOSS-Image have both.",
                ),
                io.Combo.Input(
                    "attention",
                    options=list(quantize.ATTENTION),
                    default=quantize.ATTENTION[0],
                    tooltip="Kernel each attention layer is set to run. 'as loaded' = what the "
                    "model already carries; 'like Comfy-Org' = int8 attention where Comfy-Org's own "
                    "release sets it; 'int8 everywhere' = on every layer; 'clear' = none set.",
                ),
                io.Float.Input(
                    "protect",
                    default=0.0,
                    min=0.0,
                    max=0.5,
                    step=0.01,
                    tooltip="Share of the quantized layers with the largest error kept one step "
                    "finer. 0 = none; 0.1 = the worst tenth. w6a8, w4a8, mxfp8 and nvfp4 layers "
                    "go to int8 convrot; int8 and fp8 layers stay at the model's precision.",
                ),
                io.String.Input(
                    "filename_prefix",
                    default="diffusion_models/quantized",
                    tooltip="Where the quantized file is written in ComfyUI's output folder, such "
                    "as `diffusion_models/flux3_int8`, numbered like `flux3_int8_00001.safetensors`. "
                    "Empty = no file. Move it into models/diffusion_models to load it.",
                ),
                io.Int.Input(
                    "group_size",
                    default=0,
                    min=0,
                    max=256,
                    step=16,
                    advanced=True,
                    tooltip="Weights sharing one scale in w6a8 and w4a8. 0 = 32 for w6a8 and 16 "
                    "for w4a8; 64 = smaller, a little less faithful. A layer it does not divide "
                    "falls back to int8 convrot.",
                ),
                io.Combo.Input(
                    "scales",
                    options=list(quantize.SCALES),
                    default=quantize.SCALES[0],
                    advanced=True,
                    tooltip="How each int8 row's scale is set. 'row maximum' = its largest weight, "
                    "nothing clipped; 'lowest error' = the clip of 55% to 100% with the least "
                    "error, slower; 'like Comfy-Org' = what Comfy-Org's release of this model "
                    "used, else row maximum.",
                ),
            ],
            outputs=[
                io.Model.Output(
                    display_name="model",
                    tooltip="The quantized model; MODEL, ready for a sampler.",
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model, format="int8 convrot", layers=quantize.LAYERS[0], qkv=quantize.QKV[0],
                attention=quantize.ATTENTION[0], protect=0.0,
                filename_prefix="diffusion_models/quantized", group_size=0,
                scales=quantize.SCALES[0]) -> io.NodeOutput:
        """Quantize the model, write it when named, and build it again from the quantized weights.

        Raises:
            ValueError: A choice names nothing offered, the format is not in this ComfyUI, or
                ComfyUI does not recognise the quantized weights as a diffusion model.
            PathNotAllowed: filename_prefix settles outside every permitted write folder.
        """
        import time

        import comfy.sd
        import comfy.utils

        from ...modules.interface import run_result

        started = time.time()
        bar = comfy.utils.ProgressBar(1)

        def advance(done, total):
            bar.update_absolute(done, total)

        state, report = quantize.quantize(
            model, format, layers, min(0.5, max(0.0, float(protect))), int(group_size), qkv,
            attention, scales, advance,
        )
        saved = _write(state, filename_prefix, format, layers) if filename_prefix.strip() else ""
        built = comfy.sd.load_diffusion_model_state_dict(state)
        if built is None:
            raise ValueError(
                "Quantize Model could not build the quantized weights back into a model that "
                "ComfyUI recognises. Wire the model straight from Load Diffusion Model."
            )
        seconds = time.time() - started
        quantized = sum(report.formats.values())
        before, after = report.bytes_before / GIGABYTE, report.bytes_after / GIGABYTE
        smaller = 100.0 * (1.0 - report.bytes_after / max(1, report.bytes_before))
        summary = (
            f"{quantized} layers in {format}: {before:.2f} GB to {after:.2f} GB, "
            f"{smaller:.0f}% smaller"
        )
        logger.info("Quantize Model: %s in %.0f s", summary, seconds)
        counts = {"GB before": round(before, 2), "GB after": round(after, 2),
                  "% smaller": round(smaller, 1)}
        counts.update({name: count for name, count in report.formats.items()})
        counts.update({"kept": report.kept, "seconds": round(seconds)})
        run_result.publish(
            summary=summary,
            counts=counts,
            facts={"layers": report.recipe, "scales": report.scales, "qkv": report.qkv,
                   "attention": report.attention, "protected": report.protected,
                   "refitted": report.refitted,
                   "saved": os.path.basename(saved) if saved else "not saved"},
            bodies=_bodies(run_result, report),
        )
        return io.NodeOutput(built)


def _bodies(run_result, report) -> list:
    """The projected size in every format and the layers with the largest error, as texts."""
    before = max(1, report.bytes_before)
    lines = [f"{'source':14s}{before / GIGABYTE:8.2f} GB"]
    for name, size in report.projected.items():
        lines.append(
            f"{name:14s}{size / GIGABYTE:8.2f} GB  {100.0 * (1.0 - size / before):5.1f}% smaller"
        )
    bodies = [run_result.body("projected size", "\n".join(lines))]
    worst = "\n".join(
        f"{error * 100:6.2f}%  {name:13s} {path}" for path, name, error in report.worst
    )
    if worst:
        bodies.append(run_result.body("largest errors", worst))
    return bodies


def _write(state: dict, prefix: str, format_name: str, layers: str) -> str:
    """Write the quantized weights to the output folder under the next free numbered name.

    Args:
        state: The quantized state dict.
        prefix: ``folder/leaf`` inside the output folder.
        format_name: The format chosen, recorded in the file's metadata.
        layers: The layer choice, recorded beside it.

    Returns:
        The path written.

    Raises:
        PathNotAllowed: The prefix settles outside every permitted write folder.
    """
    import safetensors.torch

    below, _, leaf = prefix.replace("\\", "/").strip().rpartition("/")
    parent = rooted.destination(rooted.OUTPUT, below)
    os.makedirs(parent, exist_ok=True)
    file_name = naming.next_name(str(parent), leaf.strip() or "quantized", DELIMITER, PADDING, EXTENSION)
    target = str(sandbox.resolve_write_file(parent, file_name))
    tensors = {key: value.contiguous() for key, value in state.items()}
    metadata = {"format": "pt", "quantized_format": format_name, "quantized_layers": layers}
    safetensors.torch.save_file(tensors, target, metadata=metadata)
    logger.info("Quantize Model wrote %s", target)
    return target
