"""Read what a loaded model is: the class behind it, its precision, its device and its size."""

from __future__ import annotations

from comfy_api.latest import io

#: Attributes a loader's wrapper keeps its torch module on, tried in this order.
INNER_ATTRIBUTES = (
    "model",
    "first_stage_model",
    "cond_stage_model",
    "control_model",
    "t2i_model",
    "patcher",
)

#: How many wrappers deep the torch module is looked for.
INNER_HOPS = 5

#: Attributes naming the device a wrapper runs its weights on, tried in this order.
DEVICE_ATTRIBUTES = ("load_device", "device")

#: Attributes naming the precision a wrapper holds its weights in, tried in this order.
DTYPE_ATTRIBUTES = ("vae_dtype", "dtype")

#: Methods listing a torch module's weights, tried in this order.
WEIGHT_READERS = ("named_parameters", "named_buffers")


def attribute(value, name):
    """One attribute of a value, without raising.

    Args:
        value: Any object.
        name: Attribute to read.

    Returns:
        The attribute, or None where it is absent or reading it raised.
    """
    try:
        return getattr(value, name, None)
    except Exception:
        return None


def text_of(value) -> str:
    """A value written out.

    Args:
        value: Any object.

    Returns:
        The text, or an empty string for None and for a value that cannot be written.
    """
    if value is None:
        return ""
    try:
        return str(value)
    except Exception:
        return ""


def short_dtype(dtype) -> str:
    """A torch dtype without its module prefix.

    Args:
        dtype: A torch dtype, or None.

    Returns:
        The name, as ``float16``, or an empty string.
    """
    text = text_of(dtype)
    return text[len("torch."):] if text.startswith("torch.") else text


def inner_module(value):
    """The torch module a loader's wrapper holds.

    Args:
        value: Whatever the socket carried.

    Returns:
        The module, or None where none is reached within :data:`INNER_HOPS`.
    """
    candidate = value
    for _ in range(INNER_HOPS):
        if candidate is None or callable(attribute(candidate, "named_parameters")):
            return candidate
        step = None
        for name in INNER_ATTRIBUTES:
            step = attribute(candidate, name)
            if step is not None:
                break
        candidate = step
    return None


def first_weight(module):
    """The first weight a torch module holds.

    Args:
        module: A torch module, or None.

    Returns:
        A tensor, or None for a module holding none.
    """
    for name in WEIGHT_READERS:
        reader = attribute(module, name)
        if not callable(reader):
            continue
        try:
            for _, tensor in reader():
                return tensor
        except Exception:
            continue
    return None


def weight_total(module) -> int:
    """How many weights a torch module holds.

    Args:
        module: A torch module, or None.

    Returns:
        Elements across every parameter, falling back to the buffers where the parameters
        count nothing, and 0 where neither can be counted.
    """
    for name in WEIGHT_READERS:
        reader = attribute(module, name)
        if not callable(reader):
            continue
        total = 0
        try:
            for _, tensor in reader():
                total += int(tensor.numel())
        except Exception:
            total = 0
        if total:
            return total
    return 0


def mapping_weights(value) -> tuple[int, object]:
    """The weights held in a mapping of tensors.

    Args:
        value: A dictionary, such as the one a LORA_MODEL socket carries.

    Returns:
        The element count across every tensor, and the first tensor found or None.
    """
    total = 0
    first = None
    for tensor in list(value.values()):
        try:
            total += int(tensor.numel())
        except Exception:
            continue
        if first is None:
            first = tensor
    return total, first


def dtype_of(value, module) -> str:
    """The precision a loaded model holds its weights in.

    Args:
        value: Whatever the socket carried.
        module: The torch module inside it, or None.

    Returns:
        The dtype name, as ``float16``, or an empty string.
    """
    for name in DTYPE_ATTRIBUTES:
        found = attribute(value, name)
        if found is not None:
            return short_dtype(found)
    reader = attribute(value, "model_dtype")
    if callable(reader):
        try:
            found = reader()
        except Exception:
            found = None
        if found is not None:
            return short_dtype(found)
    return short_dtype(attribute(first_weight(module), "dtype"))


def device_of(value, module) -> str:
    """The device a loaded model runs on.

    Args:
        value: Whatever the socket carried.
        module: The torch module inside it, or None.

    Returns:
        The device name, as ``cuda:0``, or an empty string.
    """
    for name in DEVICE_ATTRIBUTES:
        found = attribute(value, name)
        if found is not None:
            return text_of(found)
    found = attribute(attribute(value, "patcher"), "load_device")
    if found is not None:
        return text_of(found)
    return text_of(attribute(first_weight(module), "device"))


def model_facts(value) -> dict:
    """What a loaded model is, read without changing it.

    Args:
        value: Whatever the loader socket carried.

    Returns:
        ``{"kind", "dtype", "device", "parameter_count"}``, each empty or zero where the
        value does not carry it.
    """
    if isinstance(value, dict):
        total, tensor = mapping_weights(value)
        return {
            "kind": type(value).__name__,
            "dtype": short_dtype(attribute(tensor, "dtype")),
            "device": text_of(attribute(tensor, "device")),
            "parameter_count": total,
        }
    module = inner_module(value)
    return {
        "kind": type(value if module is None else module).__name__,
        "dtype": dtype_of(value, module),
        "device": device_of(value, module),
        "parameter_count": weight_total(module),
    }


def summary_of(facts: dict) -> str:
    """A model written out as one line.

    Args:
        facts: What :func:`model_facts` answered.

    Returns:
        The class, the precision, the device and the size, as far as each is known.
    """
    parts = [facts["kind"] or "unknown"]
    dtype, device = facts["dtype"], facts["device"]
    if dtype and device:
        parts.append(f"{dtype} on {device}")
    elif dtype or device:
        parts.append(dtype or device)
    count = facts["parameter_count"]
    if count >= 1_000_000:
        parts.append(f"{count / 1_000_000:.2f}M parameters")
    elif count:
        parts.append(f"{count:,} parameters")
    return ", ".join(parts)


class ModelInfo(io.ComfyNode):
    """Name and measure the model on a loader wire."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        template = io.MatchType.Template(
            "model_info",
            [
                io.Model, io.Vae, io.Clip, io.ClipVision, io.ControlNet,
                io.UpscaleModel, io.LatentUpscaleModel, io.StyleModel, io.Gligen,
                io.Photomaker, io.LoraModel, io.AudioEncoder, io.ModelPatch,
                io.BackgroundRemoval,
            ],
        )
        return io.Schema(
            node_id="WASModelInfo",
            display_name="Model Info",
            search_aliases=[
                "WASModelInfo",
                "Model Info",
                "which model",
                "model size",
                "parameter count",
                "model dtype",
                "model device",
                "checkpoint info",
                "vae info",
            ],
            category="WAS Suite/Utilities",
            description=(
                "Say which model is on a wire and how big it is: its class, as `SDXL` or "
                "`AutoencoderKL`, the precision and device of its weights, and its parameter "
                "count. Takes MODEL, VAE, CLIP, CONTROL_NET and the other loader types on one "
                "socket, changes nothing, and answers empty text for a fact it cannot reach."
            ),
            inputs=[
                io.MatchType.Input(
                    "model",
                    template=template,
                    tooltip=(
                        "Anything a loader answers: MODEL, VAE, CLIP, CLIP_VISION, "
                        "CONTROL_NET, UPSCALE_MODEL, STYLE_MODEL and the rest. The wire is "
                        "read, not changed, and nothing is moved onto the graphics card to "
                        "read it."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="kind",
                    tooltip=(
                        "The class behind the wire, which names the family: `SDXL` or "
                        "`Flux` for a checkpoint, `AutoencoderKL` for a VAE, `RRDBNet` for "
                        "an upscaler. A LoRA arrives as a plain `dict`. Falls back to the "
                        "wrapper's own name where there is nothing inside it."
                    ),
                ),
                io.String.Output(
                    display_name="dtype",
                    tooltip=(
                        "Precision the weights are held at: `float16`, `bfloat16`, "
                        "`float32`, `float8_e4m3fn`. Compare it to catch a checkpoint that "
                        "loaded at full precision when half was wanted. Empty where the "
                        "model does not say."
                    ),
                ),
                io.String.Output(
                    display_name="device",
                    tooltip=(
                        "Where the weights run: `cuda:0`, `cpu`, `mps`. For a loader that "
                        "offloads, this is the device it loads onto when it runs, not where "
                        "it is parked between runs."
                    ),
                ),
                io.Int.Output(
                    display_name="parameter_count",
                    tooltip=(
                        "Weights the model holds, counted element by element: around 860 "
                        "million for an SD1.5 checkpoint and 2.6 billion for SDXL. 0 where "
                        "nothing could be counted."
                    ),
                ),
                io.Float.Output(
                    display_name="parameter_millions",
                    tooltip=(
                        "The same count divided by a million and rounded to three "
                        "decimals, so 2567463684 reads as 2567.464. Easier to test against "
                        "a threshold than the whole number."
                    ),
                ),
                io.String.Output(
                    display_name="summary",
                    tooltip=(
                        "Every figure on one line, as `SDXL, float16 on cuda:0, 2567.46M "
                        "parameters`, with a count under a million written out in full. "
                        "Wire it to Display Any, or into a filename prefix, to label a "
                        "render with what made it."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, model=None) -> io.NodeOutput:
        """Describe the connected model.

        Args:
            model: Whatever the loader socket carried.

        Returns:
            The class name, the precision, the device, the parameter count as a whole
            number and in millions, and the four written out as one line.
        """
        if model is None:
            return io.NodeOutput("", "", "", 0, 0.0, "nothing connected")
        facts = model_facts(model)
        count = facts["parameter_count"]
        return io.NodeOutput(
            facts["kind"],
            facts["dtype"],
            facts["device"],
            count,
            round(count / 1_000_000, 3),
            summary_of(facts),
        )
