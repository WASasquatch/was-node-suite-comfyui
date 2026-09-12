"""Answering a question about an image, from one node."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from comfy_api.latest import io

from ....modules.model import marigold_v2


#: Config group this module loads under.
REQUIRES = "preprocessors"


@dataclass(frozen=True)
class Control:
    """How one preprocessor presents and bounds a shared setting.

    Attributes:
        label: Name drawn on the widget while this preprocessor is chosen.
        low: Smallest value this preprocessor accepts.
        high: Largest value this preprocessor accepts.
        start: Value this preprocessor uses where the one it was handed is out of range.
        step: Amount one click moves the widget by.
        places: Decimal places drawn, 0 for a whole number.
    """

    label: str
    low: float
    high: float
    start: float
    step: float = 1.0
    places: int = 0


@dataclass(frozen=True)
class Loaded:
    """A built model and the menu name it was built from.

    Attributes:
        backend: What the answer runs through.
        name: Menu name of the model.
    """

    backend: object
    name: str


#: Shared setting names, so a preprocessor added later reuses one rather than adding a widget.
LOW = "threshold_low"
HIGH = "threshold_high"
RADIUS = "radius"
STRENGTH = "strength"
SEED = "seed"
STEPS = "steps"
TILE = "tile"
MODEL = "model_name"

#: The inputs a wired model reads its parts from.
TRANSFORMER = "model"
DECODER = "vae"
CONDITIONING = "conditioning_name"
ADAPTER = "adapter_name"
PROMPT = "conditioning"

#: Intrinsic answer -> the checkpoints that read it, the first being what it starts on.
#: Every one of them takes a step count and a seed.
INTRINSIC: dict[str, tuple[str, ...]] = {
    "albedo": ("Marigold IID Appearance", "Marigold IID Lighting", marigold_v2.MODEL_NAME),
    "roughness": ("Marigold IID Appearance",),
    "metallicity": ("Marigold IID Appearance",),
    "material": ("Marigold IID Appearance",),
    "shading": ("Marigold IID Lighting",),
    "residual": ("Marigold IID Lighting",),
}

#: Preprocessor -> the shared settings it reads, each under the name and bounds it reads them
#: by. A preprocessor with no entry takes no setting beyond the model and the resolution.
CONTROLS: dict[str, dict[str, Control]] = {
    "canny_pyramid": {
        # At zero every gradient is an edge, the trace floods and the frame comes back
        # black, so the range starts above it.
        LOW: Control("low_threshold", 1, 255, start=100),
        HIGH: Control("high_threshold", 0, 255, start=200),
    },
    "lineart_simple": {
        # Below half a pixel the blur kernel is narrower than three samples and becomes
        # the identity, which leaves nothing to trace.
        RADIUS: Control("blur_radius", 0.5, 32.0, start=6.0, step=0.1, places=1),
        # Above about half its old range no sample survived the filter and the result
        # wrapped back to what a floor of zero gives, so the range stops short of that.
        LOW: Control("noise_floor", 0, 64, start=8),
    },
    "scribble_xdog": {LOW: Control("stroke_threshold", 1, 64, start=32)},
    "binary": {LOW: Control("split_level", 0, 255, start=100)},
    "shuffle": {SEED: Control("seed", 0, 0xFFFFFFFFFFFFFFFF, start=0)},
    "depth_map": {},
    "normal_map": {
        STRENGTH: Control("relief", 0.5, 64.0, start=16.0, step=0.5, places=1),
        RADIUS: Control("smoothing", 0.0, 8.0, start=3.0),
    },
    "openpose": {
        LOW: Control("detection_threshold", 0.05, 0.95, start=0.30, step=0.05, places=2),
    },
    "animal_pose": {
        LOW: Control("detection_threshold", 0.05, 0.95, start=0.30, step=0.05, places=2),
    },
    "ade20k_segments": {},
    "soft_edge": {},
    "lineart_model": {},
    "line_segments": {
        # The head's own scores peak well below one, so the range stops where a real
        # picture still has segments left.
        LOW: Control("score_threshold", 0.01, 0.40, start=0.10, step=0.01, places=2),
        HIGH: Control("shortest_segment", 1.0, 60.0, start=20.0, step=1.0, places=0),
    },
    "anyline": {LOW: Control("speck_size", 1, 256, start=36)},
    "denoise": {TILE: Control("tile", 0, 4096, start=0, step=64)},
    "low_light": {TILE: Control("tile", 0, 4096, start=0, step=64)},
    **{
        name: {
            STEPS: Control("steps", 1, 20, start=4),
            SEED: Control("seed", 0, 0xFFFFFFFFFFFFFFFF, start=0),
        }
        for name in INTRINSIC
    },
}

#: Preprocessor -> the models it can run, the first being what it starts on. An empty tuple
#: means it needs none, and the model widget is not drawn for it.
MODELS: dict[str, tuple[str, ...]] = {
    "canny_pyramid": (),
    "lineart_simple": (),
    "scribble_xdog": (),
    "binary": (),
    "shuffle": (),
    "depth_map": (
        "Depth Anything V2 Small", "Depth Anything V2 Base", "Depth Anything V2 Large",
        "DPT SwinV2 Tiny", "DPT Large", marigold_v2.MODEL_NAME,
    ),
    "normal_map": (
        "Depth Anything V2 Small", "Depth Anything V2 Base", "Depth Anything V2 Large",
        "DPT SwinV2 Tiny", "DPT Large", marigold_v2.MODEL_NAME,
    ),
    "openpose": ("ViTPose Base", "ViTPose Small", "ViTPose Wholebody"),
    "animal_pose": ("ViTPose Animal",),
    "ade20k_segments": ("SegFormer B0 ADE20K", "SegFormer B2 ADE20K", "SegFormer B4 ADE20K"),
    "soft_edge": ("HED Soft Edge", "PiDiNet Soft Edge", "TEED Soft Edge"),
    "lineart_model": ("Lineart", "Lineart Coarse", "Lineart Anime", "Manga Line"),
    "line_segments": ("MLSD Line Segments",),
    "anyline": ("AnyLine",),
    "denoise": ("SCUNet", "NAFNet SIDD width32", "NAFNet SIDD width64"),
    "low_light": (
        "DarkIR",
        "Retinexformer NTIRE",
        "Retinexformer LOL v1",
        "Retinexformer LOL v2 Real",
        "Retinexformer LOL v2 Synthetic",
        "Retinexformer FiveK",
        "Retinexformer Extreme Dark",
        "Retinexformer Dark Motion",
        "Retinexformer Indoor Night",
        "Retinexformer Outdoor Night",
        "HVI-CIDNet Generalization",
        "HVI-CIDNet FiveK",
        "HVI-CIDNet SICE",
        "HVI-CIDNet Extreme Dark",
    ),
    **INTRINSIC,
}

#: Widget option -> the scenario :func:`modules.model.cidnet.load` takes.
CIDNET_SCENARIOS = {
    "HVI-CIDNet Generalization": "generalization",
    "HVI-CIDNet FiveK": "fivek",
    "HVI-CIDNet SICE": "sice",
    "HVI-CIDNet Extreme Dark": "sony-total-dark",
}

#: Widget option -> the module a restoration network came from, which sets the side multiple
#: its answer is padded to.
RESTORE_FAMILY = {
    "SCUNet": "scunet",
    "NAFNet SIDD width32": "nafnet",
    "NAFNet SIDD width64": "nafnet",
    "DarkIR": "darkir",
    **{name: "cidnet" for name in CIDNET_SCENARIOS},
}

#: Every model any preprocessor names, in menu order, each listed once.
EVERY_MODEL = list(dict.fromkeys(name for names in MODELS.values() for name in names))

#: Fewest samples a side is worked out at, so a long thin frame keeps something to read.
NARROWEST = 8

#: Longest edge a result is computed at before it is scaled back to the source size. Working
#: at a fixed edge is what keeps a setting meaning the same thing on every image size.
DEFAULT_EDGE = 512


def bounded(name: str, value, kind: str):
    """Hold one setting inside the bounds the chosen preprocessor reads it by.

    Args:
        name: A shared setting name, such as :data:`LOW`.
        value: The value that arrived.
        kind: The chosen preprocessor.

    Returns:
        The value where the preprocessor accepts it, its own starting value where it does
        not, and the value untouched for a setting the preprocessor does not read.
    """
    control = CONTROLS.get(kind, {}).get(name)
    if control is None:
        return value
    # A shared widget carries one schema default, which lands at the wrong end of another
    # preprocessor's range, so a value it cannot accept becomes its own start rather than
    # the nearest bound.
    if control.low <= value <= control.high:
        return value
    return control.start


def build(model: str):
    """Build one model by its menu name.

    Args:
        model: A name from :data:`EVERY_MODEL`.

    Returns:
        What the answer runs through, already cached for the process.

    Raises:
        ValueError: ``model`` names nothing this node runs.
        ModelUnavailable: The weights are absent and ``features.network`` is off.
    """
    from ....modules.model import (
        cidnet,
        darkir,
        depth,
        hed,
        lineart,
        lineart_anime,
        manga_line,
        marigold,
        mlsd,
        pidi,
        pose,
        nafnet,
        retinexformer,
        scunet,
        segmentation,
        teed,
    )

    builders = {
        "SCUNet": scunet.load,
        "DarkIR": darkir.load,
        "HED Soft Edge": hed.load,
        "PiDiNet Soft Edge": pidi.load,
        "TEED Soft Edge": teed.load,
        "Lineart": lineart.load,
        "Lineart Coarse": lambda: lineart.load(coarse=True),
        "Lineart Anime": lineart_anime.load,
        "Manga Line": manga_line.load,
        "MLSD Line Segments": mlsd.load,
        "AnyLine": teed.load_misto,
    }
    if model in builders:
        return builders[model]()
    if model in depth.MODELS:
        return depth.load(model)
    if model in pose.MODELS:
        return pose.load(model)
    if model in segmentation.MODELS:
        return segmentation.load(model)
    if model in nafnet.CHECKPOINTS:
        return nafnet.load(model)
    if model in retinexformer.MODELS:
        return retinexformer.load(model)
    if model in CIDNET_SCENARIOS:
        return cidnet.load(CIDNET_SCENARIOS[model])
    if model in marigold.MODELS:
        return marigold.load(model)
    raise ValueError(
        f"Power Preprocessor has no model called {model!r}. "
        f"Choose one of: {', '.join(EVERY_MODEL)}."
    )


class PowerPreprocessor(io.ComfyNode):
    """Answer a question about an image, choosing which question on the node."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASPowerPreprocessor",
            display_name="Power Preprocessor",
            search_aliases=[
                "WASPowerPreprocessor",
                "Power Preprocessor",
                "controlnet preprocessor",
                "annotator",
                "hint image",
                "depth map",
                "normal map",
                "openpose",
                "animal pose",
                "hands",
                "denoise",
                "low light",
                "segmentation",
                "lineart",
                "scribble",
                "soft edge",
                "albedo",
                "roughness",
                "metallicity",
                "shading",
                "intrinsic maps",
                "relight",
            ],
            category="WAS Suite/Image/Preprocess",
            description=(
                "Measure an image and answer what it found: depth, surface direction, body "
                "pose, what every pixel is, edges, drawn lines, straight runs, the paint and "
                "the light it was lit by, or the frame with its noise or its darkness taken "
                "out. Feeding a ControlNet is the usual reason, and the same answers drive "
                "relighting, defocus, parallax, masking and stylising. Pick the question and "
                "the node draws only what that question reads. Five need no model at all and "
                "most fetch a checkpoint on first use. `Marigold v2` is the exception: it "
                "reads a transformer, a decoder and a prompt embedding placed by hand."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="The images to measure. A whole batch is processed.",
                ),
                io.Combo.Input(
                    "preprocessor",
                    options=list(CONTROLS),
                    default="canny_pyramid",
                    tooltip=(
                        "What to work out. `canny_pyramid`, `lineart_simple`, "
                        "`scribble_xdog`, `binary` and `shuffle` need no model; the rest "
                        "run the model chosen below."
                    ),
                ),
                io.Combo.Input(
                    MODEL,
                    options=EVERY_MODEL,
                    default="Depth Anything V2 Small",
                    tooltip=(
                        "Which model answers the question, listing only the ones that can. "
                        "Within a family the smaller is quicker and the larger more "
                        "accurate: `Depth Anything V2 Small` is 99 MB against `Large` at "
                        "1.3 GB. `Marigold v2` is the sharpest, and the one read from the "
                        "three inputs at the bottom rather than downloaded."
                    ),
                ),
                io.Int.Input(
                    "resolution",
                    default=DEFAULT_EDGE,
                    min=64,
                    max=8192,
                    step=64,
                    tooltip=(
                        "Longest edge the work is done at before the answer is scaled back "
                        "to the image's own size. 512 is a sensible start; 1024 resolves "
                        "finer detail and costs more. Anything above the image's own "
                        "longest edge is held to it. `openpose`, `animal_pose`, "
                        "`line_segments`, `denoise` and `low_light` ignore it."
                    ),
                ),
                io.Float.Input(
                    LOW,
                    default=100.0,
                    min=0.0,
                    max=256.0,
                    step=1.0,
                    tooltip=(
                        "The lower cut-off, or the only one where a question takes one. Each "
                        "preprocessor reads it over a range of its own, which the widget "
                        "shows: `canny_pyramid` 1 to 255, `lineart_simple` 0 to 64, "
                        "`openpose` 0.05 to 0.95, `line_segments` 0.01 to 0.40, `anyline` "
                        "1 to 256. Switching preprocessor moves it to that one's start."
                    ),
                ),
                io.Float.Input(
                    HIGH,
                    default=200.0,
                    min=0.0,
                    max=255.0,
                    step=1.0,
                    tooltip=(
                        "The upper cut-off, for a question that takes a pair. "
                        "`canny_pyramid` reads 0 to 255 as the strength an edge must reach "
                        "to start at all; `line_segments` reads 1 to 60 as the shortest run "
                        "it keeps."
                    ),
                ),
                io.Float.Input(
                    RADIUS,
                    default=6.0,
                    min=0.0,
                    max=32.0,
                    step=0.1,
                    tooltip=(
                        "A distance in pixels. `lineart_simple` reads 0.5 to 32.0 as the "
                        "blur each pixel is compared against: 6.0 gives normal line weight, "
                        "2.0 fine lines and 16.0 heavy ones. `normal_map` reads 0 to 8 as "
                        "how far the surface slope is measured across: 3 suits a depth "
                        "model, 0 is the sharpest and 6 flattens fine grain."
                    ),
                ),
                io.Float.Input(
                    STRENGTH,
                    default=16.0,
                    min=0.5,
                    max=64.0,
                    step=0.5,
                    tooltip=(
                        "How hard the answer is shaped. `normal_map` reads 0.5 to 64.0 as "
                        "relief: 16.0 shows the folds in a coat, 2.0 is nearly flat and "
                        "48.0 exaggerates every slope."
                    ),
                ),
                io.Int.Input(
                    SEED,
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=io.ControlAfterGenerate.fixed,
                    tooltip=(
                        "Chooses between equally good random answers. `shuffle` reads it as "
                        "the displacement: `0` and `1` scramble the same picture two "
                        "different ways, and one seed always gives one scramble. `albedo`, "
                        "`roughness`, `metallicity`, `material`, `shading` and `residual` "
                        "read it as the noise their first step starts from, and every frame "
                        "of a batch starts from the same one."
                    ),
                ),
                io.Int.Input(
                    TILE,
                    default=0,
                    min=0,
                    max=4096,
                    step=64,
                    tooltip=(
                        "Work a square at a time instead of the whole frame, which holds "
                        "VRAM down on a large picture. 0 reads the whole frame. 512 reads "
                        "a 512 pixel square at a time, overlapping a quarter and faded "
                        "together, so no join shows. A larger square is closer to the "
                        "whole frame. Read only by `denoise` and `low_light`."
                    ),
                ),
                io.Int.Input(
                    STEPS,
                    default=4,
                    min=1,
                    max=20,
                    tooltip=(
                        "How many passes a question that denoises takes. `albedo`, "
                        "`roughness`, `metallicity`, `material`, `shading` and `residual` "
                        "read 1 to 20: 4 is what Marigold was tuned for, 1 is roughly twice "
                        "as quick and coarser, and above 8 the answer stops changing much."
                    ),
                ),
                io.Model.Input(
                    TRANSFORMER,
                    optional=True,
                    tooltip=(
                        "The transformer `Marigold v2` runs on: Load Diffusion Model on "
                        "`qwen_image_edit_2509_int8_convrot`. One transformer answers "
                        "every map; `adapter_name` puts the adapter on it here."
                    ),
                ),
                io.Vae.Input(
                    DECODER,
                    optional=True,
                    tooltip=(
                        "The decoder this map was trained with: Load VAE on "
                        "`marigold_v2_depth_log_stage2_vae` for depth, "
                        "`marigold_v2_normals_vae` for normals, `marigold_v2_albedo_vae` "
                        "for albedo. It names the same map as the adapter."
                    ),
                ),
                io.Combo.Input(
                    ADAPTER,
                    options=marigold_v2.adapters(),
                    default=marigold_v2.AUTOMATIC,
                    optional=True,
                    tooltip=(
                        "`auto` = this map's adapter, found by name in the loras folder; "
                        "`already on the model` = apply none, for a LoRA put on the "
                        "transformer before it arrives; `marigold_v2_normals.safetensors` "
                        "= that file."
                    ),
                ),
                io.Combo.Input(
                    CONDITIONING,
                    options=marigold_v2.embeddings(),
                    default=marigold_v2.AUTOMATIC,
                    optional=True,
                    tooltip=(
                        "`auto` = this map's prompt embedding, found by name in the "
                        "embeddings folder; `marigold_v2_depth_conditioning.safetensors` = "
                        "that file. The `conditioning` socket beats this when wired."
                    ),
                ),
                io.Conditioning.Input(
                    PROMPT,
                    optional=True,
                    tooltip=(
                        "A prompt embedding to read the map against, from a text encoder "
                        "or a loaded conditioning. Used in place of `conditioning_name`. "
                        "Keep it full length: a short prompt flattens the map, while the "
                        "wording barely moves it."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="image",
                    tooltip=(
                        "The answer, the same size and batch length as the input. An Apply "
                        "ControlNet image input is the usual destination, and it is an "
                        "ordinary image that any node taking one will read."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls, image, preprocessor, model_name, resolution, threshold_low, threshold_high,
        radius, strength, seed, steps, tile, model=None, vae=None,
        adapter_name=marigold_v2.AUTOMATIC,
        conditioning_name=marigold_v2.AUTOMATIC, conditioning=None,
    ) -> io.NodeOutput:
        """Work out the chosen answer.

        Raises:
            ValueError: ``preprocessor`` names no question this node answers, or ``model``
                cannot answer the one chosen.
        """
        if preprocessor not in CONTROLS:
            raise ValueError(
                f"Power Preprocessor cannot work out {preprocessor!r}. "
                f"Choose one of: {', '.join(CONTROLS)}."
            )
        allowed = MODELS[preprocessor]
        settings = {
            LOW: bounded(LOW, float(threshold_low), preprocessor),
            HIGH: bounded(HIGH, float(threshold_high), preprocessor),
            RADIUS: bounded(RADIUS, float(radius), preprocessor),
            STRENGTH: bounded(STRENGTH, float(strength), preprocessor),
            SEED: int(bounded(SEED, int(seed), preprocessor)),
            STEPS: int(bounded(STEPS, int(steps), preprocessor)),
            TILE: int(bounded(TILE, int(tile), preprocessor)),
        }

        if model_name == marigold_v2.MODEL_NAME and model_name in allowed:
            return io.NodeOutput(
                _wired(preprocessor, model, vae, adapter_name, conditioning_name,
                       conditioning, image, int(resolution))
            )
        if not allowed:
            result = _without_model(preprocessor, image, int(resolution), settings)
        else:
            if model_name not in allowed:
                raise ValueError(
                    f"{preprocessor!r} cannot be worked out by {model_name!r}. "
                    f"Choose one of: {', '.join(allowed)}."
                )
            loaded = Loaded(backend=build(model_name), name=model_name)
            result = _with_model(preprocessor, loaded, image, int(resolution), settings)

        answer = (result.clamp(0.0, 255.0) / 255.0).permute(0, 2, 3, 1)
        return io.NodeOutput(answer.to(image.device, dtype=image.dtype).contiguous())


def _prompt_of(conditioning):
    """The embedding a wired ``CONDITIONING`` carries.

    Args:
        conditioning: A ``CONDITIONING`` value, or None.

    Returns:
        The first entry's tensor, or None where nothing was wired.

    Raises:
        ValueError: Something was wired that carries no embedding.
    """
    if conditioning is None:
        return None
    try:
        held = conditioning[0][0]
    except (TypeError, IndexError, KeyError):
        held = None
    if held is None or not hasattr(held, "ndim"):
        raise ValueError(
            f"the {PROMPT} input carries no prompt embedding. Wire a text encoder or a "
            f"loaded conditioning, or leave it empty to read the map's own file."
        )
    return held if held.ndim == 3 else held.unsqueeze(0)


def _wired(name, model, vae, adapter_name, conditioning_name, conditioning, image,
           resolution: int):
    """Work out one map from a transformer and a decoder wired into the node.

    Args:
        name: A key of :data:`marigold_v2.MODALITIES`.
        model: The ``MODEL`` wired in, or None.
        vae: The ``VAE`` wired in, or None.
        adapter_name: The adapter's file name, or the automatic entry.
        conditioning_name: The prompt embedding's file name, or the automatic entry.
        conditioning: A ``CONDITIONING`` wired in, which wins over the file above.
        image: ``(batch, height, width, channels)`` in ``[0, 1]``.
        resolution: Longest edge the map is read at.

    Returns:
        A ``(batch, height, width, 3)`` tensor in ``[0, 1]``, the size it arrived at.

    Raises:
        ValueError: Either socket is empty.
    """
    import torch.nn.functional as functional

    modality = marigold_v2.MODALITIES[name]
    missing = [
        socket
        for socket, wired in ((TRANSFORMER, model), (DECODER, vae))
        if wired is None
    ]
    if missing:
        raise ValueError(
            f"{marigold_v2.MODEL_NAME} reads its model off the node's own sockets, and "
            f"{' and '.join(missing)} {'is' if len(missing) == 1 else 'are'} not wired.\n"
            f"  Wire Load Diffusion Model on qwen_image_edit_2509_int8_convrot into "
            f"{TRANSFORMER}, and Load VAE on the {modality} VAE into {DECODER}.\n"
            f"  The {modality} adapter is put on the transformer here, so no LoRA loader "
            f"is needed.\n"
            f"  Nothing is downloaded for it: place the files in your models folders "
            f"yourself, and the conditioning in ComfyUI/models/embeddings."
        )

    planes = image[..., :3].permute(0, 3, 1, 2).to(dtype=torch.float32)
    height, width = planes.shape[-2:]
    # Held to a multiple the decoder and the transformer both take.
    working = _working_size(height, width, resolution)
    step = marigold_v2.MULTIPLE
    working = tuple(max(step, int(round(side / step)) * step) for side in working)
    if working != (height, width):
        planes = functional.interpolate(
            planes, size=working, mode="bicubic", align_corners=False
        ).clamp(0.0, 1.0)

    # A wired conditioning wins over the file.
    prompt = _prompt_of(conditioning)
    if prompt is None:
        prompt = marigold_v2.conditioning(conditioning_name, modality)
    decoded = marigold_v2.predict(
        marigold_v2.adapted(model, adapter_name, modality),
        vae,
        planes.permute(0, 2, 3, 1).contiguous(),
        prompt,
    )
    if modality == "depth":
        # Stretched over the whole batch.
        answer = marigold_v2.depth(decoded)
    elif modality == "normals":
        answer = marigold_v2.normals(decoded)
    else:
        answer = marigold_v2.albedo(decoded)

    answer = answer.permute(0, 3, 1, 2)
    if answer.shape[-2:] != (height, width):
        answer = functional.interpolate(
            answer, size=(height, width), mode="bicubic", align_corners=False
        )
    answer = answer.clamp(0.0, 1.0).permute(0, 2, 3, 1)
    return answer.to(image.device, dtype=image.dtype).contiguous()


def _without_model(name, image, resolution: int, settings):
    """Work out one of the answers that needs no weights.

    Args:
        name: A key of :data:`CONTROLS` whose :data:`MODELS` entry is empty.
        image: ``(batch, height, width, channels)`` in ``[0, 1]``.
        resolution: Longest edge the work is done at.
        settings: The shared settings, already held to this preprocessor's bounds.

    Returns:
        A ``(batch, 3, height, width)`` tensor on a 0 to 255 scale.
    """
    import torch.nn.functional as functional

    from ....modules.image import preprocess
    from ....modules.model import compute_device

    device = compute_device()
    planes = image[..., :3].permute(0, 3, 1, 2).to(device=device, dtype=torch.float32) * 255.0
    height, width = planes.shape[-2:]
    working = _working_size(height, width, resolution)
    if working != (height, width):
        planes = functional.interpolate(planes, size=working, mode="area")

    if name == "canny_pyramid":
        result = preprocess.pyramid_canny(planes, settings[LOW], settings[HIGH])
    elif name == "lineart_simple":
        result = preprocess.lineart_simple(planes, settings[RADIUS], settings[LOW])
    elif name == "scribble_xdog":
        result = preprocess.scribble_xdog(planes, settings[LOW])
    elif name == "binary":
        result = preprocess.binary(planes, settings[LOW])
    else:
        result = preprocess.shuffle(planes, settings[SEED])

    if result.shape[-2:] != (height, width):
        result = functional.interpolate(
            result, size=(height, width), mode="bicubic", align_corners=False
        )
    return result


def _with_model(name, loaded, image, resolution: int, settings):
    """Work out one of the answers a model provides.

    Args:
        name: A key of :data:`CONTROLS` whose :data:`MODELS` entry is not empty.
        loaded: The built model and the name it came from.
        image: ``(batch, height, width, channels)`` in ``[0, 1]``.
        resolution: Longest edge the model reads at.
        settings: The shared settings, already held to this preprocessor's bounds.

    Returns:
        A ``(batch, 3, height, width)`` tensor on a 0 to 255 scale.
    """
    from ....modules.image import preprocess, preprocess_models

    if name == "depth_map":
        return preprocess_models.depth(loaded, image, resolution).repeat(1, 3, 1, 1)
    if name == "normal_map":
        estimate = preprocess_models.depth(loaded, image, resolution)
        return preprocess.normal_from_depth(
            estimate, settings[STRENGTH], int(settings[RADIUS])
        )
    if name in ("openpose", "animal_pose"):
        return preprocess_models.skeleton(loaded, image, settings[LOW])
    if name == "ade20k_segments":
        return preprocess_models.segments(loaded, image, resolution)
    if name == "soft_edge":
        return preprocess_models.soft_edges(loaded, image, resolution)
    if name == "lineart_model":
        return preprocess_models.lines(loaded, image, resolution)
    if name == "anyline":
        return preprocess_models.anyline(loaded, image, resolution, settings[LOW])
    if name in ("denoise", "low_light"):
        family = RESTORE_FAMILY.get(loaded.name, "retinexformer")
        return preprocess_models.restore(loaded, image, family, settings[TILE])
    if name in INTRINSIC:
        return preprocess_models.intrinsics(
            loaded, image, resolution, settings[STEPS], settings[SEED], name
        )
    return preprocess_models.line_segments(
        loaded, image, resolution, settings[LOW], settings[HIGH]
    )


def _working_size(height: int, width: int, edge: int) -> tuple[int, int]:
    """The size an answer is worked out at, the longest edge held to ``edge``.

    Args:
        height: Source height in pixels.
        width: Source width in pixels.
        edge: Longest edge to work at.

    Returns:
        ``(height, width)``, never below one pixel on either side.
    """
    longest = max(height, width)
    # Held to the source: working larger would resample detail that is not there.
    edge = min(int(edge), longest)
    if longest == edge:
        return height, width
    scale = edge / float(longest)
    # Scaling by the longest side alone crushes a long thin frame to a sliver, so the short
    # side keeps what it can of NARROWEST without ever growing past the source.
    return (
        max(min(NARROWEST, height), int(round(height * scale))),
        max(min(NARROWEST, width), int(round(width * scale))),
    )
