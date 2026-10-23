"""Parametric Trajectory Distillation step adapters for MiniMax H3, applied as a model patch.

Reads the Ref2VA release (factors stacked by step, a 32-step head bank) and the fl2va release
(a factor pair per step, head adapters), in bfloat16 or int8.
"""

from __future__ import annotations

import copy
import json
import os
import re
import struct

import torch
import torch.nn.functional as F

from .. import log

logger = log.get_logger("modules.model.ptd")

__all__ = [
    "FL2VA", "FOLDER", "REF2VA", "Bank", "apply", "describe", "hop_of", "is_ptd_file",
    "load", "off_grid", "read_header", "register", "release_of",
]

#: The two releases a file can hold.
REF2VA = "ref2va"
FL2VA = "fl2va"

#: The model folder PTD files are kept in, also listed by ComfyUI's Load Model Patch node.
FOLDER = "ptd"

#: Key the patch's injections and wrapper are filed under on a model.
KEY = "was_ptd"

#: Bytes a safetensors header may take.
MAX_HEADER = 64 * 1024 * 1024

#: Diffusers projection names, each with the native module it adapts and the third of the fused
#: query, key and value projection it fills.
PROJECTIONS = {
    "attn.to_q": ("attn.qkv_proj", 0),
    "attn.to_k": ("attn.qkv_proj", 1),
    "attn.to_v": ("attn.qkv_proj", 2),
    "attn.to_out.0": ("attn.out_proj", None),
    "ff.net.0.proj": ("mlp.fc1", None),
    "ff.net.2": ("mlp.fc2", None),
}

#: Diffusers block names, each with the native block path it maps to.
BLOCKS = (
    (re.compile(r"transformer_blocks\.(\d+)\.(.+)"), "blocks"),
    (re.compile(r"token_refiner\.refiner_blocks\.(\d+)\.(.+)"), "token_refiner.blocks"),
)

#: Native module of each stream's output head.
VIDEO_HEAD = "final_layer.video_out"
AUDIO_HEAD = "final_layer.audio_out"

#: fl2va step times by step count: knots of a shift 6 schedule, the first held at 0.999.
FL2VA_GRIDS = {
    4: (0.999, 0.947, 0.857, 0.667, 0.0),
    6: (0.999, 0.968, 0.923, 0.857, 0.750, 0.545, 0.0),
}

#: The shift the fl2va step times are knots of.
FL2VA_SHIFT = 6.0

#: fl2va's audio clock: shifted by this factor against video, on a clock ending at FL2VA_T_MAX.
FL2VA_AUDIO_RELATIVE = 0.25
FL2VA_T_MAX = 0.999

#: Ref2VA's video and audio shifts.
REF2VA_SHIFT = 12.0
REF2VA_AUDIO_SHIFT = 3.0

#: Distance from a step time within which a sampler's sigma counts as that step.
KNOT_TOLERANCE = 5e-3

_FL2VA_KEY = re.compile(r"^transformer\.(?P<module>.+)\.lora_(?P<factor>[AB])\.(?P<hop>\d+)\.weight$")
_FL2VA_HEADS = {"proj_out.main": VIDEO_HEAD, "audio_proj_out.main": AUDIO_HEAD}
_MISSING = object()


def read_header(path) -> tuple[dict, dict]:
    """Return a safetensors file's metadata and each tensor's dtype and shape, reading no data.

    Args:
        path: The file.

    Returns:
        ``(metadata, tensors)``: the string metadata, and the header entry of every tensor.

    Raises:
        ValueError: The file is not a safetensors file, or its header claims more bytes than
            the file holds or than :data:`MAX_HEADER`.
    """
    size = os.path.getsize(path)
    with open(path, "rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"{os.path.basename(path)} is not a safetensors file")
        (length,) = struct.unpack("<Q", raw)
        if length > min(MAX_HEADER, size - 8):
            raise ValueError(f"{os.path.basename(path)} has a header larger than the file allows")
        header = json.loads(handle.read(length))
    if not isinstance(header, dict):
        raise ValueError(f"{os.path.basename(path)} has no tensor table")
    metadata = header.pop("__metadata__", None) or {}
    return metadata, header


def release_of(metadata, keys) -> str | None:
    """Which PTD release a file holds.

    Args:
        metadata: The file's metadata.
        keys: Its tensor names.

    Returns:
        :data:`REF2VA`, :data:`FL2VA`, or None for any other file.
    """
    names = list(keys)
    if "ptd_num_hops" in metadata and any(name.endswith(".lora_down") for name in names):
        return REF2VA
    if ("ptd_hops" in metadata or "ptd_recipe" in metadata) and any(".lora_A." in name for name in names):
        return FL2VA
    return None


def is_ptd_file(path) -> bool:
    """Whether a file is a PTD release, judged from its header alone. Never raises."""
    try:
        metadata, tensors = read_header(path)
    except (OSError, ValueError) as error:
        logger.debug("%s is not read as PTD (%s)", os.path.basename(str(path)), error)
        return False
    return release_of(metadata, tensors) is not None


def shifted(shift: float, t: float) -> float:
    """A flow schedule's sigma at time ``t`` under ``shift``."""
    return shift * t / (1.0 + (shift - 1.0) * t)


def fl2va_audio_sigma(t: float) -> float:
    """fl2va's audio sigma at shared time ``t``."""
    t = min(max(float(t), 0.0), FL2VA_T_MAX)
    if t == FL2VA_T_MAX:
        return FL2VA_T_MAX
    r = FL2VA_AUDIO_RELATIVE
    return r * t * FL2VA_T_MAX / (FL2VA_T_MAX + (r - 1.0) * t)


def fl2va_audio_rate(t: float) -> float:
    """How fast fl2va's audio sigma moves with shared time at ``t``."""
    t = min(max(float(t), 0.0), FL2VA_T_MAX)
    r = FL2VA_AUDIO_RELATIVE
    return r * FL2VA_T_MAX ** 2 / (FL2VA_T_MAX + (r - 1.0) * t) ** 2


def fl2va_audio_shift(t: float) -> float:
    """The audio shift H3 takes, against a video shift of 1, to place audio on fl2va's clock."""
    t = min(max(float(t), 0.0), FL2VA_T_MAX)
    if t == 0.0:
        return FL2VA_AUDIO_RELATIVE
    if t == FL2VA_T_MAX:
        return 1.0
    sigma = fl2va_audio_sigma(t)
    return sigma * (1.0 - t) / (t * (1.0 - sigma))


def hop_of(grid, sigma: float) -> tuple[int, bool]:
    """The step whose adapters serve a sigma, and whether the sigma is one of the step times.

    Args:
        grid: Step times, descending, ending at 0.
        sigma: The sigma the model is evaluated at.

    Returns:
        ``(hop, on_grid)``. Off the grid, the hop is the one whose interval holds the sigma.
    """
    hops = len(grid) - 1
    gaps = [grid[k] - grid[k + 1] for k in range(hops)]
    tolerance = min(KNOT_TOLERANCE, min(gaps) / 4.0)
    for k in range(hops):
        if abs(sigma - grid[k]) <= tolerance:
            return k, True
    if sigma > grid[0]:
        return 0, False
    for k in range(hops):
        if grid[k + 1] < sigma:
            return k, False
    return hops - 1, False


class Factor(torch.nn.Module):
    """One low-rank factor, kept in system memory.

    Attributes:
        weight: ``(out, in)``, floating point or int8.
        scale: The float32 scale of each output row of an int8 weight, or None.
    """

    def __init__(self, weight: torch.Tensor, scale: torch.Tensor | None = None):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.scale = None if scale is None else torch.nn.Parameter(
            scale.reshape(-1, 1).to(torch.float32), requires_grad=False)


#: Byte alignment of each tensor in the GPU slot a block's factors are copied into.
_ALIGN = 256

#: Rows of the feed-forward activation formed at a time for fc2's update.
_ACT_ROWS = 2048


def _dequant(weight: torch.Tensor, scale: torch.Tensor | None, dtype) -> torch.Tensor:
    """A staged factor's weight in ``dtype``, an int8 weight's rows scaled back."""
    if scale is not None:
        return (weight.to(torch.float32) * scale).to(dtype)
    return weight.to(dtype)


def _through(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor | None) -> torch.Tensor:
    """``x`` through a staged factor, in ``x``'s precision."""
    return F.linear(x, _dequant(weight, scale, x.dtype))


def _add_low(rows: torch.Tensor, low: torch.Tensor, up: torch.Tensor, up_scale, part, scale: float):
    """Add ``low`` through an up factor into ``rows``, or one third of its columns, in place.

    Args:
        rows: The projection's output as ``(tokens, width)``.
        low: The input through the down factor, ``(tokens, rank)``.
        up: The staged up factor.
        up_scale: Its int8 row scales, or None.
        part: The third of ``rows`` the update fills, or None for all of it.
        scale: Multiplier on the update.
    """
    weight = _dequant(up, up_scale, rows.dtype)
    width = weight.shape[0]
    target = rows if part is None else rows[:, part * width:(part + 1) * width]
    target.addmm_(low.to(rows.dtype), weight.t(), alpha=scale)


def _act_low(h: torch.Tensor, down: torch.Tensor, down_scale, act) -> torch.Tensor:
    """``act(h)`` through a down factor, the activation formed a slice of rows at a time."""
    flat = h.reshape(-1, h.shape[-1])
    weight = _dequant(down, down_scale, h.dtype)
    low = flat.new_empty((flat.shape[0], weight.shape[0]))
    for start in range(0, flat.shape[0], _ACT_ROWS):
        stop = min(start + _ACT_ROWS, flat.shape[0])
        torch.mm(act(flat[start:stop]), weight.t(), out=low[start:stop])
    return low


def _block_of(path: str) -> str:
    """The block a native module path belongs to, such as ``blocks.12`` or ``final_layer``."""
    match = re.match(r"^((?:token_refiner\.)?blocks\.\d+)\.", path)
    return match.group(1) if match else path.rsplit(".", 1)[0]


class HeadBank(torch.nn.Module):
    """A stacked output head in the layout H3's final layer reads a PDD head bank from.

    Attributes:
        weight: ``(steps * width, hidden)``: the first step's head, then each later step's
            difference from it.
        bias: ``(steps * width,)`` in the same layout.
    """

    comfy_cast_weights = True
    weight_function = []
    bias_function = []

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.bias = torch.nn.Parameter(bias, requires_grad=False)
        self.out_features, self.in_features = weight.shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` through the first step's head."""
        import comfy.ops

        with comfy.ops.CastBiasWeightContext(self, x, offloadable=True) as (weight, bias):
            width = self.out_features // max(1, self.steps)
            return F.linear(x, weight[:width], bias[:width])

    @property
    def steps(self) -> int:
        """How many steps the bank holds a head for."""
        return getattr(self, "_steps", 1)

    def on(self, device) -> "HeadBank":
        """A copy of the bank on ``device``."""
        copy_ = HeadBank(self.weight.to(device, non_blocking=True), self.bias.to(device, non_blocking=True))
        copy_._steps = self.steps
        return copy_


class Target(torch.nn.Module):
    """Every step's low-rank update for one native projection.

    Attributes:
        block: The block the projection belongs to, whose factors share one GPU copy.
        staged: The current step's factors on the GPU while its block is loaded, else empty.
    """

    def __init__(self, hops: int, block: str = ""):
        super().__init__()
        self.block = block
        self.downs = torch.nn.ModuleList(torch.nn.ModuleList() for _ in range(hops))
        self.ups = torch.nn.ModuleList(torch.nn.ModuleList() for _ in range(hops))
        self.parts: list[list[int | None]] = [[] for _ in range(hops)]
        self.staged: list[tuple] = []

    def add(self, hop: int, down: Factor, up: Factor, part: int | None) -> None:
        """File one step's factor pair, writing into third ``part`` of the output, or all of it."""
        self.downs[hop].append(down)
        self.ups[hop].append(up)
        self.parts[hop].append(part)

    def update(self, x: torch.Tensor, out: torch.Tensor, scale: float, factors: list) -> torch.Tensor:
        """``out`` with the step's update for input ``x`` added in place.

        Args:
            x: The projection's input.
            out: Its output.
            scale: Multiplier on the update.
            factors: The step's staged factors, from :meth:`Bank.factors`.

        Returns:
            ``out``, or a contiguous copy of it where it was not contiguous.
        """
        if not factors:
            return out
        if not out.is_contiguous():
            out = out.contiguous()
        rows = out.view(-1, out.shape[-1])
        flat = x.reshape(-1, x.shape[-1])
        for down, down_scale, up, up_scale, part in factors:
            _add_low(rows, _through(flat, down, down_scale), up, up_scale, part, scale)
        return out


class Bank(torch.nn.Module):
    """A PTD file's step adapters and heads, as the model patch ComfyUI's loader hands on.

    Attributes:
        release: :data:`REF2VA` or :data:`FL2VA`.
        name: The file name.
        hops: Steps the file was trained for.
        grid: Step times, descending, ending at 0.
        shift: Video shift the step times are knots of.
        audio_shift: Audio shift the sampler is set to.
        scale: Multiplier on every update.
        storage: ``int8`` or the factors' floating point precision.
        targets: Native module path, with dots as ``/``, to its :class:`Target`.
        video_bank: Ref2VA's video head bank, or None.
        audio_bank: Ref2VA's audio head bank, or None.
        base_heads: fl2va's reference video head weight and bias, or None.
        slot_bytes: Bytes of the GPU slot one block's factors for one step are copied into.
    """

    def __init__(self, release: str, name: str, hops: int, grid, shift: float,
                 audio_shift: float, scale: float):
        super().__init__()
        self.release = release
        self.name = name
        self.hops = hops
        self.grid = tuple(float(value) for value in grid)
        self.shift = shift
        self.audio_shift = audio_shift
        self.scale = scale
        self.storage = "bf16"
        self.targets = torch.nn.ModuleDict()
        self.video_bank: HeadBank | None = None
        self.audio_bank: HeadBank | None = None
        self.base_heads: tuple[torch.Tensor, torch.Tensor] | None = None
        self.slot_bytes = 0
        self._layout: dict[tuple[int, str], list] = {}
        self._slot: torch.Tensor | None = None
        self._hop: int | None = None
        self._loaded: str | None = None

    def target(self, path: str) -> Target:
        """The :class:`Target` for a native module path, made when new."""
        key = path.replace(".", "/")
        if key not in self.targets:
            self.targets[key] = Target(self.hops, _block_of(path))
        return self.targets[key]

    def paths(self):
        """Every native module path the bank adapts, with its :class:`Target`."""
        for key, target in self.targets.items():
            yield key.replace("/", "."), target

    def plan(self) -> None:
        """Group each step's factors by block and size the GPU slot the largest group needs."""
        self._layout = {}
        for target in self.targets.values():
            for hop in range(self.hops):
                pieces = list(zip(target.downs[hop], target.ups[hop], target.parts[hop]))
                if pieces:
                    self._layout.setdefault((hop, target.block), []).append((target, pieces))
        largest = 0
        for entries in self._layout.values():
            offset = 0
            for _, pieces in entries:
                for down, up, _ in pieces:
                    for tensor in (down.weight, down.scale, up.weight, up.scale):
                        if tensor is not None:
                            offset += -(-tensor.numel() * tensor.element_size() // _ALIGN) * _ALIGN
            largest = max(largest, offset)
        self.slot_bytes = largest

    def stage(self, hop: int, device) -> tuple[HeadBank, HeadBank] | None:
        """Ready step ``hop`` for one model call: a GPU slot for its blocks, and any head bank.

        Returns:
            The video and audio head banks on ``device``, or None for a file with none.
        """
        if not self._layout:
            self.plan()
        for target in self.targets.values():
            target.staged = []
        self._slot = torch.empty(self.slot_bytes, dtype=torch.uint8, device=device)
        self._hop = hop
        self._loaded = None
        if self.video_bank is None:
            return None
        return self.video_bank.on(device), self.audio_bank.on(device)

    def factors(self, target: Target) -> list:
        """The staged step's factors for ``target``, copying its block to the GPU if another is there."""
        if self._hop is None or self._slot is None:
            return []
        if self._loaded != target.block:
            self._load(target.block)
        return target.staged

    def _load(self, block: str) -> None:
        """Copy one block's factors for the staged step into the GPU slot."""
        if self._loaded is not None:
            for target, _ in self._layout.get((self._hop, self._loaded), ()):
                target.staged = []
        offset = 0
        for target, pieces in self._layout.get((self._hop, block), ()):
            staged = []
            for down, up, part in pieces:
                views = []
                for tensor in (down.weight, down.scale, up.weight, up.scale):
                    if tensor is None:
                        views.append(None)
                        continue
                    size = tensor.numel() * tensor.element_size()
                    view = self._slot[offset:offset + size].view(tensor.dtype).view(tensor.shape)
                    view.copy_(tensor, non_blocking=True)
                    views.append(view)
                    offset += -(-size // _ALIGN) * _ALIGN
                staged.append((*views, part))
            target.staged = staged
        self._loaded = block

    def unstage(self) -> None:
        """Release the GPU slot and every staged factor."""
        for target in self.targets.values():
            target.staged = []
        self._slot = None
        self._hop = None
        self._loaded = None

    def pin(self) -> bool:
        """Move every tensor to page-locked memory, so a block's copy to the GPU is fast.

        Returns:
            True when every tensor was pinned. False when the system refused, in which case
            the tensors stay where they were.
        """
        try:
            for module in self.modules():
                for name, parameter in list(module.named_parameters(recurse=False)):
                    if not parameter.is_pinned():
                        setattr(module, name, torch.nn.Parameter(parameter.data.pin_memory(), requires_grad=False))
        except RuntimeError as error:
            logger.warning("%s stays in pageable memory (%s); each step copies more slowly", self.name, error)
            return False
        return True

    def nbytes(self) -> int:
        """Bytes the bank holds."""
        return sum(p.numel() * p.element_size() for p in self.parameters())


def _native_block(prefix: str) -> tuple[str, str] | None:
    """Split a diffusers module name into its native block path and its projection name."""
    for pattern, native in BLOCKS:
        match = pattern.fullmatch(prefix)
        if match is not None:
            return f"{native}.{match.group(1)}", match.group(2)
    return None


def _swap_halves(tensor: torch.Tensor) -> torch.Tensor:
    """Rows reordered from diffusers' ``[up, gate]`` to H3's ``[gate, up]``."""
    first, second = tensor.chunk(2, dim=0)
    return torch.cat((second, first), dim=0).contiguous()


def _build_ref2va(sd, metadata, name) -> Bank:
    """A :class:`Bank` from a Ref2VA state dict."""
    hops = int(metadata["ptd_num_hops"])
    if hops < 1:
        raise ValueError(f"{name} names {hops} steps")
    rank = float(metadata.get("lora_rank", 1) or 1)
    alpha = float(metadata.get("lora_alpha", rank) or rank)
    grid = [shifted(REF2VA_SHIFT, 1.0 - k / hops) for k in range(hops)] + [0.0]
    bank = Bank(REF2VA, name, hops, grid, REF2VA_SHIFT, REF2VA_AUDIO_SHIFT, alpha / rank)
    int8 = False
    for key in sorted(sd):
        if not key.endswith(".lora_down"):
            continue
        prefix = key[: -len(".lora_down")]
        split = _native_block(prefix)
        if split is None or split[1] not in PROJECTIONS:
            raise ValueError(f"{name} adapts {prefix}, which MiniMax H3 has no projection for")
        native, part = PROJECTIONS[split[1]]
        downs, ups = sd[key], sd[f"{prefix}.lora_up"]
        down_scales = sd.get(f"{prefix}.lora_down_scale")
        up_scales = sd.get(f"{prefix}.lora_up_scale")
        int8 = int8 or downs.dtype == torch.int8
        if downs.shape[0] != hops or ups.shape[0] != hops:
            raise ValueError(f"{name}: {prefix} holds {downs.shape[0]} steps, not {hops}")
        target = bank.target(f"{split[0]}.{native}")
        for hop in range(hops):
            up, up_scale = ups[hop], None if up_scales is None else up_scales[hop]
            if native == "mlp.fc1":
                up = _swap_halves(up)
                up_scale = None if up_scale is None else _swap_halves(up_scale)
            target.add(hop, Factor(downs[hop], None if down_scales is None else down_scales[hop]),
                       Factor(up, up_scale), part)
    for source, attribute in (("proj_out", "video_bank"), ("audio_proj_out", "audio_bank")):
        weight, bias = sd[f"{source}.weight"].to(torch.float32), sd[f"{source}.bias"].to(torch.float32)
        steps, width, hidden = weight.shape
        weight[1:] -= weight[0].clone()
        bias[1:] -= bias[0].clone()
        head = HeadBank(weight.reshape(steps * width, hidden).contiguous(),
                        bias.reshape(steps * width).contiguous())
        head._steps = steps
        setattr(bank, attribute, head)
    bank.storage = "int8" if int8 else "bf16"
    return bank


def _build_fl2va(sd, metadata, name) -> Bank:
    """A :class:`Bank` from an fl2va state dict."""
    hops = int(metadata.get("ptd_hops") or {"h3_4step": 4, "h3_6step": 6}.get(metadata.get("ptd_recipe"), 0))
    if hops not in FL2VA_GRIDS:
        raise ValueError(f"{name} is a {hops}-step fl2va file; the 4 and 6 step releases are read")
    scale = float(metadata.get("ptd_lora_scale", 1.0) or 1.0)
    bank = Bank(FL2VA, name, hops, FL2VA_GRIDS[hops], FL2VA_SHIFT, FL2VA_SHIFT, scale)
    pairs: dict[tuple[str, int], dict[str, str]] = {}
    for key in sd:
        match = _FL2VA_KEY.fullmatch(key)
        if match is not None:
            pairs.setdefault((match["module"], int(match["hop"])), {})[match["factor"]] = key
    if not pairs:
        raise ValueError(f"{name} holds no fl2va step adapters")
    int8 = False
    for (module, hop), factors in sorted(pairs.items()):
        if hop >= hops or set(factors) != {"A", "B"}:
            raise ValueError(f"{name}: {module} step {hop} is incomplete")
        if module in _FL2VA_HEADS:
            path, part = _FL2VA_HEADS[module], None
        else:
            split = _native_block(module)
            if split is None or split[1] not in PROJECTIONS:
                raise ValueError(f"{name} adapts {module}, which MiniMax H3 has no projection for")
            native, part = PROJECTIONS[split[1]]
            path = f"{split[0]}.{native}"
        down, up = sd[factors["A"]], sd[factors["B"]]
        down_scale, up_scale = sd.get(factors["A"] + "_scale"), sd.get(factors["B"] + "_scale")
        int8 = int8 or down.dtype == torch.int8
        if path.endswith(".mlp.fc1"):
            up = _swap_halves(up)
            up_scale = None if up_scale is None else _swap_halves(up_scale)
        bank.target(path).add(hop, Factor(down, down_scale), Factor(up, up_scale), part)
    for (module, hop) in [(m, h) for m in {m for m, _ in pairs} for h in range(hops)]:
        if (module, hop) not in pairs:
            raise ValueError(f"{name}: {module} has no adapter for step {hop}")
    reference = sd.get("transformer.proj_out.orig_weight"), sd.get("transformer.proj_out.orig_bias")
    if reference[0] is not None and reference[1] is not None:
        bank.base_heads = (reference[0].to(torch.float32), reference[1].to(torch.float32))
    bank.storage = "int8" if int8 else "bf16"
    return bank


def build(sd, metadata, name: str) -> Bank:
    """A :class:`Bank` from a loaded PTD file.

    Args:
        sd: The file's tensors.
        metadata: The file's metadata.
        name: The file name, for messages.

    Returns:
        The bank.

    Raises:
        ValueError: The file is not a PTD release, or one of its adapters is incomplete.
    """
    release = release_of(metadata, sd.keys())
    if release == REF2VA:
        return _build_ref2va(sd, metadata, name)
    if release == FL2VA:
        return _build_fl2va(sd, metadata, name)
    raise ValueError(f"{name} is not a PTD file")


def load(path):
    """Load a PTD file as a model patch.

    Args:
        path: The file.

    Returns:
        A ``ModelPatcher`` whose model is the file's :class:`Bank`.
    """
    import comfy.model_management
    import comfy.model_patcher
    import comfy.utils

    name = os.path.basename(str(path))
    metadata, _ = read_header(path)
    sd = comfy.utils.load_torch_file(str(path), safe_load=True)
    bank = build(sd, metadata, name)
    del sd
    if comfy.model_management.get_torch_device().type == "cuda":
        bank.pin()
    bank.plan()
    logger.info("loaded %s: %s, %d steps, %s, %.2f GB, %.0f MB on the GPU per block", name,
                bank.release, bank.hops, bank.storage, bank.nbytes() / 1e9, bank.slot_bytes / 1e6)
    return comfy.model_patcher.CoreModelPatcher(
        bank, load_device=comfy.model_management.get_torch_device(),
        offload_device=comfy.model_management.unet_offload_device())


class _State:
    """The step a running model call serves, read by every adapted projection."""

    def __init__(self):
        self.hop: int | None = None
        self.staged: int | None = None
        self.heads = None
        self.audio_rate = 1.0
        self.warned: set[int] = set()


def _adapted(original, target: Target, bank: Bank, state: _State, scale: float, audio: bool):
    """A forward that adds ``target``'s update for the current step to ``original``'s output."""

    def forward(x, *args, **kwargs):
        out = original(x, *args, **kwargs)
        if state.hop is None:
            return out
        out = target.update(x, out, scale, bank.factors(target))
        if audio and state.audio_rate != 1.0:
            out = out.mul_(state.audio_rate)
        return out

    return forward


def _kept(original, held: dict):
    """A forward that leaves its output in ``held`` for the feed-forward wrapper to read."""

    def forward(x, *args, **kwargs):
        out = original(x, *args, **kwargs)
        held["out"] = out
        return out

    return forward


def _feed_forward(original, target: Target, bank: Bank, state: _State, scale: float, act, held: dict):
    """A feed-forward whose fused fc2 runs as ComfyUI runs it, with fc2's update added after."""

    def forward(x, *args, **kwargs):
        held.pop("out", None)
        out = original(x, *args, **kwargs)
        h = held.pop("out", None)
        if state.hop is None or h is None:
            return out
        factors = bank.factors(target)
        if not factors:
            return out
        if not out.is_contiguous():
            out = out.contiguous()
        rows = out.view(-1, out.shape[-1])
        for down, down_scale, up, up_scale, part in factors:
            _add_low(rows, _act_low(h, down, down_scale, act), up, up_scale, part, scale)
        return out

    return forward


def _swiglu():
    """ComfyUI's eager SwiGLU, the activation H3's fused fc2 applies, or None where it has none."""
    import comfy.ops

    return getattr(comfy.ops, "INPUT_ACT_EAGER", {}).get("swiglu")


def _raw_text(text_states):
    """Text states passed on unrefined, so the refiner runs inside each model call."""
    return text_states


def _check_model(model, bank: Bank) -> list[str]:
    """Refuse a model the bank cannot adapt, and return notes on what was checked.

    Raises:
        ValueError: The model is not MiniMax H3, already carries a head bank or PTD, or a
            projection's size differs from the file's.
    """
    diffusion = model.get_model_object("diffusion_model")
    final = getattr(diffusion, "final_layer", None)
    if final is None or not hasattr(diffusion, "token_refiner") or not hasattr(diffusion, "preprocess_text_embeds"):
        raise ValueError("H3 PTD Patch takes a MiniMax H3 model.")
    if model.get_injections(KEY) is not None:
        raise ValueError("This model already carries a PTD patch. Patch the model from the loader instead.")
    head_rows = final.video_out.weight.shape[0]
    if head_rows != final.video_dim:
        raise ValueError(
            "This model already carries a PDD head bank. Load a plain MiniMax H3 transformer, "
            "with no PDD or PTD weights merged into it.")
    for path, target in bank.paths():
        try:
            module = diffusion.get_submodule(path)
        except AttributeError:
            raise ValueError(f"{bank.name} adapts {path}, which this model does not have.") from None
        down, up = target.downs[0][0], target.ups[0][0]
        width = up.weight.shape[0] * (3 if target.parts[0][0] is not None else 1)
        if down.weight.shape[1] != module.in_features or width != module.out_features:
            raise ValueError(
                f"{bank.name} does not fit this model: {path} is {module.in_features} to "
                f"{module.out_features} wide, and the file's adapter is {down.weight.shape[1]} to {width}.")
    notes = []
    if bank.base_heads is not None:
        weight = final.video_out.weight
        distance = None
        if weight.dtype in (torch.float32, torch.float16, torch.bfloat16) and weight.device.type != "meta":
            reference = bank.base_heads[0]
            try:
                distance = float((weight.detach().to("cpu", torch.float32) - reference).norm() / reference.norm())
            except (RuntimeError, TypeError) as error:
                logger.debug("the transformer's video head could not be compared (%s)", error)
        if distance is not None:
            if distance > 1e-3:
                raise ValueError(
                    f"{bank.name} was trained on the fl2va transformer, and this model's video head "
                    f"differs from it ({distance:.1%}). Load an fl2va MiniMax H3 transformer, or a "
                    "Ref2VA PTD file for a ref2va one.")
            notes.append("fl2va transformer, matched")
        else:
            notes.append("transformer head not comparable, unchecked")
    else:
        notes.append("ref2va transformer expected, unchecked")
    return notes


def apply(model, bank: Bank):
    """Return a copy of a MiniMax H3 model carrying a PTD file's step adapters and heads.

    Args:
        model: The ``ModelPatcher`` of a plain MiniMax H3 transformer, LoRAs allowed.
        bank: The PTD file, from :func:`load`.

    Returns:
        ``(model, notes)``: the patched model, and what was checked about the base.

    Raises:
        ValueError: The model cannot take this file; :func:`_check_model` names why.
    """
    import comfy.model_management
    import comfy.patcher_extension

    notes = _check_model(model, bank)
    patched = model.clone()
    sampling = copy.deepcopy(patched.get_model_object("model_sampling"))
    if not hasattr(sampling, "audio_shift"):
        raise ValueError("H3 PTD Patch takes a MiniMax H3 model.")
    sampling.set_parameters(shift=bank.shift, audio_shift=bank.audio_shift,
                            multiplier=getattr(sampling, "multiplier", 1000))
    patched.add_object_patch("model_sampling", sampling)
    patched.add_object_patch("diffusion_model.preprocess_text_embeds", _raw_text)

    state = _State()
    saved: list = []
    act = _swiglu()

    def keep(module):
        saved.append((module, module.__dict__.get("forward", _MISSING),
                      module.__dict__.get("comfy_force_forward", _MISSING)))

    def inject(model_patcher):
        diffusion = model_patcher.model.diffusion_model
        targets = dict(bank.paths())
        # A fused fc2 takes its update from the feed-forward around it.
        fused = {path[:-len(".fc2")]: target for path, target in targets.items()
                 if act is not None and path.endswith(".mlp.fc2")}
        for path, target in targets.items():
            if path.endswith(".mlp.fc2") and path[:-len(".fc2")] in fused:
                continue
            module = diffusion.get_submodule(path)
            keep(module)
            audio = bank.release == FL2VA and path == AUDIO_HEAD
            module.forward = _adapted(module.forward, target, bank, state, bank.scale, audio)
            if path.endswith(".mlp.fc2"):
                module.comfy_force_forward = True
        for path, target in fused.items():
            mlp = diffusion.get_submodule(path)
            held: dict = {}
            keep(mlp.fc1)
            mlp.fc1.forward = _kept(mlp.fc1.forward, held)
            keep(mlp)
            mlp.forward = _feed_forward(mlp.forward, target, bank, state, bank.scale, act, held)

    def eject(model_patcher):
        while saved:
            module, forward, force = saved.pop()
            if forward is _MISSING:
                module.__dict__.pop("forward", None)
            else:
                module.forward = forward
            if force is _MISSING:
                module.__dict__.pop("comfy_force_forward", None)
            else:
                module.comfy_force_forward = force

    patched.set_injections(KEY, [comfy.patcher_extension.PatcherInjection(inject=inject, eject=eject)])

    def wrapper(executor, x, timestep, context, transformer_options, *args, **kwargs):
        sigma = float(timestep.flatten()[0]) / 1000.0
        hop, on_grid = hop_of(bank.grid, sigma)
        schedule = transformer_options.get("sample_sigmas")
        if not on_grid and id(schedule) not in state.warned:
            state.warned.add(id(schedule))
            logger.warning(
                "%s was trained for %d steps at sigmas %s, and the sampler asked for sigma %.4f. "
                "Set the sampler to %d steps, euler, simple, cfg 1, or use this node's sigmas.",
                bank.name, bank.hops, ", ".join(f"{s:.3f}" for s in bank.grid[:-1]), sigma, bank.hops)
        final = executor.class_obj.final_layer
        staged_heads = state.heads if state.staged == hop else None
        if state.staged != hop:
            device = (x[0] if isinstance(x, (list, tuple)) else x).device
            staged_heads = bank.stage(hop, device)
        native_heads = None
        if bank.release == FL2VA:
            t = min(bank.grid[hop] if on_grid else sigma, FL2VA_T_MAX)
            transformer_options = dict(transformer_options)
            transformer_options["minimax_h3_sigma_shift_video"] = 1.0
            transformer_options["minimax_h3_sigma_shift_audio"] = fl2va_audio_shift(t)
            state.audio_rate = fl2va_audio_rate(t)
            if on_grid:
                timestep = torch.full_like(timestep, t * 1000.0)
        elif staged_heads is not None:
            native_heads = (final.video_out, final.audio_out)
            final.video_out, final.audio_out = staged_heads
        state.hop = hop
        try:
            return executor(x, timestep, context, transformer_options, *args, **kwargs)
        finally:
            state.hop = None
            state.audio_rate = 1.0
            if native_heads is not None:
                final.video_out, final.audio_out = native_heads
            if state.staged != hop:
                bank.unstage()

    def stage(executor, x, t, *args, **kwargs):
        # The step's factors are copied before the model call opens core's allocation graph.
        hop, _ = hop_of(bank.grid, float(t.flatten()[0]))
        state.heads = bank.stage(hop, comfy.model_management.get_torch_device())
        state.staged = hop
        try:
            return executor(x, t, *args, **kwargs)
        finally:
            state.staged = None
            state.heads = None
            bank.unstage()

    def check(executor, noise, latent_image, sampler, schedule, *args, **kwargs):
        stray = off_grid(bank, schedule)
        if stray:
            raise ValueError(
                f"{bank.name} is a {bank.hops}-step {_RELEASE_NAMES[bank.release]} PTD file, trained at "
                f"sigmas {_knots(bank.grid)}, and the sampler was given {_knots(schedule)}, which "
                f"leaves {_knots(stray)} off its steps. Wire this H3 PTD Patch's sigmas output into "
                f"the sampler, or a scheduler on this model at {bank.hops} steps, simple.")
        return executor(noise, latent_image, sampler, schedule, *args, **kwargs)

    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, KEY, check)
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.APPLY_MODEL, KEY, stage)
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, KEY, wrapper)
    return patched, notes


#: Each release's name in messages.
_RELEASE_NAMES = {REF2VA: "Ref2VA", FL2VA: "fl2va"}


def _knots(values) -> str:
    """Sigmas as a short comma-separated list."""
    return ", ".join(f"{float(v):.3f}" for v in values)


def off_grid(bank: Bank, schedule) -> list[float]:
    """The sigmas of a sampler schedule that are not among a bank's step times.

    Args:
        bank: The PTD file, from :func:`load`.
        schedule: The sigmas a sampler was given, ending at 0.

    Returns:
        Every sigma above 0 that is not within the tolerance of one of the bank's steps.
    """
    found = []
    for value in schedule:
        sigma = float(value)
        if sigma > 0 and not hop_of(bank.grid, sigma)[1]:
            found.append(sigma)
    return found


def sigmas(bank: Bank) -> torch.Tensor:
    """The bank's step times as a sampler schedule."""
    return torch.tensor(bank.grid, dtype=torch.float32)


def describe(bank: Bank) -> dict:
    """What a bank is, as the rows a report shows."""
    return {
        "release": "Ref2VA (taobao-content-ai)" if bank.release == REF2VA else "fl2va (EPFL)",
        "file": bank.name,
        "steps": str(bank.hops),
        "sigmas": ", ".join(f"{s:.4f}" for s in bank.grid),
        "adapters": (f"{bank.storage}, {bank.nbytes() / 1e9:.2f} GB in system memory, "
                     f"{bank.slot_bytes / 1e6:.0f} MB on the GPU per block"),
        "sampler": f"euler, scheduler simple, {bank.hops} steps, cfg 1",
    }


def register() -> None:
    """Teach ComfyUI's Load Model Patch node to load PTD files, and list ``models/ptd`` in it."""
    import folder_paths
    import nodes

    from . import model_directories

    for directory in model_directories(FOLDER):
        folder_paths.add_model_folder_path("model_patches", str(directory))
    loader = nodes.NODE_CLASS_MAPPINGS.get("ModelPatchLoader")
    if loader is None:
        from comfy_extras import nodes_model_patch

        loader = nodes_model_patch.ModelPatchLoader
    original = loader.load_model_patch
    if getattr(original, "_was_ptd", False):
        return

    def load_model_patch(self, name):
        path = folder_paths.get_full_path_or_raise("model_patches", name)
        if not is_ptd_file(path):
            return original(self, name)
        return (load(path),)

    load_model_patch._was_ptd = True
    loader.load_model_patch = load_model_patch
