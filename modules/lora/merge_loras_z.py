"""Merging one or more LoRA files into a single LoRA.

A LoRA is read as one :class:`LoraPair` per module: ``down`` and ``up`` factors plus the
``alpha`` scaling them.
"""

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple, Optional

import torch
from safetensors import safe_open
from tqdm import tqdm

from ..log import get_logger

logger = get_logger("lora.merge")


ProgressCallback = Callable[[str, int, int, Optional[str]], None]


def _progress(cb: Optional[ProgressCallback], stage: str, current: int, total: int, message: Optional[str] = None):
    if cb is None:
        return
    try:
        cb(stage, int(current), int(total), message)
    except Exception:
        pass


@dataclass
class LoraPair:
    down: torch.Tensor
    up: torch.Tensor
    alpha: float
    is_conv: bool
    in_hw: Tuple[int, int]


def parse_weighted_paths(items: List[str]) -> List[Tuple[str, float]]:
    result: List[Tuple[str, float]] = []
    for item in items:
        if "@" in item:
            path, weight_str = item.rsplit("@", 1)
            weight = float(weight_str)
            result.append((path, weight))
        else:
            result.append((item, 1.0))
    return result


def get_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_dtype(dtype_name: str) -> torch.dtype:
    name = dtype_name.lower()
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def get_compute_dtype(dtype_name: str) -> Optional[torch.dtype]:
    """
    Compute dtype is for internal merge math alignment. If None => auto behavior.
    """
    name = dtype_name.lower()
    if name in ("auto", "none"):
        return None
    return get_dtype(name)


def is_lora_key(key: str) -> bool:
    return (
        key.endswith("lora_down.weight")
        or key.endswith("lora_up.weight")
        or key.endswith("lora_down")
        or key.endswith("lora_up")
        or key.endswith("lora_A.weight")
        or key.endswith("lora_B.weight")
        or key.endswith("lora_A")
        or key.endswith("lora_B")
    )


def scan_adapter_compat(keys: List[str], path: str) -> None:
    """Reject a source file whose adapter format cannot be stacked as a plain LoRA.

    Args:
        keys: Tensor names in the file.
        path: Path of the file, named in the message.

    Raises:
        ValueError: The file carries LyCORIS weights that have no up/down pair: LoKr,
            LoHa or OFT/BOFT.
    """
    def any_key(*subs: str) -> bool:
        return any(any(sub in k for k in keys) for sub in subs)

    unsupported: List[str] = []
    if any_key("lokr_w1", "lokr_w2"):
        unsupported.append("LoKr")
    if any_key("hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b"):
        unsupported.append("LoHa")
    if any_key("oft_blocks", "oft_diag", ".boft_"):
        unsupported.append("OFT/BOFT")

    name = os.path.basename(path)
    # A format with no up/down pair would otherwise be dropped silently, leaving a merge
    # that is missing that file's entire contribution.
    if unsupported:
        raise ValueError(
            f"'{name}' uses {', '.join(unsupported)} (LyCORIS) weights, which the WAS LoRA "
            f"merger cannot stack. It only supports plain LoRA (lora_up/lora_down or "
            f"lora_A/lora_B). Remove this file from the merge or convert it to a standard "
            f"LoRA first."
        )

    if any_key(".dora_scale", "dora_scale"):
        logger.warning(
            "'%s' is a DoRA. Its magnitude vector (dora_scale) cannot be preserved and is "
            "dropped; only the LoRA direction is merged, so the result differs from "
            "applying the DoRA directly.", name,
        )


def get_module_prefix(key: str) -> str:
    if ".lora_down" in key:
        return key.split(".lora_down", 1)[0]
    if ".lora_up" in key:
        return key.split(".lora_up", 1)[0]
    if ".lora_A" in key:
        return key.split(".lora_A", 1)[0]
    if ".lora_B" in key:
        return key.split(".lora_B", 1)[0]
    parts = key.split(".")
    return ".".join(parts[:-2])


def get_alpha_key(prefix: str) -> str:
    return f"{prefix}.alpha"


def _get_save_key_style(metadata_ref: Dict[str, str]) -> str:
    style = (metadata_ref or {}).get("_was_lora_key_style", "")
    style_norm = str(style).strip().lower()
    if style_norm in ("ab", "a/b", "lora_a_b"):
        return "ab"
    return "downup"


def _down_key(prefix: str, key_style: str) -> str:
    if key_style == "ab":
        return f"{prefix}.lora_A.weight"
    return f"{prefix}.lora_down.weight"


def _up_key(prefix: str, key_style: str) -> str:
    if key_style == "ab":
        return f"{prefix}.lora_B.weight"
    return f"{prefix}.lora_up.weight"


def module_is_included(
    prefix: str,
    include_patterns: List[str],
    exclude_patterns: List[str],
) -> bool:
    if include_patterns:
        if not any(pat in prefix for pat in include_patterns):
            return False
    if exclude_patterns:
        if any(pat in prefix for pat in exclude_patterns):
            return False
    return True


def load_lora_pairs(
    path: str,
    device: torch.device,
    progress_cb: Optional[ProgressCallback] = None,
) -> Tuple[Dict[str, LoraPair], Dict[str, str]]:
    pairs: Dict[str, LoraPair] = {}
    metadata: Dict[str, str] = {}

    with safe_open(path, framework="pt", device=str(device)) as f:
        keys = list(f.keys())
        scan_adapter_compat(keys, path)
        try:
            raw_meta = f.metadata()
            metadata = dict(raw_meta) if isinstance(raw_meta, dict) else {}
        except Exception:
            metadata = {}

        detected_key_style: Optional[str] = None
        for k in keys:
            if k.endswith("lora_A.weight") or k.endswith("lora_B.weight") or k.endswith("lora_A") or k.endswith("lora_B"):
                detected_key_style = "ab"
                break
        if detected_key_style is None:
            for k in keys:
                if k.endswith("lora_down.weight") or k.endswith("lora_up.weight") or k.endswith("lora_down") or k.endswith("lora_up"):
                    detected_key_style = "downup"
                    break
        if detected_key_style is not None:
            metadata["_was_lora_key_style"] = detected_key_style

        downs: Dict[str, torch.Tensor] = {}
        ups: Dict[str, torch.Tensor] = {}
        alphas: Dict[str, float] = {}

        total_keys = len(keys)
        for i, key in enumerate(keys, start=1):
            _progress(progress_cb, "load.keys", i, total_keys, key)
            if is_lora_key(key):
                tensor = f.get_tensor(key).to(device)
                prefix = get_module_prefix(key)
                if "lora_down" in key or "lora_A" in key:
                    downs[prefix] = tensor
                elif "lora_up" in key or "lora_B" in key:
                    ups[prefix] = tensor
            elif key.endswith(".alpha"):
                tensor = f.get_tensor(key)
                value = float(tensor.item())
                try:
                    prefix = get_module_prefix(key.replace(".alpha", ".lora_up"))
                    alphas[prefix] = value
                except Exception:
                    try:
                        prefix = get_module_prefix(key.replace(".alpha", ".lora_down"))
                        alphas[prefix] = value
                    except Exception:
                        pass

        module_keys = set(downs.keys()) | set(ups.keys())

        module_keys_list = list(module_keys)
        total_modules = len(module_keys_list)
        for i, prefix in enumerate(module_keys_list, start=1):
            _progress(progress_cb, "load.modules", i, total_modules, prefix)
            if prefix not in downs or prefix not in ups:
                continue

            down = downs[prefix]
            up = ups[prefix]

            if down.ndim == 4 or up.ndim == 4:
                is_conv = True
                k_h = down.shape[2] if down.ndim == 4 else 1
                k_w = down.shape[3] if down.ndim == 4 else 1
                in_hw = (k_h, k_w)
            else:
                is_conv = False
                in_hw = (1, 1)

            rank = down.shape[0]
            alpha = float(alphas.get(prefix, rank))
            pairs[prefix] = LoraPair(
                down=down,
                up=up,
                alpha=alpha,
                is_conv=is_conv,
                in_hw=in_hw,
            )

    return pairs, metadata


def _is_float_tensor(t: torch.Tensor) -> bool:
    return t.is_floating_point()


def _prefer_bf16_when_mixed(dtypes: List[torch.dtype]) -> torch.dtype:
    """
    If there is any mixed dtype situation and no explicit compute dtype was given,
    bf16 is preferred.
    """
    uniq = list({dt for dt in dtypes})
    if len(uniq) == 1:
        return uniq[0]
    return torch.bfloat16


def _resolve_working_dtype(
    tensors: List[torch.Tensor],
    explicit_compute_dtype: Optional[torch.dtype],
) -> torch.dtype:
    float_dtypes = [t.dtype for t in tensors if _is_float_tensor(t)]
    if not float_dtypes:
        return explicit_compute_dtype or torch.bfloat16
    if explicit_compute_dtype is not None:
        return explicit_compute_dtype
    return _prefer_bf16_when_mixed(float_dtypes)


def _cast_pair(lp: LoraPair, dtype: torch.dtype) -> LoraPair:
    if lp.down.dtype == dtype and lp.up.dtype == dtype:
        return lp
    return LoraPair(
        down=lp.down.to(dtype=dtype),
        up=lp.up.to(dtype=dtype),
        alpha=lp.alpha,
        is_conv=lp.is_conv,
        in_hw=lp.in_hw,
    )


def _safe_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Dot product that guarantees dtype/device alignment and uses fp32 for stability.
    """
    if a.device != b.device:
        b = b.to(a.device)
    a32 = a.to(torch.float32)
    b32 = b.to(torch.float32)
    return a32 @ b32


def compute_delta(lp: LoraPair, compute_dtype: torch.dtype) -> torch.Tensor:
    """Compute the full weight-space delta this LoRA module produces."""
    down = lp.down
    up = lp.up

    if down.dtype != compute_dtype:
        down = down.to(compute_dtype)
    if up.dtype != compute_dtype:
        up = up.to(compute_dtype)

    alpha = lp.alpha
    rank = down.shape[0]
    scale = alpha / max(rank, 1)

    if lp.is_conv:
        r, c_in, k_h, k_w = down.shape
        out_channels = up.shape[0]
        down_flat = down.view(r, c_in * k_h * k_w).to(torch.float32)
        up_flat = up.view(out_channels, r).to(torch.float32)
        delta_flat = (up_flat @ down_flat) * float(scale)
        delta = delta_flat.view(out_channels, c_in, k_h, k_w)
        return delta

    return (up.to(torch.float32) @ down.to(torch.float32)) * float(scale)


def tensor_norm(t: torch.Tensor) -> float:
    return float(torch.norm(t).item())


def choose_auto_rank(singular_values: torch.Tensor, threshold: float) -> int:
    if singular_values.numel() == 0:
        return 0
    s2 = singular_values * singular_values
    total = torch.sum(s2).item()
    if total <= 0.0:
        return int(singular_values.shape[0])
    cumulative = torch.cumsum(s2, dim=0)
    target = threshold * total
    mask = cumulative >= target
    indices = torch.nonzero(mask, as_tuple=False)
    if indices.numel() == 0:
        return int(singular_values.shape[0])
    r = int(indices[0, 0].item() + 1)
    return r


def delta_to_svd_factors(
    delta: torch.Tensor,
    rank: int,
    is_conv: bool,
    auto_rank_threshold: float,
    out_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if is_conv:
        out_channels, c_in, k_h, k_w = delta.shape
        mat = delta.view(out_channels, c_in * k_h * k_w)
    else:
        out_channels, c_in = delta.shape
        mat = delta

    mat32 = mat.to(torch.float32)
    u, s, vh = torch.linalg.svd(mat32, full_matrices=False)

    if rank == -1:
        r = choose_auto_rank(s, auto_rank_threshold)
    elif rank <= 0:
        r = s.shape[0]
    else:
        r = min(rank, s.shape[0])

    if r <= 0:
        if is_conv:
            return (
                torch.zeros((0, 0, 0, 0), dtype=out_dtype, device=delta.device),
                torch.zeros((0, 0), dtype=out_dtype, device=delta.device),
            )
        return (
            torch.zeros((0, 0), dtype=out_dtype, device=delta.device),
            torch.zeros((0, 0), dtype=out_dtype, device=delta.device),
        )

    u_r = u[:, :r]
    s_r = s[:r]
    vh_r = vh[:r, :]

    sqrt_s = torch.sqrt(s_r.clamp_min(0.0))
    up = u_r * sqrt_s.unsqueeze(0)
    down = sqrt_s.unsqueeze(1) * vh_r

    if is_conv:
        down = down.view(r, c_in, k_h, k_w)
        up = up.view(out_channels, r, 1, 1)

    down = down.to(out_dtype)
    up = up.to(out_dtype)

    return down.contiguous(), up.contiguous()


def merge_mode_add(
    loras: List[Tuple[Dict[str, LoraPair], float]],
    include_patterns: List[str],
    exclude_patterns: List[str],
    explicit_compute_dtype: Optional[torch.dtype],
    progress_cb: Optional[ProgressCallback] = None,
) -> Dict[str, LoraPair]:
    all_prefixes = set()
    for pairs, _w in loras:
        all_prefixes.update(pairs.keys())

    merged: Dict[str, LoraPair] = {}

    prefixes_sorted = sorted(all_prefixes)
    total_modules = len(prefixes_sorted)
    for i, prefix in enumerate(
        tqdm(prefixes_sorted, desc="Merging (add)", unit="module", disable=progress_cb is not None),
        start=1,
    ):
        _progress(progress_cb, "merge.add", i, total_modules, prefix)
        if not module_is_included(prefix, include_patterns, exclude_patterns):
            continue

        present: List[Tuple[LoraPair, float]] = []
        for pairs, weight in loras:
            if prefix in pairs:
                present.append((pairs[prefix], weight))

        if not present:
            continue

        is_conv = present[0][0].is_conv
        in_hw = present[0][0].in_hw

        for lp, _w in present[1:]:
            if lp.is_conv != is_conv or lp.in_hw != in_hw:
                raise ValueError(f"Incompatible shapes for module '{prefix}' between LoRAs")

        compute_dtype = _resolve_working_dtype(
            [p[0].down for p in present] + [p[0].up for p in present],
            explicit_compute_dtype,
        )

        downs_list: List[torch.Tensor] = []
        ups_list: List[torch.Tensor] = []
        total_rank = 0

        for lp, w in present:
            lp = _cast_pair(lp, compute_dtype)
            rank_i = lp.down.shape[0]
            scale_i = w * (lp.alpha / max(rank_i, 1))

            downs_list.append(lp.down)
            ups_list.append(lp.up * float(scale_i))
            total_rank += rank_i

        new_down = torch.cat(downs_list, dim=0)
        new_up = torch.cat(ups_list, dim=1)

        new_alpha = float(total_rank)

        merged[prefix] = LoraPair(
            down=new_down,
            up=new_up,
            alpha=new_alpha,
            is_conv=is_conv,
            in_hw=in_hw,
        )

    return merged


def merge_mode_block_mix_weighted(
    loras: List[Tuple[Dict[str, LoraPair], float]],
    method: str,
    rank_value: int,
    preset: str,
    recipe: str,
    concept_mix: float,
    style_mix: float,
    include_patterns: List[str],
    exclude_patterns: List[str],
    auto_rank_threshold: float,
    explicit_compute_dtype: Optional[torch.dtype],
    progress_cb: Optional[ProgressCallback] = None,
) -> Dict[str, LoraPair]:
    if len(loras) != 2:
        raise ValueError("block-mix weighted mode requires exactly two LoRAs (A and B).")

    method_norm = (method or "svd").lower()
    if method_norm not in ("stack", "svd"):
        raise ValueError("block-mix method must be 'stack' or 'svd'.")

    a_pairs, a_weight = loras[0]
    b_pairs, b_weight = loras[1]

    all_prefixes = set(a_pairs.keys()) | set(b_pairs.keys())
    prefixes_sorted = sorted(all_prefixes)

    preset_norm = (preset or "auto").lower().replace("_", "-")
    if preset_norm == "auto":
        preset_norm = _infer_block_mix_preset(prefixes_sorted)

    c_mix = float(concept_mix)
    s_mix = float(style_mix)
    c_mix = 0.0 if c_mix < 0.0 else (1.0 if c_mix > 1.0 else c_mix)
    s_mix = 0.0 if s_mix < 0.0 else (1.0 if s_mix > 1.0 else s_mix)

    merged: Dict[str, LoraPair] = {}
    total_modules = len(prefixes_sorted)
    for i, prefix in enumerate(
        tqdm(prefixes_sorted, desc=f"Merging (block-mix-weighted:{method_norm})", unit="module", disable=progress_cb is not None),
        start=1,
    ):
        _progress(progress_cb, f"merge.block-mix-weighted.{method_norm}", i, total_modules, prefix)
        if not module_is_included(prefix, include_patterns, exclude_patterns):
            continue

        role = _block_mix_role(prefix, preset_norm)
        mix_a = c_mix if role == "concept" else s_mix
        mix_b = 1.0 - mix_a

        present: List[Tuple[LoraPair, float]] = []
        if prefix in a_pairs and mix_a != 0.0:
            present.append((a_pairs[prefix], float(a_weight) * float(mix_a)))
        if prefix in b_pairs and mix_b != 0.0:
            present.append((b_pairs[prefix], float(b_weight) * float(mix_b)))

        if not present:
            continue

        ref_lp = present[0][0]
        is_conv = ref_lp.is_conv
        in_hw = ref_lp.in_hw
        for lp, _w in present[1:]:
            if lp.is_conv != is_conv or lp.in_hw != in_hw:
                raise ValueError(f"Incompatible shapes for module '{prefix}' in block-mix weighted mode")

        if method_norm == "stack":
            compute_dtype = _resolve_working_dtype(
                [t for lp, _w in present for t in (lp.down, lp.up)],
                explicit_compute_dtype,
            )

            downs_list: List[torch.Tensor] = []
            ups_list: List[torch.Tensor] = []
            total_rank = 0
            for lp, w in present:
                lp = _cast_pair(lp, compute_dtype)
                rank_i = lp.down.shape[0]
                scale_i = float(w) * (lp.alpha / max(rank_i, 1))
                downs_list.append(lp.down)
                ups_list.append(lp.up * float(scale_i))
                total_rank += int(rank_i)

            new_down = torch.cat(downs_list, dim=0)
            new_up = torch.cat(ups_list, dim=1)
            new_alpha = float(total_rank)
            merged[prefix] = LoraPair(
                down=new_down,
                up=new_up,
                alpha=new_alpha,
                is_conv=is_conv,
                in_hw=in_hw,
            )
            continue

        compute_dtype = _resolve_working_dtype(
            [t for lp, _w in present for t in (lp.down, lp.up)],
            explicit_compute_dtype,
        )

        merged_delta = None
        for lp, w in present:
            d = compute_delta(lp, compute_dtype) * float(w)
            merged_delta = d if merged_delta is None else (merged_delta + d)

        if merged_delta is None:
            continue

        svd_rank = rank_value
        if svd_rank == 0:
            svd_rank = -1

        new_down, new_up = delta_to_svd_factors(
            merged_delta,
            rank=svd_rank,
            is_conv=is_conv,
            auto_rank_threshold=auto_rank_threshold,
            out_dtype=compute_dtype,
        )
        new_alpha = float(new_down.shape[0])
        merged[prefix] = LoraPair(
            down=new_down,
            up=new_up,
            alpha=new_alpha,
            is_conv=is_conv,
            in_hw=in_hw,
        )

    return merged


def block_mix_routing_report(
    a_pairs: Dict[str, LoraPair],
    b_pairs: Dict[str, LoraPair],
    preset: str,
    recipe: str,
    include_patterns: List[str],
    exclude_patterns: List[str],
) -> Dict[str, Any]:
    all_prefixes = set(a_pairs.keys()) | set(b_pairs.keys())
    prefixes_sorted = sorted(all_prefixes)

    preset_norm = (preset or "auto").lower().replace("_", "-")
    if preset_norm == "auto":
        preset_norm = _infer_block_mix_preset(prefixes_sorted)

    recipe_norm = (recipe or "").lower().replace("-", "_")

    routed_a = 0
    routed_b = 0
    fallback_to_a = 0
    fallback_to_b = 0
    missing_both = 0
    skipped_filter = 0
    requested_a = 0
    requested_b = 0

    concept = 0
    style = 0

    for prefix in prefixes_sorted:
        if not module_is_included(prefix, include_patterns, exclude_patterns):
            skipped_filter += 1
            continue

        role = _block_mix_role(prefix, preset_norm)
        if role == "style":
            style += 1
        else:
            concept += 1

        idx = _block_mix_pick_lora_index(prefix, preset_norm, recipe_norm)
        if idx == 0:
            requested_a += 1
            if prefix in a_pairs:
                routed_a += 1
            elif prefix in b_pairs:
                routed_b += 1
                fallback_to_b += 1
            else:
                missing_both += 1
        else:
            requested_b += 1
            if prefix in b_pairs:
                routed_b += 1
            elif prefix in a_pairs:
                routed_a += 1
                fallback_to_a += 1
            else:
                missing_both += 1

    included_total = (len(prefixes_sorted) - skipped_filter)

    return {
        "preset": preset_norm,
        "recipe": recipe_norm,
        "modules_total": len(prefixes_sorted),
        "modules_included": included_total,
        "modules_skipped_filter": skipped_filter,
        "requested_a": requested_a,
        "requested_b": requested_b,
        "routed_a": routed_a,
        "routed_b": routed_b,
        "fallback_to_a": fallback_to_a,
        "fallback_to_b": fallback_to_b,
        "missing_both": missing_both,
        "role_concept": concept,
        "role_style": style,
    }


def _infer_block_mix_preset(prefixes: List[str]) -> str:
    for p in prefixes:
        if p.startswith("diffusion_model.layers."):
            return "zimg-turbo"
        if p.startswith("diffusion_model.blocks.") or p.startswith("diffusion_model.token_refiner."):
            return "dit"
        if p.startswith("lora_unet_double_blocks_") or p.startswith("lora_unet_single_blocks_"):
            return "flux"
        if p.startswith("lora_unet_blocks_"):
            return "wan"
        if p.startswith("transformer_blocks."):
            return "qwen"
        if p.startswith("lora_te1_") or p.startswith("lora_te2_"):
            return "sdxl"
        if p.startswith("lora_unet_"):
            return "sd"
    return "generic"


def _block_mix_role(prefix: str, preset: str) -> str:
    p = prefix.lower()

    if preset == "zimg-turbo":
        if ".attention." in p:
            return "concept"
        if ".feed_forward." in p or ".adaln_modulation." in p:
            return "style"
        return "concept"

    if preset == "flux":
        if "_attn_" in p:
            return "concept"
        if "_mlp_" in p or "_mod_" in p or "_mod_lin" in p:
            return "style"
        return "concept"

    if preset == "wan":
        if "_attn_" in p:
            return "concept"
        if "_ffn_" in p or "_mlp_" in p:
            return "style"
        return "concept"

    if preset == "qwen":
        if ".attn." in p:
            return "concept"
        if ".mlp." in p or "_mlp" in p:
            return "style"
        return "concept"

    if preset == "dit":
        if "adaln" in p or "modulation" in p or ".norm" in p:
            return "style"
        if ".attn." in p:
            return "concept"
        if ".mlp." in p or ".ffn." in p or ".feed_forward." in p:
            return "style"
        return "concept"

    attn_markers = (
        "attn",
        "to_q",
        "to_k",
        "to_v",
        "to_out",
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "proj",
    )
    if any(m in p for m in attn_markers):
        return "concept"

    style_markers = ("mlp", "ff", "feed_forward", "res", "conv", "adaln", "norm")
    if any(m in p for m in style_markers):
        return "style"

    return "concept"


def _block_mix_pick_lora_index(prefix: str, preset: str, recipe: str) -> int:
    r = (recipe or "").lower().replace("-", "_")

    if r in ("all_a", "only_a"):
        return 0
    if r in ("all_b", "only_b"):
        return 1

    if preset == "flux" and r in ("img_a_txt_b", "img_b_txt_a"):
        p = prefix.lower()
        if "_img_" in p:
            return 0 if r == "img_a_txt_b" else 1
        if "_txt_" in p:
            return 1 if r == "img_a_txt_b" else 0
        return 0

    role = _block_mix_role(prefix, preset)
    if r in ("concept_a_style_b", "concept_from_a_style_from_b", "attn_a_ffn_b"):
        return 0 if role == "concept" else 1
    if r in ("concept_b_style_a", "concept_from_b_style_from_a", "attn_b_ffn_a"):
        return 1 if role == "concept" else 0

    return 0


def merge_mode_block_mix(
    loras: List[Tuple[Dict[str, LoraPair], float]],
    method: str,
    rank_value: int,
    preset: str,
    recipe: str,
    include_patterns: List[str],
    exclude_patterns: List[str],
    auto_rank_threshold: float,
    explicit_compute_dtype: Optional[torch.dtype],
    progress_cb: Optional[ProgressCallback] = None,
) -> Dict[str, LoraPair]:
    if len(loras) != 2:
        raise ValueError("block-mix mode requires exactly two LoRAs (A and B).")

    method_norm = (method or "svd").lower()
    if method_norm not in ("stack", "svd"):
        raise ValueError("block-mix method must be 'stack' or 'svd'.")

    a_pairs, a_weight = loras[0]
    b_pairs, b_weight = loras[1]

    all_prefixes = set(a_pairs.keys()) | set(b_pairs.keys())
    prefixes_sorted = sorted(all_prefixes)

    preset_norm = (preset or "auto").lower().replace("_", "-")
    if preset_norm == "auto":
        preset_norm = _infer_block_mix_preset(prefixes_sorted)

    merged: Dict[str, LoraPair] = {}
    total_modules = len(prefixes_sorted)
    for i, prefix in enumerate(
        tqdm(prefixes_sorted, desc=f"Merging (block-mix:{method_norm})", unit="module", disable=progress_cb is not None),
        start=1,
    ):
        _progress(progress_cb, f"merge.block-mix.{method_norm}", i, total_modules, prefix)
        if not module_is_included(prefix, include_patterns, exclude_patterns):
            continue

        idx = _block_mix_pick_lora_index(prefix, preset_norm, recipe)

        present: List[Tuple[LoraPair, float]] = []
        if idx == 0:
            if prefix in a_pairs:
                present.append((a_pairs[prefix], float(a_weight)))
            elif prefix in b_pairs:
                present.append((b_pairs[prefix], float(b_weight)))
        else:
            if prefix in b_pairs:
                present.append((b_pairs[prefix], float(b_weight)))
            elif prefix in a_pairs:
                present.append((a_pairs[prefix], float(a_weight)))

        if not present:
            continue

        ref_lp = present[0][0]
        is_conv = ref_lp.is_conv
        in_hw = ref_lp.in_hw
        for lp, _w in present[1:]:
            if lp.is_conv != is_conv or lp.in_hw != in_hw:
                raise ValueError(f"Incompatible shapes for module '{prefix}' in block-mix mode")

        if method_norm == "stack":
            compute_dtype = _resolve_working_dtype(
                [t for lp, _w in present for t in (lp.down, lp.up)],
                explicit_compute_dtype,
            )

            downs_list: List[torch.Tensor] = []
            ups_list: List[torch.Tensor] = []
            total_rank = 0
            for lp, w in present:
                lp = _cast_pair(lp, compute_dtype)
                rank_i = lp.down.shape[0]
                scale_i = float(w) * (lp.alpha / max(rank_i, 1))
                downs_list.append(lp.down)
                ups_list.append(lp.up * float(scale_i))
                total_rank += int(rank_i)

            new_down = torch.cat(downs_list, dim=0)
            new_up = torch.cat(ups_list, dim=1)
            new_alpha = float(total_rank)
            merged[prefix] = LoraPair(
                down=new_down,
                up=new_up,
                alpha=new_alpha,
                is_conv=is_conv,
                in_hw=in_hw,
            )
            continue

        compute_dtype = _resolve_working_dtype(
            [t for lp, _w in present for t in (lp.down, lp.up)],
            explicit_compute_dtype,
        )
        merged_delta = None
        for lp, w in present:
            d = compute_delta(lp, compute_dtype) * float(w)
            merged_delta = d if merged_delta is None else (merged_delta + d)

        if merged_delta is None:
            continue

        svd_rank = rank_value
        if svd_rank == 0:
            svd_rank = -1

        new_down, new_up = delta_to_svd_factors(
            merged_delta,
            rank=svd_rank,
            is_conv=is_conv,
            auto_rank_threshold=auto_rank_threshold,
            out_dtype=compute_dtype,
        )
        new_alpha = float(new_down.shape[0])
        merged[prefix] = LoraPair(
            down=new_down,
            up=new_up,
            alpha=new_alpha,
            is_conv=is_conv,
            in_hw=in_hw,
        )

    return merged


def merge_mode_add_diff(
    loras: List[Tuple[Dict[str, LoraPair], float]],
    rank_value: int,
    include_patterns: List[str],
    exclude_patterns: List[str],
    auto_rank_threshold: float,
    explicit_compute_dtype: Optional[torch.dtype],
    progress_cb: Optional[ProgressCallback] = None,
) -> Dict[str, LoraPair]:
    if len(loras) < 2:
        raise ValueError("add-diff mode requires at least two LoRAs (base + one or more others).")

    base_pairs, base_weight = loras[0]

    all_prefixes = set()
    for pairs, _w in loras:
        all_prefixes.update(pairs.keys())

    merged: Dict[str, LoraPair] = {}

    prefixes_sorted = sorted(all_prefixes)
    total_modules = len(prefixes_sorted)
    for i, prefix in enumerate(
        tqdm(prefixes_sorted, desc="Merging (add-diff)", unit="module", disable=progress_cb is not None),
        start=1,
    ):
        _progress(progress_cb, "merge.add-diff", i, total_modules, prefix)
        if not module_is_included(prefix, include_patterns, exclude_patterns):
            continue

        base_lp = base_pairs.get(prefix, None)

        others: List[Tuple[LoraPair, float]] = []
        for pairs, w in loras[1:]:
            if prefix in pairs:
                others.append((pairs[prefix], w))

        if base_lp is None and not others:
            continue

        ref_lp = base_lp if base_lp is not None else others[0][0]
        is_conv = ref_lp.is_conv
        in_hw = ref_lp.in_hw

        if base_lp is not None and (base_lp.is_conv != is_conv or base_lp.in_hw != in_hw):
            raise ValueError(f"Incompatible shapes for module '{prefix}' in add-diff (base).")
        for lp, _w in others:
            if lp.is_conv != is_conv or lp.in_hw != in_hw:
                raise ValueError(f"Incompatible shapes for module '{prefix}' in add-diff (other).")

        compute_dtype = _resolve_working_dtype(
            ([base_lp.down, base_lp.up] if base_lp is not None else [])
            + [t for lp, _w in others for t in (lp.down, lp.up)],
            explicit_compute_dtype,
        )

        if base_lp is not None:
            delta_base_raw = compute_delta(base_lp, compute_dtype)
            delta_base = delta_base_raw * float(base_weight)
        else:
            delta_base_raw = None
            delta_base = torch.zeros_like(compute_delta(ref_lp, compute_dtype))

        merged_delta = delta_base.clone()

        for lp_other, weight_other in others:
            delta_other = compute_delta(lp_other, compute_dtype)
            if delta_base_raw is None:
                merged_delta = merged_delta + float(weight_other) * delta_other
            else:
                merged_delta = merged_delta + float(weight_other) * (delta_other - delta_base_raw)

        svd_rank = rank_value
        if svd_rank == 0:
            svd_rank = -1

        new_down, new_up = delta_to_svd_factors(
            merged_delta,
            rank=svd_rank,
            is_conv=is_conv,
            auto_rank_threshold=auto_rank_threshold,
            out_dtype=compute_dtype,
        )
        new_alpha = float(new_down.shape[0])

        merged[prefix] = LoraPair(
            down=new_down,
            up=new_up,
            alpha=new_alpha,
            is_conv=is_conv,
            in_hw=in_hw,
        )

    return merged


def merge_mode_svd(
    loras: List[Tuple[Dict[str, LoraPair], float]],
    rank_value: int,
    preserve_norm: bool,
    cap_mult: Optional[float],
    include_patterns: List[str],
    exclude_patterns: List[str],
    auto_rank_threshold: float,
    explicit_compute_dtype: Optional[torch.dtype],
    progress_cb: Optional[ProgressCallback] = None,
) -> Dict[str, LoraPair]:
    all_prefixes = set()
    for pairs, _w in loras:
        all_prefixes.update(pairs.keys())

    merged: Dict[str, LoraPair] = {}

    prefixes_sorted = sorted(all_prefixes)
    total_modules = len(prefixes_sorted)
    for i, prefix in enumerate(
        tqdm(prefixes_sorted, desc="Merging (svd)", unit="module", disable=progress_cb is not None),
        start=1,
    ):
        _progress(progress_cb, "merge.svd", i, total_modules, prefix)
        if not module_is_included(prefix, include_patterns, exclude_patterns):
            continue

        present: List[Tuple[LoraPair, float]] = []
        for pairs, weight in loras:
            if prefix in pairs:
                present.append((pairs[prefix], weight))

        if not present:
            continue

        ref_lp = present[0][0]
        is_conv = ref_lp.is_conv
        in_hw = ref_lp.in_hw

        for lp, _w in present[1:]:
            if lp.is_conv != is_conv or lp.in_hw != in_hw:
                raise ValueError(f"Incompatible shapes for module '{prefix}' between LoRAs")

        compute_dtype = _resolve_working_dtype(
            [t for lp, _w in present for t in (lp.down, lp.up)],
            explicit_compute_dtype,
        )

        deltas: List[torch.Tensor] = []
        norms: List[float] = []

        for lp, w in present:
            d = compute_delta(lp, compute_dtype)
            deltas.append(d * float(w))
            norms.append(tensor_norm(d))

        base_norm = sum(norms) / max(len(norms), 1)
        merged_delta = torch.zeros_like(deltas[0])
        for d in deltas:
            merged_delta = merged_delta + d

        if preserve_norm:
            merged_norm = tensor_norm(merged_delta)
            if merged_norm > 1e-8 and base_norm > 1e-8:
                merged_delta = merged_delta * (base_norm / merged_norm)

        if cap_mult is not None and cap_mult > 0.0 and base_norm > 0.0:
            merged_norm = tensor_norm(merged_delta)
            max_norm = base_norm * cap_mult
            if merged_norm > max_norm and merged_norm > 0.0:
                merged_delta = merged_delta * (max_norm / merged_norm)

        svd_rank = rank_value
        if svd_rank == 0:
            svd_rank = -1

        new_down, new_up = delta_to_svd_factors(
            merged_delta,
            rank=svd_rank,
            is_conv=is_conv,
            auto_rank_threshold=auto_rank_threshold,
            out_dtype=compute_dtype,
        )
        new_alpha = float(new_down.shape[0])

        merged[prefix] = LoraPair(
            down=new_down,
            up=new_up,
            alpha=new_alpha,
            is_conv=is_conv,
            in_hw=in_hw,
        )

    return merged


def merge_mode_add_orth(
    loras: List[Tuple[Dict[str, LoraPair], float]],
    rank_value: int,
    include_patterns: List[str],
    exclude_patterns: List[str],
    auto_rank_threshold: float,
    explicit_compute_dtype: Optional[torch.dtype],
    progress_cb: Optional[ProgressCallback] = None,
) -> Dict[str, LoraPair]:
    if len(loras) < 2:
        raise ValueError("add-orth mode requires at least two LoRAs (base + one or more others).")

    base_pairs, base_weight = loras[0]

    all_prefixes = set()
    for pairs, _w in loras:
        all_prefixes.update(pairs.keys())

    merged: Dict[str, LoraPair] = {}

    prefixes_sorted = sorted(all_prefixes)
    total_modules = len(prefixes_sorted)
    for i, prefix in enumerate(
        tqdm(prefixes_sorted, desc="Merging (add-orth)", unit="module", disable=progress_cb is not None),
        start=1,
    ):
        _progress(progress_cb, "merge.add-orth", i, total_modules, prefix)
        if not module_is_included(prefix, include_patterns, exclude_patterns):
            continue

        base_lp = base_pairs.get(prefix, None)
        present_others: List[Tuple[LoraPair, float]] = []

        for pairs, weight in loras[1:]:
            if prefix in pairs:
                present_others.append((pairs[prefix], weight))

        if base_lp is None and not present_others:
            continue

        if base_lp is None:
            temp_loras = [(pairs, weight) for (pairs, weight) in loras if prefix in pairs]
            temp_pairs_list = []
            for pairs, weight in temp_loras:
                temp_pairs_list.append(({prefix: pairs[prefix]}, weight))
            return_fallback = merge_mode_add(temp_pairs_list, include_patterns, exclude_patterns, explicit_compute_dtype)
            merged.update(return_fallback)
            continue

        is_conv = base_lp.is_conv
        in_hw = base_lp.in_hw

        for lp, _w in present_others:
            if lp.is_conv != is_conv or lp.in_hw != in_hw:
                raise ValueError(f"Incompatible shapes for module '{prefix}' in add-orth mode")

        compute_dtype = _resolve_working_dtype(
            [base_lp.down, base_lp.up] + [t for lp, _w in present_others for t in (lp.down, lp.up)],
            explicit_compute_dtype,
        )

        delta_base = compute_delta(base_lp, compute_dtype) * float(base_weight)
        base_flat = delta_base.reshape(-1)
        base_norm_sq = float(_safe_dot(base_flat, base_flat).item())

        merged_delta = delta_base.clone()
        eps = 1e-12

        for lp_other, weight_other in present_others:
            delta_other = compute_delta(lp_other, compute_dtype) * float(weight_other)
            other_flat = delta_other.reshape(-1)

            if base_norm_sq < eps:
                merged_delta = merged_delta + delta_other
                continue

            coeff = float(_safe_dot(other_flat, base_flat).item()) / (base_norm_sq + eps)
            orth_flat = other_flat - coeff * base_flat
            orth_delta = orth_flat.view_as(delta_other)
            merged_delta = merged_delta + orth_delta

        svd_rank = rank_value
        if svd_rank == 0:
            svd_rank = -1

        new_down, new_up = delta_to_svd_factors(
            merged_delta,
            rank=svd_rank,
            is_conv=is_conv,
            auto_rank_threshold=auto_rank_threshold,
            out_dtype=compute_dtype,
        )
        new_alpha = float(new_down.shape[0])

        merged[prefix] = LoraPair(
            down=new_down,
            up=new_up,
            alpha=new_alpha,
            is_conv=is_conv,
            in_hw=in_hw,
        )

    return merged


def merge_mode_diff_export(
    loras: List[Tuple[Dict[str, LoraPair], float]],
    rank_value: int,
    include_patterns: List[str],
    exclude_patterns: List[str],
    auto_rank_threshold: float,
    explicit_compute_dtype: Optional[torch.dtype],
    progress_cb: Optional[ProgressCallback] = None,
) -> Dict[str, LoraPair]:
    if len(loras) < 2:
        raise ValueError("diff-export mode requires at least two LoRAs (base and capability).")

    base_pairs, base_weight = loras[0]
    cap_pairs, cap_weight = loras[1]

    all_prefixes = set(base_pairs.keys()) | set(cap_pairs.keys())
    merged: Dict[str, LoraPair] = {}

    prefixes_sorted = sorted(all_prefixes)
    total_modules = len(prefixes_sorted)
    for i, prefix in enumerate(
        tqdm(prefixes_sorted, desc="Merging (diff-export)", unit="module", disable=progress_cb is not None),
        start=1,
    ):
        _progress(progress_cb, "merge.diff-export", i, total_modules, prefix)
        if not module_is_included(prefix, include_patterns, exclude_patterns):
            continue

        base_lp = base_pairs.get(prefix, None)
        cap_lp = cap_pairs.get(prefix, None)

        if base_lp is None and cap_lp is None:
            continue

        ref_lp = cap_lp if cap_lp is not None else base_lp
        is_conv = ref_lp.is_conv
        in_hw = ref_lp.in_hw

        if base_lp is not None and (base_lp.is_conv != is_conv or base_lp.in_hw != in_hw):
            raise ValueError(f"Incompatible shapes for module '{prefix}' in base LoRA (diff-export).")
        if cap_lp is not None and (cap_lp.is_conv != is_conv or cap_lp.in_hw != in_hw):
            raise ValueError(f"Incompatible shapes for module '{prefix}' in capability LoRA (diff-export).")

        compute_dtype = _resolve_working_dtype(
            ([base_lp.down, base_lp.up] if base_lp is not None else [])
            + ([cap_lp.down, cap_lp.up] if cap_lp is not None else []),
            explicit_compute_dtype,
        )

        if base_lp is not None:
            delta_base = compute_delta(base_lp, compute_dtype) * float(base_weight)
        else:
            delta_base = torch.zeros_like(compute_delta(ref_lp, compute_dtype))

        if cap_lp is not None:
            delta_cap = compute_delta(cap_lp, compute_dtype) * float(cap_weight)
        else:
            delta_cap = torch.zeros_like(compute_delta(ref_lp, compute_dtype))

        delta_diff = delta_cap - delta_base

        if torch.allclose(delta_diff, torch.zeros_like(delta_diff)):
            continue

        svd_rank = rank_value
        if svd_rank == 0:
            svd_rank = -1

        new_down, new_up = delta_to_svd_factors(
            delta_diff,
            rank=svd_rank,
            is_conv=is_conv,
            auto_rank_threshold=auto_rank_threshold,
            out_dtype=compute_dtype,
        )
        new_alpha = float(new_down.shape[0])

        merged[prefix] = LoraPair(
            down=new_down,
            up=new_up,
            alpha=new_alpha,
            is_conv=is_conv,
            in_hw=in_hw,
        )

    return merged


def merge_mode_moe(
    loras: List[Tuple[Dict[str, LoraPair], float]],
    rank_value: int,
    moe_temperature: float,
    moe_hard: bool,
    include_patterns: List[str],
    exclude_patterns: List[str],
    auto_rank_threshold: float,
    explicit_compute_dtype: Optional[torch.dtype],
    progress_cb: Optional[ProgressCallback] = None,
) -> Dict[str, LoraPair]:
    if len(loras) < 1:
        raise ValueError("moe mode requires at least one LoRA.")

    base_pairs, base_weight = loras[0]

    all_prefixes = set()
    for pairs, _w in loras:
        all_prefixes.update(pairs.keys())

    merged: Dict[str, LoraPair] = {}

    prefixes_sorted = sorted(all_prefixes)
    total_modules = len(prefixes_sorted)
    for i, prefix in enumerate(
        tqdm(prefixes_sorted, desc="Merging (moe)", unit="module", disable=progress_cb is not None),
        start=1,
    ):
        _progress(progress_cb, "merge.moe", i, total_modules, prefix)
        if not module_is_included(prefix, include_patterns, exclude_patterns):
            continue

        base_lp = base_pairs.get(prefix, None)
        experts: List[Tuple[LoraPair, float]] = []

        for pairs, weight in loras[1:]:
            if prefix in pairs:
                experts.append((pairs[prefix], weight))

        if base_lp is None and not experts:
            continue

        ref_lp = base_lp if base_lp is not None else (experts[0][0] if experts else None)
        if ref_lp is None:
            continue

        is_conv = ref_lp.is_conv
        in_hw = ref_lp.in_hw

        if base_lp is not None and (base_lp.is_conv != is_conv or base_lp.in_hw != in_hw):
            raise ValueError(f"Incompatible shapes for module '{prefix}' in base LoRA (moe).")
        for lp, _w in experts:
            if lp.is_conv != is_conv or lp.in_hw != in_hw:
                raise ValueError(f"Incompatible shapes for module '{prefix}' in moe experts.")

        compute_dtype = _resolve_working_dtype(
            ([base_lp.down, base_lp.up] if base_lp is not None else [])
            + [t for lp, _w in experts for t in (lp.down, lp.up)],
            explicit_compute_dtype,
        )

        if base_lp is not None:
            delta_base = compute_delta(base_lp, compute_dtype) * float(base_weight)
        else:
            delta_base = torch.zeros_like(compute_delta(ref_lp, compute_dtype))

        if not experts:
            merged_delta = delta_base
        else:
            scores: List[float] = []
            deltas_expert: List[torch.Tensor] = []

            for lp, weight in experts:
                d = compute_delta(lp, compute_dtype) * float(weight)
                deltas_expert.append(d)
                scores.append(tensor_norm(d))

            scores_tensor = torch.tensor(scores, dtype=torch.float32, device=delta_base.device)

            if moe_hard:
                best_index = int(torch.argmax(scores_tensor).item())
                gate = torch.zeros_like(scores_tensor)
                gate[best_index] = 1.0
            else:
                temp = max(moe_temperature, 1e-6)
                scaled = scores_tensor / temp
                gate = torch.softmax(scaled, dim=0)

            merged_delta = delta_base.clone()
            for g, d in zip(gate, deltas_expert):
                merged_delta = merged_delta + float(g.item()) * d

        svd_rank = rank_value
        if svd_rank == 0:
            svd_rank = -1

        new_down, new_up = delta_to_svd_factors(
            merged_delta,
            rank=svd_rank,
            is_conv=is_conv,
            auto_rank_threshold=auto_rank_threshold,
            out_dtype=compute_dtype,
        )
        new_alpha = float(new_down.shape[0])

        merged[prefix] = LoraPair(
            down=new_down,
            up=new_up,
            alpha=new_alpha,
            is_conv=is_conv,
            in_hw=in_hw,
        )

    return merged


def merge_mode_obfuscate(
    loras: List[Tuple[Dict[str, LoraPair], float]],
    include_patterns: List[str],
    exclude_patterns: List[str],
    explicit_compute_dtype: Optional[torch.dtype],
    progress_cb: Optional[ProgressCallback] = None,
) -> Dict[str, LoraPair]:
    if len(loras) < 1:
        raise ValueError("obfuscate mode requires at least one LoRA.")

    all_prefixes = set()
    for pairs, _w in loras:
        all_prefixes.update(pairs.keys())

    merged: Dict[str, LoraPair] = {}

    prefixes_sorted = sorted(all_prefixes)
    total_modules = len(prefixes_sorted)
    for i, prefix in enumerate(
        tqdm(prefixes_sorted, desc="Merging (obfuscate)", unit="module", disable=progress_cb is not None),
        start=1,
    ):
        _progress(progress_cb, "merge.obfuscate", i, total_modules, prefix)
        if not module_is_included(prefix, include_patterns, exclude_patterns):
            continue

        present: List[Tuple[LoraPair, float]] = []
        for pairs, weight in loras:
            if prefix in pairs:
                present.append((pairs[prefix], weight))

        if not present:
            continue

        ref_lp = present[0][0]
        is_conv = ref_lp.is_conv
        in_hw = ref_lp.in_hw
        device = ref_lp.down.device

        for lp, _w in present[1:]:
            if lp.is_conv != is_conv or lp.in_hw != in_hw:
                raise ValueError(f"Incompatible shapes for module '{prefix}' between LoRAs in obfuscate mode")

        compute_dtype = _resolve_working_dtype(
            [t for lp, _w in present for t in (lp.down, lp.up)],
            explicit_compute_dtype,
        )

        down_blocks: List[torch.Tensor] = []
        up_blocks_flat: List[torch.Tensor] = []
        total_rank = 0
        c_in = None
        k_h = None
        k_w = None

        for lp, w in present:
            lp = _cast_pair(lp, compute_dtype)
            r_i = lp.down.shape[0]
            if r_i == 0:
                continue

            scale_i = float(w) * (lp.alpha / max(r_i, 1))

            if is_conv:
                down_i = lp.down.to(torch.float32)
                up_i = lp.up.to(torch.float32)
                c_in = down_i.shape[1]
                k_h = down_i.shape[2]
                k_w = down_i.shape[3]

                up_i_flat = up_i.view(up_i.shape[0], r_i)
                up_i_flat = up_i_flat * scale_i

                down_blocks.append(down_i)
                up_blocks_flat.append(up_i_flat)
            else:
                down_i = lp.down.to(torch.float32)
                up_i = (lp.up.to(torch.float32) * scale_i)

                down_blocks.append(down_i)
                up_blocks_flat.append(up_i)

            total_rank += r_i

        if total_rank == 0:
            continue

        if is_conv:
            down_cat = torch.cat(down_blocks, dim=0)
            up_cat_flat = torch.cat(up_blocks_flat, dim=1)
            r_total = down_cat.shape[0]

            rand = torch.randn((r_total, r_total), device=device, dtype=torch.float32)
            q, _ = torch.linalg.qr(rand)

            down_flat = down_cat.view(r_total, -1)
            down_flat = q @ down_flat
            down_new32 = down_flat.view(r_total, c_in, k_h, k_w)

            up_cat_flat = up_cat_flat @ q.T
            up_new32 = up_cat_flat.view(up_cat_flat.shape[0], r_total, 1, 1)
        else:
            down_cat = torch.cat(down_blocks, dim=0)
            up_cat = torch.cat(up_blocks_flat, dim=1)
            r_total = down_cat.shape[0]

            rand = torch.randn((r_total, r_total), device=device, dtype=torch.float32)
            q, _ = torch.linalg.qr(rand)

            down_new32 = q @ down_cat
            up_new32 = up_cat @ q.T

        down_new = down_new32.to(compute_dtype)
        up_new = up_new32.to(compute_dtype)
        alpha_new = float(r_total)

        merged[prefix] = LoraPair(
            down=down_new.contiguous(),
            up=up_new.contiguous(),
            alpha=alpha_new,
            is_conv=is_conv,
            in_hw=in_hw,
        )

    return merged


def merge_mode_rebase(
    lora: Tuple[Dict[str, LoraPair], float],
    rank_value: int,
    include_patterns: List[str],
    exclude_patterns: List[str],
    auto_rank_threshold: float,
    explicit_compute_dtype: Optional[torch.dtype],
    progress_cb: Optional[ProgressCallback] = None,
) -> Dict[str, LoraPair]:
    pairs, weight = lora
    merged: Dict[str, LoraPair] = {}

    prefixes_sorted = sorted(pairs.keys())
    total_modules = len(prefixes_sorted)
    for i, prefix in enumerate(
        tqdm(prefixes_sorted, desc="Rebasing (rebase)", unit="module", disable=progress_cb is not None),
        start=1,
    ):
        _progress(progress_cb, "merge.rebase", i, total_modules, prefix)
        if not module_is_included(prefix, include_patterns, exclude_patterns):
            continue

        lp = pairs[prefix]
        compute_dtype = _resolve_working_dtype([lp.down, lp.up], explicit_compute_dtype)

        delta = compute_delta(lp, compute_dtype) * float(weight)
        original_rank = lp.down.shape[0]

        target_rank = -1 if rank_value == 0 else rank_value

        new_down, new_up = delta_to_svd_factors(
            delta,
            rank=target_rank,
            is_conv=lp.is_conv,
            auto_rank_threshold=auto_rank_threshold,
            out_dtype=compute_dtype,
        )

        new_rank = new_down.shape[0]
        if new_rank > original_rank:
            if lp.is_conv:
                new_down = new_down[:original_rank].contiguous()
                new_up = new_up[:, :original_rank, ...].contiguous()
            else:
                new_down = new_down[:original_rank].contiguous()
                new_up = new_up[:, :original_rank].contiguous()

        new_alpha = float(new_down.shape[0])

        merged[prefix] = LoraPair(
            down=new_down,
            up=new_up,
            alpha=new_alpha,
            is_conv=lp.is_conv,
            in_hw=lp.in_hw,
        )

    return merged


def build_state_dict(
    merged: Dict[str, LoraPair],
    metadata_ref: Dict[str, str],
    dtype: torch.dtype,
    device: torch.device,
    progress_cb: Optional[ProgressCallback] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    tensors: Dict[str, torch.Tensor] = {}

    key_style = _get_save_key_style(metadata_ref)

    items = list(merged.items())
    total_modules = len(items)
    for i, (prefix, lp) in enumerate(items, start=1):
        _progress(progress_cb, "build.state_dict", i, total_modules, prefix)
        down = lp.down.to(dtype=dtype, device=device)
        up = lp.up.to(dtype=dtype, device=device)
        alpha_value = torch.tensor(lp.alpha, dtype=torch.float32, device=device)

        tensors[_down_key(prefix, key_style)] = down
        tensors[_up_key(prefix, key_style)] = up
        tensors[get_alpha_key(prefix)] = alpha_value

    meta = dict(metadata_ref or {})
    meta.setdefault("format", "kohya-lora")

    return tensors, meta


def summarize_pairs(
    pairs: Dict[str, LoraPair],
    weight: float,
    explicit_compute_dtype: Optional[torch.dtype],
    include_patterns: List[str],
    exclude_patterns: List[str],
    max_modules: int = 25,
) -> Dict[str, float]:
    prefixes = [p for p in sorted(pairs.keys()) if module_is_included(p, include_patterns, exclude_patterns)]
    if not prefixes:
        return {
            "modules": 0.0,
            "avg_rank": 0.0,
            "avg_alpha": 0.0,
            "avg_scale": 0.0,
            "mean_delta_norm": 0.0,
            "max_delta_norm": 0.0,
        }

    norms: List[float] = []
    ranks: List[float] = []
    alphas: List[float] = []
    scales: List[float] = []

    for prefix in prefixes[: max_modules]:
        lp = pairs[prefix]
        compute_dtype = _resolve_working_dtype([lp.down, lp.up], explicit_compute_dtype)
        delta = compute_delta(lp, compute_dtype)
        norms.append(tensor_norm(delta) * float(abs(weight)))
        r = float(lp.down.shape[0])
        a = float(lp.alpha)
        ranks.append(r)
        alphas.append(a)
        scales.append(float(abs(weight)) * (a / max(r, 1.0)))

    return {
        "modules": float(len(prefixes)),
        "avg_rank": float(sum(ranks) / max(len(ranks), 1)),
        "avg_alpha": float(sum(alphas) / max(len(alphas), 1)),
        "avg_scale": float(sum(scales) / max(len(scales), 1)),
        "mean_delta_norm": float(sum(norms) / max(len(norms), 1)),
        "max_delta_norm": float(max(norms) if norms else 0.0),
    }
