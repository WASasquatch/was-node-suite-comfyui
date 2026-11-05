"""The advanced settings a merge runs with, and the defaults it runs with without them.

The Power LoRA Merger's optional options socket replaces only the settings it carries.
Every value is coerced to its declared type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .rows import to_bool

__all__ = ["LOKR_LAYERS", "MergeSettings"]

#: How the modes that recompress write a layer a LoKr is on.
LOKR_LAYERS = ("exact", "lokr", "lora")


# The options node emits a plain dictionary, so the socket may also carry one built
# elsewhere; each value is coerced to its field type instead of being trusted.
def _int(values: Mapping[str, Any], key: str, default: int) -> int:
    try:
        return int(values.get(key, default))
    except (TypeError, ValueError):
        return int(default)


def _float(values: Mapping[str, Any], key: str, default: float) -> float:
    try:
        return float(values.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _str(values: Mapping[str, Any], key: str, default: str) -> str:
    value = values.get(key, default)
    return str(default) if value is None else str(value)


@dataclass(frozen=True)
class MergeSettings:
    """Settled merge settings, one field per widget on the options node.

    Attributes:
        rank: Rank the SVD modes compress to, or 0 to choose one per module from
            ``auto_rank_threshold``.
        auto_rank_threshold: Fraction of singular-value energy an automatic rank keeps.
        preserve_norm: Whether ``svd`` rescales each merged module back to the average
            strength of its sources.
        cap_mult_enable: Whether ``svd`` caps each merged module's strength.
        cap_mult: The cap, as a multiple of the mean source strength.
        dtype: Precision the merged tensors are saved in.
        compute_dtype: Precision the merge arithmetic runs in, or ``"auto"``.
        cpu: Whether to merge on the CPU even where a GPU is available.
        include_patterns: Comma-separated substrings a module must contain to be merged.
        exclude_patterns: Comma-separated substrings that exclude a module.
        moe_temperature: Softmax temperature for ``moe`` gating.
        moe_hard: Whether ``moe`` picks one expert per module instead of blending.
        block_mix_method: How ``block-mix`` combines the routed modules.
        block_mix_preset: Model family whose block names ``block-mix`` routes by.
        block_mix_weighted: Whether ``block-mix`` blends A and B per role instead of
            choosing one of them.
        block_mix_concept_mix: Share of LoRA A in concept and attention modules.
        block_mix_style_mix: Share of LoRA A in style and feed-forward modules.
        lokr_layers: How the modes that recompress write a layer a LoKr is on: ``exact``,
            ``lokr`` or ``lora``.
    """

    rank: int = 32
    auto_rank_threshold: float = 0.99
    preserve_norm: bool = False
    cap_mult_enable: bool = False
    cap_mult: float = 1.0
    dtype: str = "bf16"
    compute_dtype: str = "auto"
    cpu: bool = False
    include_patterns: str = ""
    exclude_patterns: str = ""
    moe_temperature: float = 1.0
    moe_hard: bool = False
    block_mix_method: str = "svd"
    block_mix_preset: str = "auto"
    block_mix_weighted: bool = False
    block_mix_concept_mix: float = 0.5
    block_mix_style_mix: float = 0.5
    lokr_layers: str = "exact"

    @classmethod
    def read(cls, options: Any) -> "MergeSettings":
        """Build the settings from whatever arrived on the options socket.

        Args:
            options: The value on the socket. Anything that is not a mapping, most often
                ``None``, from an unconnected socket, produces the defaults.

        Returns:
            Settings with every field filled in.
        """
        values: Mapping[str, Any] = options if isinstance(options, Mapping) else {}
        lokr_layers = _str(values, "lokr_layers", "exact")
        return cls(
            rank=_int(values, "rank", 32),
            auto_rank_threshold=_float(values, "auto_rank_threshold", 0.99),
            preserve_norm=to_bool(values.get("preserve_norm", False), default=False),
            cap_mult_enable=to_bool(values.get("cap_mult_enable", False), default=False),
            cap_mult=_float(values, "cap_mult", 1.0),
            dtype=_str(values, "dtype", "bf16"),
            compute_dtype=_str(values, "compute_dtype", "auto"),
            cpu=to_bool(values.get("cpu", False), default=False),
            include_patterns=_str(values, "include_patterns", ""),
            exclude_patterns=_str(values, "exclude_patterns", ""),
            moe_temperature=_float(values, "moe_temperature", 1.0),
            moe_hard=to_bool(values.get("moe_hard", False), default=False),
            block_mix_method=_str(values, "block_mix_method", "svd"),
            block_mix_preset=_str(values, "block_mix_preset", "auto"),
            block_mix_weighted=to_bool(values.get("block_mix_weighted", False), default=False),
            block_mix_concept_mix=_float(values, "block_mix_concept_mix", 0.5),
            block_mix_style_mix=_float(values, "block_mix_style_mix", 0.5),
            lokr_layers=lokr_layers if lokr_layers in LOKR_LAYERS else "exact",
        )

    @property
    def include_list(self) -> list[str]:
        """``include_patterns`` split on commas, with blank entries dropped."""
        return [part.strip() for part in self.include_patterns.split(",") if part.strip()]

    @property
    def exclude_list(self) -> list[str]:
        """``exclude_patterns`` split on commas, with blank entries dropped."""
        return [part.strip() for part in self.exclude_patterns.split(",") if part.strip()]

    @property
    def cap(self) -> float | None:
        """The strength cap ``svd`` applies, or ``None`` when capping is switched off."""
        return self.cap_mult if self.cap_mult_enable else None
