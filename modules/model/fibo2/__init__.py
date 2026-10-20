"""Fibo 2.0 image generation and editing on ComfyUI's own loaders, samplers and decoders."""

from __future__ import annotations


def register() -> None:
    """Make ComfyUI's diffusion model loaders recognise Fibo 2.0 transformers."""
    from .model import register as register_model

    register_model()
