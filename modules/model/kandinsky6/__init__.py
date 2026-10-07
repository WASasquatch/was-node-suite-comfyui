"""Kandinsky 6 joint video and audio generation on ComfyUI's own loaders, samplers and decoders."""

from __future__ import annotations


def register() -> None:
    """Make ComfyUI's diffusion model loaders recognise Kandinsky 6 transformers."""
    from .model import register as register_model

    register_model()
