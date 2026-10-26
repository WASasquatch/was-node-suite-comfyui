"""Rework a clip's channels: mono, stereo, swapped, one side, mid and side, or flipped polarity."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ...modules.audio import clip
from ...modules.interface.audio_sides import before_after

REQUIRES = "audio"

#: The channel operations offered, in menu order.
OPERATIONS = (
    "to mono",
    "to stereo",
    "swap left and right",
    "left only",
    "right only",
    "mid/side encode",
    "mid/side decode",
    "invert polarity",
    "invert left",
    "invert right",
)


def _stereo(wave: torch.Tensor) -> torch.Tensor:
    """A waveform as two channels."""
    return clip.with_channels(wave, 2)


def reworked(wave: torch.Tensor, operation: str) -> torch.Tensor:
    """A waveform with one channel operation applied.

    Args:
        wave: ``(batch, channels, samples)``.
        operation: A name from :data:`OPERATIONS`.

    Returns:
        A new tensor.

    Raises:
        ValueError: ``operation`` is not offered.
    """
    if operation == "to mono":
        return wave.mean(dim=1, keepdim=True)
    if operation == "to stereo":
        return _stereo(wave).clone()
    if operation == "invert polarity":
        return -wave
    pair = _stereo(wave)
    left, right = pair[:, :1], pair[:, 1:2]
    if operation == "swap left and right":
        return torch.cat([right, left], dim=1)
    if operation == "left only":
        return torch.cat([left, left], dim=1)
    if operation == "right only":
        return torch.cat([right, right], dim=1)
    if operation == "mid/side encode":
        return torch.cat([(left + right) * 0.5, (left - right) * 0.5], dim=1)
    if operation == "mid/side decode":
        return torch.cat([left + right, left - right], dim=1)
    if operation == "invert left":
        return torch.cat([-left, right], dim=1)
    if operation == "invert right":
        return torch.cat([left, -right], dim=1)
    raise ValueError(f"operation {operation!r} is not one of {', '.join(OPERATIONS)}.")


class AudioChannels(io.ComfyNode):
    """Change how a clip's channels are laid out."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioChannels",
            display_name="Audio Channels",
            search_aliases=[
                "WASAudioChannels", "Audio Channels", "to mono", "to stereo", "swap channels",
                "mid side", "invert polarity", "phase invert",
            ],
            category="WAS Suite/Audio",
            description=(
                "Rework a clip's channels: fold to mono, spread to stereo, swap or keep one "
                "side, encode or decode mid and side, or flip polarity to fix a cancelling pair."
            ),
            is_output_node=True,
            inputs=[
                io.Audio.Input("audio", tooltip="The clip to rework."),
                io.Combo.Input(
                    "operation", options=list(OPERATIONS), default="to stereo",
                    tooltip="to mono = average every channel; mid/side encode = left is mid, right "
                    "is side; mid/side decode = back to left and right; invert = flip the waveform "
                    "upside down.",
                ),
            ],
            outputs=[io.Audio.Output(display_name="audio", tooltip="The reworked clip.")],
        )

    @classmethod
    def execute(cls, audio, operation="to stereo") -> io.NodeOutput:
        """Apply the operation."""
        wave = clip.waveform_of(audio)
        rate = clip.rate_of(audio)
        out = reworked(wave, operation).contiguous()
        return io.NodeOutput(clip.make(out, rate), ui=before_after(wave, rate, out, rate))
