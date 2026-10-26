"""Trim the silence off a clip's ends and shorten long pauses inside it."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ...modules.audio import clip, edit
from ...modules.interface.audio_sides import before_after

REQUIRES = "audio"


class AudioRemoveSilence(io.ComfyNode):
    """Take the dead air out of a clip."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioRemoveSilence",
            display_name="Audio Remove Silence",
            search_aliases=[
                "WASAudioRemoveSilence", "Audio Remove Silence", "trim silence", "strip silence",
                "remove pauses", "tighten speech",
            ],
            category="WAS Suite/Audio",
            description=(
                "Trim the silence off a clip's ends and shorten every long pause inside it to a "
                "set length, such as tightening generated speech. Pauses are crossfaded so they "
                "do not click."
            ),
            is_output_node=True,
            inputs=[
                io.Audio.Input("audio", tooltip="The clip to tighten."),
                io.Float.Input(
                    "threshold_db", default=-50.0, min=-120.0, max=0.0, step=0.5,
                    tooltip="Level in dBFS at or below which a stretch counts as silent. -50 = "
                    "studio quiet; -35 = also drops a noisy room tone.",
                ),
                io.Float.Input(
                    "max_pause", default=0.4, min=0.0, max=60.0, step=0.01, round=0.0001,
                    tooltip="Seconds any pause inside the clip is shortened to. 0.4 = natural "
                    "speech; 0 = remove pauses entirely; large = leave pauses alone.",
                ),
                io.Float.Input(
                    "keep_edges", default=0.05, min=0.0, max=10.0, step=0.01, round=0.0001,
                    tooltip="Seconds of quiet left at the very start and end. 0 = trimmed tight.",
                ),
                io.Boolean.Input(
                    "trim_ends", default=True,
                    tooltip="`true` = trim the start and end; `false` = only shorten pauses inside the clip.",
                ),
            ],
            outputs=[
                io.Audio.Output(display_name="audio", tooltip="The tightened clip. A batch is padded to its longest clip."),
                io.Float.Output(display_name="removed", tooltip="Seconds taken out of the first clip."),
            ],
        )

    @classmethod
    def execute(cls, audio, threshold_db=-50.0, max_pause=0.4, keep_edges=0.05, trim_ends=True) -> io.NodeOutput:
        """Tighten every clip of the batch.

        Raises:
            ValueError: The whole clip is silent.
        """
        wave = clip.waveform_of(audio)
        rate = clip.rate_of(audio)
        pause = clip.samples_at(max_pause, rate)
        edge = clip.samples_at(keep_edges, rate)
        joint = clip.samples_at(0.01, rate)
        results, cut_spans = [], []
        for index in range(int(wave.shape[0])):
            item = wave[index:index + 1]
            length = int(item.shape[-1])
            levels, hop = edit.silence_levels(item, rate)
            loud = edit.loud_spans(levels, hop, length, threshold_db, max(1, pause), 0, 1)
            if not loud:
                raise ValueError(
                    f"clip {index + 1} is silent below {threshold_db:.1f} dBFS throughout. Lower "
                    f"threshold_db, or check the clip."
                )
            removals = []
            for (_, end_a), (start_b, _) in zip(loud, loud[1:]):
                gap = start_b - end_a
                if gap > pause:
                    middle = end_a + gap // 2
                    removals.append((middle - (gap - pause) // 2, middle + (gap - pause + 1) // 2))
            if trim_ends:
                head = max(0, loud[0][0] - edge)
                tail = min(length, loud[-1][1] + edge)
                if head > 0:
                    removals.insert(0, (0, head))
                if tail < length:
                    removals.append((tail, length))
            if index == 0:
                cut_spans = removals
            kept = edit.without(item, removals, joint, "equal power") if removals else item.clone()
            results.append(kept)
        longest = max(int(item.shape[-1]) for item in results)
        out = torch.cat([edit.padded(item, 0, longest - int(item.shape[-1])) for item in results], dim=0)
        removed = sum(b - a for a, b in cut_spans) / rate
        facts = {"silence": {"removed": [[a / rate, b / rate] for a, b in cut_spans[:2000]]}}
        return io.NodeOutput(clip.make(out, rate), removed, ui=before_after(wave, rate, out, rate, facts))
