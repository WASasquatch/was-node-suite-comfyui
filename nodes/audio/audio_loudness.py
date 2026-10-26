"""Measure a clip's loudness, loudness range, true peak and levels, drawn as a meter on the node."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.audio import clip, loudness
from ...modules.interface.audio_sides import figures, panel_ui

REQUIRES = "audio"


class AudioLoudness(io.ComfyNode):
    """Loudness and level figures of one clip."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioLoudness",
            display_name="Audio Loudness",
            search_aliases=[
                "WASAudioLoudness", "Audio Loudness", "loudness meter", "lufs meter", "true peak",
                "loudness range", "lra", "audio meter",
            ],
            category="WAS Suite/Audio",
            description=(
                "Measure a clip's integrated loudness in LUFS, its loudness range, true peak, "
                "peak and RMS levels, to check a mix against a delivery target. The node draws "
                "the figures and the loudness over time."
            ),
            is_output_node=True,
            inputs=[
                io.Audio.Input("audio", tooltip="The clip to measure. The first clip of a batch is measured."),
                io.Float.Input(
                    "target", default=-14.0, min=-70.0, max=0.0, step=0.1, round=0.01,
                    tooltip="Loudness the meter compares against, in LUFS. -14 = streaming music; "
                    "-16 = podcasts; -23 = broadcast. Drawn as a line on the graph.",
                ),
            ],
            outputs=[
                io.Float.Output(display_name="integrated", tooltip="Integrated loudness in LUFS. -120 = silent."),
                io.Float.Output(display_name="range", tooltip="Loudness range in LU: 3 = flat and loud; 15 = dynamic."),
                io.Float.Output(display_name="true_peak", tooltip="Highest point between samples, in dBTP. Above -1 risks clipping once encoded."),
                io.Float.Output(display_name="sample_peak", tooltip="Loudest sample in dBFS. 0 = full scale."),
                io.Float.Output(display_name="short_term_max", tooltip="Loudest 3 seconds, in LUFS."),
                io.Float.Output(display_name="rms", tooltip="Average level in dBFS."),
                io.String.Output(
                    display_name="summary",
                    tooltip="Every figure on one line: `-14.2 LUFS, LRA 6.1 LU, -1.3 dBTP, peak -1.5 dBFS`.",
                ),
            ],
        )

    @classmethod
    def execute(cls, audio, target=-14.0) -> io.NodeOutput:
        """Measure the first clip."""
        wave = clip.waveform_of(audio)
        rate = clip.rate_of(audio)
        reading = loudness.measure(wave[0], rate)
        summary = (
            f"{reading['integrated']:.1f} LUFS, LRA {reading['range']:.1f} LU, "
            f"{reading['true_peak']:.1f} dBTP, peak {reading['sample_peak']:.1f} dBFS, "
            f"rms {reading['rms']:.1f} dBFS"
        )
        report = {"in": figures(wave, rate), "loudness": {**reading, "target": float(target)}}
        return io.NodeOutput(
            reading["integrated"], reading["range"], reading["true_peak"], reading["sample_peak"],
            reading["short_term_max"], reading["rms"], summary,
            ui=panel_ui([("in", wave, rate)], report),
        )
