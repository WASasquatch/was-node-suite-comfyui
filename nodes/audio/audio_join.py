"""Join clips end to end into one, crossfaded or parted by silence."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.audio import clip, curves, edit
from ...modules.interface.audio_sides import figures, panel_ui

REQUIRES = "audio"

#: Most clips one join reads.
MAX_CLIPS = 2000


def _first(value, default):
    """The first entry of a list input, or ``default``."""
    if isinstance(value, list):
        return value[0] if value else default
    return default if value is None else value


class AudioJoin(io.ComfyNode):
    """Join a list of clips into one."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioJoin",
            display_name="Audio Join",
            search_aliases=[
                "WASAudioJoin", "Audio Join", "concatenate audio", "join audio", "merge clips",
                "audio concat",
            ],
            category="WAS Suite/Audio",
            description=(
                "Join clips end to end into one, overlapped by a crossfade or parted by "
                "silence. Takes the list Audio Split answers, or one clip on each input, and "
                "chains: join the result to the next clip."
            ),
            is_output_node=True,
            is_input_list=True,
            inputs=[
                io.Audio.Input(
                    "audio",
                    tooltip="The clips that come first, in order: a list from Audio Split, or one clip.",
                ),
                io.Audio.Input(
                    "append", optional=True,
                    tooltip="Clips joined after audio, in order. Leave empty to join only the list on audio.",
                ),
                io.Float.Input(
                    "crossfade", default=0.0, min=0.0, max=60.0, step=0.01, round=0.0001,
                    tooltip="Seconds each join overlaps by. 0 = butt splice; 0.01 = click-free; 2 = "
                    "a slow blend. Held to half the shorter clip.",
                ),
                io.Float.Input(
                    "gap", default=0.0, min=0.0, max=600.0, step=0.01, round=0.0001,
                    tooltip="Seconds of silence between clips. Above 0 the clips do not overlap, and "
                    "crossfade becomes a fade at each side of the gap.",
                ),
                io.Combo.Input(
                    "curve", options=list(curves.CURVES), default=curves.DEFAULT_CURVE,
                    tooltip="Shape of the crossfades. `equal power` = steady level; `linear` = straight "
                    "ramp; `s-curve` = soft ends; `exponential` = slow start; `logarithmic` = fast start.",
                ),
            ],
            outputs=[
                io.Audio.Output(display_name="audio", tooltip="Every clip joined into one."),
                io.Float.Output(display_name="duration", tooltip="Seconds the joined clip plays for."),
            ],
        )

    @classmethod
    def execute(cls, audio, append=None, crossfade=None, gap=None, curve=None) -> io.NodeOutput:
        """Join ``audio`` then ``append``.

        Raises:
            ValueError: No clips arrived, or more than :data:`MAX_CLIPS`.
        """
        clips = [item for item in list(audio or []) + list(append or []) if item is not None]
        if not clips:
            raise ValueError("Audio Join received no clips. Wire a clip or a list of clips into audio.")
        if len(clips) > MAX_CLIPS:
            raise ValueError(f"Audio Join received {len(clips)} clips; it joins at most {MAX_CLIPS}.")
        waves, rate, _ = clip.matched(clips, [f"clip {index + 1}" for index in range(len(clips))])
        batches = {int(wave.shape[0]) for wave in waves}
        if len(batches) > 1:
            batch = max(batches)
            if batches - {1, batch}:
                raise ValueError(
                    f"the clips carry batches of {sorted(batches)}; joined clips hold the same count, or 1."
                )
            waves = [wave.expand(batch, -1, -1) for wave in waves]
        overlap = clip.samples_at(float(_first(crossfade, 0.0)), rate)
        silence = clip.samples_at(float(_first(gap, 0.0)), rate)
        shape = str(_first(curve, curves.DEFAULT_CURVE))
        out = edit.joined(waves, overlap, shape, silence)
        joins, cursor = [], 0
        for index, wave in enumerate(waves[:-1]):
            length = int(wave.shape[-1])
            following = int(waves[index + 1].shape[-1])
            step = length + silence if silence else length - min(overlap, length // 2, following // 2)
            cursor += step
            joins.append(cursor / rate)
        report = {"out": figures(out, rate), "join": {"count": len(waves), "joins": joins}}
        return io.NodeOutput(
            clip.make(out, rate), clip.duration(out, rate),
            ui=panel_ui([("out", out, rate)], report),
        )
