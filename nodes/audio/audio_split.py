"""Split a clip into several, at silences, at set times, at a fixed interval, into equal parts or on beats."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.audio import clip, edit
from ...modules.compat.lists import require_values
from ...modules.interface.audio_sides import figures, panel_ui

REQUIRES = "audio"

#: Most clips one split answers.
MAX_CLIPS = 2000


def _spans_for(choice: dict, wave, rate: int) -> list[tuple[int, int]]:
    """The sample spans one clip is split into, for the chosen mode."""
    length = int(wave.shape[-1])
    mode = choice.get("mode", "silence")
    if mode == "silence":
        levels, hop = edit.silence_levels(wave, rate)
        return edit.loud_spans(
            levels, hop, length, float(choice.get("threshold_db", -45.0)),
            clip.samples_at(choice.get("min_silence", 0.3), rate),
            clip.samples_at(choice.get("padding", 0.05), rate),
            clip.samples_at(choice.get("min_clip", 0.25), rate),
        )
    if mode == "times":
        cuts = [clip.samples_at(t, rate) for t in edit.parse_times(choice.get("times", ""), "times")]
    elif mode == "every":
        step = max(1, clip.samples_at(choice.get("interval", 10.0), rate))
        cuts = list(range(step, length, step))
    elif mode == "parts":
        parts = max(1, int(choice.get("parts", 2)))
        cuts = [round(length * index / parts) for index in range(1, parts)]
    else:
        from ...modules.media import beats

        found = beats.analyse(clip.make(wave[:1], rate), 10.0, int(choice.get("beats_per_bar", 4)))
        marks = found.downbeats if choice.get("unit") == "bars" else found.times
        cuts = [clip.samples_at(t, rate) for t in marks[:: max(1, int(choice.get("count", 4)))]]
    edges = [0] + sorted({c for c in cuts if 0 < c < length}) + [length]
    return [(a, b) for a, b in zip(edges, edges[1:]) if b > a]


class AudioSplit(io.ComfyNode):
    """Split one clip into a list of clips."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioSplit",
            display_name="Audio Split",
            search_aliases=[
                "WASAudioSplit", "Audio Split", "split on silence", "chop audio", "slice audio",
                "split audio", "beat slicer",
            ],
            category="WAS Suite/Audio",
            description=(
                "Split a clip into a list of clips: at silences, at listed times, every few "
                "seconds, into equal parts or on beats. The waveform on the node marks every "
                "clip, and in times mode a click places a cut."
            ),
            is_output_node=True,
            inputs=[
                io.Audio.Input("audio", tooltip="The clip to split. Every clip of a batch is split the same way."),
                io.DynamicCombo.Input(
                    "mode",
                    options=[
                        io.DynamicCombo.Option(key="silence", inputs=[
                            io.Float.Input(
                                "threshold_db", default=-45.0, min=-120.0, max=0.0, step=0.5,
                                tooltip="Level in dBFS at or below which a stretch counts as quiet. "
                                "-45 = quiet room; -60 = only near silence; -30 = soft passages split too.",
                            ),
                            io.Float.Input(
                                "min_silence", default=0.3, min=0.01, max=60.0, step=0.01,
                                tooltip="Seconds a quiet stretch lasts before it parts two clips. 0.3 = "
                                "between sentences; 1.5 = between tracks.",
                            ),
                            io.Float.Input(
                                "padding", default=0.05, min=0.0, max=10.0, step=0.01,
                                tooltip="Seconds of the quiet kept at each end of a clip, so a word or "
                                "note is not clipped. 0 = cut tight.",
                            ),
                            io.Float.Input(
                                "min_clip", default=0.25, min=0.0, max=600.0, step=0.01,
                                tooltip="Seconds a sound lasts to be kept. 0.25 drops coughs and clicks; "
                                "0 keeps every sound.",
                            ),
                        ]),
                        io.DynamicCombo.Option(key="times", inputs=[
                            io.String.Input(
                                "times", default="",
                                tooltip="Where to cut, in seconds: `12.5, 1:02, 1:30.25`. Click the "
                                "waveform to add one, drag to move it, double-click to remove it.",
                            ),
                        ]),
                        io.DynamicCombo.Option(key="every", inputs=[
                            io.Float.Input(
                                "interval", default=10.0, min=0.01, max=10800.0, step=0.01,
                                tooltip="Seconds each clip lasts; the last one keeps what is left.",
                            ),
                        ]),
                        io.DynamicCombo.Option(key="parts", inputs=[
                            io.Int.Input(
                                "parts", default=2, min=1, max=MAX_CLIPS,
                                tooltip="How many equal clips to make: 2 = halves, 4 = quarters.",
                            ),
                        ]),
                        io.DynamicCombo.Option(key="beats", inputs=[
                            io.Combo.Input(
                                "unit", options=["beats", "bars"], default="bars",
                                tooltip="beats = cut on detected beats; bars = cut on the first beat "
                                "of each bar.",
                            ),
                            io.Int.Input(
                                "count", default=1, min=1, max=256,
                                tooltip="How many beats or bars each clip lasts: 1 = every one; 4 = "
                                "four bars a clip.",
                            ),
                            io.Int.Input(
                                "beats_per_bar", default=4, min=1, max=16,
                                tooltip="Beats in one bar: 4 for most music, 3 for a waltz.",
                            ),
                        ]),
                    ],
                    tooltip="silence = cut where it goes quiet; times = cut where listed; every = "
                    "fixed length clips; parts = equal clips; beats = cut on the beat or bar.",
                ),
                io.Float.Input(
                    "fade", default=0.005, min=0.0, max=10.0, step=0.001, round=0.0001,
                    tooltip="Seconds each clip fades in and out over, so a cut through a sound does "
                    "not click. 0.005 = inaudible; 0 = hard edges.",
                ),
            ],
            outputs=[
                io.Audio.Output(display_name="clips", is_output_list=True, tooltip="The clips, in order, as a list."),
                io.Int.Output(display_name="count", tooltip="How many clips the split made."),
                io.Float.Output(
                    display_name="starts", is_output_list=True,
                    tooltip="Seconds into the source each clip starts, one per clip.",
                ),
            ],
        )

    @classmethod
    def execute(cls, audio, mode, fade=0.005) -> io.NodeOutput:
        """Split every clip of the batch.

        Raises:
            ValueError: The split makes no clips, more than :data:`MAX_CLIPS`, or a time is unreadable.
        """
        wave = clip.waveform_of(audio)
        rate = clip.rate_of(audio)
        choice = mode if isinstance(mode, dict) else {"mode": str(mode)}
        edge = clip.samples_at(fade, rate)
        clips, starts, first_spans = [], [], []
        for index in range(int(wave.shape[0])):
            item = wave[index:index + 1]
            spans = _spans_for(choice, item, rate)
            if index == 0:
                first_spans = spans
            for lo, hi in spans:
                piece = edit.faded(item[..., lo:hi], edge, edge, "equal power", "equal power")
                clips.append(clip.make(piece, rate))
                starts.append(lo / rate)
            if len(clips) > MAX_CLIPS:
                raise ValueError(
                    f"the split makes more than {MAX_CLIPS} clips. Raise min_silence or the "
                    f"interval, or split a shorter clip."
                )
        require_values(clips, "the split found nothing to keep: every part is quiet or shorter "
                       "than min_clip. Lower threshold_db or min_clip.")
        report = {
            "in": figures(wave, rate),
            "split": {"mode": choice.get("mode", ""), "spans": [[a / rate, b / rate] for a, b in first_spans]},
        }
        return io.NodeOutput(clips, len(clips), starts, ui=panel_ui([("in", wave, rate)], report))
