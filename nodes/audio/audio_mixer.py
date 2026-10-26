"""A mixer: tracks chained one into the next, summed by the mixer at the end of the chain."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.audio import clip, mixer
from ...modules.compat.types import AUDIO_TRACKS
from ...modules.interface.audio_sides import figures, panel_ui

REQUIRES = "audio"


class AudioMixerTrack(io.ComfyNode):
    """One clip on the mixer, with its fader, pan, mute, solo and start."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioMixerTrack",
            display_name="Audio Mixer Track",
            search_aliases=[
                "WASAudioMixerTrack", "Audio Mixer Track", "mixer channel", "channel strip",
                "audio track",
            ],
            category="WAS Suite/Audio",
            description=(
                "One channel of Audio Mixer: a clip with its fader, pan, mute, solo and start "
                "time. Chain tracks one into the next and feed the last into Audio Mixer."
            ),
            inputs=[
                io.Audio.Input("audio", tooltip="The clip this track plays."),
                AUDIO_TRACKS.Input(
                    "tracks", optional=True,
                    tooltip="The tracks before this one, from another Audio Mixer Track. Empty = the first track.",
                ),
                io.Float.Input(
                    "gain_db", default=0.0, min=-60.0, max=24.0, step=0.1, round=0.01,
                    tooltip="Fader level in dB. 0 = as recorded; -6 = half as loud; +6 = twice as loud.",
                ),
                io.Float.Input(
                    "pan", default=0.0, min=-1.0, max=1.0, step=0.01,
                    tooltip="-1 = hard left; 0 = centre; 1 = hard right. A mono clip is panned at "
                    "constant power; a stereo clip is balanced.",
                ),
                io.Boolean.Input("mute", default=False, tooltip="`true` = this track is left out of the mix; `false` = it plays."),
                io.Boolean.Input(
                    "solo", default=False,
                    tooltip="`true` = only soloed tracks play while any track is soloed.",
                ),
                io.Float.Input(
                    "start", default=0.0, min=0.0, max=clip.MAX_SECONDS, step=0.01, round=0.0001,
                    tooltip="Seconds into the mix this clip begins. 0 = at the top.",
                ),
                io.String.Input(
                    "name", default="",
                    tooltip="The label the mixer draws for this track: `Vocals`. Empty = track 1, track 2.",
                ),
            ],
            outputs=[
                AUDIO_TRACKS.Output(
                    display_name="tracks",
                    tooltip="Every track so far, this one last. Feed another Audio Mixer Track or Audio Mixer.",
                ),
            ],
        )

    @classmethod
    def execute(cls, audio, tracks=None, gain_db=0.0, pan=0.0, mute=False, solo=False, start=0.0,
                name="") -> io.NodeOutput:
        """Append this track to the chain.

        Raises:
            ValueError: ``tracks`` is not a track chain, or the chain is already full.
        """
        chain = tuple(tracks or ())
        if any(not isinstance(item, mixer.Track) for item in chain):
            raise ValueError("tracks takes the output of another Audio Mixer Track.")
        if len(chain) >= mixer.MAX_TRACKS:
            raise ValueError(f"the chain already holds {len(chain)} tracks; one mixer sums at most {mixer.MAX_TRACKS}.")
        clip.rate_of(audio)
        clip.waveform_of(audio)
        track = mixer.Track(
            audio=audio, gain_db=float(gain_db), pan=float(pan), mute=bool(mute), solo=bool(solo),
            start=float(start), name=str(name or "").strip()[:64],
        )
        return io.NodeOutput(chain + (track,))


class AudioMixer(io.ComfyNode):
    """Sum a chain of tracks into one clip."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioMixer",
            display_name="Audio Mixer",
            search_aliases=[
                "WASAudioMixer", "Audio Mixer", "mix audio", "mixdown", "audio mixdown",
                "mixing console",
            ],
            category="WAS Suite/Audio",
            description=(
                "Mix every track of an Audio Mixer Track chain into one clip, each at its own "
                "level, pan and start. The node draws each track's level and plays the mix."
            ),
            is_output_node=True,
            inputs=[
                AUDIO_TRACKS.Input("tracks", tooltip="The last Audio Mixer Track of the chain."),
                io.Float.Input(
                    "master_db", default=0.0, min=-60.0, max=24.0, step=0.1, round=0.01,
                    tooltip="Master fader in dB, applied after the tracks are summed. 0 = unchanged.",
                ),
                io.Combo.Input(
                    "length", options=list(mixer.LENGTHS), default="longest track",
                    tooltip="`longest track` = until the last track ends; `first track` = as long as "
                    "the first track plus its start; `fixed` = duration seconds.",
                ),
                io.Float.Input(
                    "duration", default=10.0, min=0.01, max=clip.MAX_SECONDS, step=0.01, round=0.0001,
                    tooltip="Seconds the mix runs while length is `fixed`, such as `30`. Ignored otherwise.",
                ),
                io.Combo.Input(
                    "layout", options=list(mixer.LAYOUTS), default="stereo",
                    tooltip="`stereo` = two channels, pan applies; `mono` = one channel, pan ignored.",
                ),
            ],
            outputs=[
                io.Audio.Output(display_name="audio", tooltip="The mix, at the highest sample rate among the tracks."),
                io.Float.Output(display_name="duration", tooltip="Seconds the mix plays for."),
            ],
        )

    @classmethod
    def execute(cls, tracks, master_db=0.0, length="longest track", duration=10.0, layout="stereo") -> io.NodeOutput:
        """Mix the chain.

        Raises:
            ValueError: ``tracks`` is empty or not a track chain.
        """
        chain = tuple(tracks or ())
        if not chain or any(not isinstance(item, mixer.Track) for item in chain):
            raise ValueError("tracks takes the last Audio Mixer Track of a chain.")
        out, rate, rows = mixer.mixed(chain, master_db, length, duration, layout)
        report = {"out": figures(out, rate), "mixer": {"tracks": rows, "master_db": float(master_db)}}
        return io.NodeOutput(
            clip.make(out, rate), clip.duration(out, rate),
            ui=panel_ui([("out", out, rate)], report),
        )
