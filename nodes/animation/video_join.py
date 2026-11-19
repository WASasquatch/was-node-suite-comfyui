"""Joining clips end to end with a transition at each join."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import EMA_VFI_MODEL
from ...modules.image import sizing
from ...modules.media import join, mixing

logger = log.get_logger("nodes.video_join")

NODE_NAME = "Video Join"


def _first(value, default):
    """The first entry of a list input, or ``default``."""
    if isinstance(value, list):
        return value[0] if value else default
    return default if value is None else value


def _floats(frames):
    """Frames as float32, uint8 codes divided by 255."""
    import torch

    if frames.dtype == torch.uint8:
        return frames.to(torch.float32) / 255.0
    return frames if frames.dtype == torch.float32 else frames.to(torch.float32)


def _with_alpha(clip, wanted: bool):
    """A clip's frames as floats, with its coverage as a fourth channel where any clip carries one."""
    import torch

    frames = _floats(clip.frames)[..., :3]
    if not wanted:
        return frames
    if clip.alpha is not None:
        alpha = _floats(clip.alpha).reshape(frames.shape[:3]).unsqueeze(-1)
    else:
        alpha = torch.ones(frames.shape[:3] + (1,), dtype=torch.float32)
    return torch.cat([frames, alpha.to(frames.device)], dim=-1)


class VideoJoin(io.ComfyNode):
    """Join clips end to end, with a transition at every join."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASVideoJoin",
            display_name=NODE_NAME,
            search_aliases=[
                "WASVideoJoin", NODE_NAME,
                "concatenate video",
                "join clips",
                "cross dissolve",
                "dip to black",
                "video transition",
                "wipe",
            ],
            category="WAS Suite/Animation",
            is_output_node=True,
            is_input_list=True,
            description=(
                "Join clips end to end with a cut, dissolve, dip or wipe at each join, the "
                "incoming clip's head laid over the outgoing clip's tail. Takes the clip so far "
                "and the next one, or the list Video Split Scenes answers. Every clip is brought "
                "to the first one's size and frame rate, its audio crossfaded under each join."
            ),
            inputs=[
                io.Video.Input(
                    "video",
                    tooltip=(
                        "The clip so far, or the first clip; a list from Video Split Scenes "
                        "joins in order. The first clip sets the joined clip's size, frame "
                        "rate and sample rate."
                    ),
                ),
                io.Combo.Input(
                    "transition",
                    options=list(join.TRANSITIONS),
                    default="dissolve",
                    tooltip=(
                        "`cut` = straight join; `dissolve` = mixes across; `dip to black`, "
                        "`dip to white` = out to the colour at the middle, then in; `wipe` = "
                        "an edge sweeps the next clip on. Used at every join."
                    ),
                ),
                io.Int.Input(
                    "duration",
                    default=12,
                    min=0,
                    max=join.MAX_DURATION,
                    tooltip=(
                        "Frames each transition lasts at the first clip's rate: 12 = half a "
                        "second at 24 fps; 0 = a cut. The joined clip holds every clip's frames "
                        "less every transition's. Held to the frames each clip has."
                    ),
                ),
                io.Combo.Input(
                    "wipe_direction",
                    options=list(join.DIRECTIONS),
                    default=join.DIRECTIONS[0],
                    tooltip=(
                        "Which way a wipe's edge travels: `left to right` shows the next clip "
                        "at the left edge first. Read only by `wipe`."
                    ),
                ),
                io.Float.Input(
                    "wipe_softness",
                    default=0.1,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Width of a wipe's edge as a share of the frame: 0 = a hard line; "
                        "0.1 = soft; 1 = as wide as the frame. Read only by `wipe`."
                    ),
                ),
                io.Combo.Input(
                    "mix",
                    options=list(mixing.SPACES),
                    default=mixing.CODES,
                    tooltip=(
                        "`codes` = mixes the values as they arrive, an editor's default and "
                        "right for linear-light frames; `light` = decodes sRGB codes to light, "
                        "mixes, encodes again: a film dissolve, brighter through the middle."
                    ),
                ),
                io.Combo.Input(
                    "fit",
                    options=list(sizing.MODES),
                    default=sizing.FIT_AND_PAD,
                    tooltip=(
                        "How a clip of another size meets the first clip's: `fit and pad` = "
                        "whole frame inside black bars; `fill and crop` = fills, trims the "
                        "overhang; `stretch` = distorts; `crop or pad` = never resamples."
                    ),
                ),
                io.Combo.Input(
                    "frame_rate",
                    options=list(join.RATE_MODES),
                    default="hold",
                    tooltip=(
                        "How a clip at another frame rate meets the first clip's, its running "
                        "time kept: `hold` = repeats or drops whole frames; `blend` = mixes "
                        "neighbours; `interpolate` = draws new ones with EMA-VFI."
                    ),
                ),
                io.Video.Input(
                    "next_video",
                    optional=True,
                    tooltip=(
                        "The clip joined after video, or a list joined in order. Leave empty "
                        "to join only the list on video."
                    ),
                ),
                EMA_VFI_MODEL.Input(
                    "ema_vfi_model",
                    optional=True,
                    tooltip=(
                        "The interpolation network from EMA-VFI Video Model Loader, for "
                        "frame_rate `interpolate`. A rate change other than doubling needs an "
                        "'ours_t' checkpoint."
                    ),
                ),
            ],
            outputs=[
                io.Video.Output(
                    display_name="video",
                    tooltip=(
                        "Every clip joined, at the first clip's size and frame rate, with the "
                        "audio crossfaded under each transition."
                    ),
                ),
                io.Int.Output(display_name="frames", tooltip="How many frames the joined clip holds."),
                io.Int.Output(
                    display_name="starts",
                    is_output_list=True,
                    tooltip="The frame each clip starts on in the joined clip, counting from 0: [0, 36, 84].",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        video,
        transition=None,
        duration=None,
        wipe_direction=None,
        wipe_softness=None,
        mix=None,
        fit=None,
        frame_rate=None,
        next_video=None,
        ema_vfi_model=None,
    ) -> io.NodeOutput:
        """Join every clip on ``video``, then every clip on ``next_video``.

        Raises:
            ValueError: No clip arrived, more than :data:`modules.media.join.MAX_CLIPS`, a
                setting names nothing the node offers, 'interpolate' has no network, or the
                clips' audio batches cannot be paired.
            MemoryError: Neither free memory nor a scratch drive can hold the joined clip.
        """
        from ...modules.image import scratch
        from ...modules.media import clip as clips

        given = [item for item in list(video or []) + list(next_video or []) if item is not None]
        if not given:
            raise ValueError(f"{NODE_NAME} received no clip. Wire a clip, or a list of clips, into video.")
        if len(given) > join.MAX_CLIPS:
            raise ValueError(f"{NODE_NAME} received {len(given)} clips; it joins at most {join.MAX_CLIPS}.")

        kind = str(_first(transition, "dissolve"))
        direction = str(_first(wipe_direction, join.DIRECTIONS[0]))
        space = str(_first(mix, mixing.CODES))
        shape = str(_first(fit, sizing.FIT_AND_PAD))
        rate_mode = str(_first(frame_rate, "hold"))
        for value, offered, name in (
            (kind, join.TRANSITIONS, "transition"),
            (direction, join.DIRECTIONS, "wipe_direction"),
            (space, mixing.SPACES, "mix"),
            (shape, sizing.MODES, "fit"),
            (rate_mode, join.RATE_MODES, "frame_rate"),
        ):
            if value not in offered:
                raise ValueError(
                    f"{NODE_NAME}'s {name} is '{value}', which is not one it offers: "
                    f"{', '.join(offered)}."
                )
        length = max(0, min(join.MAX_DURATION, int(_first(duration, 12))))
        softness = max(0.0, min(1.0, float(_first(wipe_softness, 0.1))))
        net = _first(ema_vfi_model, None)

        opened = [clips.open_clip(item, NODE_NAME) for item in given]
        first = opened[0]
        height, width = int(first.frames.shape[1]), int(first.frames.shape[2])
        rate = first.rate
        wants_alpha = any(clip.alpha is not None for clip in opened)
        step = clips.progress(len(opened))

        pieces, conformed = [], []
        for number, clip in enumerate(opened):
            frames, notes = join.conformed(
                _with_alpha(clip, wants_alpha), clip.rate, (width, height), rate, shape,
                rate_mode, net, NODE_NAME,
            )
            pieces.append(frames)
            if notes:
                conformed.append(f"clip {number + 1}: {', '.join(notes)}")
            step(1)

        lengths = [int(frames.shape[0]) for frames in pieces]
        shared = join.overlaps(lengths, length, kind)
        starts = join.starts_of(lengths, shared)
        total = starts[-1] + lengths[-1]
        channels = int(pieces[0].shape[-1])

        import torch

        out = scratch.allocate(
            (total, height, width, channels), torch.float32, node=NODE_NAME,
            advice="Join fewer or shorter clips, or bring them to a smaller size first.",
        )
        last = len(pieces) - 1
        for number, frames in enumerate(pieces):
            head = shared[number - 1] if number > 0 else 0
            tail = shared[number] if number < last else 0
            begin = starts[number]
            out[begin + head:begin + lengths[number] - tail] = frames[head:lengths[number] - tail]
            scratch.trim(out[begin + head:begin + lengths[number] - tail])
        step = clips.progress(sum(shared))
        for number, overlap in enumerate(shared):
            outgoing, incoming = pieces[number], pieces[number + 1]
            for index in range(overlap):
                out[starts[number + 1] + index] = join.transition_frame(
                    outgoing[lengths[number] - overlap + index].to(torch.float32).cpu(),
                    incoming[index].to(torch.float32).cpu(),
                    kind, join.share_of(index, overlap), space, direction, softness,
                )
                step(1)

        sound = join.joined_audio([clip.audio for clip in opened], lengths, starts, shared, rate)
        if wants_alpha:
            images, alpha = out[..., :3].contiguous(), out[..., 3].contiguous()
        else:
            images, alpha = out, None
        answer = clips.rebuild(first, images, alpha=alpha, audio=sound, rate=rate)

        held = [
            f"join {number + 1} to {overlap}"
            for number, overlap in enumerate(shared)
            if kind != "cut" and overlap < length
        ]
        _report(images, starts, shared, lengths, (kind, length, direction, softness, space),
                rate, sound, conformed, held)
        logger.info("joined %d clip(s) into %d frame(s) with %s", len(pieces), total, kind)
        return io.NodeOutput(answer, total, starts)


def _report(images, starts, shared, lengths, setting, rate, sound, conformed, held) -> None:
    """Publish what a join made, for the panel on the node.

    Args:
        images: The joined frames.
        starts: The frame each clip starts on.
        shared: Frames each join overlaps by.
        lengths: Frames in each clip.
        setting: ``(transition, duration, direction, softness, space)`` as the node read them.
        rate: Frames per second.
        sound: The joined ``AUDIO``, or None.
        conformed: One note per clip brought to the first clip's size or rate.
        held: One note per transition held short of its duration.
    """
    from ...modules.interface import preview as published
    from ...modules.interface import run_result

    kind, length, direction, softness, space = setting
    total = starts[-1] + lengths[-1]
    fps = float(rate)
    seconds = total / fps if fps > 0 else 0.0
    described = f"{kind}, {length} frame(s)" if kind != "cut" else "cut"
    if kind == "wipe":
        described += f", {direction}, softness {softness:g}"
    if kind != "cut":
        described += f", mixed in {space}"
    if sound is not None:
        audio = f"{int(sound['sample_rate'])} Hz, {int(sound['waveform'].shape[1])} channel(s)"
    else:
        audio = "none"
    facts = {
        "timeline": join.timeline_text(lengths, shared, kind),
        "size": f"{int(images.shape[2])} x {int(images.shape[1])}",
        "rate": f"{fps:.6g} fps",
        "transition": described,
        "audio": audio,
        "conformed": "; ".join(conformed) or "nothing, every clip matched",
    }
    status = run_result.OK
    summary = f"{len(lengths)} clip(s) joined into {total} frames, {seconds:.2f} s"
    if held:
        facts["held short"] = "; ".join(held) + " frame(s), to fit the clips beside them"
        status = run_result.WARNING
        summary = (
            f"{len(held)} transition(s) held short of {length} frames, since the clips beside "
            f"them are shorter"
        )
    published.publish_output(join.join_sheet(images, starts, shared, kind))
    run_result.publish(
        status=status,
        summary=summary,
        counts={"clips": len(lengths), "joins": len(shared), "frames": total, "seconds": round(seconds, 2)},
        facts=facts,
    )
