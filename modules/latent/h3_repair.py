"""Redrawing a stretch of a finished MiniMax H3 clip in a window cut from it.

Spans are half open. Rows are latent tokens, steps are audio latent frames at 40 a second,
frames run at 24 a second.
"""

from __future__ import annotations

import math

from . import h3_assets, h3_conditioning, h3_decode, h3_extend

__all__ = [
    "AUDIO_BLOCKS",
    "CONTEXT_FRAMES",
    "REPAIR_KEY",
    "closes_on_cut",
    "describe",
    "frame_of",
    "keyframes_moved",
    "plan",
    "rejoin_rows",
    "segment_holding",
    "segment_origin",
    "splice",
    "trimmed",
    "window",
    "written_rows",
    "written_steps",
]

#: Window key recording where a repair window sits on its clip and what it redraws.
REPAIR_KEY = "h3_repair"

#: Frames of context either side of the span by default, two blocks.
CONTEXT_FRAMES = 2 * h3_extend.CLIP_FRAMES

#: Blocks between block starts that land on a whole audio step.
AUDIO_BLOCKS = 3

#: Frames each row after a block's first one decodes to.
ROW_FRAMES = 4

#: Slack on a time converted to frames, so a typed boundary lands on its own frame.
EPSILON = 1e-6


def frame_of(row: int) -> int:
    """The first frame a latent row decodes to, which for a row count is the frames it covers.

    Args:
        row: A row, from 0.

    Returns:
        A frame index.
    """
    block, offset = divmod(int(row), h3_extend.CLIP_TOKENS)
    return block * h3_extend.CLIP_FRAMES + (0 if offset == 0 else 1 + ROW_FRAMES * (offset - 1))


def _unit(value: float) -> float:
    """A value held between 0.0 and 1.0."""
    return max(0.0, min(1.0, float(value)))


def _rise(step: int, steps: int) -> float:
    """One step of a half cosine opening over ``steps``, never 0."""
    return 0.5 - 0.5 * math.cos(math.pi * (step + 1) / (steps + 1))


def trimmed(latent: dict) -> dict:
    """The rows and audio a cut trimmed off each scene's end, by the row the cut lands on.

    Args:
        latent: A joined clip.

    Returns:
        ``{row: (video, audio)}`` for every cut that recorded what it trimmed.
    """
    ends = h3_extend.segment_ends(latent) or []
    found = {}
    for index, tail in (latent.get(h3_extend.TAILS_KEY) or {}).items():
        index = int(index)
        if 0 <= index < len(ends) and ends[index] % h3_extend.CLIP_FRAMES == 0:
            found[ends[index] // h3_extend.CLIP_FRAMES * h3_extend.CLIP_TOKENS] = tail
    return found


def plan(tokens: int, steps: int, start_seconds: float, end_seconds: float, picture: float,
         sound: float, context_frames: int, release: int, feather: float,
         scenes: list, trimmed_rows: dict | None = None) -> dict:
    """Where a repair window sits on its clip and what it redraws.

    Args:
        tokens: Rows the clip holds.
        steps: Audio steps the clip holds.
        start_seconds: Where the stretch starts.
        end_seconds: Where the stretch ends.
        picture: Strength the picture in the stretch is redrawn at, 0.0 to 1.0.
        sound: Strength the sound in the stretch is redrawn at, 0.0 to 1.0.
        context_frames: Frames of the clip either side of the stretch the window holds.
        release: Audio steps the redrawn sound fades over beyond the stretch.
        feather: Strength the block either side of the stretch is redrawn at.
        scenes: ``(start, stop)`` row spans of the clip's scenes, covering it.
        trimmed_rows: Rows a cut recorded trimming, by the row it lands on.

    Returns:
        The record a window carries under :data:`REPAIR_KEY`.

    Raises:
        ValueError: Both strengths are 0, the stretch is empty or outside the clip, or it
            crosses a scene cut.
    """
    tokens, steps = int(tokens), int(steps)
    total = frame_of(tokens)
    duration = total / h3_extend.FPS
    begin = max(0.0, float(start_seconds))
    end = min(float(end_seconds), duration)
    strength, heard = _unit(picture), _unit(sound)
    if strength <= 0.0 and heard <= 0.0:
        raise ValueError(
            "picture_strength and sound_strength are both 0, so the window would redraw "
            "nothing. Raise picture_strength to redraw the picture, sound_strength to redraw "
            "the sound, or both"
        )
    if begin >= duration:
        raise ValueError(
            f"start_seconds is {begin:g} and the clip runs {duration:.2f} s ({total} frames). "
            f"Set start_seconds below {duration:.2f}"
        )
    if end <= begin:
        raise ValueError(
            f"end_seconds ({float(end_seconds):g}) is not after start_seconds ({begin:g}). Set "
            f"end_seconds later than start_seconds"
        )

    size, rows_per = h3_extend.CLIP_FRAMES, h3_extend.CLIP_TOKENS
    first = int(math.floor(begin * h3_extend.FPS + EPSILON)) // size
    last = max(first + 1, math.ceil((end * h3_extend.FPS - EPSILON) / size))
    opened, closed = first * rows_per, min(last * rows_per, tokens)

    low, high = 0, tokens
    for start, stop in scenes or [(0, tokens)]:
        if start <= opened < stop:
            low, high = int(start), int(stop)
            break
    if closed > high:
        cut = frame_of(high)
        raise ValueError(
            f"the repair from {begin:.2f} s to {end:.2f} s widens to frames {frame_of(opened)} to "
            f"{frame_of(closed) - 1} and crosses the scene cut at {cut / h3_extend.FPS:.2f} s "
            f"(frame {cut}), where a new scene starts. Split it in two: one repair ending at "
            f"{cut / h3_extend.FPS:.2f} s and one starting there"
        )

    blocks = h3_extend.nearest_clips(context_frames) if int(context_frames) > 0 else 0
    floor_block = low // rows_per
    head = max(floor_block, first - blocks)
    if blocks and head % AUDIO_BLOCKS:
        # Widened back to a block whose start lands on a whole audio step.
        grown = head - head % AUDIO_BLOCKS
        if grown >= floor_block:
            head = grown
    start = head * rows_per

    after = (last + blocks) * rows_per + h3_extend.TOKEN_LEAD
    cut, lead = None, 0
    if high >= tokens:
        stop = min(tokens, after)
    elif after <= high:
        stop = after
    else:
        stop, cut = high, high
        lead = min(h3_extend.TOKEN_LEAD, int((trimmed_rows or {}).get(high, 0)))

    edge = min(_unit(feather), strength) if strength > 0.0 else 0.0
    edges = []
    if edge > 0.0:
        if opened - rows_per >= start:
            edges.append([opened - rows_per, opened])
        if closed + rows_per <= stop:
            edges.append([closed, closed + rows_per])
    if not edges:
        edge = 0.0

    audio_start = h3_extend.audio_span(frame_of(start))
    offset = (audio_start - frame_of(start) * h3_extend.AUDIO_LATENT_FPS / h3_extend.FPS) / (
        h3_extend.AUDIO_LATENT_FPS)
    length = h3_extend.audio_span(frame_of(stop - start + lead))
    scene_end = steps if high >= tokens else min(steps, h3_extend.audio_span(frame_of(high)))
    held = max(0, min(audio_start + length, scene_end) - audio_start)

    sound_from = 0 if opened == 0 else h3_extend.audio_span(frame_of(opened))
    sound_to = steps if closed >= tokens else h3_extend.audio_span(frame_of(closed))
    sound_from = max(sound_from, audio_start)
    sound_to = max(sound_from, min(sound_to, audio_start + held))
    ramps = [0, 0]
    if heard > 0.0:
        fade = max(0, int(release))
        ramps = [min(fade, sound_from - audio_start), min(fade, audio_start + held - sound_to)]

    return {
        "tokens": tokens,
        "steps": steps,
        "start": start,
        "rows": stop - start,
        "lead": lead,
        "cut": cut,
        "picture": [opened, closed],
        "picture_strength": strength,
        "feather": edges,
        "feather_strength": edge,
        "audio_start": audio_start,
        "audio_steps": held,
        "audio_length": length,
        "sound": [sound_from, sound_to],
        "sound_strength": heard,
        "ramps": ramps,
        "offset": offset,
    }


def window(video, audio, record: dict, lead=None) -> dict:
    """The repair window cut from the clip, carrying its noise mask and its record.

    Args:
        video: The clip's video half, ``[1, 24, T, H, W]``.
        audio: The clip's audio half, ``[1, 32, 2, L]``.
        record: What :func:`plan` answered.
        lead: ``(video, audio)`` the cut the window closes on trimmed, or ``None``.

    Returns:
        A joint latent holding the clip's own rows and sound, masked by strength.
    """
    import torch

    start, rows, extra = record["start"], record["rows"], record["lead"]
    picture = video[:, :, start:start + rows]
    if extra and lead is not None:
        picture = torch.cat([picture, lead[0][:, :, :extra].to(picture)], dim=2)
    picture = picture.clone()

    begin, held, length = record["audio_start"], record["audio_steps"], record["audio_length"]
    parts = [audio[..., begin:begin + held]]
    have = parts[0].shape[-1]
    if have < length and extra and lead is not None:
        parts.append(lead[1][..., :length - have].to(audio))
        have += parts[-1].shape[-1]
    if have < length:
        # The last step held repeats out to the window's length.
        source = torch.cat(parts, dim=-1) if have else audio[..., max(0, begin - 1):begin + 1]
        parts.append(source[..., -1:].expand(*source.shape[:-1], length - have))
    sound = torch.cat(parts, dim=-1)[..., :length].clone()

    video_mask = torch.zeros(
        [picture.shape[0], 1, picture.shape[2], picture.shape[3], picture.shape[4]],
        device=picture.device, dtype=torch.float32,
    )
    opened, closed = record["picture"]
    video_mask[:, :, opened - start:closed - start] = record["picture_strength"]
    for low, high in record["feather"]:
        video_mask[:, :, low - start:high - start] = record["feather_strength"]

    audio_mask = torch.zeros(
        [sound.shape[0], 1, sound.shape[2], length], device=sound.device, dtype=torch.float32,
    )
    heard = record["sound_strength"]
    if heard > 0.0:
        low, high = (step - begin for step in record["sound"])
        before, after = record["ramps"]
        audio_mask[..., low:high] = heard
        for step in range(before):
            audio_mask[..., low - before + step] = heard * _rise(step, before)
        for step in range(after):
            audio_mask[..., high + step] = heard * _rise(after - 1 - step, after)

    out = h3_extend.join(picture, sound)
    out["noise_mask"] = h3_extend.join(video_mask, audio_mask)["samples"]
    out[REPAIR_KEY] = dict(record)
    return out


def written_rows(record: dict) -> list:
    """The clip rows a repair writes back, in order.

    Args:
        record: A window's :data:`REPAIR_KEY` record.

    Returns:
        ``[start, stop]`` row spans, neighbouring spans merged.
    """
    spans = []
    if record["picture_strength"] > 0.0:
        spans.append(list(record["picture"]))
    if record["feather_strength"] > 0.0:
        spans.extend(list(span) for span in record["feather"])
    merged = []
    for low, high in sorted(spans):
        if merged and low <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], high)
        else:
            merged.append([low, high])
    return merged


def written_steps(record: dict) -> list:
    """The clip audio steps a repair writes back.

    Args:
        record: A window's :data:`REPAIR_KEY` record.

    Returns:
        ``[start, stop]``, empty where the sound is kept.
    """
    if record["sound_strength"] <= 0.0:
        return [0, 0]
    low, high = record["sound"]
    before, after = record["ramps"]
    return [low - before, high + after]


def splice(latent: dict, sampled: dict) -> dict:
    """The clip with a sampled repair window's redrawn rows and steps written back.

    Args:
        latent: The clip the window was cut from.
        sampled: The sampled window, carrying :data:`REPAIR_KEY`.

    Returns:
        The repaired clip, every other row and step and every record of the clip kept.

    Raises:
        ValueError: The window carries no record, or it was cut from another clip, or its
            shape is not the one the record names.
    """
    record = sampled.get(REPAIR_KEY) if isinstance(sampled, dict) else None
    if not record:
        raise ValueError(
            "the sampled latent carries no repair record, so there is nothing to write back. "
            "Wire the sampler's output for the window H3 Repair Window opened into sampled"
        )
    video, audio = h3_extend.split(latent)
    new_video, new_audio = h3_extend.split(sampled)
    if video.shape[2] != record["tokens"] or audio.shape[-1] != record["steps"]:
        raise ValueError(
            f"the window was cut from a clip of {record['tokens']} tokens and {record['steps']} "
            f"audio steps, and latent holds {video.shape[2]} and {audio.shape[-1]}. Wire the clip "
            f"that went into H3 Repair Window into latent"
        )
    rows = record["rows"] + record["lead"]
    if (new_video.shape[2] != rows or new_audio.shape[-1] != record["audio_length"]
            or tuple(new_video.shape[3:]) != tuple(video.shape[3:])):
        raise ValueError(
            f"the sampled window holds {new_video.shape[2]} tokens and {new_audio.shape[-1]} "
            f"audio steps at {new_video.shape[4]}x{new_video.shape[3]}, where H3 Repair Window "
            f"opened {rows} and {record['audio_length']} at {video.shape[4]}x{video.shape[3]}. "
            f"Wire the sampler's output for that window into sampled"
        )
    start, begin = record["start"], record["audio_start"]
    out_video, out_audio = video.clone(), audio.clone()
    for low, high in written_rows(record):
        out_video[:, :, low:high] = new_video[:, :, low - start:high - start].to(out_video)
    low, high = written_steps(record)
    if high > low:
        out_audio[..., low:high] = new_audio[..., low - begin:high - begin].to(out_audio)
    out = dict(latent)
    out.update(h3_extend.join(out_video, out_audio))
    out.pop("noise_mask", None)
    out.pop(REPAIR_KEY, None)
    return out


def describe(record: dict) -> str:
    """What a repair redraws and the window it sits in, for a report.

    Args:
        record: A window's :data:`REPAIR_KEY` record.

    Returns:
        One line naming the frames, seconds, tokens and steps, each at its strength.
    """
    fps, rate = h3_extend.FPS, h3_extend.AUDIO_LATENT_FPS
    opened, closed = record["picture"]
    parts = []
    if record["picture_strength"] > 0.0:
        first, after = frame_of(opened), frame_of(closed)
        parts.append(
            f"picture frames {first}-{after - 1} ({first / fps:.2f}-{after / fps:.2f} s, tokens "
            f"{opened}-{closed - 1}) at {record['picture_strength']:.2f}"
        )
        if record["feather_strength"] > 0.0:
            spans = " and ".join(f"{frame_of(low)}-{frame_of(high) - 1}"
                                 for low, high in record["feather"])
            parts.append(f"edges at {record['feather_strength']:.2f} over frames {spans}")
    else:
        parts.append("picture kept")
    if record["sound_strength"] > 0.0:
        low, high = record["sound"]
        before, after = record["ramps"]
        ramp = f"ramp {before}" if before == after else f"ramps {before} before and {after} after"
        parts.append(
            f"sound {low / rate:.2f}-{high / rate:.2f} s (steps {low}-{high - 1}) at "
            f"{record['sound_strength']:.2f}, {ramp}"
        )
    else:
        parts.append("sound kept")
    start = record["start"]
    frames = frame_of(record["rows"] + record["lead"])
    before = frame_of(opened) - frame_of(start)
    after = (start + record["rows"] - closed) // h3_extend.CLIP_TOKENS * h3_extend.CLIP_FRAMES
    context = (f"{before} frames of context each side" if before == after
               else f"{before} frames of context before and {after} after")
    return f"{', '.join(parts)}; window {frames} frames from frame {frame_of(start)}, {context}"


def segment_holding(latent: dict, first: int, last: int) -> int | None:
    """The segment of a joined clip holding most of a stretch of frames.

    Args:
        latent: A joined clip.
        first: The stretch's first frame.
        last: The frame after its last.

    Returns:
        A segment number from 0, or ``None`` where the clip records no segment ends.
    """
    ends = h3_extend.segment_ends(latent)
    if not ends:
        return None
    best, most, begin = 0, -1, 0
    for index, end in enumerate(ends):
        shared = min(end, int(last)) - max(begin, int(first))
        if shared > most:
            best, most = index, shared
        begin = end
    return best


def _scene_rows(latent: dict) -> set:
    """The row every scene of a joined clip opens on."""
    return {start for start, _ in h3_decode.scenes(latent)}


def _continuity(entry: dict) -> str:
    """A bundled segment's continuity, H3 Extend Window's default where it names none."""
    chosen = entry.get("continuity")
    chosen = h3_extend.CONTINUITY_RENAMED.get(chosen, chosen)
    return chosen if chosen in h3_extend.CONTINUITY else h3_extend.CONTINUITY[0]


def rejoin_rows(latent: dict, prompts, index: int) -> int | None:
    """Window rows a return to an earlier segment dropped ahead of its join.

    Args:
        latent: The joined clip.
        prompts: The bundle the clip was sampled from.
        index: Segment number, from 1.

    Returns:
        A multiple of five rows, or ``None`` where the segment continued the one before it.
    """
    index = int(index)
    try:
        picked = h3_conditioning.resolved_source(h3_conditioning.source_of(prompts, index), index)
    except ValueError:
        return None
    ends = h3_extend.segment_ends(latent) or []
    if picked == index - 1 or picked >= len(ends):
        return None
    end = ends[picked]
    tail = (latent.get(h3_extend.TAILS_KEY) or {}).get(picked)
    if tail is not None and end % h3_extend.CLIP_FRAMES == 0:
        rows = end // h3_extend.CLIP_FRAMES * h3_extend.CLIP_TOKENS + int(tail[0].shape[2])
    else:
        rows = h3_extend.clip_end(h3_extend.tokens_for(end))
    entry = prompts[index]
    overlap = h3_extend.snap_overlap(int(entry.get("overlap", 0)))
    head = min(h3_extend.tokens_for(overlap), rows)
    skip = h3_extend.rejoin_skip(h3_extend.seen_rows(end), rows, head)
    span = h3_extend.tokens_for(h3_extend.window_frames(overlap, int(entry.get("frames", 0))))
    return max(0, min(skip, (span - 1) // h3_extend.CLIP_TOKENS * h3_extend.CLIP_TOKENS))


def segment_origin(latent: dict, prompts, index: int) -> int | None:
    """The clip frame a segment's window opened on, where its keyframes count from.

    Args:
        latent: The joined clip.
        prompts: The bundle the clip was sampled from.
        index: Segment number, from 0.

    Returns:
        A frame, before the segment's scene where its window opened under carried sound, or
        ``None`` where the clip records no start for that segment.
    """
    index = int(index)
    if index <= 0:
        return 0
    ends = h3_extend.segment_ends(latent)
    if not ends or index >= len(ends) or not prompts or index >= len(prompts):
        return None
    begin = ends[index - 1]
    entry = prompts[index]
    overlap = int(entry.get("overlap", 0))
    continuity = h3_extend.geometry_of(_continuity(entry), entry.get("sound", "auto"))
    size, rows = h3_extend.CLIP_FRAMES, h3_extend.CLIP_TOKENS
    fresh = begin % size == 0 and begin // size * rows in _scene_rows(latent)
    if not fresh:
        return begin - (h3_extend.snap_overlap(overlap) if overlap > 0 else 0)
    if continuity in h3_extend.BRIDGING:
        return begin - h3_extend.bridge_frames(overlap)
    if continuity in ("carry", "refresh") and overlap > 0:
        skip = rejoin_rows(latent, prompts, index)
        if skip is not None:
            return begin - skip // rows * size
    return begin


def closes_on_cut(latent: dict, prompts, index: int) -> bool:
    """Whether the segment after one cut into it, which moved its closing keyframe earlier.

    Args:
        latent: The joined clip.
        prompts: The bundle the clip was sampled from.
        index: Segment number, from 0.

    Returns:
        True where the next segment opened a fresh scene.
    """
    index = int(index)
    ends = h3_extend.segment_ends(latent)
    if ends:
        if index >= len(ends) - 1:
            return False
        end = ends[index]
        size = h3_extend.CLIP_FRAMES
        return end % size == 0 and end // size * h3_extend.CLIP_TOKENS in _scene_rows(latent)
    following = index + 1
    if not prompts or following >= len(prompts):
        return False
    entry = prompts[following]
    return h3_assets.cuts_into(
        _continuity(entry), int(entry.get("overlap", 0)),
        h3_conditioning.source_of(prompts, following), following,
    )


def _guide(block: dict) -> int:
    """Frames a keyframe block pins from where it lands."""
    latent = block.get("latent")
    rows = int(latent.shape[2]) if getattr(latent, "ndim", 0) == 5 else 1
    return 1 if rows <= 1 else h3_extend.frames_for(rows)


def keyframes_moved(conditioning, shift: int | None, frames: int, steps: int) -> tuple:
    """Conditioning with every keyframe moved ``shift`` frames, those outside a window dropped.

    Args:
        conditioning: A ComfyUI conditioning, a list of ``[embedding, values]``.
        shift: Frames to move each keyframe by, or ``None`` to drop them all.
        frames: Frames the window holds.
        steps: Audio steps the window holds.

    Returns:
        ``(conditioning, kept, dropped)``: the new conditioning and the keyframes kept and
        dropped by its first entry carrying any.
    """
    moved, counted = [], None
    for embedding, values in conditioning:
        blocks = values.get("minimax_keyframes")
        if not blocks:
            moved.append([embedding, values])
            continue
        kept = []
        for block in blocks:
            if shift is None:
                continue
            index = int(block["resolved_frame_index"]) + int(shift)
            if index < 0 or index + _guide(block) > int(frames):
                continue
            block = {**block, "resolved_frame_index": index}
            sound = block.get("audio_latent")
            if sound is not None:
                most = int(math.floor(int(steps) - h3_assets.AUDIO_PER_FRAME * index))
                if most < 1:
                    block.pop("audio_latent")
                    if block.get("latent") is None:
                        continue
                elif sound.shape[-1] > most:
                    block["audio_latent"] = sound[..., :most].clone()
            kept.append(block)
        if counted is None:
            counted = (len(kept), len(blocks) - len(kept))
        values = dict(values)
        if kept:
            values["minimax_keyframes"] = kept
        else:
            values.pop("minimax_keyframes")
        moved.append([embedding, values])
    kept_count, dropped_count = counted or (0, 0)
    return moved, kept_count, dropped_count
