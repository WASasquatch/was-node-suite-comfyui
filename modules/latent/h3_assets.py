"""The pictures, clips and sounds a MiniMax H3 segment is built on, chained upstream first.

An asset names its segment, ``0`` for every one, and its part there. Pictures are
``[T, H, W, C]``, audio a ComfyUI ``AUDIO``.
"""

from __future__ import annotations

import math
from typing import NamedTuple

from . import h3_conditioning, h3_extend, h3_references

__all__ = [
    "AUDIO_PER_FRAME",
    "Asset",
    "EVERY_SEGMENT",
    "FIRST_FRAME",
    "KEYFRAME",
    "LAST_FRAME",
    "LONGEST_CLIP",
    "PINNING",
    "REFERENCE_AUDIO",
    "REFERENCE_CLIP",
    "REFERENCE_PICTURE",
    "REFERENCING",
    "ROLES",
    "CLOSING_KEY",
    "chained",
    "closing_index",
    "closing_moved",
    "collect",
    "cuts_into",
    "for_segment",
    "guide_length",
    "keyframes",
    "past_segment",
    "reference_parts",
    "resolved_index",
]

FIRST_FRAME = "first frame"
LAST_FRAME = "last frame"
KEYFRAME = "keyframe"
REFERENCE_PICTURE = "reference picture"
REFERENCE_CLIP = "reference clip"
REFERENCE_AUDIO = "reference audio"

#: Every part an asset may play, in menu order.
ROLES = (FIRST_FRAME, LAST_FRAME, KEYFRAME, REFERENCE_PICTURE, REFERENCE_CLIP, REFERENCE_AUDIO)

#: Roles that pin frames of the segment, and roles the segment's prompt references.
PINNING = (FIRST_FRAME, LAST_FRAME, KEYFRAME)
REFERENCING = (REFERENCE_PICTURE, REFERENCE_CLIP, REFERENCE_AUDIO)

#: The segment number that puts an asset on every segment.
EVERY_SEGMENT = 0

#: Frames a clip is read up to when no length is asked for, 15.08 seconds at 24 fps.
LONGEST_CLIP = 362

#: Audio latent steps per video frame.
AUDIO_PER_FRAME = h3_extend.AUDIO_LATENT_FPS / h3_extend.FPS

#: The key a keyframe block closing its segment carries.
CLOSING_KEY = h3_conditioning.CLOSING_KEY


class Asset(NamedTuple):
    """One picture, clip or sound and where it goes.

    Attributes:
        role: An entry of :data:`ROLES`.
        segment: The segment it belongs to, from 1, or :data:`EVERY_SEGMENT`.
        frame: Where a ``keyframe`` lands, counted in the segment's new frames from 0, or
            back from its end when negative.
        frames: The pictures, ``[T, H, W, C]``, or ``None``.
        audio: The sound, or ``None``.
        name: What it is called in a report.
        moment: True for a frame of the video being made, ``frame`` counted on the finished clip.
    """

    role: str
    segment: int
    frame: int
    frames: object
    audio: object
    name: str
    moment: bool = False


#: Bundle entry key holding the moments of the video a segment references, as
#: ``(frame, name, every)``.
MOMENTS_KEY = "moments"


def moments_for(assets: list[Asset], index: int) -> list[tuple[int, str, bool]]:
    """The moments of the video being made that one segment references.

    Args:
        assets: Every asset of the chain, in chain order.
        index: Segment number, from 0.

    Returns:
        ``(frame, name, every)`` per moment, ``every`` true for one shared by every segment.
    """
    return [(int(asset.frame), asset.name, int(asset.segment) == EVERY_SEGMENT)
            for asset in for_segment(assets, index) if asset.moment]


def collect(chain, named: str = "assets") -> list[Asset]:
    """The assets a chain carries, in chain order.

    Args:
        chain: What arrived on an ``assets`` input, or ``None``.
        named: The node and input it arrived on, for the error.

    Returns:
        Every asset, upstream first, or an empty list for ``None``.

    Raises:
        ValueError: The input carries something other than a chain of assets.
    """
    if chain is None:
        return []
    if isinstance(chain, Asset):
        return [chain]
    if not isinstance(chain, (tuple, list)) or not all(isinstance(each, Asset) for each in chain):
        raise ValueError(
            f"{named} holds a {type(chain).__name__}, and a chain of assets comes from "
            f"MiniMax H3 Asset. Wire that node's assets output into it"
        )
    return list(chain)


def chained(chain, asset: Asset, named: str = "assets") -> tuple[Asset, ...]:
    """A chain with one asset added at its end.

    Args:
        chain: What arrived on an ``assets`` input, or ``None``.
        asset: The asset to add.
        named: The node and input the chain arrived on, for the error.

    Returns:
        The longer chain.

    Raises:
        ValueError: The input carries something other than a chain of assets.
    """
    return (*collect(chain, named), asset)


def for_segment(assets: list[Asset], index: int) -> list[Asset]:
    """The assets one segment is built on, the shared ones first.

    Args:
        assets: Every asset of the chain, in chain order.
        index: Segment number, from 0.

    Returns:
        The assets on every segment, then the ones on this segment, each in chain order.
    """
    row = int(index) + 1
    shared = [asset for asset in assets if int(asset.segment) == EVERY_SEGMENT]
    own = [asset for asset in assets if int(asset.segment) == row]
    return shared + own


def past_segment(assets: list[Asset], count: int) -> list[Asset]:
    """The assets naming a segment the run does not have.

    Args:
        assets: Every asset of the chain.
        count: How many segments carry a prompt.

    Returns:
        The assets whose segment is past ``count``.
    """
    return [asset for asset in assets if int(asset.segment) > int(count)]


def guide_length(count: int) -> int:
    """Frames a pinned clip covers, one for a still.

    Args:
        count: Frames the asset holds.

    Returns:
        ``1`` under five frames, otherwise the longest ``17k + 5`` count at or below it.
    """
    count = int(count)
    if count < h3_extend.CLIP_LEAD:
        return 1
    return h3_extend.floor_overlap(count)


def resolved_index(frame: int, head: int, window: int, guide: int = 1, named: str = "") -> int:
    """Where a keyframe lands in the window its segment is sampled in.

    Args:
        frame: The asset's frame, from 0 in the segment's new frames or back from its end
            when negative.
        head: Frames at the start of the window that are carried rather than new.
        window: Frames the whole window holds.
        guide: Frames the asset covers from where it lands.
        named: The asset's name, for the error.

    Returns:
        A frame index into the window.

    Raises:
        ValueError: The asset would land outside the window.
    """
    frame, head, window, guide = int(frame), int(head), int(window), max(1, int(guide))
    index = head + frame if frame >= 0 else window + frame
    if index < 0 or index + guide > window:
        where = f"{guide} frames from frame {index}" if guide > 1 else f"frame {index}"
        raise ValueError(
            f"{named or 'the keyframe'} lands {where} of a {window} frame window, which runs "
            f"from frame 0 to {window - 1}. Set frame between 0 and "
            f"{max(0, window - head - guide)} to count from the segment's first new frame, or "
            f"between -{window} and -{guide} to count back from its end"
        )
    return index


def cuts_into(continuity: str, overlap: int, source: int, index: int,
              fallback: str = h3_extend.CONTINUITY[0]) -> bool:
    """Whether a segment cuts the clip it follows, which drops that clip's last lead frames.

    Args:
        continuity: The segment's continuity.
        overlap: Frames it asks to carry.
        source: Its source, as the row holds it.
        index: Its number, from 0.

    Returns:
        True for a cut, a reference, a handoff, a sound bridge or a return to an earlier
        segment; False for the opening segment and a carry from the one before.
    """
    if int(index) <= 0:
        return False
    named = h3_extend.CONTINUITY_RENAMED.get(continuity, continuity)
    if named in h3_extend.BRIDGING or named in h3_conditioning.CUT_LIKE or int(overlap) <= 0:
        return True
    try:
        return h3_conditioning.resolved_source(source, index) != int(index) - 1
    except ValueError:
        return False


def closing_index(window: int, guide: int, cut_after: bool) -> int:
    """Where a closing frame lands in its window.

    Args:
        window: Frames the window holds.
        guide: Frames the closing asset covers.
        cut_after: Whether the next segment cuts this one.

    Returns:
        The last place the frame survives the join, never before frame 0.
    """
    lead = h3_extend.CLIP_LEAD if cut_after else 0
    return max(0, int(window) - max(1, int(guide)) - lead)


def closing_moved(conditioning, frames: int) -> list:
    """Conditioning with every closing keyframe moved earlier.

    Args:
        conditioning: A ComfyUI conditioning, a list of ``[embedding, values]``.
        frames: Frames to move them by.

    Returns:
        A new conditioning; entries holding no closing keyframe are passed through.
    """
    moved = []
    for embedding, values in conditioning:
        blocks = values.get("minimax_keyframes")
        if blocks and any(block.get(CLOSING_KEY) for block in blocks):
            values = dict(values)
            values["minimax_keyframes"] = [
                {**block, "resolved_frame_index": max(0, int(block["resolved_frame_index"]) - int(frames))}
                if block.get(CLOSING_KEY) else block
                for block in blocks
            ]
        moved.append([embedding, values])
    return moved


def keyframes(vae, audio_vae, assets: list[Asset], width: int, height: int, head: int,
              window: int) -> tuple[list, list, list]:
    """The guide blocks a segment's pinning assets contribute.

    Args:
        vae: The H3 video VAE.
        audio_vae: The H3 audio VAE, or ``None`` where no asset carries sound.
        assets: The segment's assets, from :func:`for_segment`.
        width: Canvas width in pixels.
        height: Canvas height in pixels.
        head: Frames at the start of the window that are carried rather than new.
        window: Frames the whole window holds.

    Returns:
        ``(shown, blocks, notes)``: the stills the text encoder is shown, the
        ``minimax_keyframes`` blocks, and one line per block for the report.

    Raises:
        ValueError: An asset lands outside the window, carries sound with no audio VAE to
            encode it, or holds nothing to pin.
    """
    shown, blocks, notes = [], [], []
    for asset in assets:
        if asset.role not in PINNING:
            continue
        frames = asset.frames
        count = int(frames.shape[0]) if frames is not None else 0
        guide = guide_length(count) if count else 1
        if asset.role == FIRST_FRAME:
            index = int(head)
        elif asset.role == LAST_FRAME:
            index = closing_index(window, guide, False)
        else:
            index = resolved_index(asset.frame, head, window, guide, asset.name)
        block = {"resolved_frame_index": index}
        if asset.role == LAST_FRAME:
            block[CLOSING_KEY] = True
        if count:
            if guide == 1:
                onto = h3_conditioning.fitted if asset.role == FIRST_FRAME else h3_conditioning.covered
                picture = onto(frames[:1], width, height)
                if asset.role != KEYFRAME:
                    shown.append(picture)
                block["latent"] = vae.encode(picture)
            else:
                block["latent"] = vae.encode(
                    h3_conditioning.covered_batch(frames[:guide], width, height)
                )
        if asset.audio is not None:
            if audio_vae is None:
                raise ValueError(
                    f"{asset.name} carries sound to pin at frame {index}, and audio_vae is not "
                    f"wired, so there is nothing to encode it with. Wire the H3 audio VAE into "
                    f"audio_vae on MiniMax H3 Conditioning"
                )
            latent, steps = h3_references.audio_latent(audio_vae, asset.audio)
            most = int(math.floor(h3_extend.audio_span(window) - AUDIO_PER_FRAME * index))
            if most < 1:
                raise ValueError(
                    f"{asset.name} pins sound at frame {index}, past the end of the window's "
                    f"soundtrack. Move it earlier"
                )
            if steps > most:
                latent = latent[..., :most].clone()
            block["audio_latent"] = latent
        if "latent" not in block and "audio_latent" not in block:
            raise ValueError(
                f"{asset.name} is set to `{asset.role}` and holds neither a picture nor a "
                f"sound. Pick a file on its MiniMax H3 Asset node, or wire an image, video or "
                f"audio into it"
            )
        blocks.append(block)
        what = ("a still" if guide == 1 else f"{guide} frames") if count else "sound"
        if count and asset.audio is not None:
            what += " with sound"
        notes.append(f"{asset.role} at frame {index}, {what}: {asset.name}")
    return shown, blocks, notes


def reference_parts(vae, audio_vae, assets: list[Asset], canvas_width: int, canvas_height: int,
                    longest: int, size: str, cache: dict) -> tuple[list, list, list]:
    """A segment's referencing assets, each encoded once across the run.

    Args:
        vae: The H3 video VAE.
        audio_vae: The H3 audio VAE, or ``None`` where no asset carries sound.
        assets: The segment's assets, from :func:`for_segment`.
        canvas_width: The clip's width in pixels.
        canvas_height: The clip's height in pixels.
        longest: Frames the longest segment runs for, which no clip runs past.
        size: An entry from :data:`h3_references.IMAGE_SIZES`.
        cache: Encoded assets by identity, shared across segments.

    Returns:
        ``(pictures, videos, sounds)``, as :func:`h3_references.assemble` takes them.

    Raises:
        ValueError: An asset holds nothing its role can use, or sound arrives with no audio
            VAE.
    """
    pictures, videos, sounds = [], [], []
    for asset in assets:
        if asset.role not in REFERENCING or asset.moment:
            continue
        key = id(asset)
        if key not in cache:
            if asset.role == REFERENCE_PICTURE:
                if asset.frames is None:
                    raise ValueError(
                        f"{asset.name} is set to `reference picture` and holds no picture. "
                        f"Pick an image file on its MiniMax H3 Asset node, or wire one in"
                    )
                cache[key] = h3_references.encode_picture(
                    vae, asset.frames[:1], canvas_width, canvas_height, size
                )
            elif asset.role == REFERENCE_CLIP:
                if asset.frames is None:
                    raise ValueError(
                        f"{asset.name} is set to `reference clip` and holds no frames. Pick a "
                        f"video file on its MiniMax H3 Asset node, or wire a video in"
                    )
                cache[key] = h3_references.encode_video(
                    vae, audio_vae, asset.frames, asset.audio, longest, asset.name
                )
            else:
                if asset.audio is None:
                    raise ValueError(
                        f"{asset.name} is set to `reference audio` and holds no sound. Pick a "
                        f"sound file on its MiniMax H3 Asset node, or wire an audio in"
                    )
                cache[key] = h3_references.encode_sound(audio_vae, asset.audio, asset.name)
        encoded = cache[key]
        if asset.role == REFERENCE_PICTURE:
            pictures.append(encoded)
        elif asset.role == REFERENCE_CLIP:
            videos.append(encoded)
        else:
            sounds.append(encoded)
    return pictures, videos, sounds
