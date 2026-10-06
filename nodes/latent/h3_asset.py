"""One picture, clip or sound for a MiniMax H3 segment, and the part it plays there."""

from __future__ import annotations

import os

from comfy_api.latest import io

from ...modules import log
from ...modules.archive import picks
from ...modules.compat.types import H3_ASSETS
from ...modules.constants import ALLOWED_EXT
from ...modules.interface import preview, run_result
from ...modules.latent import h3_assets, h3_conditioning, h3_extend
from ...modules.media import reader
from ...modules.util import file_listing, sandbox

logger = log.get_logger("nodes.h3_asset")

NODE_NAME = "MiniMax H3 Asset"

#: The menu entry that reads the wired inputs instead of a file.
WIRED = "(wired input)"

#: The menu entry that reads a frame of the video the run is making, at ``frame``.
MOMENT = "(the video being made)"

#: Sound files the menu offers.
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".aiff", ".wma")

#: Every file the menu offers.
EXTENSIONS = (*ALLOWED_EXT, *reader.VIDEO_EXTENSIONS, *AUDIO_EXTENSIONS)

#: How many files the menu offers.
MAX_OPTIONS = 5000

FILE_HINT = (
    "Which picture, clip or sound to read, as `cast/alice.png [input]` or "
    "`takes/shot_03.mp4 [output]`. `(wired input)` reads the image, video and audio "
    "sockets instead, and a wired socket always wins over the file."
)

ROLE_HINT = (
    "What this does for its segment. `first frame` = the segment opens on it; `last frame` "
    "= closes on it; `keyframe` = pinned at `frame`, as a still, a clip, a sound, or a clip "
    "with its sound; `reference picture`, `reference clip` and `reference audio` = named "
    "`<Picture N>`, `<Video N>` and `<Audio N>` in that segment's prompt."
)

SEGMENT_HINT = (
    "Which segment it belongs to, as `1` for the opening segment or `3` for Segment 3. `0` "
    "= every segment, as a cast picture the whole run references."
)

FRAME_HINT = (
    "Where a `keyframe` lands, counted in the segment's new frames at 24 fps: `0` = its "
    "first new frame, `48` = two seconds in, `-1` = its last frame. With "
    "`(the video being made)`, the frame of the finished video to reference, as `120`."
)

CLIP_START_HINT = "Seconds into the file a clip or sound starts, as `0` or `12.5`."

CLIP_SECONDS_HINT = (
    "Seconds of the file to read, as `5`, or `0` for the first 362 frames of a clip, about "
    "15 seconds, and the whole of a sound. A reference clip is cut to the longest segment "
    "either way."
)

IMAGE_HINT = "A picture, or a batch of frames at 24 fps for a clip, in place of a file."
VIDEO_HINT = "A video in place of a file: its frames, brought to 24 fps, and its sound."
AUDIO_HINT = "A sound in place of a file, or the soundtrack of the wired image frames."

ASSETS_HINT = (
    "The assets before this one, from another MiniMax H3 Asset's assets output. This asset "
    "joins the end of that chain, so a run of any length is one wire into the next node."
)


def file_labels() -> list[str]:
    """The menu's entries, `(wired input)` first.

    Returns:
        Every picture, clip and sound under the folders the pack may read, as labels.
    """
    try:
        found = file_listing.labels(EXTENSIONS, file_listing.ROOTS, MAX_OPTIONS)
    except Exception as error:
        logger.debug("the file listing could not be read: %s", error)
        found = []
    return [WIRED, MOMENT, *found]


def chosen_path(label: str) -> str | None:
    """The file one menu entry names, resolved inside a permitted read root.

    Args:
        label: The widget's value, as ``cast/alice.png [input]``.

    Returns:
        The absolute path, or None for `(wired input)` and for an entry naming nothing there.

    Raises:
        PathNotAllowed: The file resolved outside every permitted read root.
    """
    chosen = str(label or "").strip()
    if not chosen or chosen in (WIRED, MOMENT):
        return None
    if sandbox.names_another_host(chosen):
        raise sandbox.PathNotAllowed(
            f"`{chosen}` names another machine. Pick a file from the menu"
        )
    found = file_listing.resolve(chosen, EXTENSIONS, file_listing.ROOTS)
    if found is None:
        try:
            found = sandbox.annotated_path(chosen)
        except Exception as error:
            logger.debug("`%s` could not be resolved through folder_paths: %s", chosen, error)
    if found is None:
        return None
    return str(sandbox.resolve_read(found))


def at_model_rate(frames, rate: float):
    """A batch of frames brought to the model's 24 fps by picking the nearest frame.

    Args:
        frames: A ``[T, H, W, C]`` tensor, or ``None``.
        rate: Frames per second the batch plays at.

    Returns:
        The batch at 24 fps, the same object where nothing changes.
    """
    if frames is None or int(frames.shape[0]) < 2:
        return frames
    rate = float(rate) if rate and rate > 0 else float(h3_extend.FPS)
    if abs(rate - h3_extend.FPS) < 1e-6:
        return frames
    count = int(frames.shape[0])
    wanted = max(1, int(round(count * h3_extend.FPS / rate)))
    indices = [min(count - 1, int(round(index * rate / h3_extend.FPS))) for index in range(wanted)]
    return frames[indices]


def read_file(path: str, start: float, seconds: float) -> tuple:
    """What one file holds.

    Args:
        path: The file, already contained.
        start: Seconds into the file a clip or sound starts.
        seconds: Seconds to read, ``0`` for the default length.

    Returns:
        ``(frames, audio, rate)``: the pictures or ``None``, the sound or ``None``, and the
        frames per second the pictures play at.

    Raises:
        ValueError: The file is of no kind the node reads, or holds nothing it can decode.
        DependencyError: A clip or a sound is asked for and PyAV is not installed.
    """
    suffix = os.path.splitext(path)[1].lower()
    name = os.path.basename(path)
    if suffix in ALLOWED_EXT:
        import numpy as np
        import torch
        from PIL import Image, ImageOps

        import node_helpers

        with node_helpers.pillow(Image.open, path) as opened:
            picture = node_helpers.pillow(ImageOps.exif_transpose, opened).convert("RGB")
            frames = torch.from_numpy(np.array(picture).astype(np.float32) / 255.0)[None]
        return frames, None, float(h3_extend.FPS)
    if suffix in reader.VIDEO_EXTENSIONS:
        source = reader.probe(path)
        rate = float(source.fps) if source.fps > 0 else reader.DEFAULT_RATE
        first = max(0, int(round(float(start) * rate)))
        span = (float(seconds) if float(seconds) > 0
                else h3_assets.LONGEST_CLIP / h3_extend.FPS)
        last = first + max(1, int(round(span * rate))) - 1
        clip = reader.read(path, start=first, end=last, target_fps=float(h3_extend.FPS))
        if int(clip.images.shape[0]) < 1:
            raise ValueError(f"{name} holds no frame that could be decoded")
        return clip.images, clip.audio, float(clip.fps)
    if suffix in AUDIO_EXTENSIONS:
        length = reader.audio_length(path)
        span = float(seconds) if float(seconds) > 0 else max(0.0, length - float(start))
        audio = reader.audio_span(path, float(start), span) if span > 0 else None
        if audio is None:
            raise ValueError(
                f"{name} holds no sound that could be decoded"
                + (f" from {start:g}s" if float(start) > 0 else "")
            )
        return None, audio, float(h3_extend.FPS)
    raise ValueError(
        f"{name} is not a picture, a clip or a sound this node reads. Pick a file with one "
        f"of these extensions: {', '.join(EXTENSIONS)}"
    )


def describe(asset: h3_assets.Asset) -> tuple[str, dict, dict]:
    """What an asset holds, for the report and the panel.

    Args:
        asset: The asset.

    Returns:
        ``(summary, counts, facts)``.
    """
    where = "every segment" if asset.segment == h3_assets.EVERY_SEGMENT else f"segment {asset.segment}"
    parts, counts, facts = [], {}, {"role": asset.role, "segment": where}
    if asset.frames is not None:
        count = int(asset.frames.shape[0])
        size = f"{int(asset.frames.shape[2])}x{int(asset.frames.shape[1])}"
        counts["frames"] = count
        facts["picture"] = f"{size}, {count} frame{'s' if count != 1 else ''}"
        if count > 1:
            parts.append(f"{count} frames ({h3_conditioning.duration_of(count):g}s) at {size}")
        else:
            parts.append(f"a still at {size}")
    if asset.audio is not None:
        seconds = asset.audio["waveform"].shape[-1] / max(1, int(asset.audio["sample_rate"]))
        counts["seconds"] = round(seconds, 2)
        facts["sound"] = f"{seconds:.2f}s at {int(asset.audio['sample_rate'])} Hz"
        parts.append(f"{seconds:.1f}s of sound")
    placed = f" at frame {asset.frame}" if asset.role == h3_assets.KEYFRAME else ""
    summary = f"{asset.role}{placed} for {where}: {asset.name}, {' with '.join(parts)}"
    return summary, counts, facts


class MiniMaxH3Asset(io.ComfyNode):
    """Pick a picture, clip or sound and say which segment it belongs to and what it does there."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASMiniMaxH3Asset",
            display_name=NODE_NAME,
            search_aliases=[
                "WASMiniMaxH3Asset",
                NODE_NAME,
                "minimax h3",
                "h3 keyframe",
                "h3 reference",
                "prompt timeline asset",
                "asset chain",
                "first frame",
                "last frame",
            ],
            category="WAS Suite/Latent/Video",
            description=(
                "One picture, clip or sound for a MiniMax H3 run, with the segment it belongs "
                "to and the part it plays there: the frame a segment opens or closes on, a "
                "keyframe pinned at any frame of it, or a picture, clip or sound its prompt "
                "references as `<Picture N>`, `<Video N>` or `<Audio N>`. Pick a file from "
                "ComfyUI's folders, or wire an image, a video or an audio in. Chain as many as "
                "the run needs, each into the next one's assets, and wire the last into assets "
                "on MiniMax H3 Conditioning. The Prompt Timeline window on that node places "
                "them for you."
            ),
            inputs=[
                H3_ASSETS.Input("assets", optional=True, tooltip=ASSETS_HINT),
                io.Combo.Input("file", options=file_labels(), tooltip=FILE_HINT),
                io.Combo.Input(
                    "role", options=list(h3_assets.ROLES), default=h3_assets.FIRST_FRAME,
                    tooltip=ROLE_HINT,
                ),
                io.Int.Input(
                    "segment", default=1, min=h3_assets.EVERY_SEGMENT,
                    max=h3_conditioning.MAX_ROWS, tooltip=SEGMENT_HINT,
                ),
                io.Int.Input("frame", default=0, min=-3600, max=3600, tooltip=FRAME_HINT),
                io.Float.Input(
                    "clip_start", default=0.0, min=0.0, max=36000.0, step=0.1, optional=True,
                    tooltip=CLIP_START_HINT,
                ),
                io.Float.Input(
                    "clip_seconds", default=0.0, min=0.0, max=600.0, step=0.1, optional=True,
                    tooltip=CLIP_SECONDS_HINT,
                ),
                io.Image.Input("image", optional=True, tooltip=IMAGE_HINT),
                io.Video.Input("video", optional=True, tooltip=VIDEO_HINT),
                io.Audio.Input("audio", optional=True, tooltip=AUDIO_HINT),
            ],
            outputs=[
                H3_ASSETS.Output(
                    display_name="assets",
                    tooltip=(
                        "The chain with this asset at its end, for the next MiniMax H3 Asset "
                        "or for assets on MiniMax H3 Conditioning."
                    ),
                ),
                io.Image.Output(
                    display_name="image",
                    tooltip="The picture or frames it holds, for a preview or another node.",
                ),
                io.Audio.Output(display_name="audio", tooltip="The sound it holds."),
                io.String.Output(
                    display_name="report",
                    tooltip="What was read and where it goes.",
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, assets=None, file=WIRED, role=h3_assets.FIRST_FRAME, segment=1,
                           frame=0, clip_start=0.0, clip_seconds=0.0, image=None, video=None,
                           audio=None):
        """Read the file again once it has changed on disk."""
        found = chosen_path(file)
        return picks.fingerprint(found) if found else "wired"

    @classmethod
    def execute(cls, assets=None, file=WIRED, role=h3_assets.FIRST_FRAME, segment=1, frame=0,
                clip_start=0.0, clip_seconds=0.0, image=None, video=None,
                audio=None) -> io.NodeOutput:
        """Read what is wired or chosen, label it and add it to the chain.

        Raises:
            ValueError: Nothing is wired and no file is chosen, the file cannot be read, what
                arrived is not what the role needs, or assets carries something else.
            PathNotAllowed: The chosen file resolved outside every permitted read root.
        """
        from comfy_execution.graph_utils import ExecutionBlocker

        earlier = h3_assets.collect(assets, f"assets on {NODE_NAME}")

        frames, sound, name = None, None, ""
        rate = float(h3_extend.FPS)
        if video is not None:
            components = video.get_components()
            frames, sound = components.images, components.audio
            rate = float(components.frame_rate) or rate
            name = "the wired video"
        if image is not None:
            frames, rate = image, float(h3_extend.FPS)
            name = "the wired frames" if int(image.shape[0]) > 1 else "the wired image"
        if audio is not None:
            sound = audio
            name = name or "the wired audio"
        if file == MOMENT and frames is None and sound is None:
            if role != h3_assets.REFERENCE_PICTURE:
                raise ValueError(
                    f"{NODE_NAME} reads frame {int(frame)} of the video being made, which is a "
                    f"reference picture. Set role to `reference picture`, or pick a file"
                )
            if int(frame) < 0:
                raise ValueError(
                    f"{NODE_NAME} reads a frame of the video being made, counted from its first "
                    f"frame. Set frame to 0 or more"
                )
            name = f"frame {int(frame)} of the video"
            asset = h3_assets.Asset(role, int(segment), int(frame), None, None, name, True)
            where = ("every segment" if asset.segment == h3_assets.EVERY_SEGMENT
                     else f"segment {asset.segment}")
            summary = f"reference picture for {where}: {name}"
            run_result.publish(summary=summary, counts={"frame": int(frame),
                                                        "place in chain": len(earlier) + 1},
                               facts={"role": role, "segment": where, "source": "the video being made"})
            return io.NodeOutput(
                (*earlier, asset), ExecutionBlocker(f"{name} is read during the run"),
                ExecutionBlocker(f"{name} holds no sound"), summary,
            )
        if frames is None and sound is None:
            path = chosen_path(file)
            if path is None:
                raise ValueError(
                    f"{NODE_NAME} has nothing to read: no file is chosen and nothing is wired "
                    f"into image, video or audio. Pick a file from the menu, or wire one in"
                )
            frames, sound, rate = read_file(path, clip_start, clip_seconds)
            name = os.path.basename(path)
        frames = at_model_rate(frames, rate)

        if role == h3_assets.REFERENCE_AUDIO and sound is None:
            raise ValueError(
                f"{NODE_NAME} is set to `reference audio` and {name} holds no sound. Pick a "
                f"sound file or a clip with a soundtrack, or set role to a picture role"
            )
        if role in (h3_assets.FIRST_FRAME, h3_assets.LAST_FRAME, h3_assets.REFERENCE_PICTURE,
                    h3_assets.REFERENCE_CLIP) and frames is None:
            raise ValueError(
                f"{NODE_NAME} is set to `{role}` and {name} holds no picture. Pick a picture "
                f"or a clip, or set role to `reference audio`"
            )
        if role == h3_assets.REFERENCE_CLIP and int(frames.shape[0]) < h3_extend.CLIP_LEAD:
            raise ValueError(
                f"{NODE_NAME} is set to `reference clip` and {name} holds "
                f"{int(frames.shape[0])} frame(s); a clip needs at least {h3_extend.CLIP_LEAD}. "
                f"Pick a video, or set role to `reference picture`"
            )

        asset = h3_assets.Asset(role, int(segment), int(frame), frames, sound, name)
        summary, counts, facts = describe(asset)
        counts["place in chain"] = len(earlier) + 1
        logger.info("%s: %s", NODE_NAME, summary)
        if frames is not None:
            preview.publish_output(frames[:1], slot="image")
        run_result.publish(summary=summary, counts=counts, facts=facts)
        return io.NodeOutput(
            (*earlier, asset),
            frames if frames is not None else ExecutionBlocker(f"{name} holds no picture"),
            sound if sound is not None else ExecutionBlocker(f"{name} holds no sound"),
            summary,
        )
