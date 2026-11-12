"""RefMods, saved MiniMax H3 references in ``models/refmods``, read and written.

A file holds ``latent`` (``ref_N`` in a bundle) under a ``refmod_meta`` header, and JPEG encoder
frames under ``enc_N``. Latents are ``[1, 24, T, H, W]`` or ``[1, 32, 2, T]``.
"""

from __future__ import annotations

import io
import json
import math
import os
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import NamedTuple

from .. import log
from . import h3_extend, h3_references

__all__ = [
    "FOLDER",
    "KINDS",
    "META_KEY",
    "NONE",
    "PARTS",
    "PROMPT_REF",
    "PromptRef",
    "RefMod",
    "STACK_STRIDE",
    "SUFFIX",
    "TAG_WORDS",
    "TOOL_KEY",
    "TOOL_NAME",
    "describe_file",
    "directories",
    "entries",
    "first_stored",
    "foreign",
    "load",
    "located",
    "names",
    "parts",
    "preview_picture",
    "prompt_name",
    "prompt_refs",
    "prompt_tagged",
    "role_kind",
    "save",
    "shown",
    "target",
    "tokens",
]

logger = log.get_logger("latent.h3_refmods")

#: The model folder RefMods live in, shared with every pack reading the format.
FOLDER = "refmods"

#: The one suffix a RefMod file carries.
SUFFIX = ".safetensors"

#: Header keys holding a RefMod's JSON description.
META_KEY = "refmod_meta"
AUDIO_META_KEY = "audio_refmod_meta"

#: Header key and value stamped into every RefMod this pack writes.
TOOL_KEY = "tool"
TOOL_NAME = "WAS Save RefMod"

#: The menu entry naming no file.
NONE = "(none)"

#: What a RefMod holds, and the tag a prompt names each by.
KINDS = ("image", "video", "audio")
TAG_WORDS = {"image": "Picture", "video": "Video", "audio": "Audio"}

#: Format versions written for one reference and for a bundle of several.
FORMAT_VERSION = 4
BUNDLE_VERSION = 5

#: Most members a bundle may list.
MOST_MEMBERS = 256

#: Most latent cells on a visual side, 5760 pixels.
MOST_SIDE = 360

#: Most latent frames a visual RefMod holds, and audio latent steps a voice holds.
MOST_FRAMES = 1024
MOST_STEPS = 60 * h3_extend.AUDIO_LATENT_FPS * 10

#: Most stored encoder frames read from one file, and their bytes together.
MOST_STORED = 512
MOST_STORED_BYTES = 64 << 20

#: Largest header read, and largest sidecar ``.json`` description read, in bytes.
HEADER_LIMIT = 32 << 20
SIDECAR_LIMIT = 1 << 20

#: Pictures of a stack between the ones the text encoder is shown.
STACK_STRIDE = 4

#: Factor a weakened reference is blurred toward, in latent cells.
LOW_PASS = 8

#: The rate a clip's stored encoder frames are picked at, two a second.
ENCODER_FPS = float(h3_extend.FPS)

#: JPEG quality of stored encoder frames.
JPEG_QUALITY = 95

#: Files held in memory once read, and their latent bytes together.
CACHE_FILES = 16
CACHE_BYTES = 1 << 30

#: Suffixes of a picture kept beside a RefMod as its thumbnail.
PREVIEW_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")

#: A RefMod a prompt loads: ``<ref:alice>``, ``<ref:cast/alice>``, ``<ref:alice=image,audio>``,
#: ``<ref:alice:0.6>`` or ``<ref:alice=audio:0.6>``.
PROMPT_REF = re.compile(
    r"<ref:\s*([^<>=:\n]+?)\s*(?:=\s*([A-Za-z][A-Za-z ,]*?)\s*)?(?::\s*(\d*\.?\d+)\s*)?>",
    re.IGNORECASE,
)

#: The parts a prompt reference may ask for, and the kinds each loads.
PARTS = {"image": ("image",), "video": ("video",), "audio": ("audio",), "look": ("image", "video")}

#: Most RefMod tokens one prompt reads.
MOST_PROMPT_REFS = 32

_lock = threading.Lock()
_cache: OrderedDict = OrderedDict()
_described: OrderedDict = OrderedDict()


class RefMod:
    """One saved reference: a VAE latent and what its file says about it.

    Attributes:
        name: Its name, as the file states it or the file's stem.
        kind: An entry of :data:`KINDS`.
        latent: The latent tensor on the CPU.
        source: ``image``, ``stack``, ``video``, ``audio`` or blank.
        file: The menu name of the file it came from, blank for one built in memory.
        member: Its place in a bundle, or ``None`` for a file of one.
        stored: The text encoder's frames as JPEG byte tensors, or an empty tuple.
        stored_times: Their timestamps in seconds.
        concept: What it represents, as the file states it.
    """

    __slots__ = ("name", "kind", "latent", "source", "file", "member", "stored",
                 "stored_times", "concept")

    def __init__(self, name, kind, latent, source="", file="", member=None, stored=(),
                 stored_times=(), concept="generic"):
        self.name = str(name)
        self.kind = str(kind)
        self.latent = latent
        self.source = str(source or "")
        self.file = str(file or "")
        self.member = member
        self.stored = tuple(stored)
        self.stored_times = tuple(float(time) for time in stored_times)
        self.concept = str(concept or "generic")

    @property
    def latent_t(self) -> int:
        """Latent frames, or audio latent steps."""
        return int(self.latent.shape[-1] if self.kind == "audio" else self.latent.shape[2])

    @property
    def latent_h(self) -> int:
        """Latent rows, 0 for audio."""
        return 0 if self.kind == "audio" else int(self.latent.shape[3])

    @property
    def latent_w(self) -> int:
        """Latent columns, 0 for audio."""
        return 0 if self.kind == "audio" else int(self.latent.shape[4])

    @property
    def token_count(self) -> int:
        """Tokens the reference adds to every sampling step."""
        if self.kind == "audio":
            return 2 * self.latent_t
        return self.latent_t * (self.latent_h // 2) * (self.latent_w // 2)

    def ref_block(self, strength: float = 1.0, curve=None) -> dict | None:
        """The ``minimax_refs`` block the model reads.

        Args:
            strength: ``1`` = as saved; below ``1`` = mixed toward a blurred copy; ``0`` = left out.
            curve: Accepted for bundles built by other packs and not read.

        Returns:
            The block, or ``None`` at strength 0.
        """
        strength = float(strength)
        if strength <= 0.0:
            return None
        latent = self.latent
        if strength < 1.0:
            latent = strength * latent + (1.0 - strength) * _low_pass(latent)
        if self.kind == "audio":
            return {"kind": "audio", "ref_audio_t": self.latent_t, "audio_latent": latent}
        block = {"kind": self.kind, "latent_h": self.latent_h, "latent_w": self.latent_w,
                 "latent": latent}
        if self.kind == "video":
            block.update({"latent_t": self.latent_t, "ref_audio_t": 0, "audio_latent": None})
        return block


def _low_pass(latent):
    """A latent blurred by :data:`LOW_PASS` cells and brought back to its size."""
    import torch.nn.functional as F

    if latent.dim() == 4:
        batch, channels, sides, steps = latent.shape
        if steps <= 1:
            return latent
        flat = latent.reshape(batch * channels * sides, 1, steps).float()
        down = F.adaptive_avg_pool1d(flat, max(1, steps // LOW_PASS))
        up = F.interpolate(down, size=steps, mode="linear", align_corners=False)
        return up.reshape_as(latent).to(latent.dtype)
    frames, high, wide = latent.shape[2], latent.shape[3], latent.shape[4]
    down = F.adaptive_avg_pool3d(latent.float(), (frames, max(1, high // LOW_PASS),
                                                   max(1, wide // LOW_PASS)))
    up = F.interpolate(down, size=(frames, high, wide), mode="trilinear", align_corners=False)
    return up.to(latent.dtype)


def role_kind(kind: str) -> str:
    """The asset role a RefMod of one kind plays.

    Args:
        kind: An entry of :data:`KINDS`.

    Returns:
        ``reference picture``, ``reference clip`` or ``reference audio``.
    """
    from .h3_assets import REFERENCE_AUDIO, REFERENCE_CLIP, REFERENCE_PICTURE

    return {"image": REFERENCE_PICTURE, "video": REFERENCE_CLIP}.get(kind, REFERENCE_AUDIO)


def directories() -> list[Path]:
    """Every folder ComfyUI lists for RefMods, registering ``models/refmods`` when new.

    Returns:
        The folders in search order, empty outside ComfyUI.
    """
    from ..model import model_directories

    return model_directories(FOLDER)


def names() -> list[str]:
    """The RefMod files the folders hold, as a menu lists them.

    Returns:
        Each ``.safetensors`` name relative to its folder, sorted, from ComfyUI's own cached
        listing.
    """
    try:
        directories()
        import folder_paths

        found = folder_paths.get_filename_list(FOLDER)
    except Exception as error:
        logger.debug("the RefMod folders could not be listed: %s", error)
        return []
    return sorted((name for name in found if name.lower().endswith(SUFFIX)), key=str.lower)


def located(value) -> tuple[str, str]:
    """The RefMod file one menu entry or wired path names.

    Args:
        value: A name the menu offered, or a full path inside a RefMod folder wired as text.

    Returns:
        ``(name, path)``: the name as the menu lists it and the absolute file.

    Raises:
        PathNotAllowed: The value names another machine, a file outside the RefMod folders,
            or a file that is not a ``.safetensors``.
        FileNotFoundError: No RefMod folder holds a file of that name.
    """
    from ..io import picker
    from ..util import sandbox

    directories()
    name = picker.model_name(value, (FOLDER,))
    if not name.lower().endswith(SUFFIX):
        raise sandbox.PathNotAllowed(
            f"`{sandbox._shown(name)}` is not a RefMod. Pick a .safetensors file from "
            f"models/refmods"
        )
    reason = sandbox.leaves_folder(name)
    if reason is not None:
        raise sandbox.PathNotAllowed(
            f"`{sandbox._shown(name)}` is a RefMod name that {reason}. Pick a file from the menu"
        )
    import folder_paths

    found = folder_paths.get_full_path(FOLDER, name)
    if not found:
        raise FileNotFoundError(
            f"no RefMod named `{name}` is in models/refmods or the refmods folders "
            f"extra_model_paths.yaml lists. Pick one from the menu, or save one with "
            f"MiniMax H3 Save RefMod"
        )
    real = Path(os.path.realpath(found))
    roots = [Path(os.path.realpath(folder)) for folder in directories()]
    if not any(sandbox.contains(root, real) for root in roots):
        raise sandbox.PathNotAllowed(
            f"`{sandbox._shown(name)}` leaves the RefMod folders through a link. Pick a file "
            f"stored in models/refmods"
        )
    return name, str(real)


def _stamp(path: str) -> tuple:
    """A file's identity for the cache: path, size and modification time."""
    stat = os.stat(path)
    return os.path.normcase(path), stat.st_size, stat.st_mtime_ns


def _metadata(handle, path: str) -> dict:
    """The JSON description a RefMod file carries, embedded or beside it."""
    header = handle.metadata() or {}
    for key in (META_KEY, AUDIO_META_KEY):
        if key in header:
            try:
                meta = json.loads(header[key])
            except ValueError:
                meta = None
            if isinstance(meta, dict):
                return meta
    sidecar = path[: -len(SUFFIX)] + ".json"
    if os.path.isfile(sidecar) and os.path.getsize(sidecar) <= SIDECAR_LIMIT:
        with open(sidecar, encoding="utf-8") as opened:
            meta = json.load(opened)
        if isinstance(meta, dict):
            return meta
    raise ValueError(
        f"{os.path.basename(path)} carries no RefMod description in its header, so it is not "
        f"a RefMod. Pick a file saved by a RefMod node"
    )


def _members(meta: dict, name: str) -> list[tuple[str, dict]]:
    """Each reference a file holds, as ``(tensor key, description)``."""
    if meta.get("kind") != "bundle":
        return [("latent", meta)]
    listed = meta.get("members")
    if not isinstance(listed, list) or not 0 < len(listed) <= MOST_MEMBERS:
        raise ValueError(
            f"{name} is a RefMod bundle whose member list is missing or holds more than "
            f"{MOST_MEMBERS}. Save it again"
        )
    return [(f"ref_{index}", member) for index, member in enumerate(listed)
            if isinstance(member, dict) and member.get("kind") in KINDS]


def _checked_shape(shape, kind: str, name: str) -> None:
    """Refuse a latent whose stated shape does not fit its kind or the ceilings."""
    shape = [int(side) for side in shape]
    if kind == "audio":
        fits = len(shape) == 4 and shape[:3] == [1, 32, 2] and 0 < shape[3] <= MOST_STEPS
    else:
        fits = (len(shape) == 5 and shape[:2] == [1, 24] and 0 < shape[2] <= MOST_FRAMES
                and 0 < shape[3] <= MOST_SIDE and 0 < shape[4] <= MOST_SIDE)
        if fits and kind == "image":
            fits = shape[2] == 1
    if not fits:
        raise ValueError(
            f"{name} states a {kind} latent shaped {shape}, which is not an H3 reference. "
            f"Visual RefMods are [1, 24, T, H, W] up to {MOST_SIDE} cells a side; audio is "
            f"[1, 32, 2, T]. Save it again"
        )


def _stored(handle, keys: set, times) -> tuple[tuple, tuple]:
    """The stored encoder frames a file holds, as JPEG byte tensors and their times."""
    if not isinstance(times, list) or not times:
        return (), ()
    count = min(len(times), MOST_STORED)
    wanted = [f"enc_{index}" for index in range(count)]
    if not all(key in keys for key in wanted):
        return (), ()
    total = 0
    frames = []
    for key in wanted:
        shape = handle.get_slice(key).get_shape()
        if len(shape) != 1:
            return (), ()
        total += int(shape[0])
        if total > MOST_STORED_BYTES:
            return (), ()
        frames.append(handle.get_tensor(key).clone())
    try:
        stamps = tuple(float(time) for time in times[:count])
    except (TypeError, ValueError):
        return (), ()
    return tuple(frames), stamps


def read(path: str, name: str) -> list[RefMod]:
    """Every reference one RefMod file holds.

    Args:
        path: The file, already located.
        name: Its menu name, for reports and errors.

    Returns:
        The references, in file order.

    Raises:
        ValueError: The file is not a RefMod, or states a latent no H3 reference has.
    """
    from safetensors import safe_open

    with open(path, "rb") as opened:
        length = int.from_bytes(opened.read(8), "little")
    if not 2 <= length <= HEADER_LIMIT:
        raise ValueError(f"{name} is not a safetensors file. Pick a RefMod")
    found = []
    with safe_open(path, framework="pt", device="cpu") as handle:
        meta = _metadata(handle, path)
        keys = set(handle.keys())
        stem = os.path.splitext(os.path.basename(name))[0]
        members = _members(meta, name)
        if not members:
            raise ValueError(f"{name} is a RefMod bundle holding no picture, clip or voice")
        stored_taken = False
        for key, member in members:
            kind = str(member.get("kind", "image"))
            if kind not in KINDS or key not in keys:
                raise ValueError(f"{name} names a `{kind}` reference under `{key}` that it does not hold")
            _checked_shape(handle.get_slice(key).get_shape(), kind, name)
            latent = handle.get_tensor(key).clone()
            stored, times = (), ()
            if kind != "audio" and not stored_taken and member.get("enc_times"):
                stored, times = _stored(handle, keys, member.get("enc_times"))
                stored_taken = bool(stored)
            index = None if key == "latent" else int(key.split("_")[1])
            label = str(member.get("name") or stem)
            found.append(RefMod(label, kind, latent, member.get("source", ""), name, index,
                                stored, times, member.get("concept_type", "generic")))
    return found


def load(value) -> list[RefMod]:
    """Every reference the RefMod one menu entry or wired path names, read once per version.

    Args:
        value: A menu name or a wired path.

    Returns:
        The references, in file order.

    Raises:
        PathNotAllowed: The value names a file outside the RefMod folders.
        FileNotFoundError: No such RefMod.
        ValueError: The file is not a RefMod.
    """
    name, path = located(value)
    stamp = _stamp(path)
    with _lock:
        held = _cache.get(stamp)
        if held is not None:
            _cache.move_to_end(stamp)
            return held
    found = read(path, name)
    with _lock:
        _cache[stamp] = found
        while len(_cache) > CACHE_FILES or sum(
                mod.latent.numel() * mod.latent.element_size()
                for mods in _cache.values() for mod in mods) > CACHE_BYTES:
            if len(_cache) == 1:
                break
            _cache.popitem(last=False)
    return found


def entries(mods, named: str) -> list[tuple[object, float]]:
    """The references a RefMod bundle carries, checked.

    Args:
        mods: What arrived on an ``H3_REF_MODS`` input: a list of ``(mod, strength)`` pairs.
        named: The node and input it arrived on, for the error.

    Returns:
        Each ``(mod, strength)`` pair, strength between 0 and 1.

    Raises:
        ValueError: The input holds something else, or a member is not an H3 reference.
    """
    if mods is None:
        return []
    if not isinstance(mods, (list, tuple)) or len(mods) > MOST_MEMBERS:
        raise ValueError(
            f"{named} holds a {type(mods).__name__}, and a RefMod bundle is a list of up to "
            f"{MOST_MEMBERS} RefMods from a RefMod Stack. Wire its mods output into it"
        )
    found = []
    for place, entry in enumerate(mods, start=1):
        try:
            mod, strength = entry
            strength = float(strength)
        except (TypeError, ValueError):
            raise ValueError(f"entry {place} on {named} is not a RefMod and a strength") from None
        kind = getattr(mod, "kind", None)
        latent = getattr(mod, "latent", None)
        if kind not in KINDS or not callable(getattr(mod, "ref_block", None)) or latent is None:
            raise ValueError(
                f"entry {place} on {named} is a {type(mod).__name__}, which is not an H3 "
                f"RefMod. Wire a RefMod Stack's mods output"
            )
        _checked_shape(getattr(latent, "shape", ()), kind, f"entry {place} on {named}")
        if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
            raise ValueError(f"entry {place} on {named} has strength {strength}; RefMod strengths run 0 to 1")
        found.append((mod, strength))
    return found


class PromptRef(NamedTuple):
    """One ``<ref:...>`` token of a prompt, resolved.

    Attributes:
        name: The RefMod's menu name.
        kinds: The kinds the token asks for, empty for every kind the file holds.
        strength: How strongly it is held.
        assets: One asset per reference it loads, in file order.
    """

    name: str
    kinds: tuple
    strength: float
    assets: tuple


def _stem(name: str) -> str:
    """A menu name's file name without its folders or suffix."""
    return os.path.splitext(name.replace("\\", "/").rsplit("/", 1)[-1])[0]


def _bare(name: str) -> str:
    """A menu name with ``/`` separators and no suffix."""
    return os.path.splitext(name.replace("\\", "/"))[0]


def prompt_name(written: str, listing: list[str], named: str) -> str:
    """The menu name a prompt's ``<ref:...>`` token names.

    Args:
        written: The name as the token writes it, as ``alice`` or ``cast/alice``.
        listing: The RefMods on this machine, from :func:`names`.
        named: The prompt, for the error.

    Returns:
        The menu name; a file at the top of a folder wins over one in a subfolder.

    Raises:
        ValueError: No RefMod, or more than one, answers to the name.
    """
    wanted = written.strip().replace("\\", "/")
    if wanted.lower().endswith(SUFFIX):
        wanted = wanted[: -len(SUFFIX)]
    nested = "/" in wanted
    found = [name for name in listing
             if (_bare(name) if nested else _stem(name)).lower() == wanted.lower()]
    if not nested:
        found = [name for name in found if "/" not in _bare(name)] or found
    if len(found) > 1:
        found = [name for name in found if (_bare(name) if nested else _stem(name)) == wanted] or found
    if len(found) == 1:
        return found[0]
    if found:
        raise ValueError(
            f"{named} loads <ref:{written.strip()}>, and {len(found)} RefMods answer to that name: "
            f"{', '.join(_bare(name) for name in found[:6])}. Write the folder too, as "
            f"<ref:{_bare(found[0])}>"
        )
    known = ", ".join(_bare(name) for name in listing[:8])
    raise ValueError(
        f"{named} loads <ref:{written.strip()}>, and no RefMod in models/refmods is called "
        f"`{wanted}`. "
        + (f"RefMods there: {known}{', ...' if len(listing) > 8 else ''}" if listing
           else "There are none yet; MiniMax H3 Save RefMod saves one")
    )


def prompt_refs(text: str, segment: int, named: str) -> list[PromptRef]:
    """The RefMods one prompt loads with ``<ref:...>`` tokens, in the order they are written.

    Args:
        text: The prompt, header and footer included.
        segment: The segment it prompts, from 1.
        named: The prompt, for the error.

    Returns:
        One entry per token; a reference two tokens load is one asset shared by both.

    Raises:
        ValueError: A token names no RefMod, asks for a part the file lacks, sets a strength
            outside 0 to 1, or two tokens set one reference to two strengths.
    """
    from .h3_assets import Asset

    matches = list(PROMPT_REF.finditer(str(text or "")))
    if not matches:
        return []
    if len(matches) > MOST_PROMPT_REFS:
        raise ValueError(
            f"{named} holds {len(matches)} <ref:...> tokens, and a prompt reads at most "
            f"{MOST_PROMPT_REFS}. Load the shared ones once in the header"
        )
    listing = names()
    held, found = {}, []
    for match in matches:
        written, asked, level = match.groups()
        name = prompt_name(written, listing, named)
        kinds = []
        for word in (piece.strip().lower() for piece in str(asked or "").split(",")):
            if not word:
                continue
            if word not in PARTS:
                raise ValueError(
                    f"{named}: `{match.group(0)}` asks for `{word}`, and the parts a RefMod gives "
                    f"are image, video, audio and look"
                )
            kinds.extend(kind for kind in PARTS[word] if kind not in kinds)
        strength = 1.0 if level is None else float(level)
        if not 0.0 < strength <= 1.0:
            raise ValueError(
                f"{named}: `{match.group(0)}` sets strength {level}, and a RefMod in a prompt is "
                f"held above 0 and at most 1, as `:0.6`"
            )
        mods = load(name)
        chosen = [(place, mod) for place, mod in enumerate(mods) if not kinds or mod.kind in kinds]
        if not chosen:
            holds = " and ".join(sorted({mod.kind for mod in mods}))
            raise ValueError(
                f"{named}: `{match.group(0)}` asks for {', '.join(kinds)}, and {_bare(name)} holds "
                f"{holds}. Ask for {holds}"
            )
        assets = []
        for place, mod in chosen:
            asset = held.get((name, place))
            if asset is None:
                asset = Asset(role_kind(mod.kind), int(segment), 0, None, None, mod.name, False, mod, strength)
                held[(name, place)] = asset
            elif abs(asset.strength - strength) > 1e-9:
                raise ValueError(
                    f"{named} loads {_bare(name)} at strength {asset.strength:g} and at "
                    f"{strength:g}. Give its tokens one strength"
                )
            assets.append(asset)
        found.append(PromptRef(name, tuple(kinds), strength, tuple(assets)))
    return found


def prompt_tagged(text: str, refs: list[PromptRef], pictures: list, videos: list, sounds: list,
                  cache: dict) -> tuple[str, list[str]]:
    """A prompt with each ``<ref:...>`` token written as the tags its references took.

    Args:
        text: The prompt :func:`prompt_refs` read.
        refs: What it answered, one per token.
        pictures: The prompt's encoded pictures, in tag order.
        videos: Its encoded clips, in tag order.
        sounds: Its encoded sounds, in tag order.
        cache: Encoded references by asset identity.

    Returns:
        ``(prompt, lines)``: each token replaced by its tags joined with ``and``, as
        ``<Picture 2> and <Audio 1>``, and one report line per token, as
        ``<ref:alice> = <Picture 2> and <Audio 1>``.
    """
    heard = sum(1 for video in videos if video.seconds is not None)

    def tag_of(asset) -> str:
        part = cache.get(id(asset))
        for number, each in enumerate(pictures, start=1):
            if each is part:
                return f"<Picture {number}>"
        for number, each in enumerate(videos, start=1):
            if each is part:
                return f"<Video {number}>"
        for number, each in enumerate(sounds, start=1):
            if each is part:
                return f"<Audio {heard + number}>"
        return ""

    resolved, lines = iter(refs), []

    def swapped(match) -> str:
        tags = " and ".join(tag for tag in map(tag_of, next(resolved).assets) if tag)
        lines.append(f"{match.group(0)} = {tags}")
        return tags

    return PROMPT_REF.sub(swapped, str(text)), lines


def _jpeg_frames(stored) -> object:
    """Stored JPEG byte tensors as ``[N, H, W, 3]`` frames in ``[0, 1]``."""
    import numpy as np
    import torch
    from PIL import Image

    from ..image import sizing

    import comfy.utils

    frames = []
    for raw in stored:
        with Image.open(io.BytesIO(raw.numpy().tobytes())) as picture:
            wide, high = picture.size
            if wide * high > sizing.MAX_CANVAS_PIXELS or max(wide, high) > sizing.MAX_CANVAS_SIDE:
                raise ValueError(f"a stored RefMod frame states {wide}x{high} pixels, past the ceiling")
            frame = torch.from_numpy(np.array(picture.convert("RGB"))).float() / 255.0
        if frames and frame.shape != frames[0].shape:
            # Every frame is brought to the first one's size.
            frame = comfy.utils.common_upscale(
                frame[None].movedim(-1, 1), frames[0].shape[1], frames[0].shape[0], "bilinear",
                "disabled").movedim(1, -1)[0]
        frames.append(frame)
    return torch.stack(frames)


def _decoded(vae, latent):
    """A visual latent decoded to ``[N, H, W, 3]`` frames on the CPU."""
    pixels = vae.decode(latent)
    if pixels.ndim == 5:
        pixels = pixels.reshape(-1, *pixels.shape[-3:])
    return pixels[..., :3].float().cpu()


def _foreign_stored(mod) -> tuple[tuple, tuple]:
    """The stored frames of a RefMod another pack loaded, read from its own file."""
    from ..util import sandbox

    times = getattr(mod, "enc_times", None)
    base = getattr(mod, "path", "")
    if not times or not isinstance(base, str) or not base:
        return (), ()
    path = os.path.realpath(base + SUFFIX)
    roots = [Path(os.path.realpath(folder)) for folder in directories()]
    if not os.path.isfile(path) or not any(sandbox.contains(root, Path(path)) for root in roots):
        return (), ()
    from safetensors import safe_open

    with safe_open(path, framework="pt", device="cpu") as handle:
        return _stored(handle, set(handle.keys()), list(times))


def shown(mod, vae, cache: dict) -> tuple[object, list]:
    """What the text encoder is shown for one visual RefMod.

    Args:
        mod: A RefMod, this pack's or one another pack loaded.
        vae: The H3 video VAE, used where the file stores no frames.
        cache: Frames already worked out this run, by RefMod identity.

    Returns:
        ``(frames, timestamps)``: ``[N, H, W, 3]`` frames on the CPU and their seconds.
    """
    import torch

    key = ("shown", id(mod))
    if key in cache:
        return cache[key]
    stored, times = (mod.stored, mod.stored_times) if isinstance(mod, RefMod) else _foreign_stored(mod)
    source = getattr(mod, "source", "")
    if stored:
        frames, times = _jpeg_frames(stored), list(times)
    elif source == "stack":
        latent = mod.latent
        frames = torch.cat([
            _decoded(vae, latent[:, :, index:index + 1])[:1] for index in range(latent.shape[2])
        ])
        times = [float(index) for index in range(frames.shape[0])]
    else:
        frames = _decoded(vae, mod.latent)
        if mod.kind == "image":
            frames, times = frames[:1], [0.0]
        else:
            picks = list(range(0, int(frames.shape[0]), h3_references.QWEN_STRIDE))
            frames, times = frames[picks], [index / 2.0 for index in range(len(picks))]
    if source == "stack":
        picks = list(range(0, int(frames.shape[0]), STACK_STRIDE))
        frames, times = frames[picks], [index / 2.0 for index in range(len(picks))]
    cache[key] = (frames, times)
    return frames, times


def _softened(frames, strength: float, latent_h: int, latent_w: int):
    """Encoder frames weakened as the latent is, blurred over the same cells in pixels."""
    if strength >= 1.0:
        return frames
    import torch.nn.functional as F

    samples = frames.movedim(-1, 1).float()
    down = F.adaptive_avg_pool2d(samples, (max(1, latent_h // LOW_PASS), max(1, latent_w // LOW_PASS)))
    up = F.interpolate(down, size=samples.shape[-2:], mode="bilinear", align_corners=False)
    return (strength * samples + (1.0 - strength) * up).movedim(1, -1)


def parts(mod, strength: float, vae, cache: dict, name: str = ""):
    """One RefMod as a reference a prompt carries.

    Args:
        mod: A RefMod, this pack's or one another pack loaded.
        strength: ``1`` = as saved, below ``1`` = weaker, ``0`` = left out.
        vae: The H3 video VAE.
        cache: Frames already worked out this run, by RefMod identity.
        name: What the report calls it, the RefMod's own name when blank.

    Returns:
        An :class:`h3_references.Picture`, :class:`~h3_references.Video` or
        :class:`~h3_references.Sound`, or ``None`` at strength 0.
    """
    block = mod.ref_block(float(strength))
    if block is None:
        return None
    named = name or str(getattr(mod, "name", "RefMod"))
    weaker = f", strength {strength:.2f}" if strength < 1.0 else ""
    kind = block.get("kind")
    if kind == "audio":
        seconds = int(block["ref_audio_t"]) / h3_extend.AUDIO_LATENT_FPS
        return h3_references.Sound({"type": "audio"}, block, seconds, f"RefMod {named}{weaker}")
    high, wide = int(block["latent_h"]), int(block["latent_w"])
    frames, times = shown(mod, vae, cache)
    frames = _softened(frames, float(strength), high, wide)
    size = f"{wide * h3_references.SPATIAL_STRIDE}x{high * h3_references.SPATIAL_STRIDE}"
    if kind == "image":
        return h3_references.Picture({"type": "image", "data": frames[:1]}, block,
                                     f"RefMod {named}, {size}{weaker}")
    item = {"type": "video", "data": frames, "timestamps": list(times)}
    held = "pictures" if getattr(mod, "source", "") == "stack" else "latent frames"
    return h3_references.Video([item], block,
                               f"RefMod {named}, {int(block['latent_t'])} {held} at {size}{weaker}",
                               None)


def describe_file(name: str) -> dict | None:
    """What one RefMod file holds, read from its header alone.

    Args:
        name: A menu name.

    Returns:
        ``{"name", "members", "stored", "mtime"}`` with one ``{"kind", "label", "tokens"}`` per
        member, or ``None`` where the file is missing or no RefMod.
    """
    try:
        name, path = located(name)
        stamp = _stamp(path)
    except Exception as error:
        logger.debug("RefMod `%s` could not be located: %s", name, error)
        return None
    with _lock:
        held = _described.get(stamp)
        if held is not None:
            return held
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as handle:
            meta = _metadata(handle, path)
            keys = set(handle.keys())
            stem = os.path.splitext(os.path.basename(name))[0]
            members = []
            for key, member in _members(meta, name):
                kind = str(member.get("kind"))
                if key not in keys:
                    continue
                shape = [int(side) for side in handle.get_slice(key).get_shape()]
                _checked_shape(shape, kind, name)
                tokens = 2 * shape[3] if kind == "audio" else shape[2] * (shape[3] // 2) * (shape[4] // 2)
                members.append({"kind": kind, "label": str(member.get("name") or stem)[:120],
                                "tokens": tokens})
            stored = any(member.get("enc_times") for _, member in _members(meta, name))
    except Exception as error:
        logger.debug("RefMod `%s` could not be described: %s", name, error)
        return None
    found = {"name": name, "members": members, "stored": bool(stored),
             "mtime": round(stamp[2] / 1e9, 3)}
    with _lock:
        _described[stamp] = found
        while len(_described) > 4096:
            _described.popitem(last=False)
    return found


def tokens(mod) -> int:
    """Tokens one RefMod adds to every sampling step, read from its latent's shape.

    Args:
        mod: A RefMod, this pack's or one another pack loaded.

    Returns:
        The token count.
    """
    shape = [int(side) for side in mod.latent.shape]
    if mod.kind == "audio":
        return 2 * shape[-1]
    return shape[2] * (shape[3] // 2) * (shape[4] // 2)


def first_stored(mod):
    """The first frame a RefMod stores for the text encoder.

    Args:
        mod: A RefMod, this pack's or one another pack loaded.

    Returns:
        A ``[1, H, W, 3]`` frame, or ``None`` where it stores none.
    """
    if mod.kind == "audio":
        return None
    stored = mod.stored if isinstance(mod, RefMod) else _foreign_stored(mod)[0]
    return _jpeg_frames(stored[:1]) if stored else None


def preview_picture(name: str):
    """A picture standing for one RefMod: its first stored frame, or a picture beside it.

    Args:
        name: A menu name.

    Returns:
        A PIL image in RGB, or ``None`` where the RefMod has neither.
    """
    from PIL import Image

    from ..image import sizing
    from ..util import sandbox

    name, path = located(name)
    stem = path[: -len(SUFFIX)]
    folder = Path(os.path.dirname(path))
    for suffix in PREVIEW_SUFFIXES:
        candidate = stem + suffix
        if os.path.isfile(candidate) and sandbox.contains(folder, Path(os.path.realpath(candidate))):
            with Image.open(candidate) as picture:
                wide, high = picture.size
                if wide * high <= sizing.MAX_CANVAS_PIXELS:
                    return picture.convert("RGB")
    for mod in load(name):
        if mod.stored:
            with Image.open(io.BytesIO(mod.stored[0].numpy().tobytes())) as picture:
                wide, high = picture.size
                if wide * high <= sizing.MAX_CANVAS_PIXELS:
                    return picture.convert("RGB")
    return None


def target(name: str) -> tuple[str, str]:
    """Where a RefMod saved under a name lands.

    Args:
        name: The name typed, as ``alice`` or ``characters/alice``; ``.safetensors`` is added.

    Returns:
        ``(path, menu name)``, the path inside the first RefMod folder.

    Raises:
        PathNotAllowed: The name leaves the folder.
        RuntimeError: ComfyUI lists no RefMod folder.
    """
    from ..util import sandbox

    folders = directories()
    if not folders:
        raise RuntimeError("this ComfyUI lists no folder for RefMods. Add models/refmods")
    text = str(name or "").strip().replace("\\", "/").lstrip("/")
    if not text:
        raise ValueError("MiniMax H3 Save RefMod has no name. Type one, as `alice` or `cast/alice`")
    if not text.lower().endswith(SUFFIX):
        text += SUFFIX
    reason = sandbox.leaves_folder(text)
    if reason is not None:
        raise sandbox.PathNotAllowed(
            f"refusing to save the RefMod `{sandbox._shown(text)}`: a RefMod name is a name inside "
            f"models/refmods, and this one {reason}"
        )
    folder = Path(os.path.realpath(folders[0]))
    path = Path(os.path.realpath(os.path.join(str(folder), *text.split("/"))))
    if not sandbox.contains(folder, path):
        raise sandbox.PathNotAllowed(
            f"refusing to save the RefMod `{sandbox._shown(text)}`: it leaves models/refmods "
            f"through a link"
        )
    return str(path), text.replace("/", os.sep)


def foreign(path: str) -> bool:
    """Whether a file already at a save's target was written by something other than this pack.

    Args:
        path: The target.

    Returns:
        True where a file is there whose header lacks :data:`TOOL_KEY` set to :data:`TOOL_NAME`.
    """
    if not os.path.exists(path):
        return False
    from ..lora.output import header_metadata

    held = header_metadata(Path(path))
    return not held or held.get(TOOL_KEY) != TOOL_NAME


def _jpeg(frame) -> object:
    """One ``[H, W, 3]`` frame as JPEG bytes in a uint8 tensor."""
    import torch
    from PIL import Image

    buffer = io.BytesIO()
    pixels = (frame[..., :3].float().clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    Image.fromarray(pixels).save(buffer, "JPEG", quality=JPEG_QUALITY)
    return torch.frombuffer(bytearray(buffer.getvalue()), dtype=torch.uint8)


def save(path: str, stem: str, members: list[dict]) -> str:
    """Write a RefMod file, one reference or a bundle, through a ``.part`` file.

    Args:
        path: The target, from :func:`target`.
        stem: The name stated inside it.
        members: One ``{"kind", "latent", "source", "shown", "times"}`` per reference, ``shown``
            being ``[N, H, W, 3]`` encoder frames or ``None``.

    Returns:
        The path written.
    """
    from safetensors.torch import save_file

    tensors, described, stored = {}, [], False
    for index, member in enumerate(members):
        latent = member["latent"].detach().cpu().contiguous()
        kind = member["kind"]
        meta = {
            "_format_version": FORMAT_VERSION, "name": stem if len(members) == 1 else f"{stem}_{kind}",
            "kind": kind, "mode": "encode", "source": member.get("source", kind),
            "source_shape": "", "pool": "", "optimize_steps": 0, "tags": [], "description": "",
            "concept_type": "voice" if kind == "audio" else "generic", "sample_rate": 32000,
        }
        if kind == "audio":
            meta.update({"latent_t": int(latent.shape[-1]), "latent_h": 0, "latent_w": 0})
        else:
            meta.update({"latent_t": int(latent.shape[2]), "latent_h": int(latent.shape[3]),
                         "latent_w": int(latent.shape[4])})
        if kind != "audio" and member.get("shown") is not None and not stored:
            frames = member["shown"]
            for place in range(int(frames.shape[0])):
                tensors[f"enc_{place}"] = _jpeg(frames[place])
            meta["enc_times"] = [float(time) for time in member["times"]]
            meta["enc_fps"] = ENCODER_FPS
            stored = True
        tensors["latent" if len(members) == 1 else f"ref_{index}"] = latent
        described.append(meta)
    if len(members) == 1:
        header = described[0]
    else:
        header = {"_format_version": BUNDLE_VERSION, "kind": "bundle", "name": stem,
                  "members": described}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    part = path + ".part"
    try:
        save_file(tensors, part, metadata={META_KEY: json.dumps(header), TOOL_KEY: TOOL_NAME})
        os.replace(part, path)
    finally:
        if os.path.exists(part):
            os.remove(part)
    with _lock:
        for key in [key for key in _cache if key[0] == os.path.normcase(path)]:
            _cache.pop(key, None)
    return path
