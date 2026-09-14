"""Locating the cascades and fonts bundled with the pack, and the fonts a user adds.

Fonts are read from ``modules/data/fonts`` and ``<config dir>/fonts``.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from importlib import resources
from pathlib import Path

from .. import log

logger = log.get_logger("data.paths")

__all__ = [
    "CASCADES",
    "CASCADE_DIR",
    "FONTS",
    "FONT_DIR",
    "FONT_SUFFIXES",
    "LEGACY_FONT",
    "PANTRY_SEED",
    "cascade_file",
    "data_directory",
    "font_catalog",
    "font_file",
    "font_names",
    "pantry_seed",
    "user_font_directory",
]

#: Subdirectory holding the OpenCV classifier cascades.
CASCADE_DIR = "cascades"

#: The Noodle Soup Prompts snapshot shipped with the pack, zlib-compressed JSON.
PANTRY_SEED = "nsp_pantry.pack"

#: Subdirectory holding fonts, under this package and under the config directory alike.
FONT_DIR = "fonts"

#: Extensions :func:`font_catalog` will pick up.
FONT_SUFFIXES = (".ttf", ".otf", ".ttc")

#: The font WAS Node Suite 2 shipped, which the chart nodes label with.
LEGACY_FONT = "Rheiborn Sans (v2)"

#: Bundled fonts: the name a widget stores -> the file, relative to the data directory.
FONTS = {
    "DejaVu Sans": "fonts/dejavu/DejaVuSans.ttf",
    "DejaVu Sans Bold": "fonts/dejavu/DejaVuSans-Bold.ttf",
    "DejaVu Sans Mono": "fonts/dejavu/DejaVuSansMono.ttf",
    "DejaVu Serif": "fonts/dejavu/DejaVuSerif.ttf",
    "Liberation Sans": "fonts/liberation/LiberationSans-Regular.ttf",
    "Liberation Sans Bold": "fonts/liberation/LiberationSans-Bold.ttf",
    "Liberation Serif": "fonts/liberation/LiberationSerif-Regular.ttf",
    "Liberation Mono": "fonts/liberation/LiberationMono-Regular.ttf",
    LEGACY_FONT: "font.ttf",
    # The file name v2 knew this font by.
    "font.ttf": "font.ttf",
}

#: OpenCV Haar and LBP classifier cascades bundled with the pack.
CASCADES = (
    "haarcascade_eye.xml",
    "haarcascade_frontalface_alt.xml",
    "haarcascade_frontalface_alt2.xml",
    "haarcascade_frontalface_alt_tree.xml",
    "haarcascade_frontalface_default.xml",
    "haarcascade_profileface.xml",
    "haarcascade_upperbody.xml",
    "lbpcascade_animeface.xml",
)


@lru_cache(maxsize=None)
def data_directory() -> Path:
    """This package's directory on disk.

    Returns:
        The directory holding the bundled assets.
    """
    try:
        located = Path(str(resources.files(__package__)))
    except (ModuleNotFoundError, TypeError, ValueError):
        located = Path(__file__).resolve().parent
    return located if located.is_dir() else Path(__file__).resolve().parent


def user_font_directory() -> Path | None:
    """``<config dir>/fonts``, where a user's own fonts go.

    Returns:
        The directory, whether or not it exists, or ``None`` when the config directory
        cannot be resolved. :mod:`modules.config` is imported inside the call, so this
        module imports without ComfyUI's ``folder_paths``.
    """
    try:
        from ..config.paths import config_directory

        return config_directory() / FONT_DIR
    except Exception:
        logger.debug("the user font directory could not be resolved", exc_info=True)
        return None


def _font_roots() -> list[Path]:
    """The directories :func:`font_catalog` walks, in precedence order."""
    roots = [data_directory() / FONT_DIR]
    user = user_font_directory()
    if user is not None:
        roots.append(user)
    return roots


def _signature(roots: list[Path]) -> tuple:
    """A value that changes whenever a font is added to or removed from ``roots``.

    Args:
        roots: Directories to walk. One that does not exist contributes nothing.

    Returns:
        A hashable snapshot, comparable against a later one.
    """
    marks = []
    for root in roots:
        if not root.is_dir():
            continue
        for directory in (root, *(path for path in root.rglob("*") if path.is_dir())):
            try:
                marks.append((str(directory), directory.stat().st_mtime_ns))
            except OSError:
                continue
    return tuple(sorted(marks))


def _label(path: Path, taken: dict) -> str:
    """A menu name for a discovered font that does not collide with one already taken.

    Args:
        path: The font file.
        taken: The catalog so far.

    Returns:
        The file's stem, qualified with its parent directory when that alone is taken, and
        numbered after that.
    """
    stem = path.stem
    if stem not in taken:
        return stem
    qualified = f"{path.parent.name}/{stem}"
    if qualified not in taken:
        return qualified
    suffix = 2
    while f"{qualified} ({suffix})" in taken:
        suffix += 1
    return f"{qualified} ({suffix})"


def _spelling(path: Path) -> str:
    """A key matching two spellings of one path, taken without asking the filesystem.

    Args:
        path: Any path.

    Returns:
        The absolute path, case-folded where the platform ignores case, which is the
        comparison ``resolve()`` makes on two names for one file after following every
        link on the way to it.
    """
    return os.path.normcase(os.path.abspath(path))


def _linked(path: Path) -> str | None:
    """:func:`_spelling` of the file ``path`` leads to, or ``None`` when it leads nowhere."""
    try:
        return _spelling(path.resolve())
    except OSError:
        return None


def _build_catalog() -> dict[str, Path]:
    """Every font this pack can draw with, bundled first and then discovered."""
    directory = data_directory()
    catalog: dict[str, Path] = {}
    for label, relative in FONTS.items():
        # The file-name spelling of the legacy font maps to the same file as its menu name.
        if label != "font.ttf":
            catalog[label] = directory.joinpath(*relative.split("/"))

    spellings = {_spelling(path) for path in catalog.values()}
    linked: set[str] | None = None

    for root in _font_roots():
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in FONT_SUFFIXES:
                continue
            spelling = _spelling(path)
            if spelling in spellings:
                continue
            # Links are resolved only once an unaccounted-for spelling turns up.
            if linked is None:
                linked = {key for key in map(_linked, catalog.values()) if key is not None}
            key = _linked(path)
            if key is None or key in linked:
                continue
            catalog[_label(path, catalog)] = path
            spellings.add(spelling)
            linked.add(key)
    return catalog


#: Seconds a built catalog is trusted before the directories are stat'd again.
CATALOG_TTL = 2.0

#: Last catalog built, the directory snapshot it was built from, and when that snapshot was
#: taken.
_CATALOG = {"signature": None, "fonts": {}, "checked": 0.0}


def font_catalog() -> dict[str, Path]:
    """Every font available to a widget: menu name to path.

    Returns:
        The catalog, shared rather than copied, so treat it as read-only. A scan that
        fails falls back to the bundled fonts alone rather than raising.
    """
    now = time.monotonic()
    if _CATALOG["fonts"] and (now - _CATALOG["checked"]) < CATALOG_TTL:
        return _CATALOG["fonts"]

    try:
        signature = _signature(_font_roots())
    except Exception:
        logger.debug("the font directories could not be scanned", exc_info=True)
        signature = None

    _CATALOG["checked"] = now
    if signature is not None and _CATALOG["signature"] == signature and _CATALOG["fonts"]:
        return _CATALOG["fonts"]

    try:
        fonts = _build_catalog()
    except Exception:
        logger.warning("the font catalog could not be built; using the bundled fonts only")
        logger.debug("", exc_info=True)
        directory = data_directory()
        fonts = {
            label: directory.joinpath(*relative.split("/"))
            for label, relative in FONTS.items()
            if label != "font.ttf"
        }
    _CATALOG["signature"], _CATALOG["fonts"] = signature, fonts
    return fonts


def font_names() -> tuple[str, ...]:
    """The fonts a widget may offer, in menu order.

    Returns:
        Every key of :func:`font_catalog`: the bundled families first, then whatever was
        found in the two font directories.
    """
    return tuple(font_catalog())


def font_file(name: str = "font.ttf") -> Path:
    """Path to a font, bundled or discovered.

    Args:
        name: A menu name from :func:`font_names`, or the ``font.ttf`` file name the chart
            nodes have always called this with.

    Returns:
        The path the font would occupy. Existence is not checked: a caller that has a
        fallback for a missing font needs to make that decision itself.

    Raises:
        ValueError: ``name`` names no font this pack can reach.
    """
    if name in FONTS:
        return data_directory().joinpath(*FONTS[name].split("/"))

    catalog = font_catalog()
    if name in catalog:
        return catalog[name]
    raise ValueError(
        f"{name!r} is not a font WAS Node Suite can reach. Available: "
        f"{', '.join(catalog)}. Add your own to {user_font_directory() or '<config dir>/fonts'}."
    )


def cascade_file(name: str) -> Path:
    """Path to a bundled OpenCV classifier cascade.

    Args:
        name: File name, which must be one of :data:`CASCADES`.

    Returns:
        The path the cascade would occupy. Existence is not checked.

    Raises:
        ValueError: ``name`` is not a bundled cascade.
    """
    return _resolve(name, CASCADES, data_directory() / CASCADE_DIR)


def pantry_seed() -> Path:
    """Path to the bundled Noodle Soup Prompts snapshot.

    Returns:
        The packed terminology a fresh database is seeded from. Existence is not checked.
    """
    return data_directory() / PANTRY_SEED


def _resolve(name: str, allowed: tuple[str, ...], directory: Path) -> Path:
    """Join an allowlisted file name onto its directory.

    Args:
        name: File name to look up.
        allowed: Every name that may be resolved against ``directory``.
        directory: Directory the name is joined onto.

    Returns:
        ``directory``/``name``.

    Raises:
        ValueError: ``name`` is not in ``allowed``.
    """
    if name not in allowed:
        raise ValueError(
            f"{name!r} is not an asset bundled with WAS Node Suite. "
            f"Available: {', '.join(allowed)}"
        )
    return directory / name
