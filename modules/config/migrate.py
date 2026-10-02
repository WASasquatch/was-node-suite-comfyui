"""One-shot v2 to v3 migration.

Copies v2's writable state out of the repository root and translates
``was_suite_config.json`` onto the v3 schema. Both steps are idempotent and leave the v2
originals in place.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path

from .. import log
from .defaults import VERSION
from .reconcile import as_user_copy

logger = log.get_logger("config")

#: Documented copy of every key at its default, shipped in the repository root and used as
#: the starting configuration for an install with no v2 settings to carry forward.
EXAMPLE_NAME = "config.example.yaml"

V2_CONFIG_NAME = "was_suite_config.json"
V2_STATE_FILES = (
    V2_CONFIG_NAME,
    "was_suite_settings.json",
    "was_history.json",
    "styles.json",
    "nsp_pantry.json",
)

# v2 wrote its template out in full, so every one of these keys is present in a real
# user's file and each needs an answer, carried forward, or dropped with a reason.
HANDLED = frozenset(
    {
        "show_inspiration_quote",
        "wildcards_path",
        "webui_styles",
        "ffmpeg_extra_codecs",
        "history_display_limit",
    }
)

OBSOLETE = {
    "show_startup_junk": "the loader reports only what failed, so there is nothing to silence",
    "run_requirements": "v3 never pip-installs anything",
    "ffmpeg_bin_path": "video encodes in-process through av, so no binary is located or run",
    "suppress_uncomfy_warnings": "warnings go through the logging block",
    "text_nodes_type": "text is always STRING",
    "use_legacy_ascii_text": "text is always STRING",
    "webui_styles_persistent_update": "paths.styles is imported, never written back over",
    "wildcard_api": "never read by v2 either",
    "sam_model_vith_url": "SAM loads through transformers",
    "sam_model_vitl_url": "SAM loads through transformers",
    "sam_model_vitb_url": "SAM loads through transformers",
}

SAM_URL_DEFAULTS = {
    "sam_model_vith_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "sam_model_vitl_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "sam_model_vitb_url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}

UNSET = (None, "", "none", "None")
FFMPEG_PLACEHOLDER = "/path/to/ffmpeg"

HEADER = """\
# WAS Node Suite configuration, written once from your v2 was_suite_config.json.
# Every key, with its default and what it does: config.example.yaml in the repository.
"""


def run(source: Path, target: Path, derive_config: bool) -> Path | None:
    """Copy v2 state from ``source`` into ``target``; return the config written, if any.

    Args:
        source: The repository root, where v2 wrote its state and where
            ``config.example.yaml`` ships.
        target: The configuration directory the state and the config are written into.
        derive_config: Translate the v2 config. ``False`` copies state only.

    Returns:
        The config file written, or ``None`` when none was.
    """
    copy_state(source, target)
    if not derive_config:
        return None
    legacy = read_v2_config(target, source)
    if legacy is not None:
        written = write_config(target, forward(legacy))
        if written is not None:
            report_dropped(legacy)
            return written
    return install_example(source, target)


def copy_state(source: Path, target: Path) -> list[str]:
    pending = [
        name
        for name in V2_STATE_FILES
        if (source / name).is_file() and not (target / name).exists()
    ]
    if not pending:
        return []
    try:
        target.mkdir(parents=True, exist_ok=True)
        for name in pending:
            shutil.copy2(source / name, target / name)
    except OSError as error:
        logger.warning("could not copy your v2 files into %s (%s)", target, error)
        return []
    logger.info(
        "copied %s into %s; the originals in %s are untouched but no longer read",
        ", ".join(pending), target, source,
    )
    return pending


def read_v2_config(*directories: Path) -> Mapping | None:
    for directory in directories:
        path = directory / V2_CONFIG_NAME
        if not path.is_file():
            continue
        try:
            with path.open(encoding="utf-8") as handle:
                legacy = json.load(handle)
        except (OSError, ValueError) as error:
            logger.warning("%s could not be read (%s), so nothing was carried forward", path, error)
            return None
        if isinstance(legacy, Mapping):
            return legacy
        logger.warning("%s is not a JSON object, so nothing was carried forward", path)
        return None
    return None


def forward(legacy: Mapping) -> dict:
    """v2's flat settings -> a v3 config fragment holding only what the user can still set."""
    fragment: dict = {}

    def carry(section: str, key: str, value) -> None:
        fragment.setdefault(section, {})[key] = value

    if "show_inspiration_quote" in legacy:
        carry("logging", "quotes", bool(legacy["show_inspiration_quote"]))
    if "wildcards_path" in legacy:
        carry("paths", "wildcards", existing_path(legacy["wildcards_path"]))
    if "webui_styles" in legacy:
        carry("paths", "styles", existing_path(legacy["webui_styles"]))
    if isinstance(legacy.get("ffmpeg_extra_codecs"), Mapping):
        carry("video", "extra_codecs", dict(legacy["ffmpeg_extra_codecs"]))
    if isinstance(legacy.get("history_display_limit"), int):
        carry("history", "display_limit", legacy["history_display_limit"])
    return fragment


def existing_path(value) -> str | None:
    """A v2 path setting, or ``None`` when it was unset or now points at nothing."""
    # v2 defaulted wildcards_path to a directory inside the repository, so the value is
    # only worth carrying forward when that directory still exists.
    if value in UNSET or not isinstance(value, str):
        return None
    return value if os.path.exists(value) else None


def report_dropped(legacy: Mapping) -> None:
    if legacy.get("use_legacy_ascii_text") or str(legacy.get("text_nodes_type", "")).strip() == "ASCII":
        logger.warning(
            "you had ASCII text enabled (use_legacy_ascii_text): the ASCII type is gone and "
            "every text socket is STRING now. Saved workflows still load; the setting does not."
        )
    binary = legacy.get("ffmpeg_bin_path")
    if isinstance(binary, str) and binary.strip() not in UNSET and binary != FFMPEG_PLACEHOLDER:
        logger.warning(
            "ffmpeg_bin_path (%s) dropped: video encodes in-process through av now, so no "
            "ffmpeg binary is located or run. There is nothing to install and nothing to set.",
            binary,
        )
    tuned = [key for key, url in SAM_URL_DEFAULTS.items() if legacy.get(key) not in (None, url)]
    if tuned:
        logger.warning(
            "%s dropped: SAM loads through transformers now, so a checkpoint URL is never "
            "fetched. Point features.sam at local weights under models/sams instead.",
            ", ".join(sorted(tuned)),
        )
    dropped = [
        "{} ({})".format(key, OBSOLETE.get(key, "no v3 equivalent"))
        for key in legacy
        if key not in HANDLED
    ]
    if dropped:
        logger.debug("v2 settings not carried forward: %s", ", ".join(dropped))


def install_example(source: Path, target: Path) -> Path | None:
    """Copy ``config.example.yaml`` into a fresh install as its config file.

    Args:
        source: The repository root, holding ``config.example.yaml``.
        target: The configuration directory the copy is written to.

    Returns:
        The path written, or ``None`` when the example is missing, a config file is already
        there, or the directory cannot be written to. The copy carries the user's note in
        place of the template's.
    """
    # Without the copy, an install with no v2 settings to carry forward runs on built-in
    # defaults with nothing on disk, leaving no file to change a setting in and no sign of
    # where one would go.
    example = source / EXAMPLE_NAME
    path = target / "config.yaml"
    if path.exists() or not example.is_file():
        return None
    try:
        with open(example, encoding="utf-8", newline="") as handle:
            text = as_user_copy(handle.read())
        target.mkdir(parents=True, exist_ok=True)
        with open(path, "x", encoding="utf-8", newline="") as handle:
            handle.write(text)
    except (OSError, ValueError) as error:
        # A read-only install is a supported way to run, so this is worth saying once and
        # not worth failing over: the built-in defaults are what the copy would have held.
        logger.info(
            "no configuration file was written to %s (%s), so the built-in defaults apply; "
            "%s documents every key",
            target, error, EXAMPLE_NAME,
        )
        return None
    logger.info("wrote %s, a copy of %s with every key at its default", path, EXAMPLE_NAME)
    return path


def write_config(target: Path, fragment: dict) -> Path | None:
    path = target / "config.yaml"
    if not fragment or path.exists():
        # Whatever is there now was either written by this function on an earlier run or
        # hand-edited since. Either way it wins.
        return None
    import yaml

    target.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(HEADER)
        yaml.safe_dump({"version": VERSION, **fragment}, handle, sort_keys=False, allow_unicode=True)
    logger.info("wrote %s from your v2 settings; config.example.yaml documents every key", path)
    return path
