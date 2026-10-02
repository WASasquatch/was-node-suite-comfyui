"""Rewriting a config file to match the current schema.

An outdated file is rebuilt from ``config.example.yaml`` keeping its values, with
:data:`USER_NOTE` in place of the template's notice, and the previous file kept as
``config.yaml.bak``.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path

from .. import log
from .defaults import DEFAULTS, SUPERSEDED_FEATURE_DEFAULTS, VERSION

__all__ = [
    "TEMPLATE_NAME",
    "TEMPLATE_NOTE",
    "USER_NOTE",
    "as_user_copy",
    "differences",
    "reconcile",
    "render",
    "schema_paths",
]

logger = log.get_logger("config")

#: The commented schema, shipped in the repository root, used as the rewrite template.
TEMPLATE_NAME = "config.example.yaml"

#: How the template's paragraph saying it is not the file in use begins.
TEMPLATE_NOTE = "# THIS IS THE TEMPLATE"

#: The paragraph a user's own config file carries in place of the template's.
USER_NOTE = (
    "# This is the configuration file the pack reads. A change to it takes effect when",
    "# ComfyUI is restarted. Each release brings it up to date from config.example.yaml,",
    "# keeping the values set here.",
)

#: Where the file being replaced is kept.
BACKUP_SUFFIX = ".bak"

#: Suffix of a config this can rewrite. A JSON config carries no comments to preserve and
#: no template to preserve them from, so it is left alone and only reported on.
TEMPLATE_SUFFIX = ".yaml"


def schema_paths(defaults: Mapping = DEFAULTS) -> set[tuple]:
    """Every settable key in the schema, as a path tuple.

    Args:
        defaults: The schema. One block deep, matching what the loader merges.

    Returns:
        ``{("version",), ("logging", "level"), ...}``.
    """
    paths = set()
    for key, value in defaults.items():
        if isinstance(value, Mapping):
            paths.update((key, name) for name in value)
        else:
            paths.add((key,))
    return paths


def config_paths(raw: Mapping) -> set[tuple]:
    """The same, for a config file that has been read."""
    paths = set()
    for key, value in raw.items():
        block = DEFAULTS.get(key)
        if isinstance(value, Mapping) and isinstance(block, Mapping):
            paths.update((key, name) for name in value)
        else:
            paths.add((key,))
    return paths


def config_values(raw: Mapping) -> dict[tuple, object]:
    """A config file flattened to ``{path: value}``, keeping only what the schema names."""
    known = schema_paths()
    values = {}
    for key, value in raw.items():
        if isinstance(value, Mapping) and isinstance(DEFAULTS.get(key), Mapping):
            for name, setting in value.items():
                if (key, name) in known:
                    values[(key, name)] = setting
        elif (key,) in known:
            values[(key,)] = value
    return values


def adopt_new_defaults(raw: Mapping, values: dict) -> tuple[list[str], list[str]]:
    """Bring each ``features`` key still at an older version's default forward to this one's.

    Args:
        raw: The config file as parsed, including its ``version``.
        values: Flattened ``{path: value}`` from :func:`config_values`, updated in place.

    Returns:
        ``(adopted, kept)``, each a sorted list of dotted names, for the log. Both are
        empty when the version is not one this build supersedes.
    """
    stored = raw.get("version")
    previous = SUPERSEDED_FEATURE_DEFAULTS.get(stored)
    if previous is None:
        return [], []

    adopted, kept = [], []
    for name, default in DEFAULTS["features"].items():
        path = ("features", name)
        if path not in values:
            # The file does not name this group, so the template's own default is what is
            # rendered and `differences` reports it as a key this release added.
            continue
        if name not in previous:
            # A group this record does not put in that version's block, so there is no
            # default to recognise and no case for touching what the file holds.
            continue
        if values[path] != previous[name]:
            # Not the answer that version wrote, so it is the user's and it stands. Every
            # move is named in the log and the previous file is kept.
            kept.append(f"features.{name}")
            continue
        if values[path] == default:
            # That version's default and this one's are the same answer.
            continue
        # The file holds the default that version was given, so this one's replaces it.
        values[path] = default
        adopted.append(f"features.{name}")
    return sorted(adopted), sorted(kept)


def stamp_version(values: dict) -> None:
    """Record the version this build writes, so a migration runs once and not every start.

    Args:
        values: Flattened ``{path: value}`` from :func:`config_values`, updated in place.
    """
    values[("version",)] = VERSION


def differences(raw: Mapping) -> tuple[list[str], list[str]]:
    """What the schema has that this file lacks, and what it holds that the schema drops.

    Returns:
        ``(added, removed)``, each a sorted list of dotted key names.
    """
    present = config_paths(raw)
    expected = schema_paths()
    # A key the file does not name cannot be told from one removed on purpose, so a config
    # trimmed by hand is treated as one that predates the settings it is missing.
    added = sorted(".".join(path) for path in expected - present)
    removed = sorted(".".join(path) for path in present - expected)
    return added, removed


def reconcile(path: Path, raw: Mapping, source: Path) -> bool:
    """Bring one config file up to the current schema, keeping the values in it.

    Args:
        path: The config file in use.
        raw: That file, already parsed.
        source: The repository root, holding :data:`TEMPLATE_NAME`.

    Returns:
        Whether the file was rewritten, including a rewrite of the template's own paragraph
        alone. ``False`` covers every reason not to, each of which is logged: nothing to do,
        no template, a format this cannot template, and a write that failed.
    """
    added, removed = differences(raw)
    # A version this build supersedes is a reason to rewrite on its own: the defaults it
    # was written against have moved, and the file records which ones they were.
    superseded = raw.get("version") in SUPERSEDED_FEATURE_DEFAULTS
    if not added and not removed and not superseded:
        return _drop_template_note(path, source)

    if path.suffix != TEMPLATE_SUFFIX:
        logger.info(
            "%s is behind this release: %s. It is never rewritten, because there are no "
            "comments in it to carry across. Change it by hand, or delete it and let a "
            "YAML one be written in its place",
            path,
            "it does not carry {}".format(", ".join(added)) if added else
            "it was written against version {}, and this build reads version {}".format(
                raw.get("version"), VERSION,
            ),
        )
        return False

    # Templated rather than round-tripped: dumping a parsed mapping back out would write
    # the same settings with every comment gone, and the comments are the documentation.
    template = source / TEMPLATE_NAME
    try:
        text = template.read_text(encoding="utf-8")
    except OSError as error:
        logger.warning(
            "%s could not be read (%s), so %s was left as it is. The settings it does not "
            "name are at their defaults",
            template, error, path,
        )
        return False

    values = config_values(raw)
    adopted, kept = adopt_new_defaults(raw, values)
    stamp_version(values)
    rewritten, filled = render(text, values)
    if not _is_template(path, source):
        rewritten = as_user_copy(rewritten)
    missing = sorted(".".join(p) for p in schema_paths() - filled)
    if missing:
        # The template is shipped alongside the schema, so this is a packaging fault
        # rather than anything the user did. Say so, and leave their file alone: a rewrite
        # from an incomplete template would drop settings that do exist.
        logger.warning(
            "%s does not name %s, so %s was not updated. This is a fault in the release, "
            "not in your configuration",
            TEMPLATE_NAME, ", ".join(missing), path,
        )
        return False

    backup = path.with_suffix(path.suffix + BACKUP_SUFFIX)
    try:
        shutil.copyfile(path, backup)
        path.write_text(rewritten, encoding="utf-8")
    except OSError as error:
        logger.warning(
            "%s could not be updated (%s), so it is unchanged. Every setting it does not "
            "name is at its default",
            path, error,
        )
        return False

    _report(path, backup, added, removed, adopted, kept, raw.get("version"))
    return True


def as_user_copy(text: str) -> str:
    """A config file's text with the template's paragraph replaced by :data:`USER_NOTE`.

    Args:
        text: The template's text, or a config file copied from it.

    Returns:
        The text with the comment paragraph opening with :data:`TEMPLATE_NOTE` replaced, up
        to the next bare ``#``, blank or setting line. Unchanged where there is none.
    """
    lines = text.splitlines(keepends=True)
    start = next(
        (index for index, line in enumerate(lines) if line.startswith(TEMPLATE_NOTE)), None
    )
    if start is None:
        return text
    end = start + 1
    while end < len(lines):
        stripped = lines[end].strip()
        if not stripped.startswith("#") or stripped == "#":
            break
        end += 1
    newline = "\r\n" if lines[start].endswith("\r\n") else "\n"
    note = "".join(line + newline for line in USER_NOTE)
    return "".join(lines[:start]) + note + "".join(lines[end:])


def _is_template(path: Path, source: Path) -> bool:
    """Whether ``path`` is the template itself."""
    try:
        return path.resolve() == (source / TEMPLATE_NAME).resolve()
    except OSError:
        return False


def _drop_template_note(path: Path, source: Path) -> bool:
    """Replace the template's paragraph in a user's config file, touching nothing else.

    Args:
        path: The config file in use.
        source: The repository root, holding :data:`TEMPLATE_NAME`.

    Returns:
        Whether the file was rewritten.
    """
    if path.suffix != TEMPLATE_SUFFIX or _is_template(path, source):
        return False
    try:
        with open(path, encoding="utf-8", newline="") as handle:
            text = handle.read()
    except (OSError, ValueError):
        return False
    rewritten = as_user_copy(text)
    if rewritten == text:
        return False
    try:
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(rewritten)
    except OSError as error:
        logger.debug("%s could not be updated (%s)", path, error)
        return False
    logger.info("%s now says it is the file the pack reads, rather than the template", path)
    return True


def render(text: str, values: Mapping) -> tuple[str, set[tuple]]:
    """Substitute values into the template, keeping its comments and its layout.

    Args:
        text: The template, read from :data:`TEMPLATE_NAME`.
        values: ``{path: value}`` from :func:`config_values`. A path the template names
            and this does not keeps the template's default.

    Returns:
        ``(rendered text, the paths the template named)``. The second is what proves the
        template still covers the whole schema.
    """
    lines = []
    named: set[tuple] = set()
    block: str | None = None

    for line in text.splitlines():
        parsed = _split(line)
        if parsed is None:
            lines.append(line)
            continue
        indent, key, head, body = parsed
        if indent == 0:
            block = key
        path = (key,) if indent == 0 else (block, key)

        if not _has_value(body):
            lines.append(line)
            continue

        named.add(path)
        if path not in values:
            lines.append(line)
            continue
        lines.append(head + _body(body, values[path]))

    return "\n".join(lines) + "\n", named


def _split(line: str):
    """``(indent, key, text up to and including the colon, the rest)`` for a mapping line."""
    stripped = line.lstrip(" ")
    indent = len(line) - len(stripped)
    if not stripped or stripped.startswith("#"):
        return None
    name, separator, body = stripped.partition(":")
    if not separator or not name or not name.replace("_", "").isalnum():
        return None
    return indent, name, line[: indent + len(name) + 1], body


def _has_value(body: str) -> bool:
    """Does this line set a value, rather than open a block?"""
    text = body.strip()
    return bool(text) and not text.startswith("#")


def _body(body: str, value) -> str:
    """The part of a line after the colon, with a new value and the comment kept in place."""
    value_text, hash_mark, comment = body.partition("#")
    lead = len(value_text) - len(value_text.lstrip(" "))
    old = value_text.strip()
    gap = len(value_text) - lead - len(old)
    new = _scalar(value)
    if not hash_mark:
        return " " * lead + new
    return " " * lead + new + " " * max(1, gap + len(old) - len(new)) + hash_mark + comment


def _scalar(value) -> str:
    """One value as it is written inline in YAML."""
    import yaml

    text = yaml.safe_dump(value, default_flow_style=True, allow_unicode=True, width=10**6)
    for marker in ("\n...", "..."):
        if text.rstrip().endswith(marker):
            text = text.rstrip()[: -len(marker)]
            break
    return text.strip()


def _report(
    path: Path,
    backup: Path,
    added: list[str],
    removed: list[str],
    adopted: list[str] = (),
    kept: list[str] = (),
    stored=None,
) -> None:
    lines = [f"{path} was updated to this release's settings, keeping your values:"]
    if added:
        lines.append(f"    added   {', '.join(added)}")
    if removed:
        lines.append(f"    removed {', '.join(removed)}, no longer read by this build")
    for state, names in _adopted_by_state(adopted):
        lines.append(f"    turned {state} {', '.join(names)}")
    if kept:
        names = ", ".join(kept)
        lines.append(f"    kept    {names}, which is not the answer version {stored} wrote")
    if adopted:
        lines.append("    nothing was installed for those, and nothing runs until a node or a")
        lines.append(f"    format that needs one is used. Each still held version {stored}'s own")
        lines.append("    default, which is all there is to go on: nothing on disk records")
        lines.append("    whether a value was chosen or written for you, so a group you set to")
        lines.append("    the answer it already had reads the same as one you never touched.")
        lines.append("    Set any of them back by hand to undo this")
    if not added and not removed and not adopted:
        lines.append("    no setting moved; the file now records the version of the schema it")
        lines.append("    was rewritten against, so this happens once and not on every start")
    lines.append(f"    the file as it was is at {backup.name}")
    logger.info("\n".join(lines))


def _adopted_by_state(adopted) -> list[tuple[str, list[str]]]:
    """Adopted settings split into the ones now on and the ones now off, in that order.

    Args:
        adopted: Dotted ``features.*`` names, from :func:`adopt_new_defaults`.

    Returns:
        ``[("on", names), ("off", names)]``, each pair present only when it has names.
    """
    groups = {"on": [], "off": []}
    for name in adopted:
        groups["on" if DEFAULTS["features"].get(name.partition(".")[2]) else "off"].append(name)
    return [(state, names) for state, names in groups.items() if names]
