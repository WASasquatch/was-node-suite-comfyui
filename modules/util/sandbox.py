"""Containment for every filesystem path a node accepts from a user.

:func:`resolve_read` and :func:`resolve_write` map an untrusted input onto an absolute
path inside a permitted root, or raise :class:`PathNotAllowed`. Writes exclude ComfyUI's
``input`` directory.
"""

from __future__ import annotations

import glob
import os
import unicodedata
from pathlib import Path, PureWindowsPath
from typing import Iterable

from .. import log
from ..config import WINDOWS_PATH_FIX, load_config, paths

__all__ = [
    "PathNotAllowed",
    "annotated_path",
    "contains",
    "names_another_host",
    "configured_read_roots",
    "configured_write_roots",
    "glob_read",
    "leaves_folder",
    "read_roots",
    "refuse_linked_host",
    "resolve_read",
    "resolve_write",
    "resolve_write_file",
    "save_image_path",
    "write_roots",
]

logger = log.get_logger("util.sandbox")

#: Directory name the frozen ``./ComfyUI/...`` widget defaults use for ComfyUI's own tree.
COMFY_DIRECTORY = "ComfyUI"

#: Verbs the resolution errors are phrased with, and the only two values ``purpose`` takes.
READ = "read"
WRITE = "write"

#: Config key naming the extra roots for each purpose.
KEYS = {READ: "paths.allow_read", WRITE: "paths.allow_write"}

#: Entries of the pack's state directory no node writes: its settings, and the folder view
#: extensions are installed from.
PROTECTED = ("config.yaml", "config.json", "viewer-extensions")

#: Names in the pack's state directory no node reads or writes, each with what it holds. A
#: name ending in ``*`` stands for every name that begins with the rest of it, so
#: ``config.yaml*`` covers ``config.yaml.bak`` and ``was_state.db*`` covers ``was_state.db-wal``.
PRIVATE = {
    "config.yaml*": "settings",
    "config.json*": "settings",
    "was_state.db*": "settings database",
    "was_suite_config.json*": "v2 settings",
    "was_suite_settings.json*": "v2 settings",
    "was_history.json*": "v2 file history",
}

#: Characters of a refused value repeated in the refusal.
SHOWN_LENGTH = 240

#: Names Windows opens as a device in every folder, whatever extension follows them.
DEVICE_NAMES = frozenset({
    "CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$",
    *(f"COM{digit}" for digit in "0123456789\u00b9\u00b2\u00b3"),
    *(f"LPT{digit}" for digit in "0123456789\u00b9\u00b2\u00b3"),
})

#: Links followed through one another before a path is refused.
MAX_LINKS = 16


class PathNotAllowed(ValueError):
    """A user-supplied path the containment layer refuses."""


def _shown(value) -> str:
    """A refused value as given, for the refusal to repeat.

    Args:
        value: The value as it reached the containment layer.

    Returns:
        The value stripped, with each unprintable character written as an escape, cut to
        :data:`SHOWN_LENGTH` characters.
    """
    text = "".join(
        character if character.isprintable() else f"\\u{ord(character):04x}"
        for character in str(value).strip()
    )
    return text if len(text) <= SHOWN_LENGTH else text[:SHOWN_LENGTH] + "..."


def _comfy_directory(name: str) -> Path | None:
    """Return one of ComfyUI's directories, or ``None`` outside ComfyUI.

    Args:
        name: Attribute on ``folder_paths``, such as ``"get_input_directory"``.

    Returns:
        The resolved directory, or ``None`` when folder_paths is unavailable or the
        directory does not exist.
    """
    try:
        import folder_paths
    except ImportError:
        return None
    getter = getattr(folder_paths, name, None)
    if getter is None:
        return None
    try:
        value = getter()
    except Exception:
        return None
    return Path(value).expanduser().resolve() if value else None


def _comfy_root() -> Path | None:
    """Return ComfyUI's own directory, the one holding ``input/``, ``output/`` and ``temp/``.

    Returns:
        The resolved directory, or ``None`` outside ComfyUI or where folder_paths does not
        carry ``base_path``.
    """
    try:
        import folder_paths
    except ImportError:
        return None
    value = getattr(folder_paths, "base_path", None)
    if not value:
        return None
    try:
        return Path(value).expanduser().resolve()
    except OSError:
        return None


def _configured(key: str) -> list[Path]:
    """Return the extra roots listed under a config key.

    Args:
        key: Either ``"allow_read"`` or ``"allow_write"``.

    Returns:
        Resolved directories. Entries that are not directories are dropped with a warning.
    """
    configured = (load_config().get("paths") or {}).get(key) or []
    if isinstance(configured, (str, os.PathLike)):
        configured = [configured]
    if not isinstance(configured, (list, tuple)):
        _warn_once(
            key, configured,
            f"paths.{key} should be a list of folders, not {configured!r}, so it is ignored. "
            f"Write it as ['D:/prompts'].",
        )
        return []
    roots = []
    for entry in configured:
        if not isinstance(entry, (str, os.PathLike)):
            _warn_once(
                key, entry,
                f"paths.{key} entry {entry!r} is not a path and is ignored. Write it as a "
                f"quoted path, as 'D:/prompts'.",
            )
            continue
        try:
            root = Path(entry).expanduser().resolve()
        except (OSError, ValueError, RuntimeError):
            root = None
        if root is None or not root.is_dir():
            _warn_once(key, entry, _unusable(key, entry, root))
            continue
        roots.append(root)
    return roots


#: Characters a backslash escape in a double-quoted YAML string turns into, and the escape.
YAML_ESCAPES = {
    "\0": "\\0", "\a": "\\a", "\b": "\\b", "\t": "\\t", "\n": "\\n", "\v": "\\v",
    "\f": "\\f", "\r": "\\r", "\x1b": "\\e", "\x85": "\\N", "\xa0": "\\_",
    "\u2028": "\\L", "\u2029": "\\P",
}

#: ``(key, entry)`` pairs already warned about, each logged once per process.
_warned: set[tuple[str, str]] = set()


def _warn_once(key: str, entry, message: str) -> None:
    """Log one warning about a configured entry, the first time it is seen.

    Args:
        key: ``"allow_read"`` or ``"allow_write"``.
        entry: The entry as configured.
        message: What to log.
    """
    seen = (key, repr(entry))
    if seen in _warned:
        return
    _warned.add(seen)
    logger.warning("%s", message)


def _stray(text: str) -> list[str]:
    """Control and separator characters in a configured path, as ``U+XXXX`` names.

    Args:
        text: The entry as configured.

    Returns:
        Each such character once, in order of first appearance. Empty for an ordinary path.
    """
    found: list[str] = []
    for character in text:
        category = unicodedata.category(character)
        stray = category.startswith("C") or category in ("Zl", "Zp")
        if stray or (category == "Zs" and character != " "):
            name = f"U+{ord(character):04X}"
            if name not in found:
                found.append(name)
    return found


def _unusable(key: str, entry, root: Path | None) -> str:
    """The warning for a configured entry that names no directory.

    Args:
        key: ``"allow_read"`` or ``"allow_write"``.
        entry: The entry as configured.
        root: The entry resolved, or None where it could not be.

    Returns:
        A message naming the entry and the fix. An entry holding a control or separator
        character is named as it was most likely typed, with its backslash escapes.
    """
    text = os.fspath(entry)
    stray = _stray(text)
    if stray:
        typed = "".join(YAML_ESCAPES.get(character, character) for character in text)
        return (
            f"paths.{key} entry \"{typed}\" is ignored: it was read as {text!r}, holding "
            f"{', '.join(stray)}, because a backslash inside double quotes starts an escape "
            f"sequence. {WINDOWS_PATH_FIX}"
        )
    if root is None:
        return f"paths.{key} entry {text} cannot be resolved and is ignored."
    return f"paths.{key} entry {root} is not a directory and is ignored."


def _pack_roots() -> list[Path]:
    """Return the pack's own state directory, where wildcards and styles live."""
    # config_directory(), not user_directory(): the latter is ComfyUI's whole user tree and
    # holds default/comfy.settings.json and default/workflows/, which a contained writer
    # must not reach. Every state file this pack owns resolves through paths.state_file(),
    # already inside the narrower directory.
    try:
        return [paths.config_directory().resolve()]
    except Exception:
        return []


def configured_read_roots() -> list[Path]:
    """Directories ``paths.allow_read`` names, without ComfyUI's own or this pack's."""
    return _configured("allow_read")


def configured_write_roots() -> list[Path]:
    """Directories ``paths.allow_write`` names, without ComfyUI's own or this pack's."""
    return _configured("allow_write")


def read_roots() -> list[Path]:
    """Directories a node may read from, most specific first."""
    roots = [
        _comfy_directory("get_input_directory"),
        _comfy_directory("get_output_directory"),
        _comfy_directory("get_temp_directory"),
    ]
    return [root for root in roots if root is not None] + _pack_roots() + _configured("allow_read")


def write_roots() -> list[Path]:
    """Directories a node may write to, most specific first."""
    roots = [
        _comfy_directory("get_output_directory"),
        _comfy_directory("get_temp_directory"),
    ]
    return [root for root in roots if root is not None] + _pack_roots() + _configured("allow_write")


def _state_label() -> str:
    """The pack's state directory as a refusal names it."""
    if paths.env_value(paths.ENV_CONFIG_DIR):
        return f"${paths.ENV_CONFIG_DIR}"
    return f"{COMFY_DIRECTORY}/user/{paths.CONFIG_DIR_NAME}"


def _named_roots() -> list[tuple[str, Path]]:
    """Every permitted root with the name a message gives it in place of its full path."""
    named = []
    for name, getter in (
        ("input", "get_input_directory"),
        ("output", "get_output_directory"),
        ("temp", "get_temp_directory"),
    ):
        directory = _comfy_directory(getter)
        if directory is not None:
            named.append((name, directory))
    named.extend((_state_label(), root) for root in _pack_roots())
    for root in (*_configured("allow_read"), *_configured("allow_write")):
        named.append((root.name or "a configured folder", root))
    return named


def _label(folder: str | os.PathLike) -> str:
    """A folder named by the permitted root it sits in and the path below that root.

    Args:
        folder: An absolute folder, normally one already resolved inside a root.

    Returns:
        Such as ``output/renders``, or ``the folder it was given`` for one outside every
        root. Never the full path.
    """
    try:
        target = Path(os.path.abspath(os.fspath(folder)))
    except (TypeError, ValueError):
        return "the folder it was given"
    best: tuple[str, Path] | None = None
    for name, root in _named_roots():
        if contains(root, target) and (best is None or len(root.parts) > len(best[1].parts)):
            best = (name, root)
    if best is None:
        return "the folder it was given"
    below = os.path.relpath(target, best[1]).replace("\\", "/")
    return best[0] if below == "." else f"{best[0]}/{below}"


def _permitted(purpose: str) -> str:
    """The roots a refusal lists, by name, with how many folders the config key adds.

    Args:
        purpose: :data:`READ` or :data:`WRITE`.

    Returns:
        Such as ``input, output, temp, ComfyUI/user/was-node-suite, and the 2 folders
        paths.allow_read lists``.
    """
    getters = [("output", "get_output_directory"), ("temp", "get_temp_directory")]
    if purpose == READ:
        getters.insert(0, ("input", "get_input_directory"))
    names = [name for name, getter in getters if _comfy_directory(getter) is not None]
    if _pack_roots():
        names.append(_state_label())
    key = KEYS[purpose]
    count = len(_configured(key.rsplit(".", 1)[-1]))
    if count == 0:
        listed = f"{key} lists no folder"
    elif count == 1:
        listed = f"the 1 folder {key} lists"
    else:
        listed = f"the {count} folders {key} lists"
    return f"{', '.join(names)}, and {listed}" if names else listed


def contains(root: Path, target: Path) -> bool:
    """Report whether ``target`` is ``root`` or lies beneath it.

    Args:
        root: A permitted root, already resolved.
        target: The candidate path, already resolved.

    Returns:
        True when target is inside root. Comparison is case-insensitive on Windows.
    """
    if os.name == "nt":
        root = Path(os.path.normcase(str(root)))
        target = Path(os.path.normcase(str(target)))
    return root == target or root in target.parents


def names_another_host(value: str | os.PathLike) -> bool:
    r"""Whether a path names a host rather than this machine.

    Args:
        value: The raw path, as written.

    Returns:
        True for a UNC path such as ``\\server\share\file`` and for any path under the NT
        object prefix ``\??\``. Reading one reaches that host over the network, which
        resolving the path is enough to do.
    """
    text = str(value).strip()
    # The NT object prefix reaches a share without a UNC drive: \??\UNC\server\share.
    if text.replace("/", "\\").startswith("\\??\\"):
        return True
    drive = PureWindowsPath(text).drive
    return drive.startswith("\\\\") or drive.startswith("//")


def _windows_name(part: str) -> str | None:
    """Why one path segment names something other than a file on Windows, or ``None``.

    Args:
        part: One segment, with no separator.

    Returns:
        A reason in words for a drive such as ``C:``, a segment holding ``:``, which names a
        stream of another file, or one naming a device such as ``NUL`` or ``COM1.txt``.
        ``None`` off Windows.
    """
    if os.name != "nt":
        return None
    if len(part) == 2 and part[1] == ":" and part[0].isalpha():
        return "carries a drive"
    if ":" in part:
        return "names a stream of another file with ':'"
    stem = part.split(".", 1)[0].rstrip(" ").upper()
    if stem in DEVICE_NAMES:
        return f"names the Windows device {stem}"
    return None


def leaves_folder(text: str | os.PathLike) -> str | None:
    r"""Why a relative path or glob pattern would name somewhere other than inside its folder.

    Args:
        text: The path or pattern as written, to be joined onto a folder.

    Returns:
        ``None`` when it stays inside, otherwise what it does, in words: it names another
        machine (``\\server\share`` or ``//server/share``), carries a drive, starts at a
        root, holds a segment of dots and spaces alone such as ``..`` or ``.. ``, or on
        Windows holds a segment naming a stream or a device.
    """
    value = str(text).strip()
    if names_another_host(value):
        return "names another machine"
    relative = PureWindowsPath(value)
    if relative.drive:
        return "carries a drive"
    if relative.root:
        return "starts at a filesystem root"
    if any(not part.strip(" .") for part in relative.parts):
        return "climbs out with '..'"
    for part in relative.parts:
        reason = _windows_name(part)
        if reason is not None:
            return reason
    return None


def glob_read(directory: str | os.PathLike, pattern: str, recursive: bool = False) -> list[str]:
    """Every path inside a folder that a glob pattern matches.

    Args:
        directory: The folder, already resolved inside a permitted read root.
        pattern: Glob pattern, matched under ``directory``.
        recursive: Let ``**`` in the pattern cross folders.

    Returns:
        Matches as glob spelled them, unsorted, every one of them beneath ``directory``.

    Raises:
        PathNotAllowed: The pattern is empty, or :func:`leaves_folder` gives a reason. The
            pattern is refused before anything is read.
    """
    text = str(pattern).strip()
    if not text:
        raise PathNotAllowed(f"no pattern given to match in {_label(directory)}")
    reason = leaves_folder(text)
    if reason is not None:
        raise PathNotAllowed(
            f"the pattern `{_shown(text)}` {reason}, so it would match outside "
            f"{_label(directory)}. A pattern matches inside the folder it is given, such as "
            f"`*.png` or `shots/*.png`; pick the outer folder as the folder instead."
        )
    base = os.path.normpath(str(directory))
    found = glob.glob(os.path.join(glob.escape(base), text), recursive=recursive)
    return [name for name in found if contains(Path(base), Path(os.path.normpath(name)))]


def annotated_path(name: str) -> str | None:
    """The file one of ComfyUI's annotated names, such as ``plate.png [output]``, refers to.

    Args:
        name: The name as a menu or a workflow holds it.

    Returns:
        The absolute path, or ``None`` when the name is empty, names nothing that is there,
        or would leave ComfyUI's folder as :func:`leaves_folder` reads it. That last test
        runs before ComfyUI resolves anything.
    """
    text = str(name or "").strip()
    if not text or leaves_folder(text) is not None:
        return None
    import folder_paths

    bare, _ = folder_paths.annotated_filepath(text)
    if not str(bare or "").strip() or leaves_folder(bare) is not None:
        return None
    if not folder_paths.exists_annotated_filepath(text):
        return None
    return folder_paths.get_annotated_filepath(text)


def save_image_path(prefix: str, directory: str | os.PathLike, width: int = 0, height: int = 0):
    """ComfyUI's numbered save location for a prefix, the prefix refused first if it leaves.

    Args:
        prefix: The file name prefix, which may carry folders below ``directory``.
        directory: The folder the files are written in.
        width: Image width, for the ``%width%`` token.
        height: Image height, for the ``%height%`` token.

    Returns:
        ``(folder, name, counter, subfolder, prefix)`` as ComfyUI's ``get_save_image_path``.

    Raises:
        PathNotAllowed: :func:`leaves_folder` gives a reason for ``prefix``. Nothing is
            resolved before this.
    """
    reason = leaves_folder(prefix)
    if reason is not None:
        raise PathNotAllowed(
            f"the filename prefix `{_shown(prefix)}` {reason}, so it would be written outside "
            f"{_label(directory)}. Use a prefix such as `renders/shot`"
        )
    below = os.path.dirname(str(prefix).replace("\\", "/"))
    if below:
        resolve_write_file(directory, below)
    import folder_paths

    return folder_paths.get_save_image_path(prefix, str(directory), width, height)


def _host_permitted(text: str, roots: list[Path]) -> bool:
    r"""Whether a permitted root sits on the same host and share as ``text``.

    Args:
        text: The raw path, as written.
        roots: Permitted roots.

    Returns:
        True when a root names the same ``\\server\share``. Compared as written, so no path
        is resolved to answer this.
    """
    drive = PureWindowsPath(text).drive.replace("/", "\\").lower()
    return any(
        PureWindowsPath(str(root)).drive.replace("/", "\\").lower() == drive
        for root in roots
    )


def _comfy_relative(text: str) -> tuple[Path, tuple[str, ...]] | None:
    """Split ``./ComfyUI/output/x`` into ComfyUI's root and the segments below it.

    Args:
        text: The raw widget value, stripped.

    Returns:
        ``(root, segments)``, read from the text alone. ``None`` when the value is absolute
        or carries a drive, when its leading component is not ComfyUI's directory, or when
        that directory is unknown.
    """
    relative = PureWindowsPath(text)
    if relative.drive or relative.root:
        return None
    parts = relative.parts
    if not parts:
        return None
    root = _comfy_root()
    if root is None:
        return None
    if parts[0].casefold() not in {COMFY_DIRECTORY.casefold(), root.name.casefold()}:
        return None
    return root, parts[1:]


def _rebased(text: str) -> Path | None:
    """Read ``./ComfyUI/output/x`` against ComfyUI's own root.

    Args:
        text: The raw widget value, stripped.

    Returns:
        The value with its leading ``ComfyUI`` component replaced by ComfyUI's root, so the
        frozen defaults name ComfyUI's own tree whatever directory the process was started
        in. ``None`` where :func:`_comfy_relative` gives ``None``.
    """
    split = _comfy_relative(text)
    if split is None:
        return None
    root, below = split
    # '..' segments in the remainder survive the join and are collapsed by resolve, so a
    # rebased value can still land above ComfyUI's root, where the root check refuses it.
    _refuse_linked_host(Path(os.path.abspath(root.joinpath(*below))), text)
    return root.joinpath(*below).resolve(strict=False)


def _candidates(text: str) -> list[Path]:
    """Return the absolute paths ``text`` may name, in the order they are tried.

    Args:
        text: The raw widget value, stripped.

    Returns:
        The value read absolute as written, or relative against the process working
        directory, followed, for a ``./ComfyUI/...`` value only, by the same value read
        against ComfyUI's root.
    """
    # strict=False so a write target that does not exist yet still resolves; parents and
    # symlinks along the existing prefix are resolved either way.
    _refuse_linked_host(Path(os.path.abspath(os.path.expanduser(text))), text)
    found = [Path(text).expanduser().resolve(strict=False)]
    rebased = _rebased(text)
    if rebased is not None and rebased != found[0]:
        found.append(rebased)
    return found


def _relocated(target: Path) -> Path | None:
    """Return where a write into ComfyUI's input directory goes instead.

    Args:
        target: A resolved path that landed outside every write root.

    Returns:
        The matching path under ComfyUI's temp directory, or ``None`` when ``target`` is
        not inside the input directory or either directory is unknown.
    """
    source = _comfy_directory("get_input_directory")
    # The temp directory is a write root, is readable by a loader afterwards, and is
    # cleaned up with the rest of a run's scratch data.
    temp = _comfy_directory("get_temp_directory")
    if source is None or temp is None or not contains(source, target):
        return None
    return temp / os.path.relpath(target, source)


def _refusal(text: str, key: str, purpose: str) -> PathNotAllowed:
    """The error for a path outside every permitted root.

    Args:
        text: The value as given.
        key: Config key naming the list that would permit it.
        purpose: :data:`READ` or :data:`WRITE`.

    Returns:
        An error naming the value as given, each permitted root by name, and the key. Its
        text is the same whatever is or is not on disk at that path.
    """
    message = (
        f"refusing to {purpose} `{_shown(text)}`\n"
        f"  It is outside every folder this pack may {purpose}: {_permitted(purpose)}.\n"
        f"  Add the folder to {key} in config.yaml to permit it. A link leading to another "
        f"machine is not followed even then."
    )
    if _comfy_relative(text) is not None:
        message += "\n  It was read against ComfyUI's own folder as well, and refused there too."
    return PathNotAllowed(message)


def _resolve(value: str | os.PathLike, roots: Iterable[Path], key: str, purpose: str) -> Path:
    """Resolve a user-supplied path and confirm it lies within one of ``roots``.

    Args:
        value: The raw widget value.
        roots: Permitted roots.
        key: Config key naming the list that would permit an outside path.
        purpose: Verb used in the error message, :data:`READ` or :data:`WRITE`.

    Returns:
        The resolved absolute path. It is not required to exist; a write target normally
        does not.

    Raises:
        PathNotAllowed: The path is empty, resolved outside every permitted root, names a
            Windows device or stream, or is one of the pack's own settings files.
    """
    text = str(value).strip()
    if not text:
        raise PathNotAllowed(f"no path given to {purpose}")
    roots = list(roots)
    reached = os.path.expanduser(text)
    if names_another_host(reached) and not _host_permitted(reached, roots):
        raise PathNotAllowed(
            f"`{_shown(text)}` names another machine. A path is {purpose} from this computer "
            f"only. Add the share to {key} in config.yaml to reach it."
        )
    try:
        candidates = _candidates(text)
    except (OSError, RuntimeError, ValueError) as error:
        logger.debug("refused to %s %r before resolving it: %s", purpose, text, error)
        raise _refusal(text, key, purpose) from None
    for target in candidates:
        for root in roots:
            if contains(root, target):
                _refuse_device(target, text, purpose)
                _refuse_protected(target, text, purpose)
                return target
    if purpose == WRITE:
        for target in candidates:
            moved = _relocated(target)
            if moved is not None:
                logger.warning(
                    "%s is in ComfyUI's input directory, which holds uploads and is not "
                    "written to; writing to %s instead. Add the input directory to "
                    "%s in config.yaml to write there.",
                    target, moved, key,
                )
                return moved
    logger.debug(
        "refused to %s %r, resolved as %s, outside %s", purpose, text,
        ", ".join(str(target) for target in candidates), ", ".join(str(root) for root in roots),
    )
    raise _refusal(text, key, purpose)


def _join(parent: Path, name: str | os.PathLike, purpose: str) -> Path:
    """Place a file name inside an already-resolved directory.

    Args:
        parent: Resolved directory, already confirmed to be inside a permitted root.
        name: File name to place in it. Sub-directories are allowed.
        purpose: Verb used in the error message, :data:`READ` or :data:`WRITE`.

    Returns:
        The resolved absolute path of the file, inside ``parent``.

    Raises:
        PathNotAllowed: The name is empty, names somewhere other than inside ``parent``, or
            resolves out of it through a symlink.
    """
    text = str(name).strip()
    if not text:
        raise PathNotAllowed(f"no file name given to {purpose} in {_label(parent)}")
    reason = leaves_folder(text)
    if reason is not None:
        raise PathNotAllowed(
            f"refusing to {purpose} `{_shown(text)}` in {_label(parent)}\n"
            f"  A file name is a name inside that directory, and this one {reason}.\n"
            f"  Joining it onto the directory would discard the directory and "
            f"{purpose} somewhere else entirely."
        )
    _refuse_linked_host(
        Path(os.path.abspath(parent.joinpath(*PureWindowsPath(text).parts))), text)
    target = parent.joinpath(*PureWindowsPath(text).parts).resolve(strict=False)
    if not contains(parent, target):
        logger.debug("refused to %s %r in %s, resolved as %s", purpose, text, parent, target)
        raise PathNotAllowed(
            f"refusing to {purpose} `{_shown(text)}` in {_label(parent)}\n"
            f"  It leaves that folder through a link that points out of it."
        )
    if purpose == WRITE:
        _refuse_device(target, text, purpose)
        _refuse_protected(target, text, purpose)
    return target


def _refuse_linked_host(path: Path, shown: str, depth: int = 0) -> None:
    """Refuse a path whose links lead to another machine, reading each link without following it.

    Args:
        path: An absolute path, as written; nothing along it is resolved.
        shown: The value as given, which a refusal names in place of ``path``.
        depth: Links already followed to reach it.

    Raises:
        PathNotAllowed: A link along ``path`` points at another machine, or links nest past
            :data:`MAX_LINKS`. The message names ``shown`` and never a link's target.
    """
    if depth > MAX_LINKS:
        logger.debug("%s goes through more than %d links", path, MAX_LINKS)
        raise PathNotAllowed(
            f"`{_shown(shown)}` goes through more than {MAX_LINKS} links, so it is not read")
    junction = getattr(os.path, "isjunction", None)
    # From the root down, so every link above a step is read before the step is looked at.
    for step in reversed((path, *path.parents)):
        try:
            linked = os.path.islink(step) or bool(junction and junction(step))
            target = os.readlink(step) if linked else ""
        except OSError:
            continue
        if not target:
            continue
        if names_another_host(target):
            logger.debug("%s is a link to %s, which names another machine", step, target)
            raise PathNotAllowed(
                f"`{_shown(shown)}` reaches another machine through a link, so it is not "
                f"followed"
            )
        nested = Path(target) if os.path.isabs(target) else step.parent / target
        _refuse_linked_host(Path(os.path.abspath(nested)), shown, depth + 1)


def refuse_linked_host(path: str | os.PathLike, trusted: str | os.PathLike | None = None) -> None:
    """Refuse a path that names another machine, or reaches one through a link, before it is resolved.

    Args:
        path: The path as written; nothing along it is followed.
        trusted: A folder the path sits in that may itself be on another machine, as a
            configured directory can be; only links are then checked.

    Raises:
        PathNotAllowed: It names another machine, or a link along it points at one. The
            message names the path below ``trusted`` where it sits there, and never a link's
            target.
    """
    text = str(path)
    inside = trusted is not None and contains(
        Path(os.path.abspath(str(trusted))), Path(os.path.abspath(text)))
    shown = text
    if inside:
        shown = os.path.relpath(os.path.abspath(text), os.path.abspath(str(trusted)))
    if names_another_host(text) and not inside:
        raise PathNotAllowed(f"`{_shown(shown)}` names another machine, so it is not read")
    _refuse_linked_host(Path(os.path.abspath(text)), shown)


def _refuse_device(target: Path, shown: str, purpose: str) -> None:
    """Refuse a target any segment of which names a stream or a device on Windows.

    Args:
        target: The resolved target.
        shown: The value as given, which the refusal names.
        purpose: :data:`READ` or :data:`WRITE`.

    Raises:
        PathNotAllowed: A segment holds ``:`` past the drive, or names a device such as ``NUL``.
    """
    for part in target.parts[1:]:
        reason = _windows_name(part)
        if reason is not None:
            raise PathNotAllowed(
                f"refusing to {purpose} `{_shown(shown)}`\n  A part of it {reason}.")


def _folded(path: Path) -> tuple[str, ...]:
    """A path's segments casefolded, trailing dots and spaces dropped, for comparing names."""
    return tuple(_fold(part) for part in Path(path).parts)


def _fold(name: str) -> str:
    """One name casefolded, trailing dots and spaces dropped, for comparing names."""
    return name.rstrip(" .").casefold()


def _matches(name: str, pattern: str) -> bool:
    """Whether a folded name is a folded pattern, or begins with it where it ends in ``*``."""
    return name.startswith(pattern[:-1]) if pattern.endswith("*") else name == pattern


def _private(purpose: str) -> dict[str, str]:
    """The names in the state directory a node may not reach, each with what it holds.

    Args:
        purpose: :data:`READ` or :data:`WRITE`.

    Returns:
        :data:`PRIVATE`, with :data:`PROTECTED` added for a write.
    """
    held = dict(PRIVATE)
    if purpose == WRITE:
        held.update(
            (name, PRIVATE.get(f"{name}*", "installed view extensions")) for name in PROTECTED)
    return held


def _config_files() -> list[Path]:
    """The config file in use and the one ``WAS_CONFIG`` names, resolved."""
    configs = []
    explicit = paths.env_value(paths.ENV_CONFIG_FILE)
    if explicit:
        configs.append(Path(explicit).expanduser())
    try:
        found = paths.find_config_file()
        if found is not None:
            configs.append(found)
    except Exception as error:
        logger.debug("the config file could not be located (%s)", error)
    resolved = []
    for config in configs:
        try:
            resolved.append(config.resolve())
        except (OSError, RuntimeError, ValueError):
            continue
    return resolved


def _patterns(purpose: str) -> set[str]:
    """Folded patterns some segment of a target must match for :func:`_guarded` to be read.

    Args:
        purpose: :data:`READ` or :data:`WRITE`.

    Returns:
        The names :func:`_private` gives, and the name of the file ``WAS_CONFIG`` names, as
        given and resolved, with every name beginning with it.
    """
    patterns = set(_private(purpose))
    explicit = paths.env_value(paths.ENV_CONFIG_FILE)
    if explicit:
        given = Path(explicit).expanduser()
        patterns.add(f"{given.name}*")
        try:
            patterns.add(f"{given.resolve().name}*")
        except (OSError, RuntimeError, ValueError):
            pass
    return {_fold(pattern) for pattern in patterns}


def _guarded(purpose: str) -> list[tuple[Path, str, str]]:
    """The pack's own files and folders a node may not reach.

    Args:
        purpose: :data:`READ` or :data:`WRITE`.

    Returns:
        ``(folder, pattern, what)`` triples, the folder resolved: the names :func:`_private`
        gives in the state directory, and the config file in use and the one ``WAS_CONFIG``
        names, each with every name beginning with its own.
    """
    guarded = []
    try:
        state = paths.config_directory().resolve()
        guarded.extend((state, name, what) for name, what in _private(purpose).items())
    except Exception as error:
        logger.debug("the state directory could not be resolved (%s)", error)
    guarded.extend((config.parent, f"{config.name}*", "settings") for config in _config_files())
    return guarded


def _refuse_protected(target: Path, shown: str, purpose: str) -> None:
    """Refuse the pack's settings, and for a write its view extension folder as well.

    Args:
        target: The resolved target.
        shown: The value as given, which the refusal names.
        purpose: :data:`READ` or :data:`WRITE`.

    Raises:
        PathNotAllowed: ``target`` is, or lies inside, one of the entries :func:`_guarded`
            gives.
    """
    held = _folded(target)
    patterns = _patterns(purpose)
    if not any(_matches(part, pattern) for part in held for pattern in patterns):
        return
    for folder, pattern, what in _guarded(purpose):
        mark = _folded(folder)
        if (len(held) > len(mark) and held[:len(mark)] == mark
                and _matches(held[len(mark)], _fold(pattern))):
            raise PathNotAllowed(
                f"refusing to {purpose} `{_shown(shown)}`\n"
                f"  It is this pack's {what}, which no node {purpose}s."
            )


def resolve_read(value: str | os.PathLike) -> Path:
    """Resolve a path a node intends to read.

    Args:
        value: The raw widget value.

    Returns:
        The resolved absolute path, inside a permitted read root.

    Raises:
        PathNotAllowed: The path resolved outside every permitted root, names a Windows
            device or stream, or is one of :data:`PRIVATE` or a copy of the config file in
            use.
    """
    return _resolve(value, read_roots(), "paths.allow_read", READ)


def resolve_write(value: str | os.PathLike) -> Path:
    """Resolve a path a node intends to write.

    Args:
        value: The raw widget value.

    Returns:
        The resolved absolute path, inside a permitted write root. The file need not exist.

    Raises:
        PathNotAllowed: The path resolved outside every permitted root, names a Windows
            device or stream, or is one of :data:`PRIVATE`, :data:`PROTECTED` or a copy of
            the config file in use.
    """
    return _resolve(value, write_roots(), "paths.allow_write", WRITE)


def resolve_write_file(directory: str | os.PathLike, name: str | os.PathLike) -> Path:
    """Resolve a directory a node writes into and the file name it writes there, together.

    Args:
        directory: The raw directory widget value, or an already-resolved directory.
        name: The file name to write there. Sub-directories are allowed; a drive, a leading
            root and a ``..`` segment are not.

    Returns:
        The resolved absolute file path, inside the resolved directory and so inside a
        permitted write root. It need not exist.

    Raises:
        PathNotAllowed: The directory resolved outside every permitted write root, or the
            name names somewhere other than inside it.
    """
    # os.path.join and Path both drop the left side for an absolute name and for a
    # drive-relative one such as `C:Windows\x`, and a '..' segment steps back out of a
    # directory checked a moment earlier; _join refuses all three.
    return _join(resolve_write(directory), name, WRITE)
