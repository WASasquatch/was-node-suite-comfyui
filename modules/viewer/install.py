"""Installing content-viewer view extensions.

:func:`write_manifest` lists the views into ``extension_views.json``, and
:func:`install_all` unpacks ``.zip`` packages, copies from sibling ``ComfyUI_Viewer_*``
directories, and pip-installs declared requirements.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import zipfile
from pathlib import Path

__all__ = [
    "EXTENSION_PREFIX",
    "ensure_drop_dir",
    "extension_slug",
    "install_all",
    "loads_as_pack",
    "sync_siblings",
    "write_manifest",
]

logger = logging.getLogger("was_node_suite.viewer")

#: Directory-name prefix marking a sibling pack as a view extension.
EXTENSION_PREFIX = "ComfyUI_Viewer_"

#: Where the viewer's own files live, relative to the pack root.
VIEWS_DIR = Path("web") / "viewer" / "views"
PARSERS_DIR = Path("modules") / "viewer" / "parsers"
#: Built single-page applications an extension embeds in the viewer.
APPS_DIR = Path("modules") / "viewer" / "apps"

#: ComfyUI nodes an extension ships, written against the V1 node API.
#: ``modules/viewer/extension_nodes`` imports what lands here and registers it into
#: ComfyUI's own node mappings.
NODES_DIR = Path("modules") / "viewer" / "extension_nodes"

#: Directory in an extension package -> where its contents belong in this pack.
EXTRACT = {
    "web/views": VIEWS_DIR,
    "modules/parsers": PARSERS_DIR,
    "apps": APPS_DIR,
    "nodes": NODES_DIR,
}

#: Sources whose contents go under a directory named for the package rather than straight
#: into the target, so two extensions cannot collide on a file name.
PER_EXTENSION = frozenset({"nodes"})

#: Branch names GitHub appends when a repository is downloaded as a ``.zip``, stripped
#: from the directory name so a second download of the same extension reuses its
#: directory instead of leaving a stale copy beside it.
ARCHIVE_BRANCHES = ("-main", "-master")

#: Written into the views directory and fetched by the frontend's view loader.
MANIFEST_NAME = "extension_views.json"

#: View files the viewer ships itself, listed in ``view_manifest.js`` and loaded from
#: there. They must not appear in the extension manifest as well, or each is imported
#: twice.
CORE_VIEWS = frozenset({
    "ansi.js", "canvas.js", "code_scripts.js", "css.js", "csv.js", "html.js",
    "javascript.js", "json.js", "markdown.js", "object.js", "python.js", "svg.js",
    "text.js", "yaml.js",
})

#: Machinery in the views directory that is not a view.
NOT_VIEWS = frozenset({"base_view.js", "view_loader.js", "view_manifest.js"})

#: Written into the drop directory. The name is the instruction, so it reads as one in a
#: file listing without being opened.
DROP_README = "HOW_TO_INSTALL_VIEW_EXTENSIONS.txt"

DROP_README_TEXT = """\
Content Viewer, view extensions
================================

A view extension teaches the Content Viewer a content type it does not already know: a
.js view for the browser, and usually a Python parser that decides which content the view
claims. WAS Node Suite ships twelve views built in; this directory is for adding more.

Extensions live at https://github.com/WASasquatch?tab=repositories, the ones whose names
begin ComfyUI_Viewer_ are view extensions.


Installing one by hand, works right now, nothing to turn on
------------------------------------------------------------

Copy two files out of the extension into WAS Node Suite:

    web/views/<name>.js                 ->  <pack>/web/viewer/views/
    modules/parsers/<name>_parser.py    ->  <pack>/modules/viewer/parsers/

where <pack> is ComfyUI/custom_nodes/was-node-suite-comfyui. An apps/ directory goes in
<pack>/modules/viewer/apps/. The contents of a nodes/ directory go in

    <pack>/modules/viewer/extension_nodes/<extension name>/

in a directory of its own, so two extensions cannot collide on a file name. Leave that
directory's own __init__.py behind if it has one: each node module is imported on its own
here, and the automatic route drops the file for the same reason.

An extension serving its own HTTP routes registers them from its own __init__.py, which
ComfyUI imports the way it imports any pack, and nothing of its routes is copied here.

Restart ComfyUI. The view registers itself; there is no list to edit.

If the extension ships a requirements.txt, install it yourself with the Python that runs
ComfyUI. For the Windows portable build, from the ComfyUI_windows_portable directory:



Installing one automatically, opt in first
-------------------------------------------

Put this in your config.yaml:

    viewer:
      install_extensions: true

Then drop the extension's .zip in THIS directory and restart ComfyUI. It is unpacked to
the right places, its requirements are installed with the Python that runs ComfyUI, and a
record is written to logs/ so it is not installed twice. Delete a record to reinstall.

It is off by default. With it on, a file you downloaded decides what gets pip-installed
into your ComfyUI. Nothing else in WAS Node Suite installs anything without being asked. Everything the installer does is written to the ComfyUI log, including the
pip command.


A note on extension nodes
-------------------------

Some extensions also ship a nodes/ directory of ComfyUI nodes. Those are installed and
registered by either route, and appear in the node menu after a restart like any other
node. An extension cloned into custom_nodes as ComfyUI_Viewer_<name> and carrying its own
top-level __init__.py is loaded by ComfyUI as a node pack in its own right, so its nodes
arrive that way and nothing is copied. A clone without one cannot be imported by ComfyUI,
so its nodes are copied in on the same pass as its views.

They are registered only while the viewer is on. With

    features:
      viewer: false

in config.yaml, the viewer and every extension node are left out, and the standalone
ComfyUI_Viewer pack is free to provide them instead. Saved workflows are unaffected
either way.

A node id that something else already provides, ComfyUI itself or another pack, is left
with its current owner. The extension's node is skipped and the clash is written to the
ComfyUI log, naming the id and the extension.


Full documentation
------------------

"""

DROP_LOGS_README = "delete_a_record_here_to_reinstall_that_extension.txt"

DROP_LOGS_TEXT = """\
Records of installed view extensions, one JSON file per package.

Delete one and restart ComfyUI to install that extension again. Each record lists the
files it installed, and those are checked before the extension is skipped, so updating
WAS Node Suite, which replaces its own directory, causes a reinstall on its own.
"""


def ensure_drop_dir(drop: Path) -> bool:
    """Create the extension drop directory and the note explaining what it is for.

    Args:
        drop: The drop directory, under the pack's config directory.

    Returns:
        Whether the directory is there afterwards.
    """
    try:
        # A directory that appears only once a setting is found is a directory nobody finds
        # the setting from, so it is created with the installer off as well.
        (drop / "logs").mkdir(parents=True, exist_ok=True)
        _write_if_changed(drop / DROP_README, DROP_README_TEXT)
        _write_if_changed(drop / "logs" / DROP_LOGS_README, DROP_LOGS_TEXT)
    except OSError as error:
        logger.debug("%s could not be prepared (%s)", drop, error)
        return False
    return True


def _write_if_changed(path: Path, text: str) -> None:
    try:
        if path.is_file() and path.read_text(encoding="utf-8") == text:
            return
        path.write_text(text, encoding="utf-8")
    except OSError as error:
        logger.debug("%s could not be written (%s)", path, error)


def write_manifest(pack_root: Path) -> list[str]:
    """List the extension views on disk and write the manifest the frontend reads.

    Args:
        pack_root: This pack's root directory.

    Returns:
        The view filenames written, sorted. Empty when only the built-in views are
        present, in which case an existing manifest is emptied rather than left stale.
    """
    views = pack_root / VIEWS_DIR
    try:
        found = sorted(
            path.name
            for path in views.glob("*.js")
            if path.name not in CORE_VIEWS and path.name not in NOT_VIEWS
        )
    except OSError as error:
        logger.debug("%s could not be listed (%s)", views, error)
        return []

    manifest = views / MANIFEST_NAME
    try:
        current = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else None
    except (OSError, ValueError):
        current = None
    if current == found:
        return found

    try:
        views.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(found, indent=2), encoding="utf-8")
    except OSError as error:
        logger.warning(
            "the content viewer's %s could not be written (%s), so any view extension "
            "installed alongside the built-in views will not load",
            manifest, error,
        )
        return []
    if found:
        logger.debug("content viewer: the view extensions available are %s", ", ".join(found))
        logger.info("content viewer: %s view extension(s) available", len(found))
    return found


def install_all(pack_root: Path, drop_dir: Path) -> int:
    """Unpack every ``.zip`` package in ``drop_dir`` that is not installed already.

    Args:
        pack_root: This pack's root directory, which the files are copied into.
        drop_dir: Directory holding the ``.zip`` packages, and a ``logs`` subdirectory
            recording what each one installed.

    Returns:
        How many packages were installed on this run.
    """
    try:
        packages = sorted(drop_dir.glob("*.zip"))
    except OSError as error:
        logger.debug("%s could not be listed (%s)", drop_dir, error)
        return 0
    if not packages:
        return 0

    installed = 0
    for package in packages:
        record = _record_path(drop_dir, package)
        if _is_installed(record, pack_root):
            continue
        logger.info("content viewer: installing the view extension %s", package.name)
        if _install(package, pack_root, record):
            installed += 1
    return installed


def loads_as_pack(entry: Path) -> bool:
    """Whether ComfyUI can load a directory in ``custom_nodes`` as a node pack.

    Args:
        entry: A directory beside this pack.

    Returns:
        Whether it holds the ``__init__.py`` ComfyUI imports a pack through. Without one
        the import fails and the directory registers nothing, so anything it ships has to
        reach ComfyUI by another route.
    """
    try:
        return (entry / "__init__.py").is_file()
    except OSError:
        return False


def sync_siblings(pack_root: Path, custom_nodes: Path) -> int:
    """Copy files from any ``ComfyUI_Viewer_*`` directory installed beside this pack.

    Args:
        pack_root: This pack's root directory.
        custom_nodes: The directory this pack is installed in.

    Returns:
        How many files were copied.
    """
    try:
        entries = sorted(custom_nodes.iterdir())
    except OSError as error:
        logger.debug("%s could not be listed (%s)", custom_nodes, error)
        return 0

    copied = 0
    for entry in entries:
        if not entry.name.startswith(EXTENSION_PREFIX):
            continue
        try:
            if not entry.is_dir():
                continue
        except OSError:
            continue
        registers_itself = loads_as_pack(entry)
        for source, relative in EXTRACT.items():
            if source in PER_EXTENSION:
                # A clone ComfyUI can import is loaded as a pack of its own, so its nodes
                # are registered from there and copying them in would present every one of
                # them a second time under an id that is taken. A clone with no
                # __init__.py cannot be imported at all, and its nodes reach ComfyUI only
                # from here.
                if registers_itself:
                    continue
                relative = relative / entry.name
            root = entry.joinpath(*source.split("/"))
            if not root.is_dir():
                continue
            copied += _copy_tree(root, pack_root / relative, entry.name)
    return copied


def _copy_tree(source: Path, target: Path, label: str) -> int:
    """Copy every non-private file under ``source`` into ``target``, keeping structure."""
    copied = 0
    try:
        files = [path for path in source.rglob("*") if path.is_file()]
    except OSError as error:
        logger.debug("%s could not be walked (%s)", source, error)
        return 0
    for path in files:
        if path.name.startswith("_") or path.suffix == ".pyc":
            continue
        destination = target / path.relative_to(source)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_file() and destination.stat().st_mtime >= path.stat().st_mtime:
                continue
            shutil.copy2(path, destination)
        except OSError as error:
            logger.warning(
                "the content viewer could not copy %s from %s (%s), so that part of the "
                "extension is not installed",
                path.name, label, error,
            )
            continue
        copied += 1
    return copied


def extension_slug(package_name: str) -> str:
    """A package's file name reduced to a directory name of its own.

    Args:
        package_name: The ``.zip``'s file name, with or without its suffix.

    Returns:
        The name with its suffix, any GitHub branch suffix and every character outside
        ``A-Z a-z 0-9 _`` removed. Never empty, so a package named entirely in punctuation
        still gets a directory.
    """
    stem = Path(package_name).stem
    for branch in ARCHIVE_BRANCHES:
        if stem.lower().endswith(branch) and len(stem) > len(branch):
            stem = stem[: -len(branch)]
            break
    return re.sub(r"[^0-9A-Za-z_]+", "_", stem).strip("_") or "extension"


def _record_path(drop_dir: Path, package: Path) -> Path:
    return drop_dir / "logs" / f"{package.stem}.json"


def _is_installed(record: Path, pack_root: Path) -> bool:
    """Has this package been installed, and are the files it installed still there?

    Args:
        record: The install record, ``{"package": name, "files": [...]}`` as JSON.
        pack_root: This pack's root directory, which the recorded paths are relative to.

    Returns:
        Whether the record names files and every one of them still exists.
    """
    try:
        stored = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    files = stored.get("files") if isinstance(stored, dict) else None
    if not isinstance(files, list) or not files:
        return False
    # Updating this pack replaces its directory and takes the installed view files with it,
    # so a record is checked against the filesystem rather than trusted on its own.
    return all((pack_root / name).exists() for name in files)


def _install(package: Path, pack_root: Path, record: Path) -> bool:
    """Unpack one package, install its requirements, and write its record."""
    written: list[str] = []
    requirements = ""
    try:
        with zipfile.ZipFile(package) as archive:
            root = _archive_root(archive)
            written = _extract(archive, root, pack_root, package.name, extension_slug(package.name))
            requirements = _requirements(archive, root)
    except (OSError, zipfile.BadZipFile) as error:
        logger.warning(
            "the content viewer could not read the view extension %s (%s), so it was not "
            "installed. Delete it, or download it again",
            package.name, error,
        )
        return False

    if not written:
        logger.warning(
            "the view extension %s holds none of %s, so there was nothing to install. It "
            "may be a package for something other than the content viewer",
            package.name, ", ".join(sorted(EXTRACT)),
        )
        return False

    if requirements.strip():
        _report_requirements(requirements, package)

    try:
        record.parent.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps({"package": package.name, "files": written}, indent=2), encoding="utf-8")
    except OSError as error:
        logger.warning(
            "the content viewer installed %s but could not record it (%s), so it will be "
            "installed again on the next start",
            package.name, error,
        )
    logger.info("content viewer: %s installed %s file(s)", package.name, len(written))
    return True


def _archive_root(archive: zipfile.ZipFile) -> str:
    """The single top-level directory a GitHub download wraps everything in, or ``""``.

    Args:
        archive: The open package.

    Returns:
        The wrapper directory's name, or ``""`` when the content directories are already at
        the top. A package holding exactly one top-level entry named ``web`` is of that
        second kind, not a wrapper called ``web``.
    """
    names = archive.namelist()
    tops = {name.split("/", 1)[0] for name in names if name.strip("/")}
    if any(name.startswith(f"{source}/") for source in EXTRACT for name in names):
        return ""
    if len(tops) == 1:
        only = tops.pop()
        if any(name.startswith(f"{only}/") for name in names):
            return only
    return ""


def _extract(archive: zipfile.ZipFile, root: str, pack_root: Path, label: str, slug: str) -> list[str]:
    """Copy the directories named in :data:`EXTRACT` out of one package.

    Args:
        archive: The open package.
        root: The wrapper directory inside it, or ``""``.
        pack_root: This pack's root directory, which the files are written under.
        label: The package's file name, named in any message about it.
        slug: Directory name for the sources listed in :data:`PER_EXTENSION`.

    Returns:
        Paths written, relative to ``pack_root``, for the install record.
    """
    written = []
    for source, relative in EXTRACT.items():
        if source in PER_EXTENSION:
            relative = relative / slug
        prefix = f"{root}/{source}/" if root else f"{source}/"
        for name in archive.namelist():
            if not name.startswith(prefix) or name.endswith("/"):
                continue
            tail = name[len(prefix):]
            # Private names are skipped so an extension cannot replace this package's own
            # __init__.py, and the test is on the file's name rather than on the whole
            # path: a built single-page application routinely keeps its assets in a
            # directory called _next or _app, and those are the application.
            if not tail or tail.rsplit("/", 1)[-1].startswith("_") or tail.endswith(".pyc"):
                continue
            base = (pack_root / relative).resolve()
            target = (base / tail).resolve()
            # A zip may name ../ segments, which would write outside the pack.
            if not target.is_relative_to(base):
                logger.warning(
                    "the view extension %s tried to write outside the viewer directory "
                    "(%s) and that entry was skipped",
                    label, name,
                )
                continue
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source_file, open(target, "wb") as handle:
                    shutil.copyfileobj(source_file, handle)
            except OSError as error:
                logger.warning("the content viewer could not write %s (%s)", target, error)
                continue
            written.append(str(target.relative_to(pack_root)).replace("\\", "/"))
    return written


def _requirements(archive: zipfile.ZipFile, root: str) -> str:
    """The package's ``requirements.txt``, or ``""`` when it declares none."""
    name = f"{root}/requirements.txt" if root else "requirements.txt"
    if name not in archive.namelist():
        return ""
    try:
        with archive.open(name) as handle:
            return handle.read().decode("utf-8", errors="replace")
    except (OSError, KeyError):
        return ""


def _report_requirements(requirements: str, package: Path) -> None:
    """Name what a view extension needs, for installing by hand.

    Args:
        requirements: The package's ``requirements.txt``. One holding only blank lines and
            ``#`` comments names nothing.
        package: The package being reported.
    """
    wanted = [
        line.strip()
        for line in requirements.splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not wanted:
        return
    logger.warning(
        "the view extension %s needs %s. Install them with the python that runs ComfyUI; "
        "its views are installed either way",
        package.name, ", ".join(wanted),
    )
