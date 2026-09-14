"""WAS Node Suite, ComfyUI custom node pack.

``comfy_entrypoint`` is the only name this package exports.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
import sys
import time
import traceback
from collections.abc import Mapping
from pathlib import Path

from .modules import log

#: Oldest ComfyUI carrying every API this pack calls. Set by io.NodeReplace and
#: ComfyAPI().node_replacement, which arrived in 0.14.0; io.Schema.search_aliases in
#: 0.11.0 and comfy_entrypoint in 0.3.51 are older. Kept in step with
#: requires-comfyui in pyproject.toml.
COMFYUI_MIN_VERSION = "0.14.0"

logger = log.get_logger()

try:
    from comfy_api.latest import ComfyExtension, io
except ImportError:
    ComfyExtension = io = None

REQUIRES_PATTERN = re.compile(r"^REQUIRES\s*=\s*[\"']([\w-]+)[\"']", re.MULTILINE)

#: Characters read from a node module to find the group it declares, which sits in its
#: opening lines. The whole file is read only for a module whose group is off.
HEAD_BYTES = 4096
NODE_ID_PATTERN = re.compile(r"\bnode_id\s*=\s*([\"'])(.+?)\1")


def setting(config: Mapping, section: str, key: str, default):
    block = config.get(section)
    return block.get(key, default) if isinstance(block, Mapping) else default


def id_set(value) -> set[str]:
    return set(value) if isinstance(value, (list, tuple, set)) else set()


def read_config() -> Mapping:
    """The config mapping from ``modules.config``, or built-in defaults when it cannot be read."""
    try:
        from .modules.config import load_config
    except Exception as error:
        logger.debug("modules.config is unavailable (%s), using built-in defaults", error)
        return {}
    try:
        config = load_config()
    except Exception as error:
        logger.warning("config could not be read (%s), using built-in defaults", error)
        logger.debug("%s", traceback.format_exc())
        return {}
    if not isinstance(config, Mapping):
        logger.warning(
            "load_config() returned %s instead of a mapping, using built-in defaults",
            type(config).__name__,
        )
        return {}
    return config


def group_names() -> dict[str, tuple[str, ...]]:
    """The group names a module's ``REQUIRES`` may name, per config section.

    Returns:
        ``{"features": (...), "legacy": (...)}``, or an empty mapping when
        ``modules.config`` cannot be read, which turns the name check off rather than
        rejecting every group.
    """
    try:
        from .modules.config.defaults import FEATURE_GROUPS, LEGACY_GROUPS
    except Exception as error:
        logger.debug("the config group names are unavailable (%s), so REQUIRES is unchecked", error)
        return {}
    return {"features": tuple(FEATURE_GROUPS), "legacy": tuple(LEGACY_GROUPS)}


def install_commands(keys) -> list[str]:
    """The install command for each disabled group that needs a package of its own.

    Never raises. An unreadable ``modules.deps`` costs the commands and nothing else.

    Args:
        keys: Config keys of the disabled groups, such as ``"features.rembg"``.

    Returns:
        One command per group that has a requirements file, sorted by key. Empty when no
        disabled group needs a package, which is what most of them need.
    """
    try:
        from .modules import deps
    except Exception as error:
        logger.debug("modules.deps is unavailable (%s), so no install command is shown", error)
        return []
    return [
        deps.install_command(feature=key)
        for key in sorted(keys)
        if deps.group_requirements(key) is not None
    ]


def expand_tokens(node_cls) -> None:
    """Give one node class ``[token]`` expansion on its text inputs.

    Never raises. A class left unexpanded keeps its text inputs as they were.

    Args:
        node_cls: The node class to patch in place.
    """
    try:
        # Imported inside the call and guarded: load_custom_node turns anything escaping
        # comfy_entrypoint into zero registered nodes.
        from .modules.compat.tokens import apply
    except Exception as error:
        logger.debug("token expansion is unavailable (%s)", error)
        return
    apply(node_cls)


def publish_pixels(node_cls, node_id: str) -> None:
    """Give one node class the before and after its interface draws, when it is a pixels node.

    Args:
        node_cls: The node class to patch in place.
        node_id: The id already read off its schema, so a class outside the family costs
            no second schema build.
    """
    try:
        # Imported inside the call and guarded: load_custom_node turns anything escaping
        # comfy_entrypoint into zero registered nodes.
        from .modules.interface.pixels import apply
    except Exception as error:
        logger.debug("the pixels before and after is unavailable (%s)", error)
        return
    apply(node_cls, node_id)


def register_interface_routes() -> None:
    """Register the read-only HTTP routes the node interfaces fetch from."""
    # Imported inside the call and guarded: load_custom_node turns anything escaping
    # comfy_entrypoint into zero registered nodes.
    try:
        from .modules.interface.preview import register_routes as register_previews
    except Exception as error:
        logger.debug("the interface preview channel is unavailable (%s)", error)
    else:
        register_previews()
    try:
        from .modules.interface.run_result import register_routes as register_run_results
    except Exception as error:
        logger.debug("the interface run result channel is unavailable (%s)", error)
    else:
        register_run_results()
    try:
        from .modules.interface.lines import register_routes as register_lines
    except Exception as error:
        logger.debug("the interface text line channel is unavailable (%s)", error)
    else:
        register_lines()
    try:
        from .modules.interface.fonts import register_routes as register_fonts
    except Exception as error:
        logger.debug("the interface font channel is unavailable (%s)", error)
    else:
        register_fonts()
    try:
        from .modules.interface.files import register_routes as register_files
    except Exception as error:
        logger.debug("the interface file listing channel is unavailable (%s)", error)
    else:
        register_files()
    try:
        from .modules.interface.pantry import register_routes as register_pantry
    except Exception as error:
        logger.debug("the interface terminology pantry channel is unavailable (%s)", error)
    else:
        register_pantry()
    try:
        from .modules.interface.video_probe import register_routes as register_video_probe
    except Exception as error:
        logger.debug("the interface video measurement channel is unavailable (%s)", error)
    else:
        register_video_probe()
    try:
        from .modules.interface.pause import register_routes as register_pause
    except Exception as error:
        logger.debug("the held run channel is unavailable (%s)", error)
    else:
        register_pause()
    try:
        from .modules.interface.app_exposure import register_routes as register_app_exposure
    except Exception as error:
        logger.debug("the app workflow exposure channel is unavailable (%s)", error)
    else:
        register_app_exposure()
    try:
        from .modules.interface.three_asset import register_routes as register_three_assets
    except Exception as error:
        logger.debug("the Three.js asset channel is unavailable (%s)", error)
    else:
        register_three_assets()
    try:
        from .modules.interface.three_render import register_routes as register_three_render
    except Exception as error:
        logger.debug("the Three.js render channel is unavailable (%s)", error)
    else:
        register_three_render()


def register_viewer_nodes(config: Mapping, reserved: set[str]) -> None:
    """Register the nodes shipped by installed view extensions, when that group is on.

    Args:
        config: The config mapping. Extension nodes are registered only when
            ``features.viewer`` is true.
        reserved: Node ids this pack is about to register, which an extension may not
            take.
    """
    if not setting(config, "features", "viewer", False):
        return
    try:
        # Imported inside the call and guarded: load_custom_node turns anything escaping
        # comfy_entrypoint into zero registered nodes.
        from .modules.viewer.extension_nodes import register_extension_nodes
    except Exception as error:
        logger.debug("the viewer extension nodes are unavailable (%s)", error)
        return
    try:
        register_extension_nodes(reserved)
    # An extension's node modules are third-party code imported behind this call, and
    # SystemExit is not an Exception: a sys.exit() in one would leave comfy_entrypoint and
    # reach main.py, which does not guard the call that loads custom nodes.
    except (Exception, SystemExit) as error:
        logger.warning(
            "the nodes belonging to the view extensions could not be registered (%s); the "
            "extensions' views are unaffected",
            error,
        )
        logger.debug("%s", traceback.format_exc())


async def register_replacements() -> None:
    """Register the v2 -> v3 ``io.NodeReplace`` table.

    Never raises. Without the table, workflows on retired node ids keep working and are
    not offered the swap.
    """
    try:
        # Imported inside the call and guarded: on_load runs inside load_custom_node's
        # single try/except, where anything escaping registers zero nodes.
        from .modules.compat.replacements import register_replacements as register
    except Exception as error:
        logger.debug("the node replacement table is unavailable (%s)", error)
        return
    try:
        await register()
    except Exception as error:
        logger.warning(
            "the node replacement table could not be registered (%s); workflows on retired "
            "node ids keep working, they are just not offered the swap",
            error,
        )
        logger.debug("%s", traceback.format_exc())


class NodeLoader:
    """Walks ``nodes/``, imports what the config enables, collects the node classes."""

    def __init__(self, package_name: str, config: Mapping | None = None):
        self.package_name = package_name
        self.config = config if isinstance(config, Mapping) else {}
        self.force_on = id_set(setting(self.config, "nodes", "enable", ()))
        self.force_off = id_set(setting(self.config, "nodes", "disable", ()))
        self.nodes: list[type] = []
        self.node_ids: set[str] = set()
        self.timings: dict[str, tuple[float, int, Exception | None]] = {}
        self.skipped: dict[str, list[str]] = {}
        self.dropped: list[str] = []
        self.legacy_prefix = f"{package_name}.nodes.legacy."
        self.groups = group_names()

    def short_name(self, name: str) -> str:
        return name.split(".nodes.", 1)[-1]

    def group_enabled(self, section: str, group: str) -> bool:
        return bool(setting(self.config, section, group, False))

    def node_enabled(self, node_id: str, group_on: bool) -> bool:
        """Precedence: nodes.disable > nodes.enable > features.*/legacy.* > defaults."""
        if node_id in self.force_off:
            return False
        if node_id in self.force_on:
            return True
        return group_on

    def read_source(self, finder, name: str, limit: int = 0) -> str:
        """A module's source text, read from the path the walk already found.

        Args:
            finder: The finder ``pkgutil.walk_packages`` yielded alongside ``name``.
            name: Dotted module name.
            limit: Characters to read, or 0 for the whole file. A module declares its
                group in its opening lines, so a bounded read answers that question.

        Returns:
            The source, or ``""`` when it cannot be read, which reads as a module
            declaring no group and no node id.
        """
        directory = getattr(finder, "path", None)
        if directory is not None:
            try:
                # Read as UTF-8, which is what python reads a source file as when no coding
                # cookie says otherwise. A byte sequence that is not UTF-8 becomes U+FFFD
                # rather than costing the whole file, and both patterns this is searched
                # for are ASCII.
                path = Path(directory) / f"{name.rpartition('.')[2]}.py"
                if not limit:
                    return path.read_text(encoding="utf-8", errors="replace")
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    return handle.read(limit)
            except OSError:
                pass
        try:
            # A finder that is not reading a directory, an install inside a zip say.
            return finder.find_spec(name).loader.get_source(name) or ""
        except Exception:
            return ""

    def import_module(self, name: str):
        started = time.perf_counter()
        module = error = None
        try:
            module = importlib.import_module(name)
        except Exception as failure:
            error = failure
            logger.error("%s failed to import: %s", self.short_name(name), failure)
            logger.debug("%s", traceback.format_exc())
        return module, time.perf_counter() - started, error

    def collect(self, module, group_on: bool, key: str | None) -> list[type]:
        declared = getattr(module, "NODES", None)
        if declared is None:
            declared = [
                obj
                for obj in vars(module).values()
                if isinstance(obj, type)
                and issubclass(obj, io.ComfyNode)
                and obj.__module__ == module.__name__
            ]
        elif not isinstance(declared, (list, tuple)):
            # `NODES = MyNode`, one missing pair of brackets. Say so, rather than let the
            # `for` below fail with a bare "'type' object is not iterable" naming nothing.
            raise TypeError(
                f"NODES must be a list or tuple of node classes, not {type(declared).__name__}"
            )
        kept = []
        for node_cls in declared:
            try:
                node_id = node_cls.GET_SCHEMA().node_id
            except Exception as error:
                # load_custom_node wraps the entrypoint in a single try/except, so a
                # schema that raises there takes down every node in the pack with it.
                logger.error(
                    "%s.%s has an unusable schema and was dropped: %s",
                    self.short_name(module.__name__), node_cls.__name__, error,
                )
                logger.debug("%s", traceback.format_exc())
                self.dropped.append(f"{self.short_name(module.__name__)}.{node_cls.__name__}")
                continue
            if self.node_enabled(node_id, group_on):
                expand_tokens(node_cls)
                publish_pixels(node_cls, node_id)
                kept.append(node_cls)
                self.node_ids.add(node_id)
            elif group_on:
                logger.debug("%s is turned off by nodes.disable", node_id)
            else:
                self.skipped.setdefault(key, []).append(node_id)
        return kept

    def load_module(self, finder, name: str) -> None:
        declared = REQUIRES_PATTERN.search(self.read_source(finder, name, HEAD_BYTES))
        requires = declared.group(1) if declared else None
        legacy = name.startswith(self.legacy_prefix)
        if legacy and requires is None:
            logger.warning(
                "%s sits under nodes/legacy/ but declares no REQUIRES group, so it was skipped",
                self.short_name(name),
            )
            return
        section = "legacy" if legacy else "features"
        key = f"{section}.{requires}" if requires else None
        if key is not None and self.groups and requires not in self.groups[section]:
            # An unknown group reads as off, so without this the module is skipped before
            # it is ever imported and its nodes go missing with nothing said about it.
            raise ValueError(
                f"REQUIRES = {requires!r} is not a {section}.* group in config.yaml; the "
                f"{section} groups are {', '.join(self.groups[section])}"
            )
        group_on = self.group_enabled(*key.split(".")) if key else True
        if not group_on:
            # Finding the ids costs a pass over the whole file, and only a module the
            # config is holding back needs them: nodes.enable names ids, and every id left
            # out is listed at the end of the load.
            source = self.read_source(finder, name)
            node_ids = [found.group(2) for found in NODE_ID_PATTERN.finditer(source)]
            # The group came out of the source text, so a module whose group is off costs
            # one small file read and is never imported.
            if not any(self.node_enabled(node_id, False) for node_id in node_ids):
                self.skipped.setdefault(key, []).extend(node_ids or [self.short_name(name)])
                return
        module, elapsed, error = self.import_module(name)
        collected = self.collect(module, group_on, key) if module is not None else []
        self.nodes.extend(collected)
        self.timings[name] = (elapsed, len(collected), error)

    def module_failed(self, name: str, error: BaseException) -> None:
        """Name the module, name the exception, keep it in ``timings`` so it is counted."""
        logger.error(
            "%s was dropped: %s: %s", self.short_name(name), type(error).__name__, error
        )
        logger.debug("%s", traceback.format_exc())
        self.timings[name] = (0.0, 0, error)

    def walk_error(self, name: str) -> None:
        """Record a subpackage that ``pkgutil`` could not walk, whose subtree is skipped.

        Args:
            name: The subpackage that raised.
        """
        # pkgutil calls this from inside its own except, so the live exception is the
        # subpackage's, and the failure is counted like any other.
        self.module_failed(name, sys.exc_info()[1])

    def load_all(self) -> None:
        package = self.import_nodes_package()
        if package is not None:
            for finder, name, is_package in pkgutil.walk_packages(
                package.__path__, package.__name__ + ".", onerror=self.walk_error
            ):
                if is_package:
                    continue
                try:
                    self.load_module(finder, name)
                except Exception as error:
                    # A module that raises on import, from a module __getattr__, or while
                    # building a schema is dropped alone. Ending the walk would leave
                    # load_custom_node with zero registered nodes for the whole pack.
                    self.module_failed(name, error)
        # The node list is finished by here. Rendering a report about it must not be able
        # to discard it: rich parses table cells as markup and the cells carry exception
        # text this pack did not author.
        for report in (self.print_summary, self.print_disabled):
            try:
                report()
            except Exception as error:
                logger.debug("the startup report could not be printed: %s", error)

    def reserved_ids(self) -> set[str]:
        """Every node id this pack owns, whether or not its group is on this run.

        Returns:
            The ids collected, plus the ids a disabled ``features.*`` or ``legacy.*``
            group held back. An id switched off today is still the pack's: registered to
            something else, it would be overwritten unannounced the day the group is
            turned on.
        """
        return self.node_ids.union(*self.skipped.values())

    def import_nodes_package(self):
        try:
            return importlib.import_module(".nodes", package=self.package_name)
        except Exception as error:
            logger.error("the nodes package could not be imported: %s", error)
            logger.debug("%s", traceback.format_exc())
            return None

    def print_summary(self) -> None:
        """One line for the whole load, raised to a warning when anything did not make it."""
        failed = sum(1 for _, _, error in self.timings.values() if error is not None)
        elapsed = sum(seconds for seconds, _, _ in self.timings.values())
        for name, (seconds, count, error) in self.timings.items():
            if error is None:
                logger.debug(
                    "%s loaded %s node(s) in %.1f ms",
                    self.short_name(name), count, seconds * 1000,
                )
        summary = "loaded {} node(s) from {} module(s) in {:.0f} ms".format(
            len(self.nodes), len(self.timings), elapsed * 1000
        )
        if not failed and not self.dropped:
            logger.info(summary)
            return
        trouble = []
        if failed:
            trouble.append(f"{failed} module(s) failed to import")
        if self.dropped:
            trouble.append(f"{len(self.dropped)} node(s) were dropped")
        logger.warning(
            "%s. %s. Each one is named in an error above, with the reason it did not load.",
            summary, " and ".join(trouble),
        )

    def print_disabled(self) -> None:
        """The disabled groups, how to turn one on and what to install. Ids go to debug."""
        if not self.skipped:
            return
        total = sum(len(ids) for ids in self.skipped.values())
        key_width = max(len(key) for key in self.skipped)
        for key in sorted(self.skipped):
            # The full id list is what connects the node type a workflow reports as missing
            # to the setting that brings it back.
            logger.debug("%s  %s", key.ljust(key_width), ", ".join(sorted(self.skipped[key])))
        lines = [
            f"{total} node(s) are not loaded: {', '.join(sorted(self.skipped))} turned off in "
            f"config.yaml. A workflow naming one of them opens with that node missing. "
            f"Set <group>: true and restart ComfyUI to load it."
        ]
        commands = install_commands(self.skipped)
        if commands:
            lines.append("  Only these need a package installed, one command per group:")
            lines.extend(f"    {command}" for command in commands)
        lines.append("  What each group needs, and where its models go: docs/CONFIG.md")
        logger.info("\n".join(lines))


if io is None:
    logger.error(
        "comfy_api.latest is missing, so no nodes were loaded. WAS Node Suite 3 needs "
        "ComfyUI %s or newer; update ComfyUI, or stay on WAS Node Suite 2.x.",
        COMFYUI_MIN_VERSION,
    )
else:

    class WASNodeSuite(ComfyExtension):
        def __init__(self):
            self.config: Mapping = {}

        async def on_load(self) -> None:
            self.config = read_config()
            log.configure(
                level=setting(self.config, "logging", "level", "info"),
                rich=setting(self.config, "logging", "rich", True),
            )
            register_interface_routes()
            await register_replacements()

        async def get_node_list(self) -> list[type[io.ComfyNode]]:
            loader = NodeLoader(__name__, config=self.config)
            loader.load_all()
            # After the walk, so an extension cannot take an id this pack is about to
            # register: ComfyUI writes the returned list into its mappings last and would
            # overwrite the extension's node without saying so.
            register_viewer_nodes(self.config, loader.reserved_ids())
            return loader.nodes

    async def comfy_entrypoint() -> WASNodeSuite:
        return WASNodeSuite()

    __all__ = ["comfy_entrypoint"]
