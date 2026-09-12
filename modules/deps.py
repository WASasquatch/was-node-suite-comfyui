"""Probes for what the environment around the pack provides.

Import probes that raise actionable errors rather than installing packages, and the device
a float64 computation runs on.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

from . import log

__all__ = [
    "DependencyError",
    "float64_device",
    "group_requirements",
    "install_command",
    "optional",
    "require",
]

logger = log.get_logger("deps")

#: One requirements file per feature group that needs a package, each named for the config
#: key: features.document_export is requirements/document_export.txt. A group needing no
#: package has no file here, so an absent file means there is nothing to install.
REQUIREMENTS = Path(__file__).resolve().parent.parent / "requirements"

#: Import name -> pip requirement, for packages whose two names differ.
PIP_NAMES = {
    "docx": "python-docx",
    "git": "GitPython",
    "huggingface_hub": "huggingface-hub",
}

_loaded: dict[str, ModuleType] = {}


class DependencyError(RuntimeError):
    """An optional package is missing or unusable. Not an ``ImportError`` subclass."""


def require(package: str, feature: str | None = None) -> ModuleType:
    """Import ``package``, or raise :class:`DependencyError` with remediation steps.

    Args:
        package: Module name to import.
        feature: Config key that enabled this code path, such as
            ``"features.document_export"`` or ``"legacy.sampling"``. Named in the error
            message. Omit for default-tier nodes, which have no gating key.

    Returns:
        The imported module.

    Raises:
        DependencyError: The package is absent or fails to import, for any reason.
    """
    module = _loaded.get(package)
    if module is not None:
        return module
    # A real import, not importlib.util.find_spec. find_spec reports only whether a package
    # is present on the path, and cannot tell an absent one from a present but broken one.
    try:
        module = importlib.import_module(package)
    except (Exception, SystemExit) as error:
        # Not just ImportError. A package whose module body raises, such as one built
        # against another version of a library it imports, is exactly as unusable as an
        # absent one, and a caller holding a fallback has no way to take it if the
        # library's own exception escapes.
        # SystemExit is a BaseException, so it is named rather than covered by Exception.
        # A module body that calls sys.exit() would otherwise unwind past every caller and
        # end the thread running the prompt.
        # KeyboardInterrupt and GeneratorExit still propagate.
        raise DependencyError(_explain(package, feature, error)) from error
    _loaded[package] = module
    return module


def optional(package: str) -> ModuleType | None:
    """Import ``package``, returning ``None`` when it is unavailable.

    The failure reason is written to the debug log and nothing is raised.

    Args:
        package: Module name to import.

    Returns:
        The imported module, or ``None``.
    """
    try:
        return require(package)
    except DependencyError as error:
        logger.debug("%s", error)
        return None


def install_command(*requirements: str, feature: str | None = None) -> str:
    """The command that installs what one feature group needs, ready to paste.

    Args:
        requirements: Package names to fall back on where the group has no file of its
            own, such as ``"opencv-python-headless"``. One name per argument.
        feature: Config key that gates the code path, such as
            ``"features.document_export"``.

    Returns:
        A command line naming the python that runs ComfyUI, and that group's requirements
        file where there is one, so it can be run from any directory.
    """
    path = group_requirements(feature)
    return _pip("-r", str(path)) if path else _pip(*requirements)


def group_requirements(feature: str | None) -> Path | None:
    """The requirements file a feature group installs from.

    Args:
        feature: Config key such as ``"features.yunet"``, or ``None``.

    Returns:
        The path to that group's file, or ``None`` where the key names no feature group,
        the group needs no package, or the file does not ship.
    """
    if not feature:
        return None
    section, _, group = feature.partition(".")
    if section != "features" or not group.isidentifier():
        return None
    path = REQUIREMENTS / f"{group}.txt"
    return path if path.is_file() else None


def float64_device():
    """The device a float64 array computation runs on.

    Returns:
        A ``torch.device``: ComfyUI's intermediate device, which is the CPU unless ComfyUI
        was started with ``--gpu-only``, or the CPU where that device carries no float64
        type. MPS is one such device.
    """
    import torch

    try:
        import comfy.model_management as model_management
    except ImportError:
        logger.debug("comfy.model_management is unavailable, so this computes on the CPU")
        return torch.device("cpu")

    device = model_management.intermediate_device()
    try:
        torch.zeros(1, dtype=torch.float64, device=device)
    except (RuntimeError, TypeError) as error:
        logger.debug("%s has no float64, so this computes on the CPU: %s", device, error)
        return torch.device("cpu")
    return device


def _explain(package: str, feature: str | None, error: BaseException) -> str:
    """Build a remediation message for an import failure.

    Args:
        package: Module name that failed to import.
        feature: Config key that enabled the code path, or ``None``.
        error: The exception raised by the import.

    Returns:
        A multi-line message naming the cause and the command to run.
    """
    requirement = PIP_NAMES.get(package, package)
    # Only ModuleNotFoundError names a missing module, so every other exception falls to the
    # third branch, which already says the right thing about an installed, unusable package.
    missing = error.name if isinstance(error, ModuleNotFoundError) else None
    if missing == package.partition(".")[0]:
        lines = [
            f"{package} is not installed, and this node needs it.",
            f"    {install_command(requirement, feature=feature)}",
        ]
    elif missing:
        # The package itself installed, so its group's requirements file is already
        # satisfied and running it would install nothing. Only the missing name repairs it.
        lines = [
            f"{package} is installed, but it needs {missing}, which is not.",
            f"    {_pip(PIP_NAMES.get(missing, missing))}",
        ]
    elif isinstance(error, SystemExit):
        lines = [
            f"{package} is installed, but importing it ended the process rather than",
            "    raising, so what it printed to the console above is the real error. A",
            "    backend it needs at import time, and does not install itself, is the",
            "    usual cause.",
        ]
        if group_requirements(feature) is not None:
            lines.append("    This group's file names that backend alongside the package:")
            lines.append(f"    {install_command(feature=feature)}")
    else:
        lines = [
            f"{package} is installed but will not import: {error}",
            "    That is the real error, and installing it again will not change it: a",
            "    version conflict or a half-finished install is the usual cause.",
        ]
    if feature:
        # features.document_export gates three file formats inside one node and no node of
        # its own, so this cannot promise that turning a group off stops a node loading.
        lines.append(f"    Reached because {feature} is on in config.yaml. Set it to false")
        lines.append("    to turn that group off, and nothing in it loads or runs.")
    return "\n".join(lines)


def _pip(*arguments: str) -> str:
    """Build a pip install command line.

    Args:
        arguments: What to install: one requirement, or ``-r`` and a file path.

    Returns:
        The command, quoted wherever an argument holds a space.
    """
    import sys

    # The python running ComfyUI, not a bare pip: on a portable install the pip on PATH
    # belongs to another interpreter.
    if not sys.executable:
        prefix = ["pip"]
    elif sys.flags.no_user_site:
        prefix = [sys.executable, "-s", "-m", "pip"]
    else:
        prefix = [sys.executable, "-m", "pip"]
    return " ".join(
        f'"{part}"' if " " in part else part
        for part in (*prefix, "install", *arguments)
    )
