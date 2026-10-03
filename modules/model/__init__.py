"""Model loading shared by the gated model nodes.

:func:`model_directories`, :func:`model_files` and :func:`model_file_path` work through
``folder_paths``; :func:`resolve` locates a transformers-format checkpoint and raises
:class:`ModelUnavailable` when it cannot.
"""

from __future__ import annotations

import gc
import json
import os
import re
import struct
from collections.abc import Sequence
from pathlib import Path

from .. import log

__all__ = [
    "Backend",
    "ModelUnavailable",
    "NETWORK_FEATURE",
    "cached",
    "checkpoint_shapes",
    "compute_device",
    "managed",
    "managed_module",
    "model_directories",
    "model_file_path",
    "model_files",
    "network_enabled",
    "offload_device",
    "published_checkpoint",
    "published_files",
    "release_models",
    "repository_id",
    "resolve",
    "shared_file",
    "shared_roots",
    "unpin_staged",
    "unresolved_links",
]

logger = log.get_logger("model")

#: Config key whose group permits a backend to fetch weights it cannot find on disk.
NETWORK_FEATURE = "features.network"

#: The file transformers writes into every checkpoint directory it can read.
CHECKPOINT_MARKER = "config.json"

#: Device names that mean "whichever device ComfyUI is running on". These are what v2's
#: widgets could produce; any other name is passed to ``torch.device``.
ACCELERATOR_NAMES = frozenset({"", "cuda", "gpu"})

#: Loaded backends kept resident before the least recently used one is dropped. One
#: workflow can hold six at once: BLIP counts twice, once per task, alongside CLIPSeg,
#: SAM and MiDaS.
CACHE_LIMIT = 6

_cache: dict[tuple, object] = {}


class ModelUnavailable(RuntimeError):
    """Weights were found neither on disk nor behind a permitted download."""


class Backend:
    """A loaded backend and the ComfyUI registration that keeps its weights reclaimable.

    Attributes:
        processor: The backend's transformers processor, or ``None`` for a backend that
            has none.
        model: The torch module holding the weights, in eval mode.
        load_device: Where :meth:`load` makes the weights resident and inference runs.
        offload_device: Where the weights rest between executions.
        patcher: The ``ModelPatcher`` registered with ``comfy.model_management``, or
            ``None`` for a CPU-bound backend and outside a ComfyUI process, where there is
            no VRAM budget to take part in.
        patchers: The registered ``ModelPatcher``, as a tuple that is empty where there
            is none.
        key: Cache key identifying the weights, without the device.
        build: The callable that produced ``(processor, model)``.
        name: What ComfyUI's model management calls the model: the name it was registered
            under, or its own class name.
    """

    def __init__(
        self, processor, model, load_device, offload_device, patcher=None, key=(), build=None,
        name=None,
    ):
        self.processor = processor
        self.model = model
        self.load_device = load_device
        self.offload_device = offload_device
        self.patchers = tuple(p for p in (patcher,) if p)
        self.patcher = patcher if self.patchers else None
        self.key = key
        self.build = build
        self.name = name or type(model).__name__

    def on(self, device: str | None = None) -> "Backend":
        """The same weights on another device.

        Args:
            device: Device name, or ``None`` for ComfyUI's compute device.

        Returns:
            This backend when it is already on ``device``, otherwise the cached backend
            for the same weights on ``device``, built on first use.
        """
        if compute_device(device) == self.load_device:
            return self
        # model.to(device) here would strand weights on a device comfy.model_management is
        # not accounting for and cannot reclaim.
        return managed(self.key, self.build, device=device, name=self.name)

    def load(self, memory_required: int = 0):
        """Make the weights resident on :attr:`load_device` and return that device.

        Args:
            memory_required: Bytes of working memory to make room for beside the weights.

        Returns:
            The ``torch.device`` the inputs must be moved to.
        """
        management = _management()
        if not self.patchers or management is None:
            self.model.to(self.load_device)
            return self.load_device
        # A transformers module has no cast-on-use wrappers, so a partial load would leave
        # it with weights on two devices.
        management.load_models_gpu(
            list(self.patchers), memory_required=int(memory_required), force_full_load=True
        )
        return self.load_device


def network_enabled() -> bool:
    """Whether ``features.network`` is on. Weights are downloaded only when it is."""
    from ..config import group_enabled

    return group_enabled(NETWORK_FEATURE)


def model_directories(folder: str, location: str | None = None) -> list[Path]:
    """Every directory ``folder_paths`` knows for ``folder``, registering it when new.

    Args:
        folder: A model folder name such as ``"blip"`` or ``"sams"``. This is the key an
            ``extra_model_paths`` entry uses, so it is chosen not to collide with a name
            another pack registers.
        location: Where the folder sits under ``models``, as a ``/`` separated relative
            path, when that differs from ``folder`` itself. ``"onnx/face_detection"``
            registers a directory inside the shared ``onnx`` folder while keeping a key of
            its own. ``None`` puts it at the top level, named after ``folder``.

    Returns:
        Search directories in priority order. Empty when ``folder_paths`` cannot be
        imported, which is the case outside a ComfyUI process.
    """
    try:
        import folder_paths
    except ImportError:
        logger.debug("folder_paths is unavailable, so the %s model directory is unknown", folder)
        return []
    # Registering the folder rather than joining models_dir by hand is what lets an
    # extra_model_paths entry for the same name take effect.
    parts = (location or folder).split("/")
    default = os.path.join(folder_paths.models_dir, *parts)
    known = (
        folder_paths.get_folder_paths(folder)
        if folder in folder_paths.folder_names_and_paths
        else []
    )
    # An extra_model_paths entry claims the name outright, so the directory under models is
    # registered alongside it and a checkpoint kept in either place is found.
    if os.path.normcase(default) not in {os.path.normcase(str(path)) for path in known}:
        folder_paths.add_model_folder_path(folder, default)
    return [Path(path) for path in folder_paths.get_folder_paths(folder)]


def shared_roots() -> list[Path]:
    """Directories outside this pack that already hold downloaded checkpoints.

    Returns:
        The Hugging Face client's own cache and every ``ckpts`` directory a pack under
        ``custom_nodes`` keeps, in that order. Empty entries and absent directories are
        left out, so the result is only places worth reading.
    """
    roots: list[Path] = []
    from ..config.paths import env_value

    hub = env_value("HF_HUB_CACHE") or env_value("HUGGINGFACE_HUB_CACHE")
    home = env_value("HF_HOME")
    candidates = [Path(hub)] if hub else []
    if home:
        candidates.append(Path(home) / "hub")
    candidates.append(Path.home() / ".cache" / "huggingface" / "hub")
    try:
        import folder_paths

        packs = Path(folder_paths.base_path) / "custom_nodes"
        # One convention covers every pack that follows it, rather than naming one of them.
        if os.path.isdir(packs):
            candidates.extend(
                packs / name / "ckpts" for name in sorted(os.listdir(packs))
            )
    except Exception as error:
        logger.debug("the custom_nodes directory is unreadable (%s)", error)
    for candidate in candidates:
        if os.path.isdir(candidate) and candidate not in roots:
            roots.append(candidate)
    return roots


def published_checkpoint(
    folder: str,
    repo_id: str,
    filename: str,
    *,
    subfolder: str = "",
    feature: str | None = None,
    what: str = "This model",
) -> str:
    """Where one published checkpoint file is, fetching it when that is allowed.

    Args:
        folder: ``folder_paths`` model folder to search and to download into.
        repo_id: Hugging Face repository publishing the file.
        filename: The file's own name inside that repository.
        subfolder: Directory inside the repository holding it, where there is one.
        feature: Config key of the group whose node reached here.
        what: Name for the weights in the error, such as ``"The edge network"``.

    Returns:
        An absolute path to the file.

    Raises:
        ModelUnavailable: The file is nowhere on this machine and ``features.network``
            is off.
    """
    roots = model_directories(folder)
    for name in (filename, f"{subfolder}/{filename}" if subfolder else filename):
        found = model_file_path(folder, name)
        if found is not None:
            return str(found)
    borrowed = shared_file(repo_id, filename)
    if borrowed is not None:
        logger.debug("%s was read from %s", filename, borrowed)
        return str(borrowed)
    if not network_enabled():
        raise ModelUnavailable(_missing_file(what, filename, repo_id, roots, feature))
    from .. import deps

    hub = deps.require("huggingface_hub")
    return hub.hf_hub_download(
        repo_id, filename, subfolder=subfolder or None,
        local_dir=str(roots[0]) if roots else None,
    )


def published_files(
    folder: str,
    repo_id: str,
    filenames: Sequence[str],
    *,
    also_search: Sequence[str] = (),
    feature: str | None = None,
    what: str = "This model",
) -> dict[str, Path]:
    """Where several files of one published repository are, fetching what is missing.

    Args:
        folder: ``folder_paths`` model folder to search and to download into.
        repo_id: Hugging Face repository publishing the files.
        filenames: Paths inside that repository, ``/`` separated, such as
            ``"unet/diffusion_pytorch_model.fp16.safetensors"``.
        also_search: Further repository ids to look in before downloading, for a file
            published unchanged in more than one of them.
        feature: Config key of the group whose node reached here.
        what: Name for the weights in the error, such as ``"The intrinsics network"``.

    Returns:
        Each name in ``filenames`` mapped to an absolute path.

    Raises:
        ModelUnavailable: A file is nowhere on this machine and ``features.network`` is off.
    """
    roots = model_directories(folder)
    searched = roots + shared_roots()
    repositories = [repo_id, *also_search]
    found: dict[str, Path] = {}
    wanted: list[str] = []
    for filename in filenames:
        located = _published_file(searched, repositories, filename)
        if located is None:
            wanted.append(filename)
        else:
            logger.debug("%s of %s was read from %s", filename, repo_id, located)
            found[filename] = located
    if not wanted:
        return found
    if not network_enabled():
        raise ModelUnavailable(_missing_file(what, wanted[0], repo_id, roots, feature))
    from .. import deps

    hub = deps.require("huggingface_hub")
    owner, _, name = repo_id.partition("/")
    into = roots[0] / owner / name if roots else None
    for filename in wanted:
        found[filename] = Path(
            hub.hf_hub_download(
                repo_id, filename, local_dir=str(into) if into is not None else None
            )
        )
    return found


def _published_file(
    roots: Sequence[Path], repositories: Sequence[str], filename: str
) -> Path | None:
    """The first copy of one repository file under any search root, or ``None``."""
    for root in roots:
        for repo_id in repositories:
            for relative in _layouts(repo_id):
                candidate = root / relative / filename
                if os.path.isfile(candidate):
                    return candidate
                snapshot = _newest_snapshot(root / relative / "snapshots", filename)
                if snapshot is not None:
                    return snapshot / filename
        loose = root / filename
        if os.path.isfile(loose):
            return loose
    return None


def _missing_file(what, filename, repo_id, roots, feature) -> str:
    """The message naming a checkpoint that is nowhere and how to supply it."""
    searched = "\n".join(f"    {root}" for root in roots) or "    nowhere, ComfyUI is not running"
    lines = [
        f"{what} needs {filename}, which is not in:",
        searched,
        f"Download it from https://huggingface.co/{repo_id} into that directory,",
        "or leave it where another preprocessor pack already keeps it,",
        f"or set {NETWORK_FEATURE}: true in config.yaml to fetch it on first use.",
    ]
    if feature:
        lines.append(f"This node was loaded because {feature} is enabled in config.yaml.")
    lines.extend(unresolved_links(roots, filename))
    return "\n".join(lines)


def unresolved_links(roots: Sequence[Path], filename: str = "") -> list[str]:
    """Lines naming each link among ``roots`` or ``root/filename`` that does not resolve.

    Args:
        roots: Search directories.
        filename: A file name looked for in each of them, or empty to test the roots alone.

    Returns:
        One line per link that exists but points at nothing this process can open, with
        its target where it can be read. Empty when every link resolves.
    """
    lines = []
    for root in roots:
        for candidate in (root, root / filename) if filename else (root,):
            if not os.path.lexists(candidate) or os.path.exists(candidate):
                continue
            try:
                target = os.readlink(candidate).removeprefix("\\\\?\\")
            except OSError:
                target = "a target this process cannot read"
            lines.append(
                f"{candidate} is a link to {target}, which ComfyUI cannot open. Recreate the "
                "link from Windows with mklink, or point extra_model_paths.yaml at the real folder."
                if os.name == "nt"
                else f"{candidate} is a link to {target}, which ComfyUI cannot open."
            )
    return lines


_header_cache: dict[str, tuple[tuple[int, int], dict[str, list[int]]]] = {}


def checkpoint_shapes(path: str | os.PathLike) -> dict[str, list[int]]:
    """Every tensor name in a safetensors file mapped to its shape, read from the header.

    Args:
        path: A ``.safetensors`` file.

    Returns:
        Tensor shapes by name, cached against the file's size and modification time. Empty
        when the file cannot be read or is not safetensors.
    """
    key = os.fspath(path)
    try:
        info = os.stat(key)
    except OSError:
        return {}
    stamp = (info.st_size, info.st_mtime_ns)
    cached_entry = _header_cache.get(key)
    if cached_entry is not None and cached_entry[0] == stamp:
        return cached_entry[1]
    shapes: dict[str, list[int]] = {}
    try:
        with open(key, "rb") as handle:
            (size,) = struct.unpack("<Q", handle.read(8))
            if 0 < size <= 100_000_000:
                header = json.loads(handle.read(size))
                shapes = {
                    name: list(entry.get("shape", []))
                    for name, entry in header.items()
                    if name != "__metadata__" and isinstance(entry, dict)
                }
    except (OSError, ValueError, struct.error):
        shapes = {}
    _header_cache[key] = (stamp, shapes)
    return shapes


def shared_file(repo_id: str, filename: str) -> Path | None:
    """One checkpoint file, wherever this machine already has it.

    Args:
        repo_id: Hugging Face repository the file belongs to, such as
            ``"lllyasviel/Annotators"``.
        filename: The file's own name inside that repository.

    Returns:
        The first copy found, or None. Every layout :func:`_layouts` knows is tried under
        every shared root, and the bare filename is tried too.
    """
    for root in shared_roots():
        for relative in _layouts(repo_id):
            candidate = root / relative / filename
            if os.path.isfile(candidate):
                return candidate
        loose = root / filename
        if os.path.isfile(loose):
            return loose
    return None


def model_files(
    folder: str,
    location: str | None = None,
    suffixes: Sequence[str] = (),
) -> list[str]:
    """Every model file ComfyUI can see in a folder, as a widget should list them.

    Args:
        folder: A model folder name, as :func:`model_directories` registers it.
        location: Where the folder sits under ``models``, as :func:`model_directories` takes it.
        suffixes: Lowercase extensions to keep, with the dot. Empty keeps every file. Filtered
            here rather than in the folder's registration, which is shared with anything else
            reading the same folder name.

    Returns:
        Names relative to whichever search directory holds them, sorted. Empty outside a
        ComfyUI process, and empty when the folder holds nothing, which a node turns into a
        combo holding a placeholder rather than an empty list ComfyUI cannot render.
    """
    if not model_directories(folder, location):
        return []
    try:
        import folder_paths
    except ImportError:
        return []
    # ComfyUI caches this list against each search directory's modification time and rebuilds it
    # when one moves, so the refresh key picks up a file added since startup. A schema is rebuilt
    # on every ``/object_info`` request, so a disk walk here would sit on that path.
    names = folder_paths.get_filename_list(folder)
    if not suffixes:
        return list(names)
    wanted = tuple(suffixes)
    return [name for name in names if os.path.splitext(name)[1].lower() in wanted]


def model_file_path(folder: str, name: str, location: str | None = None) -> Path | None:
    """Locate one model file by the name :func:`model_files` gave it.

    Args:
        folder: A model folder name, as :func:`model_directories` registers it.
        name: A name from :func:`model_files`, which may carry a subdirectory.
        location: Where the folder sits under ``models``, as :func:`model_directories` takes it.

    Returns:
        The full path, or None when no search directory holds it, and None for a name that
        names another machine, carries a drive, starts at a root or climbs out with ``..``.
    """
    from ..util.sandbox import leaves_folder

    if leaves_folder(name) is not None:
        return None
    directories = model_directories(folder, location)
    try:
        import folder_paths

        found = folder_paths.get_full_path(folder, name)
        if found:
            return Path(found)
    except (ImportError, KeyError):
        pass
    # Reached when folder_paths is absent, which is every run outside a ComfyUI process.
    for directory in directories:
        candidate = directory / name
        if os.path.isfile(candidate):
            return candidate
    return None


def resolve(
    folders: str | Sequence[str],
    repo_id: str,
    *,
    legacy: Sequence[str] = (),
    feature: str | None = None,
    marker: str = CHECKPOINT_MARKER,
) -> tuple[str, str | None]:
    """Locate a transformers-format checkpoint, or fall back to a repository id.

    Args:
        folders: ``folder_paths`` model folder name, or several searched in order.
        repo_id: Hugging Face repository holding the checkpoint.
        legacy: Filenames, relative to a search directory, that hold a v2-era checkpoint
            this backend cannot read. The first one present is named in the error.
        feature: Config key of the group whose node reached here, named in the error
            alongside ``features.network``.
        marker: File whose presence marks a directory as the checkpoint.

    Returns:
        A ``(pretrained, cache_dir)`` pair for ``from_pretrained``: an absolute local
        directory and ``None``, or ``repo_id`` and the directory a download lands in.

    Raises:
        ModelUnavailable: This repository is not on disk and ``features.network`` is off.
        ValueError: ``repo_id`` is a path rather than a repository id.
    """
    repo_id = repository_id(repo_id)
    names = (folders,) if isinstance(folders, str) else tuple(folders)
    roots = [root for name in names for root in model_directories(name)]
    # This pack's own folders first, then wherever the machine already has a copy, so a
    # checkpoint another pack downloaded is read rather than fetched again.
    for root in roots + shared_roots():
        found = _checkpoint(root, repo_id, marker)
        if found is not None:
            logger.debug("%s resolved to %s", repo_id, found)
            return str(found), None
    if network_enabled():
        return repo_id, str(roots[0]) if roots else None
    raise ModelUnavailable(
        _explain(roots, repo_id, _legacy_file(roots, legacy), feature, marker)
    )


def compute_device(name: str | None = None):
    """The device a backend runs inference on.

    Args:
        name: A device name a widget can produce: ``"cuda"`` or ``"gpu"`` for whichever
            accelerator ComfyUI is running on, ``"cpu"``, or an indexed name such as
            ``"cuda:1"``. A ``torch.device`` is accepted in place of a name. ``None`` also
            selects ComfyUI's device. A name naming a device type ComfyUI is not running on
            resolves to ComfyUI's device rather than raising, so ``"cuda"`` works on a
            CPU-only, MPS or XPU machine.

    Returns:
        A ``torch.device``, carrying an explicit index wherever the backend has one, so
        two spellings of one device cannot become two cache entries.
    """
    import torch

    managed = _managed_device()
    requested = "" if name is None else str(name).strip().lower()
    if requested in ACCELERATOR_NAMES:
        return managed
    device = torch.device(requested)
    if device.type == "cpu":
        return device
    if device.type != managed.type:
        logger.warning(
            "This machine runs ComfyUI on %s, not %s, so the model loads onto %s instead.",
            managed,
            requested,
            managed,
        )
        return managed
    return managed if device.index is None else device


def unpin_staged() -> bool:
    """Hand back the device pages a finished model has staged.

    Returns:
        True where the pages were handed back, False on an install that stages nothing.
    """
    try:
        import comfy.memory_management
        import comfy.model_management
        import comfy.model_prefetch
        import comfy_aimdo.model_vbar
    except ImportError:
        return False
    if not getattr(comfy.memory_management, "aimdo_enabled", False):
        return False
    comfy.model_prefetch.cleanup_prefetch_queues()
    comfy.model_management.reset_cast_buffers()
    comfy_aimdo.model_vbar.vbars_reset_watermark_limits()
    return True


def release_models() -> bool:
    """Unload every model ComfyUI holds and hand the cached device memory back.

    Returns:
        True where the models were released, False outside ComfyUI or when releasing failed.
    """
    import gc

    try:
        import comfy.model_management as management
    except ImportError:
        return False
    try:
        unpin_staged()
        management.unload_all_models()
        gc.collect()
        management.soft_empty_cache(force=True)
    except Exception as error:
        logger.warning("the loaded models could not be released: %s", error)
        return False
    return True


def offload_device(load_device=None):
    """Where a backend's weights rest between executions.

    Args:
        load_device: The device inference runs on, as a ``torch.device`` or a name. A
            CPU-bound backend rests where it computes, having no VRAM to vacate.

    Returns:
        A ``torch.device``: ComfyUI's own offload device, which is the CPU unless it was
        started with ``--highvram``, in which case weights stay put as asked.
    """
    import torch

    if load_device is not None:
        load_device = compute_device(load_device)
        if load_device.type == "cpu":
            return load_device
    management = _management()
    if management is None:
        return torch.device("cpu")
    return management.unet_offload_device()


def managed(key: tuple, build, *, device: str | None = None, name: str | None = None) -> Backend:
    """Return the memoized :class:`Backend` for ``key``, building it on first use.

    Args:
        key: Hashable identifier for the backend and the weights it was built from. The
            resolved device is appended, so one checkpoint asked for on two devices is two
            entries rather than one entry serving the wrong device.
        build: Zero-argument callable returning ``(processor, model)``, the model as
            ``from_pretrained`` returns it.
        device: Device name for inference, or ``None`` for ComfyUI's compute device.
        name: What ComfyUI's model management calls the model in its log, such as
            ``"BiRefNet"``. ``None`` uses the model's own class name.

    Returns:
        The cached backend.
    """
    load_device = compute_device(device)
    resting = offload_device(load_device)
    entry = (*key, str(load_device))
    backend = _lookup(entry)
    if backend is not None:
        return backend
    processor, model = build()
    model.eval()
    model.to(resting)
    logger.debug("%s built for %s, resting on %s", entry, load_device, resting)
    name = name or type(model).__name__
    patcher = _patcher(model, load_device, resting, name)
    backend = Backend(
        processor, model, load_device, resting, patcher, key=key, build=build, name=name
    )
    return _store(entry, backend)


def managed_module(
    key: tuple, build, *, device: str | None = None, name: str | None = None
) -> Backend:
    """Return the memoized :class:`Backend` for a bare module, building it on first use.

    Args:
        key: Hashable identifier for the module and the weights it was built from.
        build: Zero-argument callable returning the module itself.
        device: Device name for inference, or ``None`` for ComfyUI's compute device.
        name: What ComfyUI's model management calls the module in its log. ``None`` uses
            its own class name.

    Returns:
        The cached backend, whose ``processor`` is ``None``.
    """
    return managed(key, lambda: (None, build()), device=device, name=name)


def cached(key: tuple, build):
    """Return the memoized object for ``key``, building it on first use.

    Args:
        key: Hashable identifier for the backend and the weights it was built from.
        build: Zero-argument callable returning the loaded backend.

    Returns:
        The cached object.
    """
    value = _lookup(key)
    if value is not None:
        return value
    return _store(key, build())


def _management():
    """``comfy.model_management``, or ``None`` outside a ComfyUI process."""
    try:
        import comfy.model_management as model_management
    except ImportError:
        logger.debug("comfy.model_management is unavailable, so this model is not tracked")
        return None
    return model_management


def _managed_device():
    """ComfyUI's compute device, or the CPU outside a ComfyUI process."""
    import torch

    management = _management()
    if management is None:
        return torch.device("cpu")
    return management.get_torch_device()


def _patcher(model, load_device, resting, name: str):
    """Register ``model`` with ComfyUI's model management under ``name``, or return ``None``.

    Returns:
        A ``ModelPatcher`` owning ``model``, or ``None`` when there is no VRAM to manage:
        a CPU-bound backend, or a process with no ComfyUI in it.
    """
    if load_device.type == "cpu":
        return None
    try:
        import comfy.model_patcher
    except ImportError as error:
        logger.warning(
            "comfy.model_patcher will not import (%s), so this model loads onto %s without "
            "ComfyUI accounting for the memory it takes there.",
            error,
            load_device,
        )
        return None
    return comfy.model_patcher.ModelPatcher(
        _holder(model, name), load_device=load_device, offload_device=resting
    )


def _holder(model, name: str):
    """An ``nn.Module`` class called ``name``, owning ``model`` as its only submodule.

    Args:
        model: The network a ``ModelPatcher`` drives through the container.
        name: The container's class name, which ComfyUI's model management logs.
    """
    import torch

    holder = type(name, (torch.nn.Module,), {"__module__": __name__})()
    # ModelPatcher assigns model.device as it moves weights between devices, and a
    # transformers model exposes device as a read-only property, so the patcher is handed a
    # container whose attribute is writable.
    holder.device = None
    holder.model = model
    return holder


def _lookup(key: tuple):
    """The cached value for ``key``, marked as the most recently used, or ``None``."""
    value = _cache.pop(key, None)
    if value is None:
        return None
    _cache[key] = value
    return value


def _store(key: tuple, value):
    """Cache ``value`` under ``key``, dropping least recently used entries over the limit.

    Returns:
        ``value``.
    """
    _cache[key] = value
    while len(_cache) > CACHE_LIMIT:
        oldest = next(iter(_cache))
        logger.debug("dropping cached model %s to stay under %s", oldest, CACHE_LIMIT)
        _release(_cache.pop(oldest))
    return value


def _release(dropped) -> None:
    """Unregister an evicted backend and hand back the memory it was holding."""
    management = _management()
    patcher = getattr(dropped, "patcher", None)
    unload = getattr(management, "unload_model_and_clones", None)
    if patcher is not None and unload is not None:
        unload(patcher)
    # current_loaded_models holds the patcher by weak reference, so the collect has to run
    # with no strong reference left anywhere, this frame's own names included.
    del dropped, patcher
    gc.collect()
    if management is not None:
        management.soft_empty_cache()


#: One component of a repository id: a name, not a path. It cannot be ``.`` or ``..``, and
#: it holds no separator, no drive letter and no whitespace.
_SEGMENT = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*")


def repository_id(value: str) -> str:
    """Confirm a model id widget holds a repository id rather than a path.

    ``owner/name`` is joined onto each model directory to look for a checkpoint.

    Args:
        value: The raw widget value.

    Returns:
        The value, stripped.

    Raises:
        ValueError: The value is empty, carries a drive, a root, a backslash or more than
            one ``/``, or holds a ``.`` or ``..`` segment.
    """
    # Both os.path.join and pathlib.Path drop the directory they are joined to for an
    # absolute right-hand side or one carrying a drive, so a path here would reach past the
    # model directories and whatever it names would be handed to from_pretrained to read.
    text = value.strip()
    parts = text.split("/")
    if text and len(parts) <= 2 and all(_SEGMENT.fullmatch(part) for part in parts):
        return text
    raise ValueError(
        f"`{value}` is not a Hugging Face repository id. This widget takes a name such as "
        f"`Salesforce/blip-vqa-base`, not a path: a path here would be read instead of the "
        f"models directory, and this pack does not load weights from anywhere a workflow "
        f"names. Put a local checkpoint in ComfyUI's models directory, or add its folder to "
        f"extra_model_paths.yaml."
    )


def _layouts(repo_id: str) -> list[str]:
    """Directory names, relative to a search directory, that can hold ``repo_id``.

    Three layouts: the repository name alone, ``owner/name``, and ``models--owner--name``.
    """
    # All three exist on real installs. v2 created the cache tree by passing these
    # directories as cache_dir.
    owner, _, name = repo_id.partition("/")
    if not name:
        owner, name = "", owner
    if not name:
        return []
    if not owner:
        return [name, f"models--{name}"]
    return [name, f"{owner}/{name}", f"models--{owner}--{name}"]


def _checkpoint(root: Path, repo_id: str, marker: str = CHECKPOINT_MARKER) -> Path | None:
    """The directory under ``root`` holding ``repo_id`` in transformers format, if any.

    ``root`` itself is never a candidate, however plausible its ``config.json`` looks.
    """
    # One search directory holds every repository of its kind, so accepting the root would
    # serve one checkpoint for every model_size a node asks for.
    for relative in _layouts(repo_id):
        candidate = root / relative
        if os.path.isfile(candidate / marker):
            return candidate
        snapshot = _newest_snapshot(candidate / "snapshots", marker)
        if snapshot is not None:
            return snapshot
    return None


def _newest_snapshot(snapshots: Path, marker: str = CHECKPOINT_MARKER) -> Path | None:
    """The most recently written revision in a Hugging Face cache ``snapshots`` tree."""
    if not os.path.isdir(snapshots):
        return None
    revisions = [
        snapshots / name for name in os.listdir(snapshots)
        if os.path.isfile(snapshots / name / marker)
    ]
    if not revisions:
        return None
    return max(revisions, key=os.path.getmtime)


def _legacy_file(roots: Sequence[Path], legacy: Sequence[str]) -> Path | None:
    """The first v2-era checkpoint present under ``roots``, or ``None``."""
    for root in roots:
        for name in legacy:
            candidate = root / name
            if os.path.isfile(candidate):
                return candidate
    return None


def _explain(
    roots: Sequence[Path],
    repo_id: str,
    legacy_file: Path | None,
    feature: str | None,
    marker: str = CHECKPOINT_MARKER,
) -> str:
    """Build the message for weights that could not be resolved."""
    lines = []
    if legacy_file is not None:
        lines.append("This file cannot be loaded:")
        lines.append(f"    {legacy_file}")
        lines.append("It is the original single-file checkpoint. This node now runs on")
        lines.append("transformers, which reads a repository directory instead, and the")
        lines.append(f"replacement for that file is {repo_id}.")
    else:
        lines.append(f"The weights for this node are missing. It needs {repo_id},")
        lines.append("in Hugging Face repository format.")
    candidates = [root / relative for root in roots for relative in _layouts(repo_id)]
    if candidates:
        lines.append(f"No {marker} was found in any of:")
        lines += [f"    {candidate}" for candidate in candidates]
        lines.append(f"Clone or download {repo_id} to the first of those paths.")
    else:
        lines.append("No model directory could be resolved, because folder_paths is not")
        lines.append("importable here.")
    lines.append(f"Or set {NETWORK_FEATURE}: true in config.yaml to download it on first")
    lines.append("use.")
    if feature:
        lines.append(f"This node was loaded because {feature} is enabled in config.yaml.")
    return "\n".join(lines)
