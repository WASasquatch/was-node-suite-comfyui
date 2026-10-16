"""BLIP captioning and visual question answering, on transformers.

One repository per task: conditional generation for captions, question answering for VQA,
each with its own processor.
"""

from __future__ import annotations

from .. import deps
from . import ModelUnavailable, managed, resolve

__all__ = ["load"]

#: Config key of the feature group these nodes are gated on.
FEATURE = "features.blip"

#: ``folder_paths`` model folder searched for BLIP checkpoints.
FOLDER = "blip"

#: Task -> (transformers model class, default repository).
TASKS = {
    "caption": ("BlipForConditionalGeneration", "Salesforce/blip-image-captioning-base"),
    "question": ("BlipForQuestionAnswering", "Salesforce/blip-vqa-base"),
}

#: ``blip_model`` values v2 rewrote to the captioning repository before loading. The widget
#: is a free STRING, so a workflow saved when the v2 node shipped can still carry one.
LEGACY_NAMES = frozenset({"caption", "interrogate"})


def load(repo_id: str = "", task: str = "caption", device: str | None = None):
    """Load a BLIP processor and model.

    Args:
        repo_id: Hugging Face repository id, such as ``"Salesforce/blip-vqa-base"``.
            Empty, blank or one of :data:`LEGACY_NAMES` selects the default for ``task``.
        task: ``"caption"`` for conditional generation, ``"question"`` for VQA.
        device: Device name from the ``device`` widget, or ``None`` for ComfyUI's compute
            device.

    Returns:
        A :class:`~modules.model.Backend` whose ``processor`` and ``model`` are the BLIP
        pair, resting on the offload device until ``Backend.load()`` is called.

    Raises:
        ValueError: ``task`` is not a key of :data:`TASKS`.
        DependencyError: transformers is not importable.
        ModelUnavailable: No local checkpoint and ``features.network`` is off, or the
            checkpoint holds no ``model.safetensors``.
    """
    if task not in TASKS:
        raise ValueError(f"BLIP task must be one of {', '.join(TASKS)}, not {task!r}")
    model_class, default_repo = TASKS[task]
    repo_id = repo_id.strip()
    if not repo_id or (task == "caption" and repo_id in LEGACY_NAMES):
        repo_id = default_repo

    transformers = deps.require("transformers", feature=FEATURE)
    pretrained, cache_dir = resolve(FOLDER, repo_id, feature=FEATURE)

    def build():
        processor = transformers.BlipProcessor.from_pretrained(
            pretrained, cache_dir=cache_dir, token=False
        )
        try:
            model = getattr(transformers, model_class).from_pretrained(
                pretrained, cache_dir=cache_dir, use_safetensors=True, trust_remote_code=False,
                token=False,
            )
        except OSError as error:
            if "safetensors" not in str(error):
                raise
            raise ModelUnavailable(
                f"{repo_id} has no model.safetensors at {pretrained}, so it was not loaded.\n"
                f"    BLIP weights are read from model.safetensors only.\n"
                f"    Pick a repository that publishes one, or put a model.safetensors copy "
                f"of this checkpoint in models/{FOLDER}."
            ) from error
        return processor, model

    return managed(("blip", model_class, pretrained), build, device=device)
