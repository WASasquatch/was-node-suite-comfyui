"""Body, whole-body and animal pose estimation, on a transformers detector and pose pair.

:func:`load` returns a :class:`Pair` holding both backends. ``Pair.layout`` names the
keypoint set the answer is in.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import deps
from . import managed, published_checkpoint, resolve

__all__ = ["DETECTOR", "DETECTORS", "MODELS", "Pair", "Spec", "load", "remap"]

#: Config key of the feature group these nodes are gated on.
FEATURE = "features.preprocessors"

#: ``folder_paths`` model folder searched for the checkpoints.
FOLDER = "pose"

#: Repository of the detector that finds the subjects, whose boxes the pose model reads.
DETECTOR = "PekingU/rtdetr_r18vd_coco_o365"

#: The detector repositories any option names, so the folder can be described on its own.
DETECTORS = (DETECTOR,)

#: Repository publishing the original releases, which carry no transformers config.
RELEASES = "JunkyByte/easy_ViTPose"


@dataclass(frozen=True)
class Spec:
    """Where one option's pose weights come from, and what they answer.

    Attributes:
        repo: Hugging Face repository holding the weights.
        subfolder: Directory inside that repository, empty for a transformers conversion.
        filename: File inside that directory, or ``None`` for a transformers conversion.
        layout: Keypoint set the answer is in, one of ``body``, ``wholebody``, ``animal``.
        keypoints: How many points the head answers.
        hidden: Width of the backbone.
        layers: How many transformer layers the backbone holds.
        heads: How many attention heads each layer holds.
    """

    repo: str
    subfolder: str = ""
    filename: str | None = None
    layout: str = "body"
    keypoints: int = 17
    hidden: int = 384
    layers: int = 12
    heads: int = 12


#: Widget option -> where its weights are and what they answer.
MODELS = {
    "ViTPose Base": Spec("usyd-community/vitpose-base-simple"),
    "ViTPose Small": Spec("usyd-community/vitpose-plus-small"),
    "ViTPose Wholebody": Spec(
        RELEASES, "torch/wholebody", "vitpose-s-wholebody.pth", "wholebody", keypoints=133
    ),
    "ViTPose Animal": Spec(
        RELEASES, "torch/ap10k", "vitpose-s-ap10k.pth", "animal", keypoints=17
    ),
}

#: Height and width one box is read at.
READ_SIZE = (256, 192)


@dataclass(frozen=True)
class Pair:
    """The two backends a pose estimate runs through.

    Attributes:
        detector: Backend whose ``model`` finds subjects, and whose ``processor`` prepares
            a frame for it.
        poser: Backend whose ``model`` reads joints out of one subject's box.
        experts: How many experts the pose backbone mixes, 1 for a plain one. Above 1 the
            forward pass is told which dataset it is answering for.
        layout: Keypoint set the answer is in, one of ``body``, ``wholebody``, ``animal``.
    """

    detector: object
    poser: object
    experts: int
    layout: str


def load(name: str = "ViTPose Base", device: str | None = None) -> Pair:
    """Load a detector and a pose model.

    Args:
        name: One of the keys of :data:`MODELS`.
        device: Device name, or ``None`` for ComfyUI's compute device.

    Returns:
        A :class:`Pair`, each backend resting on the offload device until it is loaded.

    Raises:
        ValueError: ``name`` is not a key of :data:`MODELS`.
        DependencyError: transformers is not importable.
        ModelUnavailable: A checkpoint is absent and ``features.network`` is off.
    """
    if name not in MODELS:
        raise ValueError(f"Pose model must be one of {', '.join(MODELS)}, not {name!r}")
    spec = MODELS[name]

    transformers = deps.require("transformers", feature=FEATURE)

    detector = _backend(
        transformers, DETECTOR, "RTDetrForObjectDetection", device, "pose_detector"
    )
    if spec.filename is None:
        poser = _backend(
            transformers, spec.repo, "VitPoseForPoseEstimation", device, "pose_estimator"
        )
        backbone = getattr(poser.model.config, "backbone_config", None)
        experts = int(getattr(backbone, "num_experts", 1))
    else:
        poser = _released(transformers, name, spec, device)
        experts = 1
    return Pair(detector=detector, poser=poser, experts=experts, layout=spec.layout)


def _backend(transformers, repo_id: str, class_name: str, device, key: str):
    """Build or return the cached backend for one repository."""
    pretrained, cache_dir = resolve(FOLDER, repo_id, feature=FEATURE)

    def build():
        processor = transformers.AutoImageProcessor.from_pretrained(
            pretrained, cache_dir=cache_dir, token=False
        )
        model = getattr(transformers, class_name).from_pretrained(
            pretrained, cache_dir=cache_dir, token=False
        )
        return processor, model

    return managed((key, pretrained), build, device=device)


def _released(transformers, name: str, spec: Spec, device):
    """Build or return the cached backend for one original release."""
    import torch

    path = published_checkpoint(
        FOLDER,
        spec.repo,
        spec.filename,
        subfolder=spec.subfolder,
        feature=FEATURE,
        what="The pose network",
    )

    def build():
        processor = transformers.VitPoseImageProcessor()
        backbone = transformers.VitPoseBackboneConfig(
            image_size=list(READ_SIZE),
            patch_size=[16, 16],
            hidden_size=spec.hidden,
            num_hidden_layers=spec.layers,
            num_attention_heads=spec.heads,
            num_experts=1,
        )
        config = transformers.VitPoseConfig(
            backbone_config=backbone,
            use_simple_decoder=False,
            num_labels=spec.keypoints,
        )
        model = transformers.VitPoseForPoseEstimation(config)
        stored = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(remap(stored.get("state_dict", stored), spec))
        model.eval()
        return processor, model

    return managed((f"pose_{name}", path), build, device=device)


def remap(stored: dict, spec: Spec) -> dict:
    """Rename one original release's tensors to the transformers layout.

    Args:
        stored: The release's own ``state_dict``, whose attention weights are fused.
        spec: The option being loaded, whose ``layers`` sets how many blocks are walked.

    Returns:
        A ``state_dict`` accepted by ``VitPoseForPoseEstimation`` without slack.

    Raises:
        KeyError: A tensor the transformers layout needs is absent from ``stored``.
    """
    out = {
        "backbone.embeddings.position_embeddings": stored["backbone.pos_embed"],
        "backbone.embeddings.patch_embeddings.projection.weight": stored[
            "backbone.patch_embed.proj.weight"
        ],
        "backbone.embeddings.patch_embeddings.projection.bias": stored[
            "backbone.patch_embed.proj.bias"
        ],
        "backbone.layernorm.weight": stored["backbone.last_norm.weight"],
        "backbone.layernorm.bias": stored["backbone.last_norm.bias"],
    }
    for layer in range(spec.layers):
        source = f"backbone.blocks.{layer}."
        target = f"backbone.encoder.layer.{layer}."
        for suffix in ("weight", "bias"):
            fused = stored[f"{source}attn.qkv.{suffix}"]
            third = fused.shape[0] // 3
            for index, part in enumerate(("query", "key", "value")):
                out[f"{target}attention.attention.{part}.{suffix}"] = fused[
                    index * third : (index + 1) * third
                ]
            out[f"{target}attention.output.dense.{suffix}"] = stored[
                f"{source}attn.proj.{suffix}"
            ]
            out[f"{target}mlp.fc1.{suffix}"] = stored[f"{source}mlp.fc1.{suffix}"]
            out[f"{target}mlp.fc2.{suffix}"] = stored[f"{source}mlp.fc2.{suffix}"]
            out[f"{target}layernorm_before.{suffix}"] = stored[f"{source}norm1.{suffix}"]
            out[f"{target}layernorm_after.{suffix}"] = stored[f"{source}norm2.{suffix}"]

    head = {
        "head.deconv1.weight": "keypoint_head.deconv_layers.0.weight",
        "head.deconv2.weight": "keypoint_head.deconv_layers.3.weight",
        "head.conv.weight": "keypoint_head.final_layer.weight",
        "head.conv.bias": "keypoint_head.final_layer.bias",
    }
    for target, source in head.items():
        out[target] = stored[source]
    for index, number in ((1, 1), (2, 4)):
        for suffix in ("weight", "bias", "running_mean", "running_var", "num_batches_tracked"):
            out[f"head.batchnorm{index}.{suffix}"] = stored[
                f"keypoint_head.deconv_layers.{number}.{suffix}"
            ]
    return out
