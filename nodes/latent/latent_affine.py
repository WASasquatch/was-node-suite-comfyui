"""Scale and offset a latent through a generated or supplied mask."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ...modules.compat.types import DICT
from ...modules.interface import preview, run_result
from ...modules.latent import affine
from ...modules.latent import affine_patterns as patterns

#: What every affine node says about its mask pattern.
PATTERN_HINT = (
    "Which mask decides where the affine lands. 'solid' covers everything; 'white_noise' "
    "and the coloured noises are grain of different coarseness; 'perlin', 'checker', "
    "'bayer', 'cross_hatch', 'worley_edges' and the rest are shapes; 'detail_region', "
    "'smooth_region', 'edges_sobel' and 'edges_laplacian' are read off the latent itself; "
    "'external_mask' uses the mask wired in."
)

#: What every affine node says about its frame handling.
TEMPORAL_HINT = (
    "How a video latent's mask varies over time. 'static' = one mask on every frame, in the "
    "same place all clip. 'per_frame' = an unrelated mask each frame. 'drift' = one mask "
    "slid across the frame, set by drift_speed, drift_angle_deg and drift_renew on Affine "
    "Options. The content-aware patterns ignore this, and so does an image latent."
)

#: What every affine node says about its seed.
SEED_HINT = (
    "Seeds the mask. The same seed always draws the same mask, so change it to move the "
    "grain without changing anything else. Ignored by the content-aware patterns and by "
    "'external_mask', which read what they are given."
)

#: What every affine node says about a supplied mask.
EXTERNAL_HINT = (
    "A mask of your own, resized onto the latent. On pattern 'external_mask' it is the "
    "mask; on any other pattern it gates the generated one, so the affine reaches only "
    "where this is white. One mask covers every frame, or one per frame."
)

#: What every affine node says about its options socket.
OPTIONS_HINT = (
    "Pattern parameters and mask shaping from an Affine Options node. Leave it unwired "
    "and every value takes its default. A pattern set there wins over the pattern widget."
)

#: What every affine node says about which streams it reaches.
STREAMS_HINT = (
    "Which streams of a packed audio and video latent the affine reaches. 'video' = "
    "stream 0, 'audio' = the rest, 'both' = all of them. An ordinary latent has only a "
    "video stream, so 'audio' does nothing to it."
)


def figures(before, after, mask, scale: float, bias: float) -> dict:
    """The numbers an affine reports about what it did.

    Args:
        before: The latent stream that went in.
        after: The latent stream that came out.
        mask: The mask it was transformed through, or None.
        scale: The multiplier that was applied.
        bias: The offset that was applied.

    Returns:
        A mapping of figure name to number.
    """
    out = {"scale": round(float(scale), 4), "bias": round(float(bias), 4)}
    if mask is not None and mask.numel():
        out["mask coverage %"] = round(float(mask.float().mean()) * 100.0, 2)
    if torch.is_tensor(before) and torch.is_tensor(after):
        out["mean before"] = round(float(before.float().mean()), 4)
        out["mean after"] = round(float(after.float().mean()), 4)
        out["peak after"] = round(float(after.float().abs().max()), 4)
    return out


def shape_text(samples) -> str:
    """One line naming the shape of every stream of a latent.

    Args:
        samples: A tensor, or a NestedTensor packing several.

    Returns:
        The stream shapes, joined by ``+``.
    """
    parts = affine.streams_of(samples)
    return " + ".join("x".join(str(int(n)) for n in part.shape) for part in parts)


class LatentAffine(io.ComfyNode):
    """Multiply and offset a latent wherever a mask says to."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLatentAffine",
            display_name="Latent Affine",
            search_aliases=[
                "WASLatentAffine",
                "Latent Affine",
                "latent scale bias",
                "latent noise mask",
                "latent contrast",
                "affine",
            ],
            category="WAS Suite/Latent/Transform",
            description=(
                "Multiply a latent and add an offset to it, but only where a mask says to. "
                "The mask can be procedural grain, a repeating shape, a reading of the "
                "latent's own detail or edges, or one wired in. Small moves either side of "
                "1.0 change texture and contrast before a second sampling pass; the same "
                "transform applied during sampling is what the Affine samplers do."
            ),
            inputs=[
                io.Latent.Input(
                    "latent",
                    tooltip=(
                        "The latent to transform. Image, video and packed audio and video "
                        "latents are all handled."
                    ),
                ),
                io.Float.Input(
                    "scale",
                    default=0.96,
                    min=0.0,
                    max=2.0,
                    step=0.001,
                    tooltip=(
                        "What the latent is multiplied by where the mask is white. 1.0 = no "
                        "change; 0.96 takes a little energy out, which softens; 1.2 pushes "
                        "texture and contrast up."
                    ),
                ),
                io.Float.Input(
                    "bias",
                    default=0.0,
                    min=-2.0,
                    max=2.0,
                    step=0.001,
                    tooltip=(
                        "What is added where the mask is white, beside scale rather than "
                        "through it. 0.0 = no shift; 0.1 lifts, -0.1 drops. bias_field on "
                        "Affine Options decides whether that is one flat offset or a noise "
                        "field."
                    ),
                ),
                io.Combo.Input("pattern", options=patterns.PATTERNS, default="white_noise", tooltip=PATTERN_HINT),
                io.Combo.Input(
                    "temporal_mode",
                    options=affine.TEMPORAL_MODES,
                    default="static",
                    tooltip=TEMPORAL_HINT,
                ),
                io.Int.Input("seed", default=0, min=0, max=0x7FFFFFFF, tooltip=SEED_HINT),
                io.Combo.Input(
                    "streams",
                    options=affine.STREAM_MODES,
                    default="video",
                    optional=True,
                    tooltip=STREAMS_HINT,
                ),
                io.Mask.Input("external_mask", optional=True, tooltip=EXTERNAL_HINT),
                DICT.Input("affine_options", optional=True, tooltip=OPTIONS_HINT),
            ],
            outputs=[
                io.Latent.Output(display_name="latent", tooltip="The transformed latent."),
                io.Mask.Output(
                    display_name="mask",
                    tooltip=(
                        "The mask the transform ran through, at latent resolution. White is "
                        "where the full scale and bias landed."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        latent,
        scale,
        bias,
        pattern,
        temporal_mode,
        seed,
        streams="video",
        external_mask=None,
        affine_options=None,
    ) -> io.NodeOutput:
        options = affine.resolve(affine_options)
        chosen = str((affine_options or {}).get("pattern", pattern))

        samples = latent["samples"]
        result, mask = affine.apply_affine(
            samples,
            float(scale),
            float(bias),
            chosen,
            str(temporal_mode),
            int(seed),
            str(streams),
            external_mask,
            options,
        )
        if mask is None:
            mask = affine.mask_like(samples)

        out = {key: value for key, value in latent.items() if key != "samples"}
        out["samples"] = result

        _report(samples, result, mask, float(scale), float(bias), chosen, streams)
        return io.NodeOutput(out, mask)


def _report(samples, result, mask, scale: float, bias: float, pattern: str, streams: str) -> None:
    """Publish what the transform did, for the node's own panel.

    Args:
        samples: The latent that went in.
        result: The latent that came out.
        mask: The mask, or None.
        scale: The multiplier applied.
        bias: The offset applied.
        pattern: The pattern used.
        streams: Which streams were selected.
    """
    try:
        if mask is not None:
            preview.publish_mask_output(mask)
        if not run_result.watching():
            return
        before = affine.streams_of(samples)[0]
        after = affine.streams_of(result)[0]
        touched = affine.stream_indices(streams, len(affine.streams_of(samples)))
        counts = figures(before, after, mask, scale, bias)
        coverage = counts.get("mask coverage %")
        summary = f"{pattern} at scale {scale:g}"
        if abs(bias) > 0:
            summary += f" and bias {bias:g}"
        if coverage is not None:
            summary += f", over {coverage:g}% of the latent"
        run_result.publish(
            status=run_result.OK if touched else run_result.WARNING,
            summary=summary if touched else f"streams '{streams}': no stream matched",
            counts=counts,
            facts={
                "pattern": pattern,
                "latent": shape_text(samples),
                "streams": f"{len(touched)} of {len(affine.streams_of(samples))}",
            },
        )
    except Exception:
        run_result.publish(status=run_result.WARNING, summary="the affine report could not be built")
