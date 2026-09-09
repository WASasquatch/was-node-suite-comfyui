"""Every mask pattern's parameters, on one node that draws only the chosen pattern's."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import DICT
from ...modules.latent import affine
from ...modules.latent import affine_patterns as patterns
from .latent_affine import PATTERN_HINT


class AffineOptions(io.ComfyNode):
    """Build the options dictionary the affine nodes read their pattern settings from."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAffineOptions",
            display_name="Affine Options",
            search_aliases=[
                "WASAffineOptions",
                "Affine Options",
                "affine pattern options",
                "noise options",
                "mask options",
            ],
            category="WAS Suite/Latent/Transform",
            description=(
                "Set the pattern an affine masks through and the parameters that shape it, "
                "then wire the result into Latent Affine or any of the Affine samplers. "
                "Only the chosen pattern's own settings are drawn, so the node stays short "
                "whichever one is picked. The pattern set here is the one the affine uses, "
                "and content_gate holds it back to the flat areas, the detail or the edges "
                "of the picture."
            ),
            inputs=[
                io.Combo.Input("pattern", options=patterns.PATTERNS, default="white_noise", tooltip=PATTERN_HINT),
                # Per-pattern settings, in the order the pattern list offers them.
                io.Float.Input(
                    "green_center_frac", default=0.35, min=0.0, max=1.0, step=0.01,
                    tooltip=(
                        "Where the green noise band sits, as a share of the finest detail "
                        "the latent can hold. 0.1 = coarse blotches, 0.35 = mid, 0.8 = fine."
                    ),
                ),
                io.Float.Input(
                    "green_bandwidth_frac", default=0.15, min=0.01, max=1.0, step=0.01,
                    tooltip="How wide that band is. 0.05 = one grain size, 0.4 = a broad mix.",
                ),
                io.Int.Input(
                    "black_bins_per_kpx", default=512, min=1, max=500000,
                    tooltip=(
                        "How many frequencies black noise keeps alive, per thousand samples. "
                        "16 = a few standing ripples, 512 = a busy weave."
                    ),
                ),
                io.Float.Input(
                    "hatch_freq_cyc_px", default=0.45, min=0.01, max=2.0, step=0.01,
                    tooltip=(
                        "Cycles per sample in each hatch line. 0.1 = wide bars, 0.45 = fine "
                        "lines, above 0.5 the lines alias into moire on purpose."
                    ),
                ),
                io.Float.Input(
                    "hatch_angle1_deg", default=0.0, min=0.0, max=179.0, step=1.0,
                    tooltip="Angle of the first set of lines, in degrees. 0 = horizontal.",
                ),
                io.Float.Input(
                    "hatch_angle2_deg", default=90.0, min=0.0, max=179.0, step=1.0,
                    tooltip="Angle of the second set. 90 crosses the first at a right angle.",
                ),
                io.Boolean.Input(
                    "hatch_square", default=False,
                    tooltip=(
                        "`true` squares the waves off into hard bars; `false` leaves them "
                        "smooth sinusoids."
                    ),
                ),
                io.Float.Input(
                    "hatch_phase_jitter", default=0.0, min=0.0, max=1.0, step=0.01,
                    tooltip="How far the seed may slide the lines. 0.0 = fixed, 1.0 = anywhere.",
                ),
                io.Int.Input(
                    "hatch_supersample", default=1, min=1, max=8,
                    tooltip=(
                        "How many times over the lines are drawn before averaging down. 1 = "
                        "fast and jagged, 4 = smooth edges at four times the cost."
                    ),
                ),
                io.Float.Input(
                    "highpass_cutoff_frac", default=0.7, min=0.01, max=1.0, step=0.01,
                    tooltip=(
                        "Below this share of the finest detail the latent can hold, the noise "
                        "is rolled off. 0.3 keeps most of it, 0.9 keeps only the finest grain."
                    ),
                ),
                io.Int.Input(
                    "highpass_order", default=2, min=1, max=10,
                    tooltip="How sharply that roll-off bites. 1 = gentle, 8 = a hard edge.",
                ),
                io.Float.Input(
                    "ring_center_frac", default=0.9, min=0.0, max=1.0, step=0.01,
                    tooltip="Which single grain size the ring keeps. 0.9 is close to the finest.",
                ),
                io.Float.Input(
                    "ring_bandwidth_frac", default=0.05, min=0.005, max=1.0, step=0.005,
                    tooltip="How pure that grain is. 0.01 = one size only, 0.2 = a small spread.",
                ),
                io.Float.Input(
                    "poisson_radius_px", default=8.0, min=1.0, max=256.0, step=0.5,
                    tooltip=(
                        "Closest two points may sit, in latent samples. 4 = a dense stipple, "
                        "24 = widely spaced dots with broad space between them."
                    ),
                ),
                io.Float.Input(
                    "poisson_softness", default=6.0, min=0.1, max=256.0, step=0.1,
                    tooltip="How fast the field brightens away from a point. 1 = tight dots, 20 = soft cells.",
                ),
                io.Float.Input(
                    "worley_points_per_kpx", default=2.0, min=0.1, max=200.0, step=0.1,
                    tooltip="Cell seeds per thousand samples. 0.5 = a few large cells, 20 = a fine mesh.",
                ),
                io.Combo.Input(
                    "worley_metric", options=["L2", "L1"], default="L2",
                    tooltip="'L2' grows round cells, 'L1' grows diamond ones with straight edges.",
                ),
                io.Float.Input(
                    "worley_edge_sharpness", default=1.0, min=0.1, max=8.0, step=0.1,
                    tooltip="How thin the boundaries are drawn. 0.5 = broad seams, 4 = hairlines.",
                ),
                io.Int.Input(
                    "tile_line_tile_size", default=32, min=4, max=512,
                    tooltip="Tile side in latent samples. 8 = a fine weave, 64 = large panels.",
                ),
                io.Float.Input(
                    "tile_line_freq_cyc_px", default=0.4, min=0.01, max=2.0, step=0.01,
                    tooltip="Cycles per sample inside a tile. 0.1 = wide bands, 0.4 = fine lines.",
                ),
                io.Float.Input(
                    "tile_line_jitter", default=0.25, min=0.0, max=1.0, step=0.01,
                    tooltip="How far a tile's lines may slide. 0.0 lines the tiles up, 1.0 breaks them apart.",
                ),
                io.Int.Input(
                    "dot_cell_size", default=12, min=2, max=256,
                    tooltip="Halftone cell side in latent samples. 4 = a fine screen, 32 = a coarse one.",
                ),
                io.Float.Input(
                    "dot_jitter_px", default=1.5, min=0.0, max=10.0, step=0.1,
                    tooltip="How far a dot strays from its cell centre, in samples. 0 = a rigid grid.",
                ),
                io.Float.Input(
                    "dot_fill_ratio", default=0.3, min=0.01, max=0.95, step=0.01,
                    tooltip="Roughly what share of a cell a dot covers. 0.1 = pinpricks, 0.8 = nearly solid.",
                ),
                io.Int.Input(
                    "velvet_taps_per_kpx", default=10, min=1, max=10000,
                    tooltip="Impulses per thousand samples. 2 = sparse sparkle, 200 = dense speckle.",
                ),
                io.Float.Input(
                    "perlin_scale", default=64.0, min=4.0, max=1024.0, step=1.0,
                    tooltip="Samples per blob. 16 = small blobs, 64 = medium, 256 = broad drifts.",
                ),
                io.Int.Input(
                    "perlin_octaves", default=3, min=1, max=8,
                    tooltip="How many passes are summed. 1 = smooth blobs, 6 = detail at every size.",
                ),
                io.Float.Input(
                    "perlin_persistence", default=0.5, min=0.1, max=1.0, step=0.01,
                    tooltip="How much strength each finer pass keeps. 0.3 = smooth, 0.8 = rough.",
                ),
                io.Float.Input(
                    "perlin_lacunarity", default=2.0, min=1.0, max=4.0, step=0.1,
                    tooltip="How much finer each pass is than the last. 2.0 doubles the detail each time.",
                ),
                io.Int.Input(
                    "checker_size", default=8, min=2, max=256,
                    tooltip="Square side in latent samples. 4 = a tight grid, 32 = large blocks.",
                ),
                io.Int.Input(
                    "bayer_size", default=8, min=2, max=64,
                    tooltip="Dither tile side. 2, 4, 8 and 16 are the ones that tile without a seam.",
                ),
                io.Float.Input(
                    "solid_alpha", default=1.0, min=0.0, max=1.0, step=0.001,
                    tooltip=(
                        "How much of the affine a solid mask lets through. 1.0 = the whole "
                        "latent at full strength, 0.25 = a quarter of the way there."
                    ),
                ),
                io.Int.Input(
                    "content_window", default=7, min=3, max=63,
                    tooltip=(
                        "How wide a neighbourhood the detail and smooth patterns measure over, "
                        "in latent samples. 3 = fine texture, 15 = whole regions."
                    ),
                ),
                # Mask shaping, read whichever pattern is chosen.
                io.Float.Input(
                    "mask_strength", default=1.0, min=0.0, max=2.0, step=0.001,
                    tooltip=(
                        "Multiplies the mask before the affine reads it. 0.5 = half the effect "
                        "everywhere, 1.0 = as drawn, 2.0 = double, which pushes past the "
                        "scale and bias that were asked for."
                    ),
                ),
                io.Float.Input(
                    "threshold", default=0.0, min=0.0, max=1.0, step=0.001,
                    tooltip=(
                        "Cuts the mask into hard on and off at this level. 0.0 = off, leaving "
                        "the mask smooth; 0.5 keeps the brighter half."
                    ),
                ),
                io.Boolean.Input(
                    "invert_mask", default=False,
                    tooltip=(
                        "`true` swaps where the affine lands for where it does not; `false` "
                        "leaves the mask as drawn."
                    ),
                ),
                io.Float.Input(
                    "mask_blur", default=0.0, min=0.0, max=16.0, step=0.1,
                    tooltip=(
                        "Softens the mask's edges. 0.0 = off, 1.0 = a gentle feather, 6.0 = "
                        "smears fine grain into broad patches."
                    ),
                ),
                io.Float.Input(
                    "mask_sharpen", default=0.0, min=-5.0, max=5.0, step=0.01,
                    tooltip=(
                        "Raises the mask's contrast before anything else touches it. 0.0 = "
                        "off, 0.3 = subtle, 1.0 = strong, negative softens instead."
                    ),
                ),
                io.Float.Input(
                    "sharpen_radius", default=0.8, min=0.0, max=8.0, step=0.05,
                    tooltip="How wide the detail that sharpening lifts is. 0.5 = fine, 3.0 = broad.",
                ),
                io.Float.Input(
                    "sharpen_threshold", default=0.0, min=0.0, max=1.0, step=0.01,
                    tooltip="Detail weaker than this is left alone. 0.0 sharpens everything.",
                ),
                io.Boolean.Input(
                    "clamp", default=False,
                    tooltip=(
                        "`true` holds the transformed latent inside clamp_min and clamp_max; "
                        "`false` lets it go anywhere. Worth turning on where a large scale "
                        "or bias sends values far past what the model has seen."
                    ),
                ),
                io.Float.Input(
                    "clamp_min", default=-10.0, min=-100.0, max=0.0, step=0.1,
                    tooltip=(
                        "Lowest value the latent may hold once clamping is on. -10 is wide "
                        "enough for any ordinary latent; -4 is a tight leash."
                    ),
                ),
                io.Float.Input(
                    "clamp_max", default=10.0, min=0.0, max=100.0, step=0.1,
                    tooltip=(
                        "Highest value the latent may hold once clamping is on. 10 is wide "
                        "enough for any ordinary latent; 4 is a tight leash."
                    ),
                ),
                io.Int.Input(
                    "frame_seed_stride", default=9973, min=1, max=100000,
                    tooltip=(
                        "How far the seed moves between frames on 'per_frame' and 'drift'. "
                        "1 makes neighbouring frames similar; a large prime such as 9973 "
                        "makes each frame independent."
                    ),
                ),
                io.Float.Input(
                    "drift_speed", default=0.35, min=0.0, max=16.0, step=0.01,
                    tooltip=(
                        "How far the mask slides each frame on 'drift', in latent samples. "
                        "0.0 holds it still, 0.35 is a slow crawl, 2.0 sweeps across a short "
                        "clip."
                    ),
                ),
                io.Float.Input(
                    "drift_angle_deg", default=0.0, min=0.0, max=359.0, step=1.0,
                    tooltip=(
                        "Which way it slides, in degrees. 0 = right, 90 = down, 180 = left, "
                        "270 = up."
                    ),
                ),
                io.Float.Input(
                    "drift_renew", default=0.0, min=0.0, max=1.0, step=0.01,
                    tooltip=(
                        "How much of the mask is replaced each frame on 'drift', on top of "
                        "the slide. 0.0 slides one mask unchanged, 0.15 lets it turn over as "
                        "well, 1.0 matches 'per_frame'."
                    ),
                ),
                io.Combo.Input(
                    "bias_field", options=affine.BIAS_FIELDS, default="constant",
                    tooltip=(
                        "What the bias adds where the mask is white. 'constant' = one offset "
                        "everywhere, which shifts colour and tone. 'gaussian' = a noise "
                        "field, one value per latent element. On 'gaussian' a bias of 0.02 "
                        "is gentle and 0.1 is strong."
                    ),
                ),
                io.Combo.Input(
                    "content_gate", options=affine.CONTENT_GATES, default="off",
                    tooltip=(
                        "Hold a generated pattern back to where the picture allows it. 'off' "
                        "lets it cover the frame; 'smooth_region' keeps it to flat areas and "
                        "off detail; 'detail_region' does the opposite; 'edges_sobel' and "
                        "'edges_laplacian' keep it to edges. Read off each frame, so the "
                        "grain follows the subject. content_window sets how wide it reads."
                    ),
                ),
            ],
            outputs=[
                DICT.Output(
                    display_name="affine_options",
                    tooltip=(
                        "The pattern and its settings, for the affine_options socket of "
                        "Latent Affine or any Affine sampler."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        pattern,
        green_center_frac,
        green_bandwidth_frac,
        black_bins_per_kpx,
        hatch_freq_cyc_px,
        hatch_angle1_deg,
        hatch_angle2_deg,
        hatch_square,
        hatch_phase_jitter,
        hatch_supersample,
        highpass_cutoff_frac,
        highpass_order,
        ring_center_frac,
        ring_bandwidth_frac,
        poisson_radius_px,
        poisson_softness,
        worley_points_per_kpx,
        worley_metric,
        worley_edge_sharpness,
        tile_line_tile_size,
        tile_line_freq_cyc_px,
        tile_line_jitter,
        dot_cell_size,
        dot_jitter_px,
        dot_fill_ratio,
        velvet_taps_per_kpx,
        perlin_scale,
        perlin_octaves,
        perlin_persistence,
        perlin_lacunarity,
        checker_size,
        bayer_size,
        solid_alpha,
        content_window,
        mask_strength,
        threshold,
        invert_mask,
        mask_blur,
        mask_sharpen,
        sharpen_radius,
        sharpen_threshold,
        clamp,
        clamp_min,
        clamp_max,
        frame_seed_stride,
        drift_speed,
        drift_angle_deg,
        drift_renew,
        bias_field,
        content_gate,
    ) -> io.NodeOutput:
        settings = dict(locals())
        settings.pop("cls", None)
        unknown = set(settings) - set(affine.DEFAULTS) - {"pattern"}
        for key in unknown:
            settings.pop(key)
        return io.NodeOutput(settings)
