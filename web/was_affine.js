/**
 * What the affine nodes did, drawn on the nodes themselves.
 *
 * Affine Options draws only the settings its pattern and gate read. Latent Affine and the
 * three Affine samplers draw their figures beside the mask, and Affine Schedule its curve.
 */

import { app } from "../../scripts/app.js";
import { createPictureBand } from "./interface/picture_band.js";
import { createReportPanel } from "./interface/report_panel.js";
import { setWidgetHidden } from "./interface/visibility.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.AffineUI";
const SETTING_ID = "WAS.Affine.ShowReport";
const LOG_NAME = "WASNodeSuite.Affine";

const OPTIONS_NODE = "WASAffineOptions";
const SCHEDULE_NODE = "WASAffineSchedule";

// The nodes drawing a report beside the mask the affine ran through.
const MASK_NODES = [
  "WASLatentAffine",
  "WASKSamplerAffineAdvanced",
  "WASCustomSamplerAffineAdvanced",
];

// Affine Sampler answers a sampler rather than a picture, and its report has no band.
const PATCH_NODE = "WASAffineSampler";

// Which settings each pattern reads on Affine Options. A pattern absent from this reads
// none of its own.
const BY_PATTERN = {
  green_noise: ["green_center_frac", "green_bandwidth_frac"],
  black_noise: ["black_bins_per_kpx"],
  cross_hatch: [
    "hatch_freq_cyc_px",
    "hatch_angle1_deg",
    "hatch_angle2_deg",
    "hatch_square",
    "hatch_phase_jitter",
    "hatch_supersample",
  ],
  highpass_white: ["highpass_cutoff_frac", "highpass_order"],
  ring_noise: ["ring_center_frac", "ring_bandwidth_frac"],
  poisson_blue_mask: ["poisson_radius_px", "poisson_softness"],
  worley_edges: ["worley_points_per_kpx", "worley_metric", "worley_edge_sharpness"],
  tile_oriented_lines: ["tile_line_tile_size", "tile_line_freq_cyc_px", "tile_line_jitter"],
  dot_screen_jitter: ["dot_cell_size", "dot_jitter_px", "dot_fill_ratio"],
  velvet_noise: ["velvet_taps_per_kpx"],
  perlin: ["perlin_scale", "perlin_octaves", "perlin_persistence", "perlin_lacunarity"],
  checker: ["checker_size"],
  bayer: ["bayer_size"],
  solid: ["solid_alpha"],
  detail_region: ["content_window"],
  smooth_region: ["content_window"],
};

// Every per-pattern setting, which is what may be folded away.
const PER_PATTERN = new Set(Object.values(BY_PATTERN).flat());

// The width the content patterns read over, drawn for a content gate as well as for them.
const CONTENT_WINDOW = "content_window";

// The sharpening settings, drawn only while there is sharpening to shape.
const SHARPEN = ["sharpen_radius", "sharpen_threshold"];

// The clamp bounds, drawn only while clamping is on.
const CLAMP = ["clamp_min", "clamp_max"];

// Panel heights in node units: the summary, the figures, the fact rows and the picture.
const REPORT_HEIGHT = 210;
const SCHEDULE_HEIGHT = 220;
const PATCH_HEIGHT = 130;
const REPORT_MIN_WIDTH = 260;
const LABEL_WIDTH = 96;

const UI_WIDGET_NAME = "was_affine_ui";
const UI_WIDGET_TYPE = "was_affine_report";

/**
 * Whether the reports are drawn at all.
 *
 * @returns {boolean} True while the setting is on or cannot be read.
 */
function enabled() {
  try {
    const value = app?.extensionManager?.setting?.get?.(SETTING_ID);
    if (typeof value === "boolean") return value;
    const legacy = app?.ui?.settings?.getSettingValue?.(SETTING_ID);
    return typeof legacy === "boolean" ? legacy : true;
  } catch (error) {
    console.error(`[${EXT_NAME}] Failed to read ${SETTING_ID}:`, error);
    return true;
  }
}

/**
 * Draw the settings the chosen pattern reads on Affine Options, and fold the rest away.
 *
 * @param {object} node - The Affine Options node to lay out.
 * @returns {void}
 */
function foldOptions(node) {
  const widgets = new Map((node.widgets ?? []).map((widget) => [widget.name, widget]));
  const pattern = widgets.get("pattern");
  const shown = new Set(BY_PATTERN[pattern?.value] ?? []);
  const gated = String(widgets.get("content_gate")?.value ?? "off") !== "off";
  if (gated) shown.add(CONTENT_WINDOW);
  let moved = false;

  for (const name of PER_PATTERN) {
    if (setWidgetHidden(widgets.get(name), !shown.has(name))) moved = true;
  }
  const sharpening = Number(widgets.get("mask_sharpen")?.value ?? 0) !== 0;
  for (const name of SHARPEN) {
    if (setWidgetHidden(widgets.get(name), !sharpening)) moved = true;
  }
  const clamping = Boolean(widgets.get("clamp")?.value);
  for (const name of CLAMP) {
    if (setWidgetHidden(widgets.get(name), !clamping)) moved = true;
  }
  if (!moved) return;

  const computed = node.computeSize?.();
  if (computed) node.setSize([node.size[0], computed[1]]);
  node.graph?.setDirtyCanvas(true, true);
}

/**
 * Fold an affine node's pattern widget away while its options socket is carrying one.
 *
 * @param {object} node - The affine node to lay out.
 * @returns {void}
 */
function foldPattern(node) {
  const widget = (node.widgets ?? []).find((entry) => entry.name === "pattern");
  const socket = (node.inputs ?? []).find((entry) => entry.name === "affine_options");
  if (!setWidgetHidden(widget, Boolean(socket?.link != null))) return;

  const computed = node.computeSize?.();
  if (computed) node.setSize([node.size[0], computed[1]]);
  node.graph?.setDirtyCanvas(true, true);
}

/**
 * Call a function whenever a node is built, reconfigured, rewired or has a widget changed.
 *
 * @param {object} node - The node to watch.
 * @param {string[]} triggers - Widget names whose callback should also fire it.
 * @param {Function} refresh - What to run, taking the node.
 * @returns {void}
 */
function follow(node, triggers, refresh) {
  const run = () => {
    try {
      refresh(node);
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to lay out ${node?.type}:`, error);
    }
  };

  for (const name of triggers) {
    const widget = (node.widgets ?? []).find((entry) => entry.name === name);
    if (!widget) continue;
    const original = widget.callback;
    widget.callback = function (...args) {
      const result = original?.apply(this, args);
      run();
      return result;
    };
  }

  const originalConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    const result = originalConfigure?.apply(this, args);
    run();
    return result;
  };

  const originalConnections = node.onConnectionsChange;
  node.onConnectionsChange = function (...args) {
    const result = originalConnections?.apply(this, args);
    run();
    return result;
  };

  run();
}

/**
 * Attach a report panel to a node and release it when the node goes.
 *
 * @param {object} node - The node the panel belongs to.
 * @param {object} options - Passed through to `createReportPanel`.
 * @returns {void}
 */
function attach(node, options) {
  try {
    const panel = createReportPanel(node, options);
    appendInterfaceWidget(node, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

    const originalOnRemoved = node.onRemoved;
    node.onRemoved = function (...args) {
      const removed = originalOnRemoved?.apply(this, args);
      try {
        panel.dispose();
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to release the affine report:`, error);
      }
      return removed;
    };
  } catch (error) {
    console.error(`[${EXT_NAME}] Failed to build the affine report:`, error);
  }
}

app.registerExtension({
  name: EXT_NAME,
  settings: [
    {
      id: SETTING_ID,
      category: ["WAS Node Suite", "Affine", "Show what the affine did"],
      name: "Draw the affine report",
      tooltip:
        "Draw the scale, the bias, the share of the latent the mask covered and the mask "
        + "itself on Latent Affine and the Affine samplers, and the curve on Affine "
        + "Schedule. A run where no step carried the affine is drawn in the warning colour. "
        + "The nodes run the same either way. Affine Options folds its unused settings away "
        + "whatever this is set to. This applies to nodes added after the setting changes, "
        + "so a reload shows it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],

  async beforeRegisterNodeDef(nodeType, nodeData) {
    const id = nodeData?.name;
    const isOptions = id === OPTIONS_NODE;
    const isSchedule = id === SCHEDULE_NODE;
    const isMasked = MASK_NODES.includes(id);
    const isPatch = id === PATCH_NODE;
    if (!isOptions && !isSchedule && !isMasked && !isPatch) return;

    const proto = nodeType.prototype;
    // Definitions are registered again on a refresh, which would otherwise append a second
    // panel to every node of this type.
    if (proto.__was_affine_wrapped) return;
    proto.__was_affine_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      if (isOptions) {
        follow(this, ["pattern", "content_gate", "mask_sharpen", "clamp"], foldOptions);
        return result;
      }
      if (isMasked || isPatch) follow(this, [], foldPattern);
      if (!enabled()) return result;
      attach(this, {
        className: isSchedule ? "was-affine-schedule" : "was-affine-report",
        height: isPatch ? PATCH_HEIGHT : (isSchedule ? SCHEDULE_HEIGHT : REPORT_HEIGHT),
        minWidth: REPORT_MIN_WIDTH,
        labelWidth: LABEL_WIDTH,
        emptyLabel: isSchedule ? "run the node to see the curve" : "run the node to see the affine",
        logName: LOG_NAME,
        failure: "Failed to read the affine report:",
        sketch: isPatch ? undefined : () => createPictureBand(this, {
          label: isSchedule ? "the curve" : "the mask",
        }),
      });
      return result;
    };
  },
});
