/**
 * What Image Waveform measured, drawn on the node itself.
 *
 * The parade at the node's width, the share of a frame on the black and white rails, and five
 * figures per channel. A tab strip pages a batch through `node.imageIndex`.
 */

import { app } from "../../scripts/app.js";
import { ICON, ICON_SIZE, drawFidelityGlyph, iconTitle } from "./interface/icons.js";
import { drawStandIn, loadPlaceholder, standInDetail } from "./interface/placeholder.js";
import { PREVIEW_STATE, fetchOutputPreview } from "./interface/preview.js";
import { createFrameTabs, createReportPanel } from "./interface/report_panel.js";
import { surfaceRatio, watchSurfaceRatio } from "./interface/resolution.js";
import { onThemeChange, readTheme, themeVar } from "./interface/theme.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.ChannelWaveformUI";
const LOG_NAME = "WASNodeSuite.ChannelWaveform";
const SETTING_ID = "WAS.Analyze.ShowWaveformStats";
const PARADE_SETTING_ID = "WAS.Analyze.ShowWaveformParade";
const NODE_ID = "WASNSChannelWaveform";

const UI_WIDGET_NAME = "was_channel_waveform_ui";
const UI_WIDGET_TYPE = "was_channel_waveform";

// The slot the scope drawn for this panel is published under.
// `nodes/extras/image/channel_waveform.py` names the same.
const PARADE_SLOT = "parade_scope";

// Height in node units the panel opens at with neither band: the summary, the count tiles and
// the shared tab strip at its own 22 pixels.
const BASE_HEIGHT = 92;

// What the figures add to that: the head line and the four rows of the table.
const FIGURES_HEIGHT = 82;

// What the parade adds to it, which is the scope drawn right across the panel at its
// narrowest.
const PARADE_HEIGHT = 118;

// The narrowest the panel is worth drawing in, in node units: a 47 character row in an 11 pixel
// monospace font, plus the panel's own padding. Without it the node refits to its sockets and
// leaves the figures wrapping mid-column.
const PANEL_MIN_WIDTH = 340;

// The gap between the bands of the picker, in CSS pixels.
const BAND_GAP = 3;

// What the figures keep, in CSS pixels, before the parade may take any more room.
const FIGURES_MIN = 64;

// The shortest the parade stage is drawn, in CSS pixels.
const PICTURE_MIN = 34;

// The shape the stage holds before a picture says what the scope's own is: three plots side
// by side under one grid, which `modules/image/waveform.py` composes at a fixed size.
const PARADE_ASPECT = 2.875;

// The name the node publishes its column header under.
const COLUMNS_FACT = "columns";

// The name it publishes how the scope was resampled under.
const SCOPE_FACT = "scope";

// What the glyph claims before a run and after the panel is cleared.
const NO_FRAMES = Object.freeze({
  icon: ICON.WARNING,
  detail: "no frames have been measured yet",
});

// Animation frames `watchShownFrame` reads `node.imageIndex` over after a gesture. The frontend
// writes it during its own draw pass rather than from the pointer handler, so it is not there yet
// when the gesture ends and it never arrives at all for a gesture on something else.
const WATCH_FRAMES = 3;

/**
 * Whether one of the node's two readouts is drawn.
 *
 * @param {string} id - The setting to read.
 * @returns {boolean} True while the setting is on or cannot be read.
 */
function enabled(id) {
  try {
    const value = app?.extensionManager?.setting?.get?.(id);
    if (typeof value === "boolean") return value;
    const legacy = app?.ui?.settings?.getSettingValue?.(id);
    return typeof legacy === "boolean" ? legacy : true;
  } catch (error) {
    console.error(`[${EXT_NAME}] Failed to read ${id}:`, error);
    return true;
  }
}

/**
 * Report which frame of its own preview the frontend is showing.
 *
 * @param {object} node - The node whose preview is watched.
 * @param {(index: number) => void} onPick - Called with the frame the frontend settled on.
 * @returns {() => void} Release, for the panel's own teardown.
 */
function watchShownFrame(node, onPick) {
  let handle = 0;
  let released = false;

  const read = (left) => {
    if (released) return;
    const index = node?.imageIndex;
    if (Number.isFinite(index)) {
      onPick(Math.trunc(index));
      return;
    }
    if (left <= 0) return;
    handle = window.requestAnimationFrame(() => read(left - 1));
  };
  const start = () => {
    if (released) return;
    window.cancelAnimationFrame(handle);
    read(WATCH_FRAMES);
  };

  document.addEventListener("pointerup", start, true);
  document.addEventListener("keyup", start, true);
  return () => {
    released = true;
    window.cancelAnimationFrame(handle);
    document.removeEventListener("pointerup", start, true);
    document.removeEventListener("keyup", start, true);
  };
}

/**
 * Build the frame picker a report panel draws between its counts and its rows.
 *
 * @param {object} node - The node the picker belongs to, for its preview index and its redraws.
 * @param {object} [bands] - Which bands the picker carries.
 * @param {boolean} [bands.parade] - Draw the parade picture. On by default.
 * @param {boolean} [bands.figures] - Draw the head line and the table of figures. On by default.
 * @returns {{element: HTMLElement, update: (report: object) => void, clear: () => void,
 *   dispose: () => void}} The band, in the shape `createReportPanel` takes as its sketch.
 */
export function createFrameStats(node, bands = {}) {
  const withParade = bands.parade !== false;
  const withFigures = bands.figures !== false;

  const element = document.createElement("div");
  element.style.cssText = "flex:1 1 auto;min-height:0;display:flex;flex-direction:column;"
    + `gap:${BAND_GAP}px`;

  const tabs = createFrameTabs(pick);
  element.appendChild(tabs.element);

  // Sized to the picture rather than stretched to the band, so the figures below keep whatever
  // the parade's own shape does not need.
  const stage = document.createElement("div");
  stage.style.cssText = "flex:0 0 auto;align-self:center;position:relative;overflow:hidden;"
    + "border-radius:2px";
  const picture = document.createElement("canvas");
  picture.style.cssText = "position:absolute;inset:0;width:100%;height:100%;display:block";
  stage.appendChild(picture);
  if (withParade) element.appendChild(stage);

  const head = document.createElement("div");
  head.style.cssText = "display:flex;align-items:center;gap:6px;flex:0 0 auto;font-size:9px";
  const where = document.createElement("span");
  where.style.cssText = `flex:0 0 auto;color:${themeVar("fgMuted")}`;
  const rails = document.createElement("span");
  rails.style.cssText = "flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;"
    + `white-space:nowrap;color:${themeVar("fgMuted")}`;
  const glyph = document.createElement("canvas");
  glyph.style.cssText = `width:${ICON_SIZE}px;height:${ICON_SIZE}px;flex:0 0 auto`;
  head.append(where, rails, glyph);

  // The figures are written in fields of a fixed width and the header is written over the same
  // fields, so only a preserved run of spaces lines a column up under its name.
  const table = document.createElement("div");
  table.style.cssText = "flex:1 1 auto;min-height:0;overflow:auto;white-space:pre";
  if (withFigures) element.append(head, table);

  let disposed = false;
  // What the last report carried: the rows it holds, how many frames the batch had, and the
  // header they are written under.
  let items = [];
  let measured = 0;
  let total = 0;
  let columns = "";
  let scope = "";
  let picked = 0;
  let claim = NO_FRAMES;
  // The parade on the stage: the decoded picture, the state it came back with, how many frames
  // the slot holds and the size the node published it at.
  let parade = null;
  let paradeState = PREVIEW_STATE.WAITING;
  let paradeFrames = 0;
  let paradeSize = { width: 0, height: 0 };
  // The run the picture on the stage was fetched for.
  let fetched = null;

  /** What the figures on show are worth as a measurement of the frame above them. */
  const claimFor = () => {
    if (measured === 0) return NO_FRAMES;
    if (picked >= measured) {
      return {
        icon: ICON.WARNING,
        detail: `one report carries ${measured} of the ${total} frames, so this one has none`,
      };
    }
    return {
      icon: ICON.EXACT,
      detail: "the node's own statistics for this frame, written to four decimal places",
    };
  };

  /** What the picture on the stage is worth as a view of the parade the node plotted. */
  const paradeClaim = () => {
    if (!parade) return { icon: ICON.WARNING, detail: standInDetail(paradeState) };
    if (paradeFrames > 0 && picked >= paradeFrames) {
      return {
        icon: ICON.WARNING,
        detail: `the last of the ${paradeFrames} scopes published, not this frame's`,
      };
    }
    return {
      icon: ICON.APPROXIMATE,
      detail: scope
        ? `the three plots at ${scope}, drawn for the panel`
        : "the three plots resampled for the panel",
    };
  };

  /** Draw the fidelity glyph the picked frame earns. */
  const drawGlyph = () => {
    if (withFigures) drawFidelityGlyph(glyph, claim);
  };

  /** Draw the parade, or the pack's stand-in where the run has not published one. */
  const drawParade = () => {
    const ctx = picture.getContext("2d");
    if (!ctx) return;
    const width = picture.width;
    const height = picture.height;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = readTheme().bgDark;
    ctx.fillRect(0, 0, width, height);
    if (parade) {
      ctx.drawImage(parade, 0, 0, width, height);
      picture.style.opacity = "1";
    } else {
      drawStandIn(ctx, { x: 0, y: 0, w: width, h: height });
      picture.style.opacity = "0.35";
    }
    const shown = paradeClaim();
    picture.title = iconTitle(shown.icon, shown.detail);
  };

  /** Fit the stage to the parade's own shape, and rebuild its backing store at that size. */
  const sizeStage = () => {
    if (!withParade || disposed) return;
    // Layout pixels, not a bounding box. The graph draws the whole panel through a CSS
    // transform, so a measured rectangle is already multiplied by the zoom and writing one back
    // as a CSS size would scale it a second time.
    const room = element.clientWidth;
    if (!(room > 0)) return;
    const aspect = paradeSize.width > 0 && paradeSize.height > 0
      ? paradeSize.width / paradeSize.height
      : PARADE_ASPECT;
    const taken = tabs.element.offsetHeight
      + (withFigures ? head.offsetHeight + FIGURES_MIN : 0)
      + BAND_GAP * 3;
    const ceiling = Math.max(PICTURE_MIN, element.clientHeight - taken);
    const width = Math.max(1, Math.min(room, ceiling * aspect));
    const height = Math.max(1, width / aspect);
    stage.style.width = `${Math.round(width)}px`;
    stage.style.height = `${Math.round(height)}px`;

    // The ratio counts the graph's zoom as well as the display's density, so the picture is
    // rebuilt at the resolution it is really drawn at rather than magnified as a bitmap.
    const ratio = surfaceRatio(picture);
    const wide = Math.max(1, Math.round(width * ratio));
    const tall = Math.max(1, Math.round(height * ratio));
    if (picture.width !== wide || picture.height !== tall) {
      picture.width = wide;
      picture.height = tall;
    }
    drawParade();
  };

  /** Draw the frame the figures belong to, and the rails that frame sits on. */
  const drawHead = () => {
    if (total === 0) where.textContent = "";
    else where.textContent = total === 1 ? "one frame" : `frame ${picked + 1} of ${total}`;
    rails.textContent = picked < measured ? (items[picked]?.note ?? "") : "";
  };

  /** Draw the picked frame's three channel rows under the header the node published. */
  const drawTable = () => {
    table.textContent = "";
    const line = (text, colour) => {
      const row = document.createElement("div");
      row.style.color = colour;
      row.textContent = text;
      table.appendChild(row);
    };
    if (measured === 0) return;
    if (picked >= measured) {
      line("not measured", themeVar("fgMuted"));
      return;
    }
    if (columns) line(columns, themeVar("fgMuted"));
    for (const text of (items[picked]?.text ?? "").split("\n")) line(text, themeVar("fg"));
  };

  /** Redraw every part of the picker from the state the last report left. */
  const drawAll = () => {
    if (disposed) return;
    claim = claimFor();
    tabs.draw(total, picked, measured);
    sizeStage();
    drawHead();
    drawTable();
    drawGlyph();
    node.setDirtyCanvas?.(true, false);
  };

  let fetching = false;
  let again = false;

  /** Read the parade the node published for the frame on show, one ask at a time. */
  const loadParade = async () => {
    if (!withParade || disposed) return;
    // Asked again while a read is in flight, the picture already on its way is an older frame's,
    // so the ask is remembered and served after it rather than dropped.
    if (fetching) {
      again = true;
      return;
    }
    fetching = true;
    try {
      do {
        again = false;
        const index = paradeFrames > 0 ? Math.min(picked, paradeFrames - 1) : picked;
        let answer = await fetchOutputPreview(node, PARADE_SLOT, index);
        if (!answer?.image && index > 0) {
          // The store holds fewer scopes than the report counted frames, and a frame past the
          // end of it answers 404 with no count in it, so the head is what the count comes from.
          const head = await fetchOutputPreview(node, PARADE_SLOT, 0);
          const held = Number(head?.frameCount) || 0;
          answer = held > 1 ? await fetchOutputPreview(node, PARADE_SLOT, held - 1) : head;
        }
        if (disposed) return;
        parade = answer?.image ?? null;
        paradeState = answer?.state ?? PREVIEW_STATE.WAITING;
        paradeFrames = Number(answer?.frameCount) || 0;
        paradeSize = {
          width: Number(answer?.sourceWidth) || parade?.naturalWidth || 0,
          height: Number(answer?.sourceHeight) || parade?.naturalHeight || 0,
        };
        sizeStage();
        node.setDirtyCanvas?.(true, false);
      } while (again && !disposed);
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to read the parade the node plotted:`, error);
    } finally {
      fetching = false;
    }
  };

  /**
   * Show one frame's figures.
   *
   * @param {number} index - The frame, counting from 0, clamped into the batch.
   * @returns {void}
   */
  const show = (index) => {
    const wanted = total > 0 ? Math.max(0, Math.min(Math.trunc(index), total - 1)) : 0;
    if (wanted === picked) return;
    picked = wanted;
    drawAll();
    loadParade();
  };

  /**
   * Show one frame and put the same frame under it on the node.
   *
   * @param {number} index - The frame, counting from 0.
   * @returns {void}
   */
  function pick(index) {
    show(index);
    // The picture is the frontend's own preview, so the only way the two agree is to write the
    // index it draws from. A batch that produced fewer pictures than figures holds its last.
    const held = Array.isArray(node.imgs) ? node.imgs.length : 0;
    if (held > 0) {
      node.imageIndex = Math.min(picked, held - 1);
      node.setDirtyCanvas?.(true, true);
    }
  }

  /**
   * Read one report and draw the frame it left showing.
   *
   * @param {object|null} report - The report `createReportPanel` is drawing.
   * @returns {void}
   */
  const update = (report) => {
    if (disposed) return;
    items = Array.isArray(report?.items) ? report.items : [];
    measured = items.length;
    const stated = Number(report?.items_total);
    total = Math.max(measured, Number.isFinite(stated) ? Math.max(0, Math.trunc(stated)) : 0);
    const facts = report?.facts ?? [];
    columns = facts.find((entry) => entry.name === COLUMNS_FACT)?.value ?? "";
    scope = facts.find((entry) => entry.name === SCOPE_FACT)?.value ?? "";
    // A batch that shrank between runs leaves the picked frame past the end of the new one.
    picked = total > 0 ? Math.min(picked, total - 1) : 0;
    drawAll();
    // The same report is drawn twice, once when the node finishes and once when the run ends,
    // and both describe the one set of pictures. `run` counts every publish the process has
    // made, so two draws carrying the same number are two draws of one run.
    const run = Number(report?.run);
    if (Number.isFinite(run) && run === fetched) return;
    fetched = Number.isFinite(run) ? run : null;
    paradeFrames = 0;
    loadParade();
  };

  const onWheel = (event) => {
    // A table taller than its room scrolls itself, so the gesture is kept here and the panel
    // around it never sees it.
    if (table.scrollHeight > table.clientHeight) event.stopPropagation();
  };
  table.addEventListener("wheel", onWheel, { passive: true });

  const stopFollowing = watchShownFrame(node, show);
  const stopWatching = withFigures ? watchSurfaceRatio(glyph, () => drawGlyph()) : null;
  // `watchSurfaceRatio` reports the graph's zoom and nothing else, so the node being dragged
  // wider needs an observer of its own.
  const stopScaling = withParade ? watchSurfaceRatio(picture, () => sizeStage()) : null;
  const observer = withParade && typeof ResizeObserver === "function"
    ? new ResizeObserver(() => sizeStage())
    : null;
  observer?.observe(element);
  // The parade and the glyph are canvases, which take literal colours, so a palette change
  // draws them again.
  const stopTheme = onThemeChange(() => {
    drawGlyph();
    drawParade();
  });

  drawAll();
  if (withParade) {
    // The stand-in is one decoded picture for the page, so the stage is filled again once it
    // lands rather than each panel asking for its own.
    loadPlaceholder().then(() => {
      if (!disposed && !parade) drawParade();
    });
    loadParade();
  }

  return {
    element,
    update,
    clear() {
      items = [];
      measured = 0;
      total = 0;
      columns = "";
      scope = "";
      picked = 0;
      parade = null;
      paradeState = PREVIEW_STATE.WAITING;
      paradeFrames = 0;
      paradeSize = { width: 0, height: 0 };
      fetched = null;
      drawAll();
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      table.removeEventListener("wheel", onWheel);
      observer?.disconnect();
      stopFollowing();
      stopWatching?.();
      stopScaling?.();
      stopTheme();
    },
  };
}

app.registerExtension({
  name: EXT_NAME,
  settings: [
    {
      id: SETTING_ID,
      category: ["WAS Node Suite", "Analyze", "Show the waveform statistics"],
      name: "Draw the channel statistics on Image Waveform",
      tooltip:
        "Draw the minimum, maximum, mean, deviation and median of each colour channel on the "
        + "node, with how much of each sits on the black and white rails, and a tab per frame "
        + "for a batch. The parade picture has a switch of its own beside this one, and the "
        + "four image outputs are unchanged either way. This applies to nodes added after the "
        + "setting changes, so a reload shows it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
    {
      id: PARADE_SETTING_ID,
      category: ["WAS Node Suite", "Analyze", "Show the waveform parade"],
      name: "Draw the RGB parade on Image Waveform",
      tooltip:
        "Draw the parade on the node, composed for the node rather than shrunk from the "
        + "rgb_parade output: that output is three plots as wide as the picture, which is "
        + "about fifty pixels tall by the time it fits across a node. The one on the node "
        + "holds each channel to 240 columns and 256 levels, and is redrawn as the node is "
        + "resized and as the graph is zoomed. Drag the node wider or taller for a larger "
        + "plot, and hover it for what it was resampled from. The four image outputs are "
        + "unchanged either way, and the picture costs one PNG encode per run, only while "
        + "the panel is open. This applies to nodes added after the setting changes, so a "
        + "reload shows it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
  ],

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;

    const proto = nodeType.prototype;
    // Definitions are registered again on a refresh, which would otherwise append a second
    // panel to every node of this type.
    if (proto.__was_channel_waveform_wrapped) return;
    proto.__was_channel_waveform_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      const figures = enabled(SETTING_ID);
      const parade = enabled(PARADE_SETTING_ID);
      if (!figures && !parade) return result;
      try {
        const node = this;
        const panel = createReportPanel(node, {
          className: "was-channel-waveform",
          // The picker reads the node's facts itself: the column header goes over the figures
          // it heads, and the scope's resampling into what the picture says on hover.
          facts: false,
          height: BASE_HEIGHT + (figures ? FIGURES_HEIGHT : 0) + (parade ? PARADE_HEIGHT : 0),
          minWidth: PANEL_MIN_WIDTH,
          emptyLabel: "No frames measured yet",
          sketch: () => createFrameStats(node, { parade, figures }),
          logName: LOG_NAME,
          failure: "Failed to read the waveform report:",
        });
        appendInterfaceWidget(node, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the waveform readout:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the waveform readout:`, error);
      }
      return result;
    };
  },
});
