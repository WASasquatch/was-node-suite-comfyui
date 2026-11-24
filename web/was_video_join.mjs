/**
 * What Video Join made, drawn on the node: the joined clip's figures, its clips laid out on two
 * tracks with every transition where the tracks overlap, and the frames either side of each join.
 */

import { app } from "../../scripts/app.js";
import { CARD_LOOK } from "./interface/card.mjs";
import { ICON, ICON_SIZE, drawIcon, hoverTitles, iconTitle } from "./interface/icons.mjs";
import { createPictureBand } from "./interface/picture_band.mjs";
import { createReportPanel } from "./interface/report_panel.mjs";
import { surfaceRatio, watchSurfaceRatio } from "./interface/resolution.mjs";
import { onThemeChange, readTheme } from "./interface/theme.mjs";
import { appendInterfaceWidget } from "./interface/widget.mjs";
import { SANS, el } from "./interface/window_kit.mjs";

const EXT_NAME = "WASNodeSuite.VideoJoinUI";
const NODE_ID = "WASVideoJoin";

const UI_WIDGET_NAME = "was_video_join_ui";
const UI_WIDGET_TYPE = "was_video_join";

// Height of the panel in node units: the summary, the figures, the timeline, the facts and the
// join pictures.
const PANEL_HEIGHT = 400;

// Height of the join pictures in CSS pixels when the panel opens.
const SHEET_HEIGHT = 150;

// Height of the timeline, one track, and the ruler under the tracks, in CSS pixels.
const TIMELINE_HEIGHT = 50;
const TRACK = 15;
const TRACK_GAP = 3;

// The widest figure name the report writes, which is what the fact column is opened at.
const LABEL_WIDTH = 72;

// The fact the timeline is read from, drawn by the timeline rather than as a row.
const TIMELINE_FACT = "timeline";

// The transition each letter of the timeline names.
const KINDS = { c: "cut", d: "dissolve", k: "dip to black", w: "dip to white", x: "wipe" };

const EMPTY_LABEL = "run once to see the join";

/**
 * The clips and joins a timeline line spells, laid out on the joined clip's frames.
 *
 * @param {string} text - Such as `48 d12 60 d12 30`, ending ` +` where it was cut short.
 * @returns {{clips: object[], joins: object[], total: number, cut: boolean}|null} Each clip's
 *   start and length, each join's start, length and transition, and the frames drawn, or null
 *   where the text spells nothing.
 */
function readTimeline(text) {
  const tokens = String(text ?? "").trim().split(/\s+/).filter(Boolean);
  const clips = [];
  const joins = [];
  let cut = false;
  let start = 0;
  let pending = null;
  for (const token of tokens) {
    if (token === "+") {
      cut = true;
      break;
    }
    if (/^[0-9]+$/.test(token)) {
      const length = Number(token);
      if (pending) {
        start -= pending.length;
        joins.push({ start, length: pending.length, kind: pending.kind });
        pending = null;
      }
      clips.push({ start, length });
      start += length;
      continue;
    }
    const match = /^([a-z])([0-9]+)$/.exec(token);
    if (!match) return null;
    pending = { kind: KINDS[match[1]] ?? "cut", length: Number(match[2]) };
  }
  if (!clips.length) return null;
  const last = clips[clips.length - 1];
  return { clips, joins, total: last.start + last.length, cut };
}

/**
 * The band drawn between the figures and the facts: the timeline over the join pictures.
 *
 * @param {object} node - The node the panel is on.
 * @param {{timeline: string|null}} held - Where the panel keeps the timeline line of the report
 *   it is drawing.
 * @returns {{element: HTMLElement, update: Function, clear: Function, dispose: Function,
 *   refresh: Function}} The band, in the shape a report panel's sketch takes.
 */
function createJoinBand(node, held) {
  const element = el("div", "display:flex;flex-direction:column;gap:6px;flex:3 1 auto;min-height:0;min-width:0");
  const track = el("div", `position:relative;flex:0 0 ${TIMELINE_HEIGHT}px;min-width:0;border-radius:6px;`
    + `background:${CARD_LOOK.inset};box-shadow:${CARD_LOOK.well};overflow:hidden`);
  const canvas = el("canvas", "position:absolute;inset:0;width:100%;height:100%;display:block");
  track.append(canvas);
  const sheet = createPictureBand(node, { label: "frames at each join", height: SHEET_HEIGHT });
  element.append(track, sheet.element);
  const ctx = canvas.getContext("2d");
  const titles = hoverTitles(canvas);
  let layout = null;
  let disposed = false;

  /**
   * Paint the timeline at the graph's resolution.
   *
   * @returns {void}
   */
  const paint = () => {
    if (disposed || !ctx) return;
    const theme = readTheme();
    const width = Math.max(1, canvas.clientWidth);
    const height = Math.max(1, canvas.clientHeight);
    const ratio = surfaceRatio(canvas);
    const wide = Math.max(1, Math.round(width * ratio));
    const tall = Math.max(1, Math.round(height * ratio));
    if (canvas.width !== wide) canvas.width = wide;
    if (canvas.height !== tall) canvas.height = tall;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    if (!layout) {
      ctx.fillStyle = theme.fgMuted;
      ctx.font = `11px ${SANS}`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(held.timeline === null ? "no timeline yet" : "timeline unreadable", width / 2, height / 2);
      titles.set([]);
      return;
    }
    const pad = 6;
    const span = Math.max(1, width - pad * 2);
    const xOf = (frame) => pad + (frame / layout.total) * span;
    const top = 5;
    const regions = [];
    ctx.font = `600 10px ${SANS}`;
    ctx.textBaseline = "middle";
    ctx.textAlign = "center";
    layout.clips.forEach((clip, index) => {
      const y = top + (index % 2) * (TRACK + TRACK_GAP);
      const x0 = xOf(clip.start);
      const x1 = xOf(clip.start + clip.length);
      ctx.fillStyle = index % 2 ? theme.borderLight : theme.accent;
      ctx.globalAlpha = index % 2 ? 0.95 : 0.55;
      ctx.beginPath();
      if (typeof ctx.roundRect === "function") ctx.roundRect(x0, y, Math.max(1, x1 - x0), TRACK, 3);
      else ctx.rect(x0, y, Math.max(1, x1 - x0), TRACK);
      ctx.fill();
      ctx.globalAlpha = 1;
      if (x1 - x0 > 14) {
        ctx.fillStyle = theme.fg;
        ctx.fillText(String(index + 1), (x0 + x1) / 2, y + TRACK / 2 + 0.5);
      }
      regions.push({
        x: x0, y, width: Math.max(1, x1 - x0), height: TRACK,
        title: `clip ${index + 1}: frames ${clip.start} to ${clip.start + clip.length - 1}, ${clip.length} frames`,
      });
    });
    const joinRegions = [];
    for (const join of layout.joins) {
      if (!join.length) continue;
      const x0 = xOf(join.start);
      const x1 = xOf(join.start + join.length);
      const y = top;
      const h = TRACK * 2 + TRACK_GAP;
      ctx.strokeStyle = theme.fg;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x0, y + h);
      ctx.lineTo(x1, y);
      if (join.kind !== "dissolve" && join.kind !== "wipe") {
        ctx.moveTo(x0, y);
        ctx.lineTo((x0 + x1) / 2, y + h / 2);
        ctx.lineTo(x1, y + h);
      }
      ctx.stroke();
      joinRegions.push({
        x: x0 - 2, y, width: Math.max(4, x1 - x0 + 4), height: h,
        title: `${join.kind}, ${join.length} frames: frames ${join.start} to ${join.start + join.length - 1}`,
      });
    }
    const rulerY = top + TRACK * 2 + TRACK_GAP + 8;
    ctx.font = `10px ${SANS}`;
    ctx.fillStyle = theme.fgMuted;
    ctx.textAlign = "left";
    ctx.fillText("0", pad, rulerY);
    ctx.textAlign = "right";
    ctx.fillText(`${layout.total}${layout.cut ? "+" : ""} frames`, width - pad, rulerY);
    const glyph = drawIcon(ctx, ICON.EXACT, width / 2 - ICON_SIZE / 2, rulerY - ICON_SIZE / 2, ICON_SIZE, theme.fgMuted);
    const detail = layout.cut
      ? "the node's own figures, cut short"
      : "the node's own figures";
    titles.set([{ ...glyph, title: iconTitle(ICON.EXACT, detail) }, ...joinRegions, ...regions]);
  };

  const stopRatio = watchSurfaceRatio(canvas, () => paint());
  const stopTheme = onThemeChange(() => paint());
  const resize = typeof ResizeObserver === "function" ? new ResizeObserver(() => paint()) : null;
  resize?.observe(track);

  return {
    element,
    update(report) {
      layout = held.timeline ? readTimeline(held.timeline) : null;
      paint();
      sheet.update(report);
    },
    clear() {
      layout = null;
      paint();
      sheet.clear();
    },
    refresh() {
      sheet.refresh?.();
    },
    dispose() {
      disposed = true;
      for (const stop of [stopRatio, stopTheme]) {
        try {
          stop();
        } catch (error) {
          console.error(`[${EXT_NAME}] Failed to release the timeline:`, error);
        }
      }
      resize?.disconnect();
      titles.dispose();
      sheet.dispose();
    },
  };
}

app.registerExtension({
  name: EXT_NAME,

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;
    const proto = nodeType.prototype;
    if (proto.__was_video_join_wrapped) return;
    proto.__was_video_join_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const held = { timeline: null };
        const panel = createReportPanel(this, {
          className: "was-video-join",
          height: PANEL_HEIGHT,
          labelWidth: LABEL_WIDTH,
          emptyLabel: EMPTY_LABEL,
          logName: EXT_NAME,
          failure: "Failed to read the join:",
          // The timeline line is read here and drawn by the band rather than as a fact row.
          onAnswer: (report) => {
            if (!report) {
              held.timeline = null;
              return;
            }
            const facts = Array.isArray(report.facts) ? report.facts : [];
            const line = facts.find((fact) => fact?.name === TIMELINE_FACT);
            if (line) {
              held.timeline = String(line.value ?? "");
              report.facts = facts.filter((fact) => fact !== line);
            }
          },
          sketch: () => createJoinBand(this, held),
        });
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });
        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the panel:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the panel:`, error);
      }
      return result;
    };
  },
});
