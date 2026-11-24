/**
 * The amount Video Look Transition mixes by, drawn as a curve over the clip's frames, with its
 * keyframes dragged on the curve. The keyframes are kept in the `keyframes` widget as text.
 */

import { app } from "../../scripts/app.js";
import { findWidget, writeWidget } from "./interface/audio_view.mjs";
import { CARD_LOOK, cardChip, cardRule } from "./interface/card.mjs";
import { ICON, ICON_SIZE, drawIcon, hoverTitles, iconTitle } from "./interface/icons.mjs";
import { DEFAULT_EASING, amountAt, placed, readEntry } from "./interface/keyframes.mjs";
import { CURVE_NAMES } from "./interface/number_sequence.mjs";
import { drawStandIn, loadPlaceholder, standInTitle } from "./interface/placeholder.mjs";
import { captureWheel, elementPoint } from "./interface/pointer.mjs";
import { openPopupMenu } from "./interface/popup_menu.mjs";
import { PREVIEW_STATE, fetchOutputPreview, watchPreviews } from "./interface/preview.mjs";
import { surfaceRatio, watchSurfaceRatio } from "./interface/resolution.mjs";
import { onNodeFinished, onRunEnded } from "./interface/run_events.mjs";
import { fetchRunResult, whenReadable } from "./interface/run_result.mjs";
import { onThemeChange, readTheme } from "./interface/theme.mjs";
import { appendInterfaceWidget, chainWidgetCallback, keepOnNode } from "./interface/widget.mjs";
import { SANS, el } from "./interface/window_kit.mjs";

const EXT_NAME = "WASNodeSuite.VideoLookTransitionUI";
const NODE_ID = "WASVideoLookTransition";

const UI_WIDGET_NAME = "was_look_transition_ui";
const UI_WIDGET_TYPE = "was_look_transition";

const KEYS_WIDGET = "keyframes";

// Node property holding the clip's frame count from the last run, so a reload draws its scale.
const FRAMES_KEY = "was_look_frames";

// Height and width in node units the panel opens at.
const PANEL_HEIGHT = 250;
const PANEL_WIDTH = 420;

// Frames drawn when no run has said how long the clip is.
const ASSUMED_FRAMES = 48;

// Most pictures the strip holds, as the node publishes it.
const STRIP_FRAMES = 16;

// Margins around the plot inside the canvas, in CSS pixels.
const LEFT = 22;
const RIGHT = 8;
const TOP = 8;
const BOTTOM = 16;

// Radius of a keyframe's handle, and how far from it a press still takes it.
const HANDLE = 5;
const REACH = 9;

// Smallest gap between frames, in CSS pixels, at which every frame's amount is marked.
const MARK_FRAMES_FROM = 4;

// The footer while no keyframe is chosen, and the plot's hover text.
const HINT = "drag a key; double-click adds";
const PLOT_HINT = "right-click a key: easing";

/**
 * A small rounded button in the card's colours.
 *
 * @param {string} label - Its words.
 * @param {(event: MouseEvent) => void} onPress - Called on a press.
 * @param {string} [hint] - Hover text.
 * @returns {HTMLButtonElement} The button.
 */
function pill(label, onPress, hint = "") {
  const press = document.createElement("button");
  press.type = "button";
  press.textContent = label;
  if (hint) press.title = hint;
  const rest = `flex:none;padding:1px 9px;border-radius:999px;cursor:pointer;white-space:nowrap;`
    + `font:600 11px/18px ${SANS};border:1px solid ${CARD_LOOK.edge};background:${CARD_LOOK.inset};`
    + `color:${CARD_LOOK.muted}`;
  press.style.cssText = rest;
  press.onmouseenter = () => {
    if (press.disabled) return;
    press.style.borderColor = CARD_LOOK.rule;
    press.style.color = CARD_LOOK.title;
  };
  press.onmouseleave = () => {
    press.style.borderColor = CARD_LOOK.edge;
    press.style.color = CARD_LOOK.muted;
  };
  press.addEventListener("pointerdown", (event) => event.stopPropagation());
  press.addEventListener("click", (event) => {
    event.preventDefault();
    event.stopPropagation();
    try {
      onPress(event);
    } catch (error) {
      console.error(`[${EXT_NAME}] ${label} failed:`, error);
    }
  });
  return press;
}

/**
 * An amount written as the widget keeps it.
 *
 * @param {number} amount - 0 to 1.
 * @returns {string} At most three decimals, such as `0.25` or `1`.
 */
function amountText(amount) {
  return String(Math.round(amount * 1000) / 1000);
}

/**
 * One keyframe written as an entry of the widget's text.
 *
 * @param {{frame: number, amount: number, easing: string}} key - The keyframe.
 * @returns {string} Such as `48:1` or `0:0:ease_in_out_sine`.
 */
function entryText(key) {
  const easing = key.easing && key.easing !== DEFAULT_EASING ? `:${key.easing}` : "";
  return `${key.frame}:${amountText(key.amount)}${easing}`;
}

/**
 * A keyframes text cut into its entries and the separators between them.
 *
 * @param {string} text - The widget's text.
 * @returns {object[]} Entries at even places as `{raw, key}`, `key` null where the entry is
 *   blank, a note or unreadable; separators at odd places as `{raw}`.
 */
function piecesOf(text) {
  return String(text ?? "").split(/([;\n\r])/).map((raw, index) => {
    if (index % 2) return { raw };
    const entry = raw.trim();
    return { raw, key: entry && !entry.startsWith("#") ? readEntry(entry) : null };
  });
}

/**
 * The frame a keyframe lands on in a clip.
 *
 * @param {{frame: number}} key - The keyframe.
 * @param {number} count - Frames in the clip.
 * @returns {number} The frame, a negative one counted from the end.
 */
function landing(key, count) {
  return key.frame < 0 ? key.frame + count : key.frame;
}

/**
 * Build the panel for one node.
 *
 * @param {object} node - The Video Look Transition node.
 * @returns {object} The panel `appendInterfaceWidget` hosts.
 */
function createLookPanel(node) {
  let disposed = false;
  let lastText = null;
  let pieces = [];
  let selected = -1;
  let drag = null;
  let hoverX = null;
  let frames = 0;
  let picture = null;
  let pictureState = PREVIEW_STATE.WAITING;
  let expectedPromptId = "";
  let queued = 0;
  let headerSignature = "";

  const root = el("div", `display:flex;flex-direction:column;gap:6px;width:100%;height:100%;`
    + `box-sizing:border-box;padding:8px;font:12px ${SANS};color:${CARD_LOOK.text};outline:none;min-height:0`);
  root.tabIndex = 0;
  const head = el("div", "display:flex;align-items:center;gap:6px;flex:0 0 auto;flex-wrap:wrap;min-width:0");
  const title = el("span", `flex:none;font:600 12.5px/20px ${SANS};color:${CARD_LOOK.title}`, "Look transition");
  const summary = el("span", "flex:0 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"
    + `color:${CARD_LOOK.muted}`);
  const chips = el("span", "display:flex;gap:4px;flex:none");
  const spacer = el("span", "flex:1 1 auto");
  const addButton = pill("Add key", () => addAt(widestGap()), "add in the widest gap");
  const removeButton = pill("Remove", () => removeSelected(), "remove the chosen key");
  head.append(title, summary, chips, spacer, addButton, removeButton);

  const stage = el("div", "position:relative;flex:1 1 auto;min-width:0;min-height:0;border-radius:7px;"
    + `background:${CARD_LOOK.inset};box-shadow:${CARD_LOOK.well};overflow:hidden`);
  const canvas = el("canvas", "position:absolute;inset:0;width:100%;height:100%;cursor:crosshair;touch-action:none");
  stage.append(canvas);

  const foot = el("div", "display:flex;align-items:center;gap:6px;flex:0 0 auto;flex-wrap:wrap;min-width:0;"
    + `color:${CARD_LOOK.muted};font:11px/18px ${SANS}`);
  root.append(head, cardRule(), stage, foot);

  const ctx = canvas.getContext("2d");
  const titles = hoverTitles(canvas);

  /**
   * The widget's text cut into entries, read again only when it changed.
   *
   * @returns {void}
   */
  function readModel() {
    const text = String(findWidget(node, KEYS_WIDGET)?.value ?? "");
    if (text === lastText) return;
    lastText = text;
    pieces = piecesOf(text);
    selected = -1;
    drag = null;
  }

  /**
   * The frame count a run reported, now or before a reload.
   *
   * @returns {number} Frames, or 0 where no run has said.
   */
  function knownCount() {
    if (frames > 0) return frames;
    const kept = Number(node.properties?.[FRAMES_KEY]);
    return Number.isInteger(kept) && kept > 0 ? kept : 0;
  }

  /**
   * The readable keyframes, each with the place of its entry in the text.
   *
   * @returns {Array<{index: number, key: object}>} In written order.
   */
  function entries() {
    const found = [];
    pieces.forEach((piece, index) => {
      if (index % 2 === 0 && piece.key) found.push({ index, key: piece.key });
    });
    return found;
  }

  /**
   * The clip's frame count: a run's, else assumed from the keyframes.
   *
   * @returns {number} Frames, at least 1.
   */
  function frameCount() {
    const known = knownCount();
    if (known) return known;
    const furthest = entries().reduce((most, { key }) => Math.max(most, Math.abs(key.frame) + 1), 0);
    return Math.max(ASSUMED_FRAMES, furthest);
  }

  /**
   * Where the plot sits inside the canvas, in CSS pixels.
   *
   * @returns {{x: number, y: number, w: number, h: number}} The plot.
   */
  function plot() {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    return { x: LEFT, y: TOP, w: Math.max(1, width - LEFT - RIGHT), h: Math.max(1, height - TOP - BOTTOM) };
  }

  const xOf = (box, count, frame) => box.x + ((frame + 0.5) / count) * box.w;
  const frameOf = (box, count, x) => ((x - box.x) / box.w) * count - 0.5;
  const yOf = (box, amount) => box.y + (1 - amount) * box.h;
  const amountOf = (box, y) => 1 - (y - box.y) / box.h;
  const wholeFrame = (box, count, x) => Math.min(count - 1, Math.max(0, Math.round(frameOf(box, count, x))));

  /**
   * The keyframe entry under a point, if any.
   *
   * @param {{x: number, y: number}} point - In CSS pixels.
   * @returns {number} The entry's place in the text, or -1.
   */
  function keyAt(point) {
    const box = plot();
    const count = frameCount();
    let best = -1;
    let bestDistance = REACH;
    for (const { index, key } of entries()) {
      const x = Math.min(box.x + box.w, Math.max(box.x, xOf(box, count, landing(key, count))));
      const distance = Math.hypot(point.x - x, point.y - yOf(box, key.amount));
      if (distance <= bestDistance) {
        best = index;
        bestDistance = distance;
      }
    }
    return best;
  }

  /**
   * The middle of the widest stretch of the clip no keyframe sits inside.
   *
   * @returns {number} A whole frame.
   */
  function widestGap() {
    readModel();
    const count = frameCount();
    const marks = [...new Set(entries().map(({ key }) => landing(key, count)))]
      .filter((frame) => frame >= 0 && frame <= count - 1).sort((a, b) => a - b);
    const edges = [0, ...marks, count - 1];
    let best = Math.floor((count - 1) / 2);
    let widest = -1;
    for (let index = 1; index < edges.length; index += 1) {
      const gap = edges[index] - edges[index - 1];
      if (gap > widest) {
        widest = gap;
        best = Math.round((edges[index] + edges[index - 1]) / 2);
      }
    }
    return best;
  }

  /**
   * Write the text the pieces now spell, as one undoable edit.
   *
   * @returns {void}
   */
  function commit() {
    const text = pieces.map((piece) => piece.raw).join("");
    if (text !== lastText) {
      lastText = text;
      writeWidget(node, KEYS_WIDGET, text);
    }
    schedule();
  }

  /**
   * Take one entry and a separator beside it out of the text.
   *
   * @param {number} index - The entry's place in the text.
   * @returns {void}
   */
  function removeEntry(index) {
    if (index + 1 < pieces.length) pieces.splice(index, 2);
    else if (index > 0) pieces.splice(index - 1, 2);
    else pieces = [{ raw: "", key: null }];
  }

  /**
   * Put a keyframe into one entry, keeping the spaces around it, and drop any other on its frame.
   *
   * @param {number} index - The entry's place in the text.
   * @param {object} key - The keyframe.
   * @returns {number} The entry's place once others on its frame are gone.
   */
  function setEntry(index, key) {
    const raw = pieces[index].raw;
    const lead = raw.match(/^\s*/)[0];
    const trail = raw.match(/\s*$/)[0];
    pieces[index] = { raw: `${lead}${entryText(key)}${trail}`, key };
    const count = frameCount();
    const landed = landing(key, count);
    let kept = index;
    for (const other of entries().reverse()) {
      if (other.index === kept || landing(other.key, count) !== landed) continue;
      removeEntry(other.index);
      if (other.index < kept) kept -= 2;
    }
    return kept;
  }

  /**
   * Add a keyframe at a frame, with the easing of the stretch it lands in.
   *
   * @param {number} frame - The frame.
   * @param {number} [amount] - Its amount; the curve's own at that frame when left out.
   * @returns {void}
   */
  function addAt(frame, amount) {
    readModel();
    const count = frameCount();
    const keys = entries().map((entry) => entry.key);
    const points = keys.length ? placed(keys, count) : [];
    const level = amount ?? (points.length ? amountAt(points, frame) : 0);
    const before = [...points].reverse().find((point) => point[0] <= frame);
    const key = { frame, amount: Math.round(level * 100) / 100, easing: before ? before[2] : DEFAULT_EASING };
    if (pieces.every((piece, index) => index % 2 || !piece.raw.trim())) {
      pieces = [{ raw: entryText(key), key }];
      selected = 0;
    } else {
      pieces.push({ raw: ";" }, { raw: ` ${entryText(key)}`, key });
      selected = setEntry(pieces.length - 1, key);
    }
    commit();
  }

  /**
   * Remove the chosen keyframe, keeping at least one.
   *
   * @returns {void}
   */
  function removeSelected() {
    readModel();
    if (selected < 0 || entries().length <= 1) return;
    removeEntry(selected);
    selected = -1;
    commit();
  }

  /**
   * Give one keyframe another easing.
   *
   * @param {number} index - The entry's place in the text.
   * @param {string} easing - A curve name.
   * @returns {void}
   */
  function setEasing(index, easing) {
    readModel();
    const key = pieces[index]?.key;
    if (!key) return;
    selected = setEntry(index, { ...key, easing });
    commit();
  }

  /**
   * End a drag, writing the dragged keyframe once where it moved.
   *
   * @returns {void}
   */
  function finishDrag() {
    if (!drag) return;
    const { index, moved } = drag;
    drag = null;
    const key = pieces[index]?.key;
    if (moved && key) {
      selected = setEntry(index, key);
      commit();
    } else {
      schedule();
    }
  }

  /**
   * Open the menu for a keyframe, or for the frame under the pointer.
   *
   * @param {MouseEvent} event - The press that opened it.
   * @param {number} index - The keyframe's entry, or -1.
   * @returns {void}
   */
  function openMenu(event, index) {
    const items = [];
    if (index >= 0 && pieces[index]?.key) {
      const key = pieces[index].key;
      items.push({ header: "Easing to the next keyframe" });
      for (const name of CURVE_NAMES) {
        items.push({ label: name, checked: name === key.easing, onSelect: () => setEasing(index, name) });
      }
      items.push({ separator: true });
      items.push({
        label: "Remove keyframe",
        danger: true,
        disabled: entries().length <= 1,
        onSelect: () => {
          selected = index;
          removeSelected();
        },
      });
    } else {
      if (hoverX === null) return;
      const frame = wholeFrame(plot(), frameCount(), hoverX);
      items.push({ label: `Add keyframe at frame ${frame}`, onSelect: () => addAt(frame) });
    }
    openPopupMenu({ items, x: event.clientX, y: event.clientY, logName: EXT_NAME });
  }

  /**
   * Draw the strip of frames the node published, each picture filling its share of the plot.
   *
   * @param {{x: number, y: number, w: number, h: number}} box - The plot.
   * @param {number} count - Frames in the clip.
   * @returns {void}
   */
  function drawStrip(box, count) {
    const tiles = Math.min(STRIP_FRAMES, count);
    const tileWidth = picture.naturalWidth / tiles;
    const slot = box.w / tiles;
    const slotAspect = slot / box.h;
    for (let tile = 0; tile < tiles; tile += 1) {
      let sx = tile * tileWidth;
      let sw = tileWidth;
      let sy = 0;
      let sh = picture.naturalHeight;
      if (tileWidth / sh > slotAspect) {
        sw = sh * slotAspect;
        sx += (tileWidth - sw) / 2;
      } else {
        sh = sw / slotAspect;
        sy = (picture.naturalHeight - sh) / 2;
      }
      ctx.drawImage(picture, sx, sy, sw, sh, box.x + tile * slot, box.y, slot + 0.5, box.h);
    }
  }

  /**
   * Trace the curve across the plot, one point per CSS pixel.
   *
   * @param {{x: number, y: number, w: number, h: number}} box - The plot.
   * @param {number} count - Frames in the clip.
   * @param {Array} points - What `placed` answered.
   * @returns {void}
   */
  function traceCurve(box, count, points) {
    ctx.beginPath();
    for (let x = 0; x <= box.w; x += 1) {
      const y = yOf(box, amountAt(points, frameOf(box, count, box.x + x)));
      if (x === 0) ctx.moveTo(box.x, y);
      else ctx.lineTo(box.x + x, y);
    }
  }

  /**
   * Paint the strip, the curve and the keyframes at the graph's resolution.
   *
   * @returns {void}
   */
  function paint() {
    queued = 0;
    if (disposed || !ctx) return;
    readModel();
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

    const box = plot();
    const count = frameCount();
    const keyRegions = [];
    const backRegions = [];
    if (picture) {
      drawStrip(box, count);
      ctx.fillStyle = "rgba(0,0,0,0.38)";
      ctx.fillRect(box.x, box.y, box.w, box.h);
    } else {
      ctx.save();
      ctx.globalAlpha = 0.45;
      drawStandIn(ctx, { x: box.x, y: box.y, w: box.w, h: box.h });
      ctx.restore();
      backRegions.push({ x: box.x, y: box.y, width: box.w, height: box.h, title: standInTitle(pictureState) });
    }
    backRegions.push({ x: box.x, y: box.y, width: box.w, height: box.h, title: PLOT_HINT });

    ctx.font = `600 10px ${SANS}`;
    ctx.textBaseline = "middle";
    ctx.textAlign = "center";
    for (const [amount, label] of [[0, "A"], [0.5, ""], [1, "B"]]) {
      const y = Math.round(yOf(box, amount)) + 0.5;
      ctx.strokeStyle = theme.fgMuted;
      ctx.globalAlpha = amount === 0.5 ? 0.35 : 0.6;
      ctx.setLineDash(amount === 0.5 ? [3, 4] : []);
      ctx.beginPath();
      ctx.moveTo(box.x, y);
      ctx.lineTo(box.x + box.w, y);
      ctx.stroke();
      ctx.globalAlpha = 1;
      if (label) {
        ctx.fillStyle = theme.fgMuted;
        ctx.fillText(label, LEFT / 2, y);
      }
    }
    ctx.setLineDash([]);
    backRegions.push({ x: 0, y: box.y - 6, width: LEFT, height: 12, title: "B: image_b, amount 1" });
    backRegions.push({ x: 0, y: box.y + box.h - 6, width: LEFT, height: 12, title: "A: image_a, amount 0" });

    const listed = entries();
    const points = listed.length ? placed(listed.map((entry) => entry.key), count) : [];
    if (points.length) {
      traceCurve(box, count, points);
      ctx.lineTo(box.x + box.w, yOf(box, 0));
      ctx.lineTo(box.x, yOf(box, 0));
      ctx.closePath();
      ctx.fillStyle = theme.accentBg;
      ctx.fill();
      traceCurve(box, count, points);
      ctx.strokeStyle = theme.accent;
      ctx.lineWidth = 2;
      ctx.stroke();
      ctx.lineWidth = 1;
      if (box.w / count >= MARK_FRAMES_FROM) {
        ctx.fillStyle = theme.fg;
        for (let frame = 0; frame < count; frame += 1) {
          ctx.beginPath();
          ctx.arc(xOf(box, count, frame), yOf(box, amountAt(points, frame)), 1.4, 0, Math.PI * 2);
          ctx.fill();
        }
      }
      for (const { index, key } of listed) {
        const frame = landing(key, count);
        const outside = frame < 0 || frame > count - 1;
        const x = Math.min(box.x + box.w, Math.max(box.x, xOf(box, count, frame)));
        const y = yOf(box, key.amount);
        ctx.beginPath();
        ctx.arc(x, y, HANDLE, 0, Math.PI * 2);
        ctx.fillStyle = index === selected ? theme.accent : theme.bgDark;
        ctx.fill();
        ctx.strokeStyle = index === selected ? theme.fg : theme.accent;
        ctx.lineWidth = 2;
        ctx.setLineDash(outside ? [2, 2] : []);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.lineWidth = 1;
        keyRegions.push({
          x: x - HANDLE, y: y - HANDLE, width: HANDLE * 2, height: HANDLE * 2,
          title: `frame ${frame} · ${amountText(key.amount)} · ${key.easing}`
            + (outside ? " · off the clip" : ""),
        });
      }
    }

    ctx.fillStyle = theme.fgMuted;
    ctx.font = `10px ${SANS}`;
    ctx.textAlign = "left";
    const base = box.y + box.h + BOTTOM / 2 + 1;
    ctx.fillText("0", box.x, base);
    ctx.textAlign = "right";
    ctx.fillText(String(count - 1), box.x + box.w, base);
    const dragged = drag ? pieces[drag.index]?.key : null;
    const marked = dragged ? landing(dragged, count) : (hoverX === null ? null : wholeFrame(box, count, hoverX));
    if (marked !== null && Number.isFinite(marked)) {
      const x = Math.round(xOf(box, count, marked)) + 0.5;
      ctx.strokeStyle = theme.fgMuted;
      ctx.globalAlpha = 0.5;
      ctx.beginPath();
      ctx.moveTo(x, box.y);
      ctx.lineTo(x, box.y + box.h);
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.textAlign = "center";
      ctx.fillStyle = theme.fg;
      const level = points.length ? `  ${amountText(amountAt(points, marked))}` : "";
      ctx.fillText(`frame ${marked}${level}`, Math.min(box.x + box.w - 44, Math.max(box.x + 44, x)), base);
    }

    const glyph = drawIcon(ctx, ICON.EXACT, box.x + box.w - ICON_SIZE - 3, box.y + 3, ICON_SIZE, theme.fg);
    titles.set([
      { ...glyph, title: iconTitle(ICON.EXACT, "the node's own amounts") },
      ...keyRegions,
      ...backRegions,
    ]);
    drawHeader(count, listed);
  }

  /**
   * Fill the header's summary and chips and the footer, where what they say changed.
   *
   * @param {number} count - Frames in the clip.
   * @param {Array<{index: number, key: object}>} listed - The readable keyframes.
   * @returns {void}
   */
  function drawHeader(count, listed) {
    const unreadable = pieces.filter((piece, index) => {
      const entry = index % 2 === 0 && !piece.key ? piece.raw.trim() : "";
      return entry && !entry.startsWith("#");
    }).length;
    const key = selected >= 0 ? pieces[selected]?.key : null;
    const signature = JSON.stringify([count, listed.length, unreadable, knownCount(), selected, key]);
    if (signature === headerSignature) return;
    headerSignature = signature;

    summary.textContent = "";
    summary.append(el("span", `font-weight:650;color:${CARD_LOOK.title}`, String(listed.length)),
      ` keyframe${listed.length === 1 ? "" : "s"} over `,
      el("span", `font-weight:650;color:${CARD_LOOK.title}`, String(count)), " frames");
    chips.textContent = "";
    if (!knownCount()) {
      const chip = cardChip(`${count} assumed`, "warning");
      chip.title = "run once for the clip";
      chips.append(chip);
    }
    if (unreadable) {
      const chip = cardChip(`${unreadable} unreadable`, "error");
      chip.title = "fix these to run";
      chips.append(chip);
    }
    removeButton.disabled = !key || listed.length <= 1;
    removeButton.style.opacity = removeButton.disabled ? "0.45" : "1";

    foot.textContent = "";
    if (!key) {
      foot.append(el("span", "min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap", HINT));
      return;
    }
    foot.append(
      el("span", "", "frame"), el("span", `font-weight:650;color:${CARD_LOOK.title}`, String(landing(key, count))),
      el("span", "", "amount"), el("span", `font-weight:650;color:${CARD_LOOK.title}`, amountText(key.amount)),
      el("span", "", "easing"),
      pill(key.easing, (event) => openMenu(event, selected), "easing to the next key"),
    );
  }

  /**
   * Repaint on the next frame, once however many ask.
   *
   * @returns {void}
   */
  function schedule() {
    if (disposed || queued) return;
    queued = requestAnimationFrame(paint);
  }

  /**
   * Read the frame count and the strip the node last published.
   *
   * @returns {Promise<void>}
   */
  async function load() {
    if (disposed) return;
    try {
      const [report, strip] = await Promise.all([
        fetchRunResult(node),
        fetchOutputPreview(node, "", 0, expectedPromptId),
      ]);
      if (disposed) return;
      const counted = report?.result?.counts?.find?.((entry) => entry.name === "frames")?.value;
      if (Number.isInteger(counted) && counted > 0) {
        frames = counted;
        if (Number(node.properties?.[FRAMES_KEY]) !== counted) keepOnNode(node, FRAMES_KEY, counted);
      }
      pictureState = strip?.state ?? PREVIEW_STATE.WAITING;
      picture = pictureState === PREVIEW_STATE.READY && strip.image ? strip.image : null;
      schedule();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to read the last run:`, error);
    }
  }

  canvas.addEventListener("pointerdown", (event) => {
    if (event.button === 1) {
      app.canvas?.processMouseDown?.(event);
      return;
    }
    if (event.button !== 0) return;
    event.stopPropagation();
    root.focus({ preventScroll: true });
    readModel();
    const index = keyAt(elementPoint(canvas, event));
    selected = index;
    if (index >= 0) {
      drag = { index, moved: false };
      canvas.setPointerCapture?.(event.pointerId);
    }
    schedule();
  });
  canvas.addEventListener("pointermove", (event) => {
    const point = elementPoint(canvas, event);
    const box = plot();
    hoverX = point.x >= box.x && point.x <= box.x + box.w ? point.x : null;
    if (drag) {
      if (!(event.buttons & 1)) {
        finishDrag();
        return;
      }
      const count = frameCount();
      const frame = wholeFrame(box, count, point.x);
      const amount = Math.min(1, Math.max(0, Math.round(amountOf(box, point.y) * 100) / 100));
      const piece = pieces[drag.index];
      if (piece?.key && (landing(piece.key, count) !== frame || piece.key.amount !== amount)) {
        pieces[drag.index] = { raw: piece.raw, key: { ...piece.key, frame, amount } };
        drag.moved = true;
      }
    } else {
      canvas.style.cursor = keyAt(point) >= 0 ? "grab" : "crosshair";
    }
    schedule();
  });
  canvas.addEventListener("pointerleave", () => {
    if (drag) return;
    hoverX = null;
    schedule();
  });
  canvas.addEventListener("pointerup", finishDrag);
  canvas.addEventListener("pointercancel", finishDrag);
  canvas.addEventListener("lostpointercapture", finishDrag);
  canvas.addEventListener("dblclick", (event) => {
    event.preventDefault();
    event.stopPropagation();
    readModel();
    const point = elementPoint(canvas, event);
    if (keyAt(point) >= 0) return;
    const box = plot();
    addAt(wholeFrame(box, frameCount(), point.x), Math.min(1, Math.max(0, amountOf(box, point.y))));
  });
  canvas.addEventListener("contextmenu", (event) => {
    event.preventDefault();
    event.stopPropagation();
    readModel();
    const point = elementPoint(canvas, event);
    const box = plot();
    hoverX = point.x >= box.x && point.x <= box.x + box.w ? point.x : null;
    const index = keyAt(point);
    if (index >= 0) selected = index;
    openMenu(event, index);
    schedule();
  });
  root.addEventListener("contextmenu", (event) => event.preventDefault());
  root.addEventListener("keydown", (event) => {
    if (event.key !== "Delete" && event.key !== "Backspace") return;
    event.preventDefault();
    event.stopPropagation();
    removeSelected();
  });

  const releaseWheel = captureWheel(root);
  const releasePreviews = watchPreviews(node);
  const stopFinished = onNodeFinished(node, (info) => {
    expectedPromptId = info.promptId;
    if (!info.cached) load();
  });
  const stopEnded = onRunEnded(() => load());
  const stopSettling = whenReadable(node, () => load());
  const stopRatio = watchSurfaceRatio(canvas, () => schedule());
  const stopTheme = onThemeChange(() => schedule());
  const resize = typeof ResizeObserver === "function" ? new ResizeObserver(() => schedule()) : null;
  resize?.observe(stage);
  chainWidgetCallback(node, KEYS_WIDGET, () => schedule(), EXT_NAME);
  loadPlaceholder().then(() => schedule());

  return {
    element: root,
    height: PANEL_HEIGHT,
    minWidth: PANEL_WIDTH,
    refresh: load,
    dispose() {
      disposed = true;
      if (queued) cancelAnimationFrame(queued);
      for (const stop of [releaseWheel, releasePreviews, stopFinished, stopEnded, stopSettling, stopRatio, stopTheme]) {
        try {
          stop?.();
        } catch (error) {
          console.error(`[${EXT_NAME}] Failed to release the panel:`, error);
        }
      }
      resize?.disconnect();
      titles.dispose();
    },
  };
}

app.registerExtension({
  name: EXT_NAME,

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;
    const proto = nodeType.prototype;
    if (proto.__was_look_transition_wrapped) return;
    proto.__was_look_transition_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const panel = createLookPanel(this);
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
