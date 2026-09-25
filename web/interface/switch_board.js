/**
 * The parts of a panel that lists switches over the graph: the page timer that re-reads the
 * graph, the chip, and the switch drawn at the end of a row.
 */

import { themeVar } from "./theme.js";

// How often an open panel looks at the graph, in milliseconds.
export const REFRESH_MS = 250;

// The switch drawn at the end of a row, in CSS pixels.
export const PIP_WIDTH = 22;
export const PIP_HEIGHT = 11;

// Every open panel, as `{node, refresh, logName}`, and the one timer that calls them.
const ticking = new Set();
let tickHandle = 0;

/**
 * Stop the timer once no panel is left to call.
 *
 * @returns {void}
 */
function stopTicking() {
  if (ticking.size || !tickHandle) return;
  clearInterval(tickHandle);
  tickHandle = 0;
}

/**
 * Call every open panel's refresh once, and drop the ones whose node has left the graph.
 *
 * @returns {void}
 */
function tick() {
  // A hidden tab draws nothing, so there is nothing for a refresh to correct until it is shown.
  if (document.hidden) return;
  for (const entry of [...ticking]) {
    // A graph cleared rather than emptied node by node takes its nodes away without telling
    // each one, which would otherwise leave an entry reading a graph nobody can see.
    if (!entry.node?.graph) {
      ticking.delete(entry);
      continue;
    }
    try {
      entry.refresh();
    } catch (error) {
      console.error(`[${entry.logName}] Failed to read the graph:`, error);
    }
  }
  stopTicking();
}

/**
 * Put one panel in the page timer, starting the timer when it was stopped.
 *
 * @param {object} node - The node the panel is drawn on.
 * @param {() => void} refresh - What to call on every pass.
 * @param {string} logName - The prefix a failed pass is logged under.
 * @returns {() => void} Release, which does nothing the second time it is called.
 */
export function joinTicking(node, refresh, logName) {
  const entry = { node, refresh, logName };
  ticking.add(entry);
  if (!tickHandle) tickHandle = setInterval(tick, REFRESH_MS);
  let live = true;
  return () => {
    if (!live) return;
    live = false;
    ticking.delete(entry);
    stopTicking();
  };
}

/**
 * A small pressable label, drawn the same wherever a panel uses one.
 *
 * @param {string} hint - What the hover says, five words at most.
 * @param {() => void} onPress - What a left click or a keyboard press does.
 * @returns {HTMLButtonElement} The chip, for the caller to append and to label.
 */
export function createChip(hint, onPress) {
  const chip = document.createElement("button");
  chip.type = "button";
  chip.title = hint;
  chip.style.cssText = [
    "flex:0 0 auto",
    "padding:1px 6px",
    "border-radius:3px",
    "font:inherit",
    "line-height:15px",
    "cursor:pointer",
    "white-space:nowrap",
    `background:${themeVar("bgLight")}`,
    `border:1px solid ${themeVar("border")}`,
    `color:${themeVar("fg")}`,
  ].join(";");
  chip.addEventListener("click", (event) => {
    event.stopPropagation();
    onPress();
  });
  return chip;
}

/**
 * The on and off switch at the end of a row.
 *
 * @param {boolean} on - Which way it is drawn.
 * @returns {HTMLSpanElement} The switch, for the caller to append.
 */
export function createPip(on) {
  const pip = document.createElement("span");
  pip.style.cssText = [
    "flex:0 0 auto",
    "box-sizing:border-box",
    `width:${PIP_WIDTH}px`,
    `height:${PIP_HEIGHT}px`,
    `border-radius:${PIP_HEIGHT}px`,
    "position:relative",
    `background:${themeVar(on ? "accent" : "inputBg")}`,
    `border:1px solid ${themeVar(on ? "accent" : "inputBorder")}`,
  ].join(";");
  const knob = document.createElement("span");
  const travel = PIP_WIDTH - PIP_HEIGHT;
  knob.style.cssText = [
    "position:absolute",
    "top:0",
    `left:${on ? travel - 1 : 0}px`,
    `width:${PIP_HEIGHT - 2}px`,
    `height:${PIP_HEIGHT - 2}px`,
    "border-radius:50%",
    `background:${themeVar(on ? "selectionText" : "fgMuted")}`,
  ].join(";");
  pip.appendChild(knob);
  return pip;
}
