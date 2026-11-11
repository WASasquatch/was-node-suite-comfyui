/**
 * The reports drawn on Generate Edit Instruction, Revise Prompt and Judge Images.
 *
 * The writers show what they were asked, the judge's notes they corrected and what they wrote;
 * the judge shows every picture's scores and notes.
 */

import { app } from "../../scripts/app.js";
import { CARD_LOOK } from "./interface/card.js";
import { createReportPanel } from "./interface/report_panel.js";
import { appendInterfaceWidget, boundTextBoxes } from "./interface/widget.js";
import { LOOK, SCROLLBARS } from "./interface/window_kit.js";

const EXT_NAME = "WASNodeSuite.VisionAgentUI";

// The narrowest a picture's card is drawn before the cards stack.
const CARD_MIN = 210;

// Node id -> how its report is drawn. Heights are node units.
const PANELS = {
  WASGenerateEditInstruction: {
    className: "was-edit-instruction",
    height: 260,
    labelWidth: 72,
    emptyLabel: "run to see the instruction",
    failure: "Failed to read the instruction report:",
  },
  WASRevisePrompt: {
    className: "was-revise-prompt",
    height: 260,
    labelWidth: 72,
    emptyLabel: "run to see the prompt",
    failure: "Failed to read the prompt report:",
  },
  WASJudgeImages: {
    className: "was-judge-images",
    height: 300,
    labelWidth: 72,
    emptyLabel: "run to see the scores",
    failure: "Failed to read the scores:",
    cards: true,
  },
};

const UI_WIDGET_NAME = "was_vision_agent_ui";
const UI_WIDGET_TYPE = "was_vision_agent";

/**
 * An element with a style.
 *
 * @param {string} tag - The tag name.
 * @param {string} css - Its inline style.
 * @param {string} [text] - Its text.
 * @returns {HTMLElement} The element.
 */
function el(tag, css, text) {
  const node = document.createElement(tag);
  node.style.cssText = css;
  if (text !== undefined) node.textContent = text;
  return node;
}

/**
 * A score as it is read: whole numbers bare, others to one decimal.
 *
 * @param {number} value - The score.
 * @returns {string} The score written out.
 */
function scoreText(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "?";
  return Number.isInteger(number) ? String(number) : number.toFixed(1);
}

/**
 * One picture's card: its number, pass or fail, its scores against the least each needs, and
 * the judge's notes.
 *
 * @param {object} row - `{index, kept, passed, scores, notes}`.
 * @param {object} least - Score name to the least that passes.
 * @param {string} word - `Edit` or `Image`.
 * @returns {HTMLElement} The card.
 */
function pictureCard(row, least, word) {
  const tone = row.passed ? LOOK.success : LOOK.danger;
  const edge = row.kept ? CARD_LOOK.rule : CARD_LOOK.edge;
  const card = el(
    "div",
    `flex:1 1 ${CARD_MIN}px;min-width:0;box-sizing:border-box;display:flex;flex-direction:column;`
      + `gap:6px;padding:8px 9px;border-radius:7px;background:${CARD_LOOK.inset};`
      + `border:1px solid ${edge};${row.kept ? `box-shadow:0 0 0 1px ${CARD_LOOK.rule};` : ""}`,
  );

  const head = el("div", "display:flex;align-items:center;gap:6px;flex-wrap:wrap");
  head.appendChild(el("span", `font-weight:650;color:${CARD_LOOK.title}`, `${word} ${row.index}`));
  head.appendChild(el(
    "span",
    `font-size:10px;font-weight:650;letter-spacing:0.04em;text-transform:uppercase;padding:1px 6px;`
      + `border-radius:9px;color:${tone};border:1px solid color-mix(in srgb, ${tone} 55%, transparent)`,
    row.passed ? "pass" : "fail",
  ));
  if (row.kept) {
    head.appendChild(el(
      "span",
      `margin-left:auto;font-size:10px;font-weight:650;letter-spacing:0.04em;text-transform:uppercase;`
        + `color:${LOOK.accent}`,
      "kept",
    ));
  }
  card.appendChild(head);

  const chips = el("div", "display:flex;flex-wrap:wrap;gap:4px");
  for (const [name, value] of Object.entries(row.scores ?? {})) {
    const low = Number(value) < Number(least?.[name] ?? 0);
    const chip = el(
      "span",
      "display:inline-flex;align-items:baseline;gap:4px;padding:1px 7px;border-radius:5px;"
        + `background:color-mix(in srgb, ${low ? LOOK.danger : LOOK.text} 9%, transparent)`,
    );
    chip.title = `${name}: ${scoreText(value)} of 10, ${scoreText(least?.[name] ?? 0)} passes`;
    chip.appendChild(el("span", `font-size:10px;color:${LOOK.muted}`, name));
    chip.appendChild(el("span", `font-weight:650;color:${low ? LOOK.danger : LOOK.text}`, scoreText(value)));
    chips.appendChild(chip);
  }
  card.appendChild(chips);

  if (row.notes) {
    card.appendChild(el(
      "div",
      `font-size:11px;line-height:1.4;color:${LOOK.text};opacity:0.86;white-space:normal;`
        + "overflow-wrap:anywhere",
      row.notes,
    ));
  }
  return card;
}

/**
 * The cards Judge Images draws, one per picture, wrapping into as many columns as the node's
 * width holds.
 *
 * @returns {{element: HTMLElement, update: Function, clear: Function, dispose: Function}} The
 *   sketch, for `createReportPanel`.
 */
function createJudgeCards() {
  const element = el(
    "div",
    "flex:1 1 auto;min-height:0;overflow:auto;display:flex;flex-wrap:wrap;align-content:flex-start;"
      + `gap:6px;${SCROLLBARS}`,
  );
  let shown = "";

  const update = (report) => {
    const body = (report?.bodies ?? []).find((entry) => entry.name === "pictures");
    const text = body?.text || "";
    if (text === shown) return;
    shown = text;
    let parsed = null;
    try {
      parsed = text ? JSON.parse(text) : null;
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to read the scores:`, error);
    }
    const rows = Array.isArray(parsed?.pictures) ? parsed.pictures : [];
    const word = rows.some((row) => "consistency" in (row.scores ?? {})) ? "Edit" : "Image";
    element.replaceChildren(...rows.map((row) => pictureCard(row, parsed.least ?? {}, word)));
  };
  const clear = () => {
    shown = "";
    element.replaceChildren();
  };
  return { element, update, clear, dispose: clear };
}

app.registerExtension({
  name: EXT_NAME,

  beforeRegisterNodeDef(nodeType, nodeData) {
    const options = PANELS[nodeData?.name];
    if (!options) return;

    const proto = nodeType.prototype;
    // One panel per node type, however often the definitions are registered.
    if (proto.__was_vision_agent_wrapped) return;
    proto.__was_vision_agent_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const { cards, ...look } = options;
        const panel = createReportPanel(this, {
          ...look,
          logName: EXT_NAME,
          ...(cards ? { sketch: () => createJudgeCards(), bodies: false, facts: false, tiles: false } : {}),
        });
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });
        boundTextBoxes(this);

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the report:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the report:`, error);
      }
      return result;
    };
  },
});
