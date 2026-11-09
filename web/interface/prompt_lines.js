/**
 * Prompt text written one part per line, worked out again in the browser.
 *
 * One function per python one in `modules/prompt/prompt_lines.py`. No imports and no DOM.
 */

/** What a line is. */
export const PROMPT = "prompt";
export const OFF = "off";
export const NOTE = "note";
export const BLANK = "blank";

/** The first character of a line that is switched off. */
export const OFF_MARK = "~";

/** How kept lines are joined into one text, by name. */
export const JOINS = Object.freeze({ "line break": "\n", space: " ", comma: ", " });

// What a one-line note starts with.
const LINE_NOTES = ["#", "//"];

// What opens and closes a note running over several lines.
const BLOCK_OPEN = "/*";
const BLOCK_CLOSE = "*/";

// A line break as a text box writes one.
const BREAKS = /\r\n|\r|\n/;

// Python's `str.strip()`, `str.rstrip()` and `str.rstrip(",")`.
const strip = (line) => line.replace(/^\s+|\s+$/gu, "");
const rstrip = (line) => line.replace(/\s+$/u, "");
const rstripCommas = (line) => line.replace(/,+$/u, "");

/**
 * Cut text into its lines.
 *
 * @param {string} text - The text, with any of `\n`, `\r\n` or `\r` between lines.
 * @returns {string[]} One entry per line. The empty line after a final line break is not one.
 */
export function splitLines(text) {
  const lines = String(text ?? "").split(BREAKS);
  if (lines.length > 1 && lines[lines.length - 1] === "") lines.pop();
  return lines;
}

/**
 * Name what each line is.
 *
 * @param {string} text - The text, one part per line.
 * @returns {Array<{kind: string, line: string, block: boolean}>} One per line, `kind` one of
 *   `PROMPT`, `OFF`, `NOTE` and `BLANK`, `line` as written, and `block` true for a line of a
 *   note opened by a slash and a star.
 */
export function classify(text) {
  const kinds = [];
  let inBlock = false;
  for (const line of splitLines(text)) {
    const stripped = strip(line);
    if (inBlock) {
      kinds.push({ kind: NOTE, line, block: true });
      if (line.includes(BLOCK_CLOSE)) inBlock = false;
    } else if (!stripped) {
      kinds.push({ kind: BLANK, line, block: false });
    } else if (stripped.startsWith(BLOCK_OPEN)) {
      kinds.push({ kind: NOTE, line, block: true });
      inBlock = !stripped.slice(BLOCK_OPEN.length).includes(BLOCK_CLOSE);
    } else if (LINE_NOTES.some((mark) => stripped.startsWith(mark))) {
      kinds.push({ kind: NOTE, line, block: false });
    } else if (stripped.startsWith(OFF_MARK)) {
      kinds.push({ kind: OFF, line, block: false });
    } else {
      kinds.push({ kind: PROMPT, line, block: false });
    }
  }
  return kinds;
}

/**
 * Whether a line of a given kind is kept.
 *
 * @param {string} kind - What the line is.
 * @param {boolean} skipBlankLines - Empty lines are left out.
 * @param {boolean} skipCommentLines - Notes are left out.
 * @returns {boolean} True when the line is kept.
 */
export function handedOn(kind, skipBlankLines, skipCommentLines) {
  if (kind === OFF) return false;
  if (kind === NOTE) return !skipCommentLines;
  if (kind === BLANK) return !skipBlankLines;
  return true;
}

/**
 * The lines that are kept, as written.
 *
 * @param {string} text - The text, one part per line.
 * @param {boolean} [skipBlankLines] - Leave out empty lines.
 * @param {boolean} [skipCommentLines] - Leave out notes.
 * @returns {string[]} The kept lines in order. A line switched off is never one.
 */
export function keptLines(text, skipBlankLines = true, skipCommentLines = true) {
  return classify(text)
    .filter(({ kind }) => handedOn(kind, skipBlankLines, skipCommentLines))
    .map(({ line }) => line);
}

/**
 * The kept lines, each trimmed, one entry per line.
 *
 * @param {string} text - The text, one part per line.
 * @param {boolean} [skipBlankLines] - Leave out empty lines rather than giving each as "".
 * @param {boolean} [skipCommentLines] - Leave out notes rather than giving each as written.
 * @returns {string[]} The entries in line order.
 */
export function listLines(text, skipBlankLines = true, skipCommentLines = true) {
  return keptLines(text, skipBlankLines, skipCommentLines).map(strip);
}

/**
 * The kept lines as one text.
 *
 * @param {string} text - The text, one part per line.
 * @param {boolean} [skipBlankLines] - Leave out empty lines.
 * @param {boolean} [skipCommentLines] - Leave out notes.
 * @param {string} [joinWith] - A key of `JOINS`.
 * @returns {string} The joined text, empty when nothing is kept or `joinWith` is unknown.
 */
export function joinLines(text, skipBlankLines = false, skipCommentLines = true, joinWith = "line break") {
  if (!Object.hasOwn(JOINS, joinWith)) return "";
  const kept = keptLines(text, skipBlankLines, skipCommentLines);
  if (joinWith === "line break") return kept.map(rstrip).join(JOINS[joinWith]);
  let parts = kept.map(strip);
  if (joinWith === "comma") parts = parts.map((part) => rstrip(rstripCommas(part)));
  return parts.filter(Boolean).join(JOINS[joinWith]);
}
