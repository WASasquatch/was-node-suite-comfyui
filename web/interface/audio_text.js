/**
 * Time and region lists read from a text widget as the audio nodes read them, each entry
 * keeping its place in the text.
 *
 * Entries are separated by commas, semicolons or lines; a line opening with `#` is skipped.
 */

/**
 * A time written as seconds, `m:ss` or `h:mm:ss`.
 *
 * @param {string} text - The time, such as `12.5`, `1:02.5` or `1:00:02.5`.
 * @returns {number|null} Seconds, or null when the text is not a time.
 */
export function parseTime(text) {
  const raw = String(text).trim().toLowerCase().replace(/s$/, "").trim();
  const parts = raw.split(":").map((part) => part.trim());
  if (!raw || parts.length > 3) return null;
  let total = 0;
  for (let index = 0; index < parts.length; index += 1) {
    const piece = parts[index];
    const [whole, fraction, extra] = piece.split(".");
    const last = index === parts.length - 1;
    if (!/^\d+$/.test(whole ?? "") || extra !== undefined) return null;
    if (fraction !== undefined && (!last || !/^\d+$/.test(fraction))) return null;
    if (index > 0 && Number(whole) >= 60) return null;
    total = total * 60 + Number(piece);
  }
  return total;
}

/**
 * A region written as two times joined by `-` or `to`.
 *
 * @param {string} text - The region, such as `1.2-2.5`.
 * @returns {[number, number]|null} Start and end in seconds, or null when unreadable.
 */
export function parseRegion(text) {
  const lowered = String(text).toLowerCase();
  const halves = lowered.includes(" to ") ? lowered.split(" to ") : lowered.split("-");
  if (halves.length !== 2) return null;
  const a = parseTime(halves[0]);
  const b = parseTime(halves[1]);
  return a !== null && b !== null && b >= a ? [a, b] : null;
}

/**
 * The entries of a list widget, each with where it sits in the text.
 *
 * @param {string} text - The widget's value.
 * @param {(entry: string) => *} read - Reads one entry, answering null when it cannot.
 * @returns {Array<{from: number, to: number, value: *}>} One entry per item written; `value` is
 *   null for one that cannot be read.
 */
export function listEntries(text, read) {
  const value = String(text ?? "");
  const entries = [];
  let lineStart = 0;
  for (const line of value.split("\n")) {
    if (!line.trim().startsWith("#")) {
      for (const match of line.matchAll(/[^,;]+/g)) {
        if (!match[0].trim()) continue;
        const from = lineStart + match.index;
        entries.push({ from, to: from + match[0].length, value: read(match[0]) });
      }
    }
    lineStart += line.length + 1;
  }
  return entries;
}

/**
 * The text with one entry taken out, with the separator beside it.
 *
 * @param {string} text - The widget's value.
 * @param {{from: number, to: number}} entry - One of `listEntries(text, ...)`.
 * @returns {string} The text without it; every other entry is left as written.
 */
export function withoutEntry(text, entry) {
  const value = String(text ?? "");
  let from = entry.from;
  let to = entry.to;
  const after = value.slice(to).match(/^\s*[,;]\s*/);
  if (after) to += after[0].length;
  else {
    const before = value.slice(0, from).match(/\s*[,;]\s*$/);
    if (before) from -= before[0].length;
  }
  return (value.slice(0, from) + value.slice(to)).replace(/^\s+|\s+$/g, "");
}

/**
 * The text with one entry rewritten.
 *
 * @param {string} text - The widget's value.
 * @param {{from: number, to: number}} entry - One of `listEntries(text, ...)`.
 * @param {string} written - The entry's new text.
 * @returns {string} The text with that entry replaced and its leading space kept.
 */
export function replaceEntry(text, entry, written) {
  const value = String(text ?? "");
  const lead = value.slice(entry.from, entry.to).match(/^\s*/)[0];
  return value.slice(0, entry.from) + lead + written + value.slice(entry.to);
}

/**
 * The text with one entry added at the end.
 *
 * @param {string} text - The widget's value.
 * @param {string} written - The new entry's text.
 * @returns {string} The text with `, written` appended.
 */
export function appendEntry(text, written) {
  const value = String(text ?? "").replace(/\s+$/, "");
  return value ? `${value}, ${written}` : written;
}
