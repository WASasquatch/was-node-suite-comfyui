/**
 * Edits to prompt lines: a switch, a picked version, a mode, a new line, and the turn versions
 * and lists take after a run.
 *
 * Each takes text or a node's lists and answers them edited. No DOM.
 */

import {
  BLANK, DECREMENT, FIXED, INCREMENT, MODE, OFF_MARK, PICKED_MARK, PROMPT, RANDOM, SECTION, TERM, VERSION,
  VERSION_MARK, classify, groupLines, listEntries, parseChoices,
} from "./prompt_lines.js";

// The modes that move on after a run.
const TURNING = [INCREMENT, DECREMENT, RANDOM];

/**
 * How many space characters a line starts with.
 *
 * @param {string} line - One line.
 * @returns {number} The count.
 */
export function leadOf(line) {
  return line.length - line.replace(/^\s+/u, "").length;
}

/**
 * The group a line belongs to.
 *
 * @param {object[]} groups - From `groupLines`.
 * @param {number} index - Line index from zero.
 * @returns {{group: object, position: number}|null} The group and the line's place in it.
 */
export function groupAt(groups, index) {
  for (const group of groups) {
    const position = group.versions.indexOf(index);
    if (position >= 0) return { group, position };
  }
  return null;
}

/**
 * Write each version's mark so one of them is in use.
 *
 * @param {string[]} all - The lines, edited in place.
 * @param {object} group - The group.
 * @param {number} chosen - The position to put in use; the first needs no mark.
 * @returns {boolean} True when a line changed.
 */
function markPick(all, group, chosen) {
  let changed = false;
  group.versions.forEach((line, position) => {
    if (position === 0) return;
    const lead = leadOf(all[line]);
    const mark = position === chosen ? PICKED_MARK : VERSION_MARK;
    if (all[line][lead] === mark) return;
    all[line] = all[line].slice(0, lead) + mark + all[line].slice(lead + 1);
    changed = true;
  });
  return changed;
}

/**
 * Switch a line on or off: `~` at its start, or after a version's mark.
 *
 * @param {string} text - The text.
 * @param {number} index - Line index from zero.
 * @returns {string} The text edited, or as it was for a blank line.
 */
export function switchLine(text, index) {
  const all = text.split("\n");
  const known = classify(text)[index];
  if (!known || known.kind === BLANK) return text;
  const line = all[index];
  const at = leadOf(line) + (known.kind === VERSION ? 1 : 0);
  all[index] = line.slice(at).startsWith(OFF_MARK)
    ? line.slice(0, at) + line.slice(at + OFF_MARK.length)
    : line.slice(0, at) + OFF_MARK + line.slice(at);
  return all.join("\n");
}

/**
 * Put one version of a line in use.
 *
 * @param {string} text - The text.
 * @param {number} index - The line index of the version, or of the line itself.
 * @returns {string} The text edited.
 */
export function pickVersion(text, index) {
  const found = groupAt(groupLines(classify(text)), index);
  if (!found) return text;
  const all = text.split("\n");
  return markPick(all, found.group, found.position) ? all.join("\n") : text;
}

/**
 * Set how a line's versions take turns, by the `@mode` at its start.
 *
 * @param {string} text - The text.
 * @param {number} index - Any line of the group.
 * @param {string} mode - A value of `MODES`; `FIXED` writes no mark.
 * @returns {string} The text edited.
 */
export function setMode(text, index, mode) {
  const lines = classify(text);
  const found = groupAt(groupLines(lines), index);
  if (!found) return text;
  const head = found.group.versions[0];
  if (lines[head].kind === VERSION) return text;
  const all = text.split("\n");
  const line = all[head];
  let at = leadOf(line);
  if (line.slice(at).startsWith(OFF_MARK)) {
    at += OFF_MARK.length;
    at += leadOf(line.slice(at));
  }
  const rest = line.slice(at);
  const current = MODE.exec(rest);
  const mark = mode === FIXED ? "" : `@${mode} `;
  all[head] = line.slice(0, at) + mark + rest.slice(current ? current[0].length : 0);
  return all.join("\n");
}

/**
 * Add a version under a line's group, a copy of that line's text.
 *
 * @param {string} text - The text.
 * @param {number} index - The line whose text the version starts from.
 * @returns {{text: string, start: number, end: number}|null} The text edited and where the new
 *   version's text stands in it, or null for a line that takes no versions.
 */
export function addVersion(text, index) {
  const lines = classify(text);
  const found = groupAt(groupLines(lines), index);
  if (!found) return null;
  const all = text.split("\n");
  const versions = found.group.versions;
  const first = all[versions[0]];
  const indent = first.slice(0, leadOf(first));
  const body = lines[index].body;
  const last = versions[versions.length - 1];
  all.splice(last + 1, 0, `${indent}${VERSION_MARK} ${body}`);
  let start = 0;
  for (let line = 0; line <= last; line += 1) start += all[line].length + 1;
  start += indent.length + VERSION_MARK.length + 1;
  return { text: all.join("\n"), start, end: start + body.length };
}

/**
 * Add an empty line at the end of the lines under a heading, before any blank lines closing them.
 *
 * @param {string} text - The text.
 * @param {number} heading - The heading's line.
 * @returns {{text: string, start: number, end: number}|null} The text edited and where the caret
 *   goes, or null when the line is not a heading.
 */
export function addLine(text, heading) {
  const lines = classify(text);
  if (lines[heading]?.kind !== SECTION) return null;
  let at = heading + 1;
  while (at < lines.length && lines[at].kind !== SECTION) at += 1;
  while (at - 1 > heading && lines[at - 1].kind === BLANK) at -= 1;
  const all = text.split("\n");
  all.splice(at, 0, "");
  let start = 0;
  for (let line = 0; line < at; line += 1) start += all[line].length + 1;
  return { text: all.join("\n"), start, end: start };
}

/**
 * The position that comes next among some.
 *
 * @param {string} mode - `INCREMENT`, `DECREMENT` or `RANDOM`.
 * @param {number} at - The position in use now.
 * @param {number} count - How many there are.
 * @param {() => number} random - Draws in `[0, 1)`.
 * @returns {number} The next position: the one after, the one before, or another at random.
 */
function nextOf(mode, at, count, random) {
  if (mode === INCREMENT) return (at + 1) % count;
  if (mode === DECREMENT) return (at - 1 + count) % count;
  const step = 1 + Math.min(count - 2, Math.floor(random() * (count - 1)));
  return (at + step) % count;
}

/**
 * Give every line whose versions take turns its next version.
 *
 * @param {string} text - The text.
 * @param {() => number} [random] - Draws in `[0, 1)` for `RANDOM`.
 * @returns {string} The text with each `>` moved, or as it was when no line takes turns.
 */
export function advance(text, random = Math.random) {
  const all = text.split("\n");
  let changed = false;
  for (const group of groupLines(classify(text))) {
    if (group.off || group.live.length < 2 || !TURNING.includes(group.mode)) continue;
    const at = Math.max(0, group.live.indexOf(group.pick));
    if (markPick(all, group, group.live[nextOf(group.mode, at, group.live.length, random)])) changed = true;
  }
  return changed ? all.join("\n") : text;
}

/**
 * Whether any line's versions take turns after a run.
 *
 * @param {string} text - The text.
 * @returns {boolean} True when a switched-on line with two usable versions has a turning mode.
 */
export function takesTurns(text) {
  return groupLines(classify(text)).some((group) => !group.off && group.live.length > 1
    && TURNING.includes(group.mode));
}

/**
 * Give every list its turn after a run: its next line, and the next version of each line in it.
 *
 * @param {object[]} lists - From `readLists`.
 * @param {number} seed - The node's seed.
 * @param {() => number} [random] - Draws in `[0, 1)` for `RANDOM`.
 * @returns {object[]} The lists, a new array when anything moved, or the same one.
 */
export function advanceLists(lists, seed, random = Math.random) {
  const counts = new Map(listEntries(lists, seed).map((found) => [found.list, found.candidates.length]));
  let changed = false;
  const moved = lists.map((entry, index) => {
    const lines = entry.on ? advance(entry.lines, random) : entry.lines;
    const count = counts.get(index) ?? 0;
    let pick = entry.pick;
    if (entry.on && TURNING.includes(entry.mode) && count > 1) {
      pick = nextOf(entry.mode, (entry.pick - 1) % count, count, random) + 1;
    }
    if (lines === entry.lines && pick === entry.pick) return entry;
    changed = true;
    return { ...entry, lines, pick };
  });
  return changed ? moved : lists;
}

/**
 * Whether anything among the lists takes turns after a run.
 *
 * @param {object[]} lists - From `readLists`.
 * @param {number} seed - The node's seed.
 * @returns {boolean} True when a switched-on list, or a line inside one, has something to move to.
 */
export function listsTakeTurns(lists, seed) {
  const counts = new Map(listEntries(lists, seed).map((found) => [found.list, found.candidates.length]));
  return lists.some((entry, index) => entry.on && ((TURNING.includes(entry.mode) && (counts.get(index) ?? 0) > 1)
    || takesTurns(entry.lines)));
}

/**
 * Whether some parts hold a choice that draws.
 *
 * @param {Array<string|object>} parts - From `parseChoices`.
 * @returns {boolean} True for a choice with a count or more than one option, at any depth.
 */
function draws(parts) {
  return parts.some((part) => typeof part !== "string"
    && (!part.literal || part.options.some((option) => draws(option.parts))));
}

/**
 * Whether a switched-on line holds a `{a|b}` choice, so the seed decides what it hands on.
 *
 * @param {string} text - The text.
 * @returns {boolean} True when a prompt, term or version draws from the seed.
 */
export function holdsChoice(text) {
  if (!text.includes("{")) return false;
  return classify(text).some((line) => !line.off
    && (line.kind === PROMPT || line.kind === TERM || line.kind === VERSION)
    && draws(parseChoices(line.body)));
}
