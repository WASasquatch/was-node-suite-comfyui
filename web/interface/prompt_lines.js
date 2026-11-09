/**
 * Prompt text written one part per line, worked out again in the browser.
 *
 * One function per python one in `modules/prompt/prompt_lines.py`. No imports and no DOM. A
 * parsed choice also carries where its braces, bars and prefixes stand, for colouring.
 */

/** What a line is. */
export const PROMPT = "prompt";
export const TERM = "term";
export const VERSION = "version";
export const SECTION = "section";
export const NOTE = "note";
export const BLANK = "blank";

/** What a heading starts with. */
export const SECTION_MARK = "==";

/** How several lists are put together into entries. */
export const CYCLE = "cycle";
export const EVERY = "every combination";
export const COMBINES = Object.freeze([CYCLE, EVERY]);

/** The most entries every combination may make. */
export const MAX_COMBINED = 10000;

/** The most lists one node reads, and the highest line a list's pick names. */
export const LIST_LIMIT = 64;
export const PICK_LIMIT = 999999;

/** The first character of a line that is switched off, and the one after a version's mark. */
export const OFF_MARK = "~";

/** What a version of the line above starts with, and what the version in use starts with. */
export const VERSION_MARK = "|";
export const PICKED_MARK = ">";

/** How the versions of a line take turns from one run to the next. */
export const FIXED = "fixed";
export const INCREMENT = "increment";
export const DECREMENT = "decrement";
export const RANDOM = "random";
export const ALL = "all";
export const MODES = Object.freeze([FIXED, INCREMENT, DECREMENT, RANDOM, ALL]);

/** How kept lines are joined into one text, by name. */
export const JOINS = Object.freeze({ "line break": "\n", space: " ", comma: ", " });

/** `@mode` and the space after it, at the start of a line. */
export const MODE = /^@(fixed|increment|decrement|random|all)(?=\s|$)\s*/;

/** `$name = value`, the whole of a term's line after its marks. */
export const TERM_LINE = /^\$([A-Za-z_][A-Za-z0-9_]*)[ \t]*=[ \t]*(.*)$/;

/** `$name` or `${name}` in a line, unless its dollar is escaped. */
export const REFERENCE = /(?<!\\)\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))/g;

// What a one-line note starts with.
const LINE_NOTES = ["#", "//"];

// What opens and closes a note running over several lines.
const BLOCK_OPEN = "/*";
const BLOCK_CLOSE = "*/";

// A line break as a text box writes one.
const BREAKS = /\r\n|\r|\n/;

// `2$$` or `1-3$$` before a choice's first option.
const COUNT = /^\s*([0-9]+)(?:\s*-\s*([0-9]+))?\s*\$\$/;

// `3::` or `0.5::` before an option.
const WEIGHT = /^\s*([0-9]+(?:\.[0-9]+)?)::/;

// `{name}` right after a dollar, which is a reference rather than a choice.
const BRACED_NAME = /^\{[A-Za-z_][A-Za-z0-9_]*\}/;

// A character kept as text by a backslash.
const ESCAPED = /\\([{}|$])/g;

// What several drawn options are joined with when the choice names nothing else.
const JOINER = ", ";

// How many terms deep a term's value is expanded.
const TERM_DEPTH = 8;

// Arithmetic of the draw.
const GOLDEN = 0x9e3779b9;
const SPAN = 4294967296;

// Python's `str.strip()`, `str.lstrip()`, `str.rstrip()` and `str.rstrip(",")`.
const strip = (line) => line.replace(/^\s+|\s+$/gu, "");
const lstrip = (line) => line.replace(/^\s+/u, "");
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
 * One line and what it is.
 *
 * @param {string} kind - What the line is.
 * @param {string} line - The line as written.
 * @param {object} [rest] - Any of `body`, `off`, `mode`, `picked`, `name` and `block`.
 * @returns {object} The line, every field present.
 */
function lineOf(kind, line, rest = {}) {
  return { kind, line, body: "", off: false, mode: FIXED, picked: false, name: "", block: false, ...rest };
}

/**
 * Read a line that may be a heading.
 *
 * @param {string} line - The line as written.
 * @param {string} stripped - The line without the space around it.
 * @returns {object|null} The line as a `SECTION`, or null when it is not one.
 */
function section(line, stripped) {
  let rest = stripped;
  const off = rest.startsWith(OFF_MARK);
  if (off) rest = lstrip(rest.slice(OFF_MARK.length));
  if (!rest.startsWith(SECTION_MARK)) return null;
  return lineOf(SECTION, line, { body: strip(rest.replace(/^=+|=+$/g, "")), off });
}

/**
 * Read a line that is a prompt or a term.
 *
 * @param {string} line - The line as written.
 * @param {string} stripped - The line without the space around it.
 * @returns {object} The line as a `PROMPT` or `TERM`.
 */
function headed(line, stripped) {
  let rest = stripped;
  const off = rest.startsWith(OFF_MARK);
  if (off) rest = lstrip(rest.slice(OFF_MARK.length));
  let mode = FIXED;
  const found = MODE.exec(rest);
  if (found) {
    mode = found[1];
    rest = rest.slice(found[0].length);
  }
  const term = TERM_LINE.exec(rest);
  if (term) return lineOf(TERM, line, { body: strip(term[2]), off, mode, name: term[1] });
  return lineOf(PROMPT, line, { body: strip(rest), off, mode });
}

/**
 * Name what each line is.
 *
 * @param {string} text - The text, one part per line.
 * @returns {object[]} One line per line: `{kind, line, body, off, mode, picked, name, block}`.
 */
export function classify(text) {
  const found = [];
  let inBlock = false;
  for (const line of splitLines(text)) {
    const stripped = strip(line);
    if (inBlock) {
      found.push(lineOf(NOTE, line, { body: line, block: true }));
      if (line.includes(BLOCK_CLOSE)) inBlock = false;
    } else if (!stripped) {
      found.push(lineOf(BLANK, line));
    } else if (stripped.startsWith(BLOCK_OPEN)) {
      found.push(lineOf(NOTE, line, { body: line, block: true }));
      inBlock = !stripped.slice(BLOCK_OPEN.length).includes(BLOCK_CLOSE);
    } else if (LINE_NOTES.some((mark) => stripped.startsWith(mark))) {
      found.push(lineOf(NOTE, line, { body: line }));
    } else if (section(line, stripped)) {
      found.push(section(line, stripped));
    } else if (stripped[0] === VERSION_MARK || stripped[0] === PICKED_MARK) {
      let rest = stripped.slice(1);
      const off = rest.startsWith(OFF_MARK);
      if (off) rest = rest.slice(OFF_MARK.length);
      found.push(lineOf(VERSION, line, { body: strip(rest), off, picked: stripped[0] === PICKED_MARK }));
    } else {
      found.push(headed(line, stripped));
    }
  }
  return found;
}

/**
 * Gather one line and its versions.
 *
 * @param {object[]} lines - Every line, from `classify`.
 * @param {number[]} indices - The group's line indices, the line itself first.
 * @returns {object} The group.
 */
function group(lines, indices) {
  const head = lines[indices[0]];
  const live = indices
    .map((index, position) => (position === 0 || !lines[index].off ? position : -1))
    .filter((position) => position >= 0);
  const pick = live.find((position) => position && lines[indices[position]].picked) ?? 0;
  const stray = head.kind === VERSION;
  return {
    kind: head.kind === TERM ? TERM : PROMPT,
    versions: indices.slice(),
    mode: stray ? FIXED : head.mode,
    off: head.off,
    name: head.name,
    pick,
    live,
  };
}

/**
 * Gather each prompt and term with the versions written under it.
 *
 * @param {object[]} lines - Every line, from `classify`.
 * @returns {object[]} The groups in line order: `{kind, versions, mode, off, name, pick, live}`.
 *   A version with no prompt or term above it starts its own.
 */
export function groupLines(lines) {
  const groups = [];
  let current = [];
  lines.forEach((line, index) => {
    if (line.kind === VERSION && current.length) {
      current.push(index);
      return;
    }
    if (current.length) {
      groups.push(group(lines, current));
      current = [];
    }
    if (line.kind === PROMPT || line.kind === TERM || line.kind === VERSION) current = [index];
  });
  if (current.length) groups.push(group(lines, current));
  return groups;
}

/**
 * Scramble a 32-bit number.
 *
 * @param {number} value - Any integer; its low 32 bits are used.
 * @returns {number} A 32-bit number.
 */
function mix(value) {
  let x = value >>> 0;
  x ^= x >>> 16;
  x = Math.imul(x, 0x85ebca6b);
  x ^= x >>> 13;
  x = Math.imul(x, 0xc2b2ae35);
  x ^= x >>> 16;
  return x >>> 0;
}

/**
 * The draws one line takes, the same for the same seed and line.
 *
 * @param {number} seed - The seed; its low 32 bits are used.
 * @param {number} line - The line's index.
 * @returns {() => number} Answers the next draw in `[0, 1)` each call.
 */
export function stream(seed, line) {
  let state = mix(((Number(seed) || 0) >>> 0) ^ mix(line + GOLDEN));
  return () => {
    state = (state + GOLDEN) >>> 0;
    return mix(state) / SPAN;
  };
}

/**
 * Find the brace that closes the one at `start`.
 *
 * @param {string} text - The text.
 * @param {number} start - Where a `{` stands.
 * @returns {number} The index of its `}`, or -1 when it is never closed.
 */
function closing(text, start) {
  let depth = 0;
  let index = start;
  while (index < text.length) {
    const char = text[index];
    if (char === "\\") {
      index += 2;
      continue;
    }
    if (char === "{") depth += 1;
    else if (char === "}") {
      depth -= 1;
      if (depth === 0) return index;
    }
    index += 1;
  }
  return -1;
}

/**
 * Split a choice's inside at each `|` not nested or escaped.
 *
 * @param {string} text - What stands between the braces.
 * @returns {Array<{text: string, start: number}>} The options as written, and where each starts.
 */
function options(text) {
  const found = [];
  let depth = 0;
  let start = 0;
  let index = 0;
  while (index < text.length) {
    const char = text[index];
    if (char === "\\") {
      index += 2;
      continue;
    }
    if (char === "{") depth += 1;
    else if (char === "}") depth -= 1;
    else if (char === "|" && depth === 0) {
      found.push({ text: text.slice(start, index), start });
      start = index + 1;
    }
    index += 1;
  }
  found.push({ text: text.slice(start), start });
  return found;
}

/**
 * Read what stands between a choice's braces.
 *
 * @param {string} inside - The text between `{` and `}`.
 * @param {number} at - Where `inside` starts in the parsed line.
 * @returns {object} `{low, high, joiner, options, literal, start, end, head, bars}`: each option
 *   `{weight, parts, start, weightEnd, end}`, `head` the count prefix's span or null, `bars` where
 *   each `|` stands.
 */
function choice(inside, at) {
  const pieces = options(inside);
  const bars = pieces.slice(1).map((piece) => at + piece.start - 1);
  let low = 1;
  let high = 1;
  let joiner = JOINER;
  let head = null;
  const counted = COUNT.exec(pieces[0].text);
  if (counted) {
    low = Number(counted[1]);
    high = Number(counted[2] || counted[1]);
    if (high < low) [low, high] = [high, low];
    let rest = pieces[0].text.slice(counted[0].length);
    let restStart = pieces[0].start + counted[0].length;
    const named = rest.indexOf("$$");
    if (named >= 0) {
      joiner = rest.slice(0, named);
      rest = rest.slice(named + 2);
      restStart += named + 2;
    }
    head = [at + pieces[0].start, at + restStart];
    pieces[0] = { text: rest, start: restStart };
  }
  const drawn = pieces.map((piece) => {
    const weighted = WEIGHT.exec(piece.text);
    const skip = weighted ? weighted[0].length : 0;
    return {
      weight: weighted ? Number(weighted[1]) : 1,
      parts: parseChoices(piece.text.slice(skip), at + piece.start + skip),
      start: at + piece.start,
      weightEnd: at + piece.start + skip,
      end: at + piece.start + piece.text.length,
    };
  });
  return {
    low, high, joiner, options: drawn, literal: pieces.length === 1 && !counted,
    start: at - 1, end: at + inside.length, head, bars,
  };
}

/**
 * Find the `{a|b}` choices in a text.
 *
 * @param {string} text - One line, or one option of a choice.
 * @param {number} [offset] - Where `text` starts in the line, for the positions answered.
 * @returns {Array<string|object>} Its parts in order, each a string or a choice. A `{` never
 *   closed, an escaped one and the brace of `${name}` are text.
 */
export function parseChoices(text, offset = 0) {
  const parts = [];
  let buffer = "";
  let index = 0;
  while (index < text.length) {
    const char = text[index];
    if (char === "\\" && index + 1 < text.length) {
      buffer += text.slice(index, index + 2);
      index += 2;
      continue;
    }
    if (char === "{") {
      const afterDollar = index > 0 && text[index - 1] === "$" && BRACED_NAME.test(text.slice(index));
      const close = afterDollar ? -1 : closing(text, index);
      if (close > index) {
        if (buffer) {
          parts.push(buffer);
          buffer = "";
        }
        parts.push(choice(text.slice(index + 1, close), offset + index + 1));
        index = close + 1;
        continue;
      }
    }
    buffer += char;
    index += 1;
  }
  if (buffer) parts.push(buffer);
  return parts;
}

/**
 * Draw one position, each as likely as its weight.
 *
 * @param {number[]} weights - One weight per option; zero is never drawn.
 * @param {() => number} drawNext - The line's draws.
 * @returns {number} The position drawn, or -1 when no weight is above zero.
 */
function weighted(weights, drawNext) {
  let total = 0;
  for (const weight of weights) {
    if (weight > 0) total += weight;
  }
  if (total <= 0) return -1;
  const target = drawNext() * total;
  let reached = 0;
  let last = -1;
  for (let position = 0; position < weights.length; position += 1) {
    const weight = weights[position];
    if (weight <= 0) continue;
    reached += weight;
    last = position;
    if (target < reached) return position;
  }
  return last;
}

/**
 * Draw every choice among some parts.
 *
 * @param {Array<string|object>} parts - From `parseChoices`.
 * @param {() => number} drawNext - The line's draws.
 * @returns {string} The text with each choice replaced by what it drew.
 */
function drawn(parts, drawNext) {
  let out = "";
  for (const part of parts) {
    if (typeof part === "string") {
      out += part;
      continue;
    }
    if (part.literal) {
      out += `{${drawn(part.options[0].parts, drawNext)}}`;
      continue;
    }
    let count = part.low;
    if (part.high > part.low) count = part.low + Math.floor(drawNext() * (part.high - part.low + 1));
    const weights = part.options.map((option) => option.weight);
    const picks = [];
    for (let taken = 0; taken < count; taken += 1) {
      const position = weighted(weights, drawNext);
      if (position < 0) break;
      picks.push(position);
      weights[position] = 0;
    }
    out += picks.map((position) => drawn(part.options[position].parts, drawNext)).join(part.joiner);
  }
  return out;
}

/**
 * Draw every `{a|b}` choice in one line.
 *
 * @param {string} text - The line's text.
 * @param {number} seed - The seed.
 * @param {number} line - The line's index, so two lines draw apart.
 * @returns {string} The text with each choice drawn. Escapes are kept for `unescape`.
 */
export function draw(text, seed, line) {
  if (!text.includes("{")) return text;
  return drawn(parseChoices(text), stream(seed, line));
}

/**
 * Replace each `$name` and `${name}` with its term.
 *
 * @param {string} text - The text.
 * @param {Map<string, string>} terms - Name to value, from `glossary`.
 * @param {string[]} [inside] - The names whose values this text is part of.
 * @returns {string} The text with every known name replaced. An unknown name, and one naming a
 *   term it is already inside, is left as written.
 */
export function expand(text, terms, inside = []) {
  if (!text.includes("$")) return text;
  return text.replace(REFERENCE, (whole, braced, bare) => {
    const name = braced || bare;
    if (!terms.has(name) || inside.includes(name) || inside.length >= TERM_DEPTH) return whole;
    return expand(terms.get(name), terms, [...inside, name]);
  });
}

/**
 * Close the gaps a choice or term left behind: empty list items and doubled spaces.
 *
 * @param {string} text - The line after drawing and expanding.
 * @param {string} written - The line as written, whose own leading or trailing comma is kept.
 * @returns {string} The line tidied and trimmed.
 */
export function tidy(text, written) {
  let out = text.replace(/[ \t]*,(?:[ \t]*,)+/g, ",");
  out = out.replace(/[ \t]+,/g, ",");
  out = out.replace(/[ \t]{2,}/g, " ");
  if (!lstrip(written).startsWith(",")) out = out.replace(/^[ \t]*,[ \t]*/, "");
  if (!rstrip(written).endsWith(",")) out = out.replace(/[ \t]*,[ \t]*$/, "");
  return strip(out);
}

/**
 * Drop the backslash from each escaped brace, bar and dollar.
 *
 * @param {string} text - A finished line.
 * @returns {string} The line as it is handed on.
 */
export function unescape(text) {
  return text.includes("\\") ? text.replace(ESCAPED, "$1") : text;
}

/**
 * One line drawn, expanded, tidied and unescaped.
 *
 * @param {string} body - The text after its marks.
 * @param {number} seed - The seed.
 * @param {number} line - The line's index.
 * @param {Map<string, string>} terms - Name to value.
 * @returns {string} The line as it is handed on.
 */
function finished(body, seed, line, terms) {
  const done = expand(draw(body, seed, line), terms);
  return unescape(done === body ? done : tidy(done, body));
}

/**
 * Which lines sit in a section that is switched off.
 *
 * @param {object[]} lines - Every line, from `classify`.
 * @returns {boolean[]} One flag per line, true from a switched-off section's heading to the next
 *   heading.
 */
export function hiddenLines(lines) {
  let off = false;
  return lines.map((line) => {
    if (line.kind === SECTION) off = line.off;
    return off;
  });
}

/**
 * Every term switched on, by name, its choices drawn once.
 *
 * @param {object[]} lines - Every line, from `classify`.
 * @param {object[]} groups - From `groupLines`.
 * @param {number} seed - The seed.
 * @returns {Map<string, string>} Name to value. A name written twice takes its last value.
 */
export function glossary(lines, groups, seed) {
  const hidden = hiddenLines(lines);
  const terms = new Map();
  for (const entry of groups) {
    if (entry.kind !== TERM || entry.off || hidden[entry.versions[0]]) continue;
    const index = entry.versions[entry.pick];
    terms.set(entry.name, draw(lines[index].body, seed, index));
  }
  return terms;
}

/**
 * Every line a node could hand on, finished, before blanks and notes are left out.
 *
 * @param {string} text - The text, one part per line.
 * @param {number} [seed] - What the choices draw from.
 * @param {Map<string, string>} [terms] - Terms from elsewhere, such as another list; the text's own
 *   replace them.
 * @returns {Array<{kind: string, text: string, line: number}>} In order, `kind` one of `PROMPT`,
 *   `SECTION`, `NOTE` and `BLANK`, a heading giving its name, and `line` the index of the line it
 *   came from. A prompt in mode `ALL` gives each of its versions. A switched-off heading gives
 *   nothing, nor do the lines under it.
 */
export function resolve(text, seed = 0, terms = null) {
  const lines = classify(text);
  const groups = groupLines(lines);
  const known = new Map([...(terms ?? new Map()), ...glossary(lines, groups, seed)]);
  const hidden = hiddenLines(lines);
  const heads = new Map(groups.map((entry) => [entry.versions[0], entry]));
  const out = [];
  lines.forEach((line, index) => {
    if (hidden[index]) return;
    if (line.kind === SECTION) {
      out.push({ kind: SECTION, text: line.body, line: index });
    } else if (line.kind === NOTE) {
      out.push({ kind: NOTE, text: rstrip(line.line), line: index });
    } else if (line.kind === BLANK) {
      out.push({ kind: BLANK, text: "", line: index });
    } else if (heads.has(index)) {
      const entry = heads.get(index);
      if (entry.off || entry.kind === TERM) return;
      const indent = line.line.slice(0, line.line.length - lstrip(line.line).length);
      const chosen = entry.mode === ALL ? entry.live : [entry.pick];
      for (const position of chosen) {
        const at = entry.versions[position];
        out.push({ kind: PROMPT, text: indent + finished(lines[at].body, seed, at, known), line: at });
      }
    }
  });
  return out;
}

/**
 * Whether a finished line of a given kind is kept.
 *
 * @param {string} kind - `PROMPT`, `SECTION`, `NOTE` or `BLANK`.
 * @param {boolean} skipBlankLines - Empty lines are left out.
 * @param {boolean} skipCommentLines - Notes are left out.
 * @returns {boolean} True when the line is kept. A section's heading never is.
 */
export function handedOn(kind, skipBlankLines, skipCommentLines) {
  if (kind === SECTION) return false;
  if (kind === NOTE) return !skipCommentLines;
  if (kind === BLANK) return !skipBlankLines;
  return true;
}

/**
 * The finished lines that are kept.
 *
 * @param {string} text - The text, one part per line.
 * @param {boolean} [skipBlankLines] - Leave out empty lines.
 * @param {boolean} [skipCommentLines] - Leave out notes.
 * @param {number} [seed] - What the choices draw from.
 * @returns {string[]} The kept lines in order. Terms, versions not in use and lines switched off
 *   are never kept.
 */
export function keptLines(text, skipBlankLines = true, skipCommentLines = true, seed = 0) {
  return resolve(text, seed)
    .filter(({ kind }) => handedOn(kind, skipBlankLines, skipCommentLines))
    .map(({ text: line }) => line);
}

/**
 * The kept lines, each trimmed, one entry per line.
 *
 * @param {string} text - The text, one part per line.
 * @param {boolean} [skipBlankLines] - Leave out empty lines rather than giving each as "".
 * @param {boolean} [skipCommentLines] - Leave out notes rather than giving each as written.
 * @param {number} [seed] - What the choices draw from.
 * @returns {string[]} The entries in line order.
 */
export function listLines(text, skipBlankLines = true, skipCommentLines = true, seed = 0) {
  return keptLines(text, skipBlankLines, skipCommentLines, seed).map(strip);
}

/**
 * The kept lines as one text.
 *
 * @param {string} text - The text, one part per line.
 * @param {boolean} [skipBlankLines] - Leave out empty lines.
 * @param {boolean} [skipCommentLines] - Leave out notes.
 * @param {string} [joinWith] - A key of `JOINS`.
 * @param {number} [seed] - What the choices draw from.
 * @returns {string} The joined text, empty when nothing is kept or `joinWith` is unknown.
 */
export function joinLines(text, skipBlankLines = false, skipCommentLines = true, joinWith = "line break", seed = 0) {
  if (!Object.hasOwn(JOINS, joinWith)) return "";
  return joinParts(keptLines(text, skipBlankLines, skipCommentLines, seed), joinWith);
}

/**
 * Join finished lines into one text.
 *
 * @param {string[]} parts - The lines.
 * @param {string} joinWith - A key of `JOINS`.
 * @returns {string} The joined text, empty when `joinWith` is unknown.
 */
export function joinParts(parts, joinWith) {
  if (!Object.hasOwn(JOINS, joinWith)) return "";
  if (joinWith === "line break") return parts.map(rstrip).join(JOINS[joinWith]);
  let trimmed = parts.map(strip);
  if (joinWith === "comma") trimmed = trimmed.map((part) => rstrip(rstripCommas(part)));
  return trimmed.filter(Boolean).join(JOINS[joinWith]);
}

/**
 * One list read from what a node stored.
 *
 * @param {*} entry - An object with any of `name`, `mode`, `pick`, `on` and `lines`, or a string of
 *   lines.
 * @returns {object|null} `{name, mode, pick, on, lines}` with every field filled, or null for
 *   anything else.
 */
function oneList(entry) {
  const found = typeof entry === "string" ? { lines: entry } : entry;
  if (!found || typeof found !== "object" || Array.isArray(found)) return null;
  const { name, mode, pick, lines } = found;
  const counted = typeof pick === "number" && Number.isInteger(pick);
  return {
    name: typeof name === "string" ? name : "",
    mode: MODES.includes(mode) ? mode : ALL,
    pick: counted && pick >= 1 && pick <= PICK_LIMIT ? pick : 1,
    on: found.on !== false,
    lines: typeof lines === "string" ? lines : "",
  };
}

/**
 * The lists a String List (Sections) node stores, as JSON.
 *
 * @param {string} value - A JSON array of lists, each `{name, mode, pick, on, lines}`. Text that is
 *   not one is read as a single list of lines.
 * @returns {object[]} Up to `LIST_LIMIT` lists, every field filled.
 */
export function readLists(value) {
  const text = String(value ?? "");
  let data;
  try {
    data = strip(text) ? JSON.parse(text) : [];
  } catch {
    return [oneList(text)];
  }
  if (!Array.isArray(data)) return [oneList(text)];
  return data.slice(0, LIST_LIMIT).map(oneList).filter(Boolean);
}

/**
 * The seed one list draws from, so two lists draw apart.
 *
 * @param {number} seed - The node's seed.
 * @param {number} index - The list's place, from zero.
 * @returns {number} A 32-bit seed.
 */
export function listSeed(seed, index) {
  return ((Number(seed) || 0) + index * GOLDEN) >>> 0;
}

/**
 * Each switched-on list's finished prompts, a list in a mode other than `ALL` giving one.
 *
 * @param {object[]} lists - From `readLists`.
 * @param {number} [seed] - The node's seed.
 * @returns {Array<{list: number, entries: string[], candidates: Array<{text: string, line: number}>}>}
 *   One per switched-on list, in order: its place among all the lists, its entries, and every
 *   prompt it could hand on with the line it came from. Terms from any list reach every list.
 */
export function listEntries(lists, seed = 0) {
  const texts = lists
    .map((entry, index) => ({ entry, index, own: listSeed(seed, index) }))
    .filter(({ entry }) => entry.on);
  const terms = new Map();
  for (const { entry, own } of texts) {
    const lines = classify(entry.lines);
    for (const [name, value] of glossary(lines, groupLines(lines), own)) terms.set(name, value);
  }
  return texts.map(({ entry, index, own }) => {
    const candidates = resolve(entry.lines, own, terms)
      .filter((item) => item.kind === PROMPT && strip(item.text))
      .map((item) => ({ text: strip(item.text), line: item.line }));
    let entries = candidates.map((item) => item.text);
    if (entry.mode !== ALL && entries.length) entries = [entries[(entry.pick - 1) % entries.length]];
    return { list: index, entries, candidates };
  });
}

/**
 * Put several lists together into one, each entry taking one line from every list.
 *
 * @param {string[][]} lists - The lists, in the order their lines are joined. An empty one is left
 *   out.
 * @param {string} [how] - `CYCLE` or `EVERY`.
 * @param {string} [joinWith] - A key of `JOINS`.
 * @returns {string[]|null} The entries, or null where every combination would make more than
 *   `MAX_COMBINED`, or `how` or `joinWith` is unknown.
 */
export function combine(lists, how = CYCLE, joinWith = "comma") {
  if (!COMBINES.includes(how) || !Object.hasOwn(JOINS, joinWith)) return null;
  const kept = lists.filter((entries) => entries.length);
  if (!kept.length) return [];
  if (how === CYCLE) {
    const longest = Math.max(...kept.map((entries) => entries.length));
    const rows = [];
    for (let index = 0; index < longest; index += 1) {
      rows.push(kept.map((entries) => entries[index % entries.length]));
    }
    return rows.map((row) => joinParts(row, joinWith));
  }
  let total = 1;
  for (const entries of kept) total *= entries.length;
  if (total > MAX_COMBINED) return null;
  let rows = [[]];
  for (const entries of kept) rows = rows.flatMap((row) => entries.map((entry) => [...row, entry]));
  return rows.map((row) => joinParts(row, joinWith));
}
