/**
 * Rows and columns for a list drawn several items to a row.
 *
 * A grid is runs, each under one header row, then a tail with no header. Items run row major.
 * `buildGrid` answers the shape and `gridPosition` inverts `gridCell`.
 */

/**
 * How many columns a room holds.
 *
 * @param {number} room - Pixels across the grid has.
 * @param {number} cell - Pixels one cell wants.
 * @param {number} limit - Most columns to answer.
 * @returns {number} Columns, at least one and never past `limit`.
 */
export function gridColumns(room, cell, limit) {
  const ceiling = Math.max(1, Math.trunc(limit) || 1);
  if (!(room > 0) || !(cell > 0)) return 1;
  return Math.max(1, Math.min(ceiling, Math.floor(room / cell)));
}

/**
 * How many lines a run of items covers.
 *
 * @param {number} count - Items in the run.
 * @param {number} columns - Cells across.
 * @returns {number} Lines, zero for an empty run.
 */
export function gridLines(count, columns) {
  const across = Math.max(1, Math.trunc(columns) || 1);
  return count <= 0 ? 0 : Math.ceil(count / across);
}

/**
 * Items on one line of a run.
 *
 * @param {number} count - Items in the run.
 * @param {number} line - Which line, counting from the first.
 * @param {number} columns - Cells across.
 * @returns {number} Filled cells on that line, from zero to `columns`.
 */
export function gridSpan(count, line, columns) {
  const across = Math.max(1, Math.trunc(columns) || 1);
  const rest = count - line * across;
  return rest <= 0 ? 0 : Math.min(across, rest);
}

/**
 * Lay a sequence of runs and a tail out as flat rows.
 *
 * @param {Array<{count: number}>} runs - One entry per header, carrying the items under it.
 * @param {number} tail - Items after the last run, drawn with no header.
 * @param {number} columns - Cells across.
 * @returns {{columns: number, sections: Array<{count: number, lines: number, offset: number,
 *   rows: number}>, tailOffset: number, tailCount: number, tailLines: number, total: number}}
 *   The shape, with each run's first row in `offset` and the rows altogether in `total`.
 */
export function buildGrid(runs, tail, columns) {
  const across = Math.max(1, Math.trunc(columns) || 1);
  const sections = [];
  let total = 0;
  for (const run of runs ?? []) {
    const count = Math.max(0, Math.trunc(Number(run?.count) || 0));
    const lines = gridLines(count, across);
    sections.push({ count, lines, offset: total, rows: 1 + lines });
    total += 1 + lines;
  }
  const tailCount = Math.max(0, Math.trunc(Number(tail) || 0));
  const tailLines = gridLines(tailCount, across);
  return {
    columns: across,
    sections,
    tailOffset: total,
    tailCount,
    tailLines,
    total: total + tailLines,
  };
}

/**
 * Which run holds a row.
 *
 * @param {Array<{offset: number}>} sections - Sections from `buildGrid`.
 * @param {number} position - Flat row position, inside the runs.
 * @returns {number} Index of the run, or -1 when there are none.
 */
function sectionAt(sections, position) {
  let low = 0;
  let high = sections.length - 1;
  if (high < 0) return -1;
  while (low < high) {
    const middle = (low + high + 1) >> 1;
    if (sections[middle].offset <= position) low = middle;
    else high = middle - 1;
  }
  return low;
}

/**
 * The row at one flat position.
 *
 * @param {object} grid - The shape from `buildGrid`.
 * @param {number} position - Flat row position.
 * @returns {{kind: string, section: number, line: number, first: number, span: number,
 *   position: number}|null} `kind` is `header`, `run` or `tail`; `first` is the item at column
 *   zero and `span` the filled cells. A header answers `first` of -1. Null when the position
 *   is on no row.
 */
export function gridRow(grid, position) {
  const at = Math.floor(position);
  if (!grid || !(at >= 0) || at >= grid.total) return null;
  if (at >= grid.tailOffset) {
    const line = at - grid.tailOffset;
    return {
      kind: "tail",
      section: -1,
      line,
      first: line * grid.columns,
      span: gridSpan(grid.tailCount, line, grid.columns),
      position: at,
    };
  }
  const section = sectionAt(grid.sections, at);
  const held = grid.sections[section];
  if (!held) return null;
  if (at === held.offset) {
    return { kind: "header", section, line: -1, first: -1, span: 1, position: at };
  }
  const line = at - held.offset - 1;
  return {
    kind: "run",
    section,
    line,
    first: line * grid.columns,
    span: gridSpan(held.count, line, grid.columns),
    position: at,
  };
}

/**
 * The item at one row and column.
 *
 * @param {object} grid - The shape from `buildGrid`.
 * @param {number} position - Flat row position.
 * @param {number} column - Which cell across, ignored on a header row.
 * @returns {{kind: string, section: number, index: number, line: number, first: number,
 *   span: number, position: number, column: number}|null} The item, its row and its cell, or
 *   null when the cell holds nothing. `index` is `line * columns + column`.
 */
export function gridCell(grid, position, column) {
  const row = gridRow(grid, position);
  if (!row) return null;
  if (row.kind === "header") return { ...row, index: -1, column: 0 };
  const at = Math.floor(column);
  if (!(at >= 0) || at >= row.span) return null;
  return { ...row, index: row.first + at, column: at };
}

/**
 * Where an item sits, which inverts `gridCell`.
 *
 * @param {object} grid - The shape from `buildGrid`.
 * @param {{kind: string, section?: number, index?: number}} item - `header` with a section,
 *   `run` with a section and an index, or `tail` with an index.
 * @returns {{position: number, column: number}|null} Its row and cell, or null when the grid
 *   does not hold it. An item sits on line `Math.floor(index / columns)` at column
 *   `index % columns`, counting from its own run's first row.
 */
export function gridPosition(grid, item) {
  if (!grid || !item) return null;
  if (item.kind === "header") {
    const held = grid.sections[item.section];
    return held ? { position: held.offset, column: 0 } : null;
  }
  if (item.kind === "tail") {
    const index = Math.floor(item.index);
    if (!(index >= 0) || index >= grid.tailCount) return null;
    return {
      position: grid.tailOffset + Math.floor(index / grid.columns),
      column: index % grid.columns,
    };
  }
  if (item.kind !== "run") return null;
  const held = grid.sections[item.section];
  const index = Math.floor(item.index);
  if (!held || !(index >= 0) || index >= held.count) return null;
  return {
    position: held.offset + 1 + Math.floor(index / grid.columns),
    column: index % grid.columns,
  };
}

/**
 * Pixels from one column to the next.
 *
 * @param {number} room - Pixels across the grid has.
 * @param {number} columns - Cells across.
 * @returns {number} The pitch, which the cells share the room in.
 */
export function cellPitch(room, columns) {
  const across = Math.max(1, Math.trunc(columns) || 1);
  return Math.max(1, room) / across;
}

/**
 * The left edge of one column, on a whole pixel.
 *
 * @param {number} left - Left edge of the grid.
 * @param {number} pitch - Pixels from one column to the next.
 * @param {number} column - Which cell across.
 * @returns {number} Its left edge.
 */
export function cellLeft(left, pitch, column) {
  return left + Math.round(column * pitch);
}

/**
 * The width of one column.
 *
 * @param {number} left - Left edge of the grid.
 * @param {number} pitch - Pixels from one column to the next.
 * @param {number} column - Which cell across.
 * @returns {number} Its width, from its own edge to the next one.
 */
export function cellWidth(left, pitch, column) {
  return cellLeft(left, pitch, column + 1) - cellLeft(left, pitch, column);
}

/**
 * Which column a point falls in, which inverts `cellLeft`.
 *
 * @param {number} x - Position across, in the same pixels as `left`.
 * @param {number} left - Left edge of the grid.
 * @param {number} pitch - Pixels from one column to the next.
 * @param {number} columns - Cells across.
 * @returns {number|null} The column, or null when the point is outside the grid.
 */
export function cellColumn(x, left, pitch, columns) {
  const across = Math.max(1, Math.trunc(columns) || 1);
  if (!(pitch > 0) || x < left || x >= cellLeft(left, pitch, across)) return null;
  for (let at = across - 1; at > 0; at -= 1) {
    if (x >= cellLeft(left, pitch, at)) return at;
  }
  return 0;
}
