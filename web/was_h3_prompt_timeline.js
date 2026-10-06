/**
 * The Prompt Timeline: a floating editor for the scenes and assets of MiniMax H3 Conditioning.
 *
 * A window of media, scene settings and tracks. Every edit writes the node's own widgets or a
 * MiniMax H3 Asset node on its chain.
 */

import { api } from "../../scripts/api.js";
import { app } from "../../scripts/app.js";
import { KIND_EXTENSIONS, createAssetBrowser, kindOf as entryKind } from "./interface/asset_browser.js";
import { addButton } from "./interface/decoration.js";
import { createFloatingWindow } from "./interface/floating_window.js";
import {
  ASSET_NODE_ID, appendToChain, chainNumberedSockets, chainOf, placeInChain, removeFromChain,
  wiredInto,
} from "./interface/h3_asset_chain.js";
import {
  AUDIO_REFERENCE, BRIDGING, CONTINUITY, resolvedTransition, CUT_LIKE, EVERY_SEGMENT, FPS, MAX_ROWS, PREVIOUS_SOURCE,
  REFERENCE_FRAMES, REFERENCE_VIDEO, ROW_CONTINUITY, bridgeFrames, durationOf, framesOf, guideLength, resolvedIndex, snapClip, snapOverlap,
  snapOverlapFor, timeline,
} from "./interface/h3_extend.js";
import {
  ROLE_NAMES, TIMELINE_HEIGHT, TRANSITIONS, TRANSITION_TINTS, clockOf, createTimeline, drawnEndOf,
  effectiveTransition, sceneAt, sceneTint, sheetFrameAt, tintKey,
} from "./interface/h3_timeline.js";
import {
  JOBS, answerOf, defaultSystemPrompt, digest, llmSettings, modelSource, queueJob, samplingInputs, storeSettings,
} from "./interface/h3_writer.js";
import { executionId } from "./interface/preview.js";
import { surfaceRatio, watchSurfaceRatio } from "./interface/resolution.js";
import {
  PREVIEW_KEY, drawSheetFrame, ensurePreviewKey, followSegmentPreviews,
} from "./interface/segment_previews.js";
import { closePopupMenu, openPopupMenu } from "./interface/popup_menu.js";
import { withGraphChange } from "./interface/region.js";
import { ROW_PALETTE, rowColour, setRowColour } from "./interface/row_headers.js";
import { joinTicking } from "./interface/switch_board.js";
import {
  CARD, FIELD, FLOATING_PANEL, LABEL, LOOK, MONO, ROW, SANS, SCROLLBARS, SELECTED_FILL,
  button as kitButton, createNotice, el, field, focusRing, heading, numberField as makeNumber,
  pill, select as makeSelect, setDisabled, tabStrip, textBox,
} from "./interface/window_kit.js";

const EXT_NAME = "WASNodeSuite.H3PromptTimelineUI";
const LOG_NAME = EXT_NAME;
const SETTING_ID = "WAS.H3.ShowPromptTimelineButton";

const NODE_ID = "WASMiniMaxH3Conditioning";

const BUTTON_NAME = "was_h3_prompt_timeline_button";
const BUTTON_LABEL = "Open Prompt Timeline";
const WINDOW_NAME = "Prompt Timeline";

// The window's size when first opened, and the least it may be dragged to, in CSS pixels.
const WINDOW_WIDTH = 1560;
const WINDOW_HEIGHT = 860;
const WINDOW_MIN_WIDTH = 1100;
const WINDOW_MIN_HEIGHT = 600;

// The media bin's width, and its tiles.
const MEDIA_WIDTH = 330;
const TILE_WIDTH = 92;
const TILE_HEIGHT = 66;

// The preview monitor's width, and how long a hand-moved playhead stays put during a run.
const MONITOR_WIDTH = 420;
const FOLLOW_AFTER_MS = 4000;

// The narrowest each column is dragged to, the dividers between them, and where widths are kept.
const MEDIA_MIN = 200;
const INSPECTOR_MIN = 300;
const MONITOR_MIN = 240;
const DIVIDER_WIDTH = 7;
const PANES_KEY = "was.h3.prompt_timeline.columns";

// Pixels a side a thumbnail is asked for.
const THUMB_EDGE = 192;

// Pixels a side the thumbnail route draws an inspected file at, where ComfyUI's view cannot serve it.
const INSPECT_EDGE = 512;

// How long a frame is waited for when the finished video seeks to it, in milliseconds.
const STILL_WAIT_MS = 3000;

// The longest scene Write asks for, in seconds.
const WRITER_LONGEST = 12;

// The look Write starts from.
const WRITER_STYLE = "a 2D-animated cartoon with clean inked outlines and flat cel-painted colours";

// What a new scene holds until it is written.
const NEW_SCENE_SECONDS = 5.2;
const NEW_SCENE_OVERLAP = 22;
const NEW_SCENE_PROMPT = "New scene";

// How long typing is left alone before a prompt edit is written to the node.
const TYPING_IDLE_MS = 700;

// The parts an asset may play, in the order the role menu lists them.
const ROLES = [
  "first frame", "last frame", "keyframe", "reference picture", "reference clip", "reference audio",
];
const PINNING = new Set(["first frame", "last frame", "keyframe"]);

// Which kinds of file each role takes.
const ROLE_KINDS = {
  "first frame": ["picture", "clip"],
  "last frame": ["picture", "clip"],
  "keyframe": ["picture", "clip", "sound"],
  "reference picture": ["picture", "clip"],
  "reference clip": ["clip"],
  "reference audio": ["sound", "clip"],
};

// Most references of each kind one prompt carries, and the word its tag uses.
const MOST = { "reference picture": 9, "reference clip": 3, "reference audio": 3 };
const TAG_WORDS = { "reference picture": "Picture", "reference clip": "Video", "reference audio": "Audio" };

/**
 * A symbol drawn in the text's colour rather than as an emoji.
 *
 * @param {string} symbol - The symbol, as `⏮`.
 * @returns {string} The symbol and its text-style selector.
 */
function textGlyph(symbol) {
  return `${symbol}\uFE0E`;
}

// Files a run's outputs name that the Final view plays.
const VIDEO_FILE = /\.(mp4|webm|mov|mkv|m4v)$/i;

// Both canvas sides are a multiple of this.
const CANVAS_MULTIPLE = 32;

// The menu entry on the asset node that reads its sockets instead of a file.
const WIRED = "(wired input)";

// The menu entry on the asset node that reads a frame of the video the run is making.
const MOMENT = "(the video being made)";

// Inputs on the conditioning node whose wiring the window shows.
const WATCHED_SOCKETS = Object.freeze([
  "model_fl2va", "model_ref2va", "audio_vae", "first_frame", "last_frame", "images",
  ...Array.from({ length: 9 }, (unused, at) => `ref_image_${at + 1}`),
  ...Array.from({ length: 3 }, (unused, at) => [`ref_video_${at + 1}`, `ref_video_audio_${at + 1}`]).flat(),
  ...Array.from({ length: 3 }, (unused, at) => `ref_audio_${at + 1}`),
]);

// How each shared text choice reads, and each mode.
const WRAP_NAMES = {
  both: "Header and footer", "header only": "Header only", "footer only": "Footer only",
  neither: "Neither",
};
const MODE_HINTS = {
  t2va: "Text only. Pinned frames and references come from the assets.",
  i2va: "Opens on the picture wired into first_frame.",
  fl2va: "Opens on first_frame and closes on last_frame.",
  fl2va_batched: "Each scene runs between two neighbouring pictures of images.",
  ref2va: "Every scene is built on the ref inputs, named <Picture 1>, <Video 1>, <Audio 1>.",
};
const MODEL_NAMES = { auto: "Automatic", fl2va: "fl2va", ref2va: "ref2va" };

// The length line of a prompt, as `duration_seconds: 8`.
const DURATION_LINE = /^[ \t]*duration_seconds:[ \t]*(\d+(?:\.\d+)?)[ \t]*$/m;

// Seconds a stated length may differ from the scene's before it is worth saying.
const STATED_TOLERANCE = 0.05;

// One window per conditioning node.
const WINDOWS = new WeakMap();

// The window's icon: three tracks of clips, in a 24 unit box.
const WINDOW_ICON = ["M2 4h11v4H2z", "M15 4h7v4h-7z", "M2 10h6v4H2z", "M10 10h12v4H10z", "M2 16h14v4H2z", "M18 16h4v4h-4z"];

/**
 * The workflow open on the canvas.
 *
 * @returns {object|null} The workflow store's active workflow, or null where there is none.
 */
function activeWorkflow() {
  return app?.extensionManager?.workflow?.activeWorkflow ?? null;
}

/**
 * Read whether the button is drawn at all.
 *
 * @returns {boolean} True while the setting is on or cannot be read.
 */
function enabled() {
  try {
    const value = app?.extensionManager?.setting?.get?.(SETTING_ID);
    if (typeof value === "boolean") return value;
    const legacy = app?.ui?.settings?.getSettingValue?.(SETTING_ID, true);
    if (typeof legacy === "boolean") return legacy;
  } catch (error) {
    console.error(`[${EXT_NAME}] Failed to read ${SETTING_ID}:`, error);
  }
  return true;
}

/**
 * One of a node's widgets, by name.
 *
 * @param {object} node - The node.
 * @param {string} name - The widget's name.
 * @returns {object|undefined} The widget.
 */
function widgetOf(node, name) {
  return node?.widgets?.find((widget) => widget?.name === name);
}

/**
 * The value one of a node's widgets holds.
 *
 * @param {object} node - The node.
 * @param {string} name - The widget's name.
 * @param {*} [fallback] - What to answer where the node carries no such widget.
 * @returns {*} The value.
 */
function valueOf(node, name, fallback) {
  const widget = widgetOf(node, name);
  return widget ? widget.value : fallback;
}

/**
 * The options a combo widget offers.
 *
 * @param {object} node - The node.
 * @param {string} name - The widget's name.
 * @returns {string[]} The options, empty where the widget is not a combo.
 */
function optionsOf(node, name) {
  const values = widgetOf(node, name)?.options?.values;
  return Array.isArray(values) ? values.map(String) : [];
}

/**
 * Put a value on a widget and run its callback.
 *
 * @param {object} node - The node.
 * @param {string} name - The widget's name.
 * @param {*} value - The value.
 * @returns {boolean} Whether the widget was found and changed.
 */
function write(node, name, value) {
  const widget = widgetOf(node, name);
  if (!widget || widget.value === value) return false;
  widget.value = value;
  try {
    widget.callback?.(value, app.canvas, node);
  } catch (error) {
    console.error(`[${LOG_NAME}] ${name}'s callback failed:`, error);
  }
  return true;
}

/**
 * Make one graph change, as one undo entry, and redraw.
 *
 * @param {object} node - The node the change belongs to.
 * @param {() => *} change - The writes.
 * @returns {*} What the writes answered, or undefined when they failed.
 */
function commit(node, change) {
  let answer;
  withGraphChange(() => {
    try {
      answer = change();
    } catch (error) {
      console.error(`[${LOG_NAME}] A change failed:`, error);
    }
  });
  try {
    node.__was_refold?.();
  } catch (error) {
    console.error(`[${LOG_NAME}] Failed to refold the rows:`, error);
  }
  node.setDirtyCanvas?.(true, true);
  node.graph?.setDirtyCanvas?.(true, true);
  return answer;
}

/**
 * The kind of file a label names.
 *
 * @param {string} label - A file label, as `cast/alice.png [input]`.
 * @returns {string} `picture`, `clip`, `sound`, or an empty string.
 */
function kindOf(label) {
  return entryKind({ label: String(label ?? "") }) ?? "";
}

/**
 * The file name a label carries, without its folder or tag.
 *
 * @param {string} label - A file label.
 * @returns {string} The name.
 */
function baseName(label) {
  const bare = String(label ?? "").replace(/\s\[[^\]]*\]$/, "");
  return bare.split(/[\\/]/).pop() || bare;
}

/**
 * The thumbnail route for a label.
 *
 * @param {string} label - A file label.
 * @returns {string} The URL.
 */
function thumbnailUrl(label) {
  const query = new URLSearchParams({ label, edge: String(THUMB_EDGE) });
  return api.apiURL(`/was/interface/api/file_thumbnail?${query.toString()}`);
}

/**
 * What the monitor plays for a file, at full size.
 *
 * @param {string} label - A file label, as `cast/alice.png [input]`.
 * @param {string} kind - `picture`, `clip` or `sound`.
 * @returns {{kind: string, url: string}} ComfyUI's own view of a file under input, output or
 *   temp; elsewhere the largest picture the thumbnail route draws, and no URL for a sound.
 */
function inspectedMedia(label, kind) {
  const found = /^(.*) \[(input|output|temp)\]$/.exec(String(label ?? ""));
  if (found) {
    const cut = found[1].lastIndexOf("/");
    const query = new URLSearchParams({
      filename: found[1].slice(cut + 1), subfolder: cut >= 0 ? found[1].slice(0, cut) : "", type: found[2],
    });
    return { kind: kind || "picture", url: api.apiURL(`/view?${query.toString()}`) };
  }
  if (kind === "sound") return { kind, url: "" };
  const query = new URLSearchParams({ label, edge: String(INSPECT_EDGE) });
  return { kind: "picture", url: api.apiURL(`/was/interface/api/file_thumbnail?${query.toString()}`) };
}

/**
 * The first line of a prompt that reads as a title.
 *
 * @param {string} prompt - The scene's prompt.
 * @returns {string} The line, shortened.
 */
function titleOf(prompt) {
  const lines = String(prompt ?? "").split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  // A key line, as `id="x001"` or `duration_seconds: 4.5`, names nothing a person reads.
  const keyed = (line) => /^[\w.-]+\s*[:=]/.test(line);
  const prose = lines.filter((line) => !keyed(line));
  const plain = prose.find((line) => line.split(/\s+/).length >= 3) ?? prose[0] ?? lines[0] ?? "";
  const bare = plain.replace(/^\[[^\]]{1,24}\]\s*/, "");
  return bare.length > 80 ? `${bare.slice(0, 77)}…` : bare;
}

/**
 * The scenes a node holds, read from its row widgets.
 *
 * @param {object} node - The conditioning node.
 * @returns {object[]} One scene per row carrying a prompt, in row order.
 */
function readScenes(node) {
  const scenes = [];
  for (let row = 1; row <= MAX_ROWS; row += 1) {
    const prompt = valueOf(node, `prompt_${row}`, "");
    if (typeof prompt !== "string" || !prompt.trim()) continue;
    const colour = rowColour(node, row);
    scenes.push({
      row,
      prompt,
      seconds: Number(valueOf(node, `duration_${row}`, NEW_SCENE_SECONDS)) || NEW_SCENE_SECONDS,
      overlap: Number(valueOf(node, `overlap_${row}`, NEW_SCENE_OVERLAP)) || 0,
      continuity: String(valueOf(node, `continuity_${row}`, CONTINUITY[0]) ?? CONTINUITY[0]),
      source: Number(valueOf(node, `source_${row}`, PREVIOUS_SOURCE)),
      wrap: String(valueOf(node, `header_footer_${row}`, "both") ?? "both"),
      model: String(valueOf(node, `model_${row}`, "auto") ?? "auto"),
      hold: Math.max(0, Math.min(1, Number(valueOf(node, `strength_${row}`, 1) ?? 1))),
      sound: String(valueOf(node, `sound_${row}`, "auto") ?? "auto"),
      seed: Math.max(0, Math.round(Number(valueOf(node, `seed_${row}`, 0) ?? 0)) || 0),
      title: titleOf(prompt),
      colourName: colour,
      colour: colour ? ROW_PALETTE[colour] : null,
      warning: "",
    });
  }
  return scenes;
}

/**
 * The assets on a node's chain, read from the asset nodes behind its `assets` input.
 *
 * @param {object} node - The conditioning node.
 * @returns {object[]} One entry per asset node, upstream first.
 */
function readAssets(node) {
  return chainOf(node).map((origin, place) => {
    const file = String(valueOf(origin, "file", WIRED) ?? WIRED);
    const moment = file === MOMENT;
    const fromFile = Boolean(file) && file !== WIRED && !moment;
    let kind = fromFile ? kindOf(file) : moment ? "picture" : "";
    let name = fromFile ? baseName(file) : moment ? `frame ${Number(valueOf(origin, "frame", 0)) || 0} of the video` : "";
    if (!fromFile && !moment) {
      if (wiredInto(origin, "video")) [kind, name] = ["clip", "wired video"];
      else if (wiredInto(origin, "image")) [kind, name] = ["picture", "wired image"];
      else if (wiredInto(origin, "audio")) [kind, name] = ["sound", "wired audio"];
      else name = "nothing chosen";
    }
    const seconds = Number(valueOf(origin, "clip_seconds", 0)) || 0;
    return {
      id: origin.id,
      node: origin,
      place,
      segment: Number(valueOf(origin, "segment", 1)) || 0,
      role: String(valueOf(origin, "role", "first frame")),
      frame: Number(valueOf(origin, "frame", 0)) || 0,
      file,
      kind,
      name,
      thumb: fromFile && kind !== "sound" ? thumbnailUrl(file) : "",
      moment,
      frames: kind === "clip" ? (seconds > 0 ? framesOf(seconds) : 2) : 1,
      tag: "",
    };
  });
}

/**
 * The references a `ref2va` node's own sockets put ahead of every asset.
 *
 * @param {object} node - The conditioning node.
 * @returns {{pictures: string[], videos: Array<{name: string, sound: boolean}>, sounds: string[]}}
 *   The wired sockets, in the order the prompt numbers them. Empty outside `ref2va`.
 */
function socketReferences(node) {
  const found = { pictures: [], videos: [], sounds: [] };
  if (String(valueOf(node, "mode", "t2va")) !== "ref2va") return found;
  for (let slot = 1; slot <= 9; slot += 1) {
    if (wiredInto(node, `ref_image_${slot}`)) found.pictures.push(`ref_image_${slot}`);
  }
  for (let slot = 1; slot <= 3; slot += 1) {
    if (!wiredInto(node, `ref_video_${slot}`)) continue;
    found.videos.push({ name: `ref_video_${slot}`, sound: wiredInto(node, `ref_video_audio_${slot}`) });
  }
  for (let slot = 1; slot <= 3; slot += 1) {
    if (wiredInto(node, `ref_audio_${slot}`)) found.sounds.push(`ref_audio_${slot}`);
  }
  return found;
}

/**
 * Give every reference asset the tag its scene's prompt names it by, sockets numbered first.
 *
 * @param {object[]} assets - The assets, from `readAssets`.
 * @param {number} count - How many scenes there are.
 * @param {object} sockets - The node's own references, from `socketReferences`.
 * @returns {void}
 */
function tagReferences(assets, count, sockets) {
  const shared = assets.filter((asset) => asset.segment === EVERY_SEGMENT);
  for (let segment = 1; segment <= Math.max(1, count); segment += 1) {
    const carried = [...shared, ...assets.filter((asset) => asset.segment === segment)];
    // Moments of the video join a scene's prompt after its other pictures.
    const of = (role) => [
      ...carried.filter((asset) => asset.role === role && !asset.moment),
      ...carried.filter((asset) => asset.role === role && asset.moment),
    ];
    const clips = of("reference clip");
    // A clip's soundtrack takes an `<Audio N>` ahead of every sound.
    let heard = sockets.videos.filter((video) => video.sound).length;
    const tags = new Map();
    of("reference picture").forEach((asset, at) => tags.set(asset, `<Picture ${sockets.pictures.length + at + 1}>`));
    clips.forEach((asset, at) => {
      const sound = asset.kind === "clip" ? ` with <Audio ${(heard += 1)}>` : "";
      tags.set(asset, `<Video ${sockets.videos.length + at + 1}>${sound}`);
    });
    heard += sockets.sounds.length;
    of("reference audio").forEach((asset) => tags.set(asset, `<Audio ${(heard += 1)}>`));
    for (const [asset, tag] of tags) {
      if (asset.segment === segment || !asset.tag) asset.tag = tag;
    }
  }
}

/**
 * The canvas MiniMax H3 Conditioning samples at, as it works it out.
 *
 * @param {number} megapixels - Millions of pixels aimed for.
 * @param {number} width - Width in pixels, 0 to work it out.
 * @param {number} height - Height in pixels, 0 to work it out.
 * @param {string} aspect - The aspect ratio choice, as `16:9`.
 * @returns {number[]|null} `[width, height]`, or null where it comes from a picture or is unset.
 */
function canvasSize(megapixels, width, height, aspect) {
  const snap = (value) => Math.max(CANVAS_MULTIPLE, Math.round(Number(value) / CANVAS_MULTIPLE) * CANVAS_MULTIPLE);
  const wide = Math.max(0, Math.round(Number(width) || 0));
  const high = Math.max(0, Math.round(Number(height) || 0));
  if (wide && high) return [snap(wide), snap(high)];
  const area = Math.max(0, Number(megapixels) || 0) * 1e6;
  if (area <= 0) return null;
  if (wide) return [snap(wide), snap(area / snap(wide))];
  if (high) return [snap(area / snap(high)), snap(high)];
  const parts = String(aspect || "").split(":").map(Number);
  if (parts.length !== 2 || !(parts[0] > 0) || !(parts[1] > 0)) return null;
  const ratio = parts[0] / parts[1];
  const across = snap(Math.sqrt(area * ratio));
  return [across, snap(across / ratio)];
}

/**
 * Write a list of scenes onto a node's rows, blanking the rows past them.
 *
 * @param {object} node - The conditioning node.
 * @param {object[]} scenes - The scenes, in the order they are to sit.
 * @returns {void}
 */
function writeScenes(node, scenes) {
  const colours = {};
  scenes.forEach((scene, index) => {
    const row = index + 1;
    write(node, `prompt_${row}`, scene.prompt);
    write(node, `duration_${row}`, scene.seconds);
    write(node, `overlap_${row}`, scene.overlap);
    write(node, `continuity_${row}`, scene.continuity);
    write(node, `source_${row}`, scene.source);
    write(node, `header_footer_${row}`, scene.wrap);
    write(node, `model_${row}`, scene.model);
    write(node, `strength_${row}`, scene.hold ?? 1);
    write(node, `sound_${row}`, scene.sound ?? "auto");
    write(node, `seed_${row}`, scene.seed ?? 0);
    if (scene.colourName) colours[String(row)] = scene.colourName;
  });
  for (let row = scenes.length + 1; row <= MAX_ROWS; row += 1) {
    write(node, `prompt_${row}`, "");
  }
  node.properties ??= {};
  if (Object.keys(colours).length) node.properties.was_row_colours = colours;
  else delete node.properties.was_row_colours;
}

/**
 * Renumber the assets after scenes moved.
 *
 * @param {object[]} assets - The assets, from `readAssets`.
 * @param {(segment: number) => number|null} map - Old scene to new, null to drop the asset.
 * @returns {void}
 */
function remapAssets(assets, map) {
  for (const asset of assets) {
    if (asset.segment === EVERY_SEGMENT) continue;
    const to = map(asset.segment);
    if (to === null) removeFromChain(asset.node);
    else if (to !== asset.segment) write(asset.node, "segment", to);
  }
}

/**
 * The scene a source value names, as an index from 0.
 *
 * @param {number} source - The source widget's value.
 * @param {number} index - The scene's index, from 0.
 * @returns {number} The resolved index, or -1 where it names nothing before this scene.
 */
function sourceIndex(source, index) {
  const picked = source === 0 ? index - 1 : source < 0 ? index + source : source - 1;
  return picked >= 0 && picked < index ? picked : -1;
}

// ------------------------------------------------------------------------------ controls

// Button kinds the window's own call sites name, as the kit draws them.
const BUTTON_KINDS = { primary: "primary", secondary: "plain", ghost: "ghost", danger: "danger" };

/**
 * A button drawn by the kit.
 *
 * @param {string} label - The text on it.
 * @param {(event: MouseEvent) => void} onPress - What it does.
 * @param {object} [options] - `{hint, kind}`, `kind` a kit kind or `primary`, `secondary`,
 *   `ghost` or `danger`.
 * @returns {HTMLButtonElement} The button.
 */
function makeButton(label, onPress, { hint = "", kind = "secondary" } = {}) {
  return kitButton(label, onPress, { hint, kind: BUTTON_KINDS[kind] ?? kind, logName: LOG_NAME });
}

/**
 * Put a value on a control unless it is being typed in.
 *
 * @param {HTMLElement} control - The control.
 * @param {*} value - The value.
 * @returns {void}
 */
function show(control, value) {
  if (document.activeElement === control) return;
  const text = value === null || value === undefined ? "" : String(value);
  if (control.value !== text) control.value = text;
}

/**
 * Whether an event was aimed at something that takes typing.
 *
 * @param {Event} event - The event.
 * @returns {boolean} True for an input, a textarea, a select or an editable element.
 */
function typingInto(event) {
  const target = event.target;
  return ["INPUT", "TEXTAREA", "SELECT"].includes(target?.tagName) || target?.isContentEditable === true;
}

// ------------------------------------------------------------------------------ the window

/**
 * Build the Prompt Timeline for one conditioning node.
 *
 * @param {object} first - The conditioning node.
 * @returns {object} `{open, close, toggle, nodeRemoved, dispose}`.
 */
function createPromptTimeline(first) {
  let node = first;
  let selected = null;
  let selectedAsset = null;
  // The asset the screen shows full size, `{id, file, kind, name, role, tag, segment}`, or null.
  let inspected = null;
  let playhead = 0;
  let tab = "scene";
  let dragging = null;
  let showMore = false;
  let signature = "";
  let state = { scenes: [], assets: [], layout: [], warnings: [] };
  let typingTimer = 0;
  let renderPending = false;
  let disposed = false;
  let release = null;
  const cards = new Map();

  const workflow = activeWorkflow();
  const win = createFloatingWindow({
    title: WINDOW_NAME,
    badge: node.title || "MiniMax H3 Conditioning",
    icon: WINDOW_ICON,
    width: WINDOW_WIDTH,
    height: WINDOW_HEIGHT,
    minWidth: WINDOW_MIN_WIDTH,
    minHeight: WINDOW_MIN_HEIGHT,
    storageKey: "was.h3.prompt_timeline",
    logName: LOG_NAME,
    onClose: () => close(),
  });
  win.body.style.position = "relative";
  win.body.style.background = LOOK.body;

  // -------------------------------------------------------------------- timeline bar
  // The tools that act on the tracks, placed directly above them.
  const toolbar = el("div", "display:flex;gap:8px;align-items:center;padding:0 10px;flex:0 0 auto;"
    + `min-height:40px;border-top:1px solid ${LOOK.border};background:${LOOK.surface};min-width:0`);
  const addButtonEl = makeButton("+  Add scene", () => addScene(), {
    kind: "go", hint: "Add a scene at the end of the video (N)",
  });
  const writeButton = makeButton("Write scenes…", () => openWriter(), {
    kind: "tool", hint: "Write the video from an idea",
  });
  const planButton = makeButton("Plan transitions", () => togglePlan(), {
    kind: "tool", hint: "Pick every cut and carry",
  });
  const duplicateButton = makeButton("Duplicate", () => {
    if (selected !== null) duplicateScene(selected);
  }, { kind: "tool", hint: "Copy the chosen scene after itself (D)" });
  const deleteButton = makeButton("Delete", () => removeSelection(), {
    kind: "danger", hint: "Delete the chosen scene, or the chosen asset (Delete)",
  });
  const summary = el("div", "display:flex;align-items:center;gap:8px;margin-left:6px;min-width:0;"
    + "flex:0 1 auto;white-space:nowrap;overflow:hidden");
  const legend = el("div", "display:flex;align-items:center;gap:14px;min-width:0;flex:0 1 auto;overflow:hidden;"
    + `white-space:nowrap;font:11px ${SANS};color:${LOOK.muted}`);
  const issuesButton = makeButton("", () => toggleIssues(), { kind: "tool", hint: "What needs fixing before a run" });
  const zoomOut = makeButton("−", () => strip.zoomBy(1 / 1.3), { kind: "tool", hint: "Zoom out (-)" });
  const fitButton = makeButton("Fit", () => strip.fit(), { kind: "tool", hint: "Fit the whole video (F)" });
  const zoomIn = makeButton("+", () => strip.zoomBy(1.3), { kind: "tool", hint: "Zoom in (+)" });
  const helpButton = makeButton("?", () => toggleHelp(), { kind: "tool", hint: "How the Prompt Timeline works" });
  for (const narrow of [zoomOut, zoomIn, helpButton]) narrow.style.minWidth = "28px";
  toolbar.append(
    addButtonEl, writeButton, planButton, duplicateButton, deleteButton, summary,
    el("span", "flex:1 1 auto"), legend,
    el("span", `flex:0 0 1px;height:18px;background:${LOOK.border}`),
    zoomOut, fitButton, zoomIn, helpButton,
  );

  // -------------------------------------------------------------------- popovers
  const popover = el("div", "position:absolute;z-index:3;display:none;"
    + `max-width:440px;overflow:auto;padding:12px 14px;${FLOATING_PANEL};${SCROLLBARS}`);
  let popoverFor = "";
  win.body.appendChild(popover);

  /**
   * Show or hide the list of what needs fixing.
   *
   * @returns {void}
   */
  function toggleIssues() {
    popoverFor = popoverFor === "issues" ? "" : "issues";
    renderPopover();
  }

  /**
   * Show or hide the guide.
   *
   * @returns {void}
   */
  function toggleHelp() {
    popoverFor = popoverFor === "help" ? "" : "help";
    renderPopover();
  }

  // The language model jobs, by kind: `{promptId, status, step, steps, result, error, ...}`, or null.
  const jobs = { write: null, rewrite: null, plan: null };
  const writerDraft = { idea: "", scenes: 0, seconds: 8, style: WRITER_STYLE };
  // Where the Write form is drawn: the dialog's panel, the empty Scene tab's card, or nowhere.
  let writerHost = null;
  // The scene Rewrite works on, by row, and the directions typed for it.
  let rewriteDraft = { row: 0, directions: "" };
  let rewriteAnchor = null;
  // The nodes' own rules, read from their definition.
  let systemText = "";
  defaultSystemPrompt().then((text) => {
    systemText = text;
    if (!disposed && tab === "llm") refresh(true);
  });

  /**
   * Whether a job is queued or running.
   *
   * @param {object|null} job - A job.
   * @returns {boolean} True until it answers, fails or stops.
   */
  function busy(job) {
    return Boolean(job && job.result == null && !job.error);
  }

  /**
   * One line saying where a job is.
   *
   * @param {object|null} job - A job.
   * @param {string} doing - What it does, as `Writing`.
   * @returns {string} The line, empty for no job.
   */
  function statusOf(job, doing) {
    if (!job) return "";
    if (job.error) return job.error;
    if (job.status === "queued") return "Queued behind the runs ahead of it";
    return job.steps ? `${doing} · ${job.step}/${job.steps}` : `${doing}…`;
  }

  /**
   * Store language model settings on the node, as one undo entry.
   *
   * @param {object} patch - The settings that change.
   * @param {boolean} [redraw] - Whether to draw the inspector again.
   * @returns {void}
   */
  function saveSettings(patch, redraw = false) {
    commit(node, () => storeSettings(node, patch));
    if (redraw) refresh(true);
  }

  /**
   * The system prompt input a job takes, left out where the node's own default applies.
   *
   * @param {object} settings - From `llmSettings`.
   * @returns {object} `{system_prompt}` or nothing.
   */
  function systemInput(settings) {
    return settings.system === null || settings.system === undefined ? {} : { system_prompt: String(settings.system) };
  }

  /**
   * Queue a job and follow it.
   *
   * @param {string} kind - `write`, `rewrite` or `plan`.
   * @param {object} inputs - The job node's inputs.
   * @param {object} [options] - Passed to `queueJob`.
   * @param {object} [extra] - Kept on the job, as the rows it applies to.
   * @returns {Promise<void>}
   */
  async function startJob(kind, inputs, options = {}, extra = {}) {
    jobs[kind] = { promptId: null, status: "queued", step: 0, steps: 0, result: null, error: null, ...extra };
    repaintJob(kind);
    try {
      jobs[kind].promptId = await queueJob(node, kind, inputs, options);
    } catch (error) {
      console.error(`[${LOG_NAME}] ${JOBS[kind].title} could not be queued:`, error);
      jobs[kind] = { ...jobs[kind], status: "error", error: `Not queued: ${error.message ?? error}` };
    }
    repaintJob(kind);
  }

  /**
   * Stop a job, queued or running.
   *
   * @param {string} kind - `write`, `rewrite` or `plan`.
   * @returns {Promise<void>}
   */
  async function stopJob(kind) {
    const job = jobs[kind];
    try {
      if (job?.status === "running") await api.interrupt(job.promptId);
      else if (job?.promptId) await api.deleteItem("queue", job.promptId);
    } catch (error) {
      console.error(`[${LOG_NAME}] ${JOBS[kind].title} could not be stopped:`, error);
    }
    jobs[kind] = null;
    repaintJob(kind);
  }

  /**
   * Draw whatever shows a job again: its form or popover, and the button that started it.
   *
   * @param {string} kind - `write`, `rewrite` or `plan`.
   * @returns {void}
   */
  function repaintJob(kind) {
    const job = jobs[kind];
    if (kind === "write") {
      writeButton.textContent = busy(job) ? statusOf(job, "Writing") : "Write scenes…";
      paintWriter();
    } else if (kind === "plan") {
      planButton.textContent = busy(job) ? "Planning…" : "Plan transitions";
      if (popoverFor === "plan") renderPopover();
    } else {
      if (rewriteAnchor?.isConnected) rewriteAnchor.textContent = busy(job) ? "Rewriting…" : rewriteAnchor.dataset.idle;
      if (popoverFor === "rewrite") renderPopover();
    }
  }

  // -------------------------------------------------------------------- write dialog
  const dialog = el("div", "position:absolute;inset:0;z-index:4;display:none;align-items:center;justify-content:center;"
    + "background:rgba(0, 0, 0, 0.45)");
  const dialogPanel = el("div", "width:min(560px, calc(100% - 32px));max-height:calc(100% - 48px);overflow:auto;"
    + `padding:18px 20px;${FLOATING_PANEL};${SCROLLBARS}`);
  dialog.appendChild(dialogPanel);
  dialog.addEventListener("pointerdown", (event) => {
    if (event.target === dialog) closeWriter();
  });
  win.body.appendChild(dialog);

  /**
   * Open Write in its dialog.
   *
   * @returns {void}
   */
  function openWriter() {
    popoverFor = "";
    renderPopover();
    dialog.style.display = "flex";
    writerHost = dialogPanel;
    paintWriter();
  }

  /**
   * Close the Write dialog. A job it started carries on.
   *
   * @returns {void}
   */
  function closeWriter() {
    dialog.style.display = "none";
    dialogPanel.replaceChildren();
    if (writerHost === dialogPanel) writerHost = null;
  }

  /**
   * Draw the Write form wherever it is shown.
   *
   * @returns {void}
   */
  function paintWriter() {
    if (!writerHost?.isConnected) {
      writerHost = null;
      return;
    }
    writerHost.replaceChildren();
    renderWriter(writerHost, writerHost === dialogPanel);
  }

  /**
   * The chain's reference pictures, one per file, in chain order.
   *
   * @returns {object[]} Asset entries.
   */
  function castPictures() {
    const seen = new Set();
    return state.assets.filter((asset) => {
      const fromFile = /\[[^\]]+\]$/.test(String(asset.file ?? ""));
      if (asset.role !== "reference picture" || asset.moment || !fromFile || seen.has(asset.name)) return false;
      seen.add(asset.name);
      return true;
    });
  }

  /**
   * One muted line of facts.
   *
   * @param {string} text - The facts.
   * @param {string} [tip] - The hover.
   * @returns {HTMLElement} The line.
   */
  function factLine(text, tip = "") {
    const line = el("div", `font:12px/1.5 ${SANS};color:${LOOK.muted};font-variant-numeric:tabular-nums`, text);
    if (tip) line.title = tip;
    return line;
  }

  /**
   * Draw Write: the description, the count and length, the job's progress and what it wrote.
   *
   * @param {HTMLElement} host - Where to draw it.
   * @param {boolean} inDialog - Whether it is drawn in the dialog, which has a close button.
   * @returns {void}
   */
  function renderWriter(host, inDialog) {
    const head = el("div", "display:flex;align-items:center;gap:10px;margin-bottom:12px");
    head.appendChild(el("div", `font:600 15px ${SANS};color:${LOOK.text}`, "Write the video"));
    head.appendChild(el("span", "flex:1 1 auto"));
    if (inDialog) head.appendChild(makeButton("✕", () => closeWriter(), { kind: "tool", hint: "Close (Esc)" }));
    host.appendChild(head);
    const source = modelSource(node);
    if (!source.wired) {
      host.appendChild(factLine("Wire a language model into vlm_clip on MiniMax H3 Conditioning, as Load CLIP "
        + "with qwen3vl_8b_fp8_scaled.safetensors."));
      return;
    }
    const job = jobs.write;
    const settings = llmSettings(node);
    if (job?.result) {
      const result = job.result;
      host.appendChild(el("div", `font:13px ${SANS};color:${LOOK.text};margin-bottom:8px`,
        `${result.scenes.length} scenes · ${result.scenes.reduce((sum, scene) => sum + Number(scene.seconds || 0), 0).toFixed(1)}s`));
      const list = el("div", "display:flex;flex-direction:column;gap:4px;margin-bottom:10px");
      result.scenes.forEach((scene, index) => {
        const row = el("div", `display:flex;gap:8px;align-items:baseline;font:12px ${SANS};color:${LOOK.muted};min-width:0`);
        row.title = scene.summary || scene.title;
        row.appendChild(el("span", `flex:0 0 22px;color:${LOOK.text}`, `${index + 1}.`));
        row.appendChild(el("span", `flex:1 1 auto;color:${LOOK.text};white-space:nowrap;overflow:hidden;text-overflow:ellipsis`, scene.title));
        row.appendChild(el("span", "flex:0 0 auto", `${Number(scene.seconds).toFixed(1)}s`));
        const join = el("span", "flex:0 0 150px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis",
          index === 0 ? "Opens the video" : (TRANSITIONS[scene.continuity]?.name ?? scene.continuity));
        if (scene.why) join.title = scene.why;
        row.appendChild(join);
        list.appendChild(row);
      });
      host.appendChild(list);
      const words = (text) => String(text ?? "").split(/\s+/).filter(Boolean).length;
      host.appendChild(factLine(`Header · ${result.header ? `${words(result.header)} words` : "empty"}   ·   `
        + `Footer · ${result.footer ? `${words(result.footer)} words` : "empty"}   ·   `
        + `Cast · ${result.cast.length} picture${result.cast.length === 1 ? "" : "s"} read`));
      host.appendChild(factLine("Apply replaces every scene, the header, the footer and the reference pictures. "
        + "Pinned frames stay. Ctrl+Z undoes it."));
      const buttons = el("div", "display:flex;gap:8px;justify-content:flex-end;margin-top:12px");
      buttons.append(
        makeButton("Discard", () => {
          jobs.write = null;
          repaintJob("write");
        }, { kind: "tool", hint: "Forget what was written" }),
        makeButton("Apply", () => applyWritten(result), { kind: "primary", hint: "Replace the video with these" }),
      );
      host.appendChild(buttons);
      return;
    }

    const idea = textBox(6, "min-height:120px");
    idea.placeholder = "Who is in it, where it goes, what happens, who speaks which language";
    idea.value = writerDraft.idea;
    host.appendChild(field("Describe the video", idea, "", "Names, languages and beats carry into the scenes."));
    const grid = el("div", "display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-top:12px");
    const count = makeNumber(writerDraft.scenes || state.scenes.length || 8, { min: 1, max: MAX_ROWS, step: 1 }, "scenes");
    count.input.addEventListener("change", () => {
      writerDraft.scenes = Math.max(1, Math.min(MAX_ROWS, Math.round(Number(count.input.value) || 8)));
    });
    const seconds = makeNumber(writerDraft.seconds, { min: 5.2, max: WRITER_LONGEST, step: 0.1 }, "s");
    seconds.input.addEventListener("change", () => {
      writerDraft.seconds = Math.max(5.2, Math.min(WRITER_LONGEST, Number(seconds.input.value) || 8));
    });
    const planning = makeSelect([["true", "Planned"], ["false", "All cuts"]], String(Boolean(settings.plan)));
    planning.addEventListener("change", () => saveSettings({ plan: planning.value === "true" }));
    grid.append(
      field("Scenes", count.box),
      field("Each about", seconds.box),
      field("Transitions", planning, "", "Planned reads the scenes back and picks each cut or carry."),
    );
    host.appendChild(grid);
    const look = textBox(2, "margin-top:2px");
    look.value = writerDraft.style;
    look.addEventListener("input", () => {
      writerDraft.style = look.value;
    });
    const lookBox = el("div", "margin-top:12px");
    lookBox.appendChild(field("Look", look, "", "Finishes the sentence: The target video is ..."));
    host.appendChild(lookBox);
    const cast = castPictures();
    host.appendChild(el("div", "height:10px"));
    host.appendChild(factLine(`Cast · ${cast.length ? `${cast.length} reference picture${cast.length === 1 ? "" : "s"}` : "described in words"}`
      + `   ·   Model · ${source.detail || source.name}`, "Reference pictures come from the asset chain."));

    const status = el("div", `font:12px ${SANS};color:${job?.error ? LOOK.warning : LOOK.text};min-height:18px;margin:10px 0 8px`,
      statusOf(job, "Writing"));
    host.appendChild(status);
    const buttons = el("div", "display:flex;gap:8px;justify-content:flex-end");
    if (busy(job)) {
      buttons.appendChild(makeButton("Stop", () => stopJob("write"), { kind: "danger", hint: "Stop writing" }));
    } else {
      const go = makeButton("Write", () => startWriting(), { kind: "primary", hint: "Queue the writing" });
      setDisabled(go, !writerDraft.idea.trim());
      buttons.appendChild(go);
      idea.addEventListener("input", () => setDisabled(go, !idea.value.trim()));
    }
    idea.addEventListener("input", () => {
      writerDraft.idea = idea.value;
    });
    host.appendChild(buttons);
  }

  /**
   * Queue MiniMax H3 Scene Writer on the model wired into vlm_clip and the asset chain.
   *
   * @returns {Promise<void>}
   */
  async function startWriting() {
    const settings = llmSettings(node);
    await startJob("write", {
      idea: writerDraft.idea.trim(),
      scenes: writerDraft.scenes || state.scenes.length || 8,
      seconds: writerDraft.seconds,
      style: writerDraft.style.trim() || WRITER_STYLE,
      plan_transitions: Boolean(settings.plan),
      ...samplingInputs(settings),
      ...systemInput(settings),
    }, { assets: true });
  }

  /**
   * Replace every scene, the header, the footer and the reference pictures with what was written.
   *
   * @param {{cast: object[], header: string, footer: string, scenes: object[]}} result - The answer.
   * @returns {void}
   */
  function applyWritten(result) {
    const labels = result.cast.map((member) =>
      castPictures().find((asset) => asset.name === member.file || baseName(asset.file) === member.file)?.file ?? null);
    commit(node, () => {
      write(node, "prompt_header", String(result.header ?? ""));
      write(node, "prompt_footer", String(result.footer ?? ""));
      for (let row = 1; row <= MAX_ROWS; row += 1) {
        const scene = result.scenes[row - 1];
        write(node, `prompt_${row}`, scene ? scene.prompt : "");
        if (!scene) continue;
        write(node, `duration_${row}`, Number(scene.seconds));
        write(node, `continuity_${row}`, scene.continuity);
        write(node, `overlap_${row}`, Number(scene.overlap));
        write(node, `source_${row}`, PREVIOUS_SOURCE);
        write(node, `sound_${row}`, "auto");
        write(node, `header_footer_${row}`, scene.wrap ?? "both");
      }
      for (const asset of chainOf(node)) {
        if (valueOf(asset, "role", "") === "reference picture" && valueOf(asset, "file", "") !== MOMENT) removeFromChain(asset);
      }
      result.scenes.forEach((scene, index) => {
        for (const member of scene.references) {
          if (labels[member]) makeAsset(labels[member], "reference picture", index + 1, 0);
        }
      });
    });
    say(`${result.scenes.length} scenes written in. Ctrl+Z brings the old ones back`);
    jobs.write = null;
    closeWriter();
    repaintJob("write");
    refresh(true);
  }

  // -------------------------------------------------------------------- rewrite
  /**
   * Show or hide Rewrite for a scene, under the button that opened it.
   *
   * @param {number} index - The scene, from 0.
   * @param {HTMLElement} anchor - The button.
   * @returns {void}
   */
  function openRewrite(index, anchor) {
    const scene = state.scenes[index];
    if (!scene) return;
    if (rewriteDraft.row !== scene.row) rewriteDraft = { row: scene.row, directions: "" };
    rewriteAnchor = anchor;
    popoverFor = popoverFor === "rewrite" ? "" : "rewrite";
    renderPopover();
  }

  /**
   * What the model is told around a scene: its place, its neighbours and the references it names.
   *
   * @param {number} index - The scene, from 0.
   * @returns {string} The lines.
   */
  function rewriteContext(index) {
    const scene = state.scenes[index];
    const lines = [`This is scene ${index + 1} of ${state.scenes.length}, lasting exactly `
      + `${durationOf(snapClip(framesOf(scene.seconds))).toFixed(2)} seconds.`];
    if (index > 0) lines.push(`The scene before: ${digest(state.scenes[index - 1].prompt)}`);
    if (index + 1 < state.scenes.length) lines.push(`The scene after: ${digest(state.scenes[index + 1].prompt)}`);
    const named = state.assets.filter((asset) => asset.tag && asset.role.startsWith("reference")
      && (asset.segment === index + 1 || asset.segment === EVERY_SEGMENT));
    for (const asset of named) lines.push(`${asset.tag.replace(/ with .*$/, "")} is ${asset.name}, a ${asset.role}.`);
    for (const asset of state.assets.filter((item) => item.segment === index + 1 && PINNING.has(item.role))) {
      const at = asset.role === "keyframe" ? ` at ${durationOf(asset.frame).toFixed(2)} seconds` : "";
      lines.push(`Its ${asset.role}${at} is pinned to the picture ${asset.name}.`);
    }
    const header = ["both", "header only"].includes(scene.wrap) ? String(valueOf(node, "prompt_header", "") ?? "").trim() : "";
    const footer = ["both", "footer only"].includes(scene.wrap) ? String(valueOf(node, "prompt_footer", "") ?? "").trim() : "";
    if (header) lines.push(`The HEADER placed before this scene's prompt:\n${header}`);
    if (footer) lines.push(`The FOOTER placed after it:\n${footer}`);
    return lines.join("\n");
  }

  /**
   * Queue MiniMax H3 Prompt Rewrite on the scene Rewrite is open for.
   *
   * @returns {Promise<void>}
   */
  async function startRewrite() {
    const index = state.scenes.findIndex((scene) => scene.row === rewriteDraft.row);
    const scene = state.scenes[index];
    if (!scene) return;
    const settings = llmSettings(node);
    await startJob("rewrite", {
      prompt: scene.prompt === NEW_SCENE_PROMPT ? "" : scene.prompt,
      directions: rewriteDraft.directions.trim(),
      context: rewriteContext(index),
      ...samplingInputs(settings),
      ...systemInput(settings),
    }, {}, { row: scene.row });
  }

  /**
   * Draw Rewrite: the directions, the job's progress and the new prompt to keep or discard.
   *
   * @returns {void}
   */
  function renderRewrite() {
    const index = state.scenes.findIndex((scene) => scene.row === rewriteDraft.row);
    const scene = state.scenes[index];
    if (!scene) return;
    const empty = !scene.prompt.trim() || scene.prompt === NEW_SCENE_PROMPT;
    popover.appendChild(el("div", `font:600 15px ${SANS};margin-bottom:10px`, `${empty ? "Write" : "Rewrite"} scene ${index + 1}`));
    if (!modelSource(node).wired) {
      popover.appendChild(factLine("Wire a language model into vlm_clip on MiniMax H3 Conditioning."));
      return;
    }
    const job = jobs.rewrite?.row === scene.row ? jobs.rewrite : null;
    if (job?.result != null) {
      const preview = textBox(14, `min-height:240px;font:12px/1.55 ${MONO}`);
      preview.readOnly = true;
      preview.value = job.result;
      popover.appendChild(field("New prompt", preview, `${job.result.split(/\s+/).filter(Boolean).length} words`));
      const buttons = el("div", "display:flex;gap:8px;justify-content:flex-end;margin-top:10px");
      buttons.append(
        makeButton("Discard", () => {
          jobs.rewrite = null;
          renderPopover();
        }, { kind: "tool", hint: "Keep the prompt as it is" }),
        makeButton("Use it", () => {
          commit(node, () => write(node, `prompt_${scene.row}`, job.result));
          jobs.rewrite = null;
          rewriteDraft.directions = "";
          popoverFor = "";
          renderPopover();
          say(`Scene ${index + 1} rewritten. Ctrl+Z brings the old prompt back`);
          refresh(true);
        }, { kind: "primary", hint: "Replace the scene's prompt" }),
      );
      popover.appendChild(buttons);
      return;
    }
    const directions = textBox(4, "min-height:80px");
    directions.placeholder = empty ? "What happens in the scene" : "What to change, as: make it night and add rain";
    directions.value = rewriteDraft.directions;
    popover.appendChild(field("Directions", directions, "",
      empty ? "Written under the system prompt in LLM settings." : "Blank brings the prompt in line with the system prompt."));
    popover.appendChild(el("div", `font:12px ${SANS};color:${job?.error ? LOOK.warning : LOOK.text};min-height:18px;margin:10px 0 8px`,
      statusOf(job, empty ? "Writing" : "Rewriting")));
    const buttons = el("div", "display:flex;gap:8px;justify-content:flex-end");
    if (busy(job)) {
      buttons.appendChild(makeButton("Stop", () => stopJob("rewrite"), { kind: "danger", hint: "Stop rewriting" }));
    } else {
      const go = makeButton(empty ? "Write" : "Rewrite", () => startRewrite(), { kind: "primary", hint: "Queue it" });
      setDisabled(go, busy(jobs.rewrite) || (empty && !rewriteDraft.directions.trim()));
      directions.addEventListener("input", () => setDisabled(go, busy(jobs.rewrite) || (empty && !directions.value.trim())));
      buttons.appendChild(go);
    }
    directions.addEventListener("input", () => {
      rewriteDraft.directions = directions.value;
    });
    popover.appendChild(buttons);
  }

  // -------------------------------------------------------------------- plan transitions
  /**
   * Queue MiniMax H3 Plan Transitions over the scenes as they stand.
   *
   * @returns {Promise<void>}
   */
  async function startPlan() {
    if (state.scenes.length < 2) return;
    const scenes = state.scenes.map((scene, index) => ({
      prompt: scene.prompt,
      seconds: Number(scene.seconds),
      pictures: state.assets.filter((asset) => asset.role === "reference picture"
        && (asset.segment === index + 1 || asset.segment === EVERY_SEGMENT)).length,
    }));
    await startJob("plan", { scenes: JSON.stringify(scenes), ...samplingInputs(llmSettings(node)) }, {},
      { rows: state.scenes.map((scene) => scene.row) });
  }

  /**
   * Show Plan transitions, starting a plan where none is held.
   *
   * @returns {void}
   */
  function togglePlan() {
    popoverFor = popoverFor === "plan" ? "" : "plan";
    renderPopover();
  }

  /**
   * Draw Plan transitions: the job's progress, then each scene's transition to keep or discard.
   *
   * @returns {void}
   */
  function renderPlan() {
    popover.appendChild(el("div", `font:600 15px ${SANS};margin-bottom:10px`, "Plan transitions"));
    if (!modelSource(node).wired) {
      popover.appendChild(factLine("Wire a language model into vlm_clip on MiniMax H3 Conditioning."));
      return;
    }
    const job = jobs.plan;
    if (Array.isArray(job?.result)) {
      const list = el("div", "display:flex;flex-direction:column;gap:8px;margin-bottom:10px");
      for (const choice of job.result) {
        const row = el("div", "display:flex;flex-direction:column;gap:1px;min-width:0");
        const was = state.scenes.find((scene) => scene.row === job.rows[choice.scene - 1]);
        const line = el("div", `font:12px ${SANS};color:${LOOK.text}`,
          `${choice.scene - 1} → ${choice.scene}   ${TRANSITIONS[choice.continuity]?.name ?? choice.continuity}`);
        if (was && was.continuity === choice.continuity) line.style.color = LOOK.muted;
        row.appendChild(line);
        if (choice.why) row.appendChild(el("div", `font:12px/1.4 ${SANS};color:${LOOK.muted}`, choice.why));
        list.appendChild(row);
      }
      popover.appendChild(list);
      const buttons = el("div", "display:flex;gap:8px;justify-content:flex-end");
      buttons.append(
        makeButton("Discard", () => {
          jobs.plan = null;
          repaintJob("plan");
        }, { kind: "tool", hint: "Keep the transitions as they are" }),
        makeButton("Apply", () => applyPlan(job), { kind: "primary", hint: "Set these transitions" }),
      );
      popover.appendChild(buttons);
      return;
    }
    popover.appendChild(factLine(`${state.scenes.length} scenes · the model reads each one and picks how it follows the last`));
    popover.appendChild(el("div", `font:12px ${SANS};color:${job?.error ? LOOK.warning : LOOK.text};min-height:18px;margin:10px 0 8px`,
      statusOf(job, "Reading the scenes")));
    const buttons = el("div", "display:flex;gap:8px;justify-content:flex-end");
    if (busy(job)) {
      buttons.appendChild(makeButton("Stop", () => stopJob("plan"), { kind: "danger", hint: "Stop planning" }));
    } else {
      const go = makeButton("Plan", () => startPlan(), { kind: "primary", hint: "Queue the plan" });
      setDisabled(go, state.scenes.length < 2);
      buttons.appendChild(go);
    }
    popover.appendChild(buttons);
  }

  /**
   * Set each scene's transition and overlap as planned.
   *
   * @param {{result: object[], rows: number[]}} job - The finished plan.
   * @returns {void}
   */
  function applyPlan(job) {
    commit(node, () => {
      for (const choice of job.result) {
        const row = job.rows[choice.scene - 1];
        if (!row) continue;
        write(node, `continuity_${row}`, choice.continuity);
        write(node, `overlap_${row}`, Number(choice.overlap));
        write(node, `sound_${row}`, "auto");
      }
    });
    say(`${job.result.length} transitions set. Ctrl+Z brings the old ones back`);
    jobs.plan = null;
    popoverFor = "";
    renderPopover();
    repaintJob("plan");
    refresh(true);
  }

  /**
   * Draw whichever popover is open.
   *
   * @returns {void}
   */
  function renderPopover() {
    popover.replaceChildren();
    popover.style.display = popoverFor ? "block" : "none";
    if (popoverFor) {
      // Opens above the toolbar button that opened it, on its right edge or its left for Plan;
      // Rewrite opens below its button in the Scene tab.
      const anchor = { help: helpButton, plan: planButton, rewrite: rewriteAnchor }[popoverFor] ?? issuesButton;
      const host = win.body.getBoundingClientRect();
      const box = (anchor?.isConnected ? anchor : issuesButton).getBoundingClientRect();
      const leftward = popoverFor === "plan" || popoverFor === "rewrite";
      const below = popoverFor === "rewrite";
      popover.style.left = leftward ? `${Math.max(8, Math.min(box.left - host.left, host.width - 480))}px` : "auto";
      popover.style.right = leftward ? "auto" : `${Math.max(8, host.right - box.right)}px`;
      popover.style.width = leftward ? "460px" : "";
      popover.style.top = below ? `${box.bottom - host.top + 6}px` : "auto";
      popover.style.bottom = below ? "auto" : `${Math.max(8, host.bottom - box.top + 6)}px`;
      popover.style.maxHeight = below ? `${Math.max(160, host.bottom - box.bottom - 16)}px`
        : `${Math.max(160, box.top - host.top - 16)}px`;
    }
    if (popoverFor === "plan") {
      renderPlan();
    } else if (popoverFor === "rewrite") {
      renderRewrite();
    } else if (popoverFor === "help") {
      popover.appendChild(el("div", `font:600 15px ${SANS};margin-bottom:10px`, "Building a video"));
      const tips = [
        ["Scenes", "Each block on the Scenes track is one prompt, played in order. Click one to edit it above, drag it to reorder, drag its right edge to change its length."],
        ["Transitions", "The round button between two scenes says how the next one follows: the same shot carrying on, a cut, or a cut that keeps the sound. Click it to choose, drag it to change how many frames carry over."],
        ["Pictures and sounds", "Drag a file from Media onto a track. On Scenes or Pinned frames it holds the scene to that picture at that moment; near a scene's start or end it becomes the opening or closing frame. On References the scene's prompt can name it as <Picture 1>; on All scenes every prompt can. Drag a reference to another scene to move it."],
        ["Files from your computer", "Drop them on Media to upload them, or straight onto a track."],
        ["Reference frame", "Reference frame, under Preview, makes the frame at the playhead a picture reference: a later scene reads it from the video as it is made, this scene and earlier ones take it saved as a picture in the input folder, and A new scene at the end adds a scene for it."],
        ["Preview", "With taeh3 in models/vae_approx, each scene is drawn on its clip and in Preview as it samples, and keeps its last step. Space plays from the playhead. A scene changed since its render is marked, and so is every scene after it."],
        ["The node", "Everything here is written to MiniMax H3 Conditioning and to a chain of MiniMax H3 Asset nodes wired into its assets input, so the graph runs the same with this window closed. Ctrl+Z undoes a change."],
        ["Keys", "N adds a scene, D duplicates it, Delete removes the chosen scene or asset, ← and → choose a scene, Space plays, V switches Preview and Final, + and − zoom, F fits, Esc closes."],
        ["Final", "Once a run has saved its video, Final plays it, sound included, on the same playhead as the preview frames."],
      ];
      for (const [title, text] of tips) {
        const tip = el("div", "margin:0 0 10px");
        tip.appendChild(el("div", `${LABEL};color:${LOOK.text};margin-bottom:2px`, title));
        tip.appendChild(el("div", `font:12px/1.5 ${SANS};color:${LOOK.muted}`, text));
        popover.appendChild(tip);
      }
    } else if (popoverFor === "issues") {
      const found = state.warnings;
      popover.appendChild(el("div", `font:600 15px ${SANS};margin-bottom:10px`,
        found.length ? `${found.length} to fix before a run` : "Nothing to fix"));
      
      const list = el("div", "display:flex;flex-direction:column;gap:4px");
      popover.appendChild(list);
      for (const issue of found) {
        const row = makeButton(`⚠  ${issue.text}`, () => {
          if (issue.scene !== null && issue.scene !== undefined) selectScene(issue.scene);
          if (issue.asset !== null && issue.asset !== undefined) selectedAsset = issue.asset;
          popoverFor = "";
          renderPopover();
          refresh(true);
        }, { kind: "plain" });
        row.style.cssText += `;width:100%;justify-content:flex-start;text-align:left;white-space:normal;`
          + `padding:7px 10px;font-size:12px;border-left:3px solid ${LOOK.warning}`;
        list.appendChild(row);
      }
    }
  }

  // -------------------------------------------------------------------- main area
  const main = el("div", "display:grid;flex:1 1 auto;min-height:0");
  win.body.appendChild(main);

  const media = el("div", "display:flex;flex-direction:column;min-height:0;min-width:0;"
    + `background:${LOOK.body}`);
  const mediaHead = el("div", "flex:0 0 auto;padding:12px 12px 2px");
  mediaHead.appendChild(heading("Media"));
  mediaHead.title = "Drop files here to upload";
  media.appendChild(mediaHead);
  const browser = createAssetBrowser({
    logName: LOG_NAME,
    roles: [],
    thumbWidth: TILE_WIDTH,
    thumbHeight: TILE_HEIGHT,
    acceptFiles: true,
    onChoose: (entry, roleId) => chooseFromBin(entry, roleId),
    onSelect: (entry) => inspect({
      id: null, file: entry.label, kind: entry.kind || kindOf(entry.label), name: baseName(entry.label),
      role: "", tag: "", segment: -1,
    }),
    onDragStart: (entry) => {
      dragging = { kind: entry.kind, label: entry.label };
      strip.repaint();
      lightDropZones(true);
    },
    onDragEnd: () => {
      dragging = null;
      strip.repaint();
      lightDropZones(false);
    },
  });
  const browserBody = el("div", "flex:1 1 auto;min-height:0");
  browserBody.appendChild(browser.element);
  media.appendChild(browserBody);
  main.appendChild(media);

  const inspector = el("div", "display:flex;flex-direction:column;min-height:0;min-width:0");
  const tabs = tabStrip([["scene", "Scene"], ["video", "Video settings"], ["llm", "LLM settings"]], (id) => {
    tab = id;
    refresh(true);
  });
  inspector.appendChild(tabs.element);
  const pane = el("div", "flex:1 1 auto;min-height:0;overflow:auto;padding:14px 20px 20px;"
    + `display:flex;flex-direction:column;gap:14px;${SCROLLBARS}`);
  inspector.appendChild(pane);
  main.appendChild(inspector);

  // -------------------------------------------------------------------- monitor
  const monitor = el("div", "display:flex;flex-direction:column;gap:10px;min-height:0;min-width:0;"
    + `padding:12px;background:${LOOK.body};overflow:auto;${SCROLLBARS}`);
  const monitorHead = el("div", "display:flex;align-items:center;gap:8px;min-height:24px");
  // Preview draws the preview VAE's frames; Final plays the video the run saved.
  const viewSwitch = el("div", `display:flex;border:1px solid ${LOOK.border};border-radius:6px;overflow:hidden;flex:0 0 auto`);
  const viewButton = (label, id, hint) => {
    const button = el("button", `border:0;padding:4px 12px;font:600 12px/18px ${SANS};cursor:pointer;background:transparent;`
      + `color:${LOOK.muted}`, label);
    button.type = "button";
    button.title = hint;
    button.addEventListener("click", () => setView(id));
    return button;
  };
  const previewView = viewButton("Preview", "preview", "Preview VAE frames (V)");
  const finalView = viewButton("Final", "final", "No finished video yet");
  viewSwitch.append(previewView, finalView);
  monitorHead.appendChild(viewSwitch);
  const monitorState = el("span", "margin-left:auto;display:flex;gap:6px;align-items:center");
  monitorHead.appendChild(monitorState);
  const closeInspect = makeButton("Close", () => inspect(null), { kind: "tool", hint: "Back to the frames (Esc)" });
  monitor.appendChild(monitorHead);
  // Shorter than its shape where the column is, and the frame letterboxes inside it.
  const screen = el("div", "position:relative;width:100%;aspect-ratio:16 / 9;border-radius:8px;overflow:hidden;"
    + `background:#000;border:1px solid ${LOOK.border};flex:0 1 auto;min-height:120px;cursor:pointer`);
  screen.title = "Play or pause (Space)";
  screen.addEventListener("click", () => (inspected ? inspect(null) : togglePlay()));
  const screenCanvas = document.createElement("canvas");
  screenCanvas.style.cssText = "position:absolute;inset:0;width:100%;height:100%";
  const screenNote = el("div", "position:absolute;inset:0;display:flex;align-items:center;justify-content:center;"
    + `padding:18px;text-align:center;font:12px/1.5 ${SANS};color:rgba(255, 255, 255, 0.72);pointer-events:none`);
  // An asset shown full size in place of the frames, until the playhead is wanted again.
  const inspectPicture = document.createElement("img");
  inspectPicture.draggable = false;
  inspectPicture.style.cssText = "position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#000;display:none";
  inspectPicture.addEventListener("load", () => drawMonitor());
  const inspectClip = document.createElement("video");
  inspectClip.controls = true;
  inspectClip.loop = true;
  inspectClip.playsInline = true;
  inspectClip.preload = "metadata";
  inspectClip.style.cssText = "position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#000;display:none";
  const inspectSound = document.createElement("audio");
  inspectSound.controls = true;
  inspectSound.preload = "metadata";
  inspectSound.style.cssText = "position:absolute;left:16px;bottom:16px;width:calc(100% - 32px);display:none";
  for (const each of [inspectPicture, inspectClip, inspectSound]) each.addEventListener("error", () => drawMonitor());
  for (const each of [inspectClip, inspectSound]) each.addEventListener("click", (event) => event.stopPropagation());
  const finalVideo = document.createElement("video");
  finalVideo.preload = "auto";
  finalVideo.playsInline = true;
  finalVideo.style.cssText = "position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#000;display:none";
  finalVideo.addEventListener("ended", () => stopPlaying());
  // Animation frames stall in a hidden page; the video's own clock still moves the playhead.
  finalVideo.addEventListener("timeupdate", () => {
    if (!playing || !showingFinal()) return;
    playhead = finalVideo.currentTime;
    strip.repaint();
    drawMonitor();
  });
  finalVideo.addEventListener("loadedmetadata", () => drawMonitor());
  finalVideo.addEventListener("error", () => drawMonitor());
  screen.append(screenCanvas, finalVideo, inspectPicture, inspectClip, inspectSound, screenNote);
  monitor.appendChild(screen);
  // The whole video as a bar: a tick at each scene's start, the playhead as a knob.
  const seek = el("div", "position:relative;height:16px;flex:0 0 auto;cursor:pointer;touch-action:none");
  seek.title = "Drag to move the playhead";
  seek.append(
    el("div", `position:absolute;left:0;right:0;top:6px;height:4px;border-radius:2px;background:${LOOK.border}`),
  );
  const seekFill = el("div", `position:absolute;left:0;top:6px;height:4px;border-radius:2px;background:${LOOK.accent}`);
  const seekMarks = el("div", "position:absolute;inset:0;pointer-events:none");
  const seekKnob = el("div", "position:absolute;top:2px;width:12px;height:12px;margin-left:-6px;border-radius:50%;"
    + `background:${LOOK.accent};box-shadow:0 0 0 2px ${LOOK.body}`);
  seek.append(seekFill, seekMarks, seekKnob);
  monitor.appendChild(seek);
  const seekTo = (event) => {
    const box = seek.getBoundingClientRect();
    const share = box.width > 0 ? Math.max(0, Math.min(1, (event.clientX - box.left) / box.width)) : 0;
    if (inspected) inspect(null);
    playhead = (share * lastFrame()) / FPS;
    movedAt = performance.now();
    browser.setRoles(roleEntries());
    strip.repaint();
    drawMonitor();
  };
  seek.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    event.preventDefault();
    try {
      seek.setPointerCapture(event.pointerId);
    } catch (error) {
      console.warn(`[${LOG_NAME}] The seek bar could not capture the pointer:`, error);
    }
    seekTo(event);
    const move = (moved) => seekTo(moved);
    const end = () => {
      seek.removeEventListener("pointermove", move);
      seek.removeEventListener("pointerup", end);
      seek.removeEventListener("pointercancel", end);
    };
    seek.addEventListener("pointermove", move);
    seek.addEventListener("pointerup", end);
    seek.addEventListener("pointercancel", end);
  });
  const transport = el("div", "display:flex;gap:6px;align-items:center;min-width:0");
  const backButton = makeButton(textGlyph("⏮"), () => stepScene(-1), {
    kind: "tool", hint: "The start of this scene, or the one before (Home)",
  });
  const playButton = makeButton(textGlyph("▶"), () => togglePlay(), { kind: "tool", hint: "Play from the playhead (Space)" });
  const nextButton = makeButton(textGlyph("⏭"), () => stepScene(1), { kind: "tool", hint: "The start of the next scene (End)" });
  const momentButton = makeButton("Reference frame", (event) => openMomentMenu(event), {
    kind: "tool", hint: "Use the frame at the playhead as a picture reference",
  });
  playButton.style.minWidth = "40px";
  const captionScene = el("span", `flex:1 1 auto;min-width:0;margin-left:6px;overflow:hidden;text-overflow:ellipsis;`
    + `white-space:nowrap;font:12px ${SANS};color:${LOOK.muted}`);
  const captionClock = el("span", `flex:0 0 auto;font:12px ${MONO};color:${LOOK.text};white-space:nowrap`);
  transport.append(backButton, playButton, nextButton, captionScene, captionClock, momentButton);
  monitor.appendChild(transport);
  // Running the graph, held at the foot of the column.
  const footer = el("div", "display:flex;gap:8px;align-items:center;margin-top:auto;padding-top:10px;"
    + `border-top:1px solid ${LOOK.border};min-width:0`);
  const runStatus = el("span", `flex:1 1 auto;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;`
    + `font:12px ${SANS};color:${LOOK.muted}`);
  const stopButton = makeButton("■  Stop", () => api.interrupt(null), { kind: "danger", hint: "Stop the run that is sampling" });
  const runButton = makeButton("▶  Run", () => app.queuePrompt(0), { kind: "primary", hint: "Queue the graph, as the Run button does" });
  footer.append(runStatus, issuesButton, stopButton, runButton);
  monitor.appendChild(footer);
  main.appendChild(monitor);

  // -------------------------------------------------------------------- column dividers
  const leftDivider = makeDivider("Drag to widen Media or the scene; double-click to reset");
  const rightDivider = makeDivider("Drag to widen the preview or the scene; double-click to reset");
  main.replaceChildren(media, leftDivider, inspector, rightDivider, monitor);
  const panes = { media: MEDIA_WIDTH, monitor: MONITOR_WIDTH };
  try {
    const stored = JSON.parse(localStorage.getItem(PANES_KEY) || "null");
    if (Number.isFinite(stored?.media)) panes.media = stored.media;
    if (Number.isFinite(stored?.monitor)) panes.monitor = stored.monitor;
  } catch (error) {
    console.warn(`[${LOG_NAME}] The column widths could not be read:`, error);
  }

  /**
   * Lay the columns out at the widths asked for, the scene column keeping its least.
   *
   * @returns {void}
   */
  function layoutPanes() {
    const room = Math.max(0, main.clientWidth - DIVIDER_WIDTH * 2 - INSPECTOR_MIN);
    let monitorWidth = Math.max(MONITOR_MIN, panes.monitor);
    let mediaWidth = Math.max(MEDIA_MIN, panes.media);
    if (room > 0 && mediaWidth + monitorWidth > room) {
      monitorWidth = Math.max(MONITOR_MIN, room - mediaWidth);
      mediaWidth = Math.max(MEDIA_MIN, room - monitorWidth);
    }
    main.style.gridTemplateColumns = `${mediaWidth}px ${DIVIDER_WIDTH}px minmax(${INSPECTOR_MIN}px,1fr) `
      + `${DIVIDER_WIDTH}px ${monitorWidth}px`;
  }

  /**
   * One draggable rule between two columns.
   *
   * @param {string} hint - What the hover says.
   * @returns {HTMLElement} The divider.
   */
  function makeDivider(hint) {
    const divider = el("div", "position:relative;cursor:col-resize;touch-action:none;user-select:none;"
      + `background:linear-gradient(${LOOK.border},${LOOK.border}) center / 1px 100% no-repeat`);
    divider.title = hint;
    divider.addEventListener("mouseenter", () => {
      divider.style.background = `linear-gradient(${LOOK.accent},${LOOK.accent}) center / 2px 100% no-repeat`;
    });
    divider.addEventListener("mouseleave", () => {
      divider.style.background = `linear-gradient(${LOOK.border},${LOOK.border}) center / 1px 100% no-repeat`;
    });
    return divider;
  }

  /**
   * Remember the column widths for the next window.
   *
   * @returns {void}
   */
  function rememberPanes() {
    try {
      localStorage.setItem(PANES_KEY, JSON.stringify(panes));
    } catch (error) {
      console.warn(`[${LOG_NAME}] The column widths could not be stored:`, error);
    }
  }

  for (const [divider, side] of [[leftDivider, "media"], [rightDivider, "monitor"]]) {
    divider.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      event.preventDefault();
      const startX = event.clientX;
      const start = side === "media" ? media.offsetWidth : monitor.offsetWidth;
      const other = side === "media" ? monitor.offsetWidth : media.offsetWidth;
      const widest = Math.max(0, main.clientWidth - DIVIDER_WIDTH * 2 - INSPECTOR_MIN - other);
      try {
        divider.setPointerCapture(event.pointerId);
      } catch (error) {
        console.warn(`[${LOG_NAME}] The divider could not capture the pointer:`, error);
      }
      const move = (moved) => {
        const shift = moved.clientX - startX;
        const wanted = side === "media" ? start + shift : start - shift;
        panes[side] = Math.round(Math.max(side === "media" ? MEDIA_MIN : MONITOR_MIN, Math.min(widest, wanted)));
        layoutPanes();
      };
      const end = () => {
        divider.removeEventListener("pointermove", move);
        divider.removeEventListener("pointerup", end);
        divider.removeEventListener("pointercancel", end);
        rememberPanes();
      };
      divider.addEventListener("pointermove", move);
      divider.addEventListener("pointerup", end);
      divider.addEventListener("pointercancel", end);
    });
    divider.addEventListener("dblclick", () => {
      panes[side] = side === "media" ? MEDIA_WIDTH : MONITOR_WIDTH;
      layoutPanes();
      rememberPanes();
    });
  }
  const mainSize = new ResizeObserver(() => layoutPanes());
  mainSize.observe(main);
  layoutPanes();

  let playing = false;
  let frameRequest = 0;
  let lastTick = 0;
  let running = false;
  let live = null;
  let movedAt = 0;
  const stamps = new Map();

  const sheets = followSegmentPreviews(() => node.properties?.[PREVIEW_KEY] ?? executionId(node), (segment, preview) => {
    const fields = preview.fields;
    if (fields.prompt_id && fields.version > latestVersion) {
      latestRun = fields.prompt_id;
      latestVersion = fields.version;
    }
    stamps.set(segment, stampOf(segment));
    if (!fields.final && running) live = { segment, step: fields.step, steps: fields.steps };
    else if (live?.segment === segment) live = null;
    screen.style.aspectRatio = `${fields.width} / ${fields.height}`;
    // The playhead follows the scene being sampled unless it was moved by hand a moment ago.
    const entry = state.layout[segment];
    if (live?.segment === segment && entry && !playing && performance.now() - movedAt > FOLLOW_AFTER_MS
        && sceneAt(state.layout, Math.round(playhead * FPS)) !== segment) {
      playhead = entry.start / FPS;
    }
    strip.repaint();
    drawMonitor();
  });

  const runListeners = {
    execution_start: () => {
      running = true;
      live = null;
      drawMonitor();
    },
    execution_success: () => endRun(),
    execution_error: () => endRun(),
    execution_interrupted: () => endRun(),
    executed: (event) => {
      const detail = event?.detail;
      if (detail?.prompt_id && detail.prompt_id === latestRun) adoptFinished(detail.prompt_id, detail.output);
    },
  };
  for (const [name, listener] of Object.entries(runListeners)) api.addEventListener(name, listener);

  // The language model jobs, each followed by its prompt id.
  const jobOf = (detail) => Object.keys(jobs).find((kind) => jobs[kind]?.promptId
    && detail?.prompt_id === jobs[kind].promptId) ?? null;
  const writerListeners = {
    execution_start: (event) => {
      const kind = jobOf(event?.detail);
      if (!kind) return;
      jobs[kind].status = "running";
      repaintJob(kind);
    },
    progress: (event) => {
      const detail = event?.detail;
      const kind = jobOf(detail);
      if (!kind || String(detail.node) !== JOBS[kind].id) return;
      jobs[kind].step = detail.value;
      jobs[kind].steps = detail.max;
      repaintJob(kind);
    },
    executed: (event) => {
      const detail = event?.detail;
      const kind = jobOf(detail);
      if (!kind || String(detail.node) !== JOBS[kind].id) return;
      try {
        jobs[kind].result = answerOf(kind, detail.output);
      } catch (error) {
        jobs[kind].error = `Nothing to apply: ${error.message ?? error}`;
      }
      if (kind === "write" && !writerHost) openWriter();
      if (kind === "rewrite") rewriteDraft.row = jobs.rewrite.row;
      if (kind !== "write" && popoverFor !== kind) {
        popoverFor = kind;
        renderPopover();
      }
      repaintJob(kind);
    },
    execution_error: (event) => {
      const kind = jobOf(event?.detail);
      if (!kind) return;
      jobs[kind].error = event.detail.exception_message || `${JOBS[kind].title} failed`;
      repaintJob(kind);
    },
    execution_interrupted: (event) => {
      const kind = jobOf(event?.detail);
      if (!kind) return;
      jobs[kind] = null;
      repaintJob(kind);
    },
  };
  for (const [name, listener] of Object.entries(writerListeners)) api.addEventListener(name, listener);

  /**
   * Note that a run has stopped, however it stopped.
   *
   * @returns {void}
   */
  function endRun() {
    running = false;
    live = null;
    strip.repaint();
    drawMonitor();
  }

  // What the run saved: the last video it wrote to the output folder.
  let view = "preview";
  let finished = null;
  let latestRun = "";
  let latestVersion = -1;

  /**
   * Videos a run's outputs list.
   *
   * @param {object} outputs - One node's or one run's outputs, as ComfyUI reports them.
   * @returns {object[]} `{filename, subfolder, type}` entries naming a video, in order.
   */
  function videosOf(outputs) {
    const found = [];
    const visit = (value) => {
      if (!Array.isArray(value)) return;
      for (const item of value) {
        if (item && typeof item.filename === "string" && VIDEO_FILE.test(item.filename)) found.push(item);
      }
    };
    for (const value of Object.values(outputs ?? {})) {
      if (Array.isArray(value)) visit(value);
      else if (value && typeof value === "object") Object.values(value).forEach(visit);
    }
    return found;
  }

  /**
   * Take a run's last saved video as the finished one.
   *
   * @param {string} promptId - The run.
   * @param {object} outputs - What it reported.
   * @returns {void}
   */
  function adoptFinished(promptId, outputs) {
    const chosen = videosOf(outputs).filter((item) => (item.type ?? "output") === "output").at(-1);
    if (!chosen) return;
    const url = api.apiURL(`/view?filename=${encodeURIComponent(chosen.filename)}`
      + `&subfolder=${encodeURIComponent(chosen.subfolder ?? "")}&type=output`);
    if (finished?.url === url) return;
    finished = { url, name: chosen.filename, promptId };
    finalVideo.src = url;
    markView();
    drawMonitor();
  }

  /**
   * Find the finished video of the run the held sheets came from.
   *
   * @returns {Promise<void>}
   */
  async function recoverFinished() {
    let newest = null;
    for (const preview of sheets.previews.values()) {
      if (preview.fields.prompt_id && (!newest || preview.fields.version > newest.version)) newest = preview.fields;
    }
    if (!newest) return;
    latestRun = newest.prompt_id;
    latestVersion = newest.version;
    try {
      const response = await api.fetchApi(`/history/${encodeURIComponent(latestRun)}`, { cache: "no-store" });
      if (response.status !== 200) return;
      const held = (await response.json())?.[latestRun];
      if (held?.outputs) adoptFinished(latestRun, held.outputs);
    } catch (error) {
      console.warn(`[${LOG_NAME}] The finished video could not be found:`, error);
    }
  }

  /**
   * Whether the screen plays the finished video.
   *
   * @returns {boolean} True in the Final view with a finished video to play.
   */
  function showingFinal() {
    return view === "final" && finished !== null;
  }

  /**
   * Show an asset full size on the screen in place of the frames.
   *
   * @param {object|null} asset - `{id, file, kind, name, role, tag, segment}`, or null for the
   *   frames again.
   * @returns {void}
   */
  function inspect(asset) {
    const showable = Boolean(asset?.file) && !asset.moment;
    if (showable && playing) stopPlaying();
    inspected = showable ? asset : null;
    if (!inspected) {
      inspectClip.pause();
      inspectSound.pause();
    }
    markView();
    drawMonitor();
  }

  /**
   * Switch the screen between the preview frames and the finished video.
   *
   * @param {string} next - `preview` or `final`.
   * @returns {void}
   */
  function setView(next) {
    if (inspected) {
      inspect(null);
      if (next === view) return;
    }
    if (next === view || (next === "final" && !finished)) return;
    const resume = playing;
    stopPlaying();
    view = next;
    markView();
    drawMonitor();
    if (resume) togglePlay();
  }

  /**
   * Draw the view switch for the view shown.
   *
   * @returns {void}
   */
  function markView() {
    for (const [button, id] of [[previewView, "preview"], [finalView, "final"]]) {
      const on = !inspected && (id === "final") === showingFinal();
      button.style.background = on ? SELECTED_FILL : "transparent";
      button.style.color = on ? LOOK.text : LOOK.muted;
    }
    setDisabled(finalView, !finished);
    finalView.title = finished ? `${finished.name} (V)` : "No finished video yet";
  }
  markView();

  const screenSize = new ResizeObserver(() => drawMonitor());
  screenSize.observe(screen);
  const stopWatchingScreen = watchSurfaceRatio(screen, () => drawMonitor());

  /**
   * What a scene and everything before it was set to, for telling a preview gone stale.
   *
   * @param {number} index - The scene, from 0.
   * @returns {string} The settings, as text.
   */
  function stampOf(index) {
    const run = ["mode", "aspect_ratio", "megapixels", "width", "height", "prompt_header", "prompt_footer", "loop"]
      .map((name) => valueOf(node, name));
    const scenes = state.scenes.slice(0, index + 1).map((scene, at) => [
      scene.prompt, scene.seconds, scene.overlap, scene.continuity, scene.source, scene.wrap, scene.model, scene.hold, scene.sound, scene.seed,
      state.assets.filter((asset) => asset.segment === at + 1 || asset.segment === EVERY_SEGMENT)
        .map((asset) => [asset.role, asset.file, asset.frame, asset.segment]),
    ]);
    return JSON.stringify([run, scenes]);
  }

  /**
   * Whether a scene, or one before it, has changed since its preview was drawn.
   *
   * @param {number} index - The scene, from 0.
   * @returns {boolean} True when it no longer shows what the scene would render.
   */
  function isStale(index) {
    return stamps.has(index) && stamps.get(index) !== stampOf(index);
  }

  /**
   * What the Scenes track draws on one scene from its preview.
   *
   * @param {number} index - The scene, from 0.
   * @returns {object|null} `{sheet, stale, live, partial}`, or null where it has none.
   */
  function previewOf(index) {
    const sheet = sheets.previews.get(index) ?? null;
    const sampling = live?.segment === index ? live : null;
    if (!sheet && !sampling) return null;
    return {
      sheet, stale: isStale(index), live: sampling,
      partial: Boolean(sheet && !sheet.fields.final && !sampling),
    };
  }

  /**
   * The frame after the last one the laid-out scenes reach.
   *
   * @returns {number} A frame count.
   */
  function lastFrame() {
    return state.layout.filter(Boolean).at(-1)?.end ?? 0;
  }

  /**
   * Put a tick on the seek bar where each scene after the first starts.
   *
   * @returns {void}
   */
  function renderSeekMarks() {
    seekMarks.replaceChildren();
    const end = lastFrame();
    if (end <= 0) return;
    for (const entry of state.layout.filter(Boolean).slice(1)) {
      seekMarks.appendChild(el("span", `position:absolute;top:3px;width:2px;height:10px;margin-left:-1px;`
        + `border-radius:1px;background:${LOOK.muted};left:${(entry.start / end) * 100}%`));
    }
  }

  /**
   * Draw the frame under the playhead, and say where it is and how the run is going.
   *
   * @returns {void}
   */
  function drawMonitor() {
    if (disposed) return;
    const ratio = surfaceRatio(screen);
    const w = Math.max(1, Math.round(screen.clientWidth * ratio));
    const h = Math.max(1, Math.round(screen.clientHeight * ratio));
    if (screenCanvas.width !== w || screenCanvas.height !== h) {
      screenCanvas.width = w;
      screenCanvas.height = h;
    }
    const pen = screenCanvas.getContext("2d");
    pen.fillStyle = "#000";
    pen.fillRect(0, 0, w, h);
    const end = lastFrame();
    const frame = Math.max(0, Math.min(Math.round(playhead * FPS), end - 1));
    const index = sceneAt(state.layout, frame);
    const sheet = index >= 0 ? sheets.previews.get(index) : null;
    captionClock.textContent = `${clockOf(playhead)} / ${clockOf(end / FPS)}`;
    const share = end > 0 ? Math.max(0, Math.min(1, (playhead * FPS) / end)) : 0;
    seekFill.style.width = `${share * 100}%`;
    seekKnob.style.left = `${share * 100}%`;
    captionScene.textContent = index >= 0
      ? `Scene ${index + 1}  ·  ${state.scenes[index]?.title || "Untitled scene"}` : "";
    const media = inspected ? inspectedMedia(inspected.file, inspected.kind) : null;
    const final = !media && showingFinal();
    screenCanvas.style.display = final || media ? "none" : "block";
    finalVideo.style.display = final ? "block" : "none";
    for (const [element, kind] of [[inspectPicture, "picture"], [inspectClip, "clip"], [inspectSound, "sound"]]) {
      const on = Boolean(media?.url) && media.kind === kind;
      element.style.display = on ? "block" : "none";
      if (on && element.dataset.source !== media.url) {
        element.dataset.source = media.url;
        element.src = media.url;
      }
    }
    if (final && !playing && finalVideo.readyState >= 1) {
      // Paused, the video shows the playhead's frame; playing, it leads the playhead and is never sought.
      const wanted = Math.min(finalVideo.duration || Infinity, (frame + 0.5) / FPS);
      if (Math.abs(finalVideo.currentTime - wanted) > 0.5 / FPS) finalVideo.currentTime = wanted;
    }
    if (sheet && !final && !media) {
      drawSheetFrame(pen, sheet, sheetFrameAt(state.layout[index], sheet.fields, frame), 0, 0, w, h, true);
    }
    monitorState.replaceChildren();
    if (media) {
      const element = { picture: inspectPicture, clip: inspectClip, sound: inspectSound }[media.kind];
      screenNote.textContent = !media.url ? "No preview for this file"
        : element?.error || (media.kind === "picture" && inspectPicture.complete && !inspectPicture.naturalWidth)
          ? "File unreadable" : media.kind === "sound" ? `♪  ${inspected.name}` : "";
      captionScene.textContent = [inspected.tag, inspected.name].filter(Boolean).join("  ");
      const where = inspected.segment > 0 ? `Scene ${inspected.segment}`
        : inspected.segment === EVERY_SEGMENT ? "Every scene" : "";
      const role = inspected.role ? ROLE_NAMES[inspected.role] ?? inspected.role : "Media";
      monitorState.append(pill(where ? `${role} · ${where}` : role, LOOK.accent), closeInspect);
    } else {
      screenNote.textContent = final ? (finalVideo.error ? "Video unreadable" : "") : sheet ? "" : "Idle Preview";
      const shown = index >= 0 ? previewOf(index) : null;
      const length = finalVideo.duration;
      if (final && Number.isFinite(length) && Math.abs(length - end / FPS) > 0.25) {
        monitorState.appendChild(pill("Different length", LOOK.warning));
      } else if (final && shown?.stale) {
        monitorState.appendChild(pill("Changed", LOOK.warning));
      } else if (final) {
        const tag = pill("Final", LOOK.success);
        tag.title = finished.name;
        monitorState.appendChild(tag);
      } else if (shown?.live) {
        monitorState.appendChild(pill(`Sampling · ${shown.live.step}/${shown.live.steps}`, LOOK.accent));
      } else if (shown?.stale) {
        monitorState.appendChild(pill("Changed", LOOK.warning));
      } else if (shown?.partial) {
        monitorState.appendChild(pill(`Stopped · ${shown.sheet.fields.step}/${shown.sheet.fields.steps}`, LOOK.warning));
      }
    }
    const drawn = state.scenes.filter((scene, at) => sheets.previews.get(at)?.fields.final).length;
    const stale = state.scenes.filter((scene, at) => sheets.previews.has(at) && isStale(at)).length;
    runStatus.textContent = live
      ? `Scene ${live.segment + 1}/${state.scenes.length} · step ${live.step}/${live.steps}`
      : running ? "Running"
        : drawn ? `${drawn}/${state.scenes.length} drawn${stale ? ` · ${stale} changed` : ""}`
          : "";
    setDisabled(stopButton, !running);
  }

  /**
   * Start playing from the playhead, or stop.
   *
   * @returns {void}
   */
  function togglePlay() {
    if (playing) {
      stopPlaying();
      return;
    }
    if (inspected) inspect(null);
    const end = lastFrame() / FPS;
    if (end <= 0) return;
    if (playhead >= end - 1 / FPS) playhead = 0;
    playing = true;
    lastTick = performance.now();
    playButton.textContent = textGlyph("⏸");
    if (showingFinal()) {
      finalVideo.currentTime = playhead;
      finalVideo.play().catch((error) => {
        console.warn(`[${LOG_NAME}] The finished video would not play:`, error);
        stopPlaying();
      });
    }
    frameRequest = requestAnimationFrame(tick);
  }

  /**
   * Move the playhead on by the time since the last frame drawn.
   *
   * @param {number} now - The frame's timestamp.
   * @returns {void}
   */
  function tick(now) {
    if (!playing || disposed) return;
    const end = lastFrame() / FPS;
    playhead = Math.min(end, showingFinal() ? finalVideo.currentTime : playhead + (now - lastTick) / 1000);
    lastTick = now;
    strip.repaint();
    drawMonitor();
    if (playhead >= end) {
      stopPlaying();
      return;
    }
    frameRequest = requestAnimationFrame(tick);
  }

  /**
   * Stop playing where the playhead is.
   *
   * @returns {void}
   */
  function stopPlaying() {
    const was = playing;
    if (playing && showingFinal() && finalVideo.readyState >= 1) playhead = finalVideo.currentTime;
    playing = false;
    cancelAnimationFrame(frameRequest);
    if (!finalVideo.paused) finalVideo.pause();
    playButton.textContent = textGlyph("▶");
    if (was && !disposed) {
      strip.repaint();
      drawMonitor();
    }
  }

  /**
   * Offer the scenes after the playhead a reference to the frame under it.
   *
   * @param {MouseEvent} event - The press that opened the menu.
   * @returns {void}
   */
  function openMomentMenu(event) {
    const frame = Math.max(0, Math.min(Math.round(playhead * FPS), lastFrame() - 1));
    const here = sceneAt(state.layout, frame);
    // A later scene reads the frame from the video as it is made; this scene and earlier ones take
    // it saved as a picture, from the finished video or the preview.
    const drawn = Boolean(finished) || sheets.previews.has(here);
    const items = [{ header: `Reference frame ${frame} (${clockOf(frame / FPS)}) in` }];
    state.scenes.forEach((scene, index) => {
      const later = index > here;
      items.push({
        label: `Scene ${index + 1}`, detail: later ? scene.title : "saved as a picture",
        disabled: !later && !drawn, onSelect: () => (later ? addMoment(frame, index + 1) : addStill(frame, index + 1)),
      });
    });
    items.push({ separator: true });
    if (here < state.scenes.length - 1) {
      items.push({ label: "Every later scene", onSelect: () => addMoment(frame, EVERY_SEGMENT) });
    }
    items.push({
      label: "Every scene", detail: "saved as a picture", disabled: !drawn, onSelect: () => addStill(frame, EVERY_SEGMENT),
    });
    items.push({ separator: true });
    items.push({
      label: "A new scene at the end", detail: `Scene ${state.scenes.length + 1}`,
      disabled: state.scenes.length >= MAX_ROWS, onSelect: () => momentInNewScene(frame),
    });
    openPopupMenu({ items, x: event.clientX, y: event.clientY, logName: LOG_NAME });
  }

  /**
   * One frame of the video as a picture: the finished video's where it holds the frame, else the preview's.
   *
   * @param {number} frame - The frame of the video.
   * @returns {Promise<Blob|null>} A PNG, or null where neither has the frame.
   */
  async function frameStill(frame) {
    const canvas = document.createElement("canvas");
    if (finished && finalVideo.readyState >= 1 && frame / FPS < (finalVideo.duration || 0)) {
      const wanted = (frame + 0.5) / FPS;
      if (Math.abs(finalVideo.currentTime - wanted) > 0.5 / FPS || finalVideo.readyState < 2) {
        await new Promise((resolve) => {
          const done = () => {
            finalVideo.removeEventListener("seeked", done);
            resolve();
          };
          finalVideo.addEventListener("seeked", done);
          setTimeout(done, STILL_WAIT_MS);
          finalVideo.currentTime = wanted;
        });
      }
      if (finalVideo.videoWidth && finalVideo.readyState >= 2) {
        canvas.width = finalVideo.videoWidth;
        canvas.height = finalVideo.videoHeight;
        canvas.getContext("2d").drawImage(finalVideo, 0, 0);
      }
    }
    if (!canvas.width) {
      const index = sceneAt(state.layout, frame);
      const sheet = index >= 0 ? sheets.previews.get(index) : null;
      if (!sheet) return null;
      canvas.width = Number(sheet.fields.cell_width) || 1;
      canvas.height = Number(sheet.fields.cell_height) || 1;
      drawSheetFrame(canvas.getContext("2d"), sheet, sheetFrameAt(state.layout[index], sheet.fields, frame),
        0, 0, canvas.width, canvas.height, true);
    }
    return new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  }

  /**
   * Save one frame of the video to the input folder and reference it in a scene, as one undo entry.
   *
   * @param {number} frame - The frame of the video.
   * @param {number} segment - Its scene, from 1, or 0 for every scene.
   * @returns {Promise<void>}
   */
  async function addStill(frame, segment) {
    say(`Saving frame ${frame}…`);
    try {
      const blob = await frameStill(frame);
      if (!blob) {
        say("Nothing is drawn at this frame yet. Run the graph first");
        return;
      }
      const file = new File([blob], `h3_frame_${String(frame).padStart(5, "0")}.png`, { type: "image/png" });
      const [saved] = await browser.upload([file]);
      if (!saved) {
        say("The frame could not be saved to the input folder");
        return;
      }
      commit(node, () => makeAsset(saved.label, "reference picture", segment, 0));
      say(`Frame ${frame} saved as ${baseName(saved.label)}, referenced in `
        + `${segment === EVERY_SEGMENT ? "every scene" : `scene ${segment}`}`);
      refresh(true);
    } catch (error) {
      console.error(`[${LOG_NAME}] Frame ${frame} could not be saved:`, error);
      say("The frame could not be saved to the input folder");
    }
  }

  /**
   * Add a scene at the end that references one frame of the video being made, as one undo entry.
   *
   * @param {number} frame - The frame of the finished video.
   * @returns {void}
   */
  function momentInNewScene(frame) {
    const index = commit(node, () => {
      const made = writeNewScene();
      if (made >= 0) makeAsset(MOMENT, "reference picture", made + 1, frame);
      return made;
    });
    if (index === undefined || index < 0) {
      say(`The node holds ${MAX_ROWS} scenes at most`);
      return;
    }
    say(`Frame ${frame} referenced in the new scene ${index + 1}`);
    selectScene(index);
    refresh(true);
    strip.reveal(index);
  }

  /**
   * Chain an asset reading one frame of the video being made into the scenes it is for.
   *
   * @param {number} frame - The frame of the finished video.
   * @param {number} segment - Its scene, from 1, or 0 for every scene.
   * @returns {void}
   */
  function addMoment(frame, segment) {
    commit(node, () => makeAsset(MOMENT, "reference picture", segment, frame));
    say(`Frame ${frame} referenced in ${segment === EVERY_SEGMENT ? "every later scene" : `scene ${segment}`}`);
    refresh(true);
  }

  /**
   * Move the playhead to a scene's start.
   *
   * @param {number} direction - `-1` for this scene's start or the one before, `1` for the next.
   * @returns {void}
   */
  function stepScene(direction) {
    const entries = state.layout.filter(Boolean);
    if (!entries.length) return;
    if (inspected) inspect(null);
    const frame = Math.round(playhead * FPS);
    const here = Math.max(0, sceneAt(state.layout, frame));
    let target = here + (direction > 0 ? 1 : 0);
    if (direction < 0 && frame - entries[here].start <= 2) target = here - 1;
    target = Math.max(0, Math.min(entries.length - 1, target));
    playhead = direction > 0 && here === entries.length - 1 ? lastFrame() / FPS : entries[target].start / FPS;
    movedAt = performance.now();
    strip.repaint();
    drawMonitor();
  }

  pane.addEventListener("focusout", () => {
    if (!renderPending) return;
    // After the focus has moved, so a field handing focus to its neighbour is not rebuilt under it.
    setTimeout(() => {
      if (!renderPending || disposed) return;
      const active = document.activeElement;
      if (pane.contains(active) && (active?.tagName === "TEXTAREA" || active?.tagName === "INPUT")) return;
      renderPending = false;
      renderInspector();
    }, 0);
  });

  // -------------------------------------------------------------------- tracks
  win.body.appendChild(toolbar);

  /**
   * List the transition colours the run uses, beside what each one means.
   *
   * @returns {void}
   */
  function renderLegend() {
    legend.replaceChildren();
    const used = new Set(state.scenes.map((scene, index) => tintKey(index, scene.continuity, scene.overlap, scene.sound)));
    if (!used.size) {
      legend.style.display = "none";
      return;
    }
    legend.style.display = "flex";
    for (const [key, tint] of Object.entries(TRANSITION_TINTS)) {
      if (!used.has(key)) continue;
      const chip = el("span", "display:inline-flex;align-items:center;gap:6px;white-space:nowrap;flex:0 0 auto");
      chip.appendChild(el("span", `width:9px;height:9px;border-radius:50%;background:${tint.stripe}`));
      chip.appendChild(el("span", `color:${LOOK.text}`, tint.label));
      chip.title = `${key === "opening" ? "The scene the video opens on" : TRANSITIONS[key]?.detail ?? ""}. `
        + "A colour chosen for a scene replaces its transition colour.";
      legend.appendChild(chip);
    }
  }

  const stripHost = el("div", `flex:0 0 ${TIMELINE_HEIGHT}px;min-height:0;`
    + `border-top:1px solid ${LOOK.border}`);
  win.body.appendChild(stripHost);
  const strip = createTimeline({
    logName: LOG_NAME,
    read: () => ({
      rows: state.scenes.map((scene, index) => ({
        seconds: scene.seconds, overlap: scene.overlap, continuity: scene.continuity,
        source: scene.source, title: scene.title, colour: scene.colour, model: scene.model,
        warning: scene.warning, preview: previewOf(index), sound: scene.sound,
      })),
      assets: state.assets,
      selected,
      selectedAsset,
      playhead,
      dragging,
    }),
    onSelect: (index) => {
      selectScene(index);
      refresh(true);
    },
    onSelectAsset: (id) => {
      selectedAsset = id;
      const asset = state.assets.find((each) => each.id === id);
      if (asset && asset.segment > 0) selected = asset.segment - 1;
      tab = "scene";
      refresh(true);
      inspect(asset ?? null);
    },
    onChange: (change) => applyTimelineChange(change),
    onDrop: (target, event) => dropOnTimeline(target, event),
    onMenu: (index, event) => openSceneMenu(index, event),
    onTransitionMenu: (index, event) => openTransitionMenu(index, event),
    onAssetMenu: (id, event) => openAssetMenu(id, event),
  });
  stripHost.appendChild(strip.element);

  const notice = createNotice(win.body, TIMELINE_HEIGHT + 56);

  /**
   * Say something briefly over the window.
   *
   * @param {string} text - What to say.
   * @returns {void}
   */
  function say(text) {
    notice.say(text, /cannot|holds|needs|Only|could not/.test(text) ? "warning" : "accent");
  }

  // -------------------------------------------------------------------- reading
  /**
   * Follow the node that replaced this one, or take the window down when none did.
   *
   * @returns {boolean} True when a replacement was found and taken.
   */
  function follow() {
    if (disposed) return false;
    // An undo rebuilds the graph with new node objects, so the window follows its node's id;
    // another workflow's node with the same id is not this one.
    const replacement = app.graph?.getNodeById?.(node.id);
    if (!replacement || replacement.type !== NODE_ID || replacement === node
        || activeWorkflow() !== workflow) {
      dispose();
      return false;
    }
    WINDOWS.delete(node);
    node = replacement;
    WINDOWS.set(node, self);
    release?.();
    release = joinTicking(node, () => refresh(), LOG_NAME);
    signature = "";
    return true;
  }

  /**
   * The node this window was on left the graph: wait a tick for a rebuild to put it back.
   *
   * @param {object} gone - The node that was removed.
   * @returns {void}
   */
  function nodeRemoved(gone) {
    if (disposed || gone !== node) return;
    setTimeout(() => {
      if (!disposed && !node.graph) follow();
    }, 0);
  }

  /**
   * Choose a scene, dropping an asset choice that belongs to another.
   *
   * @param {number|null} index - The scene, from 0.
   * @returns {void}
   */
  function selectScene(index) {
    if (selected !== index) {
      const asset = state.assets.find((each) => each.id === selectedAsset);
      if (!asset || asset.segment !== (index ?? -1) + 1) selectedAsset = null;
    }
    selected = index;
    tab = "scene";
    if (index === null) return;
    strip.reveal(index);
    const entry = state.layout[index];
    if (entry && sceneAt(state.layout, Math.round(playhead * FPS)) !== index) playhead = entry.start / FPS;
  }

  /**
   * Read the node and the graph again, and redraw whatever changed.
   *
   * @param {boolean} [force] - Redraw even when nothing the node holds has changed.
   * @returns {void}
   */
  function refresh(force = false) {
    if (disposed) return;
    if (!node.graph && !follow()) return;
    const scenes = readScenes(node);
    const assets = readAssets(node);
    tagReferences(assets, scenes.length, socketReferences(node));
    let layout = [];
    try {
      layout = scenes.length ? timeline(scenes) : [];
    } catch (error) {
      console.error(`[${LOG_NAME}] The run could not be laid out:`, error);
    }
    state = { scenes, assets, layout, warnings: [] };
    state.warnings = warnings();
    for (const issue of state.warnings) {
      const scene = scenes[issue.scene ?? -1];
      if (scene && !scene.warning) scene.warning = issue.text;
    }
    if (selected !== null && selected >= scenes.length) selected = scenes.length ? scenes.length - 1 : null;
    if (selected === null && scenes.length) selected = 0;
    if (selectedAsset !== null && !assets.some((asset) => asset.id === selectedAsset)) selectedAsset = null;
    // An inspected asset follows its node's edits, and leaves the screen with its node.
    if (inspected?.id !== null && inspected?.id !== undefined) {
      inspected = assets.find((asset) => asset.id === inspected.id) ?? null;
    }
    const next = JSON.stringify({
      scenes: scenes.map(({ colour, ...rest }) => rest),
      assets: assets.map(({ node: unused, ...rest }) => rest),
      warnings: state.warnings, selected, selectedAsset, tab, showMore,
      wired: WATCHED_SOCKETS.map((name) => wiredInto(node, name)),
      run: ["mode", "aspect_ratio", "megapixels", "width", "height", "prompt_header", "prompt_footer", "loop"]
        .map((name) => valueOf(node, name)),
    });
    if (next === signature && !force) return;
    signature = next;
    win.setBadge(node.title || "MiniMax H3 Conditioning");
    renderToolbar();
    renderSeekMarks();
    renderLegend();
    if (popoverFor === "issues") renderPopover();
    // A field being typed into is left alone; the pane is rebuilt once it is left.
    const active = document.activeElement;
    const typing = pane.contains(active) && (active?.tagName === "TEXTAREA" || active?.tagName === "INPUT");
    if (typing) renderPending = true;
    else renderInspector();
    browser.setRoles(roleEntries());
    strip.repaint();
    drawMonitor();
  }

  /**
   * Everything worth fixing, in the order it is worth saying.
   *
   * @returns {Array<{text: string, scene: number|null, asset: number|null}>} The issues.
   */
  function warnings() {
    const found = [];
    const add = (text, scene = null, asset = null) => found.push({ text, scene, asset });
    const fl = wiredInto(node, "model_fl2va");
    const rf = wiredInto(node, "model_ref2va");
    const mode = String(valueOf(node, "mode", "t2va"));
    const refused = state.layout.indexOf(null);
    if (refused >= 0) add(`Scene ${refused + 1} cannot continue from the scene it names`, refused);
    state.scenes.forEach((scene, index) => {
      const named = `Scene ${index + 1}`;
      if (scene.model === "fl2va" && !fl) add(`${named} asks for fl2va and model_fl2va is not wired`, index);
      if (scene.model === "ref2va" && !rf) add(`${named} asks for ref2va and model_ref2va is not wired`, index);
      const stated = DURATION_LINE.exec(scene.prompt);
      const length = durationOf(framesOf(scene.seconds));
      if (stated && Math.abs(Number(stated[1]) - length) > STATED_TOLERANCE) {
        add(`${named}'s prompt says duration_seconds: ${stated[1]} and its length is ${length}s`, index);
      }
      if (index > 0 && sourceIndex(scene.source, index) < 0 && scene.source !== PREVIOUS_SOURCE && scene.source !== 0) {
        add(`${named} continues from a scene it cannot reach`, index);
      }
      const entry = state.layout[index];
      const own = state.assets.filter((asset) => asset.segment === index + 1 || asset.segment === EVERY_SEGMENT);
      for (const role of Object.keys(MOST)) {
        const count = own.filter((asset) => asset.role === role).length;
        if (count > MOST[role]) add(`${named} has ${count} ${ROLE_NAMES[role].toLowerCase()}s, the model takes ${MOST[role]}`, index);
      }
      if (!entry) return;
      for (const asset of own) {
        if (asset.role !== "keyframe" || asset.segment === EVERY_SEGMENT) continue;
        const guide = asset.frames > 1 ? guideLength(asset.frames) : 1;
        if (resolvedIndex(asset.frame, entry.head, entry.window, guide) === null) {
          add(`${named}: the keyframe ${asset.name} lands outside the scene`, index, asset.id);
        }
      }
    });
    const sounding = state.assets.some((asset) =>
      asset.kind === "sound" || asset.kind === "clip" || asset.role === "reference audio");
    if (sounding && !wiredInto(node, "audio_vae")) {
      add("A clip or sound is placed and audio_vae is not wired");
    }
    if (mode === "fl2va_batched" && state.assets.some((asset) => PINNING.has(asset.role))) {
      add("fl2va_batched takes every pinned frame from images, so pinned assets have nowhere to go");
    }
    for (const asset of state.assets) {
      const scene = asset.segment > 0 ? asset.segment - 1 : null;
      if (asset.segment > state.scenes.length) {
        add(`${asset.name} names scene ${asset.segment}, which has no prompt`, null, asset.id);
      }
      if (asset.kind && !ROLE_KINDS[asset.role]?.includes(asset.kind)) {
        add(`${asset.name} is a ${asset.kind} and cannot be a ${ROLE_NAMES[asset.role]?.toLowerCase() ?? asset.role}`, scene, asset.id);
      }
      if (asset.name === "nothing chosen") add("An asset node has no file chosen and nothing wired", scene, asset.id);
    }
    return found;
  }

  /**
   * Draw the toolbar's figures and the state of its buttons.
   *
   * @returns {void}
   */
  function renderToolbar() {
    const total = state.layout.filter(Boolean).at(-1)?.end ?? 0;
    summary.replaceChildren(
      pill(`${state.scenes.length} scene${state.scenes.length === 1 ? "" : "s"}`),
      el("span", `font:600 13px ${SANS};color:${LOOK.text};font-variant-numeric:tabular-nums`,
        `${durationOf(total).toFixed(2)}s`),
      el("span", `font:12px ${SANS};color:${LOOK.muted}`, `${total} frames`),
      pill(String(valueOf(node, "mode", "t2va"))),
    );
    setDisabled(duplicateButton, selected === null || state.scenes.length >= MAX_ROWS);
    setDisabled(deleteButton, selected === null && selectedAsset === null);
    setDisabled(addButtonEl, state.scenes.length >= MAX_ROWS);
    const count = state.warnings.length;
    issuesButton.textContent = count ? `⚠  ${count} to fix` : "✓  Ready";
    issuesButton.style.color = count ? LOOK.warning : LOOK.success;
    issuesButton.style.fontWeight = "600";
  }

  /**
   * The roles the media bin's menu offers for the chosen scene.
   *
   * @returns {object[]} One `{id, label, kinds}` per role.
   */
  function roleEntries() {
    const entries = [];
    if (selected !== null && state.scenes[selected]) {
      const n = selected + 1;
      const entry = state.layout[selected];
      const inside = entry && playhead * FPS >= entry.start && playhead * FPS < entry.end;
      entries.push(
        { id: `first frame|${n}`, label: `Opening frame of scene ${n}`, kinds: ROLE_KINDS["first frame"] },
        { id: `last frame|${n}`, label: `Closing frame of scene ${n}`, kinds: ROLE_KINDS["last frame"] },
        {
          id: `keyframe|${n}`,
          label: inside ? `Keyframe at the playhead (${clockOf(playhead)})` : `Keyframe at the start of scene ${n}`,
          kinds: ROLE_KINDS.keyframe,
        },
        { id: `reference picture|${n}`, label: `Picture reference in scene ${n}`, kinds: ROLE_KINDS["reference picture"] },
        { id: `reference clip|${n}`, label: `Clip reference in scene ${n}`, kinds: ROLE_KINDS["reference clip"] },
        { id: `reference audio|${n}`, label: `Sound reference in scene ${n}`, kinds: ROLE_KINDS["reference audio"] },
      );
    }
    entries.push(
      { id: "reference picture|0", label: "Picture reference in every scene", kinds: ROLE_KINDS["reference picture"] },
      { id: "reference clip|0", label: "Clip reference in every scene", kinds: ROLE_KINDS["reference clip"] },
      { id: "reference audio|0", label: "Sound reference in every scene", kinds: ROLE_KINDS["reference audio"] },
      { id: "new|0", label: "A new scene opening on this frame", kinds: ["picture", "clip"] },
    );
    return entries;
  }

  // -------------------------------------------------------------------- inspector
  /**
   * Draw the chosen scene, or the video's settings.
   *
   * @returns {void}
   */
  function renderInspector() {
    tabs.show(tab);
    const scrolled = pane.scrollTop;
    pane.replaceChildren();
    cards.clear();
    if (tab === "video") renderVideo();
    else if (tab === "llm") renderLlm();
    else if (!state.scenes.length) renderWelcome();
    else renderScene();
    pane.scrollTop = scrolled;
  }

  /**
   * Draw the first steps, for a node with no scene yet.
   *
   * @returns {void}
   */
  function renderWelcome() {
    const card = el("div", `${CARD};margin:auto;width:min(560px, 100%);padding:22px 24px;gap:0`);
    if (modelSource(node).wired) {
      const host = el("div", "display:flex;flex-direction:column");
      card.appendChild(host);
      writerHost = host;
      renderWriter(host, false);
      const start = makeButton("+  Add the first scene by hand", () => addScene(), { kind: "tool" });
      start.style.alignSelf = "flex-start";
      start.style.marginTop = "14px";
      card.appendChild(start);
      pane.appendChild(card);
      return;
    }
    card.appendChild(el("div", `font:600 15px ${SANS};color:${LOOK.text};margin-bottom:6px`,
      "Build the video scene by scene"));

    const steps = [
      "Add a scene and write its prompt.",
      "Drag media onto the timeline to pin a frame or add a reference.",
      "Set each scene's length and transition.",
    ];
    steps.forEach((text, index) => {
      const step = el("div", "display:flex;gap:10px;align-items:flex-start;margin:0 0 10px");
      step.appendChild(el("span", "flex:0 0 22px;height:22px;border-radius:50%;display:flex;align-items:center;"
        + `justify-content:center;font:600 12px ${SANS};border:1px solid ${LOOK.border};color:${LOOK.text}`,
      String(index + 1)));
      step.appendChild(el("span", `font:13px/1.5 ${SANS};color:${LOOK.text}`, text));
      card.appendChild(step);
    });
    const start = makeButton("+  Add the first scene", () => addScene(), { kind: "primary" });
    start.style.alignSelf = "flex-start";
    start.style.marginTop = "8px";
    card.appendChild(start);
    const wire = factLine("Wire a language model into vlm_clip to write the video from a description.",
      "As Load CLIP with qwen3vl_8b_fp8_scaled");
    wire.style.marginTop = "14px";
    card.appendChild(wire);
    pane.appendChild(card);
  }

  /**
   * Draw the language model settings: the model, the system prompt and how words are drawn.
   *
   * @returns {void}
   */
  function renderLlm() {
    const settings = llmSettings(node);
    const source = modelSource(node);
    pane.appendChild(heading("Model"));
    const model = el("div", "display:flex;gap:8px;align-items:baseline;min-width:0");
    model.appendChild(el("span", `font:600 13px ${SANS};color:${source.wired ? LOOK.success : LOOK.muted}`, source.wired ? "●" : "○"));
    model.appendChild(el("span", `font:13px ${MONO};color:${source.wired ? LOOK.text : LOOK.muted}`, "vlm_clip"));
    model.appendChild(el("span", `font:12px ${SANS};color:${LOOK.muted};white-space:nowrap;overflow:hidden;text-overflow:ellipsis`,
      source.wired ? [source.name, source.detail].filter(Boolean).join(" · ") : "not wired"));
    model.title = "Never loaded by a render";
    pane.appendChild(model);

    const rules = settings.system ?? systemText;
    const edited = settings.system !== null && settings.system !== undefined && settings.system !== systemText;
    const reset = makeButton("Reset to default", () => saveSettings({ system: null }, true), {
      kind: "tool", hint: "Use the nodes' own rules",
    });
    setDisabled(reset, !edited);
    pane.appendChild(heading("System prompt", reset));
    const box = textBox(20, `min-height:320px;font:12px/1.55 ${MONO}`);
    box.placeholder = systemText ? "" : "Loading the default rules…";
    show(box, rules);
    box.addEventListener("change", () => saveSettings({ system: box.value === systemText ? null : box.value }, true));
    const words = String(rules ?? "").split(/\s+/).filter(Boolean).length;
    const rulesField = field("Rules", box, `${edited ? "edited" : "default"} · ${words} words`,
      "Every scene, header, footer and rewrite is written under these rules.");
    pane.appendChild(rulesField);

    pane.appendChild(heading("Sampling"));
    const grid = el("div", "display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:14px 18px");
    for (const [label, key, limits, tip] of [
      ["Temperature", "temperature", { min: 0.01, max: 2, step: 0.01 }, "0.7 is balanced, 0.3 keeps to the likeliest words, 1.1 wanders."],
      ["Top P", "top_p", { min: 0, max: 1, step: 0.01 }, "Picks among the likeliest words whose chances add up to this; 1.0 is off."],
      ["Top K", "top_k", { min: 0, max: 1000, step: 1 }, "Picks among this many likeliest words; 0 is off."],
      ["Min P", "min_p", { min: 0, max: 1, step: 0.01 }, "Drops words under this share of the likeliest; 0 is off."],
      ["Repetition penalty", "repetition_penalty", { min: 0, max: 5, step: 0.01 }, "Above 1.0 makes a used word less likely again; 1.0 is off."],
    ]) {
      const number = makeNumber(settings[key], limits);
      number.input.addEventListener("change", () => {
        const value = Number(number.input.value);
        if (!Number.isFinite(value)) return;
        const kept = Math.max(limits.min, Math.min(limits.max, key === "top_k" ? Math.round(value) : value));
        saveSettings({ [key]: kept });
      });
      grid.appendChild(field(label, number.box, "", tip));
    }
    for (const [label, key, entries, tip] of [
      ["Thinking", "thinking", [["false", "Off"], ["true", "On"]], "On lets a reasoning model such as Qwen3 think before each answer."],
      ["Fast decode", "fast_decode", [["true", "On"], ["false", "Off"]], "Off decodes exactly as core Generate Text."],
    ]) {
      const choice = makeSelect(entries, String(Boolean(settings[key])));
      choice.addEventListener("change", () => saveSettings({ [key]: choice.value === "true" }));
      grid.appendChild(field(label, choice, "", tip));
    }
    pane.appendChild(grid);

    pane.appendChild(heading("Writing"));
    const writing = el("div", "display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:14px 18px");
    const plan = makeSelect([["true", "Planned"], ["false", "All cuts"]], String(Boolean(settings.plan)));
    plan.addEventListener("change", () => saveSettings({ plan: plan.value === "true" }));
    writing.appendChild(field("Transitions", plan, "", "Planned reads the written scenes back and picks each cut or carry."));
    const seedMode = makeSelect([["false", "New each time"], ["true", "Fixed"]], String(Boolean(settings.fixed_seed)));
    seedMode.addEventListener("change", () => saveSettings({ fixed_seed: seedMode.value === "true" }, true));
    writing.appendChild(field("Seed", seedMode, "", "Fixed writes the same answer for the same request."));
    if (settings.fixed_seed) {
      const seed = makeNumber(settings.seed, { min: 0, max: 2 ** 32 - 1, step: 1 });
      seed.input.addEventListener("change", () => saveSettings({ seed: Math.max(0, Math.round(Number(seed.input.value) || 0)) }));
      writing.appendChild(field("Seed value", seed.box));
    }
    pane.appendChild(writing);
  }

  /**
   * Draw the chosen scene's prompt, timing and assets.
   *
   * @returns {void}
   */
  function renderScene() {
    const index = selected ?? 0;
    const scene = state.scenes[index];
    const entry = state.layout[index];
    if (!scene) return;

    // The heading: colour, name, place in the video, more.
    const head = el("div", "display:flex;align-items:center;gap:10px");
    const swatch = el("button", "flex:0 0 auto;width:18px;height:18px;border-radius:50%;cursor:pointer;padding:0;"
      + `background:${(scene.colour ?? sceneTint(index, scene.continuity, scene.overlap, scene.sound)).stripe};`
      + `border:2px solid ${LOOK.body};box-shadow:0 0 0 1px ${LOOK.border}`);
    swatch.type = "button";
    swatch.title = scene.colour ? "Colour this scene"
      : `Coloured by its transition: ${sceneTint(index, scene.continuity, scene.overlap, scene.sound).label}. Click to choose a colour`;
    swatch.addEventListener("click", (event) => openColourMenu(index, event));
    head.appendChild(swatch);
    const before = makeButton("‹", () => { selectScene(index - 1); refresh(true); }, { kind: "tool", hint: "The scene before (←)" });
    const after = makeButton("›", () => { selectScene(index + 1); refresh(true); }, { kind: "tool", hint: "The scene after (→)" });
    setDisabled(before, index <= 0);
    setDisabled(after, index >= state.scenes.length - 1);
    head.appendChild(before);
    head.appendChild(el("span", `font:600 17px ${SANS};color:${LOOK.text}`, `Scene ${index + 1}`));
    head.appendChild(el("span", `font:13px ${SANS};color:${LOOK.muted}`, `of ${state.scenes.length}`));
    head.appendChild(after);
    head.appendChild(el("span", "flex:1 1 auto"));
    head.appendChild(makeButton("More ▾", (event) => openSceneMenu(index, event), { kind: "tool", hint: "Insert, duplicate, delete, colour" }));
    pane.appendChild(head);
    if (entry) {
      const end = drawnEndOf(state.layout, index);
      pane.appendChild(el("div", `font:12px ${SANS};color:${LOOK.muted};margin-top:-8px`,
        `${clockOf(entry.start / FPS)} → ${clockOf(end / FPS)}  ·  ${end - entry.start} frames on screen`
        + `  ·  ${entry.window} sampled` + (entry.head ? `, ${entry.head} carried` : "")));
    }

    // The prompt on the left, everything else on the right, one column when narrow.
    const columns = el("div", "display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));"
      + "gap:18px 26px;align-items:stretch");
    const left = el("div", "display:flex;flex-direction:column;min-width:0");
    const right = el("div", "display:flex;flex-direction:column;gap:14px;min-width:0");
    columns.append(left, right);
    pane.appendChild(columns);

    const promptBox = el("div", "display:flex;flex-direction:column;gap:5px;flex:1 1 auto");
    const promptHead = el("div", "display:flex;align-items:center;gap:8px;min-height:24px");
    promptHead.appendChild(el("span", LABEL, "Prompt"));
    promptHead.appendChild(el("span", "flex:1 1 auto"));
    const blank = !scene.prompt.trim() || scene.prompt === NEW_SCENE_PROMPT;
    const rewriting = busy(jobs.rewrite) && jobs.rewrite.row === scene.row;
    const rewriteButton = makeButton(rewriting ? "Rewriting…" : blank ? "Write…" : "Rewrite…",
      () => openRewrite(index, rewriteButton), {
        kind: "tool", hint: blank ? "Write it with the language model" : "Rewrite it with the language model",
      });
    rewriteButton.dataset.idle = blank ? "Write…" : "Rewrite…";
    rewriteAnchor = rewriteButton;
    promptHead.appendChild(rewriteButton);
    promptBox.appendChild(promptHead);
    const prompt = textBox(10, "min-height:220px;flex:1 1 auto");
    prompt.placeholder = "What happens in this scene: who is in it, what they do, the camera, the sound";
    show(prompt, scene.prompt);
    prompt.addEventListener("input", () => {
      clearTimeout(typingTimer);
      typingTimer = setTimeout(() => commitPrompt(scene.row, prompt.value), TYPING_IDLE_MS);
    });
    prompt.addEventListener("change", () => {
      clearTimeout(typingTimer);
      commitPrompt(scene.row, prompt.value);
    });
    promptBox.appendChild(prompt);
    left.appendChild(promptBox);
    if (scene.prompt === NEW_SCENE_PROMPT) {
      prompt.focus();
      prompt.select();
    }

    // The timing: length, transition, carried frames.
    const timing = el("div", "display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));gap:14px 18px");
    const length = makeNumber(scene.seconds, { min: 0.2, max: 150, step: 0.1 }, "seconds");
    const snapped = snapClip(framesOf(scene.seconds));
    length.input.addEventListener("change", () => {
      const seconds = Math.max(0.2, Math.min(150, Number(length.input.value) || scene.seconds));
      commit(node, () => write(node, `duration_${scene.row}`, seconds));
      refresh();
    });
    timing.appendChild(field("Length", length.box, `${durationOf(snapped).toFixed(2)} s · ${snapped} frames sampled`,
      "Snapped to the model's frame grid. Carried frames and the 5 a cut trims are not on screen."));

    const effective = effectiveTransition(scene.continuity, scene.overlap);
    if (index === 0) {
      const opens = makeSelect([["", "Opens the video"]], "");
      opens.disabled = true;
      opens.style.opacity = "0.6";
      timing.appendChild(field("Transition in", opens));
    } else {
      const choices = ROW_CONTINUITY.map((value) => [value, TRANSITIONS[value]?.name ?? value]);
      const transition = makeSelect(choices, scene.continuity);
      transition.addEventListener("change", () => {
        commit(node, () => write(node, `continuity_${scene.row}`, transition.value));
        refresh();
      });
      timing.appendChild(field("Transition in", transition,
        entry && entry.trimmed > 0 ? `${entry.trimmed} frames trimmed before` : "", describeTransition(scene, entry)));
      const soundChoice = makeSelect([["auto", "As the transition"], ["carry", "Carry over"], ["fresh", "Fresh"]], scene.sound);
      soundChoice.addEventListener("change", () => {
        commit(node, () => write(node, `sound_${scene.row}`, soundChoice.value));
        refresh();
      });
      const resolved = resolvedTransition(scene.continuity, scene.sound);
      const heard = resolved.bridged ? "carried across the cut"
        : resolved.fresh ? "new under the carried shot"
          : CUT_LIKE.includes(resolved.picture) || scene.overlap <= 0 ? "new" : "carried with the shot";
      timing.appendChild(field("Sound", soundChoice, heard,
        "Carry over runs the last scene's sound on across any cut; Fresh gives a carried shot new sound."));
      const sounding = BRIDGING.includes(effective);
      const carries = !CUT_LIKE.includes(effective) || sounding;
      const overlap = makeNumber(scene.overlap, { min: 0, max: 362, step: 1 }, "frames");
      overlap.input.addEventListener("change", () => {
        const frames = Math.max(0, Math.min(362, Math.round(Number(overlap.input.value) || 0)));
        commit(node, () => write(node, `overlap_${scene.row}`, frames));
        refresh();
      });
      const bridged = bridgeFrames(snapOverlapFor(framesOf(scene.seconds), scene.overlap));
      const reach = (least) => Math.max(REFERENCE_FRAMES, least);
      const carried = snapOverlap(scene.overlap);
      if (effective === REFERENCE_VIDEO) {
        timing.appendChild(field("Referenced frames", overlap.box,
          `${reach(scene.overlap > 0 ? carried : 0)} frames`, "Frames of the scene before it references."));
      } else if (carries) {
        const meta = scene.overlap <= 0 ? "0 = cut"
          : effective === AUDIO_REFERENCE ? `${bridged} frames of sound · ${reach(bridged)} referenced`
          : sounding ? `${bridged} frames · ${durationOf(bridged).toFixed(2)} s`
          : `${carried} frames · ${durationOf(carried).toFixed(2)} s`;
        timing.appendChild(field(sounding ? "Sound carried" : "Carried frames", overlap.box, meta,
          sounding ? "Sound of the scene before, carried across the cut in whole clips."
            : "Frames of the scene before that this one continues from. 0 cuts."));
      }
    }
    const hold = makeNumber(scene.hold, { min: 0, max: 1, step: 0.05 }, "");
    hold.input.addEventListener("change", () => {
      const value = Number(hold.input.value);
      if (!Number.isFinite(value)) return;
      commit(node, () => write(node, `strength_${scene.row}`, Math.round(Math.max(0, Math.min(1, value)) * 100) / 100));
      refresh();
    });
    timing.appendChild(field("Hold", hold.box, scene.hold >= 1 ? "as given" : `loosened to ${scene.hold.toFixed(2)}`,
      "How firmly this scene holds its pinned frames and references, transition references included. "
      + "1.0 holds them as given; lower values add noise to them before sampling."));
    right.appendChild(timing);

    // More options, folded.
    const more = makeButton(`${showMore ? "▾" : "▸"}  More options`, () => {
      showMore = !showMore;
      refresh(true);
    }, { kind: "ghost" });
    more.style.alignSelf = "flex-start";
    more.style.marginLeft = "-8px";
    right.appendChild(more);
    if (showMore) {
      const extra = el("div", `${CARD};display:grid;grid-template-columns:repeat(auto-fill,minmax(170px,1fr));`
        + "gap:14px 18px;padding:12px 14px");
      const sources = [[String(PREVIOUS_SOURCE), "The scene before"]];
      for (let earlier = 1; earlier <= index; earlier += 1) sources.push([String(earlier), `Scene ${earlier}`]);
      const resolved = sourceIndex(scene.source, index);
      const sourceValue = scene.source === PREVIOUS_SOURCE || scene.source === 0 || resolved < 0
        ? String(PREVIOUS_SOURCE) : String(resolved + 1);
      const source = makeSelect(sources, sourceValue);
      source.disabled = index === 0;
      source.addEventListener("change", () => {
        commit(node, () => write(node, `source_${scene.row}`, Number(source.value)));
        refresh();
      });
      extra.appendChild(field("Continues from", source, "", "An earlier scene to come back to after a cutaway."));
      const wrap = makeSelect(optionsOf(node, `header_footer_${scene.row}`).map((value) => [value, WRAP_NAMES[value] ?? value]), scene.wrap);
      wrap.addEventListener("change", () => {
        commit(node, () => write(node, `header_footer_${scene.row}`, wrap.value));
        refresh();
      });
      extra.appendChild(field("Shared text", wrap, "", "Which of the video's header and footer wrap this prompt."));
      const model = makeSelect(optionsOf(node, `model_${scene.row}`).map((value) => [value, MODEL_NAMES[value] ?? value]), scene.model);
      model.addEventListener("change", () => {
        commit(node, () => write(node, `model_${scene.row}`, model.value));
        refresh();
      });
      const fl = wiredInto(node, "model_fl2va");
      const rf = wiredInto(node, "model_ref2va");
      const seed = makeNumber(scene.seed, { min: 0, max: 2 ** 32 - 1, step: 1 }, "");
      seed.input.addEventListener("change", () => {
        const value = Math.max(0, Math.round(Number(seed.input.value) || 0));
        commit(node, () => write(node, `seed_${scene.row}`, value));
        refresh();
      });
      extra.appendChild(field("Seed", seed.box, scene.seed > 0 ? "its own" : "run seed + scene",
        "0 samples this scene from H3 Extend Window's seed plus its number; any other value is its own seed."));
      extra.appendChild(field("Model", model, `fl2va ${fl ? "wired" : "not wired"} · ref2va ${rf ? "wired" : "not wired"}`,
        "Automatic picks ref2va for a scene carrying references, fl2va otherwise."));
      right.appendChild(extra);
    }

    // The scene's assets, then the ones every scene shares.
    const own = state.assets.filter((asset) => asset.segment === index + 1);
    right.appendChild(heading("In this scene", own.length ? pill(String(own.length)) : undefined));
    for (const asset of own) right.appendChild(assetCard(asset, entry));
    right.appendChild(dropZone(index, own.length === 0));
    const shared = state.assets.filter((asset) => asset.segment === EVERY_SEGMENT);
    if (shared.length) {
      right.appendChild(heading("In every scene"));
      for (const asset of shared) right.appendChild(assetCard(asset, entry));
    }
    const stray = state.assets.filter((asset) => asset.segment > state.scenes.length);
    if (stray.length) {
      right.appendChild(heading("On no scene"));
      for (const asset of stray) right.appendChild(assetCard(asset, null));
    }
  }

  /**
   * The line under a scene's transition.
   *
   * @param {object} scene - The scene.
   * @param {object} entry - Its layout entry, or undefined.
   * @returns {string} The line.
   */
  function describeTransition(scene, entry) {
    const effective = effectiveTransition(scene.continuity, scene.overlap);
    const detail = TRANSITIONS[effective]?.detail ?? "";
    const text = detail ? `${detail.charAt(0).toUpperCase()}${detail.slice(1)}.` : "";
    if (effective === "cut" && scene.continuity !== "cut") return `${text} An overlap of 0 cuts.`;
    return entry && entry.trimmed > 0 ? `${text} The last ${entry.trimmed} frames before it are dropped.` : text;
  }

  /**
   * One asset's card.
   *
   * @param {object} asset - The asset.
   * @param {object} [entry] - The chosen scene's layout entry.
   * @returns {HTMLElement} The card.
   */
  function assetCard(asset, entry) {
    const chosen = asset.id === selectedAsset;
    const card = el("div", `${ROW};gap:12px;min-width:0;`
      + `background:${chosen ? SELECTED_FILL : LOOK.surface};border-color:${chosen ? LOOK.accent : LOOK.border}`);
    cards.set(asset.id, card);
    card.addEventListener("pointerdown", (event) => {
      if (!event.target.closest?.("button, select, input, textarea")) inspect(asset);
      if (selectedAsset === asset.id) return;
      selectedAsset = asset.id;
      for (const [id, each] of cards) {
        each.style.background = id === asset.id ? SELECTED_FILL : LOOK.surface;
        each.style.borderColor = id === asset.id ? LOOK.accent : LOOK.border;
      }
      strip.repaint();
      renderToolbar();
    });
    const thumb = el("div", "flex:0 0 64px;height:42px;border-radius:5px;overflow:hidden;display:flex;"
      + `align-items:center;justify-content:center;background:${LOOK.body};`
      + `font:18px ${SANS};color:${LOOK.muted}`);
    if (asset.thumb) {
      const image = el("img", "width:100%;height:100%;object-fit:cover;display:block");
      image.src = asset.thumb;
      image.loading = "lazy";
      image.draggable = false;
      image.onerror = () => {
        image.remove();
        thumb.textContent = asset.kind === "clip" ? "▶" : "▣";
      };
      thumb.appendChild(image);
    } else {
      thumb.textContent = asset.kind === "sound" ? "♪" : asset.kind === "clip" ? "▶" : "▣";
    }
    card.appendChild(thumb);

    const middle = el("div", "flex:1 1 auto;min-width:0;display:flex;flex-direction:column;gap:5px");
    const top = el("div", "display:flex;gap:8px;align-items:center;min-width:0;flex-wrap:wrap");
    const fits = ROLES.filter((role) => !asset.kind || ROLE_KINDS[role].includes(asset.kind) || role === asset.role);
    const role = makeSelect(fits.map((value) => [value, ROLE_NAMES[value]]), asset.role);
    role.style.width = "auto";
    role.style.padding = "4px 8px";
    role.title = "Its part in the scene";
    role.addEventListener("change", () => {
      commit(node, () => write(asset.node, "role", role.value));
      refresh();
    });
    top.appendChild(role);
    const where = [[String(EVERY_SEGMENT), "Every scene"]];
    state.scenes.forEach((unused, at) => where.push([String(at + 1), `Scene ${at + 1}`]));
    if (asset.segment > state.scenes.length) where.push([String(asset.segment), `Scene ${asset.segment} (none)`]);
    const scene = makeSelect(where, String(asset.segment));
    scene.style.width = "auto";
    scene.style.padding = "4px 8px";
    scene.title = "Which scene it belongs to";
    scene.addEventListener("change", () => {
      commit(node, () => write(asset.node, "segment", Number(scene.value)));
      const to = Number(scene.value);
      if (to > 0) selected = to - 1;
      refresh();
    });
    top.appendChild(scene);
    if (asset.role === "keyframe" && asset.segment !== EVERY_SEGMENT) {
      const at = makeNumber(durationOf(Math.max(0, asset.frame)).toFixed(2), { min: 0, max: 150, step: 0.04 }, "s in");
      at.input.style.width = "78px";
      at.input.style.padding = "4px 8px";
      at.input.title = `Frame ${asset.frame} of the scene's new frames`;
      at.input.addEventListener("change", () => {
        const frame = Math.max(0, Math.round((Number(at.input.value) || 0) * FPS));
        commit(node, () => write(asset.node, "frame", frame));
        refresh();
      });
      top.appendChild(at.box);
    }
    middle.appendChild(top);
    const name = el("div", `font:12px ${SANS};color:${LOOK.text};white-space:nowrap;overflow:hidden;text-overflow:ellipsis`);
    const tag = String(asset.role).startsWith("reference") && asset.tag ? `${asset.tag}  ` : "";
    name.textContent = `${tag}${asset.name}`;
    name.title = `${asset.file}\nMiniMax H3 Asset #${asset.id}, number ${asset.place + 1} on the chain`;
    if (tag) name.style.fontFamily = MONO;
    middle.appendChild(name);
    card.appendChild(middle);

    card.appendChild(makeButton("Show", () => revealNode(asset.node), {
      kind: "ghost", small: true, hint: "Select its MiniMax H3 Asset node on the canvas",
    }));
    card.appendChild(makeButton("✕", () => removeAsset(asset), {
      kind: "ghost", small: true, hint: "Take it off the video",
    }));
    return card;
  }

  /**
   * The place under a scene's assets a file is dropped on.
   *
   * @param {number} index - The scene, from 0.
   * @param {boolean} empty - Whether the scene has no asset yet.
   * @returns {HTMLElement} The zone.
   */
  function dropZone(index, empty) {
    const zone = el("div", `padding:${empty ? 18 : 10}px 14px;border-radius:8px;text-align:center;`
      + `font:12px ${SANS};color:${LOOK.muted};border:1px dashed ${LOOK.border}`);
    zone.dataset.wasDropZone = "1";
    zone.textContent = empty
      ? "Drop a picture, clip or sound"
      : "Drop another file";
    zone.addEventListener("dragover", (event) => {
      if (!dragging && !Array.from(event.dataTransfer?.types ?? []).includes("Files")) return;
      event.preventDefault();
      event.stopPropagation();
      event.dataTransfer.dropEffect = "copy";
      zone.style.borderColor = LOOK.accent;
      zone.style.background = SELECTED_FILL;
    });
    zone.addEventListener("dragleave", () => {
      zone.style.borderColor = LOOK.border;
      zone.style.background = "";
    });
    zone.addEventListener("drop", async (event) => {
      event.preventDefault();
      event.stopPropagation();
      zone.style.borderColor = LOOK.border;
      zone.style.background = "";
      const files = await filesOf(event);
      if (!files.length) return;
      selected = index;
      openRoleMenu(files[0], event.clientX, event.clientY);
    });
    return zone;
  }

  /**
   * Light every drop zone in the pane while a file is dragged.
   *
   * @param {boolean} on - Whether a drag is under way.
   * @returns {void}
   */
  function lightDropZones(on) {
    for (const zone of pane.querySelectorAll("[data-was-drop-zone]")) {
      zone.style.borderColor = (on ? LOOK.accent : LOOK.border);
      zone.style.color = (on ? LOOK.text : LOOK.muted);
    }
  }

  /**
   * Draw the video's own settings.
   *
   * @returns {void}
   */
  function renderVideo() {
    const grid = el("div", "display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:14px 18px");
    const mode = String(valueOf(node, "mode", "t2va"));
    const modeSelect = makeSelect(optionsOf(node, "mode").map((value) => [value, value]), mode);
    modeSelect.addEventListener("change", () => {
      commit(node, () => write(node, "mode", modeSelect.value));
      refresh();
    });
    grid.appendChild(field("Mode", modeSelect, "", MODE_HINTS[mode] ?? ""));
    const aspect = makeSelect(optionsOf(node, "aspect_ratio").map((value) => [value, value]), valueOf(node, "aspect_ratio"));
    aspect.addEventListener("change", () => {
      commit(node, () => write(node, "aspect_ratio", aspect.value));
      refresh();
    });
    grid.appendChild(field("Aspect ratio", aspect, "", "custom takes the shape of the first picture read."));
    const loop = makeSelect([["false", "Ends on its last scene"], ["true", "Loops to the first frame"]],
      String(Boolean(valueOf(node, "loop", false))));
    loop.addEventListener("change", () => {
      commit(node, () => write(node, "loop", loop.value === "true"));
      refresh();
    });
    grid.appendChild(field("Ending", loop, "",
      "Loops closes the last scene on the video's first frame, so it plays round."));
    const size = canvasSize(valueOf(node, "megapixels", 1), valueOf(node, "width", 0), valueOf(node, "height", 0),
      valueOf(node, "aspect_ratio", "16:9"));
    const canvas = size ? `${size[0]} × ${size[1]}` : "from the first picture";
    for (const [label, name, limits, meta, tip] of [
      ["Megapixels", "megapixels", { min: 0, max: 16, step: 0.05 }, canvas, "0.4 is 832 × 480, 1.0 is 1344 × 736."],
      ["Width", "width", { min: 0, max: 16384, step: 32 }, "0 = auto", "0 works it out from megapixels."],
      ["Height", "height", { min: 0, max: 16384, step: 32 }, "0 = auto", "0 works it out from megapixels."],
    ]) {
      const number = makeNumber(valueOf(node, name, 0), limits, name === "megapixels" ? "MP" : "px");
      number.input.addEventListener("change", () => {
        const value = Number(number.input.value);
        if (!Number.isFinite(value)) return;
        commit(node, () => write(node, name, name === "megapixels" ? value : Math.round(value)));
        refresh();
      });
      grid.appendChild(field(label, number.box, meta, tip));
    }
    pane.appendChild(grid);
    for (const [label, name, hint] of [
      ["Header", "prompt_header", "Put before every scene's prompt, as the cast and wardrobe that hold for the whole video."],
      ["Footer", "prompt_footer", "Put after every scene's prompt, as the soundscape and music for the whole video."],
    ]) {
      const box = focusRing(el("textarea", `${FIELD};resize:vertical;font:13px/1.5 ${SANS};min-height:90px;width:100%;padding:10px 12px`));
      box.rows = 4;
      box.spellcheck = false;
      show(box, valueOf(node, name, ""));
      box.addEventListener("change", () => {
        commit(node, () => write(node, name, box.value));
        refresh();
      });
      pane.appendChild(field(label, box, "", hint));
    }
    pane.appendChild(heading("Wired into the node"));
    const wires = el("div", "display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:8px 18px");
    const sockets = socketReferences(node);
    let heard = 0;
    const referenced = [
      ...sockets.pictures.map((name, at) => [name, `<Picture ${at + 1}>`, "A picture every scene's prompt names."]),
      ...sockets.videos.map((video, at) => [
        video.name, `<Video ${at + 1}>${video.sound ? ` + <Audio ${(heard += 1)}>` : ""}`, "A clip every scene's prompt names.",
      ]),
      ...sockets.sounds.map((name) => [name, `<Audio ${(heard += 1)}>`, "A sound every scene's prompt names."]),
    ];
    const assets = `${state.assets.length} asset${state.assets.length === 1 ? "" : "s"}`;
    for (const [name, meta, tip] of [
      ["model_fl2va", "wired", "Samples the scenes set to fl2va."],
      ["model_ref2va", "wired", "Samples the scenes set to ref2va."],
      ["audio_vae", "wired", "Encodes clips and sounds."],
      ["assets", assets, "The asset chain."],
      ["first_frame", "wired", "The frame scene 1 opens on."],
      ["last_frame", "wired", "The frame the video closes on."],
      ["images", "wired", "Keyframes for fl2va_batched."],
      ...referenced,
    ]) {
      const on = wiredInto(node, name);
      const row = el("div", "display:flex;gap:8px;align-items:baseline;min-width:0");
      row.title = tip;
      row.appendChild(el("span", `font:600 13px ${SANS};color:${(on ? LOOK.success : LOOK.muted)}`, on ? "●" : "○"));
      row.appendChild(el("span", `font:13px ${MONO};color:${on ? LOOK.text : LOOK.muted}`, name));
      row.appendChild(el("span", `font:12px ${SANS};color:${LOOK.muted};white-space:nowrap`, on ? meta : "not wired"));
      wires.appendChild(row);
    }
    pane.appendChild(wires);
  }

  // -------------------------------------------------------------------- changes
  /**
   * Write a prompt edit to its row.
   *
   * @param {number} row - The row, from 1.
   * @param {string} text - The prompt.
   * @returns {void}
   */
  function commitPrompt(row, text) {
    if (valueOf(node, `prompt_${row}`) === text) return;
    if (!text.trim()) {
      say("A scene needs a prompt. Delete the scene to take it out");
      return;
    }
    commit(node, () => write(node, `prompt_${row}`, text));
    refresh();
  }

  /**
   * Apply a gesture the tracks finished.
   *
   * @param {object} change - What the tracks reported.
   * @returns {void}
   */
  function applyTimelineChange(change) {
    if (change.type === "playhead") {
      playhead = change.seconds;
      movedAt = performance.now();
      browser.setRoles(roleEntries());
      strip.repaint();
      drawMonitor();
      return;
    }
    if (change.type === "add") {
      addScene();
      return;
    }
    if (change.type === "edit") {
      selectScene(change.index);
      refresh(true);
      pane.querySelector("textarea")?.focus();
      return;
    }
    const scene = state.scenes[change.index];
    if (change.type === "duration" && scene) {
      commit(node, () => write(node, `duration_${scene.row}`, change.seconds));
    } else if (change.type === "overlap" && scene) {
      commit(node, () => write(node, `overlap_${scene.row}`, change.frames));
    } else if (change.type === "move") {
      moveScene(change.from, change.to);
      return;
    } else if (change.type === "keyframe") {
      const asset = state.assets.find((each) => each.id === change.id);
      if (asset) commit(node, () => write(asset.node, "frame", change.frame));
    } else if (change.type === "reference") {
      const asset = state.assets.find((each) => each.id === change.id);
      if (!asset) return;
      const entry = change.segment > 0 ? state.layout[change.segment - 1] : null;
      if (asset.moment && entry && entry.start <= asset.frame) {
        say(`Frame ${asset.frame} of the video plays in or after scene ${change.segment}. `
          + "Reference frame saves it as a picture for an earlier scene");
        refresh(true);
        return;
      }
      commit(node, () => write(asset.node, "segment", change.segment));
      say(`${asset.name} moved to ${change.segment === EVERY_SEGMENT ? (asset.moment ? "every later scene" : "every scene")
        : `scene ${change.segment}`}`);
    }
    refresh();
  }

  /**
   * Write a new scene's row, after the last.
   *
   * @returns {number} The new scene's index, or -1 when the node holds no more.
   */
  function writeNewScene() {
    const last = state.scenes.length ? state.scenes[state.scenes.length - 1].row : 0;
    const row = last + 1;
    if (state.scenes.length >= MAX_ROWS || row > MAX_ROWS) return -1;
    write(node, `prompt_${row}`, NEW_SCENE_PROMPT);
    write(node, `duration_${row}`, NEW_SCENE_SECONDS);
    write(node, `overlap_${row}`, NEW_SCENE_OVERLAP);
    write(node, `continuity_${row}`, CONTINUITY[0]);
    write(node, `source_${row}`, PREVIOUS_SOURCE);
    write(node, `header_footer_${row}`, "both");
    write(node, `model_${row}`, "auto");
    write(node, `strength_${row}`, 1);
    write(node, `sound_${row}`, "auto");
    write(node, `seed_${row}`, 0);
    const colours = node.properties?.was_row_colours;
    if (colours && Object.hasOwn(colours, String(row))) {
      delete colours[String(row)];
      if (!Object.keys(colours).length) delete node.properties.was_row_colours;
    }
    return state.scenes.length;
  }

  /**
   * Append a scene after the last.
   *
   * @returns {void}
   */
  function addScene() {
    if (state.scenes.length >= MAX_ROWS) {
      say(`The node holds ${MAX_ROWS} scenes at most`);
      return;
    }
    const index = commit(node, () => writeNewScene());
    if (index === undefined || index < 0) {
      say(`Row ${MAX_ROWS} is the node's last. Delete a scene first`);
      return;
    }
    selectScene(index);
    refresh(true);
    strip.reveal(index);
  }

  /**
   * A copy of the scenes with a blank one inserted.
   *
   * @param {number} at - Where it goes, from 0.
   * @returns {void}
   */
  function insertScene(at) {
    if (state.scenes.length >= MAX_ROWS) return;
    const scenes = state.scenes.map((scene) => ({ ...scene }));
    scenes.splice(at, 0, {
      prompt: NEW_SCENE_PROMPT, seconds: NEW_SCENE_SECONDS, overlap: NEW_SCENE_OVERLAP,
      continuity: CONTINUITY[0], source: PREVIOUS_SOURCE, wrap: "both", model: "auto", colourName: null,
    });
    commit(node, () => {
      writeScenes(node, scenes.map((scene) => ({
        ...scene, source: scene.source > at ? scene.source + 1 : scene.source,
      })));
      remapAssets(state.assets, (segment) => (segment > at ? segment + 1 : segment));
    });
    selectScene(at);
    refresh(true);
  }

  /**
   * Copy a scene after itself.
   *
   * @param {number} index - The scene, from 0.
   * @returns {void}
   */
  function duplicateScene(index) {
    if (state.scenes.length >= MAX_ROWS || !state.scenes[index]) return;
    const scenes = state.scenes.map((scene) => ({ ...scene }));
    scenes.splice(index + 1, 0, { ...scenes[index] });
    commit(node, () => {
      writeScenes(node, scenes.map((scene) => ({
        ...scene, source: scene.source > index + 1 ? scene.source + 1 : scene.source,
      })));
      remapAssets(state.assets, (segment) => (segment > index + 1 ? segment + 1 : segment));
    });
    selectScene(index + 1);
    refresh(true);
  }

  /**
   * Take a scene and its assets out of the video.
   *
   * @param {number} index - The scene, from 0.
   * @returns {void}
   */
  function deleteScene(index) {
    if (!state.scenes[index]) return;
    const scenes = state.scenes.map((scene) => ({ ...scene }));
    scenes.splice(index, 1);
    const owned = state.assets.filter((asset) => asset.segment === index + 1).length;
    commit(node, () => {
      writeScenes(node, scenes.map((scene) => ({
        ...scene,
        source: scene.source === index + 1 ? PREVIOUS_SOURCE
          : scene.source > index + 1 ? scene.source - 1 : scene.source,
      })));
      remapAssets(state.assets, (segment) =>
        (segment === index + 1 ? null : segment > index + 1 ? segment - 1 : segment));
    });
    selected = scenes.length ? Math.min(index, scenes.length - 1) : null;
    selectedAsset = null;
    say(`Scene ${index + 1} deleted${owned ? ` with its ${owned} asset${owned === 1 ? "" : "s"}` : ""}. Ctrl+Z brings it back`);
    refresh(true);
  }

  /**
   * Move a scene to another place in the video.
   *
   * @param {number} from - Where it is, from 0.
   * @param {number} to - Where it goes, from 0.
   * @returns {void}
   */
  function moveScene(from, to) {
    const scenes = state.scenes.map((scene) => ({ ...scene }));
    const [moved] = scenes.splice(from, 1);
    scenes.splice(to, 0, moved);
    const order = state.scenes.map((unused, index) => index);
    const [movedIndex] = order.splice(from, 1);
    order.splice(to, 0, movedIndex);
    const newOf = new Map(order.map((old, at) => [old + 1, at + 1]));
    commit(node, () => {
      writeScenes(node, scenes.map((scene) => ({
        ...scene, source: scene.source > 0 ? (newOf.get(scene.source) ?? PREVIOUS_SOURCE) : scene.source,
      })));
      remapAssets(state.assets, (segment) => newOf.get(segment) ?? segment);
    });
    selectScene(to);
    refresh(true);
  }

  /**
   * Delete whatever is chosen: the asset, or else the scene.
   *
   * @returns {void}
   */
  function removeSelection() {
    const asset = state.assets.find((each) => each.id === selectedAsset);
    if (asset) removeAsset(asset);
    else if (selected !== null) deleteScene(selected);
  }

  /**
   * Take one asset off the chain.
   *
   * @param {object} asset - The asset.
   * @returns {void}
   */
  function removeAsset(asset) {
    const removed = commit(node, () => removeFromChain(asset.node));
    if (selectedAsset === asset.id) selectedAsset = null;
    say(removed === false
      ? `${asset.name} is off the video; its node stays, since another node reads it`
      : `${asset.name} removed. Ctrl+Z brings it back`);
    refresh(true);
  }

  /**
   * Add a MiniMax H3 Asset node for a file and chain it in. Called inside a commit.
   *
   * @param {string} label - The file's label.
   * @param {string} role - The part it plays.
   * @param {number} segment - Its scene, from 1, or 0 for every scene.
   * @param {number} frame - Where a keyframe lands, in the scene's new frames.
   * @returns {object} The new node.
   * @throws {Error} The asset node is not registered, or the node has no assets input.
   */
  function makeAsset(label, role, segment, frame) {
    const made = window.LiteGraph?.createNode?.(ASSET_NODE_ID);
    if (!made) throw new Error("MiniMax H3 Asset is not registered on this server");
    made.pos = placeInChain(node, chainOf(node), made);
    node.graph.add(made);
    write(made, "file", label);
    write(made, "role", role);
    write(made, "segment", segment);
    write(made, "frame", frame);
    appendToChain(node, made);
    return made;
  }

  /**
   * Place one file where a drop or a menu said.
   *
   * @param {{label: string, kind: string}} file - The file.
   * @param {object} target - `{action, scene, segment, role, frame}`.
   * @returns {void}
   */
  function place(file, target) {
    const kind = file.kind || kindOf(file.label);
    if (kind && target.action !== "new-scene" && !ROLE_KINDS[target.role]?.includes(kind)) {
      say(`${baseName(file.label)} is a ${kind} and cannot be a ${ROLE_NAMES[target.role].toLowerCase()}`);
      return;
    }
    let made = null;
    if (target.action === "new-scene") {
      const index = commit(node, () => {
        const added = writeNewScene();
        if (added < 0) return -1;
        made = makeAsset(file.label, "first frame", added + 1, 0);
        return added;
      });
      if (index === undefined || index < 0) {
        say(`The node holds ${MAX_ROWS} scenes at most`);
        return;
      }
      selectScene(index);
    } else {
      made = commit(node, () => makeAsset(file.label, target.role, target.segment, target.frame ?? 0));
      if (target.segment > 0) selectScene(target.segment - 1);
    }
    if (!made) {
      say("The asset could not be added. The console says why");
      return;
    }
    selectedAsset = made.id;
    say(`${baseName(file.label)}: ${target.action === "new-scene" ? "a new scene opens on it" : target.label ?? ROLE_NAMES[target.role]}`);
    refresh(true);
  }

  /**
   * The files a drop carries: the dragged media tile, or files from the computer, uploaded.
   *
   * @param {DragEvent} event - The drop.
   * @returns {Promise<Array<{label: string, kind: string}>>} The files, in order.
   */
  async function filesOf(event) {
    if (dragging) return [{ label: dragging.label, kind: dragging.kind }];
    const files = Array.from(event.dataTransfer?.files ?? []).filter((file) => {
      const name = file.name.toLowerCase();
      return Object.values(KIND_EXTENSIONS).some((suffixes) => suffixes.some((suffix) => name.endsWith(suffix)));
    });
    if (!files.length) {
      say("Only pictures, clips and sounds can be placed");
      return [];
    }
    say(`Uploading ${files.length} file${files.length === 1 ? "" : "s"} to the input folder`);
    return browser.upload(files);
  }

  /**
   * Place what was dropped on the tracks.
   *
   * @param {object} target - Where the tracks said it lands.
   * @param {DragEvent} event - The drop.
   * @returns {Promise<void>} Settles once it is placed.
   */
  async function dropOnTimeline(target, event) {
    const files = await filesOf(event);
    if (!files.length || disposed) return;
    if (target.action === "reference") {
      for (const file of files) place(file, target);
      return;
    }
    place(files[0], target);
    if (files.length > 1) say(`${files.length - 1} more uploaded to Media; drag each where it goes`);
  }

  /**
   * Place a file picked from the media bin's menu.
   *
   * @param {object} entry - The file, as the bin lists it.
   * @param {string} roleId - `<role>|<scene>`.
   * @returns {void}
   */
  function chooseFromBin(entry, roleId) {
    const [role, segmentText] = String(roleId).split("|");
    const segment = Number(segmentText) || 0;
    const file = { label: entry.label, kind: entry.kind };
    if (role === "new") {
      place(file, { action: "new-scene" });
      return;
    }
    if (!ROLES.includes(role)) return;
    let frame = 0;
    if (role === "keyframe" && segment > 0) {
      const layout = state.layout[segment - 1];
      if (layout) {
        const at = Math.round(playhead * FPS) - layout.start;
        frame = Math.max(0, Math.min(Math.max(0, layout.window - layout.head - 1), at));
      }
    }
    place(file, {
      action: role.startsWith("reference") ? "reference" : "pin", segment, role, frame,
      label: roleEntries().find((each) => each.id === roleId)?.label,
    });
  }

  /**
   * Open the menu of where a dropped file may go.
   *
   * @param {{label: string, kind: string}} file - The file.
   * @param {number} x - Where, in client pixels.
   * @param {number} y - Where, in client pixels.
   * @returns {void}
   */
  function openRoleMenu(file, x, y) {
    const kind = file.kind || kindOf(file.label);
    const items = [{ header: baseName(file.label) }];
    for (const entry of roleEntries()) {
      if (kind && !entry.kinds.includes(kind)) continue;
      items.push({ label: entry.label, onSelect: () => chooseFromBin(file, entry.id) });
    }
    openPopupMenu({ items, x, y, logName: LOG_NAME });
  }

  /**
   * Select a node on the canvas and bring it into view.
   *
   * @param {object} target - The node.
   * @returns {void}
   */
  function revealNode(target) {
    const canvas = app.canvas;
    if (!canvas || !target) return;
    try {
      canvas.deselectAll?.();
      canvas.selectNode?.(target);
      canvas.centerOnNode?.(target);
      canvas.setDirty?.(true, true);
      win.collapse();
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to show the asset node:`, error);
    }
  }

  // -------------------------------------------------------------------- menus
  /**
   * The colour swatches for one scene.
   *
   * @param {number} index - The scene, from 0.
   * @returns {object[]} Menu rows.
   */
  function colourItems(index) {
    const scene = state.scenes[index];
    return [
      { header: "Colour" },
      {
        swatches: Object.entries(ROW_PALETTE).map(([name, colour]) => ({ name, colour: colour.stripe, title: colour.label })),
        current: scene?.colourName,
        onPick: (name) => {
          setRowColour(node, scene.row, name);
          refresh(true);
        },
      },
      {
        label: "Colour by transition", checked: !scene?.colourName,
        detail: sceneTint(index, scene?.continuity, scene?.overlap, scene?.sound).label,
        colour: sceneTint(index, scene?.continuity, scene?.overlap, scene?.sound).stripe,
        onSelect: () => {
          setRowColour(node, scene.row, null);
          refresh(true);
        },
      },
    ];
  }

  /**
   * Open the colour menu of one scene.
   *
   * @param {number} index - The scene, from 0.
   * @param {MouseEvent} event - What opened it.
   * @returns {void}
   */
  function openColourMenu(index, event) {
    openPopupMenu({ items: colourItems(index), x: event.clientX, y: event.clientY, logName: LOG_NAME });
  }

  /**
   * Open the menu of one scene.
   *
   * @param {number} index - The scene, from 0.
   * @param {MouseEvent} event - What opened it.
   * @returns {void}
   */
  function openSceneMenu(index, event) {
    const scene = state.scenes[index];
    if (!scene) return;
    const full = state.scenes.length >= MAX_ROWS;
    const items = [
      { header: `Scene ${index + 1}` },
      { label: "Insert a scene before", disabled: full, onSelect: () => insertScene(index) },
      { label: "Insert a scene after", disabled: full, onSelect: () => insertScene(index + 1) },
      { label: "Duplicate", detail: "D", disabled: full, onSelect: () => duplicateScene(index) },
      { separator: true },
    ];
    if (index > 0) {
      items.push({ header: "Transition in" });
      for (const value of ROW_CONTINUITY) {
        items.push({
          label: TRANSITIONS[value]?.name ?? value, checked: value === scene.continuity,
          onSelect: () => {
            commit(node, () => write(node, `continuity_${scene.row}`, value));
            refresh();
          },
        });
      }
      items.push({ separator: true });
    }
    items.push({ header: "Model" });
    for (const value of optionsOf(node, `model_${scene.row}`)) {
      items.push({
        label: MODEL_NAMES[value] ?? value, checked: value === scene.model,
        onSelect: () => {
          commit(node, () => write(node, `model_${scene.row}`, value));
          refresh();
        },
      });
    }
    items.push({ separator: true }, ...colourItems(index), { separator: true });
    items.push({ label: "Delete scene", detail: "Del", danger: true, onSelect: () => deleteScene(index) });
    openPopupMenu({ items, x: event.clientX, y: event.clientY, logName: LOG_NAME });
  }

  /**
   * Open the transition menu between a scene and the one before it.
   *
   * @param {number} index - The scene, from 0.
   * @param {MouseEvent} event - What opened it.
   * @returns {void}
   */
  function openTransitionMenu(index, event) {
    const scene = state.scenes[index];
    if (!scene || index === 0) return;
    selectScene(index);
    refresh(true);
    const items = [{ header: `Into scene ${index + 1}` }];
    for (const value of ROW_CONTINUITY) {
      const shown = TRANSITIONS[value];
      items.push({
        label: `${shown?.glyph ?? ""}  ${shown?.name ?? value}`, detail: value, checked: value === scene.continuity,
        colour: TRANSITION_TINTS[value]?.stripe,
        onSelect: () => {
          commit(node, () => {
            write(node, `continuity_${scene.row}`, value);
            const cuts = CUT_LIKE.includes(value);
            if (!cuts && scene.overlap <= 0) write(node, `overlap_${scene.row}`, NEW_SCENE_OVERLAP);
          });
          refresh();
        },
      });
    }
    openPopupMenu({ items, x: event.clientX, y: event.clientY, logName: LOG_NAME });
  }

  /**
   * Open the menu of one asset.
   *
   * @param {number} id - The asset node's id.
   * @param {MouseEvent} event - What opened it.
   * @returns {void}
   */
  function openAssetMenu(id, event) {
    const asset = state.assets.find((each) => each.id === id);
    if (!asset) return;
    const items = [{ header: asset.name }];
    for (const role of ROLES) {
      if (asset.kind && !ROLE_KINDS[role].includes(asset.kind)) continue;
      items.push({
        label: ROLE_NAMES[role], checked: role === asset.role,
        onSelect: () => {
          commit(node, () => write(asset.node, "role", role));
          refresh();
        },
      });
    }
    items.push(
      { separator: true },
      { label: "Show its node on the canvas", onSelect: () => revealNode(asset.node) },
      { label: "Remove", detail: "Del", danger: true, onSelect: () => removeAsset(asset) },
    );
    openPopupMenu({ items, x: event.clientX, y: event.clientY, logName: LOG_NAME });
  }

  // -------------------------------------------------------------------- keys
  // Escape closes the Write dialog, a popover or a full-size asset before the window itself.
  win.element.addEventListener("keydown", (event) => {
    if (event.key !== "Escape") return;
    if (dialog.style.display !== "none") {
      closeWriter();
    } else if (popoverFor) {
      popoverFor = "";
      renderPopover();
    } else if (inspected) {
      inspect(null);
    } else {
      return;
    }
    event.preventDefault();
    event.stopImmediatePropagation();
  }, true);
  win.element.addEventListener("keydown", (event) => {
    if (typingInto(event) || event.ctrlKey || event.metaKey || event.altKey) return;
    let handled = true;
    const key = event.key;
    if (key === "Delete" || key === "Backspace") {
      removeSelection();
    } else if (key === "ArrowLeft" || key === "ArrowRight") {
      if (state.scenes.length) {
        const step = key === "ArrowLeft" ? -1 : 1;
        selectScene(Math.max(0, Math.min(state.scenes.length - 1, (selected ?? 0) + step)));
        refresh(true);
      }
    } else if (key === "n" || key === "N") {
      addScene();
    } else if (key === "d" || key === "D") {
      if (selected !== null) duplicateScene(selected);
    } else if (key === "+" || key === "=") {
      strip.zoomBy(1.3);
    } else if (key === "-" || key === "_") {
      strip.zoomBy(1 / 1.3);
    } else if (key === "f" || key === "F") {
      strip.fit();
    } else if (key === " ") {
      togglePlay();
    } else if (key === "v" || key === "V") {
      setView(view === "final" ? "preview" : "final");
    } else if (key === "Home") {
      stepScene(-1);
    } else if (key === "End") {
      stepScene(1);
    } else {
      handled = false;
    }
    if (handled) {
      event.preventDefault();
      event.stopPropagation();
    }
  });

  // Drops anywhere else in the window are refused rather than loaded as a workflow.
  win.element.addEventListener("dragover", (event) => {
    if (event.defaultPrevented) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "none";
  });
  win.element.addEventListener("drop", (event) => {
    event.preventDefault();
    event.stopPropagation();
  });

  const self = {
    open,
    close,
    toggle: () => (win.isOpen() ? close() : open()),
    nodeRemoved,
    dispose,
  };

  /**
   * Show the window and start following the graph.
   *
   * @returns {void}
   */
  function open() {
    if (disposed) return;
    win.open();
    win.expand();
    signature = "";
    refresh(true);
    release?.();
    release = joinTicking(node, () => refresh(), LOG_NAME);
    browser.refresh();
    strip.fit();
    let keyed = false;
    withGraphChange(() => {
      keyed = ensurePreviewKey(node);
    });
    // A property is not a widget or a link, so the workflow is told it changed.
    if (keyed) workflow?.changeTracker?.checkState?.();
    // A run made before the node had a key filed its sheets under the node's id.
    sheets.reload(keyed ? executionId(node) : undefined).then(() => recoverFinished());
  }

  /**
   * Hide the window and stop following the graph.
   *
   * @returns {void}
   */
  function close() {
    clearTimeout(typingTimer);
    release?.();
    release = null;
    closePopupMenu();
    popoverFor = "";
    renderPopover();
    closeWriter();
    stopPlaying();
    win.close();
  }

  /**
   * Take the window down for good.
   *
   * @returns {void}
   */
  function dispose() {
    if (disposed) return;
    disposed = true;
    clearTimeout(typingTimer);
    notice.dispose();
    release?.();
    release = null;
    closePopupMenu();
    stopPlaying();
    finalVideo.removeAttribute("src");
    finalVideo.load();
    sheets.dispose();
    screenSize.disconnect();
    mainSize.disconnect();
    stopWatchingScreen();
    for (const [name, listener] of Object.entries(runListeners)) api.removeEventListener(name, listener);
    for (const [name, listener] of Object.entries(writerListeners)) api.removeEventListener(name, listener);
    try {
      strip.dispose();
      browser.dispose();
      win.dispose();
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to release the window:`, error);
    }
    WINDOWS.delete(node);
  }

  return self;
}

/**
 * The Prompt Timeline for a node, built on first use.
 *
 * @param {object} node - The conditioning node.
 * @returns {object} The window.
 */
function windowOf(node) {
  let held = WINDOWS.get(node);
  if (!held) {
    held = createPromptTimeline(node);
    WINDOWS.set(node, held);
  }
  return held;
}

app.registerExtension({
  name: EXT_NAME,
  settings: [
    {
      id: SETTING_ID,
      category: ["WAS Node Suite", "MiniMax H3", "Prompt Timeline"],
      name: "Show the Open Prompt Timeline button",
      tooltip:
        "Draw a button at the top of MiniMax H3 Conditioning that opens the Prompt Timeline, a "
        + "window where scenes, transitions, pinned frames and references are laid out on tracks "
        + "and placed by dragging. Everything it does is written to the node's own rows and to a "
        + "chain of MiniMax H3 Asset nodes, so the graph runs the same with the window closed. "
        + "This applies to nodes added after the setting changes.",
      type: "boolean",
      defaultValue: true,
    },
  ],

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;
    const proto = nodeType.prototype;
    if (proto.__was_h3_prompt_timeline_wrapped) return;
    proto.__was_h3_prompt_timeline_wrapped = true;

    const originalOnConfigure = proto.onConfigure;
    proto.onConfigure = function (...args) {
      const result = originalOnConfigure?.apply(this, args);
      // Runs once the load or paste has wired every node.
      queueMicrotask(() => {
        // A pasted copy carries the original's key; it gets one of its own.
        if (this.properties?.[PREVIEW_KEY]) withGraphChange(() => ensurePreviewKey(this));
        try {
          const chained = chainNumberedSockets(this);
          if (chained) console.info(`[${EXT_NAME}] Chained ${chained} asset node(s) from numbered sockets on #${this.id}.`);
        } catch (error) {
          console.error(`[${EXT_NAME}] Failed to chain the numbered asset sockets:`, error);
        }
      });
      return result;
    };

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      if (!enabled()) return result;
      try {
        addButton(this, {
          name: BUTTON_NAME,
          label: BUTTON_LABEL,
          before: "mode",
          onClick: (node) => windowOf(node).toggle(),
        });
        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            WINDOWS.get(this)?.nodeRemoved(this);
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to close the Prompt Timeline:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to add the Prompt Timeline button:`, error);
      }
      return result;
    };
  },
});
