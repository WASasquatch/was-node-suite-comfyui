/**
 * Point Cloud Navigator's panel: the cloud in a live view to fly or orbit through, a flight
 * recorder, placed keys, playback, and camera rig import and export. Writes the `path` widget.
 */

import { app } from "../../scripts/app.js";
import { CARD_LOOK, CHIP_TONES, cardChip, cardRule } from "./interface/card.mjs";
import { captureWheel } from "./interface/pointer.mjs";
import { openPopupMenu } from "./interface/popup_menu.mjs";
import { nodeLocator, ownOutput } from "./interface/preview.mjs";
import { surfaceRatio, watchSurfaceRatio } from "./interface/resolution.mjs";
import { themeVar } from "./interface/theme.mjs";
import { appendInterfaceWidget } from "./interface/widget.mjs";

const EXT_NAME = "WASNodeSuite.PointCloudNavigatorUI";

// three.js, loaded when the first navigator is built.
let THREE = null;
let threeLoading = null;

/**
 * Load three.js once for the page.
 *
 * @returns {Promise<object>} The module.
 */
function loadThree() {
  threeLoading ??= import("./vendor/three/three.module.mjs").then((module) => {
    THREE = module;
    return module;
  });
  return threeLoading;
}
const NODE_ID = "WASPointCloudNavigator";
const UI_WIDGET_NAME = "was_point_cloud_navigator_ui";
const UI_WIDGET_TYPE = "was_point_cloud_navigator";

// Height the panel opens at, in node units, and the width a new node opens at.
const PANEL_HEIGHT = 560;
const PANEL_MIN_WIDTH = 270;

// The only address a run may hand the panel: the pack's asset route and a hex key.
const ASSET_PATTERN = /^\/was\/threejs\/api\/asset\?key=[0-9a-f]{32}$/;

// Seconds between placed keys, and between recorded samples.
const KEY_SPACING = 1.0;
const SAMPLE_SECONDS = 1 / 30;

// Most samples one recording keeps, a little over 20 minutes at the sample rate.
const MAX_SAMPLES = 36000;

// Fly speed as a share of the scene depth per second, and the wheel's step on it.
const BASE_SPEED = 0.25;
const SPEED_STEP = 1.15;

// Degrees the view turns per pixel of drag.
const LOOK_RATE = 0.18;

// Output frame shapes the guide can show, as width over height.
const ASPECTS = { "16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1, "4:3": 4 / 3 };

// Bytes one point takes in the cloud a run sends: position, spread and colour.
const POINT_BYTES = 19;

// Widest dot drawn, as a radius in pixels, and a dot's radius as a multiple of its spread, as the
// node's own render draws them.
const MAX_RADIUS = 24;
const COVER = 1.5;

// Round dots sized from each point's spread and distance, the size held under the widest dot.
const DOT_VERTEX = `
attribute float spread;
attribute vec3 tint;
uniform float focal;
uniform float widest;
varying vec3 vTint;
void main() {
  vec4 view = modelViewMatrix * vec4(position, 1.0);
  gl_Position = projectionMatrix * view;
  gl_PointSize = clamp(2.0 * ${COVER.toFixed(1)} * spread * focal / max(-view.z, 1e-6), 1.0, widest);
  vTint = tint;
}`;
const DOT_FRAGMENT = `
varying vec3 vTint;
void main() {
  vec2 offset = gl_PointCoord - 0.5;
  if (dot(offset, offset) > 0.25) discard;
  gl_FragColor = vec4(vTint, 1.0);
}`;

// Bytes one fitted splat takes in what a run sends: position, covariance, colour and opacity.
const SPLAT_BYTES = 40;

// Splats per row of the textures fitted splats are held in.
const SPLAT_ROW = 2048;

// Depth buckets the back to front sort uses.
const SORT_BUCKETS = 65536;

// A fitted splat as an ellipse facing the camera, projected from its covariance.
const SPLAT_VERTEX = `
precision highp float;
precision highp int;
attribute float order;
uniform highp sampler2D centres;
uniform highp sampler2D shapesA;
uniform highp sampler2D shapesB;
uniform highp sampler2D tints;
uniform float focal;
uniform vec2 viewport;
varying vec4 vTint;
varying vec2 vOffset;
void main() {
  int index = int(order);
  ivec2 at = ivec2(index % ${SPLAT_ROW}, index / ${SPLAT_ROW});
  vec4 view = modelViewMatrix * vec4(texelFetch(centres, at, 0).xyz, 1.0);
  vec4 clip = projectionMatrix * view;
  float depth = -view.z;
  if (depth <= 0.0 || abs(clip.x) > 1.3 * clip.w || abs(clip.y) > 1.3 * clip.w) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    return;
  }
  vec4 a = texelFetch(shapesA, at, 0);
  vec4 b = texelFetch(shapesB, at, 0);
  mat3 cov = mat3(a.x, a.y, a.z, a.y, a.w, b.x, a.z, b.x, b.y);
  float inverse = focal / depth;
  mat3 jac = mat3(inverse, 0.0, 0.0, 0.0, inverse, 0.0, inverse * view.x / depth, inverse * view.y / depth, 0.0);
  mat3 t = jac * mat3(modelViewMatrix);
  mat3 flat2 = t * cov * transpose(t);
  float p = flat2[0][0] + 0.3;
  float q = flat2[0][1];
  float r = flat2[1][1] + 0.3;
  float middle = 0.5 * (p + r);
  float reach = length(vec2(0.5 * (p - r), q));
  float major = middle + reach;
  float minor = middle - reach;
  if (minor <= 0.0) {
    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
    return;
  }
  vec2 raw = vec2(q, major - p);
  vec2 along = dot(raw, raw) > 1e-12 ? normalize(raw) : vec2(1.0, 0.0);
  vec2 axisA = min(sqrt(2.0 * major), 1024.0) * along;
  vec2 axisB = min(sqrt(2.0 * minor), 1024.0) * vec2(-along.y, along.x);
  vTint = texelFetch(tints, at, 0);
  vOffset = position.xy;
  gl_Position = vec4(clip.xy / clip.w + (position.x * axisA + position.y * axisB) * 2.0 / viewport, clip.z / clip.w, 1.0);
}`;
const SPLAT_FRAGMENT = `
precision highp float;
varying vec4 vTint;
varying vec2 vOffset;
void main() {
  float power = -dot(vOffset, vOffset);
  if (power < -4.0) discard;
  float alpha = min(0.99, vTint.a * exp(power));
  if (alpha < 1.0 / 255.0) discard;
  gl_FragColor = vec4(vTint.rgb * alpha, alpha);
}`;

/**
 * A texture holding four values per splat, one splat per texel.
 *
 * @param {Float32Array|Uint8Array} values Four values per splat, in splat order.
 * @param {number} count Splats held.
 * @returns {object} The texture.
 */
function splatTexture(values, count) {
  const rows = Math.max(1, Math.ceil(count / SPLAT_ROW));
  const data = new values.constructor(SPLAT_ROW * rows * 4);
  data.set(values);
  const type = values instanceof Float32Array ? THREE.FloatType : THREE.UnsignedByteType;
  const texture = new THREE.DataTexture(data, SPLAT_ROW, rows, THREE.RGBAFormat, type);
  texture.minFilter = THREE.NearestFilter;
  texture.magFilter = THREE.NearestFilter;
  texture.generateMipmaps = false;
  texture.needsUpdate = true;
  return texture;
}

/**
 * Fitted splats from a run's bytes, as a mesh and the sort that keeps them back to front.
 *
 * @param {ArrayBuffer} body Positions, covariances, then colours with opacity.
 * @returns {{object: object, sort: function(object): void}} The mesh and its sort.
 */
function buildSplats(body) {
  const count = Math.floor(body.byteLength / SPLAT_BYTES);
  const positions = new Float32Array(body, 0, count * 3);
  const covariances = new Float32Array(body, count * 12, count * 6);
  const centres = new Float32Array(count * 4);
  const shapesA = new Float32Array(count * 4);
  const shapesB = new Float32Array(count * 4);
  for (let i = 0; i < count; i++) {
    centres[i * 4] = positions[i * 3];
    centres[i * 4 + 1] = positions[i * 3 + 1];
    centres[i * 4 + 2] = positions[i * 3 + 2];
    shapesA.set(covariances.subarray(i * 6, i * 6 + 4), i * 4);
    shapesB[i * 4] = covariances[i * 6 + 4];
    shapesB[i * 4 + 1] = covariances[i * 6 + 5];
  }
  const geometry = new THREE.InstancedBufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array([-2, -2, 0, 2, -2, 0, 2, 2, 0, -2, 2, 0]), 3));
  geometry.setIndex([0, 1, 2, 0, 2, 3]);
  const order = new THREE.InstancedBufferAttribute(new Float32Array(count), 1);
  order.setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute("order", order);
  geometry.instanceCount = count;
  const material = new THREE.ShaderMaterial({
    uniforms: {
      centres: { value: splatTexture(centres, count) },
      shapesA: { value: splatTexture(shapesA, count) },
      shapesB: { value: splatTexture(shapesB, count) },
      tints: { value: splatTexture(new Uint8Array(body, count * 36, count * 4), count) },
      focal: { value: 1 },
      viewport: { value: new THREE.Vector2(1, 1) },
    },
    vertexShader: SPLAT_VERTEX,
    fragmentShader: SPLAT_FRAGMENT,
    transparent: true,
    depthTest: false,
    depthWrite: false,
    blending: THREE.CustomBlending,
    blendSrc: THREE.OneFactor,
    blendDst: THREE.OneMinusSrcAlphaFactor,
    blendSrcAlpha: THREE.OneFactor,
    blendDstAlpha: THREE.OneMinusSrcAlphaFactor,
  });
  const object = new THREE.Mesh(geometry, material);
  object.frustumCulled = false;
  const depths = new Float32Array(count);
  const buckets = new Uint32Array(SORT_BUCKETS);
  const sorted = order.array;
  let last = null;
  const sort = (camera) => {
    const m = camera.matrixWorldInverse.elements;
    const view = [m[2], m[6], m[10], m[14]];
    if (last && view.every((v, i) => Math.abs(v - last[i]) < 1e-5)) return;
    last = view;
    let near = Infinity;
    let far = -Infinity;
    for (let i = 0; i < count; i++) {
      const z = m[2] * positions[i * 3] + m[6] * positions[i * 3 + 1] + m[10] * positions[i * 3 + 2] + m[14];
      depths[i] = z;
      if (z < near) near = z;
      if (z > far) far = z;
    }
    const scale = (SORT_BUCKETS - 1) / Math.max(far - near, 1e-9);
    buckets.fill(0);
    for (let i = 0; i < count; i++) buckets[Math.floor((depths[i] - near) * scale)]++;
    for (let i = 1; i < SORT_BUCKETS; i++) buckets[i] += buckets[i - 1];
    for (let i = count - 1; i >= 0; i--) sorted[--buckets[Math.floor((depths[i] - near) * scale)]] = i;
    order.needsUpdate = true;
  };
  return { object, sort };
}

/**
 * Release a cloud's mesh, its material and any textures the material holds.
 *
 * @param {object|null} object The mesh or points.
 */
function disposeCloud(object) {
  if (!object) return;
  object.geometry.dispose();
  for (const uniform of Object.values(object.material.uniforms ?? {})) uniform.value?.isTexture && uniform.value.dispose();
  object.material.dispose();
}

// Digits a stored number keeps.
const DIGITS = 5;

// Frames a second rigs are written and read at when a file names none.
const RIG_FPS = 24;

// Keys that steer the camera while the view has focus.
const MOVE_KEYS = {
  KeyW: [0, 0, -1], KeyS: [0, 0, 1], KeyA: [-1, 0, 0], KeyD: [1, 0, 0],
  KeyE: [0, 1, 0], KeyQ: [0, -1, 0], Space: [0, 1, 0], KeyC: [0, -1, 0],
  ArrowUp: [0, 0, -1], ArrowDown: [0, 0, 1], ArrowLeft: [-1, 0, 0], ArrowRight: [1, 0, 0],
};

/**
 * A number rounded to the digits a stored path keeps.
 *
 * @param {number} value - The number.
 * @returns {number} The rounded number.
 */
function kept(value) {
  const scale = 10 ** DIGITS;
  return Math.round(value * scale) / scale;
}

/**
 * A pill button in the card's look.
 *
 * @param {string} label - The words on it.
 * @param {string} hint - What it does, shown on hover.
 * @param {Function} onPress - Called when it is pressed.
 * @returns {HTMLButtonElement} The button.
 */
function pill(label, hint, onPress) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  button.title = hint;
  button.style.cssText = "flex:none;font:600 11px/1.4 system-ui,sans-serif;padding:3px 9px;border-radius:999px;"
    + `cursor:pointer;background:${CARD_LOOK.inset};color:${CARD_LOOK.text};border:1px solid ${CARD_LOOK.edge};`
    + "white-space:nowrap";
  button.onmouseenter = () => { button.style.borderColor = CARD_LOOK.rule; };
  button.onmouseleave = () => { button.style.borderColor = CARD_LOOK.edge; };
  button.addEventListener("click", (event) => {
    event.stopPropagation();
    onPress(event);
  });
  return button;
}

// Each tool's glyph, as SVG path data on a 16 unit square.
const GLYPHS = {
  fly: "M2 8 14 2 10 14 8 9Z",
  orbit: "M8 3a5 5 0 1 1-4.6 3M3 2v4h4",
  record: "M8 3a5 5 0 1 0 0 10A5 5 0 1 0 8 3Z",
  stop: "M4 4h8v8H4Z",
  key: "M8 2 13 8 8 14 3 8Z",
  play: "M5 3 13 8 5 13Z",
  start: "M4 3v10M13 3 6 8 13 13Z",
  frame: "M2 4h12v8H2Z",
  import: "M8 2v8M5 7l3 3 3-3M3 12v2h10v-2",
  export: "M8 10V2M5 5l3-3 3 3M3 12v2h10v-2",
  align: "M8 2v3M8 11v3M2 8h3M11 8h3M8 5a3 3 0 1 0 0 6 3 3 0 1 0 0-6",
  clear: "M4 4l8 8M12 4l-8 8",
};

// Glyphs drawn solid rather than outlined.
const SOLID = ["fly", "record", "stop", "play"];

// Panel width under which the toolbar shows glyphs without words.
const COMPACT_WIDTH = 520;

/**
 * A glyph as an inline SVG in the current text colour.
 *
 * @param {string} name - A key of `GLYPHS`.
 * @returns {SVGElement} The glyph.
 */
function glyph(name) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 16 16");
  svg.setAttribute("width", "13");
  svg.setAttribute("height", "13");
  svg.style.cssText = "flex:none;display:block";
  const shape = document.createElementNS("http://www.w3.org/2000/svg", "path");
  const solid = SOLID.includes(name);
  shape.setAttribute("d", GLYPHS[name] ?? "");
  shape.setAttribute("fill", solid ? "currentColor" : "none");
  shape.setAttribute("stroke", "currentColor");
  shape.setAttribute("stroke-width", solid ? "0" : "1.6");
  shape.setAttribute("stroke-linecap", "round");
  shape.setAttribute("stroke-linejoin", "round");
  svg.append(shape);
  return svg;
}

/**
 * One segment of a toolbar group: a glyph and a word, filling its share of the group.
 *
 * @param {object} spec - `{icon, label, hint, onPress}`.
 * @returns {HTMLButtonElement} The segment, with `setIcon`, `setLabel` and its `word`.
 */
function tool({ icon, label, hint, onPress }) {
  const button = document.createElement("button");
  button.type = "button";
  button.title = hint;
  button.style.cssText = "flex:1 1 0;min-width:0;display:flex;align-items:center;justify-content:center;gap:5px;"
    + "height:26px;padding:0 8px;border:none;margin:0;cursor:pointer;background:transparent;"
    + `color:${CARD_LOOK.text};font:600 11px/1 system-ui,sans-serif;white-space:nowrap`;
  const word = document.createElement("span");
  word.style.cssText = "min-width:0;overflow:hidden;text-overflow:ellipsis";
  word.textContent = label;
  let mark = glyph(icon);
  button.append(mark, word);
  button.word = word;
  button.setIcon = (name) => {
    const next = glyph(name);
    mark.replaceWith(next);
    mark = next;
  };
  button.setLabel = (text) => { word.textContent = text; };
  button.onmouseenter = () => {
    if (!button.on) button.style.background = `color-mix(in srgb, ${CARD_LOOK.text} 7%, transparent)`;
  };
  button.onmouseleave = () => {
    if (!button.on) button.style.background = "transparent";
  };
  button.addEventListener("click", (event) => {
    event.stopPropagation();
    onPress(event);
  });
  return button;
}

/**
 * Show a segment as on or off.
 *
 * @param {HTMLElement} button - The segment.
 * @param {boolean} on - Whether it is on.
 * @param {string} [tone] - The chip tone it takes while on.
 * @returns {void}
 */
function setSegment(button, on, tone = "accent") {
  const look = CHIP_TONES[tone] ?? CHIP_TONES.accent;
  button.on = on;
  button.style.background = on ? look.background : "transparent";
  button.style.color = on ? look.color : (button.tone ?? CARD_LOOK.text);
}

/**
 * Segments joined into one inset bar with hairlines between them.
 *
 * @param {HTMLElement[]} segments - The segments.
 * @param {number} grow - The group's share of the toolbar's width.
 * @returns {HTMLElement} The group.
 */
function toolGroup(segments, grow) {
  const group = document.createElement("div");
  group.style.cssText = `flex:${grow} 1 0;min-width:0;display:flex;align-items:stretch;border-radius:6px;`
    + `overflow:hidden;background:${CARD_LOOK.inset};border:1px solid ${CARD_LOOK.edge};box-shadow:${CARD_LOOK.shine}`;
  segments.forEach((segment, index) => {
    if (index) {
      const line = document.createElement("span");
      line.style.cssText = `flex:none;width:1px;background:${CARD_LOOK.edge}`;
      group.append(line);
    }
    group.append(segment);
  });
  return group;
}

/**
 * Catmull-Rom point between `p1` and `p2`.
 *
 * @returns {THREE.Vector3} The point.
 */
function catmull(p0, p1, p2, p3, t) {
  const t2 = t * t;
  const t3 = t2 * t;
  const out = new THREE.Vector3();
  for (const axis of ["x", "y", "z"]) {
    out[axis] = 0.5 * ((2 * p1[axis]) + (-p0[axis] + p2[axis]) * t
      + (2 * p0[axis] - 5 * p1[axis] + 4 * p2[axis] - p3[axis]) * t2
      + (-p0[axis] + 3 * p1[axis] - 3 * p2[axis] + p3[axis]) * t3);
  }
  return out;
}

/**
 * The camera at a time along keys, as the node computes it.
 *
 * @param {object[]} keys - Keys with `t`, `position`, `quaternion` and `fov`, times rising.
 * @param {number} when - Seconds.
 * @returns {object} `{position, quaternion, fov}`.
 */
function poseAt(keys, when) {
  if (keys.length === 1) return keys[0];
  const last = keys.length - 1;
  let span = keys.findIndex((key, index) => index < last && when <= keys[index + 1].t);
  if (span < 0) span = last - 1;
  const a = keys[span];
  const b = keys[span + 1];
  const local = Math.min(1, Math.max(0, (when - a.t) / Math.max(1e-9, b.t - a.t)));
  const position = catmull(keys[Math.max(span - 1, 0)].position, a.position, b.position,
    keys[Math.min(span + 2, last)].position, local);
  const quaternion = a.quaternion.clone().slerp(b.quaternion, local);
  return { position, quaternion, fov: a.fov + (b.fov - a.fov) * local };
}

/**
 * Keys out of the node's stored path text.
 *
 * @param {string} text - The `path` widget's value.
 * @returns {{kind: string, keys: object[]}} The path; no keys for empty or unreadable text.
 */
function parsePath(text) {
  try {
    const data = JSON.parse(text || "{}");
    const rows = Array.isArray(data?.keys) ? data.keys : [];
    const keys = rows.filter((row) => Array.isArray(row) && row.length === 9 && row.every(Number.isFinite))
      .map(([t, x, y, z, qx, qy, qz, qw, fov]) => ({
        t, position: new THREE.Vector3(x, y, z), quaternion: new THREE.Quaternion(qx, qy, qz, qw).normalize(), fov,
      }))
      .sort((a, b) => a.t - b.t);
    return { kind: data?.kind === "recording" ? "recording" : "keys", keys };
  } catch {
    return { kind: "keys", keys: [] };
  }
}

/**
 * The path text the node reads.
 *
 * @param {string} kind - `keys` or `recording`.
 * @param {object[]} keys - The keys.
 * @returns {string} JSON, or empty for no keys.
 */
function pathText(kind, keys) {
  if (!keys.length) return "";
  const rows = keys.map((key) => [
    kept(key.t), kept(key.position.x), kept(key.position.y), kept(key.position.z),
    kept(key.quaternion.x), kept(key.quaternion.y), kept(key.quaternion.z), kept(key.quaternion.w), kept(key.fov),
  ]);
  return JSON.stringify({ v: 1, kind, keys: rows });
}

/**
 * Keys read from a camera rig file.
 *
 * @param {File} file - A nerfstudio camera path or this node's own JSON, a Nuke `.chan`, or a glTF.
 * @returns {Promise<object[]>} Keys with times in seconds.
 * @throws {Error} When the file holds no camera.
 */
async function readRig(file) {
  const name = file.name.toLowerCase();
  if (name.endsWith(".glb") || name.endsWith(".gltf")) return readGltfRig(file);
  const text = await file.text();
  if (name.endsWith(".chan") || name.endsWith(".txt")) return readChan(text);
  const data = JSON.parse(text);
  if (Array.isArray(data?.keys)) return parsePath(text).keys;
  const path = Array.isArray(data?.camera_path) ? data.camera_path : null;
  if (!path?.length) throw new Error("the JSON holds no camera_path");
  const seconds = Number(data.seconds) || 0;
  const fps = Number(data.fps) || RIG_FPS;
  return path.map((entry, index) => {
    const flat = (entry.camera_to_world ?? []).flat();
    if (flat.length < 12) throw new Error(`camera ${index + 1} has no camera_to_world matrix`);
    const matrix = new THREE.Matrix4().set(...flat.slice(0, 12), 0, 0, 0, 1);
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    matrix.decompose(position, quaternion, new THREE.Vector3());
    const t = seconds > 0 && path.length > 1 ? (seconds * index) / (path.length - 1) : index / fps;
    return { t, position, quaternion, fov: Number(entry.fov) || 50 };
  });
}

/**
 * Keys from a Nuke `.chan`: frame, translate, rotate in degrees applied Z then X then Y, and an
 * optional vertical field of view.
 *
 * @param {string} text - The file.
 * @returns {object[]} The keys.
 */
function readChan(text) {
  const keys = [];
  for (const line of text.split(/\r?\n/)) {
    const values = line.trim().split(/\s+/).map(Number);
    if (values.length < 7 || !values.every(Number.isFinite)) continue;
    const [frame, x, y, z, rx, ry, rz, fov] = values;
    const euler = new THREE.Euler(THREE.MathUtils.degToRad(rx), THREE.MathUtils.degToRad(ry),
      THREE.MathUtils.degToRad(rz), "YXZ");
    keys.push({ t: frame / RIG_FPS, position: new THREE.Vector3(x, y, z),
      quaternion: new THREE.Quaternion().setFromEuler(euler), fov: Number.isFinite(fov) ? fov : 50 });
  }
  if (!keys.length) throw new Error("the .chan file holds no frame lines");
  const first = keys[0].t;
  for (const key of keys) key.t -= first;
  return keys;
}

/**
 * Keys from the first camera in a glTF, sampled through its animation at the rig rate.
 *
 * @param {File} file - A `.glb` or `.gltf`.
 * @returns {Promise<object[]>} The keys.
 */
async function readGltfRig(file) {
  const { GLTFLoader } = await import("./vendor/three/loaders/GLTFLoader.mjs");
  const gltf = await new GLTFLoader().parseAsync(await file.arrayBuffer(), "");
  const camera = gltf.cameras?.[0];
  if (!camera) throw new Error("the glTF holds no camera");
  const mixer = new THREE.AnimationMixer(gltf.scene);
  const clip = gltf.animations?.[0];
  if (clip) mixer.clipAction(clip).play();
  const seconds = clip?.duration ?? 0;
  const count = Math.max(1, Math.round(seconds * RIG_FPS) + 1);
  const keys = [];
  for (let index = 0; index < count; index += 1) {
    mixer.setTime(index / RIG_FPS);
    gltf.scene.updateMatrixWorld(true);
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    camera.matrixWorld.decompose(position, quaternion, new THREE.Vector3());
    keys.push({ t: index / RIG_FPS, position, quaternion, fov: camera.fov ?? 50 });
  }
  return keys;
}

/**
 * Move keys rigidly so the first lands on a camera, keeping their spacing and turns.
 *
 * @param {object[]} keys - The keys.
 * @param {object} anchor - `{position, quaternion}` the first key moves onto.
 * @returns {object[]} The moved keys.
 */
function alignTo(keys, anchor) {
  const first = keys[0];
  const turn = anchor.quaternion.clone().multiply(first.quaternion.clone().invert());
  return keys.map((key) => ({
    ...key,
    position: key.position.clone().sub(first.position).applyQuaternion(turn).add(anchor.position),
    quaternion: turn.clone().multiply(key.quaternion),
  }));
}

/**
 * A rig file's text for a path.
 *
 * @param {string} format - `nerfstudio` or `chan`.
 * @param {object[]} frames - One pose per frame.
 * @param {number} fps - Frames a second.
 * @param {number} aspect - Width over height.
 * @returns {string} The file.
 */
function writeRig(format, frames, fps, aspect) {
  if (format === "chan") {
    return frames.map((pose, index) => {
      const euler = new THREE.Euler().setFromQuaternion(pose.quaternion, "YXZ");
      const deg = THREE.MathUtils.radToDeg;
      return [index + 1, pose.position.x, pose.position.y, pose.position.z, deg(euler.x), deg(euler.y), deg(euler.z),
        pose.fov].map((value) => (Number.isInteger(value) ? value : value.toFixed(6))).join(" ");
    }).join("\n") + "\n";
  }
  const cameraPath = frames.map((pose) => {
    const matrix = new THREE.Matrix4().compose(pose.position, pose.quaternion, new THREE.Vector3(1, 1, 1));
    const rows = matrix.clone().transpose().toArray();
    return { camera_to_world: rows.map(kept), fov: kept(pose.fov), aspect: kept(aspect) };
  });
  return JSON.stringify({
    camera_type: "perspective", render_height: 768, render_width: Math.round(768 * aspect),
    camera_path: cameraPath, fps, seconds: Math.max(0, (frames.length - 1) / fps),
  }, null, 1);
}

/**
 * Hand the browser a file to save.
 *
 * @param {string} text - The contents.
 * @param {string} name - The file name.
 * @returns {void}
 */
function download(text, name) {
  const url = URL.createObjectURL(new Blob([text], { type: "application/octet-stream" }));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/**
 * Build the navigator for one node.
 *
 * @param {object} node - A Point Cloud Navigator node.
 * @returns {object} The panel for `appendInterfaceWidget`.
 */
function createNavigator(node) {
  const root = document.createElement("div");
  root.style.cssText = "display:flex;flex-direction:column;gap:6px;padding:8px;height:100%;min-width:0;"
    + "font:12px/1.4 system-ui,sans-serif";

  const header = document.createElement("div");
  header.style.cssText = "display:flex;flex-wrap:wrap;align-items:center;gap:6px;min-width:0;flex:none";
  const title = document.createElement("span");
  title.textContent = "Navigator";
  title.style.cssText = `font-weight:650;color:${CARD_LOOK.title};flex:none`;
  const summary = document.createElement("span");
  summary.style.cssText = `color:${CARD_LOOK.muted};min-width:0;flex:1 1 120px;overflow:hidden;`
    + "text-overflow:ellipsis;white-space:nowrap";
  const modeChip = cardChip("fly", "accent");
  header.append(title, modeChip, summary);

  const tools = document.createElement("div");
  tools.style.cssText = "display:flex;flex-wrap:wrap;gap:6px;min-width:0;flex:none";

  const well = document.createElement("div");
  well.style.cssText = `position:relative;flex:1 1 auto;min-height:0;border-radius:6px;overflow:hidden;`
    + `background:${CARD_LOOK.inset};box-shadow:${CARD_LOOK.well};outline:none`;
  well.tabIndex = 0;
  const canvas = document.createElement("canvas");
  canvas.style.cssText = "position:absolute;inset:0;width:100%;height:100%;display:block";
  const guide = document.createElement("div");
  guide.style.cssText = `position:absolute;pointer-events:none;border:1px solid ${CARD_LOOK.rule};`
    + "box-shadow:0 0 0 9999px rgba(0,0,0,0.35)";
  const hud = document.createElement("div");
  hud.style.cssText = `position:absolute;left:8px;bottom:6px;right:8px;pointer-events:none;color:${CARD_LOOK.muted};`
    + "font:11px/1.3 system-ui,sans-serif;text-shadow:0 1px 2px rgba(0,0,0,0.8)";
  const empty = document.createElement("div");
  empty.style.cssText = `position:absolute;inset:0;display:flex;align-items:center;justify-content:center;`
    + `color:${CARD_LOOK.muted};text-align:center;padding:12px`;
  empty.textContent = "run to see the cloud";
  well.append(canvas, guide, hud, empty);

  const strip = document.createElement("div");
  strip.style.cssText = "display:flex;flex-wrap:wrap;gap:4px;min-width:0;flex:0 1 auto;max-height:64px;overflow:auto";

  root.append(header, cardRule(), tools, well, strip);

  const fileInput = document.createElement("input");
  fileInput.type = "file";
  fileInput.accept = ".json,.chan,.txt,.glb,.gltf";
  fileInput.style.display = "none";
  root.append(fileInput);

  // Live view, built once three.js has loaded.
  let renderer = null;
  let scene = null;
  let camera = null;
  let pathLine = null;
  let keyDots = null;
  let cloudPoints = null;
  let cloudSort = null;
  let disposed = false;
  const ready = loadThree().then(() => {
    if (disposed) return;
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(50, 16 / 9, 0.01, 1000);
    pathLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xffc04d }));
    keyDots = new THREE.Points(new THREE.BufferGeometry(),
      new THREE.PointsMaterial({ color: 0xffc04d, size: 8, sizeAttenuation: false }));
    pathLine.frustumCulled = false;
    keyDots.frustumCulled = false;
    scene.add(pathLine, keyDots);
    pivot = new THREE.Vector3(0, 0, -1);
    path = parsePath(pathWidget()?.value);
    showPath();
    describe();
  });
  let payload = null;
  let shownAsset = "";
  let depth = 1;
  let fov = 50;
  let mode = "fly";
  let aspectName = "16:9";
  let speed = BASE_SPEED;
  let pivot = null;
  const held = new Set();
  let recording = null;
  let playing = null;
  let alignImports = true;
  let path = { kind: "keys", keys: [] };
  let frame = 0;
  let lastTime = 0;

  function pathWidget() {
    return node.widgets?.find((widget) => widget.name === "path");
  }

  function numberWidget(name, fallback) {
    const value = Number(node.widgets?.find((widget) => widget.name === name)?.value);
    return Number.isFinite(value) ? value : fallback;
  }

  function ensureRenderer() {
    if (renderer) return renderer;
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, preserveDrawingBuffer: false });
    renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    renderer.setClearColor(0x000000, 0);
    return renderer;
  }

  // The guide rectangle and the camera's vertical view, so the guide shows `fov` across its shorter side.
  function layout() {
    if (!camera) return;
    const width = Math.max(1, well.clientWidth);
    const height = Math.max(1, well.clientHeight);
    const aspect = ASPECTS[aspectName] ?? ASPECTS["16:9"];
    let guideW = width - 16;
    let guideH = guideW / aspect;
    if (guideH > height - 16) {
      guideH = height - 16;
      guideW = guideH * aspect;
    }
    guideW = Math.max(8, guideW);
    guideH = Math.max(8, guideH);
    Object.assign(guide.style, {
      left: `${(width - guideW) / 2}px`, top: `${(height - guideH) / 2}px`,
      width: `${guideW}px`, height: `${guideH}px`,
    });
    const half = Math.tan(THREE.MathUtils.degToRad(fov) / 2) * height / Math.min(guideW, guideH);
    camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(half));
    camera.aspect = width / height;
    camera.near = Math.max(1e-4, depth * 0.002);
    camera.far = depth * 200;
    camera.updateProjectionMatrix();
    const ratio = surfaceRatio(canvas);
    ensureRenderer().setSize(Math.round(width * ratio), Math.round(height * ratio), false);
    if (cloudPoints) {
      const uniforms = cloudPoints.material.uniforms;
      uniforms.focal.value = (canvas.height / 2) / Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
      if (uniforms.widest) uniforms.widest.value = (2 * MAX_RADIUS + 1) * (canvas.height / Math.max(1, height));
      uniforms.viewport?.value.set(canvas.width, canvas.height);
    }
    draw();
  }

  function draw() {
    if (!renderer || !scene) return;
    if (cloudSort) {
      camera.updateMatrixWorld();
      cloudSort(camera);
    }
    renderer.render(scene, camera);
  }

  function describe() {
    const count = path.keys.length;
    const frames = numberWidget("frames", 158);
    const what = !count ? "no path, the start camera is held"
      : path.kind === "recording" ? `flight of ${path.keys.at(-1).t.toFixed(1)} s`
        : `${count} key${count === 1 ? "" : "s"}`;
    const points = payload ? `${payload.points.toLocaleString()} ${payload.style === "splats" ? "splats" : "points"}` : "not run yet";
    summary.textContent = `${points}, ${what}, ${frames || "as flown"} frames`;
    hud.textContent = mode === "fly"
      ? `drag to look, W A S D to move, Q E down and up, Shift faster, wheel speed ${(speed / BASE_SPEED).toFixed(2)}x`
      : "drag to orbit, right drag to pan, wheel to move in and out";
    keyStrip();
  }

  function keyStrip() {
    strip.replaceChildren();
    if (path.kind === "recording" && path.keys.length) {
      strip.append(cardChip(`recorded ${path.keys.length} samples`, "accent"));
      return;
    }
    path.keys.forEach((key, index) => {
      const chip = pill(`${index + 1}  ${key.t.toFixed(1)} s`, "Go to this key; Shift-click to remove it", (event) => {
        if (event.shiftKey) {
          path.keys.splice(index, 1);
          path.keys.forEach((other, at) => { other.t = at * KEY_SPACING; });
          store();
          return;
        }
        place(key);
      });
      strip.append(chip);
    });
  }

  function showPath() {
    if (!pathLine) return;
    const points = path.keys.map((key) => key.position);
    const dense = [];
    if (points.length > 1) {
      const end = path.keys.at(-1).t;
      const steps = Math.min(600, Math.max(32, path.keys.length * 12));
      for (let index = 0; index <= steps; index += 1) dense.push(poseAt(path.keys, (end * index) / steps).position);
    }
    pathLine.geometry.dispose();
    pathLine.geometry = new THREE.BufferGeometry().setFromPoints(dense);
    keyDots.geometry.dispose();
    keyDots.geometry = new THREE.BufferGeometry().setFromPoints(path.kind === "recording" ? [] : points);
  }

  // Write the path into its widget, where the node reads it.
  function store() {
    const widget = pathWidget();
    const text = pathText(path.kind, path.keys);
    if (widget && widget.value !== text) {
      widget.value = text;
      widget.callback?.(text);
      node.setDirtyCanvas?.(true, true);
      app.extensionManager?.workflow?.activeWorkflow?.changeTracker?.checkState?.();
    }
    showPath();
    describe();
    draw();
  }

  function current() {
    if (!camera) return null;
    return { position: camera.position.clone(), quaternion: camera.quaternion.clone(), fov };
  }

  function place(pose) {
    if (!camera || !pose) return;
    camera.position.copy(pose.position);
    camera.quaternion.copy(pose.quaternion);
    fov = pose.fov ?? fov;
    pivot = camera.position.clone().add(new THREE.Vector3(0, 0, -depth).applyQuaternion(camera.quaternion));
    layout();
  }

  function startPose() {
    const start = payload?.start;
    if (!start?.position) return { position: new THREE.Vector3(), quaternion: new THREE.Quaternion(), fov: 50 };
    const position = new THREE.Vector3(start.position.x, start.position.y, start.position.z);
    const q = start.quaternion;
    const quaternion = q ? new THREE.Quaternion(q.x, q.y, q.z, q.w).normalize()
      : new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().lookAt(position,
        new THREE.Vector3(start.target?.x ?? 0, start.target?.y ?? 0, start.target?.z ?? -1), new THREE.Vector3(0, 1, 0)));
    return { position, quaternion, fov: Number(start.fov) || 50 };
  }

  async function show(data) {
    await ready;
    if (disposed || !scene) return;
    if (!data || typeof data.asset !== "string" || !ASSET_PATTERN.test(data.asset)) return;
    const first = !payload;
    payload = data;
    depth = Math.max(1e-3, Number(data.depth) || 1);
    describe();
    if (data.asset === shownAsset) return;
    shownAsset = data.asset;
    try {
      const response = await fetch(data.asset);
      if (!response.ok) throw new Error(`the cloud is no longer held (${response.status}); run the node again`);
      const body = await response.arrayBuffer();
      let built;
      if (data.style === "splats") {
        const splats = buildSplats(body);
        built = splats.object;
        cloudSort = splats.sort;
      } else {
        const count = Math.floor(body.byteLength / POINT_BYTES);
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(body, 0, count * 3), 3));
        geometry.setAttribute("spread", new THREE.BufferAttribute(new Float32Array(body, count * 12, count), 1));
        geometry.setAttribute("tint", new THREE.BufferAttribute(new Uint8Array(body, count * 16, count * 3), 3, true));
        const material = new THREE.ShaderMaterial({
          uniforms: { focal: { value: 1 }, widest: { value: 2 * MAX_RADIUS + 1 } },
          vertexShader: DOT_VERTEX,
          fragmentShader: DOT_FRAGMENT,
        });
        built = new THREE.Points(geometry, material);
        built.frustumCulled = false;
        cloudSort = null;
      }
      if (cloudPoints) {
        scene.remove(cloudPoints);
        disposeCloud(cloudPoints);
      }
      cloudPoints = built;
      scene.add(cloudPoints);
      empty.style.display = "none";
      if (first) place(path.keys[0] ?? startPose());
      else layout();
    } catch (error) {
      empty.style.display = "flex";
      empty.textContent = String(error?.message ?? error);
    }
  }

  function tick(time) {
    frame = 0;
    if (!camera) return;
    const seconds = lastTime ? Math.min(0.1, (time - lastTime) / 1000) : 0;
    lastTime = time;
    let busy = false;
    if (playing) {
      const elapsed = (time - playing.start) / 1000;
      const end = path.keys.at(-1)?.t ?? 0;
      const length = playing.seconds;
      if (elapsed >= length || path.keys.length < 2) {
        playing = null;
        setSegment(playButton, false);
      } else {
        place(poseAt(path.keys, (elapsed / length) * end));
        busy = true;
      }
    }
    if (held.size && !playing) {
      const move = new THREE.Vector3();
      for (const code of held) {
        const step = MOVE_KEYS[code];
        if (step) move.add(new THREE.Vector3(...step));
      }
      if (move.lengthSq() > 0) {
        const fast = held.has("ShiftLeft") || held.has("ShiftRight") ? 4 : 1;
        move.normalize().multiplyScalar(speed * depth * fast * seconds).applyQuaternion(camera.quaternion);
        camera.position.add(move);
        pivot.add(move);
        busy = true;
      }
    }
    if (recording) {
      const now = (time - recording.start) / 1000;
      if (now - recording.last >= SAMPLE_SECONDS && recording.keys.length < MAX_SAMPLES) {
        recording.keys.push({ t: now, ...current() });
        recording.last = now;
      }
      busy = true;
    }
    draw();
    if (busy || held.size) frame = requestAnimationFrame(tick);
    else lastTime = 0;
  }

  function wake() {
    if (!frame) frame = requestAnimationFrame(tick);
  }

  // Pointer: look or orbit with the left button, pan with the right, the middle is the graph's.
  let drag = null;
  well.addEventListener("pointerdown", (event) => {
    if (event.button === 1) {
      app.canvas?.processMouseDown?.(event);
      return;
    }
    event.stopPropagation();
    well.focus({ preventScroll: true });
    try {
      well.setPointerCapture(event.pointerId);
    } catch {
      // A pointer the browser no longer tracks cannot be captured; the drag still follows moves over the view.
    }
    drag = { x: event.clientX, y: event.clientY, button: event.button };
  });
  well.addEventListener("pointermove", (event) => {
    event.stopPropagation();
    if (!drag || playing || !camera) return;
    const scale = well.clientWidth / Math.max(1, well.getBoundingClientRect().width);
    const dx = (event.clientX - drag.x) * scale;
    const dy = (event.clientY - drag.y) * scale;
    drag.x = event.clientX;
    drag.y = event.clientY;
    if (mode === "fly" && drag.button === 0) {
      const euler = new THREE.Euler().setFromQuaternion(camera.quaternion, "YXZ");
      euler.y -= THREE.MathUtils.degToRad(dx * LOOK_RATE);
      euler.x = Math.max(-1.55, Math.min(1.55, euler.x - THREE.MathUtils.degToRad(dy * LOOK_RATE)));
      euler.z = 0;
      camera.quaternion.setFromEuler(euler);
    } else if (drag.button === 2 || (mode === "orbit" && event.shiftKey)) {
      const reach = camera.position.distanceTo(pivot) * 2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
      const shift = new THREE.Vector3(-dx, dy, 0).multiplyScalar(reach / Math.max(1, well.clientHeight))
        .applyQuaternion(camera.quaternion);
      camera.position.add(shift);
      pivot.add(shift);
    } else {
      const offset = camera.position.clone().sub(pivot);
      const spherical = new THREE.Spherical().setFromVector3(offset);
      spherical.theta -= THREE.MathUtils.degToRad(dx * LOOK_RATE);
      spherical.phi = Math.max(0.05, Math.min(Math.PI - 0.05, spherical.phi - THREE.MathUtils.degToRad(dy * LOOK_RATE)));
      camera.position.copy(pivot).add(new THREE.Vector3().setFromSpherical(spherical));
      camera.lookAt(pivot);
    }
    wake();
  });
  const endDrag = (event) => {
    event.stopPropagation();
    drag = null;
  };
  well.addEventListener("pointerup", endDrag);
  well.addEventListener("pointercancel", endDrag);
  well.addEventListener("contextmenu", (event) => {
    event.preventDefault();
    event.stopPropagation();
  });
  const releaseWheel = captureWheel(root, (event) => {
    if (!well.contains(event.target) || !payload || !camera) return false;
    if (mode === "fly") {
      speed = Math.min(BASE_SPEED * 64, Math.max(BASE_SPEED / 64, speed * (event.deltaY < 0 ? SPEED_STEP : 1 / SPEED_STEP)));
      describe();
    } else {
      const toward = pivot.clone().sub(camera.position).multiplyScalar(event.deltaY < 0 ? 0.12 : -0.12);
      camera.position.add(toward);
      draw();
    }
    return true;
  });
  well.addEventListener("keydown", (event) => {
    if (event.ctrlKey || event.metaKey) return;
    event.stopPropagation();
    if (MOVE_KEYS[event.code] || event.code.startsWith("Shift")) {
      event.preventDefault();
      held.add(event.code);
      wake();
    }
  });
  well.addEventListener("keyup", (event) => {
    event.stopPropagation();
    held.delete(event.code);
  });
  well.addEventListener("blur", () => held.clear());

  // Tools.
  const flyTool = tool({ icon: "fly", label: "Fly", hint: "Fly: drag to look, W A S D to move, Q E down and up",
    onPress: () => setMode("fly") });
  const orbitTool = tool({ icon: "orbit", label: "Orbit", hint: "Orbit: drag round the point ahead, right drag to pan",
    onPress: () => setMode("orbit") });
  const aspectTool = tool({ icon: "frame", label: aspectName, hint: "The output frame the guide shows; press for the next shape",
    onPress: () => {
      const names = Object.keys(ASPECTS);
      aspectName = names[(names.indexOf(aspectName) + 1) % names.length];
      aspectTool.setLabel(aspectName);
      layout();
    } });
  function setMode(next) {
    mode = next;
    modeChip.textContent = mode;
    setSegment(flyTool, mode === "fly");
    setSegment(orbitTool, mode === "orbit");
    if (camera) pivot = camera.position.clone().add(new THREE.Vector3(0, 0, -depth).applyQuaternion(camera.quaternion));
    describe();
  }
  const recordTool = tool({ icon: "record", label: "Record", hint: "Record the flight; press again to stop and keep it as the path",
    onPress: () => {
      if (recording) {
        const samples = recording.keys;
        recording = null;
        setSegment(recordTool, false, "error");
        recordTool.setIcon("record");
        recordTool.setLabel("Record");
        if (samples.length > 1) {
          path = { kind: "recording", keys: samples };
          store();
        }
        return;
      }
      if (!camera) return;
      playing = null;
      recording = { start: performance.now(), last: -1, keys: [] };
      setSegment(recordTool, true, "error");
      recordTool.setIcon("stop");
      recordTool.setLabel("Stop");
      well.focus({ preventScroll: true });
      wake();
    } });
  const keyTool = tool({ icon: "key", label: "Key", hint: "Keep this view as the next key, a second after the last",
    onPress: () => {
      if (!camera) return;
      if (path.kind === "recording") path = { kind: "keys", keys: [] };
      path.keys.push({ t: path.keys.length * KEY_SPACING, ...current() });
      store();
    } });
  const playButton = tool({ icon: "play", label: "Play", hint: "Play the path at the node's frames and fps",
    onPress: () => {
      if (playing || path.keys.length < 2) {
        playing = null;
        setSegment(playButton, false);
        return;
      }
      const frames = numberWidget("frames", 158);
      const fps = numberWidget("fps", 24) || 24;
      const seconds = frames > 0 ? frames / fps : path.keys.at(-1).t;
      playing = { start: performance.now(), seconds };
      setSegment(playButton, true);
      wake();
    } });
  const startTool = tool({ icon: "start", label: "Start", hint: "Go back to the start camera",
    onPress: () => place(path.keys[0] ?? startPose()) });
  const clearTool = tool({ icon: "clear", label: "Clear", hint: "Remove the path; the node then holds the start camera",
    onPress: () => {
      path = { kind: "keys", keys: [] };
      store();
    } });
  clearTool.tone = themeVar("error");
  clearTool.style.color = clearTool.tone;
  const importTool = tool({ icon: "import", label: "Import",
    hint: "Read a camera rig: nerfstudio camera_path.json, Nuke .chan, or a glTF camera", onPress: () => fileInput.click() });
  const exportTool = tool({ icon: "export", label: "Export", hint: "Save the path as a camera rig",
    onPress: () => {
      const box = exportTool.getBoundingClientRect();
      openPopupMenu({
        x: box.left,
        y: box.bottom + 2,
        logName: EXT_NAME,
        items: [
          { header: "Save the path as" },
          { label: "nerfstudio camera_path.json", detail: "splat tools", onSelect: () => exportRig("nerfstudio") },
          { label: "Nuke .chan", detail: "Blender, Nuke, Houdini", onSelect: () => exportRig("chan") },
        ],
      });
    } });
  const alignTool = tool({ icon: "align", label: "Align", hint: "Move an imported rig so its first camera sits on the start camera",
    onPress: () => {
      alignImports = !alignImports;
      setSegment(alignTool, alignImports);
    } });
  setSegment(alignTool, alignImports);
  setSegment(flyTool, true);
  tools.append(
    toolGroup([flyTool, orbitTool, aspectTool], 3),
    toolGroup([recordTool, keyTool, playButton, startTool, clearTool], 5),
    toolGroup([importTool, exportTool, alignTool], 3),
  );
  const worded = [flyTool, orbitTool, recordTool, keyTool, playButton, startTool, clearTool, importTool, exportTool, alignTool];
  function fitTools() {
    const compact = root.clientWidth > 0 && root.clientWidth < COMPACT_WIDTH;
    for (const button of worded) button.word.style.display = compact ? "none" : "";
  }

  fileInput.addEventListener("change", async () => {
    const file = fileInput.files?.[0];
    fileInput.value = "";
    if (!file) return;
    await ready;
    try {
      let keys = await readRig(file);
      if (!keys.length) throw new Error("the file holds no camera");
      if (alignImports) keys = alignTo(keys, startPose());
      path = { kind: keys.length > 2 && keys[1].t - keys[0].t < 0.5 ? "recording" : "keys", keys };
      store();
      place(keys[0]);
    } catch (error) {
      summary.textContent = `Could not read ${file.name}: ${error?.message ?? error}`;
    }
  });

  function exportRig(format) {
    if (!THREE) return;
    if (path.keys.length < 1) {
      summary.textContent = "no path to export";
      return;
    }
    const frames = numberWidget("frames", 158);
    const fps = numberWidget("fps", 24) || 24;
    const end = path.keys.at(-1).t;
    const count = frames > 0 ? frames : Math.max(1, Math.round(end * fps) + 1);
    const poses = [];
    for (let index = 0; index < count; index += 1) poses.push(poseAt(path.keys, count > 1 ? (end * index) / (count - 1) : 0));
    const text = writeRig(format, poses, fps, ASPECTS[aspectName] ?? 16 / 9);
    download(text, format === "chan" ? "camera_path.chan" : "camera_path.json");
  }

  const resize = typeof ResizeObserver === "function" ? new ResizeObserver(() => {
    fitTools();
    layout();
  }) : null;
  resize?.observe(well);
  resize?.observe(root);
  const stopRatio = watchSurfaceRatio(canvas, () => layout());

  function refresh() {
    if (!THREE) {
      ready.then(() => refresh());
      return;
    }
    path = parsePath(pathWidget()?.value);
    const entry = app.nodeOutputs?.[nodeLocator(node)];
    if (entry && ownOutput(entry)) show(entry.was_point_cloud?.[0]);
    showPath();
    describe();
  }

  describe();

  return {
    element: root,
    height: PANEL_HEIGHT,
    minWidth: PANEL_MIN_WIDTH,
    show,
    refresh,
    dispose() {
      disposed = true;
      cancelAnimationFrame(frame);
      resize?.disconnect();
      stopRatio?.();
      releaseWheel?.();
      disposeCloud(cloudPoints);
      pathLine?.geometry.dispose();
      keyDots?.geometry.dispose();
      renderer?.dispose();
      renderer?.forceContextLoss?.();
      renderer = null;
    },
  };
}

app.registerExtension({
  name: EXT_NAME,

  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;
    const proto = nodeType.prototype;
    if (proto.__was_point_cloud_navigator_wrapped) return;
    proto.__was_point_cloud_navigator_wrapped = true;

    const originalCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalCreated?.apply(this, arguments);
      try {
        const panel = createNavigator(this);
        this.wasNavigator = panel;
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the navigator:`, error);
      }
      return result;
    };

    const originalExecuted = proto.onExecuted;
    proto.onExecuted = function (message) {
      const result = originalExecuted?.apply(this, arguments);
      try {
        this.wasNavigator?.show(message?.was_point_cloud?.[0]);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to show the cloud:`, error);
      }
      return result;
    };

    const originalConfigure = proto.onConfigure;
    proto.onConfigure = function () {
      const result = originalConfigure?.apply(this, arguments);
      try {
        this.wasNavigator?.refresh();
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to read the stored path:`, error);
      }
      return result;
    };

    const originalRemoved = proto.onRemoved;
    proto.onRemoved = function () {
      try {
        this.wasNavigator?.dispose();
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to release the navigator:`, error);
      }
      return originalRemoved?.apply(this, arguments);
    };
  },
});
