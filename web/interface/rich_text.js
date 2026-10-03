/**
 * The vendored rich text editor, as a panel a node can host.
 *
 * `createRichTextPanel` answers an element, the height it is pinned to and its teardown. The
 * document is read and written through the callbacks the caller hands over.
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { captureWheel } from "./pointer.js";
import { UPLOAD_TIMEOUT, fetchWithin } from "./request.js";
import { applyThemeVars, onThemeChange, readTheme, themeVar } from "./theme.js";

const LOG_NAME = "WASNodeSuite.RichText";

// The vendored editor's directory, taken from this module's own URL rather than spelled out, so
// the prefix follows whatever ComfyUI mounted the pack under. A default install resolves it to
// `/extensions/was-node-suite-comfyui/vendor/hugerte`.
const BASE_URL = new URL("../vendor/hugerte", import.meta.url).pathname;

// The editor is a global script rather than a module: its bundle is an IIFE that assigns
// `window.hugerte`, and it reads `window.hugeRTEPreInit` while that bundle runs. Loading it as a
// classic script keeps it in the sloppy mode it was compiled for.
const CORE_URL = `${BASE_URL}/hugerte.min.mjs`;
const ICONS_URL = `${BASE_URL}/icons/default/icons.min.mjs`;

// Every plugin vendored under `plugins/`, named by URL. The pack serves them as `.mjs`, which
// keeps ComfyUI's own `extensions/**/*.js` glob from importing them as extensions.
const PLUGINS = [
  "advlist",
  "autolink",
  "charmap",
  "code",
  "image",
  "link",
  "lists",
  "searchreplace",
  "table",
  "wordcount",
];

const TOOLBAR = [
  "undo redo",
  "blocks",
  "bold italic underline strikethrough",
  "forecolor backcolor removeformat",
  "alignleft aligncenter alignright alignjustify",
  "bullist numlist outdent indent",
  "link image table blockquote hr",
  "charmap searchreplace",
  "code",
].join(" | ");

// The policy of the frame a document is edited in. No script runs there: inline handlers,
// `<script>` and `javascript:` URLs included. Pictures and media load from ComfyUI's own origin,
// `data:` and `blob:`; fonts from that origin and `data:`; stylesheets from that origin and
// inline. Frames, requests, form posts and `<base>` are refused.
const CONTENT_SECURITY_POLICY = [
  "script-src 'none'",
  "object-src 'none'",
  "img-src 'self' data: blob:",
  "media-src 'self' data: blob:",
  "font-src 'self' data:",
  "style-src 'self' 'unsafe-inline'",
  "frame-src 'none'",
  "child-src 'none'",
  "connect-src 'none'",
  "form-action 'none'",
  "base-uri 'none'",
].join("; ");

// How long the editor may sit idle before an edit is written to the widget, and before the undo
// bracket around an editing session is closed. Short enough that a pause hands the work over,
// long enough that a paragraph is one entry in the graph's undo stack rather than fifty.
const COMMIT_IDLE_MS = 1500;

// How long a widget change made somewhere else waits before it is pushed into the editor.
// Typing into the node's own text box fires a change per keystroke, and each one would otherwise
// reload the document and drop the editor's own undo history.
const INBOUND_IDLE_MS = 300;

// Where an uploaded image is filed, under ComfyUI's input directory. Its own folder, so the
// pictures a document is built from do not land among the images somebody queues a workflow on.
const UPLOAD_SUBFOLDER = "rich-text";

const PLACEHOLDER = "Write the document here.";

const OPEN_LABEL = "Open the rich text editor";

// What the status bar says on hover. There is no route from the pack's configuration to the
// browser, so this states the default rather than the live setting. The last sentence is the
// part that holds whichever way that setting is left, and `editorOptions` says why.
const CLEANING_TITLE =
  "The node cleans the HTML it emits unless document.clean_html is set to false in config.yaml."
  + " Cleaning removes script and iframe elements, object and embed tags, on* handler attributes"
  + " and javascript: URLs, and writes a bare < in text as &lt;. Three of those change what a"
  + " reader sees: an iframe goes with everything between its tags, an object or an embed leaves"
  + " only the fallback that was between its tags, and a javascript: link keeps its words and"
  + " stops being a link. A script and an on* handler are already inert in this frame, so losing"
  + " them changes nothing on screen. Everything else goes out exactly as this editor holds it."
  + " This editor wraps loose text in a paragraph and moves an element out of a parent HTML does"
  + " not allow it in, whether or not cleaning is on.";

/**
 * The rules the editor's own chrome is repainted by.
 *
 * @returns {Array<string[]>} Each rule as a selector tail and its declarations.
 */
function chromeRules() {
  return [
    [
      ".tox-editor-header",
      `background-color:${themeVar("panelBg")};border-bottom-color:${themeVar("border")}`,
    ],
    [
      ".tox-toolbar,.tox-toolbar__primary,.tox-toolbar__overflow,.tox-toolbar-overlord,.tox-menubar",
      `background-color:${themeVar("panelBg")};background-image:none`,
    ],
    [
      ".tox-statusbar",
      `background-color:${themeVar("panelBg")};border-top-color:${themeVar("border")};`
        + `color:${themeVar("fgMuted")}`,
    ],
    [
      ".tox-statusbar a,.tox-statusbar__path-item,.tox-statusbar__wordcount",
      `color:${themeVar("fgMuted")}`,
    ],
    [".tox-edit-area__iframe", `background-color:${themeVar("inputBg")}`],
    [".tox-edit-area::before", `border-color:${themeVar("accent")}`],
    [".tox-tbtn", `background:transparent;color:${themeVar("fg")}`],
    [".tox-tbtn svg", "fill:currentColor"],
    [
      ".tox-tbtn:hover,.tox-tbtn:focus",
      `background:${themeVar("accentBg")};color:${themeVar("fg")}`,
    ],
    [
      ".tox-tbtn--active,.tox-tbtn--enabled,.tox-tbtn--enabled:hover,.tox-tbtn--enabled:focus,.tox-tbtn:active",
      `background:${themeVar("accent")};color:${themeVar("selectionText")}`,
    ],
    [
      ".tox-tbtn--disabled,.tox-tbtn--disabled:hover,.tox-tbtn:disabled,.tox-tbtn:disabled:hover",
      `background:transparent;color:${themeVar("fgDisabled")}`,
    ],
    [".tox-tbtn--bespoke", `background:${themeVar("bgDark")}`],
    [".tox-tbtn__select-chevron svg,.tox-split-button__chevron svg", `fill:${themeVar("fgMuted")}`],
    [".tox-menu", `background-color:${themeVar("bgDark")};border-color:${themeVar("border")}`],
    [".tox-collection__item", `color:${themeVar("fg")}`],
    [
      ".tox-collection--list .tox-collection__item--active,"
        + ".tox-collection--toolbar .tox-collection__item--active,"
        + ".tox-collection--grid .tox-collection__item--active",
      `background-color:${themeVar("accentBg")};color:${themeVar("fg")}`,
    ],
    [
      ".tox-collection--list .tox-collection__item--enabled,"
        + ".tox-collection--toolbar .tox-collection__item--enabled,"
        + ".tox-collection--grid .tox-collection__item--enabled",
      `background-color:${themeVar("accent")};color:${themeVar("selectionText")}`,
    ],
    [".tox-label,.tox-toolbar-label", `color:${themeVar("fgMuted")}`],
    [
      ".tox-textfield,.tox-textarea,.tox-listbox--select,.tox-toolbar-textfield",
      `background-color:${themeVar("inputBg")};border-color:${themeVar("border")};color:${themeVar("fg")}`,
    ],
    [
      ".tox-textfield:focus,.tox-textarea:focus,.tox-listbox--select:focus",
      `background-color:${themeVar("inputBg")};border-color:${themeVar("accent")};box-shadow:none`,
    ],
    [".tox-dialog", `background-color:${themeVar("bgDark")};border-color:${themeVar("border")}`],
    [
      ".tox-dialog__header,.tox-dialog__footer",
      `background-color:${themeVar("bgDark")};color:${themeVar("fg")}`,
    ],
    [".tox-dialog__body", `color:${themeVar("fg")}`],
    [".tox-dialog__body-content svg", "fill:currentColor"],
    [".tox-dialog__body-nav-item", `color:${themeVar("fgMuted")}`],
    [
      ".tox-dialog__body-nav-item--active,.tox-dialog__body-content a",
      `color:${themeVar("accent")};border-bottom-color:${themeVar("accent")}`,
    ],
    [
      ".tox-button",
      `background-color:${themeVar("accent")};border-color:${themeVar("accent")};`
        + `color:${themeVar("selectionText")}`,
    ],
    [
      ".tox-button:hover:not(:disabled),.tox-button:focus:not(:disabled),.tox-button:active:not(:disabled)",
      `background-color:${themeVar("accentHover")};border-color:${themeVar("accentHover")}`,
    ],
    [
      ".tox-button--secondary",
      `background-color:${themeVar("bgDark")};border-color:${themeVar("border")};color:${themeVar("fg")}`,
    ],
    [
      ".tox-button--secondary:hover:not(:disabled),.tox-button--secondary:focus:not(:disabled)",
      `background-color:${themeVar("accentBg")};border-color:${themeVar("border")};color:${themeVar("fg")}`,
    ],
    [".tox-button--naked", `color:${themeVar("fg")}`],
    [
      ".tox-tooltip__body",
      `background-color:${themeVar("bgDark")};color:${themeVar("fg")};`
        + `border:1px solid ${themeVar("border")}`,
    ],
    [".tox-tooltip--down .tox-tooltip__arrow", `border-top-color:${themeVar("bgDark")}`],
    [".tox-tooltip--up .tox-tooltip__arrow", `border-bottom-color:${themeVar("bgDark")}`],
    [".tox-tooltip--left .tox-tooltip__arrow", `border-right-color:${themeVar("bgDark")}`],
    [".tox-tooltip--right .tox-tooltip__arrow", `border-left-color:${themeVar("bgDark")}`],
  ];
}

// The editor's own containers. `tox-hugerte` is the box drawn on the node; `tox-hugerte-aux` is
// the sink it appends to `document.body` for dialogs, which sits outside the node and so outside
// the graph's transform, and is therefore drawn at page size.
const HOSTS = [".tox.tox-hugerte", ".tox.tox-hugerte-aux"];

const STYLE_ID = "was-rich-text-editor-theme";

// The stylesheet in the frame's own head, which the panel rewrites when the palette moves.
const CONTENT_STYLE_ID = "was-rich-text-content-theme";

/**
 * Expand one selector tail across the editor's containers.
 *
 * @param {string} tail - One or more comma separated selectors, relative to a container.
 * @returns {string} The same selectors under every container.
 */
function scopeRule(tail) {
  const parts = [];
  for (const one of tail.split(",")) {
    for (const host of HOSTS) parts.push(`${host} ${one.trim()}`);
  }
  return parts.join(",");
}

/**
 * The stylesheet that repaints the editor's chrome.
 *
 * @returns {string} CSS in terms of the palette's properties, with no colour of its own.
 */
function chromeStylesheet() {
  const rules = [
    `.tox.tox-hugerte{background-color:${themeVar("panelBg")};border:1px solid ${themeVar("border")};`
      + "border-radius:4px}",
    // The sink is an empty div appended to `document.body`, and the editor gives it an inline
    // `position:relative` and the page's width and nothing else. The skin has no rule for it at
    // all: `tox-silver-sink` does not appear anywhere in skin.min.css. So its height is whatever
    // the host page does to a plain child of its body, and under ComfyUI that is the full
    // viewport, at z-index 1300, over the graph canvas, painting nothing. Every click on the
    // graph lands on it instead, which is not a slow canvas but a dead one: the pointer never
    // reaches the node it was aimed at, and the only way out is a reload.
    //
    // Zero height takes it out of the page without touching the width the editor set for its own
    // layout, and `pointer-events` hands the empty area back to the canvas while leaving the
    // menus and dialogs inside it clickable. `!important` overrides the host's layout
    // stretching an in-flow child, which a plain declaration does not settle.
    ".tox.tox-silver-sink{height:0!important;pointer-events:none}",
    ".tox.tox-silver-sink>*{pointer-events:auto}",
    // The editor is started with a pixel height and keeps it, so dragging the node taller
    // grew the panel around an editor that stayed the size it opened at and left the
    // document in a sliver. Its own layout is flex the whole way down, so following the
    // container is one rule: the toolbar keeps its height and the edit area takes the rest.
    ".was-rich-text-panel .tox.tox-hugerte{height:100%!important}",
  ];
  for (const [tail, body] of chromeRules()) rules.push(`${scopeRule(tail)}{${body}}`);
  return rules.join("\n");
}

/**
 * Read one CSS colour back as the browser normalises it.
 *
 * @param {string} colour - Any CSS colour, in any of the syntaxes a palette may be written in.
 * @returns {number[]|null} Its red, green and blue in 0 to 255, or null when it is not a colour.
 */
function channels(colour) {
  const probe = document.createElement("span");
  probe.style.color = "";
  probe.style.color = String(colour ?? "");
  if (!probe.style.color) return null;
  probe.style.display = "none";
  document.body.appendChild(probe);
  const computed = getComputedStyle(probe).color;
  probe.remove();
  const found = /rgba?\(([^)]+)\)/.exec(computed);
  if (!found) return null;
  const numbers = found[1].split(/[,\s/]+/).map(Number);
  return numbers.length >= 3 && numbers.slice(0, 3).every(Number.isFinite)
    ? numbers.slice(0, 3)
    : null;
}

/**
 * Whether a colour is dark enough that white text sits on it.
 *
 * @param {string} colour - The paper colour the document is drawn on.
 * @returns {boolean} True when it is dark, and true when it cannot be read, since the base skin
 *   the overrides sit on is the dark one.
 */
function isDark(colour) {
  const rgb = channels(colour);
  if (!rgb) return true;
  return (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]) / 255 < 0.5;
}

/**
 * The stylesheet the document inside the frame is drawn with.
 *
 * @param {object} theme - Tokens from `readTheme`, read for the colour scheme alone.
 * @returns {string} CSS for the frame, in terms of the properties written onto its root.
 */
function contentStylesheet(theme) {
  const scheme = isDark(theme.inputBg) ? "dark" : "light";
  return [
    `:root{color-scheme:${scheme};`
      + `scrollbar-color:${themeVar("scrollbarThumb")} ${themeVar("scrollbarTrack")}}`,
    `body{background-color:${themeVar("inputBg")};color:${themeVar("fg")}}`,
    `a{color:${themeVar("accent")}}`,
    `hr{border-color:${themeVar("border")}}`,
    `code{background-color:${themeVar("panelBg")};color:${themeVar("fg")}}`,
    `figure figcaption{color:${themeVar("fgMuted")}}`,
    ".mce-content-body:not([dir=rtl]) blockquote,.mce-content-body[dir=rtl] blockquote"
      + `{border-color:${themeVar("border")}}`,
    'table[border]:not([border="0"]):not([style*=border-color]) td,'
      + 'table[border]:not([border="0"]):not([style*=border-color]) th'
      + `{border-color:${themeVar("border")}}`,
    "::selection{"
      + `background-color:${themeVar("selection")};color:${themeVar("selectionText")}}`,
  ].join("\n");
}

// One page, one core, one stylesheet, one key guard and one palette listener, however many nodes
// carry an editor. Each is taken by the first panel to open one and released by the last to go.
let corePromise = null;
let styleElement = null;
let keyGuard = null;
let stopTheme = null;
const livePanels = new Set();

/**
 * Fetch one classic script and resolve once it has run.
 *
 * @param {string} url - The script's URL, under the pack's own extension directory.
 * @returns {Promise<void>} Resolved on load, rejected when the browser could not run it.
 */
function loadScript(url) {
  return new Promise((resolve, reject) => {
    const element = document.createElement("script");
    element.src = url;
    element.async = false;
    element.addEventListener("load", () => resolve());
    element.addEventListener("error", () => reject(new Error(`Could not load ${url}`)));
    document.head.appendChild(element);
  });
}

/**
 * Load the vendored editor, once for the page.
 *
 * @returns {Promise<object>} The editor's global namespace, with its icon pack already
 *   registered so the loader does not go looking for one.
 */
function loadEditorCore() {
  if (corePromise) return corePromise;
  corePromise = (async () => {
    if (!window.hugerte) {
      // Read while the bundle runs, and the only way to tell it where it lives: it otherwise
      // works that out from the `src` of the script tag whose name matches its own.
      window.hugeRTEPreInit = { base: BASE_URL, suffix: ".min" };
      await loadScript(CORE_URL);
    }
    const core = window.hugerte;
    if (!core) throw new Error("The rich text editor loaded without registering itself.");
    if (!core.IconManager.has("default")) await loadScript(ICONS_URL);
    return core;
  })();
  return corePromise;
}

/**
 * Put one palette into every open editor's frame.
 *
 * @param {object} theme - Tokens from `readTheme`.
 * @returns {void}
 */
function applyTheme(theme) {
  for (const panel of livePanels) panel.applyContentTheme(theme);
}

/**
 * Follow a palette change while an editor is open, or add one holder to the subscription.
 *
 * @returns {void}
 */
function acquireTheme() {
  if (stopTheme) return;
  // The chrome reads the page root's properties on its own. This covers the frame, whose
  // document the page root does not reach.
  stopTheme = onThemeChange((theme) => {
    try {
      applyTheme(theme);
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to follow a palette change:`, error);
    }
  });
}

/**
 * Stop a keystroke reaching the graph from where it was typed.
 *
 * Propagation only: the default action is left alone.
 *
 * @param {KeyboardEvent} event - The keystroke, on its way up to the window.
 * @returns {void}
 */
function holdKey(event) {
  event.stopPropagation();
}

/**
 * Stop the keystrokes of the editor's dialogs from reaching the graph.
 *
 * @returns {void}
 */
function acquireKeyGuard() {
  if (keyGuard) return;
  keyGuard = (event) => {
    const target = event.target;
    if (!(target instanceof Element)) return;
    if (!target.closest(".tox, .was-rich-text-panel")) return;
    holdKey(event);
  };
  document.body.addEventListener("keydown", keyGuard);
}

/**
 * Put the chrome stylesheet in the page, after the skin it overrides.
 *
 * @returns {void}
 */
function acquireStylesheet() {
  if (!styleElement) {
    styleElement = document.createElement("style");
    styleElement.id = STYLE_ID;
    styleElement.textContent = chromeStylesheet();
  }
  // Appending an element already in the head moves it to the end.
  document.head.appendChild(styleElement);
}

/**
 * Take the page-wide parts an editor on screen needs, and count the panel holding them.
 *
 * @param {object} panel - The panel taking them, which the palette watcher calls back.
 * @returns {void}
 */
function acquireMounted(panel) {
  livePanels.add(panel);
  acquireStylesheet();
  acquireKeyGuard();
  acquireTheme();
}

/**
 * Give back the page-wide parts one panel took, once no panel is left holding them.
 *
 * @param {object} panel - The panel giving them back.
 * @returns {void}
 */
function releaseMounted(panel) {
  // A panel that never opened an editor is not in the set and took none of what follows.
  if (!livePanels.delete(panel)) return;
  if (livePanels.size) return;
  styleElement?.remove();
  styleElement = null;
  stopTheme?.();
  stopTheme = null;
  if (keyGuard) document.body.removeEventListener("keydown", keyGuard);
  keyGuard = null;
}

/**
 * Send one image to ComfyUI's own upload route and answer where it can be read back.
 *
 * @param {File|Blob} file - What to upload. A blob with no name is given one, since the route
 *   files an upload under the name it arrives with.
 * @returns {Promise<string>} A URL the frame can load the image from.
 * @throws {Error} When the route refuses the upload or answers something without a name in it.
 */
async function uploadImage(file) {
  const body = new FormData();
  body.append("image", file, file.name || "pasted-image.png");
  body.append("type", "input");
  body.append("subfolder", UPLOAD_SUBFOLDER);
  // The route renames rather than replaces, so two pictures of the same name both survive and
  // neither overwrites an image a workflow already points at.
  body.append("overwrite", "false");

  const response = await fetchWithin(
    "/upload/image", { method: "POST", body }, UPLOAD_TIMEOUT,
  );
  if (!response.ok) {
    throw new Error(`ComfyUI refused the upload with ${response.status} ${response.statusText}.`);
  }
  const filed = await response.json();
  if (!filed?.name) throw new Error("ComfyUI accepted the upload without saying where it went.");

  // Read back through the same view route the rest of the frontend uses, and take the folder and
  // the type from the answer rather than from what was asked for: the route is free to file an
  // upload somewhere other than where it was pointed.
  const query = new URLSearchParams({
    filename: filed.name,
    subfolder: filed.subfolder ?? "",
    type: filed.type ?? "input",
  });
  return api.apiURL(`/view?${query}`);
}

/**
 * The editor's handler for an image it wants uploaded, for paste and drop.
 *
 * @param {object} blobInfo - The editor's wrapper around the pasted or dropped image.
 * @returns {Promise<string>} The URL to put in the document.
 */
async function uploadBlob(blobInfo) {
  const blob = blobInfo.blob();
  const named = blob instanceof File ? blob : new File([blob], blobInfo.filename(), { type: blob.type });
  return uploadImage(named);
}

/**
 * The editor's handler for the browse button beside the Insert Image dialog's URL field.
 *
 * @param {(url: string, meta: object) => void} callback - Fills the dialog in with the answer.
 * @param {string} value - What the field holds now, unused: a pick replaces it.
 * @param {object} meta - What the dialog wants, of which only an image is offered.
 * @returns {void}
 */
function pickImage(callback, value, meta) {
  if (meta?.filetype !== "image") return;

  const chooser = document.createElement("input");
  chooser.type = "file";
  chooser.accept = "image/*";
  chooser.addEventListener("change", async () => {
    const file = chooser.files?.[0];
    if (!file) return;
    try {
      // The alt text starts as the file name.
      callback(await uploadImage(file), { alt: file.name });
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to upload an image:`, error);
      window.alert(`The image could not be uploaded: ${error.message}`);
    }
  });
  chooser.click();
}

/**
 * The options the editor is started with.
 *
 * @param {HTMLElement} host - The element the editor replaces.
 * @param {(editor: object) => void} setup - Run before the editor renders, to bind its events.
 * @param {boolean} autoUpload - Whether a pasted or dropped image is uploaded rather than kept
 *   in the document as a data URL.
 * @returns {object} Options for the editor's `init`.
 */
function editorOptions(host, setup, autoUpload) {
  const external = {};
  for (const name of PLUGINS) external[name] = `${BASE_URL}/plugins/${name}/plugin.min.mjs`;
  return {
    target: host,
    base_url: BASE_URL,
    suffix: ".min",
    theme_url: `${BASE_URL}/themes/silver/theme.min.mjs`,
    model_url: `${BASE_URL}/models/dom/model.min.mjs`,
    external_plugins: external,
    skin: "oxide-dark",
    content_css: "default",
    content_style: contentStylesheet(readTheme()),
    content_security_policy: CONTENT_SECURITY_POLICY,
    // Both are percentages so the editor is the size of the panel and follows it when the
    // node is dragged, rather than being fixed at whatever the panel measured on the frame
    // it started.
    height: "100%",
    width: "100%",
    menubar: false,
    statusbar: true,
    elementpath: true,
    branding: false,
    resize: false,
    toolbar: TOOLBAR,
    toolbar_mode: "sliding",
    placeholder: PLACEHOLDER,
    browser_spellcheck: true,
    // A pasted or dropped image stays a data URL in the document unless the caller asked for
    // uploads. Either way nothing leaves the machine: an upload is a POST to the ComfyUI this
    // page is served by, which is the same box.
    // A data URL travels with the document and every byte lives in the workflow. An upload
    // keeps the workflow small, and the document renders only while this ComfyUI is serving
    // `/api/view`.
    automatic_uploads: autoUpload,
    images_upload_handler: uploadBlob,
    // The Insert/Edit Image dialog takes a URL by hand, for a picture already on the web or
    // already in ComfyUI. This adds the other half: a button that uploads one from disk. It is
    // there whether or not paste and drop upload.
    file_picker_types: "image",
    file_picker_callback: pickImage,
    // Everything from here to `setup` is a rewrite the editor can be told to skip. `verify_html`
    // off widens `valid_elements` to `*[*]`, so no element is dropped for being unknown to the
    // schema and no attribute for not belonging to its element. It does not widen the schema's
    // lists of which children a parent may hold, and it does not reach the parser's `validate`
    // flag, which is hardcoded on.
    verify_html: false,
    xss_sanitization: false,
    convert_urls: false,
    entity_encoding: "raw",
    indent: false,
    allow_conditional_comments: true,
    allow_html_in_named_anchor: true,
    allow_unsafe_link_target: true,
    preserve_cdata: true,
    convert_unsafe_embeds: false,
    sandbox_iframes: false,
    // Left unset, `javascript:`, `vbscript:`, `mhtml:` and any non-image `data:` are deleted from
    // `src`, `href`, `data`, `background`, `action`, `formaction`, `poster` and `xlink:href`. None
    // of that is part of `xss_sanitization` and all of it happens with that off. This switch
    // reaches those attributes only. The same filtering inside a `style` attribute has no
    // switch behind it.
    allow_script_urls: true,
    // `<font>` would otherwise come back as a `<span>` carrying the same colour, family and
    // size, and `<strike>` as `<s>`. Both filters are installed together behind `inline_styles`,
    // which is the only switch that reaches the second of them; the first is named as well so
    // the intent survives `inline_styles`, which the editor marks deprecated.
    inline_styles: false,
    convert_fonts_to_spans: false,
    // The last `<br>` of a block would otherwise be dropped, a block left empty by that dropped
    // or padded with a non-breaking space, and a lone `<br>` replaced by one. Off, a trailing
    // `<br>` and a pair of them are kept wherever they were written. One trim still runs outside
    // this option, and it takes the whole document rather than a `<br>`: a document that
    // serialises to nothing but one empty forced root block comes back as the empty string, so
    // `<p>&nbsp;</p>` on its own reads back as "".
    remove_trailing_brs: false,
    // Every other rewrite the editor makes has no switch on this path, in either state of
    // `document.clean_html`. The panel compares what the editor holds rather than what the
    // widget held, and never writes the widget for a difference it made itself.
    setup,
  };
}

/**
 * Build the rich text editor panel for one node.
 *
 * @param {object} options - How the panel reaches the document and the graph.
 * @param {number} options.height - The height the widget starts at, in node units, and the
 *   least it is drawn at.
 * @param {number} [options.maxHeight] - The most it is drawn at, so the node can be dragged
 *   taller and the editor takes the room. Omitted, the panel is pinned to `height`.
 * @param {() => string} options.read - The document as the node's own widget holds it.
 * @param {(html: string) => void} options.commit - Store an edit in that widget.
 * @param {() => void} options.beginEdit - Open an undo bracket around an editing session.
 * @param {() => void} options.endEdit - Close it, which is what gives the session one undo entry.
 * @param {() => boolean} options.linked - Whether a link fills the input instead of the widget.
 * @param {() => boolean} [options.uploadImages] - Whether a pasted or dropped image is uploaded
 *   to ComfyUI rather than kept in the document as a data URL. Read once, when the editor is
 *   opened. The browse button in the Insert Image dialog uploads either way.
 * @returns {object} The panel: its element, height and maximum for `appendInterfaceWidget`, a
 *   `handleValueChanged` for the widget's callback, and `dispose`.
 */
export function createRichTextPanel(options) {
  const height = options.height;
  const maxHeight = options.maxHeight;

  const root = document.createElement("div");
  root.className = "was-rich-text-panel";
  root.style.cssText = [
    "position:relative",
    "box-sizing:border-box",
    "width:100%",
    "height:100%",
    // The editor's menus and colour pickers are drawn inside this element and are taller than
    // the room a node gives them, so the panel does not clip its own children.
    "overflow:visible",
  ].join(";");

  const host = document.createElement("div");
  host.style.cssText = "width:100%;height:100%";
  root.appendChild(host);

  const cover = document.createElement("div");
  cover.style.cssText = [
    "position:absolute",
    "inset:0",
    "display:flex",
    "flex-direction:column",
    "align-items:center",
    "justify-content:center",
    "gap:8px",
    "box-sizing:border-box",
    "padding:8px",
    "text-align:center",
    "font:12px sans-serif",
    `background:${themeVar("bg")}`,
    `color:${themeVar("fgMuted")}`,
  ].join(";");
  root.appendChild(cover);

  const openButton = document.createElement("button");
  openButton.type = "button";
  openButton.textContent = OPEN_LABEL;
  openButton.style.cssText = [
    "font:inherit",
    "padding:6px 14px",
    "border-radius:4px",
    "cursor:pointer",
    `background:${themeVar("inputBg")}`,
    `border:1px solid ${themeVar("border")}`,
    `color:${themeVar("fg")}`,
  ].join(";");
  cover.appendChild(openButton);

  const status = document.createElement("div");
  status.title = CLEANING_TITLE;
  cover.appendChild(status);

  const state = {
    editor: null,
    mounting: false,
    disposed: false,
    lastSync: null,
    session: false,
    commitTimer: 0,
    inboundTimer: 0,
    resizeFrame: 0,
    readonly: false,
  };

  /**
   * The document as the node's widget holds it.
   *
   * @returns {string} Its text, empty when the widget cannot be read.
   */
  function readValue() {
    const value = options.read?.();
    return typeof value === "string" ? value : "";
  }

  /**
   * Whether a link fills the input, which the schema's socketless flag should prevent.
   *
   * @returns {boolean} True while the run would read a link instead of the widget.
   */
  function isLinked() {
    try {
      return options.linked?.() === true;
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to read whether the input is linked:`, error);
      return false;
    }
  }

  /**
   * Repaint the cover, which is what stands in for the editor until it is opened.
   *
   * @returns {void}
   */
  function paintCover() {
    // Mid-mount the button and the words below it are the editor's progress, so a redraw leaves
    // them alone rather than putting the closed state back.
    if (state.mounting) return;

    const linked = isLinked();
    openButton.disabled = linked;
    openButton.style.opacity = linked ? "0.5" : "1";
    openButton.style.cursor = linked ? "not-allowed" : "pointer";

    if (linked) {
      status.style.color = themeVar("warning");
      status.textContent = "Driven by link";
      return;
    }
    status.style.color = themeVar("fgMuted");
    const length = readValue().length;
    status.textContent = length
      ? `${length.toLocaleString()} characters of HTML in the html box`
      : "the html box is empty";
  }

  /**
   * Draw whichever of the cover and the editor belongs on screen.
   *
   * @returns {void}
   */
  function refresh() {
    if (state.disposed) return;
    // The target element is hidden by the editor itself once it has replaced it, so the panel
    // only ever draws or hides its own cover. The cover comes back over a mounted editor when a
    // link fills the input, since the document on screen is then not what the run reads.
    const mounted = !!state.editor;
    const linked = isLinked();
    cover.style.display = mounted && !linked ? "none" : "flex";
    if (!mounted || linked) paintCover();
    if (!mounted) return;
    if (linked === state.readonly) return;
    state.readonly = linked;
    try {
      state.editor.mode.set(linked ? "readonly" : "design");
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to change the editor's mode:`, error);
    }
  }

  /**
   * Put the palette's colours inside the editor's frame.
   *
   * @param {object} theme - Tokens from `readTheme`.
   * @returns {void}
   */
  function applyContentTheme(theme) {
    if (!state.editor) return;
    const doc = state.editor.getDoc?.();
    if (!doc?.head) return;
    // A frame is its own document, which the page root's properties do not reach.
    applyThemeVars(doc.documentElement, theme);
    let element = doc.getElementById(CONTENT_STYLE_ID);
    if (!element) {
      element = doc.createElement("style");
      element.id = CONTENT_STYLE_ID;
      doc.head.appendChild(element);
    }
    element.textContent = contentStylesheet(theme);
  }

  /**
   * Open the undo bracket for an editing session, if one is not open.
   *
   * @returns {void}
   */
  function beginSession() {
    if (state.session) return;
    state.session = true;
    try {
      options.beginEdit?.();
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to open an undo bracket:`, error);
    }
  }

  /**
   * Close it, which is what gives the session one entry in the graph's undo stack.
   *
   * @returns {void}
   */
  function endSession() {
    if (!state.session) return;
    state.session = false;
    try {
      options.endEdit?.();
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to close an undo bracket:`, error);
    }
  }

  /**
   * Whether the editor reports its own document as changed since it was loaded.
   *
   * @returns {boolean} What `isDirty` answers, and false when it cannot be asked.
   */
  function isEdited() {
    try {
      return state.editor?.isDirty() === true;
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to read whether the document was edited:`, error);
      return false;
    }
  }

  /**
   * Write what the editor holds into the widget, and end the editing session.
   *
   * @returns {void}
   */
  function commitNow() {
    if (state.commitTimer) clearTimeout(state.commitTimer);
    state.commitTimer = 0;
    if (state.disposed || !state.editor) {
      endSession();
      return;
    }
    try {
      const html = state.editor.getContent();
      // `state.lastSync` is what the editor answered last, never what the widget held, so this
      // is a comparison of two strings the editor produced and a difference in it is an edit
      // rather than the editor's own normalisation of markup it was handed.
      if (html !== state.lastSync) {
        state.lastSync = html;
        // The bracket is opened here as well as by the typing timer, so a write that arrives by
        // any other path is still one entry in the graph's undo stack rather than none. It is
        // already open for a write the timer scheduled, and opening it twice does nothing.
        beginSession();
        options.commit?.(html);
      }
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to store the document:`, error);
    } finally {
      endSession();
    }
  }

  /**
   * Write the document once the typing has settled.
   *
   * @returns {void}
   */
  function scheduleCommit() {
    beginSession();
    if (state.commitTimer) clearTimeout(state.commitTimer);
    state.commitTimer = setTimeout(() => {
      state.commitTimer = 0;
      commitNow();
    }, COMMIT_IDLE_MS);
  }

  /**
   * Load into the editor a document that changed somewhere other than the editor.
   *
   * @returns {void}
   */
  function applyInbound() {
    state.inboundTimer = 0;
    if (state.disposed || !state.editor) return;
    const value = readValue();
    if (value === state.lastSync) return;
    // Written twice on purpose. The widget's own text first, so a load that throws still leaves
    // the widget counted as read and cannot be committed back over; then what the editor answers
    // once it holds the document, which is the only string a later `getContent` can be compared
    // with. See `editorOptions` for what the editor changes on the way in.
    state.lastSync = value;
    try {
      state.editor.setContent(value, { format: "html" });
      state.lastSync = state.editor.getContent();
      state.editor.undoManager.clear();
      state.editor.setDirty(false);
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to load the document into the editor:`, error);
    }
  }

  /**
   * Follow the widget after something else wrote it.
   *
   * The handler only reads.
   *
   * @returns {void}
   */
  function handleValueChanged() {
    if (state.disposed) return;
    // Also the hook a link change comes in on, so the panel's own state is redrawn either way.
    refresh();
    if (!state.editor) return;
    if (readValue() === state.lastSync) return;
    if (state.inboundTimer) clearTimeout(state.inboundTimer);
    state.inboundTimer = setTimeout(applyInbound, INBOUND_IDLE_MS);
  }

  /**
   * Bind one editor to the widget it is a view onto.
   *
   * @param {object} editor - The editor being started.
   * @returns {void}
   */
  function bindEditor(editor) {
    // Held from here rather than from the promise `init` answers, which lands after the
    // editor's own `init` event.
    state.editor = editor;
    editor.on("init", () => {
      if (state.disposed) return;
      try {
        editor.setContent(readValue(), { format: "html" });
        // What the editor holds, not what the widget held. The editor normalises a document as
        // it takes it in, and `editorOptions` lists what of that is left, so recording the
        // widget's own text here would leave the two different for any markup the editor did
        // not write and the first blur would store that difference as though it were an edit.
        state.lastSync = editor.getContent();
        editor.undoManager.clear();
        editor.setDirty(false);
        applyContentTheme(readTheme());
        const bar = editor.getContainer?.()?.querySelector(".tox-statusbar");
        if (bar) bar.title = CLEANING_TITLE;
        refresh();
        editor.focus();
      } catch (error) {
        console.error(`[${LOG_NAME}] Failed to show the document:`, error);
      }
    });
    // `change` is the editor's own undo level, so it lands once a paragraph rather than once a
    // keystroke; `input` covers a long burst of typing that has not reached one yet.
    editor.on("change undo redo", scheduleCommit);
    editor.on("input", scheduleCommit);
    editor.on("blur", () => {
      // A blur is not an edit. The editor takes focus as it mounts, so a click anywhere else on
      // the canvas blurs it whether or not anything was typed, and the widget is not the
      // editor's to write on the strength of that alone. Either signal is enough: a session is
      // open once the editor has reported a change, and the dirty flag covers a change it set
      // without reporting.
      if (!state.session && !isEdited()) return;
      commitNow();
    });
  }

  /**
   * Start the editor, which is what the cover's button does.
   *
   * @returns {Promise<void>} Resolved once the editor is on the node or the attempt has failed.
   */
  async function mount() {
    if (state.disposed || state.editor || state.mounting || isLinked()) return;
    if (!root.isConnected) return;
    state.mounting = true;
    openButton.disabled = true;
    status.textContent = "loading the editor";
    // Taken here rather than at construction, which leaves the page-wide cost of a panel nobody
    // opens at nothing. Every part of it is idempotent, so a second attempt after a failed one
    // takes them again for free, and `releaseMounted` gives them back either way.
    acquireMounted(panel);
    try {
      const core = await loadEditorCore();
      if (state.disposed) return;
      await core.init(editorOptions(host, bindEditor, options.uploadImages?.() === true));
      // Teardown that ran while the editor was starting has already released it.
      if (state.disposed) return;
      if (!state.editor) throw new Error("The editor did not start on this node.");
      refresh();
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to open the editor:`, error);
      status.textContent = "Editor failed, see console";
      openButton.disabled = false;
    } finally {
      state.mounting = false;
    }
  }

  openButton.addEventListener("click", () => {
    mount().catch((error) => console.error(`[${LOG_NAME}] Failed to open the editor:`, error));
  });

  /**
   * Wrap an event handler so a failure is logged rather than thrown at the browser.
   *
   * @param {(event: Event) => void} handler - Handler to wrap.
   * @returns {(event: Event) => void} The wrapped handler.
   */
  function guard(handler) {
    return (event) => {
      try {
        handler(event);
      } catch (error) {
        console.error(`[${LOG_NAME}] Rich text panel input failed:`, error);
      }
    };
  }

  // Bound on the panel rather than waiting for the body guard the editor's dialogs need. The
  // cover's Open button is focusable from the moment the panel exists, and Delete on it reaches
  // the node it is drawn on.
  root.addEventListener("keydown", guard(holdKey));

  root.addEventListener(
    "contextmenu",
    guard((event) => {
      // The graph canvas suppresses the browser menu on its own element only, and this is a
      // different element. The document itself is inside a frame and keeps its own menu, which is
      // where a spelling suggestion comes from.
      event.preventDefault();
      event.stopPropagation();
    }),
  );

  // Nothing reaching this handler scrolls: the toolbar slides rather than scrolls, the status
  // bar is one line, and the document is inside a frame whose own wheel events never leave it.
  // The editor's menus and dialogs are appended to `document.body` outside this element, so
  // their scrolling is not reached from here either. The panel takes the gesture and the graph
  // zooms from the canvas around the node.
  const releaseWheel = captureWheel(root);

  root.addEventListener(
    "pointerdown",
    guard((event) => {
      // Middle button panning belongs to the canvas underneath. Nothing on the chrome is a
      // middle button gesture, so the button is forwarded whole rather than shared.
      if (event.button === 1) app.canvas?.processMouseDown?.(event);
    }),
  );

  root.addEventListener(
    "pointermove",
    guard((event) => {
      if (event.buttons & 4) app.canvas?.processMouseMove?.(event);
    }),
  );

  root.addEventListener(
    "pointerup",
    guard((event) => {
      if (event.button === 1) app.canvas?.processMouseUp?.(event);
    }),
  );

  // What a closed cover follows a link change by: the pointer arriving over it, and the box
  // changing. Both are the panel's own, so a panel nobody opened still costs the page nothing.
  root.addEventListener(
    "pointerenter",
    guard(() => {
      if (!state.editor) refresh();
    }),
  );

  let observer = null;
  if (typeof ResizeObserver === "function") {
    observer = new ResizeObserver(() => {
      if (state.resizeFrame) return;
      state.resizeFrame = requestAnimationFrame(() => {
        state.resizeFrame = 0;
        if (!state.editor) {
          // Fires when the node is resized and when the widget is drawn again after a zoom
          // hid it.
          try {
            refresh();
          } catch (error) {
            console.error(`[${LOG_NAME}] Failed to repaint the cover:`, error);
          }
          return;
        }
        // The toolbar works out its own overflow from the width it last saw, and it hears about
        // a window resize but not about the node being dragged wider.
        try {
          state.editor?.dispatch("ResizeEditor");
        } catch (error) {
          console.error(`[${LOG_NAME}] Failed to relayout the toolbar:`, error);
        }
      });
    });
    observer.observe(root);
  }

  const panel = {
    element: root,
    height,
    maxHeight,
    handleValueChanged,
    applyContentTheme,
    /**
     * Release the editor, the timers and the observers this panel holds.
     *
     * @returns {void}
     */
    dispose() {
      if (state.disposed) return;
      state.disposed = true;
      releaseWheel();
      if (state.commitTimer) clearTimeout(state.commitTimer);
      if (state.inboundTimer) clearTimeout(state.inboundTimer);
      if (state.resizeFrame) cancelAnimationFrame(state.resizeFrame);
      state.commitTimer = 0;
      state.inboundTimer = 0;
      state.resizeFrame = 0;
      observer?.disconnect();
      observer = null;
      // Left open, the bracket stops the graph's change tracker taking any snapshot at all, for
      // every node on the canvas, so it is closed before anything else can throw.
      endSession();
      const editor = state.editor;
      state.editor = null;
      if (editor) {
        try {
          window.hugerte?.remove(editor);
        } catch (error) {
          console.error(`[${LOG_NAME}] Failed to release the editor:`, error);
        }
      }
      releaseMounted(panel);
    },
  };

  refresh();
  return panel;
}
