/**
 * Rich text editor for the Rich Text Editor node.
 *
 * Drawn under the node's widgets as a view onto the `html` widget: every edit is written back
 * into it, and a change made elsewhere is loaded back in.
 */

import { app } from "../../scripts/app.js";
import { createRichTextPanel } from "./interface/rich_text.js";
import { appendInterfaceWidget, boundTextBoxes } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.RichTextEditorUI";
const NODE_NAME = "WASRichTextEditor";
const SETTING_ID = "WAS.RichTextEditor.ShowInterface";
const UPLOAD_SETTING_ID = "WAS.RichTextEditor.UploadImages";
const HEIGHT_SETTING_ID = "WAS.RichTextEditor.PanelHeight";

const HTML_WIDGET = "html";

const UI_WIDGET_NAME = "was_rich_text_editor_ui";
const UI_WIDGET_TYPE = "was_rich_text_editor";

// The height the panel opens at, in node units, and the range the setting may put it in. The
// panel is not pinned to it: dragging the node taller hands the extra room to the editor, which
// is the only widget on the node that can use it.
const DEFAULT_HEIGHT = 320;
const MIN_HEIGHT = 160;
const MAX_HEIGHT = 1200;

/**
 * Find a widget on a node by name.
 *
 * @param {object} node - Node to search.
 * @param {string} name - Widget name.
 * @returns {object|null} The widget, or null when the node does not carry it.
 */
function findWidget(node, name) {
  const widgets = Array.isArray(node?.widgets) ? node.widgets : [];
  for (const widget of widgets) {
    if (widget?.name === name) return widget;
  }
  return null;
}

/**
 * Test whether one of a node's inputs is linked.
 *
 * @param {object} node - Node to search.
 * @param {string} name - Input name.
 * @returns {boolean} True while a link is attached to that input.
 */
function inputLinked(node, name) {
  const inputs = Array.isArray(node?.inputs) ? node.inputs : [];
  for (const input of inputs) {
    if (input?.name === name) return input.link !== null && input.link !== undefined;
  }
  return false;
}

/**
 * Read one of the pack's settings.
 *
 * @param {string} id - The setting's id.
 * @returns {unknown} Its value, or undefined when it cannot be read.
 */
function readSetting(id) {
  try {
    const value = app?.extensionManager?.setting?.get?.(id);
    if (value !== undefined && value !== null) return value;
    return app?.ui?.settings?.getSettingValue?.(id);
  } catch (error) {
    console.error(`[${EXT_NAME}] Failed to read ${id}:`, error);
    return undefined;
  }
}

/**
 * Read whether the editor is drawn at all.
 *
 * @returns {boolean} True while the setting is on or cannot be read.
 */
function interfaceEnabled() {
  const value = readSetting(SETTING_ID);
  return typeof value === "boolean" ? value : true;
}

/**
 * Read the height the panel is drawn at.
 *
 * @returns {number} The setting, held inside the range the panel is usable over, and the default
 *   for anything that is not a number.
 */
function panelHeight() {
  const value = Number(readSetting(HEIGHT_SETTING_ID));
  if (!Number.isFinite(value)) return DEFAULT_HEIGHT;
  return Math.min(MAX_HEIGHT, Math.max(MIN_HEIGHT, Math.round(value)));
}

/**
 * Chain a handler onto a widget's callback.
 *
 * @param {object} node - Node holding the widget.
 * @param {string} name - Widget name.
 * @param {() => void} onChange - Called after the original callback.
 * @returns {void}
 */
function chainWidgetCallback(node, name, onChange) {
  const widget = findWidget(node, name);
  if (!widget) return;
  const original = widget.callback;
  widget.callback = function (...args) {
    const result = original?.apply(this, args);
    try {
      onChange();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to follow a widget change:`, error);
    }
    return result;
  };
}


/**
 * Append the editor to a node and wire it to the widget the document lives in.
 *
 * @param {object} node - The node being created.
 * @returns {void}
 */
function attachEditor(node) {
  if (!findWidget(node, HTML_WIDGET)) return;

  // The canvas the bracket was opened on, so a graph swapped underneath a live session still
  // gets its closing event. Both halves are feature-detected together: emitting only the first
  // leaves the change tracker's nesting count above zero and stops it snapshotting anything.
  let bracketed = null;

  const panel = createRichTextPanel({
    height: panelHeight(),
    // No ceiling worth naming: the node's own height is the bound, and a node cannot be dragged
    // taller than the room it is given.
    maxHeight: Number.MAX_SAFE_INTEGER,
    read: () => findWidget(node, HTML_WIDGET)?.value ?? "",
    commit: (html) => {
      const widget = findWidget(node, HTML_WIDGET);
      if (!widget || widget.value === html) return;
      widget.value = html;
      node.setDirtyCanvas?.(true, true);
    },
    beginEdit: () => {
      const canvas = app.canvas;
      if (typeof canvas?.emitBeforeChange !== "function") return;
      if (typeof canvas?.emitAfterChange !== "function") return;
      // Recorded only once the opening event has gone out, so a throw there does not leave a
      // closing event to be sent on its own and drive the tracker's nesting count negative.
      canvas.emitBeforeChange();
      bracketed = canvas;
    },
    endEdit: () => {
      const canvas = bracketed;
      bracketed = null;
      canvas?.emitAfterChange();
    },
    linked: () => inputLinked(node, HTML_WIDGET),
    uploadImages: () => readSetting(UPLOAD_SETTING_ID) === true,
  });

  // Bounded like every other box in the pack rather than pinned at a fixed height. The editor
  // still takes the room past the ceiling, and the source box grows with the node up to it
  // instead of being the one box that never moves.
  boundTextBoxes(node);

  // Appended after every schema widget, with both serialize flags set, which is what
  // `appendInterfaceWidget` is for.
  appendInterfaceWidget(node, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

  chainWidgetCallback(node, HTML_WIDGET, panel.handleValueChanged);

  // A widget value is the default until `configure` has run, so the saved document reaches the
  // panel from here rather than on creation.
  const originalOnConfigure = node.onConfigure;
  node.onConfigure = function (...args) {
    const result = originalOnConfigure?.apply(this, args);
    try {
      panel.handleValueChanged();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to read the document after a workflow load:`, error);
    }
    return result;
  };

  // A link cannot be made to a socketless input, so this only matters on a frontend that ignores
  // the flag: the panel goes read only rather than showing a document the run will not read.
  const originalOnConnectionsChange = node.onConnectionsChange;
  node.onConnectionsChange = function (...args) {
    const result = originalOnConnectionsChange?.apply(this, args);
    try {
      panel.handleValueChanged();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to follow a connection change:`, error);
    }
    return result;
  };

  // The original runs first: `addDOMWidget` chains the frontend's own widget teardown onto
  // `onRemoved`, so anything that ran before it and threw would leave the widget registered and
  // its element in the page.
  const originalOnRemoved = node.onRemoved;
  node.onRemoved = function (...args) {
    const result = originalOnRemoved?.apply(this, args);
    try {
      panel.dispose();
    } catch (error) {
      console.error(`[${EXT_NAME}] Failed to release the editor:`, error);
    }
    return result;
  };
}

app.registerExtension({
  name: EXT_NAME,
  settings: [
    {
      id: SETTING_ID,
      category: ["WAS Node Suite", "Rich Text Editor", "Editor"],
      name: "Show the rich text editor",
      tooltip:
        "Draw a rich text editor under the widgets of Rich Text Editor. The html box is always "
        + "available and is where the document lives either way. This applies to nodes added "
        + "after the setting changes, so a reload shows it everywhere.",
      type: "boolean",
      defaultValue: true,
    },
    {
      id: UPLOAD_SETTING_ID,
      category: ["WAS Node Suite", "Rich Text Editor", "Images"],
      name: "Upload pasted and dropped images",
      tooltip:
        "Send an image pasted or dropped into the editor to ComfyUI's input folder and link to "
        + "it, instead of embedding it in the document as a data URL. Uploading keeps the "
        + "workflow small and puts the picture in the Load Image menu, and the document then "
        + "only renders while this ComfyUI is running. Embedding makes the document carry its "
        + "own pictures anywhere it is saved or pasted, at the cost of the workflow holding "
        + "every byte. Either way the Insert Image dialog can upload one from disk on demand.",
      type: "boolean",
      defaultValue: false,
    },
    {
      id: HEIGHT_SETTING_ID,
      category: ["WAS Node Suite", "Rich Text Editor", "Height"],
      name: "Editor height in pixels",
      tooltip:
        "How tall the rich text editor opens on the node. The toolbar and the status bar take "
        + "about 70 pixels of it and the rest is the document. Dragging the node taller grows "
        + "the editor past this, so it is a starting height rather than a limit. This applies to "
        + "nodes added after the setting changes, so a reload resizes the ones already on the "
        + "canvas.",
      type: "number",
      defaultValue: DEFAULT_HEIGHT,
      attrs: { min: MIN_HEIGHT, max: MAX_HEIGHT, step: 20 },
    },
  ],
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) return;

    const proto = nodeType.prototype;

    // Node definitions are registered again on a definitions refresh, which would otherwise wrap
    // the prototype a second time and append a second editor.
    if (proto.__was_rich_text_editor_wrapped) return;
    proto.__was_rich_text_editor_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        if (interfaceEnabled()) attachEditor(this);
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the rich text editor:`, error);
      }
      return result;
    };
  },
});
