/**
 * The Three.js viewer drawn on a node.
 *
 * The panel is built when the node is; three.js and the live view load the first time a viewer
 * is placed, from `web/vendor/three`.
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import { captureWheel } from "../interface/pointer.mjs";
import { appendInterfaceWidget } from "../interface/widget.mjs";
import { onNodeFinished } from "../interface/run_events.mjs";
import { createViewportBar, hideViewWidget } from "./viewport_bar.mjs";

const VIEWER_NODE = "WASThreeViewer";
const TRACE_VIEWER_NODE = "WASThreePathTraceViewer";

const UI_WIDGET_NAME = "was_threejs_viewer";
const UI_WIDGET_TYPE = "was_threejs";

// The surface opens at this many node units tall, plus the toolbar and status line under
// it, and is never drawn narrower than the minimum. It grows with the node from there.
const CHROME_HEIGHT = 54;
// The viewport tools add a row, which wraps to two on a narrow node.
const VIEWPORT_ROW_HEIGHT = 64;
const DEFAULT_VIEWER_HEIGHT = 360;
const MIN_VIEWER_WIDTH = 320;

function applyViewerStyles(element) {
    element.style.width = "100%";
    element.style.height = "100%";
    element.style.display = "flex";
    element.style.flexDirection = "column";
    element.style.gap = "6px";
    element.style.boxSizing = "border-box";
    element.style.padding = "6px";
    element.style.background = "rgba(0,0,0,0.16)";
    element.style.borderRadius = "6px";
    element.style.overflow = "hidden";
}

function createButton(label) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.style.minHeight = "28px";
    button.style.padding = "3px 9px";
    button.style.borderRadius = "4px";
    button.style.border = "1px solid var(--border-color, #555)";
    button.style.background = "var(--comfy-input-bg, #222)";
    button.style.color = "var(--input-text, #ddd)";
    button.style.cursor = "pointer";
    return button;
}

function createViewerDOM(withViewport = false) {
    const root = document.createElement("div");
    applyViewerStyles(root);

    const canvasWrap = document.createElement("div");
    canvasWrap.style.position = "relative";
    canvasWrap.style.width = "100%";
    canvasWrap.style.flex = "1 1 auto";
    canvasWrap.style.minHeight = "0";
    canvasWrap.style.background = "#0c0c0c";
    canvasWrap.style.borderRadius = "4px";
    canvasWrap.style.overflow = "hidden";

    const canvas = document.createElement("canvas");
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    canvas.style.display = "block";
    canvas.setAttribute("aria-label", "Three.js viewer canvas");

    const status = document.createElement("div");
    status.style.position = "absolute";
    status.style.left = "8px";
    status.style.top = "8px";
    status.style.maxWidth = "calc(100% - 16px)";
    status.style.padding = "4px 7px";
    status.style.borderRadius = "4px";
    status.style.background = "rgba(0,0,0,0.62)";
    status.style.color = "#ddd";
    status.style.font = "11px/1.35 sans-serif";
    status.style.pointerEvents = "none";
    status.textContent = "Execute the node to render.";

    const toolbar = document.createElement("div");
    toolbar.style.display = "flex";
    toolbar.style.flexWrap = "wrap";
    toolbar.style.gap = "6px";
    toolbar.style.alignItems = "center";

    const playButton = createButton("Pause");
    const resetButton = createButton("Reset Camera");
    const reloadButton = createButton("Reload");

    const detail = document.createElement("span");
    detail.style.font = "11px/1.35 sans-serif";
    detail.style.color = "var(--descrip-text, #aaa)";
    detail.style.marginLeft = "auto";
    detail.textContent = `Three.js r185`;

    toolbar.append(playButton, resetButton, reloadButton, detail);
    canvasWrap.append(canvas, status);
    const viewportBar = withViewport ? createViewportBar() : null;
    if (viewportBar) {
        canvasWrap.append(viewportBar.overlay);
        root.append(canvasWrap, viewportBar.row, toolbar);
    } else {
        root.append(canvasWrap, toolbar);
    }

    const releaseWheel = claimPointer(root);

    return {
        root,
        canvasWrap,
        canvas,
        status,
        playButton,
        resetButton,
        reloadButton,
        detail,
        releaseWheel,
        viewportBar,
    };
}

function claimPointer(root) {
    // Stopped at the widget root, so nothing anywhere in the viewer reaches the node's widget
    // grid, which forwards pointer events to the graph canvas and stops them.
    for (const type of ["pointerdown", "pointermove", "pointerup", "pointercancel"]) {
        root.addEventListener(type, (event) => event.stopPropagation());
    }
    // A right drag pans the camera, so a right click opens neither the node's menu nor the
    // browser's.
    root.addEventListener("contextmenu", (event) => {
        event.preventDefault();
        event.stopPropagation();
    });
    // The wheel dollies the camera on the scene's own canvas, which has already run by the time
    // the gesture reaches here, so the viewer claims it rather than dollying and zooming at
    // once. Ctrl or Cmd held reaches the graph.
    return captureWheel(root, () => true);
}

/**
 * Hand one run's payload to a node's viewer, unless that payload is already drawn.
 *
 * @param {object} node - The node the viewer is drawn on.
 * @param {string|object} raw - The payload, as JSON text or as an object.
 * @returns {void}
 */
function applyPayload(node, raw) {
    if (!raw || !node) return;
    const controller = node.threeViewerController;
    if (!controller) {
        // Drawn once the live view has loaded.
        node.threeViewerPending = raw;
        return;
    }
    const text = typeof raw === "string" ? raw : JSON.stringify(raw);
    if (text === node.threeViewerPayloadText) return;
    try {
        controller.renderPayload(typeof raw === "string" ? JSON.parse(raw) : raw);
        node.threeViewerPayloadText = text;
    } catch (error) {
        console.error("[WAS ThreeJS] Invalid viewer payload.", error);
        controller.elements.status.textContent = `Payload error: ${error?.message || error}`;
    }
}

/**
 * Read one node's viewer payload out of a finished run.
 *
 * @param {string} promptId - The run to read.
 * @param {string} nodeId - The node's execution id.
 * @returns {Promise<string|null>} The payload, or null where the run holds none for that node.
 */
async function payloadFromRun(promptId, nodeId) {
    if (!promptId || !nodeId) return null;
    const answer = await fetch(api.apiURL(`/history/${encodeURIComponent(promptId)}`));
    if (!answer.ok) return null;
    const history = await answer.json();
    const outputs = history?.[promptId]?.outputs ?? {};
    return outputs[nodeId]?.three_app?.[0] ?? null;
}

let controllerModule = null;

/**
 * The live view's module, imported the first time a viewer asks for it.
 *
 * @returns {Promise<object>} The module, exporting `ViewerController`.
 */
function loadController() {
    controllerModule ??= import("./viewer_controller.mjs");
    return controllerModule;
}

/**
 * Give a node's panel its live view once three.js has loaded, and draw any run it missed.
 *
 * @param {object} node - A Three Viewer node.
 * @param {object} elements - What `createViewerDOM` built for it.
 * @param {boolean} tracing - True for the path traced viewer.
 * @returns {void}
 */
function attachController(node, elements, tracing) {
    loadController()
        .then(({ ViewerController }) => {
            if (node.threeViewerRemoved) return;
            node.threeViewerController = new ViewerController(node, elements, tracing);
            const pending = node.threeViewerPending;
            node.threeViewerPending = null;
            if (pending) applyPayload(node, pending);
        })
        .catch((error) => {
            console.error("[WAS ThreeJS] The viewer did not load.", error);
            elements.status.textContent = `Error: ${error?.message || error}`;
        });
}

app.registerExtension({
    name: "WASNodeSuite.ThreeJS",

    beforeRegisterNodeDef(nodeType, nodeData) {
        const name = nodeData?.name;
        if (name !== VIEWER_NODE && name !== TRACE_VIEWER_NODE) return;
        const tracing = name === TRACE_VIEWER_NODE;

        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            originalCreated?.apply(this, arguments);

            const elements = createViewerDOM(!tracing);
            if (!tracing) hideViewWidget(this);
            this.threeViewerElements = elements;
            attachController(this, elements, tracing);

            appendInterfaceWidget(
                this,
                {
                    element: elements.root,
                    height: DEFAULT_VIEWER_HEIGHT,
                    maxHeight: Number.MAX_SAFE_INTEGER,
                    minWidth: MIN_VIEWER_WIDTH,
                },
                { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE },
            );

            this.setSize?.([
                Math.max(this.size?.[0] || 0, MIN_VIEWER_WIDTH),
                Math.max(this.size?.[1] || 0, DEFAULT_VIEWER_HEIGHT + CHROME_HEIGHT + (tracing ? 0 : VIEWPORT_ROW_HEIGHT)),
            ]);

            this.threeViewerUnwatch = onNodeFinished(this, ({ promptId, nodeId }) => {
                payloadFromRun(promptId, nodeId)
                    .then((raw) => applyPayload(this, raw))
                    .catch((error) => console.error("[WAS ThreeJS] Run payload failed.", error));
            });
        };

        const originalExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            originalExecuted?.apply(this, arguments);
            applyPayload(this, message?.three_app?.[0]);
        };

        const originalRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            this.threeViewerUnwatch?.();
            this.threeViewerUnwatch = null;
            this.threeViewerRemoved = true;
            if (this.threeViewerController) {
                this.threeViewerController.destroy?.();
            } else {
                this.threeViewerElements?.releaseWheel?.();
                this.threeViewerElements?.root.remove();
            }
            this.threeViewerController = null;
            this.threeViewerElements = null;
            originalRemoved?.apply(this, arguments);
        };
    },
});
