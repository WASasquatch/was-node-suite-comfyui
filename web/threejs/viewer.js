/**
 * The Three.js viewer drawn on a node.
 *
 * A descriptor from the node's run is resolved into Three.js objects here. three.js is loaded
 * from `web/vendor/three`, so nothing is fetched over the network.
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import * as THREE from "../vendor/three/three.module.js";
import {
    assertDescriptor,
    createOrbitControls,
    createRuntime,
    toneMappingConstant,
} from "./runtime.js";
import { captureWheel } from "../interface/pointer.js";
import { surfaceRatio, watchSurfaceRatio } from "../interface/resolution.js";
import { appendInterfaceWidget } from "../interface/widget.js";
import { onNodeFinished } from "../interface/run_events.js";
import { createPathTracer, prepareFrame } from "./pathtrace.js";

const VIEWER_NODE = "WASThreeViewer";
const TRACE_VIEWER_NODE = "WASThreePathTraceViewer";

const UI_WIDGET_NAME = "was_threejs_viewer";
const UI_WIDGET_TYPE = "was_threejs";

// The surface opens at this many node units tall, plus the toolbar and status line under
// it, and is never drawn narrower than the minimum. It grows with the node from there.
const CHROME_HEIGHT = 54;
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

function createViewerDOM() {
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
    root.append(canvasWrap, toolbar);

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
    };
}

// How far the camera may drift before a traced picture is started over. Damping moves it by
// a hair for a while after a drag, and starting over on that would never let it settle.
const CAMERA_SETTLED = 1e-5;

function claimPointer(root) {
    // Stopped at the widget root, so nothing anywhere in the viewer reaches the node's widget
    // grid, which forwards pointer events to the graph canvas and stops them.
    for (const type of ["pointerdown", "pointermove", "pointerup", "pointercancel"]) {
        root.addEventListener(type, (event) => event.stopPropagation());
    }
    // The wheel dollies the camera on the scene's own canvas, which has already run by the time
    // the gesture reaches here, so the viewer claims it rather than dollying and zooming at
    // once. Ctrl or Cmd held reaches the graph.
    return captureWheel(root, () => true);
}

class ViewerController {
    constructor(node, elements, tracing = false) {
        this.node = node;
        this.elements = elements;
        this.tracing = Boolean(tracing);
        this.tracer = null;
        this.tracedFrom = null;
        this.tracedShape = null;
        this.wanted = 512;
        this.runtime = null;
        this.renderer = null;
        this.scene = null;
        this.camera = null;
        this.controls = null;
        this.resizeObserver = null;
        this.running = true;
        this.lastTime = null;
        this.lastPayload = null;
        this.initialCamera = null;

        this.elements.playButton.addEventListener("click", () => {
            this.running = !this.running;
            this.elements.playButton.textContent = this.running ? "Pause" : "Play";
        });

        this.elements.resetButton.addEventListener("click", () => {
            this.resetCamera();
        });

        this.elements.reloadButton.addEventListener("click", () => {
            if (this.lastPayload) {
                this.renderPayload(this.lastPayload);
            }
        });
    }

    async renderPayload(payload) {
        this.lastPayload = payload;
        await this.disposeRenderer();

        const elements = this.elements;
        elements.status.textContent = "Loading Three.js r185…";
        elements.detail.textContent = "Three.js r185";

        try {
            const appSpec = payload.app;
            assertDescriptor(appSpec, "app");

            const rect = elements.canvasWrap.getBoundingClientRect();
            const width = Math.max(2, Math.floor(rect.width || 420));
            const height = Math.max(2, Math.floor(rect.height || payload.viewerHeight || 360));
            const aspect = width / height;

            const runtime = createRuntime(elements.canvas, elements.status);
            this.runtime = runtime;

            const renderer = new THREE.WebGLRenderer({
                canvas: elements.canvas,
                antialias: appSpec.params?.antialias !== false && !appSpec.deps?.effects,
                alpha: appSpec.deps?.scene?.params?.backgroundMode === "transparent",
            });
            this.renderer = renderer;
            runtime.ctx.renderer = renderer;

            const ratioLimit = Math.max(0.25, Number(appSpec.params?.pixelRatioLimit) || 2);
            const drawRatio = () => {
                const measured = Number(surfaceRatio(elements.canvas));
                return Math.min(Number.isFinite(measured) && measured > 0 ? measured : 1, ratioLimit);
            };
            renderer.setPixelRatio(drawRatio());
            renderer.setSize(width, height, false);
            renderer.shadowMap.enabled = appSpec.params?.shadows !== false;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            renderer.toneMapping = toneMappingConstant(appSpec.params?.toneMapping);
            renderer.toneMappingExposure = Number(appSpec.params?.exposure) || 1;
            renderer.outputColorSpace = THREE.SRGBColorSpace;

            runtime.ctx.shadowMapSize = Number(appSpec.params?.shadowMapSize) || 2048;
            runtime.ctx.tracing = this.tracing;
            const scene = await runtime.buildScene(appSpec.deps.scene);
            const camera = runtime.buildCamera(appSpec.deps.camera, aspect);
            runtime.attachCameraTrack(camera, appSpec.deps.camera, scene);
            this.scene = scene;
            this.camera = camera;
            runtime.ctx.scene = scene;
            runtime.ctx.camera = camera;

            // A traced picture is presented by the tracer itself, so no effect chain is
            // built for one.
            this.composer = this.tracing ? null : await runtime.createComposer(
                appSpec.deps?.effects ?? null,
                { renderer, scene, camera, width, height },
            );

            if (this.tracing) {
                elements.status.textContent = "Building the path tracer…";
                this.tracer = await createPathTracer(renderer, payload.trace || {});
                this.wanted = Math.max(1, Math.floor(payload.trace?.samples) || 512);
                prepareFrame(this.tracer, scene, camera, true);

            }

            this.initialCamera = {
                position: camera.position.clone(),
                quaternion: camera.quaternion.clone(),
                target: camera.userData.comfyTarget?.clone?.() ?? new THREE.Vector3(),
            };

            // The live view has no capture, so a per-capture animation loops over this.
            runtime.ctx.timelineSeconds = Number(appSpec.params?.loopSeconds) || 4;
            runtime.ctx.duration = runtime.ctx.timelineSeconds;
            runtime.ctx.timeOrigin = 0;

            if (appSpec.params?.orbitControls) {
                const controls = createOrbitControls(camera, elements.canvas);
                controls.enableDamping = true;
                controls.target.copy(this.initialCamera.target);
                controls.sync();
                controls.autoRotate = Boolean(appSpec.params?.autoRotate);
                controls.autoRotateSpeed = Number(appSpec.params?.autoRotateSpeed) || 1;
                controls.update();
                this.controls = controls;
                runtime.ctx.controls = controls;
                elements.canvas.__wasControls = controls;
            }

            // A moving scene starts a traced picture over every frame, so the animation is
            // held still and Play is what releases it.
            this.running = !this.tracing;
            this.lastTime = null;
            elements.playButton.textContent = this.running ? "Pause" : "Play";
            elements.status.textContent = this.tracing ? "Tracing" : "Running";

            const resize = () => {
                if (!this.renderer || !this.camera || !this.runtime) return;
                const nextRect = elements.canvasWrap.getBoundingClientRect();
                const nextWidth = Math.max(2, Math.floor(nextRect.width));
                const nextHeight = Math.max(2, Math.floor(nextRect.height));
                this.renderer.setSize(nextWidth, nextHeight, false);
                this.composer?.setSize(nextWidth, nextHeight);
                this.runtime.updateCameraAspect(this.camera, nextWidth / nextHeight);
            };

            this.resizeObserver = new ResizeObserver(resize);
            this.resizeObserver.observe(elements.canvasWrap);

            this.stopRatio = watchSurfaceRatio(elements.canvas, () => {
                if (!this.renderer) return;
                this.renderer.setPixelRatio(drawRatio());
                resize();
            });

            renderer.setAnimationLoop((milliseconds) => {
                if (!this.renderer || !this.scene || !this.camera || !this.runtime) return;

                const time = milliseconds * 0.001;
                if (this.lastTime === null) this.lastTime = time;
                const delta = Math.min(0.1, Math.max(0, time - this.lastTime));
                this.lastTime = time;

                if (this.running) {
                    for (const update of this.runtime.ctx.updateFunctions) {
                        update({ time, delta, ctx: this.runtime.ctx });
                    }
                }

                // Damping is on, so the camera only moves on update. Outside the paused
                // check, so a paused scene can still be orbited and inspected.
                this.controls?.update?.();

                const currentRect = elements.canvasWrap.getBoundingClientRect();
                this.runtime.updateShaderUniforms(
                    time,
                    Math.max(2, Math.floor(currentRect.width)),
                    Math.max(2, Math.floor(currentRect.height))
                );
                if (this.tracer) {
                    this.traceStep();
                    return;
                }
                if (this.composer) this.composer.render();
                else this.renderer.render(this.scene, this.camera);
            });
        } catch (error) {
            console.error("[WAS ThreeJS] Viewer failed.", error);
            elements.status.textContent = `Error: ${error?.message || error}`;
        }
    }

    /**
     * Add one tile of samples, starting the picture over where the view has moved.
     *
     * @returns {void}
     */
    traceStep() {
        const tracer = this.tracer;
        const camera = this.camera;
        camera.updateMatrixWorld();

        // A resized canvas throws away what was drawn on it, and the shape it was drawn at is
        // baked into the camera's projection, of which the tracer holds its own copy. Losing
        // the pose is what makes the step below hand it the new one and start over.
        const canvas = this.elements.canvas;
        const shape = `${canvas.width}x${canvas.height}`;
        if (this.tracedShape !== shape) {
            this.tracedShape = shape;
            this.tracedFrom = null;
        }

        // Only a scene that actually moves is rebuilt each frame. A still one keeps gathering
        // whether the clock is running or not.
        if (this.running && this.runtime.ctx.updateFunctions.length) {
            prepareFrame(tracer, this.scene, camera, true);
        } else {
            const now = camera.matrixWorld.elements;
            const held = this.tracedFrom;
            let drift = Number.POSITIVE_INFINITY;
            if (held) {
                drift = 0;
                for (let i = 0; i < 16; i += 1) {
                    drift = Math.max(drift, Math.abs(now[i] - held[i]));
                }
            }
            // Damping nudges the camera for a while after a drag, and starting over on that
            // would never let the picture settle.
            if (drift > CAMERA_SETTLED) {
                this.tracedFrom = Float32Array.from(now);
                tracer.updateCamera();
            }
        }

        const standing = Math.floor(tracer.samples);
        if (standing >= this.wanted) {
            this.elements.status.textContent = `Settled at ${standing} samples`;
            return;
        }
        tracer.renderSample();
        this.elements.status.textContent = tracer.isCompiling
            ? "Compiling the tracer…"
            : `${Math.floor(tracer.samples)} of ${this.wanted} samples`;
    }

    resetCamera() {
        if (!this.camera || !this.initialCamera) return;
        this.camera.position.copy(this.initialCamera.position);
        this.camera.quaternion.copy(this.initialCamera.quaternion);
        if (this.controls) {
            this.controls.target.copy(this.initialCamera.target);
            this.controls.sync?.();
            this.controls.update();
        }
    }

    async disposeRenderer() {
        this.composer = null;
        try {
            this.tracer?.dispose?.();
        } catch {}
        this.tracer = null;
        this.tracedFrom = null;
        this.tracedShape = null;
        try {
            this.resizeObserver?.disconnect();
        } catch {}
        this.resizeObserver = null;

        try {
            this.renderer?.setAnimationLoop?.(null);
        } catch {}

        try {
            this.controls?.dispose?.();
            try {
                this.stopRatio?.();
            } catch (error) {
                /* the watcher is already gone */
            }
            this.stopRatio = null;
        } catch {}

        try {
            this.runtime?.dispose?.();
        } catch {}

        try {
            this.renderer?.dispose?.();
        } catch {}

        this.runtime = null;
        this.renderer = null;
        this.scene = null;
        this.camera = null;
        this.controls = null;
        this.initialCamera = null;
        this.lastTime = null;
    }

    async destroy() {
        await this.disposeRenderer();
        this.elements.releaseWheel?.();
        this.elements.root.remove();
    }
}

/**
 * Hand one run's payload to a node's viewer, unless that payload is already drawn.
 *
 * @param {object} node - The node the viewer is drawn on.
 * @param {string|object} raw - The payload, as JSON text or as an object.
 * @returns {void}
 */
function applyPayload(node, raw) {
    const controller = node?.threeViewerController;
    if (!raw || !controller) return;
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

app.registerExtension({
    name: "WASNodeSuite.ThreeJS",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        const name = nodeData?.name;
        if (name !== VIEWER_NODE && name !== TRACE_VIEWER_NODE) return;
        const tracing = name === TRACE_VIEWER_NODE;

        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            originalCreated?.apply(this, arguments);

            const elements = createViewerDOM();
            const controller = new ViewerController(this, elements, tracing);
            this.threeViewerController = controller;

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
                Math.max(this.size?.[1] || 0, DEFAULT_VIEWER_HEIGHT + CHROME_HEIGHT),
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
            this.threeViewerController?.destroy?.();
            this.threeViewerController = null;
            originalRemoved?.apply(this, arguments);
        };
    },
});
