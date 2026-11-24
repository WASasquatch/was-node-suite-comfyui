/**
 * The Three Viewer's live view: the renderer, scene, camera and frame loop behind one node's panel.
 */

import * as THREE from "../vendor/three/three.module.mjs";
import {
    assertDescriptor,
    createOrbitControls,
    createRuntime,
    onRenderBusy,
    toneMappingConstant,
} from "./runtime.mjs";
import { surfaceRatio, watchSurfaceRatio } from "../interface/resolution.mjs";
import { Viewport } from "./viewport.mjs";
import {
    TilePacer,
    VIEW_LATENCY_MS,
    compilesInline,
    createTracer,
    disposeTracer,
    prepareFrame,
    traceBatch,
    tracingStation,
} from "./pathtrace.mjs";

// How far the camera may drift before a traced picture is started over. Damping moves it by
// a hair for a while after a drag, and starting over on that would never let it settle.
const CAMERA_SETTLED = 1e-5;

// Shortest gap between two live frames, in milliseconds: 60 a second on any display.
const FRAME_MS = 1000 / 60;

// Milliseconds a paused view keeps drawing after the pointer or a key touches it.
const STIR_MS = 500;

export class ViewerController {
    constructor(node, elements, tracing = false) {
        this.node = node;
        this.elements = elements;
        this.tracing = Boolean(tracing);
        this.tracer = null;
        this.pacer = null;
        this.station = null;
        this.screen = null;
        this.frame = null;
        this.loopToken = null;
        this.compileNoted = false;
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
        this.viewport = elements.viewportBar && !this.tracing ? new Viewport(this, elements.viewportBar) : null;
        this.renderHeld = false;
        this.heldStatus = "";
        this.stopRenderWatch = onRenderBusy((busy) => this.holdForRender(busy));
        this.drawnAt = 0;
        this.drawnView = null;
        this.stirred = 0;
        const stir = () => {
            this.stirred = performance.now();
        };
        for (const type of ["pointerdown", "pointermove", "wheel", "keydown"]) {
            elements.root.addEventListener(type, stir, { capture: true, passive: true });
        }

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

            const runtime = createRuntime(elements.canvas, elements.status, { network: payload.network === true });
            this.runtime = runtime;

            const transparent = appSpec.deps?.scene?.params?.backgroundMode === "transparent";
            const ratioLimit = Math.max(0.25, Number(appSpec.params?.pixelRatioLimit) || 2);
            const drawRatio = () => {
                const measured = Number(surfaceRatio(elements.canvas));
                return Math.min(Number.isFinite(measured) && measured > 0 ? measured : 1, ratioLimit);
            };
            this.drawRatio = drawRatio;
            this.toneMapping = toneMappingConstant(appSpec.params?.toneMapping);
            this.exposure = Number(appSpec.params?.exposure) || 1;

            let renderer = null;
            if (this.tracing) {
                // A traced view draws on the page's tracing station and copies each batch
                // onto its own canvas.
                this.station = tracingStation({
                    alpha: transparent,
                    antialias: appSpec.params?.antialias !== false,
                });
                renderer = this.station.renderer;
                this.screen = elements.canvas.getContext("2d");
                this.fitScreen(width, height);
            } else {
                renderer = new THREE.WebGLRenderer({
                    canvas: elements.canvas,
                    antialias: appSpec.params?.antialias !== false && !appSpec.deps?.effects,
                    alpha: transparent,
                });
                renderer.setPixelRatio(drawRatio());
                renderer.setSize(width, height, false);
                renderer.shadowMap.enabled = appSpec.params?.shadows !== false;
                renderer.shadowMap.type = THREE.PCFShadowMap;
                renderer.toneMapping = this.toneMapping;
                renderer.toneMappingExposure = this.exposure;
                renderer.outputColorSpace = THREE.SRGBColorSpace;
            }
            this.renderer = renderer;
            runtime.ctx.renderer = renderer;

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
                this.tracer = await createTracer(this.station, payload.trace || {});
                this.pacer = new TilePacer(renderer, VIEW_LATENCY_MS);
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
            if (this.viewport) await this.viewport.attach();

            // A moving scene starts a traced picture over every frame, so the animation is
            // held still and Play is what releases it.
            this.running = !this.tracing;
            this.lastTime = null;
            elements.playButton.textContent = this.running ? "Pause" : "Play";
            elements.status.textContent = this.tracing ? "Tracing" : "Running";
            if (this.renderHeld) {
                this.heldStatus = elements.status.textContent;
                elements.status.textContent = "Paused while a render draws";
            }

            const resize = () => {
                if (!this.renderer || !this.camera || !this.runtime) return;
                const nextRect = elements.canvasWrap.getBoundingClientRect();
                const nextWidth = Math.max(2, Math.floor(nextRect.width));
                const nextHeight = Math.max(2, Math.floor(nextRect.height));
                if (this.station) this.fitScreen(nextWidth, nextHeight);
                else this.renderer.setSize(nextWidth, nextHeight, false);
                this.composer?.setSize(nextWidth, nextHeight);
                this.runtime.updateCameraAspect(this.camera, nextWidth / nextHeight);
            };

            this.resizeObserver = new ResizeObserver(resize);
            this.resizeObserver.observe(elements.canvasWrap);

            this.stopRatio = watchSurfaceRatio(elements.canvas, () => {
                if (!this.renderer) return;
                if (!this.station) this.renderer.setPixelRatio(drawRatio());
                resize();
            });

            const step = (milliseconds) => {
                if (!this.renderer || !this.scene || !this.camera || !this.runtime) return;
                if (this.renderHeld) return;
                const now = performance.now();
                if (!this.tracer) {
                    const since = now - this.drawnAt;
                    if (since < FRAME_MS) return;
                    // Carries the remainder, so a 144 Hz display still averages 60.
                    this.drawnAt = now - (since % FRAME_MS);
                }

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
                if (!this.running && !this.viewChanged(now)) return;
                this.viewport?.beforeRender();
                if (this.composer) this.composer.render();
                else this.renderer.render(this.scene, this.camera);
            };

            if (this.station) {
                // The station is shared, so a traced view keeps its own frame loop.
                const token = {};
                this.loopToken = token;
                const tick = (milliseconds) => {
                    if (this.loopToken !== token) return;
                    step(milliseconds);
                    this.frame = requestAnimationFrame(tick);
                };
                this.frame = requestAnimationFrame(tick);
            } else {
                renderer.setAnimationLoop(step);
            }
        } catch (error) {
            console.error("[WAS ThreeJS] Viewer failed.", error);
            elements.status.textContent = `Error: ${error?.message || error}`;
        }
    }

    /**
     * Add a batch of tiles, starting the picture over where the view has moved.
     *
     * @returns {void}
     */
    traceStep() {
        // Nothing more is sent while the GPU is still on the last batch.
        if (!this.pacer.idle) return;
        if (this.station.holder) {
            this.elements.status.textContent = "Waiting for a render";
            return;
        }

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
        if (!this.station.warm && !this.compileNoted && compilesInline(this.renderer)) {
            // Shows the notice for one frame before the compile starts.
            this.compileNoted = true;
            this.elements.status.textContent = "Compiling the tracer, tab pauses";
            return;
        }
        this.configureStation();
        const batch = traceBatch(tracer, this.pacer, this.wanted);
        if (batch.drawn > 0) this.station.warm = true;
        const screen = this.screen;
        screen.clearRect(0, 0, screen.canvas.width, screen.canvas.height);
        screen.drawImage(this.station.canvas, 0, 0);
        this.elements.status.textContent = tracer.isCompiling
            ? "Compiling the tracer…"
            : `${Math.floor(tracer.samples)} of ${this.wanted} samples`;
    }

    /**
     * Size the traced view's own canvas to its box at the drawing ratio.
     *
     * @param {number} width - Box width in CSS pixels.
     * @param {number} height - Box height in CSS pixels.
     * @returns {void}
     */
    fitScreen(width, height) {
        const canvas = this.elements.canvas;
        const ratio = this.drawRatio?.() || 1;
        const nextWidth = Math.max(1, Math.floor(width * ratio));
        const nextHeight = Math.max(1, Math.floor(height * ratio));
        if (canvas.width !== nextWidth) canvas.width = nextWidth;
        if (canvas.height !== nextHeight) canvas.height = nextHeight;
    }

    /**
     * Set the shared station to this view's size and output before it draws for the view.
     *
     * @returns {void}
     */
    configureStation() {
        const renderer = this.renderer;
        const canvas = this.elements.canvas;
        renderer.setRenderTarget(null);
        if (renderer.getPixelRatio() !== 1) renderer.setPixelRatio(1);
        const surface = this.station.canvas;
        if (surface.width !== canvas.width || surface.height !== canvas.height) {
            renderer.setSize(canvas.width, canvas.height, false);
        }
        renderer.toneMapping = this.toneMapping;
        renderer.toneMappingExposure = this.exposure;
        renderer.outputColorSpace = THREE.SRGBColorSpace;
    }

    /**
     * Whether a paused view has anything new to draw, and note what it is drawing if so.
     *
     * @param {number} now - The frame's `performance.now()`.
     * @returns {boolean} True when the camera, the canvas size or recent input has moved.
     */
    viewChanged(now) {
        const camera = this.camera;
        const canvas = this.elements.canvas;
        const pose = [
            ...camera.position.toArray(),
            ...camera.quaternion.toArray(),
            canvas.width,
            canvas.height,
        ];
        const held = this.drawnView;
        let moved = !held || now - this.stirred < STIR_MS;
        for (let i = 0; !moved && i < pose.length; i += 1) {
            moved = Math.abs(pose[i] - held[i]) > CAMERA_SETTLED;
        }
        if (moved) this.drawnView = pose;
        return moved;
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

    /**
     * Stop drawing while a Three Render job uses the page, and pick up again after.
     *
     * @param {boolean} busy - True while a job draws.
     * @returns {void}
     */
    holdForRender(busy) {
        if (busy === this.renderHeld) return;
        this.renderHeld = busy;
        const status = this.elements.status;
        if (busy) {
            this.heldStatus = status.textContent;
            status.textContent = "Paused while a render draws";
            return;
        }
        if (status.textContent === "Paused while a render draws") status.textContent = this.heldStatus;
        // The clock starts again from the next frame rather than jumping the gap.
        this.lastTime = null;
    }

    async disposeRenderer() {
        this.viewport?.detach();
        this.drawnView = null;
        this.composer = null;
        this.loopToken = null;
        if (this.frame !== null) cancelAnimationFrame(this.frame);
        this.frame = null;
        this.pacer?.dispose();
        this.pacer = null;
        try {
            disposeTracer(this.tracer);
        } catch {}
        this.tracer = null;
        this.tracedFrom = null;
        this.tracedShape = null;
        try {
            this.resizeObserver?.disconnect();
        } catch {}
        this.resizeObserver = null;

        // The station's renderer is shared and kept for the page.
        const ownRenderer = this.station ? null : this.renderer;
        this.station = null;
        try {
            ownRenderer?.setAnimationLoop?.(null);
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
            ownRenderer?.dispose?.();
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
        this.stopRenderWatch?.();
        this.stopRenderWatch = null;
        await this.disposeRenderer();
        this.elements.releaseWheel?.();
        this.elements.root.remove();
    }
}
