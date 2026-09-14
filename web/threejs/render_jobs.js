/**
 * Frames drawn off screen for a Three Render node that is waiting on them.
 *
 * Polls for a filed job, draws it on a canvas of its own and posts the PNG back.
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";
import * as THREE from "../vendor/three/three.module.js";

import { createPathTracer, prepareFrame, traceFrame } from "./pathtrace.js";
import { createRuntime, toneMappingConstant } from "./runtime.js";

const ROUTE = "/was/threejs/api/render";

// How often the browser looks for a job, in milliseconds. A node files one and then waits, so
// this is the delay before its render starts.
const POLL_MS = 250;

// Backed off to this while nothing is queued, so an idle tab is not polling four times a second.
const IDLE_MS = 1500;

// Seconds per step while the scene is wound forward to the capture time, and the most steps
// taken. Part of a scene's motion is added up from each step rather than read off the clock,
// so the capture time is reached by stepping rather than jumped to.
const STEP_SECONDS = 1 / 60;
const MAX_STEPS = 4096;

// The most the frame is drawn oversize before it is scaled down. Four is 16 times the
// pixels, which is where a browser starts refusing the drawing buffer.
const MAX_SUPERSAMPLE = 4;

// Draws thrown away before the first frame is kept, so both composer targets are
// written at least once.
const WARM_UP_DRAWS = 3;

let running = false;

/**
 * Run the scene's updates from one moment to the next.
 *
 * @param {object} runtime - The runtime holding the scene's update functions.
 * @param {number} from - Where the scene already stands, in seconds.
 * @param {number} to - The moment to wind to, in seconds.
 * @returns {void}
 */
function windForward(runtime, from, to) {
    const updates = runtime.ctx.updateFunctions;
    if (!updates.length) return;
    const span = to - from;
    if (span <= 0) {
        // A run beginning at its own first moment has nothing to wind through. The updates
        // still run once there, which is what takes a rigged model out of its bind pose.
        for (const update of updates) update({ time: to, delta: 0, ctx: runtime.ctx });
        return;
    }
    const steps = Math.min(MAX_STEPS, Math.max(1, Math.ceil(span / STEP_SECONDS)));
    const delta = span / steps;
    for (let step = 1; step <= steps; step += 1) {
        const time = from + delta * step;
        for (const update of updates) update({ time, delta, ctx: runtime.ctx });
    }
}

/**
 * Draw one job and answer the PNG.
 *
 * @param {object} job - `{token, app, width, height, transparent, time}` from the route.
 * @param {Function} onFrame - Called with each frame's index and its `{png, depth, normal}`
 *   data URLs.
 * @param {Function} onProgress - Called with `{done, note}` as the work advances.
 * @returns {Promise<void>} Nothing. Every frame is handed to `onFrame` as it is drawn.
 */
async function drawJob(job, onFrame, onProgress) {
    const width = Math.max(16, Math.floor(job.width));
    const height = Math.max(16, Math.floor(job.height));
    // A traced frame is already antialiased by where its samples land inside each pixel, so
    // it is drawn at the size asked for.
    const asked = job.trace ? 1 : Math.floor(job.supersample) || 1;
    const over = Math.max(1, Math.min(MAX_SUPERSAMPLE, asked));
    const drawWidth = width * over;
    const drawHeight = height * over;
    const spec = job.app;
    const times = Array.isArray(job.times) && job.times.length ? job.times : [0];

    const canvas = document.createElement("canvas");
    canvas.width = drawWidth;
    canvas.height = drawHeight;

    // Drawn oversize and scaled down here, which is what smooths an edge the renderer's own
    // antialias leaves stepped. At 1 the two are the same canvas.
    const output = over > 1 ? document.createElement("canvas") : canvas;
    if (over > 1) {
        output.width = width;
        output.height = height;
    }
    const flatten = () => {
        if (over === 1) return canvas;
        const pen = output.getContext("2d");
        pen.clearRect(0, 0, width, height);
        pen.imageSmoothingEnabled = true;
        pen.imageSmoothingQuality = "high";
        pen.drawImage(canvas, 0, 0, width, height);
        return output;
    };

    const runtime = createRuntime(canvas, null);
    let renderer = null;
    let tracer = null;
    try {
        renderer = new THREE.WebGLRenderer({
            canvas,
            antialias: spec.params?.antialias !== false && !spec.deps?.effects,
            alpha: Boolean(job.transparent),
            preserveDrawingBuffer: true,
        });
        renderer.setPixelRatio(1);
        renderer.setSize(drawWidth, drawHeight, false);
        renderer.shadowMap.enabled = spec.params?.shadows !== false;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        renderer.toneMapping = toneMappingConstant(spec.params?.toneMapping);
        renderer.toneMappingExposure = Number(spec.params?.exposure) || 1;
        if (job.transparent) renderer.setClearAlpha(0);
        runtime.ctx.renderer = renderer;

        // Per-capture motion is read against the run this job covers, so the last frame
        // lands exactly on the amount that was asked for.
        runtime.ctx.timelineSeconds = Number(spec.params?.loopSeconds) || 4;
        runtime.ctx.duration = Math.max(0, times[times.length - 1] - times[0]);
        runtime.ctx.timeOrigin = times[0];

        // Read before the scene is built, since a light takes its shadow map size on the way
        // in and the surroundings are captured differently for a traced render.
        runtime.ctx.shadowMapSize = Number(spec.params?.shadowMapSize) || 2048;
        runtime.ctx.tracing = Boolean(job.trace);
        const scene = await runtime.buildScene(spec.deps?.scene);
        if (job.transparent) scene.background = null;
        const camera = runtime.buildCamera(spec.deps?.camera, width / height);
        runtime.attachCameraTrack(camera, spec.deps?.camera, scene);
        // A traced frame is presented by the tracer itself, so the effect chain is not built
        // for one.
        const composer = job.trace ? null : await runtime.createComposer(spec.deps?.effects ?? null, {
            renderer, scene, camera, width: drawWidth, height: drawHeight,
        });

        // How the picture for one frame is drawn, either at once or by accumulating samples.
        let drawPicture = null;
        const wanted = Math.max(1, Math.floor(job.trace?.samples) || 1);
        if (job.trace) {
            tracer = await createPathTracer(renderer, job.trace);
            // Only a scene that moves needs its hierarchy rebuilt each frame; one that only
            // has the camera moving is built once.
            const moves = runtime.ctx.updateFunctions.length > 0;
            let built = false;
            drawPicture = async (index) => {
                prepareFrame(tracer, scene, camera, moves || !built);
                built = true;
                await traceFrame(tracer, wanted, job.trace.patience, (standing) => {
                    onProgress?.({
                        done: index * wanted + Math.floor(standing),
                        note: `frame ${index + 1} of ${times.length}, `
                            + `${Math.floor(standing)} of ${wanted} samples`,
                    });
                });
            };
        } else {
            drawPicture = async () => {
                if (composer) composer.render();
                else renderer.render(scene, camera);
            };
        }

        // Throwaway draws first. A composer ping-pongs between two targets, and its first
        // present can come from one nothing has written, which loses the background.
        windForward(runtime, 0, times[0]);
        runtime.updateShaderUniforms(times[0], drawWidth, drawHeight);
        if (!job.trace) {
            for (let warm = 0; warm < (composer ? WARM_UP_DRAWS : 1); warm += 1) {
                if (composer) composer.render();
                else renderer.render(scene, camera);
            }
        }

        // Wound forward once through the whole run, so a frame costs only the step from the
        // frame before it however far along the run it sits.
        let standing = 0;
        for (let index = 0; index < times.length; index += 1) {
            const moment = Number(times[index]) || 0;
            windForward(runtime, standing, moment);
            standing = moment;
            runtime.updateShaderUniforms(moment, drawWidth, drawHeight);
            await drawPicture(index);
            const png = flatten().toDataURL("image/png");

            // The two data passes are drawn from the same pose, straight after it, so they
            // line up with the picture frame for frame.
            // A range given on the node wins; 0 and 0 means fit it to what is in shot.
            const asked = { near: Number(job.depthNear) || 0, far: Number(job.depthFar) || 0 };
            const bounds = asked.far > asked.near
                ? asked
                : runtime.depthBounds(scene, camera);
            runtime.renderOverridePass({ renderer, scene, camera, kind: "depth" }, bounds);
            const depth = flatten().toDataURL("image/png");
            runtime.renderOverridePass({ renderer, scene, camera, kind: "normal" }, bounds);
            const normal = flatten().toDataURL("image/png");

            await onFrame(index, {
                png, depth, normal,
                done: (index + 1) * wanted,
                note: `frame ${index + 1} of ${times.length}`,
            });
        }
    } finally {
        try {
            tracer?.dispose?.();
        } catch (error) {
            console.warn("[WAS ThreeJS] The path tracer was not fully released.", error);
        }
        try {
            runtime.dispose();
        } catch (error) {
            console.warn("[WAS ThreeJS] Render job resources were not fully released.", error);
        }
        renderer?.dispose?.();
    }
}

/**
 * This tab's ComfyUI session id, which is the one its prompts are queued under.
 *
 * @returns {string} The session id, or an empty string before the socket has connected.
 */
function sessionId() {
    return String(api?.clientId || api?.initialClientId || "");
}

/**
 * Take whatever jobs are waiting, draw them and post each one back.
 *
 * @returns {Promise<number>} How many jobs were taken.
 */
async function serveJobs() {
    let taken = [];
    try {
        const asking = `${ROUTE}?client_id=${encodeURIComponent(sessionId())}`;
        const answer = await fetch(asking, { cache: "no-store" });
        if (!answer.ok) return 0;
        taken = (await answer.json()).jobs || [];
    } catch (error) {
        return 0;
    }

    for (const job of taken) {
        const post = (payload) => fetch(ROUTE, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ token: job.token, ...payload }),
        });
        try {
            await drawJob(
                job,
                (index, pictures) => post({ index, ...pictures }),
                (progress) => post(progress),
            );
        } catch (error) {
            console.error("[WAS ThreeJS] Render job failed:", error);
            try {
                await post({ error: String(error?.message || error).slice(0, 400) });
            } catch (reportError) {
                console.error("[WAS ThreeJS] Could not report the failure:", reportError);
            }
        }
    }
    return taken.length;
}

/**
 * Poll for jobs for as long as the tab is open.
 *
 * @returns {void}
 */
function startPolling() {
    if (running) return;
    running = true;
    const tick = async () => {
        let served = 0;
        if (!document.hidden) {
            try {
                served = await serveJobs();
            } catch (error) {
                console.error("[WAS ThreeJS] Render poll failed:", error);
            }
        }
        setTimeout(tick, served ? POLL_MS : IDLE_MS);
    };
    setTimeout(tick, IDLE_MS);
}

app.registerExtension({
    name: "WASNodeSuite.ThreeJSRender",
    async setup() {
        startPolling();
    },
});
