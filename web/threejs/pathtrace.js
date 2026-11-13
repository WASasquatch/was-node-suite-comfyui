/**
 * Path-traced frames, drawn on one renderer the whole page shares.
 *
 * The tracer's shader compiles once a page, on a renderer kept for the page's life. Tiles go
 * to the GPU in batches, each sent once the last has finished.
 */

import * as THREE from "../vendor/three/three.module.js";

// The vendored build, loaded the first time a frame is traced rather than with the pack.
const TRACER_MODULE = "../vendor/pathtracer/index.module.js";

// Milliseconds from sending a batch to seeing it finished that a batch is sized to, the delay
// before the page sees a finished batch included.
export const RENDER_LATENCY_MS = 16;
export const VIEW_LATENCY_MS = 10;

// Most tiles one batch may hold.
const MAX_BATCH = 64;

// Milliseconds between looks at whether the GPU has finished a batch.
const POLL_MS = 2;

// Milliseconds a batch waits for an animation frame before going ahead without one.
const FRAME_WAIT_MS = 50;

// Milliseconds between progress reports while a frame accumulates.
const PROGRESS_MS = 250;

let pending = null;
const stations = new Map();

/**
 * Load the path tracer.
 *
 * @returns {Promise<object>} The tracer module's exports.
 */
export function loadPathTracer() {
    if (!pending) pending = import(TRACER_MODULE);
    return pending;
}

/**
 * Wait a number of milliseconds.
 *
 * @param {number} milliseconds - How long to wait.
 * @returns {Promise<void>} Settles once the time has passed.
 */
function pause(milliseconds) {
    return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

/**
 * Wait for the next animation frame, or a short timeout where none comes. A hidden page draws
 * nothing, so it does not wait at all.
 *
 * @returns {Promise<void>} Settles on the frame or the timeout, whichever is first.
 */
function nextFrame() {
    if (document.hidden) return Promise.resolve();
    return new Promise((resolve) => {
        let settled = false;
        const finish = () => {
            if (settled) return;
            settled = true;
            resolve();
        };
        requestAnimationFrame(finish);
        setTimeout(finish, FRAME_WAIT_MS);
    });
}

/**
 * The page's tracing renderer for one set of canvas attributes, made on first use and kept.
 *
 * @param {object} [attributes] - `{alpha, antialias}` for the canvas.
 * @returns {object} The station: `canvas`, off the page; `renderer`; `holder`, the render job
 *   using it alone or null; `warm`, whether a tile has been traced.
 */
export function tracingStation({ alpha = false, antialias = true } = {}) {
    const key = `${alpha ? "a" : "o"}${antialias ? "s" : "j"}`;
    const held = stations.get(key);
    if (held && !held.renderer.getContext().isContextLost()) return held;
    held?.renderer.dispose();

    const canvas = document.createElement("canvas");
    canvas.width = 1;
    canvas.height = 1;
    const renderer = new THREE.WebGLRenderer({
        canvas,
        alpha: Boolean(alpha),
        antialias: Boolean(antialias),
        preserveDrawingBuffer: true,
    });
    renderer.setPixelRatio(1);
    renderer.shadowMap.enabled = false;
    const station = { canvas, renderer, holder: null, warm: false };
    stations.set(key, station);
    return station;
}

/**
 * Whether the browser compiles a shader with the page waiting on it, as Firefox does.
 *
 * @param {THREE.WebGLRenderer} renderer - The renderer to ask about.
 * @returns {boolean} True where the first traced tile holds the page until the shader is built.
 */
export function compilesInline(renderer) {
    return !renderer.extensions.has("KHR_parallel_shader_compile");
}

/**
 * Sends work to the GPU a batch at a time and sizes each batch from how long the last took.
 */
export class TilePacer {
    /**
     * @param {THREE.WebGLRenderer} renderer - The renderer the tiles are drawn with.
     * @param {number} [budget] - Milliseconds from sending a batch to seeing it finished.
     */
    constructor(renderer, budget = RENDER_LATENCY_MS) {
        this.gl = renderer.getContext();
        this.budget = budget;
        this.size = 1;
        this.fence = null;
        this.sentAt = 0;
    }

    /**
     * Tiles the next batch may hold.
     *
     * @returns {number} At least one.
     */
    get batch() {
        return Math.max(1, Math.floor(this.size));
    }

    /**
     * Whether the GPU has finished every batch sent.
     *
     * @returns {boolean} True when another batch may go.
     */
    get idle() {
        return this.fence === null;
    }

    /**
     * Mark the end of a batch and watch for the GPU to finish it.
     *
     * @param {number} tiles - Tiles the batch traced, which the next batch is sized from.
     * @returns {Promise<void>} Settles once the GPU has finished the batch.
     */
    send(tiles) {
        const gl = this.gl;
        const fence = gl.isContextLost() ? null : gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
        gl.flush();
        if (!fence) return Promise.resolve();
        this.fence = fence;
        this.sentAt = performance.now();
        return this.watch(fence, tiles);
    }

    /**
     * Poll one fence until the GPU passes it, then resize the next batch.
     *
     * @param {WebGLSync} fence - The fence closing the batch.
     * @param {number} tiles - Tiles the batch traced.
     * @returns {Promise<void>} Settles once the fence has signalled or been dropped.
     */
    async watch(fence, tiles) {
        const gl = this.gl;
        let hidden = document.hidden;
        while (this.fence === fence) {
            await pause(POLL_MS);
            if (this.fence !== fence) return;
            hidden = hidden || document.hidden;
            const lost = gl.isContextLost();
            if (!lost && gl.getSyncParameter(fence, gl.SYNC_STATUS) !== gl.SIGNALED) continue;
            if (!lost) gl.deleteSync(fence);
            this.fence = null;
            // Timings taken while the page was hidden are not used.
            if (tiles > 0 && !hidden) {
                const took = Math.max(0.5, performance.now() - this.sentAt);
                const grown = Math.max(this.size + 1, this.size * 1.5);
                const shrunk = this.size * (this.budget / took);
                this.size = Math.min(MAX_BATCH, Math.max(1, took <= this.budget ? grown : shrunk));
            }
        }
    }

    /**
     * Drop the fence still being watched.
     *
     * @returns {void}
     */
    dispose() {
        const fence = this.fence;
        this.fence = null;
        if (fence && !this.gl.isContextLost()) this.gl.deleteSync(fence);
    }
}

/**
 * Make a tracer on a station. A shader the station has already compiled is not compiled again.
 *
 * @param {object} station - The station from `tracingStation`.
 * @param {object} settings - `{bounces, transmissiveBounces, filterGlossy, tiles, textureSize,
 *   multipleImportanceSampling}`, each optional.
 * @returns {Promise<object>} The tracer, to free with `disposeTracer`.
 */
export async function createTracer(station, settings = {}) {
    const { WebGLPathTracer } = await loadPathTracer();
    const tracer = new WebGLPathTracer(station.renderer);
    // Preview behaviour off: no delay before the first sample, no fade, no sample floor, and
    // no ordinary draw in place of a frame short of its samples.
    tracer.renderDelay = 0;
    tracer.fadeDuration = 0;
    tracer.minSamples = 0;
    tracer.dynamicLowRes = false;
    tracer.renderScale = 1;
    tracer.synchronizeRenderSize = true;
    tracer.rasterizeScene = false;
    tracer.renderToCanvas = true;
    // Each pass takes its sample pattern from a seeded sequence restarted with the picture.
    tracer.stableNoise = true;

    tracer.bounces = Math.max(1, Math.floor(settings.bounces ?? 5));
    tracer.transmissiveBounces = Math.max(0, Math.floor(settings.transmissiveBounces ?? 10));
    tracer.filterGlossyFactor = Math.max(0, Number(settings.filterGlossy) || 0);
    tracer.multipleImportanceSampling = settings.multipleImportanceSampling !== false;

    const tiles = Math.max(1, Math.floor(settings.tiles ?? 3));
    tracer.tiles.set(tiles, tiles);

    const texture = Math.max(16, Math.floor(settings.textureSize ?? 1024));
    tracer.textureSize.set(texture, texture);
    return tracer;
}

/**
 * Free a tracer's targets and the scene data its material holds, keeping the compiled shader.
 *
 * @param {object} tracer - The tracer, or null.
 * @returns {void}
 */
export function disposeTracer(tracer) {
    if (!tracer) return;
    const material = tracer._pathTracer?.material;
    tracer.dispose();
    tracer._lowResPathTracer?.dispose?.();
    // Frees what the material holds; the material itself stays, keeping its program. A struct
    // with no dispose of its own has its textures freed one by one.
    for (const uniform of Object.values(material?.uniforms ?? {})) {
        const value = uniform.value;
        try {
            if (typeof value?.dispose === "function") value.dispose();
            else if (value && typeof value === "object") {
                for (const part of Object.values(value)) if (part?.isTexture) part.dispose();
            }
        } catch (error) {
            console.warn("[WAS ThreeJS] A path tracer resource was not released.", error);
        }
    }
}

/**
 * Hand the tracer the pose to trace, and clear what it had accumulated.
 *
 * @param {object} tracer - The tracer.
 * @param {THREE.Scene} scene - The scene at this moment.
 * @param {THREE.Camera} camera - The camera at this moment.
 * @param {boolean} rebuild - Whether the geometry moved and its hierarchy needs rebuilding.
 * @returns {void}
 */
export function prepareFrame(tracer, scene, camera, rebuild) {
    if (rebuild) {
        tracer.setScene(scene, camera);
        return;
    }
    tracer.setCamera(camera);
}

/**
 * Whether the tracer holds a whole pass at the sample count asked for.
 *
 * @param {object} tracer - The tracer.
 * @param {number} target - Samples per pixel to reach.
 * @returns {boolean} True once the frame is finished.
 */
function reached(tracer, target) {
    // A count is fractional part way through a pass; a frame runs on to the end of its pass.
    return tracer.samples >= target && Number.isInteger(tracer.samples);
}

/**
 * Trace one batch of tiles, present the result and send the batch to the GPU.
 *
 * @param {object} tracer - The tracer, already given a pose.
 * @param {TilePacer} pacer - The pacer sizing and closing the batch.
 * @param {number} target - Samples per pixel to stop at.
 * @returns {{drawn: number, done: Promise<void>}} Tiles traced, and a promise that settles
 *   once the GPU has finished them.
 */
export function traceBatch(tracer, pacer, target) {
    let drawn = 0;
    tracer.renderToCanvas = false;
    try {
        while (drawn < pacer.batch && !reached(tracer, target)) {
            const before = tracer.samples;
            tracer.renderSample();
            // No sample lands while the shader compiles.
            if (tracer.samples === before) break;
            drawn += 1;
        }
    } finally {
        tracer.renderToCanvas = true;
    }
    presentTrace(tracer);
    return { drawn, done: pacer.send(drawn) };
}

/**
 * Draw what the tracer has accumulated onto its renderer's canvas, tracing nothing.
 *
 * @param {object} tracer - The tracer.
 * @returns {void}
 */
export function presentTrace(tracer) {
    tracer.pausePathTracing = true;
    try {
        tracer.renderSample();
    } finally {
        tracer.pausePathTracing = false;
    }
}

// The edge-preserving filter each renderer draws a finished frame through, made on first use.
const denoisers = new WeakMap();

/**
 * A function that redraws a finished traced frame through the tracer's own denoise filter.
 *
 * @param {THREE.WebGLRenderer} renderer - The renderer whose canvas the frames are read from.
 * @returns {Promise<Function>} `(tracer, strength) => void`; a strength of 0 draws nothing.
 */
export async function createDenoiser(renderer) {
    let quad = denoisers.get(renderer);
    if (!quad) {
        const { DenoiseMaterial } = await loadPathTracer();
        const { FullScreenQuad } = await import("../vendor/three/postprocessing/Pass.js");
        quad = new FullScreenQuad(new DenoiseMaterial());
        denoisers.set(renderer, quad);
    }
    return (tracer, strength) => {
        const sigma = Number(strength) || 0;
        const texture = tracer?.target?.texture;
        if (sigma <= 0 || !texture) return;
        const material = quad.material;
        material.map = texture;
        material.sigma = sigma;
        material.kSigma = 1.0;
        material.threshold = 0.03 + 0.01 * sigma;
        material.opacity = 1;
        const held = renderer.getRenderTarget();
        renderer.setRenderTarget(null);
        quad.render(renderer);
        renderer.setRenderTarget(held);
    };
}

/**
 * Accumulate one frame.
 *
 * @param {object} tracer - The tracer, already given a pose.
 * @param {TilePacer} pacer - The pacer the tiles are sent through.
 * @param {number} samples - Samples per pixel to reach.
 * @param {number} patience - Milliseconds the frame is given.
 * @param {Function} [onSample] - Called with the samples standing, as they accumulate.
 * @param {Function} [beforeBatch] - Called just before each batch, to set the renderer up.
 * @param {Function} [stopped] - Answers true once the frame is no longer wanted, which ends
 *   it after the batch in hand.
 * @returns {Promise<number>} How many tiles were traced.
 *
 * @throws {Error} The frame did not reach its sample count within its patience.
 */
export async function traceFrame(tracer, pacer, samples, patience, onSample, beforeBatch, stopped) {
    const target = Math.max(1, Math.floor(samples));
    const deadline = performance.now() + Math.max(1000, patience);
    let drawn = 0;
    let reported = performance.now();

    while (!reached(tracer, target)) {
        if (stopped?.()) break;
        if (performance.now() > deadline) {
            throw new Error(
                `Path tracing reached ${Math.floor(tracer.samples)} of ${target} samples per `
                + `pixel before the ${Math.round(patience / 1000)}s timeout ran out. Raise `
                + `timeout, or lower samples, bounces, width and height.`
            );
        }
        beforeBatch?.();
        const batch = traceBatch(tracer, pacer, target);
        await batch.done;
        drawn += batch.drawn;
        if (onSample && performance.now() - reported >= PROGRESS_MS) {
            reported = performance.now();
            onSample(tracer.samples);
        }
        // The page draws between batches.
        await nextFrame();
    }
    return drawn;
}
