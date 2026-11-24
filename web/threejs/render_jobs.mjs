/**
 * Frames drawn off screen for a Three Render node that is waiting on them.
 *
 * Polls for a filed job and posts each frame back. The drawing, and three.js with it, loads with
 * the first job.
 */

import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const ROUTE = "/was/threejs/api/render";

// How often the browser looks for a job, in milliseconds. A node files one and then waits, so
// this is the delay before its render starts.
const POLL_MS = 250;

// Backed off to this while nothing is queued, so an idle tab is not polling four times a second.
const IDLE_MS = 1500;

let running = false;
let drawModule = null;

/**
 * The drawing module, imported with the first job.
 *
 * @returns {Promise<object>} The module, exporting `drawJob` and `holdViewers`.
 */
function loadDraw() {
    drawModule ??= import("./render_draw.mjs");
    return drawModule;
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
    // Live viewers pause while a job draws. With no job yet taken there is nothing to wake.
    if (taken.length || drawModule) (await loadDraw()).holdViewers(taken.length > 0);
    if (!taken.length) return 0;
    const { drawJob } = await loadDraw();

    for (const job of taken) {
        const post = async (payload) => {
            const answer = await fetch(ROUTE, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ token: job.token, ...payload }),
            });
            // 404 means no node waits under the token any longer, as after a cancelled run,
            // and the job stops drawing.
            if (answer.status === 404) job.dropped = true;
            return answer;
        };
        try {
            await drawJob(
                job,
                (index, pictures) => post({ index, ...pictures }),
                (progress) => post(progress),
            );
            if (job.dropped) console.info("[WAS ThreeJS] Render job stopped: its run was cancelled.");
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
