/**
 * The Prompt Timeline's language model jobs: writing a video, rewriting a scene, planning
 * transitions.
 *
 * Each job is a prompt of its own on the model wired into `vlm_clip`. Settings live on the
 * conditioning node's properties.
 */

import { api } from "../../../scripts/api.js";
import { app } from "../../../scripts/app.js";

const LOG_NAME = "WASNodeSuite.H3Writer";

// The node property holding the language model settings.
export const LLM_PROPERTY = "was_h3_llm";

// Every job: the id its node takes in the queued prompt, its type, title and the ui key it answers on.
export const JOBS = Object.freeze({
  write: { id: "was_h3_scene_writer", type: "WASH3SceneWriter", title: "MiniMax H3 Scene Writer", key: "was_h3_scenes" },
  rewrite: { id: "was_h3_prompt_rewrite", type: "WASH3PromptRewrite", title: "MiniMax H3 Prompt Rewrite", key: "was_h3_rewrite" },
  plan: { id: "was_h3_plan_transitions", type: "WASH3PlanTransitions", title: "MiniMax H3 Plan Transitions", key: "was_h3_transitions" },
});

// The settings a node starts with. `system` null writes under the nodes' own default rules.
export const LLM_DEFAULTS = Object.freeze({
  system: null,
  temperature: 0.7,
  top_k: 64,
  top_p: 0.95,
  min_p: 0.05,
  repetition_penalty: 1.05,
  thinking: false,
  fast_decode: true,
  fixed_seed: false,
  seed: 0,
  plan: true,
});

// The largest seed the nodes take.
const MAX_SEED = 2 ** 32;

let systemDefault = null;

/**
 * Whether a value is a plain object.
 *
 * @param {*} value - Anything.
 * @returns {boolean} True for a non-null, non-array object.
 */
function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/**
 * The language model settings a node holds, defaults filled in.
 *
 * @param {object} node - The conditioning node.
 * @returns {object} Every key of `LLM_DEFAULTS`.
 */
export function llmSettings(node) {
  const saved = node?.properties?.[LLM_PROPERTY];
  return { ...LLM_DEFAULTS, ...(isRecord(saved) ? saved : {}) };
}

/**
 * Store settings on a node, merged over what it holds.
 *
 * @param {object} node - The conditioning node.
 * @param {object} patch - The settings that change.
 * @returns {object} The settings now held.
 */
export function storeSettings(node, patch) {
  node.properties ??= {};
  const next = { ...llmSettings(node), ...patch };
  node.properties[LLM_PROPERTY] = next;
  return next;
}

/**
 * The rules the writing nodes use when none are given, read from their definition.
 *
 * @returns {Promise<string>} The default system prompt, or an empty string where it cannot be read.
 */
export function defaultSystemPrompt() {
  systemDefault ??= api.fetchApi(`/object_info/${JOBS.write.type}`)
    .then((response) => response.json())
    .then((info) => String(info?.[JOBS.write.type]?.input?.optional?.system_prompt?.[1]?.default ?? ""))
    .catch((error) => {
      console.error(`[${LOG_NAME}] Failed to read the default system prompt:`, error);
      systemDefault = null;
      return "";
    });
  return systemDefault;
}

/**
 * The sampling inputs every job node takes, from a node's settings.
 *
 * @param {object} settings - From `llmSettings`.
 * @returns {object} Input name to value, with a fresh seed unless the seed is fixed.
 */
export function samplingInputs(settings) {
  return {
    temperature: Number(settings.temperature),
    top_k: Math.round(Number(settings.top_k)),
    top_p: Number(settings.top_p),
    min_p: Number(settings.min_p),
    repetition_penalty: Number(settings.repetition_penalty),
    thinking: Boolean(settings.thinking),
    fast_decode: Boolean(settings.fast_decode),
    seed: settings.fixed_seed ? Math.max(0, Math.round(Number(settings.seed) || 0)) : Math.floor(Math.random() * MAX_SEED),
  };
}

/**
 * What loads the model wired into a node's `vlm_clip`.
 *
 * @param {object} node - The conditioning node.
 * @returns {{wired: boolean, name: string, detail: string}} The loader's title and the file and
 *   type it loads, where they can be read.
 */
export function modelSource(node) {
  const socket = node?.inputs?.find((input) => input.name === "vlm_clip");
  const link = socket?.link != null ? node.graph?.links?.get?.(socket.link) ?? node.graph?.links?.[socket.link] : null;
  const origin = link ? node.graph?.getNodeById?.(link.origin_id) : null;
  if (!origin) return { wired: false, name: "", detail: "" };
  const values = (origin.widgets ?? []).map((widget) => widget.value).filter((value) => typeof value === "string");
  return { wired: true, name: origin.title || origin.type || "", detail: values.slice(0, 2).join(" · ") };
}

/**
 * Queue one job on the model wired into a node's `vlm_clip`, as a prompt of its own.
 *
 * @param {object} node - The conditioning node.
 * @param {string} kind - A key of `JOBS`.
 * @param {object} inputs - The job node's own inputs, sampling and system prompt included.
 * @param {{assets?: boolean}} [options] - `assets` wires the node's asset chain in too.
 * @returns {Promise<string>} The prompt id.
 * @throws {Error} When `vlm_clip` is not wired or the server refuses the prompt.
 */
export async function queueJob(node, kind, inputs, { assets = false } = {}) {
  const job = JOBS[kind];
  const { output, workflow } = await app.graphToPrompt();
  const own = output[String(node.id)];
  const clip = own?.inputs?.vlm_clip;
  if (!Array.isArray(clip)) throw new Error("vlm_clip is not wired");
  const chain = own.inputs.assets;
  const wanted = {};
  const visit = (id) => {
    if (wanted[id] || !output[id]) return;
    wanted[id] = output[id];
    for (const value of Object.values(output[id].inputs ?? {})) {
      if (Array.isArray(value) && value.length === 2 && output[String(value[0])]) visit(String(value[0]));
    }
  };
  visit(String(clip[0]));
  if (assets && Array.isArray(chain)) visit(String(chain[0]));
  wanted[job.id] = {
    class_type: job.type,
    inputs: { vlm_clip: clip, ...inputs, ...(assets && Array.isArray(chain) ? { assets: chain } : {}) },
    _meta: { title: job.title },
  };
  const answer = await api.queuePrompt(0, { output: wanted, workflow });
  if (answer?.node_errors && Object.keys(answer.node_errors).length) {
    throw new Error(Object.values(answer.node_errors)[0]?.errors?.[0]?.message ?? "the prompt was refused");
  }
  if (!answer?.prompt_id) throw new Error("the server gave no prompt id");
  return answer.prompt_id;
}

/**
 * What a finished job answered, read from its node's output.
 *
 * @param {string} kind - A key of `JOBS`.
 * @param {object} output - The `executed` event's output.
 * @returns {*} The writer's `{cast, header, footer, scenes}`, the rewritten text, or the planned
 *   transitions.
 * @throws {Error} When the answer is missing or empty.
 */
export function answerOf(kind, output) {
  const raw = output?.[JOBS[kind].key]?.[0];
  if (kind === "rewrite") {
    if (!String(raw ?? "").trim()) throw new Error("no prompt came back");
    return String(raw);
  }
  const parsed = JSON.parse(String(raw ?? "null"));
  if (kind === "write" && !parsed?.scenes?.length) throw new Error("no scenes came back");
  if (kind === "plan" && !Array.isArray(parsed)) throw new Error("no transitions came back");
  return parsed;
}

/**
 * What a scene shows, in a sentence or two, from its prompt.
 *
 * @param {string} prompt - A scene prompt.
 * @param {number} [most] - The most words to keep.
 * @returns {string} The prompt's summary section where it has one, else the start of its shots.
 */
export function digest(prompt, most = 60) {
  const text = String(prompt ?? "");
  const summary = /^summary:\s*([\s\S]+?)(?:\n\s*\n|\n[a-z_]+:|$)/m.exec(text);
  let body = summary ? summary[1] : text;
  if (!summary) {
    const shot = body.indexOf("[Shot 1]");
    if (shot >= 0) body = body.slice(shot + "[Shot 1]".length);
    body = body.split(/\n[a-z_]+:/)[0];
  }
  const words = body.replace(/\[[a-z +]+\]/g, "").replace(/<d>\[[^\]]*\]\s*/g, "\"").replace(/<\/d>/g, "\"")
    .split(/\s+/).filter(Boolean);
  return words.slice(0, most).join(" ") + (words.length > most ? "…" : "");
}
