// Run with: node --experimental-vm-modules --test test/was_pause.test.mjs
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import { createContext, SourceTextModule, SyntheticModule } from "node:vm";

async function setup() {
  const nodes = new Map();
  const listeners = new Map();
  const requests = [];
  const errors = [];
  let extension;
  const app = {
    graph: { getNodeById: (id) => nodes.get(Number(id)) },
    registerExtension: (value) => { extension = value; },
  };
  const api = {
    addEventListener: (name, listener) => listeners.set(name, listener),
    async fetchApi(route, options) {
      requests.push({ route, method: options.method, body: JSON.parse(options.body) });
      return { ok: true };
    },
  };
  const context = createContext({
    AbortController, DOMException, setTimeout, clearTimeout,
    console: { error: (...args) => errors.push(args) },
  });
  const appModule = new SyntheticModule(["app"], function () {
    this.setExport("app", app);
  }, { context });
  const apiModule = new SyntheticModule(["api"], function () {
    this.setExport("api", api);
  }, { context });
  const modules = new Map();
  async function load(url) {
    if (!modules.has(url.href)) {
      modules.set(url.href, new SourceTextModule(await readFile(url, "utf8"), {
        context, identifier: url.href,
      }));
    }
    return modules.get(url.href);
  }
  const module = await load(new URL("../web/was_pause.js", import.meta.url));
  await module.link((specifier, importer) => {
    if (specifier.endsWith("/scripts/app.js")) return appModule;
    if (specifier.endsWith("/scripts/api.js")) return apiModule;
    return load(new URL(specifier, importer.identifier));
  });
  await module.evaluate();
  await extension.setup();
  class Node {
    constructor(id) {
      this.id = id;
      this.widgets = [{ name: "unrelated", disabled: false }];
    }
    setDirtyCanvas() {}
  }
  await extension.beforeRegisterNodeDef(Node, { name: "WASPause" });
  return {
    api, requests, errors,
    emit: (name, node_id) => listeners.get(name)({ detail: { node_id } }),
    create(id) {
      const node = new Node(id);
      // ComfyUI can create widgets before the node enters the graph.
      node.onNodeCreated();
      nodes.set(id, node);
      return node;
    },
  };
}

function states(node) {
  return node.widgets.map((widget) => [widget.name, widget.disabled]);
}

function expected(disabled) {
  return [
    ["unrelated", false],
    ["was_pause_resume", disabled],
    ["was_pause_cancel", disabled],
  ];
}

function adoptWidgets(node) {
  // Model the frontend boundary: adoption replaces the property with stored state.
  // These tests do not load ComfyUI's renderer or its widget store.
  for (const widget of node.widgets) {
    let disabled = widget.disabled;
    Object.defineProperty(widget, "disabled", {
      configurable: true,
      get: () => disabled,
      set: (value) => { disabled = value; },
    });
  }
  return node;
}

for (const [kind, prepare] of [["plain", (node) => node], ["store-backed", adoptWidgets]]) {
  test(`both buttons follow pause lifecycle with ${kind} properties`, async () => {
    const { create, emit } = await setup();
    const first = prepare(create(7));
    const second = prepare(create(23));
    assert.deepEqual(states(first), expected(true));
    emit("was-pause", "7");
    assert.deepEqual(states(first), expected(false));
    assert.deepEqual(states(second), expected(true));
    emit("was-pause-done", 7);
    assert.deepEqual(states(first), expected(true));
    emit("was-pause", 7);
    emit("was-pause", "23");
    assert.deepEqual(states(first), expected(false));
    assert.deepEqual(states(second), expected(false));
    emit("execution_start");
    assert.deepEqual(states(first), expected(true));
    assert.deepEqual(states(second), expected(true));
    emit("was-pause", "7");
    assert.deepEqual(states(first), expected(false));
  });
}

test("adoption during a pause retains the state and receives the done event", async () => {
  const { create, emit } = await setup();
  const node = create(29);
  emit("was-pause", "29");
  adoptWidgets(node);
  assert.deepEqual(states(node), expected(false));
  emit("was-pause-done", "29");
  assert.deepEqual(states(node), expected(true));
});

test("a node created after its pause event starts enabled", async () => {
  const { create, emit } = await setup();
  emit("was-pause", "31");
  const node = create(31);
  assert.deepEqual(states(node), expected(false));
  emit("was-pause-done", "31");
  assert.deepEqual(states(node), expected(true));
});

for (const [name, action] of [["was_pause_resume", "resume"], ["was_pause_cancel", "cancel"]]) {
  test(`${action} posts only while paused and waits for the done event`, async () => {
    const { create, emit, requests, errors } = await setup();
    const node = adoptWidgets(create(17));
    const button = node.widgets.find((widget) => widget.name === name);
    button.callback();
    assert.deepEqual(requests, []);
    emit("was-pause", "17");
    button.callback();
    await new Promise(setImmediate);
    assert.deepEqual(requests, [{
      route: "/was/interface/api/pause", method: "POST",
      body: { node_id: "17", action },
    }]);
    assert.deepEqual(states(node), expected(false));
    assert.equal(node.__was_viewer_held, true);
    emit("was-pause-done", "17");
    button.callback();
    assert.equal(requests.length, 1);
    assert.deepEqual(states(node), expected(true));
    assert.deepEqual(errors, []);
  });
}

test("a failed request leaves both buttons enabled for retry", async () => {
  const { create, emit, api, errors } = await setup();
  const node = create(19);
  api.fetchApi = async () => { throw new Error("offline"); };
  emit("was-pause", "19");
  node.widgets.find((widget) => widget.name === "was_pause_resume").callback();
  await new Promise(setImmediate);
  assert.equal(errors.length, 1);
  assert.deepEqual(states(node), expected(false));
  assert.equal(node.__was_viewer_held, true);
});
