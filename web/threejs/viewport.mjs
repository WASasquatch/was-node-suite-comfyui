/**
 * The viewport tools on Three Viewer: picking, a transform gizmo, view presets, framing, a grid,
 * guides, a render gate and the saved view. The `view` widget holds the view as JSON.
 */

import { app } from "../../../scripts/app.js";
import * as THREE from "../vendor/three/three.module.mjs";
import { applyObjectOverrides, pathToObject, wrapForOverride } from "./runtime.mjs";
import { VIEW_WIDGET, setPressed } from "./viewport_bar.mjs";

// The nodes a render gate is read from, by their width and height widgets.
const RENDER_NODES = ["WASThreeRender", "WASThreePathTraceRender"];

// Directions a view preset looks from, toward the camera's target.
const VIEWS = {
    front: [0, 0, 1],
    back: [0, 0, -1],
    left: [-1, 0, 0],
    right: [1, 0, 0],
    top: [0, 1, 1e-4],
};

// Pointer travel, in CSS pixels, under which a press and release count as a click.
const CLICK_SLOP = 4;

// Digits a saved coordinate keeps.
const DIGITS = 5;

// Milliseconds after the last change before the view is written to its widget.
const WRITE_DELAY_MS = 250;

const MODE_KEYS = { q: "orbit", w: "move", e: "rotate", r: "scale" };

/**
 * Round a list of numbers to the digits a saved view keeps.
 *
 * @param {number[]} values - The numbers.
 * @returns {number[]} The rounded numbers.
 */
function rounded(values) {
    const scale = 10 ** DIGITS;
    return values.map((value) => Math.round(value * scale) / scale);
}

/**
 * Whether an object, or anything above it, is a viewport helper rather than scene content.
 *
 * @param {THREE.Object3D} object - The object.
 * @returns {boolean} True for a helper.
 */
function isHelper(object) {
    for (let node = object; node; node = node.parent) if (node.userData?.wasHelper) return true;
    return false;
}

/**
 * The object a click picks: the innermost ancestor a node built, wrappers skipped.
 *
 * @param {THREE.Object3D} hit - The mesh the ray met.
 * @param {THREE.Scene} scene - The scene.
 * @returns {THREE.Object3D|null} The picked object.
 */
function pickable(hit, scene) {
    for (let node = hit; node && node !== scene; node = node.parent) {
        if (node.userData?.wasSpecId && !node.userData.wasOverride && !node.userData.wasOverrideInner) return node;
    }
    return null;
}

/**
 * The picked object's next ancestor a node built, for picking the parent.
 *
 * @param {THREE.Object3D} object - The picked object.
 * @param {THREE.Scene} scene - The scene.
 * @returns {THREE.Object3D|null} The parent object, or null at the top.
 */
function pickableParent(object, scene) {
    return object?.parent ? pickable(object.parent, scene) : null;
}

/** The viewport tools bound to one viewer. */
export class Viewport {
    /**
     * @param {object} controller - The viewer's controller.
     * @param {object} bar - What `createViewportBar` built.
     */
    constructor(controller, bar) {
        this.controller = controller;
        this.bar = bar;
        this.mode = "orbit";
        this.selected = null;
        this.box = null;
        this.gizmo = null;
        this.helpers = null;
        this.gridOn = false;
        this.guidesOn = false;
        this.gateOn = true;
        this.writeTimer = null;
        this.listeners = [];
        this.lastSize = "";

        const { buttons } = bar;
        for (const mode of ["orbit", "select", "move", "rotate", "scale"]) {
            buttons[mode].addEventListener("click", () => this.setMode(mode));
        }
        buttons.frame.addEventListener("click", () => this.frame());
        buttons.grid.addEventListener("click", () => this.toggleGrid());
        buttons.guides.addEventListener("click", () => {
            this.guidesOn = !this.guidesOn;
            this.lastSize = "";
            this.refreshButtons();
        });
        buttons.gate.addEventListener("click", () => {
            this.gateOn = !this.gateOn;
            this.lastSize = "";
            this.refreshButtons();
        });
        buttons.override.addEventListener("click", () => this.setOverride(!this.overrideOn()));
        buttons.capture.addEventListener("click", () => {
            this.writeView();
            if (!this.overrideOn()) this.setOverride(true, false);
            this.note("View kept. Override is on, so Three Render draws this framing.");
        });
        buttons.clear.addEventListener("click", () => this.clearView());
        bar.viewSelect.addEventListener("change", () => {
            const value = bar.viewSelect.value;
            bar.viewSelect.value = "";
            if (value === "camera") this.controller.resetCamera();
            else if (value) this.lookFrom(value);
            this.changed();
        });
        this.refreshButtons();
    }

    /** The node's widget by name. */
    widget(name) {
        return this.controller.node?.widgets?.find((widget) => widget.name === name) ?? null;
    }

    /** Whether override is on. */
    overrideOn() {
        return Boolean(this.widget("override")?.value);
    }

    /** The saved view, parsed, or null. */
    savedView() {
        const text = String(this.widget(VIEW_WIDGET)?.value || "").trim();
        if (!text) return null;
        try {
            return JSON.parse(text);
        } catch (error) {
            console.warn("[WAS ThreeJS] The saved view could not be read.", error);
            return null;
        }
    }

    /** Show a short line on the viewer's status. */
    note(text) {
        this.controller.elements.status.textContent = text;
    }

    /**
     * Bind to a freshly built scene: apply the saved view where override is on, and arm the tools.
     *
     * @returns {Promise<void>}
     */
    async attach() {
        const { scene, camera, controls, elements } = this.controller;
        if (!scene || !camera) return;

        const view = this.overrideOn() ? this.savedView() : null;
        if (view?.objects?.length) applyObjectOverrides(scene, view.objects);
        if (view?.camera) {
            camera.position.fromArray(view.camera.position);
            const target = new THREE.Vector3().fromArray(view.camera.target);
            if (camera.isPerspectiveCamera && Number.isFinite(view.camera.fov)) camera.fov = view.camera.fov;
            camera.lookAt(target);
            camera.updateProjectionMatrix();
            camera.userData.comfyTarget = target.clone();
            if (controls) {
                controls.target.copy(target);
                controls.sync?.();
                controls.update();
            }
        }
        if (camera.isPerspectiveCamera) camera.userData.wasFov = camera.fov;

        const { TransformControls } = await import("../vendor/three/controls/TransformControls.mjs");
        if (this.controller.scene !== scene) return;
        this.gizmo = new TransformControls(camera, elements.canvas);
        this.gizmo.setSize(0.9);
        const handle = this.gizmo.getHelper();
        handle.userData.wasHelper = true;
        scene.add(handle);
        this.gizmo.addEventListener("dragging-changed", (event) => {
            if (this.controller.controls) this.controller.controls.enabled = !event.value;
            if (!event.value) {
                if (!this.overrideOn()) this.setOverride(true, false);
                this.changed();
            }
        });

        const helpers = new THREE.Group();
        helpers.userData.wasHelper = true;
        helpers.add(new THREE.GridHelper(10, 20, 0x8a8a8a, 0x4a4a4a), new THREE.AxesHelper(1.5));
        helpers.visible = this.gridOn;
        scene.add(helpers);
        this.helpers = helpers;

        if (!elements.canvasWrap.contains(this.bar.overlay)) elements.canvasWrap.append(this.bar.overlay);
        this.listen(elements.canvas, "pointerdown", (event) => this.onDown(event), true);
        this.listen(elements.canvas, "pointerup", (event) => this.onUp(event));
        this.listen(elements.canvas, "wheel", () => this.changed());
        this.listen(elements.root, "keydown", (event) => this.onKey(event));
        this.listen(elements.root, "pointerenter", () => elements.root.focus({ preventScroll: true }));
        elements.root.tabIndex = 0;
        elements.root.style.outline = "none";

        this.watchOverride();
        this.setMode(this.mode);
        this.lastSize = "";
    }

    /** Add an event listener that `detach` removes. */
    listen(target, type, handler, capture = false) {
        target.addEventListener(type, handler, capture);
        this.listeners.push(() => target.removeEventListener(type, handler, capture));
    }

    /** Re-render the scene when the override widget is flipped on the node itself. */
    watchOverride() {
        const widget = this.widget("override");
        if (!widget || widget.__wasViewportWatched) return;
        widget.__wasViewportWatched = true;
        const original = widget.callback;
        widget.callback = (...args) => {
            const result = original?.apply(widget, args);
            const viewport = this.controller.viewport;
            viewport?.refreshButtons();
            viewport?.rebuild();
            return result;
        };
    }

    /** Draw the last run's scene again, so a change of override shows at once. */
    rebuild() {
        const payload = this.controller.lastPayload;
        if (payload) this.controller.renderPayload(payload);
    }

    /**
     * Turn override on or off, on the node and on screen.
     *
     * @param {boolean} on - The new state.
     * @param {boolean} [redraw] - Build the scene again, so the saved view is applied or dropped.
     * @returns {void}
     */
    setOverride(on, redraw = true) {
        const widget = this.widget("override");
        if (!widget) return;
        widget.value = Boolean(on);
        this.controller.node?.setDirtyCanvas?.(true, true);
        app.graph?.change?.();
        this.refreshButtons();
        if (redraw) this.rebuild();
    }

    /** Forget the saved view and every move, turn override off and draw the upstream scene. */
    clearView() {
        const view = this.widget(VIEW_WIDGET);
        if (view) view.value = "";
        this.setOverride(false, true);
        this.note("View cleared. The camera and objects follow the nodes upstream.");
    }

    /** Pick a tool. */
    setMode(mode) {
        this.mode = mode;
        if (this.gizmo) {
            if (["move", "rotate", "scale"].includes(mode)) {
                this.gizmo.setMode({ move: "translate", rotate: "rotate", scale: "scale" }[mode]);
                if (this.selected) this.attachGizmo(this.selected);
            } else {
                this.gizmo.detach();
            }
        }
        this.refreshButtons();
    }

    /** Light the buttons that are on. */
    refreshButtons() {
        const { buttons } = this.bar;
        for (const mode of ["orbit", "select", "move", "rotate", "scale"]) setPressed(buttons[mode], this.mode === mode);
        setPressed(buttons.grid, this.gridOn);
        setPressed(buttons.guides, this.guidesOn);
        setPressed(buttons.gate, this.gateOn);
        setPressed(buttons.override, this.overrideOn());
    }

    toggleGrid() {
        this.gridOn = !this.gridOn;
        if (this.helpers) this.helpers.visible = this.gridOn;
        this.refreshButtons();
    }

    onDown(event) {
        this.down = { x: event.clientX, y: event.clientY };
        // A press on the gizmo moves the object, never the camera.
        if (this.controller.controls) this.controller.controls.enabled = !this.gizmo?.axis;
    }

    onUp(event) {
        const start = this.down;
        this.down = null;
        if (this.controller.controls && !this.gizmo?.dragging) this.controller.controls.enabled = true;
        if (!start) return;
        const travel = Math.hypot(event.clientX - start.x, event.clientY - start.y);
        if (travel > CLICK_SLOP) {
            if (this.overrideOn()) this.changed();
            return;
        }
        if (this.mode === "orbit" || this.gizmo?.axis) return;
        this.pickAt(event.clientX, event.clientY);
    }

    onKey(event) {
        if (event.target?.closest?.("input, textarea, select")) return;
        const key = event.key.toLowerCase();
        if (MODE_KEYS[key]) this.setMode(MODE_KEYS[key]);
        else if (key === "f") this.frame();
        else if (key === "g") this.toggleGrid();
        else if (key === "p") this.select(pickableParent(this.selected, this.controller.scene));
        else if (key === "escape") this.select(null);
        else return;
        event.preventDefault();
        event.stopPropagation();
    }

    /** Pick what lies under a point on the canvas. */
    pickAt(clientX, clientY) {
        const { scene, camera, elements } = this.controller;
        const box = elements.canvas.getBoundingClientRect();
        const pointer = new THREE.Vector2(
            ((clientX - box.left) / box.width) * 2 - 1,
            -((clientY - box.top) / box.height) * 2 + 1,
        );
        const ray = new THREE.Raycaster();
        ray.setFromCamera(pointer, camera);
        const hit = ray.intersectObjects(scene.children, true).find((entry) => !isHelper(entry.object));
        this.select(hit ? pickable(hit.object, scene) : null);
    }

    /** Make an object the picked one. */
    select(object) {
        const { scene } = this.controller;
        this.selected = object || null;
        if (this.box) {
            scene?.remove(this.box);
            this.box.dispose?.();
            this.box = null;
        }
        this.gizmo?.detach();
        if (!this.selected) {
            this.note("Nothing picked.");
            return;
        }
        this.box = new THREE.BoxHelper(this.selected, 0xffb02e);
        this.box.userData.wasHelper = true;
        scene.add(this.box);
        if (["move", "rotate", "scale"].includes(this.mode)) this.attachGizmo(this.selected);
        this.note(`Picked ${this.selected.name || this.selected.type}. P picks its parent, Esc clears.`);
    }

    /** Hang the gizmo on an object's override wrapper, making the wrapper where there is none. */
    attachGizmo(object) {
        if (!this.gizmo || !object?.parent) return;
        const middle = new THREE.Box3().setFromObject(object).getCenter(new THREE.Vector3());
        const pivot = object.parent.worldToLocal(middle).toArray();
        const wrapper = wrapForOverride(object, pivot);
        this.gizmo.attach(wrapper);
    }

    /** Look from a preset direction, keeping the distance to the target. */
    lookFrom(name) {
        const { camera, controls } = this.controller;
        const target = controls?.target ?? camera.userData.comfyTarget ?? new THREE.Vector3();
        const distance = camera.position.distanceTo(target) || 5;
        const direction = new THREE.Vector3().fromArray(VIEWS[name]).normalize();
        camera.position.copy(target).addScaledVector(direction, distance);
        camera.lookAt(target);
        controls?.sync?.();
        controls?.update?.();
    }

    /** Fit the picked object, or the whole scene, in view. */
    frame() {
        const { scene, camera, controls } = this.controller;
        const box = new THREE.Box3();
        if (this.selected) box.setFromObject(this.selected);
        else scene.traverse((node) => {
            if (node.isMesh && !isHelper(node) && node.geometry) box.expandByObject(node);
        });
        if (box.isEmpty()) return;
        const sphere = box.getBoundingSphere(new THREE.Sphere());
        const fov = THREE.MathUtils.degToRad(camera.userData.wasFov ?? camera.fov ?? 50);
        const distance = sphere.radius / Math.sin(fov / 2) * 1.08;
        const target = controls?.target ?? new THREE.Vector3();
        const direction = camera.position.clone().sub(target).normalize();
        target.copy(sphere.center);
        camera.position.copy(sphere.center).addScaledVector(direction, distance);
        camera.lookAt(sphere.center);
        controls?.sync?.();
        controls?.update?.();
        this.changed();
    }

    /** Note that the view moved; it is written once things settle, where override is on. */
    changed() {
        if (!this.overrideOn()) return;
        clearTimeout(this.writeTimer);
        this.writeTimer = setTimeout(() => this.writeView(), WRITE_DELAY_MS);
    }

    /** Write the camera and every moved object to the node's view widget. */
    writeView() {
        const { scene, camera, controls, node } = this.controller;
        const widget = this.widget(VIEW_WIDGET);
        if (!scene || !camera || !widget) return;
        const target = controls?.target ?? camera.userData.comfyTarget ?? new THREE.Vector3();
        const saved = {
            camera: { position: rounded(camera.position.toArray()), target: rounded(target.toArray()) },
            objects: [],
        };
        if (camera.isPerspectiveCamera) saved.camera.fov = rounded([camera.userData.wasFov ?? camera.fov])[0];
        if (camera.isOrthographicCamera) saved.camera.height = rounded([(camera.top - camera.bottom) / (camera.zoom || 1)])[0];
        scene.traverse((wrapper) => {
            if (!wrapper.userData.wasOverride) return;
            const pivot = wrapper.userData.pivot;
            const still = wrapper.position.distanceTo(new THREE.Vector3().fromArray(pivot)) < 1e-6
                && Math.abs(wrapper.quaternion.w) > 1 - 1e-9
                && wrapper.scale.distanceTo(new THREE.Vector3(1, 1, 1)) < 1e-9;
            if (still) return;
            const inner = wrapper.children[0]?.children[0];
            const path = inner ? pathToObject(scene, inner) : null;
            if (!path) return;
            saved.objects.push({
                path,
                pivot: rounded(pivot),
                position: rounded(wrapper.position.toArray()),
                quaternion: rounded(wrapper.quaternion.toArray()),
                scale: rounded(wrapper.scale.toArray()),
            });
        });
        widget.value = JSON.stringify(saved);
        node?.setDirtyCanvas?.(true, true);
        app.graph?.change?.();
    }

    /** The width and height of the render this viewer feeds, or null. */
    renderSize() {
        const node = this.controller.node;
        for (const output of node?.outputs ?? []) {
            for (const linkId of output.links ?? []) {
                const link = node.graph?.links?.[linkId] ?? node.graph?.links?.get?.(linkId);
                const target = link ? node.graph.getNodeById(link.target_id) : null;
                if (!target || !RENDER_NODES.includes(target.comfyClass ?? target.type)) continue;
                const width = Number(target.widgets?.find((widget) => widget.name === "width")?.value);
                const height = Number(target.widgets?.find((widget) => widget.name === "height")?.value);
                if (width > 0 && height > 0) return [width, height];
            }
        }
        return null;
    }

    /**
     * Run before each drawn frame: keep the gate and its field of view in step with the canvas.
     *
     * @returns {void}
     */
    beforeRender() {
        this.box?.update?.();
        const { camera, elements } = this.controller;
        if (!camera) return;
        const rect = elements.canvasWrap.getBoundingClientRect();
        const size = this.gateOn ? this.renderSize() : null;
        const key = `${rect.width}x${rect.height}:${size?.join("x")}:${this.gateOn}:${this.guidesOn}:${camera.userData.wasFov}`;
        if (key === this.lastSize) return;
        this.lastSize = key;

        const { gate, thirds } = this.bar;
        const canvasAspect = rect.width / Math.max(1, rect.height);
        let width = rect.width;
        let height = rect.height;
        if (camera.isPerspectiveCamera) camera.fov = camera.userData.wasFov ?? camera.fov;
        if (size) {
            const renderAspect = size[0] / size[1];
            if (renderAspect <= canvasAspect) {
                width = rect.height * renderAspect;
            } else {
                // The render is wider than the canvas, so the view widens until it fits.
                height = rect.width / renderAspect;
                if (camera.isPerspectiveCamera) {
                    const half = THREE.MathUtils.degToRad(camera.fov) / 2;
                    camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(Math.tan(half) * rect.height / height));
                }
            }
        }
        camera.updateProjectionMatrix?.();
        const framed = Boolean(size) || this.guidesOn;
        gate.style.display = framed ? "block" : "none";
        gate.style.width = `${width}px`;
        gate.style.height = `${height}px`;
        gate.style.left = `${(rect.width - width) / 2}px`;
        gate.style.top = `${(rect.height - height) / 2}px`;
        gate.style.boxShadow = size ? "0 0 0 9999px rgba(0,0,0,0.45)" : "none";
        gate.style.border = size ? "1px solid rgba(255,255,255,0.55)" : "none";
        for (const line of thirds) line.style.display = this.guidesOn ? "block" : "none";
    }

    /** Let go of the scene, keeping the toolbar and its state for the next one. */
    detach() {
        clearTimeout(this.writeTimer);
        for (const remove of this.listeners.splice(0)) remove();
        try {
            this.gizmo?.detach();
            this.gizmo?.dispose?.();
        } catch (error) {
            console.warn("[WAS ThreeJS] The gizmo was not released.", error);
        }
        this.gizmo = null;
        this.box = null;
        this.helpers = null;
        this.selected = null;
        this.lastSize = "";
    }
}
