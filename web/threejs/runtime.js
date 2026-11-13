/**
 * Turning Three.js descriptors into a live scene, and the camera control that drives it.
 *
 * Nothing here touches ComfyUI, so the exported page runs this same file unchanged.
 */

import * as THREE from "../vendor/three/three.module.js";

// Longest a model's own textures are waited for before the scene is drawn anyway.
const LOAD_PATIENCE_MS = 20000;

// Side of the cube the surroundings are captured into for a traced render. The map the
// tracer samples is four of these across and two down.
const ENVIRONMENT_CAPTURE = 256;

let areaLightTables = null;

/**
 * Load the lookup tables an area light is shaded from, once for the page.
 *
 * @returns {Promise<void>} Nothing. The renderer's uniforms carry the tables afterwards.
 */
function loadAreaLightTables() {
    if (!areaLightTables) {
        areaLightTables = import("../vendor/three/lights/RectAreaLightUniformsLib.js")
            .then(({ RectAreaLightUniformsLib }) => RectAreaLightUniformsLib.init());
    }
    return areaLightTables;
}

// The wrapper key and schema version every descriptor carries.
const WRAPPER_KEY = "__was_threejs__";
const SCHEMA_VERSION = 1;

// Where ComfyUI serves an asset a Three node holds.
const ASSET_ROUTE = "/was/threejs/api/asset";

// Folders an exported page keeps its pictures and models in, beside the page.
const ARCHIVE_FOLDERS = ["./textures/", "./models/"];

// What an address that may not be fetched is replaced with. It reads as empty bytes.
const REFUSED_ADDRESS = "data:,";

/**
 * The address relative ones are read against.
 *
 * @returns {string} The document's base, or the location where there is no document.
 */
function baseAddress() {
    return globalThis.document?.baseURI || globalThis.location?.href || "";
}

/**
 * Whether an address names an asset held by ComfyUI's asset route.
 *
 * @param {string} address - The address a loader asked for.
 * @returns {boolean} True for the asset route on this page's own origin.
 */
function isAssetRoute(address) {
    try {
        const base = baseAddress();
        const resolved = new URL(String(address || ""), base);
        return resolved.origin !== "null"
            && resolved.origin === new URL(base).origin
            && resolved.pathname === ASSET_ROUTE;
    } catch (error) {
        return false;
    }
}

/**
 * Whether an address may be read without the network.
 *
 * @param {string} address - The address a loader asked for.
 * @returns {boolean} True for a `data:` or `blob:` URL, the asset route, or a file inside an
 *     exported page's own folders.
 */
function isHeldAddress(address) {
    const text = String(address || "").trim();
    if (/^(data|blob):/i.test(text)) return true;
    if (isAssetRoute(text)) return true;
    try {
        const base = baseAddress();
        const resolved = new URL(text, base).href;
        return ARCHIVE_FOLDERS.some((folder) => resolved.startsWith(new URL(folder, base).href));
    } catch (error) {
        return false;
    }
}

/**
 * Whether an address is an absolute `http` or `https` URL.
 *
 * @param {string} address - The address a texture names.
 * @returns {boolean} True for an absolute web address.
 */
function isWebAddress(address) {
    try {
        const protocol = new URL(String(address || "").trim()).protocol;
        return protocol === "http:" || protocol === "https:";
    } catch (error) {
        return false;
    }
}

function sideConstant(side) {
    if (side === "back") return THREE.BackSide;
    if (side === "double") return THREE.DoubleSide;
    return THREE.FrontSide;
}

function wrapConstant(wrap) {
    if (wrap === "repeat") return THREE.RepeatWrapping;
    if (wrap === "mirrored-repeat") return THREE.MirroredRepeatWrapping;
    return THREE.ClampToEdgeWrapping;
}

function colorSpaceConstant(colorSpace) {
    if (colorSpace === "srgb") return THREE.SRGBColorSpace;
    if (colorSpace === "linear-srgb") return THREE.LinearSRGBColorSpace;
    return THREE.NoColorSpace;
}

export function toneMappingConstant(toneMapping) {
    const mapping = {
        none: THREE.NoToneMapping,
        linear: THREE.LinearToneMapping,
        reinhard: THREE.ReinhardToneMapping,
        cineon: THREE.CineonToneMapping,
        aces: THREE.ACESFilmicToneMapping,
        agx: THREE.AgXToneMapping,
        neutral: THREE.NeutralToneMapping,
    };
    return mapping[toneMapping] ?? THREE.ACESFilmicToneMapping;
}

function applyTransform(object, params = {}) {
    if (Array.isArray(params.position)) object.position.fromArray(params.position);
    if (Array.isArray(params.rotation)) object.rotation.fromArray(params.rotation);
    if (Array.isArray(params.scale)) object.scale.fromArray(params.scale);
    if (typeof params.visible === "boolean") object.visible = params.visible;
    if (typeof params.name === "string") object.name = params.name;
    return object;
}

/**
 * The parameter names a script body receives, and the values to pass for them.
 *
 * @param {object} params - The descriptor's `params`. `names` renames the slots in declared
 *     order and `values` appends widgets; without them the slot names are used as they are.
 * @param {string[]} slots - The node's own slot names, in order.
 * @param {object} resolved - The resolved value per slot name.
 * @returns {{names: string[], args: unknown[]}} Names and values, in matching order.
 */
function scriptBindings(params, slots, resolved) {
  const declared = Array.isArray(params?.names) ? params.names : null;
  const names = declared ? declared.slice(0, slots.length) : slots.slice();
  const args = names.map((_name, index) => resolved[slots[index]] ?? null);
  const values = params?.values;
  if (values && typeof values === "object") {
    for (const [name, value] of Object.entries(values)) {
      names.push(name);
      args.push(value);
    }
  }
  return { names, args };
}

function createAsyncFunction(...args) {
    const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
    return new AsyncFunction(...args);
}

export function assertDescriptor(spec, kind = null) {
    if (!spec || typeof spec !== "object" || spec[WRAPPER_KEY] !== SCHEMA_VERSION) {
        throw new Error(
            "This input did not arrive as a Three.js descriptor. Wire it from a Three "
                + "node rather than from another socket that happens to carry an object."
        );
    }
    if (kind && spec.kind !== kind) {
        throw new Error(`Expected ${kind} descriptor, received ${spec.kind}.`);
    }
    return spec;
}


/**
 * Orbit, dolly and pan a camera from pointer input on one canvas.
 *
 * @param {object} camera - The camera to place.
 * @param {HTMLCanvasElement} canvas - The surface the pointer is read from.
 * @returns {object} The control, with `target`, `update`, `sync` and `dispose`.
 */
export function createOrbitControls(camera, canvas) {
    const TURN_PER_WIDTH = 2 * Math.PI;
    const WHEEL_STEP = 1.12;
    const MIN_RADIUS = 0.01;
    const MAX_RADIUS = 1e6;
    const POLAR_EDGE = 1e-4;
    const EASE = 0.25;

    const target = new THREE.Vector3();
    const offset = new THREE.Vector3();
    const right = new THREE.Vector3();
    const up = new THREE.Vector3();

    let radius = 1;
    let theta = 0;
    let phi = 1;
    let wantRadius = 1;
    let wantTheta = 0;
    let wantPhi = 1;

    const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

    const readCamera = () => {
        offset.copy(camera.position).sub(target);
        radius = clamp(offset.length(), MIN_RADIUS, MAX_RADIUS);
        theta = Math.atan2(offset.x, offset.z);
        phi = Math.acos(clamp(offset.y / radius, -1, 1));
        wantRadius = radius;
        wantTheta = theta;
        wantPhi = phi;
    };
    readCamera();

    let pointerId = null;
    let panning = false;
    let lastX = 0;
    let lastY = 0;

    const onDown = (event) => {
        if (!api.enabled || pointerId !== null) return;
        pointerId = event.pointerId;
        panning = event.button === 1 || event.button === 2 || event.shiftKey;
        lastX = event.clientX;
        lastY = event.clientY;
        try {
            canvas.setPointerCapture(pointerId);
        } catch (error) {
            /* the pointer ended before capture could be taken */
        }
        event.preventDefault();
    };

    const onMove = (event) => {
        if (pointerId === null || event.pointerId !== pointerId) return;
        const box = canvas.getBoundingClientRect();
        const dx = (event.clientX - lastX) / Math.max(1, box.width);
        const dy = (event.clientY - lastY) / Math.max(1, box.height);
        lastX = event.clientX;
        lastY = event.clientY;

        if (panning) {
            right.setFromMatrixColumn(camera.matrix, 0);
            up.setFromMatrixColumn(camera.matrix, 1);
            const reach = wantRadius * Math.tan((camera.fov ?? 50) * Math.PI / 360) * 2;
            target.addScaledVector(right, -dx * reach);
            target.addScaledVector(up, dy * reach);
        } else {
            wantTheta -= dx * TURN_PER_WIDTH;
            wantPhi = clamp(wantPhi - dy * Math.PI, POLAR_EDGE, Math.PI - POLAR_EDGE);
        }
        event.preventDefault();
    };

    const onUp = (event) => {
        if (pointerId === null) return;
        if (event && event.pointerId !== undefined && event.pointerId !== pointerId) return;
        try {
            if (canvas.hasPointerCapture?.(pointerId)) canvas.releasePointerCapture(pointerId);
        } catch (error) {
            /* capture was already given up */
        }
        pointerId = null;
        panning = false;
    };

    const onWheel = (event) => {
        if (!api.enabled || !event.deltaY) return;
        const step = event.deltaY < 0 ? 1 / WHEEL_STEP : WHEEL_STEP;
        wantRadius = clamp(wantRadius * step, MIN_RADIUS, MAX_RADIUS);
        event.preventDefault();
    };

    const onContextMenu = (event) => event.preventDefault();

    canvas.addEventListener("pointerdown", onDown);
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerup", onUp);
    canvas.addEventListener("pointercancel", onUp);
    canvas.addEventListener("lostpointercapture", onUp);
    canvas.addEventListener("wheel", onWheel, { passive: false });
    canvas.addEventListener("contextmenu", onContextMenu);
    window.addEventListener("pointerup", onUp);
    window.addEventListener("blur", onUp);

    const api = {
        target,
        enabled: true,
        enableDamping: true,
        autoRotate: false,
        autoRotateSpeed: 1,
        getAzimuthalAngle: () => theta,
        getPolarAngle: () => phi,
        get dragging() {
            return pointerId !== null;
        },

        update() {
            if (this.autoRotate && pointerId === null) {
                wantTheta -= 0.0025 * this.autoRotateSpeed;
            }
            const ease = this.enableDamping ? EASE : 1;
            theta += (wantTheta - theta) * ease;
            phi += (wantPhi - phi) * ease;
            radius += (wantRadius - radius) * ease;

            const sinPhi = Math.sin(phi);
            offset.set(radius * sinPhi * Math.sin(theta), radius * Math.cos(phi), radius * sinPhi * Math.cos(theta));
            camera.position.copy(target).add(offset);
            camera.lookAt(target);
        },

        // Re-read the angles after the camera has been moved by something else.
        sync() {
            readCamera();
        },

        dispose() {
            onUp(null);
            canvas.removeEventListener("pointerdown", onDown);
            canvas.removeEventListener("pointermove", onMove);
            canvas.removeEventListener("pointerup", onUp);
            canvas.removeEventListener("pointercancel", onUp);
            canvas.removeEventListener("lostpointercapture", onUp);
            canvas.removeEventListener("wheel", onWheel);
            canvas.removeEventListener("contextmenu", onContextMenu);
            window.removeEventListener("pointerup", onUp);
            window.removeEventListener("blur", onUp);
        },
    };
    return api;
}

/**
 * A loading manager answering a model's references from its held files, by file name, and
 * anything not held with empty bytes.
 *
 * @param {object} resources - `{name: url}` for the files held beside the model.
 * @returns {THREE.LoadingManager} The manager, to hand to a loader.
 */
function sidecarManager(resources) {
    const manager = new THREE.LoadingManager();
    const held = resources && typeof resources === "object" ? resources : {};
    const byName = new Map();
    for (const [name, url] of Object.entries(held)) {
        byName.set(name.toLowerCase(), url);
    }

    // A loader answers when it has parsed the model, while its textures are still arriving.
    // Drawing then gives one untextured frame, so the wait is until the manager is idle.
    let done = false;
    const idle = new Promise((resolve) => {
        manager.onLoad = () => {
            done = true;
            resolve();
        };
    });
    manager.whenIdle = (patience = LOAD_PATIENCE_MS) => (done
        ? Promise.resolve()
        : Promise.race([idle, new Promise((resolve) => setTimeout(resolve, patience))]));

    manager.setURLModifier((url) => {
        const asked = String(url || "");
        // An address already pointing at the asset route is the model itself.
        if (isAssetRoute(asked)) return asked;
        // Bytes embedded in the model are read where they are.
        if (/^(data|blob):/i.test(asked.trim())) return asked;
        let name = "";
        try {
            name = decodeURIComponent(asked.split(/[?#]/)[0].split("/").pop() || "").toLowerCase();
        } catch (error) {
            name = "";
        }
        const held = byName.get(name);
        if (held !== undefined) return isHeldAddress(held) ? held : REFUSED_ADDRESS;
        return isHeldAddress(asked) ? asked : REFUSED_ADDRESS;
    });
    return manager;
}

/**
 * Read one model file into an Object3D.
 *
 * @param {object} params - `{url, format, resources}` from a Three Load Model descriptor.
 * @returns {Promise<object>} The loaded object.
 */
async function loadModelFile(params) {
    const manager = sidecarManager(params.resources);
    const object = await readModel(params, manager);
    // A loader answers on the model alone, so the wait here is for its textures.
    await manager.whenIdle?.();
    return object;
}

async function readModel(params, manager) {
    const address = String(params.url || "");
    const format = String(params.format || "").toLowerCase();
    if (!address) throw new Error("Three Load Model gave no address to read from.");

    if (format === "glb" || format === "gltf") {
        const { GLTFLoader } = await import("../vendor/three/loaders/GLTFLoader.js");
        const loader = new GLTFLoader(manager);
        const decoders = Array.isArray(params.decoders) ? params.decoders : [];
        if (decoders.includes("meshopt")) {
            const { MeshoptDecoder } = await import("../vendor/three/libs/meshopt_decoder.module.js");
            loader.setMeshoptDecoder(MeshoptDecoder);
        }
        if (decoders.includes("draco")) loader.setDRACOLoader(await dracoLoader());
        const gltf = await loader.loadAsync(address);
        // Clips ride on the object, the way three's own loaders leave them.
        gltf.scene.animations = gltf.animations || [];
        return gltf.scene;
    }
    if (format === "obj") {
        const { OBJLoader } = await import("../vendor/three/loaders/OBJLoader.js");
        const loader = new OBJLoader(manager);
        // A .obj keeps its materials in a .mtl beside it, which is held under its own name.
        const library = Object.keys(params.resources || {}).find(
            (name) => name.toLowerCase().endsWith(".mtl")
        );
        if (library) {
            const { MTLLoader } = await import("../vendor/three/loaders/MTLLoader.js");
            const materials = await new MTLLoader(manager).loadAsync(params.resources[library]);
            materials.preload();
            loader.setMaterials(materials);
        }
        return await loader.loadAsync(address);
    }
    if (format === "dae") {
        const { ColladaLoader } = await import("../vendor/three/loaders/ColladaLoader.js");
        const collada = await new ColladaLoader(manager).loadAsync(address);
        const scene = collada.scene;
        scene.animations = scene.animations || collada.animations || [];
        return scene;
    }
    if (format === "fbx") {
        const { FBXLoader } = await import("../vendor/three/loaders/FBXLoader.js");
        return await new FBXLoader(manager).loadAsync(address);
    }
    if (format === "3mf") {
        const { ThreeMFLoader } = await import("../vendor/three/loaders/3MFLoader.js");
        return await new ThreeMFLoader(manager).loadAsync(address);
    }
    if (format === "stl" || format === "ply") {
        const module = format === "stl"
            ? await import("../vendor/three/loaders/STLLoader.js")
            : await import("../vendor/three/loaders/PLYLoader.js");
        const Loader = module.STLLoader || module.PLYLoader;
        const geometry = await new Loader(manager).loadAsync(address);
        geometry.computeVertexNormals();
        // A vertex-coloured PLY says so on the geometry, and the material has to opt in.
        const coloured = Boolean(geometry.getAttribute("color"));
        return new THREE.Mesh(
            geometry,
            new THREE.MeshStandardMaterial({
                color: coloured ? "#ffffff" : "#cccccc",
                vertexColors: coloured,
            })
        );
    }
    throw new Error(`No loader reads a .${format} model.`);
}

/**
 * The object a path of child indices leads to, read through the override wrappers.
 *
 * @param {THREE.Object3D} root - Where the path starts, the scene.
 * @param {number[]} path - Child indices, from the root down.
 * @returns {THREE.Object3D|null} The object, or null where the path leads nowhere.
 */
export function objectAtPath(root, path) {
    let node = root;
    for (const index of path) {
        const children = node.children.filter((child) => !child.userData.wasHelper);
        let next = children[index];
        while (next?.userData.wasOverride) next = next.children[0]?.children[0];
        if (!next) return null;
        node = next;
    }
    return node;
}

/**
 * The path of child indices from the root to an object, read through the override wrappers.
 *
 * @param {THREE.Object3D} root - The scene.
 * @param {THREE.Object3D} object - An object inside it.
 * @returns {number[]|null} The path, or null where the object is not inside the root.
 */
export function pathToObject(root, object) {
    const path = [];
    let node = object;
    while (node && node !== root) {
        let parent = node.parent;
        let holder = node;
        // An override wrapper and its inner offset group stand in the object's place.
        if (parent?.userData.wasOverrideInner) {
            holder = parent.parent;
            parent = holder.parent;
        }
        if (!parent) return null;
        const children = parent.children.filter((child) => !child.userData.wasHelper);
        path.unshift(children.indexOf(holder));
        node = parent;
    }
    return node === root ? path : null;
}

/**
 * Put an override wrapper around an object, turning about the pivot given, in the object's place.
 *
 * @param {THREE.Object3D} object - The object to wrap.
 * @param {number[]} pivot - The point the wrapper turns about, in the parent's space.
 * @returns {THREE.Object3D} The wrapper; already wrapped objects answer their own.
 */
export function wrapForOverride(object, pivot) {
    if (object.parent?.userData.wasOverrideInner) return object.parent.parent;
    const parent = object.parent;
    const wrapper = new THREE.Group();
    wrapper.name = `${object.name || "object"} (moved)`;
    wrapper.userData.wasOverride = true;
    wrapper.userData.pivot = pivot.slice();
    wrapper.position.fromArray(pivot);
    const inner = new THREE.Group();
    inner.userData.wasOverrideInner = true;
    inner.position.set(-pivot[0], -pivot[1], -pivot[2]);
    const at = parent.children.indexOf(object);
    parent.remove(object);
    inner.add(object);
    wrapper.add(inner);
    parent.add(wrapper);
    // Kept at the object's own index, so the paths of its siblings do not move.
    parent.children.splice(parent.children.indexOf(wrapper), 1);
    parent.children.splice(at, 0, wrapper);
    return wrapper;
}

/**
 * Move the objects a viewer's saved view names, each inside an override wrapper.
 *
 * @param {THREE.Object3D} scene - The built scene.
 * @param {object[]} overrides - `[{path, pivot, position, quaternion, scale}]`.
 * @returns {void}
 */
export function applyObjectOverrides(scene, overrides) {
    if (!Array.isArray(overrides) || !overrides.length) return;
    // Deepest first, so wrapping an object never moves the path to one inside it.
    const ordered = [...overrides].sort((a, b) => (b.path?.length || 0) - (a.path?.length || 0));
    for (const entry of ordered) {
        const object = Array.isArray(entry.path) ? objectAtPath(scene, entry.path) : null;
        if (!object || !Array.isArray(entry.pivot)) continue;
        const wrapper = wrapForOverride(object, entry.pivot);
        if (Array.isArray(entry.position)) wrapper.position.fromArray(entry.position);
        if (Array.isArray(entry.quaternion)) wrapper.quaternion.fromArray(entry.quaternion);
        if (Array.isArray(entry.scale)) wrapper.scale.fromArray(entry.scale);
    }
}

// The page's one Draco decoder, made on first use; its workers outlive a single model.
let dracoShared = null;

/**
 * The Draco decoder glTF geometry compressed with `KHR_draco_mesh_compression` is read through.
 *
 * Its own loading manager fetches the vendored WebAssembly, outside any model's sidecar list.
 *
 * @returns {Promise<object>} A `DRACOLoader`.
 */
function dracoLoader() {
    dracoShared ??= import("../vendor/three/loaders/DRACOLoader.js").then(({ DRACOLoader }) =>
        new DRACOLoader().setDecoderPath({
            js: new URL("../vendor/three/libs/draco/gltf/draco_wasm_wrapper.txt", import.meta.url).href,
            wasm: new URL("../vendor/three/libs/draco/gltf/draco_decoder.wasm", import.meta.url).href,
        })
    );
    return dracoShared;
}

/**
 * Scale a loaded model, then move it so the point `origin` names sits at its parent's origin.
 *
 * @param {THREE.Object3D} object - The loaded model, not yet parented.
 * @param {object} params - `{scale, fitSize, origin}` from a Three Load Model descriptor.
 * @returns {void}
 */
function placeModel(object, params) {
    let factor = Number(params.scale) || 1;
    const fit = Number(params.fitSize) || 0;
    if (fit > 0) {
        object.updateMatrixWorld(true);
        const size = new THREE.Box3().setFromObject(object).getSize(new THREE.Vector3());
        const longest = Math.max(size.x, size.y, size.z);
        if (Number.isFinite(longest) && longest > 0) factor *= fit / longest;
    }
    object.scale.multiplyScalar(factor);

    const origin = params.origin || (params.centre === false ? "as saved" : "centre");
    if (origin === "as saved") return;
    object.updateMatrixWorld(true);
    const target = modelCentre(object);
    if (origin === "base") {
        const box = new THREE.Box3().setFromObject(object);
        if (!box.isEmpty()) target.y = box.min.y;
    }
    object.position.sub(target);
}

/**
 * Turn a model's metal down and take the mirror shine off its smoothest surfaces.
 *
 * @param {THREE.Object3D} object - The loaded model, changed in place.
 * @param {number} metalness - Multiplies how metallic every surface is; 1 keeps the file's.
 * @param {number} minRoughness - The least roughness any surface keeps; 0 keeps the file's.
 * @returns {void}
 */
function adjustSurfaces(object, metalness, minRoughness) {
    const scale = Math.min(Math.max(Number.isFinite(metalness) ? metalness : 1, 0), 1);
    const floor = Math.min(Math.max(Number.isFinite(minRoughness) ? minRoughness : 0, 0), 1);
    if (scale === 1 && floor === 0) return;

    // A glTF model packs roughness into the green channel of one picture, shared by many
    // materials, so each picture is remade once with its green raised to the floor.
    const remade = new Map();
    const raised = (texture, factor) => {
        const image = texture?.image;
        if (!image || !image.width || !image.height || image.data) return texture;
        const key = `${texture.uuid}:${factor}`;
        if (remade.has(key)) return remade.get(key);
        const canvas = typeof document !== "undefined"
            ? Object.assign(document.createElement("canvas"), { width: image.width, height: image.height })
            : new OffscreenCanvas(image.width, image.height);
        const context = canvas.getContext("2d", { willReadFrequently: true });
        context.drawImage(image, 0, 0);
        const pixels = context.getImageData(0, 0, image.width, image.height);
        const low = Math.min(255, Math.ceil((floor / Math.max(factor, 1e-3)) * 255));
        const data = pixels.data;
        for (let index = 1; index < data.length; index += 4) {
            if (data[index] < low) data[index] = low;
        }
        context.putImageData(pixels, 0, 0);
        const copy = texture.clone();
        copy.source = new THREE.Source(canvas);
        copy.needsUpdate = true;
        remade.set(key, copy);
        return copy;
    };

    object.traverse((child) => {
        if (!child.isMesh) return;
        const materials = Array.isArray(child.material) ? child.material : [child.material];
        for (const material of materials) {
            if (!material || !("metalness" in material) || !("roughness" in material)) continue;
            material.metalness *= scale;
            if (floor > 0) {
                if (material.roughnessMap) {
                    const shared = material.metalnessMap === material.roughnessMap;
                    material.roughnessMap = raised(material.roughnessMap, material.roughness);
                    if (shared) material.metalnessMap = material.roughnessMap;
                } else {
                    material.roughness = Math.max(material.roughness, floor);
                }
            }
            material.needsUpdate = true;
        }
    });
}

/**
 * A top-down map of a ground's height, from its vertices.
 *
 * @param {THREE.Object3D} ground - The ground, in place in the scene.
 * @param {number} cell - Width of one cell in scene units.
 * @param {number} ceiling - Vertices above this height are left out, so walls and roofs over
 *   the path are not read as floor.
 * @returns {{heights: Float32Array, nx: number, nz: number, cell: number, minX: number,
 *   minZ: number}} The highest point in each cell, -Infinity where the ground has none.
 */
function groundHeights(ground, cell, ceiling) {
    ground.updateWorldMatrix(true, true);
    const box = new THREE.Box3().setFromObject(ground);
    const width = Math.max(box.max.x - box.min.x, 1e-3);
    const depth = Math.max(box.max.z - box.min.z, 1e-3);
    // Large grounds take coarser cells rather than an unbounded grid.
    const size = Math.max(cell, Math.sqrt((width * depth) / 4e6));
    const nx = Math.ceil(width / size) + 1;
    const nz = Math.ceil(depth / size) + 1;
    const heights = new Float32Array(nx * nz).fill(-Infinity);
    const point = new THREE.Vector3();
    ground.traverse((mesh) => {
        const position = mesh.isMesh ? mesh.geometry?.attributes?.position : null;
        if (!position) return;
        for (let index = 0; index < position.count; index += 1) {
            point.fromBufferAttribute(position, index).applyMatrix4(mesh.matrixWorld);
            if (point.y > ceiling) continue;
            const at = Math.floor((point.z - box.min.z) / size) * nx + Math.floor((point.x - box.min.x) / size);
            if (point.y > heights[at]) heights[at] = point.y;
        }
    });
    return { heights, nx, nz, cell: size, minX: box.min.x, minZ: box.min.z };
}

/**
 * The height of a ground map under a patch.
 *
 * @param {object} map - From groundHeights.
 * @param {number} x - Patch centre, world x.
 * @param {number} z - Patch centre, world z.
 * @param {number} footprint - Patch width in scene units.
 * @returns {number|null} The 60th percentile height in the patch, or null off the ground.
 */
function heightUnder(map, x, z, footprint) {
    const reach = Math.max(0, Math.floor(footprint / 2 / map.cell));
    const cx = Math.floor((x - map.minX) / map.cell);
    const cz = Math.floor((z - map.minZ) / map.cell);
    const found = [];
    for (let dz = -reach; dz <= reach; dz += 1) {
        for (let dx = -reach; dx <= reach; dx += 1) {
            const ix = cx + dx;
            const iz = cz + dz;
            if (ix < 0 || iz < 0 || ix >= map.nx || iz >= map.nz) continue;
            const value = map.heights[iz * map.nx + ix];
            if (value > -Infinity) found.push(value);
        }
    }
    if (!found.length) return null;
    found.sort((a, b) => a - b);
    return found[Math.min(found.length - 1, Math.floor(found.length * 0.6))];
}

/**
 * The middle of a loaded model, measured from its bones where it carries a skeleton.
 *
 * @param {THREE.Object3D} object - The loaded model.
 * @returns {THREE.Vector3} The middle, in the model's own space.
 */
function modelCentre(object) {
    object.updateMatrixWorld(true);
    const bones = [];
    object.traverse((child) => {
        if (child.isBone) bones.push(child);
    });

    const box = new THREE.Box3();
    if (bones.length) {
        // The bind pose's own bounds are ignored; the placed bones are measured instead.
        const at = new THREE.Vector3();
        for (const bone of bones) box.expandByPoint(bone.getWorldPosition(at));
    } else {
        box.setFromObject(object);
    }
    if (box.isEmpty()) return new THREE.Vector3();

    const middle = box.getCenter(new THREE.Vector3());
    return object.parent ? object.parent.worldToLocal(middle) : middle;
}

/**
 * Build the chain of passes a finished frame is put through.
 *
 * @param {object} spec - The last effect descriptor in the chain, or null for none.
 * @param {object} parts - `{renderer, scene, camera, width, height}`.
 * @returns {Promise<object|null>} An `EffectComposer` to render with, or null where the chain
 *   is empty and the renderer should be used directly.
 */
async function createComposer(spec, parts) {
    if (!spec) return null;

    // The chain is described from its last effect back, so it is unwound into order first.
    const ordered = [];
    for (let step = spec; step; step = step.deps?.input ?? null) {
        assertDescriptor(step, "effect");
        ordered.unshift(step);
    }

    const { EffectComposer } = await import("../vendor/three/postprocessing/EffectComposer.js");
    const { RenderPass } = await import("../vendor/three/postprocessing/RenderPass.js");
    const { OutputPass } = await import("../vendor/three/postprocessing/OutputPass.js");

    const composer = new EffectComposer(parts.renderer);
    composer.setSize(parts.width, parts.height);
    composer.addPass(new RenderPass(parts.scene, parts.camera));

    for (const step of ordered) {
        const params = step.params || {};
        if (step.type === "Bloom") {
            const { UnrealBloomPass } = await import(
                "../vendor/three/postprocessing/UnrealBloomPass.js"
            );
            composer.addPass(new UnrealBloomPass(
                new THREE.Vector2(parts.width, parts.height),
                Number(params.strength) || 0,
                Number(params.radius) || 0,
                Number(params.threshold) || 0
            ));
        } else if (step.type === "DepthOfField") {
            const { BokehPass } = await import("../vendor/three/postprocessing/BokehPass.js");
            composer.addPass(new BokehPass(parts.scene, parts.camera, {
                focus: Number(params.focus) || 1,
                aperture: Number(params.aperture) || 0,
                maxblur: Number(params.maxBlur) || 0,
            }));
        } else if (step.type === "Antialias") {
            const { SMAAPass } = await import("../vendor/three/postprocessing/SMAAPass.js");
            composer.addPass(new SMAAPass(parts.width, parts.height));
        } else {
            throw new Error(`Unsupported effect type: ${step.type}`);
        }
    }

    // Tone mapping and the colour space are applied once, at the end of the chain.
    composer.addPass(new OutputPass());
    return composer;
}

/**
 * How far through its own clock an animation is, as a fraction.
 *
 * @param {string} units - `per second`, `per capture` or `per timeline`.
 * @param {number} time - Seconds since the scene started.
 * @param {object} ctx - The runtime context, carrying both clocks.
 * @returns {number} 0 to 1 for the two spans, or the elapsed seconds for `per second`.
 */
function alongFor(units, time, ctx) {
    if (units === "per timeline") {
        // The timeline is the whole animation. A capture takes a window out of it, so a
        // shorter capture shows less of the motion rather than a faster version of all of it.
        const timeline = Number(ctx?.timelineSeconds) || 0;
        return timeline > 0 ? (time / timeline) % 1 : 0;
    }
    if (units === "per capture") {
        const span = Number(ctx?.duration) || 0;
        const origin = Number(ctx?.timeOrigin) || 0;
        return span > 0 ? (time - origin) / span : 0;
    }
    return time;
}

const renderBusyListeners = new Set();
let renderBusy = false;

/**
 * Say whether a Three Render job is drawing in this page.
 *
 * @param {boolean} busy - True while a job draws.
 */
export function setRenderBusy(busy) {
    if (renderBusy === Boolean(busy)) return;
    renderBusy = Boolean(busy);
    for (const listener of renderBusyListeners) {
        try {
            listener(renderBusy);
        } catch (error) {
            console.error("[WAS ThreeJS] Render state listener failed:", error);
        }
    }
}

/**
 * Hear when Three Render jobs start and stop drawing in this page.
 *
 * @param {(busy: boolean) => void} listener - Called now with the current state, then on
 *   every change.
 * @returns {() => void} Stops listening.
 */
export function onRenderBusy(listener) {
    renderBusyListeners.add(listener);
    listener(renderBusy);
    return () => renderBusyListeners.delete(listener);
}

/**
 * The space a node's position is written in, measured in the root's space.
 *
 * @param {THREE.Object3D} root - The animated model.
 * @param {THREE.Object3D} node - The node the position track moves.
 * @returns {THREE.Matrix4} From the node's parent space to the root's.
 */
function parentToRoot(root, node) {
    root.updateWorldMatrix(true, true);
    const toRoot = root.matrixWorld.clone().invert();
    return node.parent ? toRoot.multiply(node.parent.matrixWorld) : toRoot.identity();
}

/**
 * The position track that carries a clip across the ground, and how far one pass moves it.
 *
 * @param {THREE.Object3D} root - The object the clip animates, posed at rest.
 * @param {THREE.AnimationClip} clip - The clip to read.
 * @returns {{name: string, node: THREE.Object3D, travel: THREE.Vector3, keys: Array}|null}
 *   The track's name, the node it moves, its ground travel per pass in the root's space and
 *   each key's parent space and root space position, or null when nothing travels.
 */
function groundTravel(root, clip) {
    const tracks = [];
    for (const track of clip.tracks) {
        if (!track.name.endsWith(".position") || track.times.length < 2) continue;
        const parsed = THREE.PropertyBinding.parseTrackName(track.name);
        const node = THREE.PropertyBinding.findNode(root, parsed.nodeName);
        if (node) tracks.push({ track, node });
    }
    if (!tracks.length) return null;

    // The rest pose comes back afterwards, whatever the sampling left on the bones.
    const rest = [];
    root.traverse((child) => {
        rest.push([child, child.position.clone(), child.quaternion.clone(), child.scale.clone()]);
    });
    const mixer = new THREE.AnimationMixer(root);
    const action = mixer.clipAction(clip);
    action.setLoop(THREE.LoopOnce, 1);
    action.clampWhenFinished = true;
    action.play();

    const valueAt = (track, key) => {
        const stride = track.values.length / track.times.length;
        const base = key * stride + (stride === 9 ? 3 : 0);
        return new THREE.Vector3().fromArray(track.values, base);
    };
    // Each key is measured with the clip's whole pose applied, so a parent turning under the
    // moving node is part of the measurement.
    const measure = (entry, key) => {
        // A finished one-shot action holds its end pose until it is reset.
        action.reset();
        mixer.setTime(entry.track.times[key]);
        const space = parentToRoot(root, entry.node);
        return { space, at: valueAt(entry.track, key).applyMatrix4(space) };
    };

    // Every node under the moving one travels with it, so the highest moving node is the one.
    const depthOf = (node) => {
        let depth = 0;
        for (let up = node.parent; up && up !== root; up = up.parent) depth += 1;
        return depth;
    };
    const measured = tracks.map((entry) => {
        const first = measure(entry, 0).at;
        const last = measure(entry, entry.track.times.length - 1).at;
        const travel = last.sub(first).setY(0);
        return { entry, travel, distance: travel.length(), depth: depthOf(entry.node) };
    });
    const furthest = Math.max(...measured.map((one) => one.distance));
    let best = null;
    for (const one of measured) {
        if (one.distance <= 1e-6 || one.distance < furthest * 0.25) continue;
        if (!best || one.depth < best.depth
            || (one.depth === best.depth && one.distance > best.distance)) best = one;
    }

    let motion = null;
    if (best) {
        const keys = [];
        for (let key = 0; key < best.entry.track.times.length; key += 1) {
            keys.push(measure(best.entry, key));
        }
        motion = { name: best.entry.track.name, node: best.entry.node, travel: best.travel, keys };
    }

    action.stop();
    mixer.uncacheRoot(root);
    for (const [child, position, quaternion, scale] of rest) {
        child.position.copy(position);
        child.quaternion.copy(quaternion);
        child.scale.copy(scale);
    }
    return motion;
}

/**
 * Turn each foot and toe so a planted foot lies the way the model's rest pose holds it.
 *
 * @param {THREE.Object3D} root - The object the clip animates, posed at rest.
 * @param {THREE.AnimationClip} clip - The clip, changed in place.
 * @returns {number} How many feet were turned.
 */
function plantFeet(root, clip) {
    const trackFor = (bone) => clip.tracks.find((track) => track.name === `${bone.name}.quaternion`);
    const feet = [];
    root.traverse((bone) => {
        if (!bone.isBone || !/foot|ankle/i.test(bone.name)) return;
        if (/toe|ball|twist|end|ik|pole|roll|ctrl/i.test(bone.name)) return;
        const toe = bone.children.find((child) => child.isBone && /toe|ball/i.test(child.name));
        const footTrack = trackFor(bone);
        if (toe && footTrack) feet.push({ bone, toe, footTrack, toeTrack: trackFor(toe) || null });
    });
    if (!feet.length) return 0;

    const up = new THREE.Vector3(0, 1, 0);
    const inRoot = (object, rootInverse) => new THREE.Vector3()
        .setFromMatrixPosition(object.matrixWorld).applyMatrix4(rootInverse);
    const pitchOf = (from, to) => {
        const along = to.clone().sub(from);
        return Math.atan2(along.y, Math.hypot(along.x, along.z));
    };

    const rest = [];
    root.traverse((child) => {
        rest.push([child, child.position.clone(), child.quaternion.clone(), child.scale.clone()]);
    });
    root.updateWorldMatrix(true, true);
    let rootInverse = root.matrixWorld.clone().invert();
    for (const foot of feet) {
        foot.restPitch = pitchOf(inRoot(foot.bone, rootInverse), inRoot(foot.toe, rootInverse));
        foot.restToe = foot.toe.quaternion.clone();
    }

    const mixer = new THREE.AnimationMixer(root);
    const action = mixer.clipAction(clip);
    action.setLoop(THREE.LoopOnce, 1);
    action.clampWhenFinished = true;
    action.play();
    const poseAt = (time) => {
        action.reset();
        mixer.setTime(time);
        root.updateWorldMatrix(true, true);
        rootInverse = root.matrixWorld.clone().invert();
    };

    // The lowest moment of each ankle is where that foot stands.
    const samples = Math.max(24, Math.ceil(clip.duration * 60));
    for (const foot of feet) foot.lowest = { height: Infinity, time: 0 };
    for (let step = 0; step <= samples; step += 1) {
        const time = (clip.duration * step) / samples;
        poseAt(time);
        for (const foot of feet) {
            const height = inRoot(foot.bone, rootInverse).y;
            if (height < foot.lowest.height) foot.lowest = { height, time };
        }
    }

    // At that moment the foot takes a constant turn putting its heel to toe line at the rest
    // angle, and the toe takes its rest bend, so every other moment keeps its angle to it.
    const footTurn = new THREE.Quaternion();
    const toeTurn = new THREE.Quaternion();
    const held = new THREE.Quaternion();
    const rootTurn = new THREE.Quaternion();
    for (const foot of feet) {
        poseAt(foot.lowest.time);
        const ankle = inRoot(foot.bone, rootInverse);
        const ball = inRoot(foot.toe, rootInverse);
        const flat = ball.clone().sub(ankle).setY(0);
        if (flat.lengthSq() < 1e-12) continue;
        const side = flat.normalize().cross(up).normalize();
        const turn = new THREE.Quaternion().setFromAxisAngle(side, foot.restPitch - pitchOf(ankle, ball));

        // The turn is made in the root's space, then carried into the foot's own frame.
        rootTurn.copy(root.getWorldQuaternion(held).invert())
            .multiply(foot.bone.getWorldQuaternion(new THREE.Quaternion()));
        footTurn.copy(rootTurn).invert().multiply(turn).multiply(rootTurn);
        toeTurn.copy(foot.toe.quaternion).invert().multiply(foot.restToe);

        for (const [track, offset] of [[foot.footTrack, footTurn], [foot.toeTrack, toeTurn]]) {
            if (!track) continue;
            const stride = track.values.length / track.times.length;
            const at = stride === 12 ? 4 : 0;
            for (let key = 0; key < track.times.length; key += 1) {
                held.fromArray(track.values, key * stride + at).multiply(offset).normalize();
                held.toArray(track.values, key * stride + at);
            }
        }
    }

    action.stop();
    mixer.uncacheRoot(root);
    for (const [child, position, quaternion, scale] of rest) {
        child.position.copy(position);
        child.quaternion.copy(quaternion);
        child.scale.copy(scale);
    }
    return feet.length;
}

/**
 * Take a clip's ground travel out of it, keeping its sway and its rise and fall.
 *
 * @param {THREE.AnimationClip} clip - The clip, changed in place.
 * @param {{name: string, travel: THREE.Vector3, keys: Array}} motion - From groundTravel.
 */
function holdInPlace(clip, motion) {
    const track = clip.tracks.find((one) => one.name === motion.name);
    if (!track) return;
    const times = track.times;
    const values = track.values;
    const stride = values.length / times.length;
    const start = times[0];
    const span = times[times.length - 1] - start || 1;
    const slope = motion.travel.clone().divideScalar(span);
    for (let key = 0; key < times.length; key += 1) {
        const share = (times[key] - start) / span;
        const { space, at } = motion.keys[key];
        const back = space.clone().invert();
        const held = at.clone().addScaledVector(motion.travel, -share).applyMatrix4(back);
        held.toArray(values, key * stride + (stride === 9 ? 3 : 0));
        if (stride === 9) {
            // Cubic spline tangents are rates, so the ramp's slope comes off both.
            const rate = slope.clone().applyMatrix3(new THREE.Matrix3().setFromMatrix4(back));
            for (const tangent of [key * stride, key * stride + 6]) {
                values[tangent] -= rate.x;
                values[tangent + 1] -= rate.y;
                values[tangent + 2] -= rate.z;
            }
        }
    }
}

/**
 * A material writing view space distance as an even grey, white for near.
 *
 * @param {{near: number, far: number}} bounds - The range mapped onto white to black.
 * @returns {THREE.ShaderMaterial} The material, for `scene.overrideMaterial`.
 */
function createDepthMaterial(bounds) {
    // The skinning chunks are included, so a rigged model is measured where it is posed.
    return new THREE.ShaderMaterial({
        uniforms: {
            uNear: { value: bounds.near },
            uFar: { value: bounds.far },
        },
        vertexShader: `
            #include <common>
            #include <skinning_pars_vertex>
            #include <morphtarget_pars_vertex>
            varying float vViewDepth;
            void main() {
                #include <skinbase_vertex>
                #include <begin_vertex>
                #include <morphtarget_vertex>
                #include <skinning_vertex>
                vec4 viewPosition = modelViewMatrix * vec4( transformed, 1.0 );
                vViewDepth = -viewPosition.z;
                gl_Position = projectionMatrix * viewPosition;
            }
        `,
        fragmentShader: `
            uniform float uNear;
            uniform float uFar;
            varying float vViewDepth;
            void main() {
                float along = clamp( ( vViewDepth - uNear ) / max( uFar - uNear, 1e-6 ), 0.0, 1.0 );
                gl_FragColor = vec4( vec3( 1.0 - along ), 1.0 );
            }
        `,
    });
}

/**
 * Draw a scene with one material over everything, for a depth or a normal pass.
 *
 * @param {object} parts - `{renderer, scene, camera, kind}`. `kind` is `depth` or `normal`.
 * @param {object} bounds - `{near, far}` the depth pass is spread across.
 * @returns {void} The frame is left on the renderer's canvas.
 */
function renderOverridePass(parts, bounds) {
    const { renderer, scene, camera, kind } = parts;
    const heldOverride = scene.overrideMaterial;
    const heldBackground = scene.background;
    const heldNear = camera.near;
    const heldFar = camera.far;
    const heldTone = renderer.toneMapping;

    // A pass is data, not a picture, so nothing may be tone mapped on the way out.
    renderer.toneMapping = THREE.NoToneMapping;
    scene.background = new THREE.Color(kind === "depth" ? 0x000000 : 0x7f7fff);

    if (kind === "depth") {
        scene.overrideMaterial = createDepthMaterial(bounds);
    } else {
        scene.overrideMaterial = new THREE.MeshNormalMaterial();
    }

    renderer.render(scene, camera);

    scene.overrideMaterial?.dispose?.();
    scene.overrideMaterial = heldOverride;
    scene.background = heldBackground;
    camera.near = heldNear;
    camera.far = heldFar;
    camera.updateProjectionMatrix();
    renderer.toneMapping = heldTone;
}

/**
 * The nearest and furthest a scene reaches along the way the camera is looking.
 *
 * @param {THREE.Object3D} scene - The scene to measure.
 * @param {THREE.Camera} camera - The camera to measure from.
 * @returns {{near: number, far: number}} A range with the subject inside it.
 */
function depthBounds(scene, camera) {
    const box = new THREE.Box3().setFromObject(scene);
    if (box.isEmpty()) return { near: 0.1, far: 100 };

    camera.updateMatrixWorld(true);
    const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion).normalize();
    const at = new THREE.Vector3();
    const corner = new THREE.Vector3();
    let nearest = Infinity;
    let furthest = -Infinity;
    for (let index = 0; index < 8; index += 1) {
        corner.set(
            index & 1 ? box.max.x : box.min.x,
            index & 2 ? box.max.y : box.min.y,
            index & 4 ? box.max.z : box.min.z
        );
        const along = at.copy(corner).sub(camera.position).dot(forward);
        nearest = Math.min(nearest, along);
        furthest = Math.max(furthest, along);
    }
    // A margin either side, and a floor, so nothing lands exactly on the edge of the range.
    const pad = Math.max(0.01, (furthest - nearest) * 0.02);
    const far = Math.max(nearest + 0.02, furthest + pad);
    return { near: Math.max(0, nearest - pad), far };
}

/**
 * A resolver turning descriptors into Three.js objects for one canvas.
 *
 * @param {HTMLCanvasElement} canvas - The surface the scene is drawn on.
 * @param {HTMLElement|null} statusElement - Where status lines are written, or null.
 * @param {object} [options] - `network: true` lets a texture name an `http` or `https`
 *     address. Off by default.
 * @returns {object} The context and the resolve, build and dispose functions.
 */
export function createRuntime(canvas, statusElement, options = {}) {
    const networkAllowed = options?.network === true;
    const cache = {
        texture: new Map(),
        geometry: new Map(),
        material: new Map(),
        module: new Map(),
    };
    const updateFunctions = [];
    const shaderMaterials = new Set();
    const disposableResources = new Set();

    function trackMaterialResources(material) {
        if (!material) return;
        const materials = Array.isArray(material) ? material : [material];
        for (const item of materials) {
            if (!item) continue;
            disposableResources.add(item);
            for (const value of Object.values(item)) {
                if (value?.isTexture) disposableResources.add(value);
            }
        }
    }

    function trackObjectResources(object) {
        if (!object?.traverse) return;
        object.traverse((child) => {
            if (child.geometry?.isBufferGeometry) disposableResources.add(child.geometry);
            trackMaterialResources(child.material);
        });
    }

    const ctx = {
        THREE,
        cache,
        updateFunctions,
        shaderMaterials,
        modules: Object.create(null),
        mixers: [],
        timelineSeconds: 4,
        shadowMapSize: 2048,
        shadowReach: 20,
        tracing: false,
        canvas,
        renderer: null,
        scene: null,
        camera: null,
        controls: null,
        status(message) {
            if (statusElement) statusElement.textContent = String(message);
        },
    };

    async function resolveModule(spec) {
        assertDescriptor(spec, "module");
        if (cache.module.has(spec.id)) return cache.module.get(spec.id);

        const promise = (async () => {
            if (spec.type !== "ScriptModule") {
                throw new Error(`Unsupported module type: ${spec.type}`);
            }
            const name = spec.params?.name || spec.id;
            const source = spec.params?.javascript || "return {};";
            const fn = createAsyncFunction("THREE", "ctx", `"use strict";\n${source}`);
            const exportsValue = await fn(THREE, ctx);
            if (!exportsValue || typeof exportsValue !== "object") {
                throw new Error(`Module "${name}" must return an object of exports.`);
            }
            ctx.modules[name] = exportsValue;
            return exportsValue;
        })();

        cache.module.set(spec.id, promise);
        return promise;
    }

    async function resolveModuleExport(spec, expected) {
        const moduleSpec = spec.deps?.module;
        const moduleExports = await resolveModule(moduleSpec);
        const exportName = spec.params?.exportName;
        if (!Object.prototype.hasOwnProperty.call(moduleExports, exportName)) {
            throw new Error(`Module export "${exportName}" does not exist.`);
        }
        const value = moduleExports[exportName];

        if (expected === "material" && !value?.isMaterial) {
            throw new Error(`Module export "${exportName}" is not a THREE.Material.`);
        }
        if (expected === "geometry" && !value?.isBufferGeometry) {
            throw new Error(`Module export "${exportName}" is not a THREE.BufferGeometry.`);
        }
        if (expected === "object" && !value?.isObject3D) {
            throw new Error(`Module export "${exportName}" is not a THREE.Object3D.`);
        }

        return value;
    }

    async function resolveTexture(spec) {
        assertDescriptor(spec, "texture");
        if (cache.texture.has(spec.id)) return cache.texture.get(spec.id);

        const promise = (async () => {
            if (spec.type !== "TextureURL") {
                throw new Error(`Unsupported texture type: ${spec.type}`);
            }

            const params = spec.params || {};
            const address = String(params.url || "");
            if (!isHeldAddress(address) && !(networkAllowed && isWebAddress(address))) {
                throw new Error(
                    "A texture names an address outside ComfyUI, and this page fetches only "
                        + "pictures ComfyUI holds, so it was not fetched. Use Three Texture From "
                        + "Image, or paste a data:image/png;base64, address into Three Texture URL."
                );
            }
            const loader = new THREE.TextureLoader();
            loader.setCrossOrigin("anonymous");
            const texture = await loader.loadAsync(address);
            texture.colorSpace = colorSpaceConstant(params.colorSpace);
            texture.wrapS = wrapConstant(params.wrapS);
            texture.wrapT = wrapConstant(params.wrapT);
            if (Array.isArray(params.repeat)) texture.repeat.fromArray(params.repeat);
            if (Array.isArray(params.offset)) texture.offset.fromArray(params.offset);
            texture.rotation = Number(params.rotation) || 0;
            texture.flipY = params.flipY !== false;
            texture.anisotropy = Math.max(1, Number(params.anisotropy) || 1);
            texture.needsUpdate = true;
            disposableResources.add(texture);
            return texture;
        })();

        cache.texture.set(spec.id, promise);
        return promise;
    }

    async function resolveGeometry(spec) {
        assertDescriptor(spec, "geometry");
        if (cache.geometry.has(spec.id)) return cache.geometry.get(spec.id);

        const promise = (async () => {
            let geometry;
            if (spec.type === "ModuleExport") {
                geometry = await resolveModuleExport(spec, "geometry");
            } else if (spec.type === "CustomGeometry") {
                const source = spec.params?.javascript || "";
                const fn = createAsyncFunction("THREE", "ctx", `"use strict";\n${source}`);
                geometry = await fn(THREE, ctx);
                if (!geometry?.isBufferGeometry) {
                    throw new Error("Custom geometry code must return THREE.BufferGeometry.");
                }
            } else {
                const GeometryClass = THREE[spec.type];
                if (typeof GeometryClass !== "function") {
                    throw new Error(`THREE.${spec.type} is not available.`);
                }
                const args = Array.isArray(spec.params?.args) ? spec.params.args : [];
                geometry = new GeometryClass(...args);
            }

            disposableResources.add(geometry);
            return geometry;
        })();

        cache.geometry.set(spec.id, promise);
        return promise;
    }

    function resolveUniformValue(definition) {
        if (
            definition === null ||
            typeof definition === "number" ||
            typeof definition === "boolean" ||
            typeof definition === "string"
        ) {
            return definition;
        }

        if (Array.isArray(definition)) return definition;

        const type = definition?.type;
        const value = definition?.value;

        if (type === "color") return new THREE.Color(value ?? "#ffffff");
        if (type === "vec2") return new THREE.Vector2().fromArray(value ?? [0, 0]);
        if (type === "vec3") return new THREE.Vector3().fromArray(value ?? [0, 0, 0]);
        if (type === "vec4") return new THREE.Vector4().fromArray(value ?? [0, 0, 0, 0]);
        if (type === "float" || type === "int") return Number(value ?? 0);
        if (type === "bool") return Boolean(value);
        if (type === "array") return Array.isArray(value) ? value : [];
        return value;
    }

    async function resolveMaterial(spec) {
        assertDescriptor(spec, "material");
        if (cache.material.has(spec.id)) return cache.material.get(spec.id);

        const promise = (async () => {
            let material;

            if (spec.type === "ModuleExport") {
                material = await resolveModuleExport(spec, "material");
            } else if (spec.type === "CustomMaterial") {
                const deps = {};
                for (const [key, value] of Object.entries(spec.deps || {})) {
                    deps[key] = value ? await resolveTexture(value) : null;
                }
                const source = spec.params?.javascript || "";
                const bound = scriptBindings(
                    spec.params,
                    ["texture1", "texture2", "texture3", "texture4"],
                    deps
                );
                const fn = createAsyncFunction(
                    "THREE",
                    "ctx",
                    ...bound.names,
                    `"use strict";\n${source}`
                );
                material = await fn(THREE, ctx, ...bound.args);
                if (!material?.isMaterial) {
                    throw new Error("Custom material code must return THREE.Material.");
                }
            } else if (spec.type === "ShaderMaterial") {
                const params = spec.params || {};
                const uniforms = {};
                for (const [name, definition] of Object.entries(params.uniforms || {})) {
                    uniforms[name] = { value: resolveUniformValue(definition) };
                }
                material = new THREE.ShaderMaterial({
                    uniforms,
                    vertexShader: params.vertexShader || "",
                    fragmentShader: params.fragmentShader || "",
                    transparent: Boolean(params.transparent),
                    depthWrite: params.depthWrite !== false,
                    depthTest: params.depthTest !== false,
                    side: sideConstant(params.side),
                    glslVersion: params.glslVersion === "glsl3" ? THREE.GLSL3 : null,
                });
                shaderMaterials.add(material);
            } else {
                const MaterialClass = THREE[spec.type];
                if (typeof MaterialClass !== "function") {
                    throw new Error(`THREE.${spec.type} is not available.`);
                }

                const params = { ...(spec.params || {}) };
                if ("side" in params) params.side = sideConstant(params.side);
                // A few material settings are vectors rather than numbers, and a bare number
                // reaches the shader as an unusable uniform.
                for (const name of ["normalScale", "clearcoatNormalScale"]) {
                    if (typeof params[name] === "number") {
                        params[name] = new THREE.Vector2(params[name], params[name]);
                    }
                }
                for (const name of ["specularColor", "sheenColor", "attenuationColor"]) {
                    if (typeof params[name] === "string") {
                        params[name] = new THREE.Color(params[name]);
                    }
                }
                // Absorption through glass reaches here as 0 for none, since the distance
                // that means none is infinite and JSON carries no such number.
                if ("attenuationDistance" in params && !(params.attenuationDistance > 0)) {
                    params.attenuationDistance = Infinity;
                }
                material = new MaterialClass(params);

                for (const [property, dependency] of Object.entries(spec.deps || {})) {
                    if (!dependency) continue;
                    material[property] = await resolveTexture(dependency);
                }
                material.needsUpdate = true;
            }

            disposableResources.add(material);
            return material;
        })();

        cache.material.set(spec.id, promise);
        return promise;
    }

    async function resolveDependencyValue(spec) {
        if (!spec) return null;
        if (spec.kind === "texture") return resolveTexture(spec);
        if (spec.kind === "geometry") return resolveGeometry(spec);
        if (spec.kind === "material") return resolveMaterial(spec);
        if (spec.kind === "module") return resolveModule(spec);
        if (spec.kind === "object") return buildObject(spec);
        throw new Error(`Unsupported dependency kind: ${spec.kind}`);
    }

    async function buildObject(spec) {
        assertDescriptor(spec, "object");
        const params = spec.params || {};
        const tag = (object) => {
            if (object) object.userData.wasSpecId = spec.id;
            return object;
        };

        if (spec.type === "ModuleExport") {
            const exported = await resolveModuleExport(spec, "object");
            return tag(exported.clone(true));
        }

        if (spec.type === "Mesh") {
            const geometry = await resolveGeometry(spec.deps.geometry);
            const material = await resolveMaterial(spec.deps.material);
            const mesh = new THREE.Mesh(geometry, material);
            mesh.name = params.name || "Mesh";
            mesh.castShadow = params.castShadow !== false;
            mesh.receiveShadow = params.receiveShadow !== false;
            return tag(mesh);
        }

        if (spec.type === "Group") {
            const group = applyTransform(new THREE.Group(), params);
            for (const childSpec of spec.children || []) {
                group.add(await buildObject(childSpec));
            }
            return tag(group);
        }

        if (spec.type === "AnimatedGroup") {
            const group = new THREE.Group();
            for (const childSpec of spec.children || []) {
                group.add(await buildObject(childSpec));
            }
            const baseY = group.position.y;
            const baseScale = group.scale.clone();
            const rotate = Array.isArray(params.rotate) ? params.rotate : [0, 0, 0];
            const phase = Number(params.phase) || 0;
            const units = params.units || "per second";
            const spread = units === "per capture" || units === "per timeline";
            const baseRotation = group.rotation.clone();
            updateFunctions.push(({ time, delta, ctx: runtimeCtx }) => {
                const bobAmplitude = Number(params.bobAmplitude) || 0;
                const bobFrequency = Number(params.bobFrequency) || 0;
                const pulseAmplitude = Number(params.pulseAmplitude) || 0;
                const pulseFrequency = Number(params.pulseFrequency) || 0;

                // A spread unit fixes each pose by the moment alone, so no step can drift.
                const along = alongFor(units, time, runtimeCtx);
                if (spread) {
                    group.rotation.x = baseRotation.x + (Number(rotate[0]) || 0) * along;
                    group.rotation.y = baseRotation.y + (Number(rotate[1]) || 0) * along;
                    group.rotation.z = baseRotation.z + (Number(rotate[2]) || 0) * along;
                } else {
                    group.rotation.x += (Number(rotate[0]) || 0) * delta;
                    group.rotation.y += (Number(rotate[1]) || 0) * delta;
                    group.rotation.z += (Number(rotate[2]) || 0) * delta;
                }

                group.position.y =
                    baseY + Math.sin(along * bobFrequency * Math.PI * 2 + phase) * bobAmplitude;
                const scale =
                    1 + Math.sin(along * pulseFrequency * Math.PI * 2 + phase) * pulseAmplitude;
                group.scale.copy(baseScale).multiplyScalar(scale);
            });
            return tag(group);
        }

        if (spec.type === "ClipGroup") {
            const group = new THREE.Group();
            for (const childSpec of spec.children || []) {
                group.add(await buildObject(childSpec));
            }

            // The clips ride on whichever object in the subtree the loader left them on.
            let owner = null;
            group.traverse((child) => {
                if (!owner && Array.isArray(child.animations) && child.animations.length) {
                    owner = child;
                }
            });
            if (!owner) {
                throw new Error(
                    "Three Play Animation was given an object carrying no animation. A .glb or "
                        + ".gltf saved with a clip carries one; an .obj or .stl never does."
                );
            }

            const clips = owner.animations;
            const wanted = String(params.clip ?? "").trim();
            let clip = null;
            if (!wanted) {
                clip = clips[0];
            } else if (/^\d+$/.test(wanted)) {
                clip = clips[Number(wanted)] ?? null;
            } else {
                clip = THREE.AnimationClip.findByName(clips, wanted);
            }
            if (!clip) {
                const names = clips.map((one, index) => `${index}: ${one.name}`).join(", ");
                throw new Error(
                    `Three Play Animation found no clip called "${wanted}". This model holds ${names}.`
                );
            }

            const rootMotion = params.root_motion || "as authored";
            const motion = rootMotion === "as authored" ? null : groundTravel(owner, clip);
            const flatFeet = params.feet === "flat when planted";
            if ((motion && rootMotion === "in place") || flatFeet) clip = clip.clone();
            if (motion && rootMotion === "in place") holdInPlace(clip, motion);
            if (flatFeet) plantFeet(owner, clip);
            const passShift = new THREE.Vector3();

            const mixer = new THREE.AnimationMixer(owner);
            const action = mixer.clipAction(clip);
            const modes = {
                repeat: THREE.LoopRepeat,
                once: THREE.LoopOnce,
                "ping pong": THREE.LoopPingPong,
            };
            action.setLoop(modes[params.loop] ?? THREE.LoopRepeat, Infinity);
            action.clampWhenFinished = params.loop === "once";
            action.play();
            ctx.mixers.push(mixer);

            const units = params.units || "per capture";
            const speed = Number(params.speed) || 1;
            const offset = Number(params.offset) || 0;
            const duration = clip.duration || 1;

            updateFunctions.push(({ time, ctx: runtimeCtx }) => {
                // The pose is set from the moment alone, so a frame drawn out of order and a
                // frame drawn twice both land on the same pose.
                const origin = Number(runtimeCtx?.timeOrigin) || 0;
                const along = alongFor(units, time, runtimeCtx);
                const at = units === "per second"
                    ? offset + (time - origin) * speed
                    : offset + along * duration * speed;
                mixer.setTime(Math.max(0, at));
                if (motion && rootMotion === "travel" && params.loop !== "ping pong") {
                    // Each finished pass leaves the model where that pass ended.
                    const passes = params.loop === "once" ? 0 : Math.floor(Math.max(0, at) / duration);
                    if (passes) {
                        const back = new THREE.Matrix3()
                            .setFromMatrix4(parentToRoot(owner, motion.node).invert());
                        passShift.copy(motion.travel).multiplyScalar(passes).applyMatrix3(back);
                        motion.node.position.add(passShift);
                    }
                }
            });
            return tag(group);
        }

        if (spec.type === "FollowGround") {
            const group = new THREE.Group();
            for (const childSpec of spec.children || []) {
                group.add(await buildObject(childSpec));
            }
            const wanted = String(params.groundId || "");
            const offset = Number(params.offset) || 0;
            const footprint = Math.max(0.05, Number(params.footprint) || 0.7);
            const stand = new THREE.Vector3();
            const scale = new THREE.Vector3();
            let map = null;
            let warned = false;

            updateFunctions.push(() => {
                // Where the object stands with no lift, measured fresh each frame.
                group.position.y = 0;
                const base = group.children[0] || group;
                base.updateWorldMatrix(true, false);
                stand.setFromMatrixPosition(base.matrixWorld);

                if (!map) {
                    let root = group;
                    while (root.parent) root = root.parent;
                    let ground = null;
                    root.traverse((child) => {
                        if (!ground && child.userData?.wasSpecId === wanted) ground = child;
                    });
                    if (!ground) {
                        if (root.isScene && !warned) {
                            warned = true;
                            console.warn(
                                "[WAS ThreeJS] Three Follow Ground was given a ground that is not "
                                    + "in the scene. Wire the same object into Three Group or Three "
                                    + "Scene as well."
                            );
                        }
                        return;
                    }
                    // A step above where it was placed: higher ground is wall or roof.
                    map = groundHeights(ground, footprint / 7, stand.y + Math.max(1, footprint * 2));
                }

                const height = heightUnder(map, stand.x, stand.z, footprint);
                if (height === null) return;
                const parentScale = group.parent ? group.parent.getWorldScale(scale).y || 1 : 1;
                group.position.y = (height + offset - stand.y) / parentScale;
            });
            return tag(group);
        }

        if (spec.type === "CustomUpdateGroup") {
            const group = new THREE.Group();
            for (const childSpec of spec.children || []) {
                group.add(await buildObject(childSpec));
            }
            const source = params.javascript || "";
            const fn = new Function(
                "object",
                "time",
                "delta",
                "THREE",
                "ctx",
                `"use strict";\n${source}`
            );
            updateFunctions.push(({ time, delta }) => {
                fn(group, time, delta, THREE, ctx);
            });
            return tag(group);
        }

        if (spec.type === "Light") {
            let light;
            const color = params.color || "#ffffff";
            const intensity = Number(params.intensity) || 0;

            if (params.lightType === "ambient") {
                light = new THREE.AmbientLight(color, intensity);
            } else if (params.lightType === "hemisphere") {
                light = new THREE.HemisphereLight(
                    color,
                    params.groundColor || "#404040",
                    intensity
                );
            } else if (params.lightType === "point") {
                light = new THREE.PointLight(
                    color,
                    intensity,
                    Number(params.distance) || 0,
                    Number(params.decay) || 2
                );
            } else if (params.lightType === "spot") {
                light = new THREE.SpotLight(
                    color,
                    intensity,
                    Number(params.distance) || 0,
                    Number(params.angle) || Math.PI / 4,
                    Number(params.penumbra) || 0,
                    Number(params.decay) || 2
                );
            } else {
                light = new THREE.DirectionalLight(color, intensity);
            }

            if (Array.isArray(params.position)) light.position.fromArray(params.position);
            if ("castShadow" in light) light.castShadow = params.castShadow !== false;
            if (light.shadow) {
                // three's own default is 512 a side, which is coarse enough to read as blobs
                // on anything but a small subject.
                const side = Math.max(256, Math.min(8192, Number(ctx.shadowMapSize) || 2048));
                light.shadow.mapSize.set(side, side);
                light.shadow.bias = -0.0005;
                light.shadow.normalBias = 0.02;
                if (light.shadow.camera?.isOrthographicCamera) {
                    const reach = Math.max(1, Number(ctx.shadowReach) || 20);
                    Object.assign(light.shadow.camera, {
                        left: -reach, right: reach, top: reach, bottom: -reach,
                        near: 0.1, far: reach * 4,
                    });
                    light.shadow.camera.updateProjectionMatrix();
                }
            }
            return tag(light);
        }

        if (spec.type === "AreaLight") {
            // Without the tables a rect area light shades as black, and they are large enough
            // to be worth fetching only for a scene that holds one.
            await loadAreaLightTables();
            const light = new THREE.RectAreaLight(
                params.color || "#ffffff",
                Number(params.intensity) || 0,
                Number(params.width) || 1,
                Number(params.height) || 1
            );
            // A tracer samples this one as a circle. The rasteriser has no such shape and
            // shades the rectangle either way.
            light.isCircular = params.shape === "disc";
            if (Array.isArray(params.position)) light.position.fromArray(params.position);
            if (Array.isArray(params.target)) {
                light.lookAt(new THREE.Vector3().fromArray(params.target));
            }
            return tag(light);
        }

        if (spec.type === "GridHelper") {
            return new THREE.GridHelper(
                Number(params.size) || 10,
                Number(params.divisions) || 10,
                params.centerColor || "#666666",
                params.gridColor || "#333333"
            );
        }

        if (spec.type === "ModelFile") {
            const object = await loadModelFile(params);
            object.name = params.name || "Model";
            const shadows = params.castShadow !== false;
            object.traverse((child) => {
                if (child.isMesh) {
                    child.castShadow = shadows;
                    child.receiveShadow = shadows;
                }
                // A skinned mesh keeps the bounds of its bind pose, so an animated one that
                // moves out of them is culled and vanishes.
                if (child.isSkinnedMesh) child.frustumCulled = false;
            });
            placeModel(object, params);
            adjustSurfaces(object, Number(params.metalness ?? 1), Number(params.minRoughness ?? 0));
            // Wrapped, so centring and scaling survive being placed by a transform above it.
            const holder = new THREE.Group();
            holder.name = object.name;
            holder.add(object);
            return tag(holder);
        }

        if (spec.type === "CustomObject") {
            const resolved = {};
            for (const [name, dep] of Object.entries(spec.deps || {})) {
                resolved[name] = await resolveDependencyValue(dep);
            }
            const source = params.javascript || "";
            const fn = createAsyncFunction(
                "THREE",
                "ctx",
                "geometry1",
                "geometry2",
                "material1",
                "material2",
                "object1",
                "object2",
                `"use strict";\n${source}`
            );
            const object = await fn(
                THREE,
                ctx,
                resolved.geometry1 ?? null,
                resolved.geometry2 ?? null,
                resolved.material1 ?? null,
                resolved.material2 ?? null,
                resolved.object1 ?? null,
                resolved.object2 ?? null
            );
            if (!object?.isObject3D) {
                throw new Error("Custom object code must return THREE.Object3D.");
            }
            return tag(object);
        }

        throw new Error(`Unsupported object type: ${spec.type}`);
    }

    async function buildScene(spec) {
        assertDescriptor(spec, "scene");
        const params = spec.params || {};
        const scene = new THREE.Scene();

        if (params.backgroundMode === "color") {
            scene.background = new THREE.Color(params.background || "#111111");
        } else {
            scene.background = null;
        }

        if (params.fogEnabled) {
            scene.fog = new THREE.Fog(
                params.fogColor || "#111111",
                Number(params.fogNear) || 0,
                Number(params.fogFar) || 100
            );
        }

        if (spec.deps?.environment) {
            await applyEnvironment(scene, spec.deps.environment);
        }

        if (spec.deps?.root) {
            scene.add(await buildObject(spec.deps.root));
        }
        applyObjectOverrides(scene, params.overrides);

        trackObjectResources(scene);
        return scene;
    }

    /**
     * Attach any tracking a camera carries, once the scene it looks at exists.
     *
     * @param {THREE.Camera} camera - The built camera.
     * @param {object} spec - The camera descriptor.
     * @param {THREE.Object3D} scene - The built scene.
     * @returns {void}
     */
    function attachCameraTrack(camera, spec, scene) {
        if (spec?.deps?.track) applyTrack(camera, spec.deps.track, scene);
    }

    /**
     * Light a scene from all around, from a room, a picture or a high dynamic range file.
     *
     * @param {THREE.Scene} scene - The scene to light.
     * @param {object} spec - The environment descriptor.
     * @returns {Promise<void>} Nothing. The scene is changed in place.
     */
    async function applyEnvironment(scene, spec) {
        assertDescriptor(spec, "environment");
        const params = spec.params || {};
        const source = params.source || "none";
        if (source === "none") return;
        if (!ctx.renderer) {
            throw new Error("Three Environment needs a renderer, which is built before the scene.");
        }

        // The room is a scene of glowing panels; every other source arrives as one wide picture.
        let room = null;
        let picture = null;
        if (source === "studio room") {
            const { RoomEnvironment } = await import("../vendor/three/environments/RoomEnvironment.js");
            room = new RoomEnvironment();
        } else {
            const address = String(params.url || "");
            if (!address) throw new Error("Three Environment gave no address to read from.");
            if (!isHeldAddress(address)) {
                throw new Error(
                    "Three Environment names an address outside ComfyUI, so it was not "
                        + "fetched. Pick the file or wire the image again and queue the graph."
                );
            }
            const format = String(params.format || "png").toLowerCase();
            if (format === "hdr") {
                const { HDRLoader } = await import("../vendor/three/loaders/HDRLoader.js");
                picture = await new HDRLoader().loadAsync(address);
            } else if (format === "exr") {
                const { EXRLoader } = await import("../vendor/three/loaders/EXRLoader.js");
                picture = await new EXRLoader().loadAsync(address);
            } else {
                picture = await new THREE.TextureLoader().loadAsync(address);
                picture.colorSpace = THREE.SRGBColorSpace;
            }
            picture.mapping = THREE.EquirectangularReflectionMapping;
        }

        let texture = null;
        if (ctx.tracing) {
            // A traced render samples the surroundings from a map it reads on the processor,
            // which a filtered environment holds nowhere. A cube capture is handed over and
            // unwrapped on the way into the tracer.
            const cube = new THREE.WebGLCubeRenderTarget(ENVIRONMENT_CAPTURE, {
                type: THREE.HalfFloatType,
                generateMipmaps: true,
                minFilter: THREE.LinearMipmapLinearFilter,
            });
            if (room) new THREE.CubeCamera(0.1, 1000, cube).update(ctx.renderer, room);
            else cube.fromEquirectangularTexture(ctx.renderer, picture);
            texture = cube.texture;
            disposableResources.add(cube);
        } else {
            const pmrem = new THREE.PMREMGenerator(ctx.renderer);
            pmrem.compileEquirectangularShader();
            const target = room ? pmrem.fromScene(room, 0.04) : pmrem.fromEquirectangular(picture);
            pmrem.dispose();
            texture = target.texture;
        }

        if (room) {
            room.traverse((child) => {
                if (child.geometry) child.geometry.dispose();
                if (child.material) child.material.dispose();
            });
        }
        picture?.dispose();
        disposableResources.add(texture);
        scene.environment = texture;
        scene.environmentIntensity = Number(params.intensity) ?? 1;
        scene.environmentRotation.y = THREE.MathUtils.degToRad(Number(params.rotation) || 0);
        if (params.asBackground) {
            scene.background = texture;
            scene.backgroundBlurriness = Number(params.backgroundBlur) || 0;
            scene.backgroundRotation.y = scene.environmentRotation.y;
        }
    }

    /**
     * Make a camera follow or aim at an object already in the scene.
     *
     * @param {THREE.Camera} camera - The camera to drive.
     * @param {object} spec - The track descriptor.
     * @param {THREE.Object3D} scene - The built scene the target is found in.
     * @returns {void}
     */
    function applyTrack(camera, spec, scene) {
        assertDescriptor(spec, "track");
        const params = spec.params || {};
        const wanted = String(params.targetId || "");

        let target = null;
        scene.traverse((child) => {
            if (!target && child.userData?.wasSpecId === wanted) target = child;
        });
        if (!target) {
            throw new Error(
                "Three Track was given an object that is not in the scene. Wire the same object "
                    + "into Three Group or Three Scene as well, so there is something to track."
            );
        }

        const mode = params.mode || "aim";
        const aims = mode === "aim" || mode === "aim and follow";
        const follows = mode === "follow" || mode === "aim and follow";
        const offset = Array.isArray(params.offset) ? params.offset : [0, 0, 0];
        const lag = Math.min(0.99, Math.max(0, Number(params.damping) || 0));
        const aimUp = Number(params.aimOffsetY) || 0;

        const rest = camera.position.clone();
        const at = new THREE.Vector3();
        const want = new THREE.Vector3();
        const look = new THREE.Vector3();
        const shift = new THREE.Vector3();
        const turn = new THREE.Quaternion();
        const mount = new THREE.Quaternion();
        const ridesObject = params.offsetSpace === "object";
        let started = false;

        updateFunctions.push(() => {
            target.getWorldPosition(at);
            if (ridesObject) target.getWorldQuaternion(turn);

            if (follows) {
                shift.set(Number(offset[0]) || 0, Number(offset[1]) || 0, Number(offset[2]) || 0);
                if (ridesObject) shift.applyQuaternion(turn);
                want.copy(at).add(shift);
                // The first frame lands on the mark, so a damped camera does not fly in from
                // wherever it happened to start.
                camera.position.lerp(want, started ? 1 - lag : 1);
                if (ridesObject && !aims) {
                    // Mounted on the object: the camera keeps the angle it started at to it.
                    if (!started) mount.copy(turn).invert().multiply(camera.quaternion);
                    camera.quaternion.copy(turn).multiply(mount);
                }
            } else {
                camera.position.copy(rest);
            }

            if (aims) {
                look.copy(at).setY(at.y + aimUp);
                camera.userData.comfyTarget.copy(look);
                camera.lookAt(look);
            }
            started = true;
        });
    }

    function buildCamera(spec, aspect) {
        assertDescriptor(spec, "camera");
        const params = spec.params || {};
        let camera;

        if (spec.type === "OrthographicCamera") {
            const height = Number(params.viewHeight) || 6;
            const width = height * aspect;
            camera = new THREE.OrthographicCamera(
                -width / 2,
                width / 2,
                height / 2,
                -height / 2,
                Number(params.near) || 0.1,
                Number(params.far) || 1000
            );
            camera.userData.comfyViewHeight = height;
        } else {
            camera = new THREE.PerspectiveCamera(
                Number(params.fov) || 50,
                aspect,
                Number(params.near) || 0.1,
                Number(params.far) || 1000
            );
        }

        if (Array.isArray(params.position)) camera.position.fromArray(params.position);
        if (Array.isArray(params.target)) {
            camera.userData.comfyTarget = new THREE.Vector3().fromArray(params.target);
            camera.lookAt(camera.userData.comfyTarget);
        } else {
            camera.userData.comfyTarget = new THREE.Vector3();
        }

        return camera;
    }

    function updateCameraAspect(camera, aspect) {
        if (camera.isPerspectiveCamera) {
            camera.aspect = aspect;
        } else if (camera.isOrthographicCamera) {
            const height = camera.userData.comfyViewHeight || 6;
            const width = height * aspect;
            camera.left = -width / 2;
            camera.right = width / 2;
            camera.top = height / 2;
            camera.bottom = -height / 2;
        }
        camera.updateProjectionMatrix();
    }

    function updateShaderUniforms(time, width, height) {
        // How far through the capture the scene is, on the same window a per-capture animation
        // reads. In the live viewer the window is the app's loop length.
        const span = Number(ctx.duration) || 0;
        const origin = Number(ctx.timeOrigin) || 0;
        const along = span > 0 ? (time - origin) / span : 0;
        const timeline = Number(ctx.timelineSeconds) || 0;
        const round = timeline > 0 ? (time / timeline) % 1 : 0;
        for (const material of shaderMaterials) {
            const uniforms = material.uniforms || {};
            if (uniforms.time) uniforms.time.value = time;
            if (uniforms.uTime) uniforms.uTime.value = time;
            if (uniforms.progress) uniforms.progress.value = along;
            if (uniforms.uProgress) uniforms.uProgress.value = along;
            if (uniforms.timeline) uniforms.timeline.value = round;
            if (uniforms.uTimeline) uniforms.uTimeline.value = round;
            if (uniforms.resolution) {
                const value = uniforms.resolution.value;
                if (value?.isVector2) value.set(width, height);
            }
            if (uniforms.uResolution) {
                const value = uniforms.uResolution.value;
                if (value?.isVector2) value.set(width, height);
            }
        }
    }

    function dispose() {
        for (const resource of disposableResources) {
            try {
                resource.dispose?.();
            } catch (error) {
                console.warn("[WAS ThreeJS] Resource disposal failed.", error);
            }
        }
        disposableResources.clear();
        cache.texture.clear();
        cache.geometry.clear();
        cache.material.clear();
        cache.module.clear();
        updateFunctions.length = 0;
        shaderMaterials.clear();
    }

    return {
        ctx,
        resolveTexture,
        resolveGeometry,
        resolveMaterial,
        resolveModule,
        buildObject,
        buildScene,
        buildCamera,
        renderOverridePass,
        depthBounds,
        attachCameraTrack,
        createComposer,
        updateCameraAspect,
        updateShaderUniforms,
        dispose,
    };
}
