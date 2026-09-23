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
 * Read one model file into an Object3D.
 *
 * @param {object} params - `{url, format}` from a Three Load Model descriptor.
 * @returns {Promise<object>} The loaded object.
 */
/**
 * A loading manager that answers a model's relative references from the held files.
 *
 * Each request is matched on its file name alone.
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

    if (byName.size) {
        manager.setURLModifier((url) => {
            const asked = String(url || "");
            // An address already pointing at the asset route is the model itself.
            if (asked.includes("key=")) return asked;
            const name = decodeURIComponent(asked.split(/[?#]/)[0].split("/").pop() || "").toLowerCase();
            return byName.get(name) ?? asked;
        });
    }
    return manager;
}

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
        const gltf = await new GLTFLoader(manager).loadAsync(address);
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

export function createRuntime(canvas, statusElement) {
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
            statusElement.textContent = String(message);
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
            const loader = new THREE.TextureLoader();
            loader.setCrossOrigin("anonymous");
            const texture = await loader.loadAsync(params.url);
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
                if (params.centre !== false) {
                const middle = modelCentre(object);
                object.position.sub(middle);
            }
            const factor = Number(params.scale) || 1;
            object.scale.multiplyScalar(factor);
            // Wrapped, so centring and scaling survive being placed by a transform above it.
            const holder = new THREE.Group();
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
        let started = false;

        updateFunctions.push(() => {
            target.getWorldPosition(at);

            if (follows) {
                want.set(at.x + Number(offset[0]) || 0, at.y + Number(offset[1]) || 0,
                         at.z + Number(offset[2]) || 0);
                // The first frame lands on the mark, so a damped camera does not fly in from
                // wherever it happened to start.
                camera.position.lerp(want, started ? 1 - lag : 1);
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
