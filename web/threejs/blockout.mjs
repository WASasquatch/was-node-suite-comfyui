/**
 * Blockout shapes standing on the ground, and preset camera moves laid along a shot.
 */

const FIGURE_HEIGHT = 1.8;

const BACK = 1.70158;

const EASES = {
    linear: (t) => t,
    "ease in": (t) => t * t,
    "ease out": (t) => 1 - (1 - t) * (1 - t),
    "ease in and out": (t) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2),
    "slow in": (t) => t * t * t,
    "slow out": (t) => 1 - Math.pow(1 - t, 3),
    snap: (t) => 1 - Math.pow(1 - t, 6),
    anticipate: (t) => t * t * ((BACK + 1) * t - BACK),
    overshoot: (t) => 1 + (BACK + 1) * Math.pow(t - 1, 3) + BACK * Math.pow(t - 1, 2),
    hold: (t) => (t < 1 ? 0 : 1),
};

/**
 * A point on a smooth curve through four points, at a fraction of the way from the second to the third.
 *
 * @param {Array<number>} a - The point before the segment.
 * @param {Array<number>} b - The segment's start.
 * @param {Array<number>} c - The segment's end.
 * @param {Array<number>} d - The point after the segment.
 * @param {number} t - 0 to 1 along the segment.
 * @returns {Array<number>} The point.
 */
function spline(a, b, c, d, t) {
    const t2 = t * t;
    const t3 = t2 * t;
    return b.map((_, i) => 0.5 * (2 * b[i] + (-a[i] + c[i]) * t + (2 * a[i] - 5 * b[i] + 4 * c[i] - d[i]) * t2
        + (-a[i] + 3 * b[i] - 3 * c[i] + d[i]) * t3));
}

/**
 * A steady wobble read from the moment alone, so a frame drawn twice lands on the same shake.
 *
 * @param {number} seconds - Time into the shot.
 * @param {number} seed - Which of the independent wobbles.
 * @returns {number} About -1 to 1.
 */
function wobble(seconds, seed) {
    // Slow drift with a little tremor on top, as a hand-held operator breathes and steadies.
    const tau = Math.PI * 2;
    return (Math.sin(tau * 0.43 * seconds + seed * 1.7) * 0.45 + Math.sin(tau * 0.91 * seconds + seed * 4.1) * 0.3
        + Math.sin(tau * 1.73 * seconds + seed * 2.9) * 0.17 + Math.sin(tau * 3.1 * seconds + seed * 0.7) * 0.08);
}

/**
 * How far along a walk is, speeding up and slowing down over its first and last third of a second.
 *
 * @param {number} f - 0 to 1 along the step in time.
 * @param {number} seconds - The step's length.
 * @returns {number} 0 to 1 along the step's distance.
 */
function ramped(f, seconds) {
    const r = Math.min(0.35 / Math.max(seconds, 1e-6), 1 / 3);
    const top = 1 / (1 - r);
    if (f < r) return (top * f * f) / (2 * r);
    if (f > 1 - r) return 1 - (top * (1 - f) * (1 - f)) / (2 * r);
    return top * (f - r / 2);
}

/**
 * A side profile extruded across the width, rising toward the back.
 *
 * @param {object} THREE - The three.js namespace.
 * @param {Array<Array<number>>} points - Profile points as [depth, height], front at negative depth.
 * @param {number} width - Extrusion along X.
 * @returns {THREE.BufferGeometry} The solid, centred on X and Z, its base at 0.
 */
function sideProfile(THREE, points, width) {
    const shape = new THREE.Shape(points.map(([s, y]) => new THREE.Vector2(s, y)));
    const geometry = new THREE.ExtrudeGeometry(shape, { depth: width, bevelEnabled: false });
    geometry.rotateY(Math.PI / 2);
    geometry.translate(-width / 2, 0, 0);
    geometry.computeVertexNormals();
    return geometry;
}

/**
 * The geometry of one plain shape, its base on y = 0.
 *
 * @param {object} THREE - The three.js namespace.
 * @param {string} shape - One of the Three Blockout Prop shapes, other than person.
 * @param {Array<number>} size - Width, height and depth.
 * @param {number} steps - Steps in a flight of stairs.
 * @returns {THREE.BufferGeometry} The geometry.
 */
function shapeGeometry(THREE, shape, size, steps) {
    const [w, h, d] = size;
    let geometry;
    if (shape === "ground") {
        geometry = new THREE.PlaneGeometry(w, d);
        geometry.rotateX(-Math.PI / 2);
        return geometry;
    }
    if (shape === "ramp") {
        return sideProfile(THREE, [[-d / 2, 0], [d / 2, 0], [d / 2, h]], w);
    }
    if (shape === "stairs") {
        const count = Math.max(1, Math.floor(steps) || 1);
        const at = (i) => -d / 2 + (i * d) / count;
        const rise = (i) => (i * h) / count;
        // Drawn round the profile: under the flight, up the back, then down each step.
        const points = [[-d / 2, 0], [d / 2, 0], [d / 2, h]];
        for (let i = count - 1; i >= 0; i -= 1) {
            points.push([at(i), rise(i + 1)]);
            if (i > 0) points.push([at(i), rise(i)]);
        }
        return sideProfile(THREE, points, w);
    }
    if (shape === "arch") {
        const post = Math.min(w, h) * 0.2;
        const radius = Math.max(0.001, w / 2 - post);
        const spring = Math.max(0, h - post - radius);
        const outline = new THREE.Shape();
        outline.moveTo(-w / 2, 0);
        outline.lineTo(-radius, 0);
        outline.lineTo(-radius, spring);
        outline.absarc(0, spring, radius, Math.PI, 0, true);
        outline.lineTo(radius, 0);
        outline.lineTo(w / 2, 0);
        outline.lineTo(w / 2, h);
        outline.lineTo(-w / 2, h);
        outline.closePath();
        geometry = new THREE.ExtrudeGeometry(outline, { depth: d, bevelEnabled: false, curveSegments: 24 });
        geometry.translate(0, 0, -d / 2);
        geometry.computeVertexNormals();
        return geometry;
    }
    if (shape === "cylinder") {
        geometry = new THREE.CylinderGeometry(0.5, 0.5, 1, 40);
    } else if (shape === "cone") {
        geometry = new THREE.ConeGeometry(0.5, 1, 40);
    } else if (shape === "sphere") {
        geometry = new THREE.SphereGeometry(0.5, 40, 24);
    } else if (shape === "capsule") {
        const radius = Math.min(w, h) / 2;
        geometry = new THREE.CapsuleGeometry(radius, Math.max(0, h - 2 * radius), 8, 24);
        geometry.scale(1, 1, d / Math.max(1e-6, w));
        geometry.translate(0, h / 2, 0);
        return geometry;
    } else {
        geometry = new THREE.BoxGeometry(1, 1, 1);
    }
    geometry.scale(w, h, d);
    geometry.translate(0, h / 2, 0);
    return geometry;
}

/** Poses a figure takes. */
const POSES = new Set(["stand", "sit", "walk", "run", "point", "wave"]);

/**
 * A figure in a pose, its face toward +Z.
 *
 * @param {object} THREE - The three.js namespace.
 * @param {THREE.Material} material - Its surface.
 * @param {THREE.Material} dark - The eyes.
 * @param {number} height - Its height standing.
 * @param {object} options - `pose`, `seat` (hip height when sitting) and `head` (radians the head turns).
 * @returns {THREE.Group} The figure, its feet on y = 0.
 */
function figure(THREE, material, dark, height, options = {}) {
    const s = height / FIGURE_HEIGHT;
    const pose = POSES.has(options.pose) ? options.pose : "stand";
    const seat = Math.max(0.2, Number(options.seat) || 0.46);
    const group = new THREE.Group();
    const mesh = (geometry, surface = material) => {
        const m = new THREE.Mesh(geometry, surface);
        m.castShadow = true;
        m.receiveShadow = true;
        return m;
    };
    const capsule = (radius, length) => new THREE.CapsuleGeometry(radius * s, Math.max(0, (length - 2 * radius) * s), 6, 16);
    // A limb hangs from its pivot, so a turn of the pivot swings it.
    const limb = (radius, length, x, y, z, swing = 0, spread = 0) => {
        const pivot = new THREE.Group();
        pivot.position.set(x, y, z);
        const part = mesh(capsule(radius, length));
        part.position.y = -length * s / 2;
        pivot.add(part);
        pivot.rotation.x = swing;
        pivot.rotation.z = spread;
        group.add(pivot);
        return pivot;
    };

    const sitting = pose === "sit";
    const hip = sitting ? seat : 0.9 * s;
    const torsoY = hip + 0.27 * s;
    const torso = mesh(capsule(0.17, 0.6));
    torso.geometry.scale(1, 1, 0.62);
    torso.position.set(0, torsoY, 0);
    group.add(torso);

    if (sitting) {
        // Thighs forward along the seat, shins down to the floor.
        for (const x of [-0.1, 0.1]) {
            limb(0.075, 0.46, x * s, hip + 0.02 * s, -0.04 * s, -Math.PI / 2);
            limb(0.07, Math.max(0.2, hip / s), x * s, hip, 0.38 * s, 0);
        }
    } else {
        const stride = pose === "walk" ? 0.38 : pose === "run" ? 0.75 : 0;
        const legs = [limb(0.075, 0.9, -0.1 * s, hip, 0, stride), limb(0.075, 0.9, 0.1 * s, hip, 0, -stride)];
        group.userData.wasLegs = legs;
    }

    const shoulder = torsoY + 0.25 * s;
    const rest = sitting ? -0.55 : 0;
    // The figure faces +Z, so its right hand is on -X.
    const rightSwing = { point: -1.5, wave: -2.7, walk: -0.3, run: -0.7 }[pose] ?? rest;
    const leftSwing = { walk: 0.3, run: 0.7 }[pose] ?? rest;
    const arms = [
        limb(0.05, 0.64, -0.235 * s, shoulder, 0, rightSwing, pose === "wave" ? -0.35 : 0),
        limb(0.05, 0.64, 0.235 * s, shoulder, 0, leftSwing),
    ];
    // Only free arms swing with the stride; a pointing or waving arm holds.
    group.userData.wasArms = pose === "point" || pose === "wave" ? [null, arms[1]] : arms;

    const head = new THREE.Group();
    head.position.set(0, torsoY + 0.36 * s, 0);
    const skull = mesh(new THREE.SphereGeometry(0.11 * s, 24, 16));
    skull.position.y = 0.1 * s;
    head.add(skull);
    const nose = mesh(new THREE.BoxGeometry(0.035 * s, 0.05 * s, 0.06 * s));
    nose.position.set(0, 0.09 * s, 0.12 * s);
    head.add(nose);
    for (const x of [-0.04, 0.04]) {
        const eye = mesh(new THREE.SphereGeometry(0.018 * s, 12, 8), dark);
        eye.position.set(x * s, 0.13 * s, 0.095 * s);
        head.add(eye);
    }
    head.rotation.y = Math.max(-1.4, Math.min(1.4, Number(options.head) || 0));
    group.add(head);
    return group;
}

/**
 * A block figure: a box body and a cube head with eyes, nose and mouth, its face toward +Z.
 *
 * @param {object} THREE - The three.js namespace.
 * @param {THREE.Material} material - Its surface.
 * @param {THREE.Material} dark - The face's features.
 * @param {number} height - Its height standing.
 * @param {object} options - `pose`, `seat` (hip height when sitting) and `head` (radians the head turns).
 * @returns {THREE.Group} The figure, its feet on y = 0.
 */
function blockFigure(THREE, material, dark, height, options = {}) {
    const group = new THREE.Group();
    const box = (w, h, d, x, y, z, surface = material) => {
        const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), surface);
        mesh.position.set(x, y, z);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        group.add(mesh);
        return mesh;
    };
    const side = 0.19 * height;
    const width = 0.26 * height;
    const depth = 0.16 * height;
    const sitting = options.pose === "sit";
    const seat = Math.max(0.2, Number(options.seat) || 0.46);
    const headY = sitting ? seat + 0.406 * height : 0.906 * height;
    if (sitting) {
        const torso = headY - side / 2 - seat - 0.01;
        box(width, torso, depth, 0, seat + torso / 2, -0.02 * height);
        box(width * 0.9, 0.09 * height, 0.27 * height, 0, seat + 0.03 * height, 0.12 * height);
        box(width * 0.85, seat, 0.08 * height, 0, seat / 2, 0.25 * height);
    } else {
        const body = headY - side / 2 - 0.01;
        box(width, body, depth, 0, body / 2, 0);
    }
    const head = new THREE.Group();
    head.position.set(0, headY, 0);
    head.rotation.y = Math.max(-1.4, Math.min(1.4, Number(options.head) || 0));
    const skull = new THREE.Mesh(new THREE.BoxGeometry(side, side, side), material);
    skull.castShadow = true;
    head.add(skull);
    const feature = (w, h, x, y) => {
        const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, 0.01), dark);
        mesh.position.set(x, y, side / 2 + 0.005);
        head.add(mesh);
    };
    feature(side * 0.12, side * 0.12, -side * 0.18, side * 0.12);
    feature(side * 0.12, side * 0.12, side * 0.18, side * 0.12);
    feature(side * 0.1, side * 0.16, 0, -side * 0.04);
    feature(side * 0.3, side * 0.05, 0, -side * 0.24);
    group.add(head);
    return group;
}

/**
 * One Three Blockout Prop.
 *
 * @param {object} THREE - The three.js namespace.
 * @param {object} params - The prop's descriptor params.
 * @returns {{object: THREE.Object3D, resources: Array<object>}} The prop and what it allocated.
 */
export function buildBlockoutProp(THREE, params) {
    const size = Array.isArray(params.size) ? params.size.map((v) => Math.max(1e-3, Number(v) || 1)) : [1, 1, 1];
    const material = new THREE.MeshStandardMaterial({
        color: new THREE.Color(params.colour || "#a0a0a0"),
        roughness: 0.85,
        metalness: 0,
        side: params.shape === "ground" ? THREE.DoubleSide : THREE.FrontSide,
    });
    const resources = [material];
    let object;
    if (params.shape === "person") {
        const dark = new THREE.MeshStandardMaterial({ color: 0x202020, roughness: 0.6, metalness: 0 });
        resources.push(dark);
        const build = params.figure === "block" ? blockFigure : figure;
        object = build(THREE, material, dark, size[1], { pose: params.pose, seat: params.seat, head: params.head });
        object.traverse((child) => child.geometry && resources.push(child.geometry));
    } else {
        const geometry = shapeGeometry(THREE, params.shape, size, params.steps);
        const square = Number(params.checker) || 0;
        if (params.shape === "ground" && square > 0) {
            // A two-tone checkerboard of the ground's colour, one square per `checker` metres.
            const base = new THREE.Color(params.colour || "#a6a6a6");
            const light = base.clone().lerp(new THREE.Color(0xffffff), 0.35);
            const data = new Uint8Array(16);
            [light, base, base, light].forEach((c, i) => {
                const rgb = c.clone().convertLinearToSRGB();
                data.set([rgb.r * 255, rgb.g * 255, rgb.b * 255, 255], i * 4);
            });
            const texture = new THREE.DataTexture(data, 2, 2, THREE.RGBAFormat);
            texture.colorSpace = THREE.SRGBColorSpace;
            texture.magFilter = THREE.NearestFilter;
            texture.minFilter = THREE.LinearMipmapLinearFilter;
            texture.generateMipmaps = true;
            texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
            texture.repeat.set(size[0] / (2 * square), size[2] / (2 * square));
            texture.anisotropy = 8;
            texture.needsUpdate = true;
            material.color.set(0xffffff);
            material.map = texture;
            resources.push(texture);
        }
        // A part placed by its centre rather than its base.
        if (params.pivot === "center" && params.shape !== "ground") geometry.translate(0, -size[1] / 2, 0);
        resources.push(geometry);
        object = new THREE.Mesh(geometry, material);
        object.castShadow = params.shape !== "ground";
        object.receiveShadow = true;
        const rotate = Array.isArray(params.rotate) ? params.rotate.map((v) => Number(v) || 0) : null;
        if (rotate && rotate.some((v) => v !== 0)) {
            object.rotation.set(rotate[0], rotate[1], rotate[2]);
            const holder = new THREE.Group();
            holder.add(object);
            object = holder;
        }
    }
    object.name = `Blockout ${params.shape || "box"}`;
    if (Array.isArray(params.position)) object.position.fromArray(params.position.map((v) => Number(v) || 0));
    object.rotation.y = Number(params.turn) || 0;
    return { object, resources };
}

/**
 * A camera pose carried through a list of moves.
 *
 * @param {object} THREE - The three.js namespace.
 * @param {object} pose - `position` and `target` vectors, `fov` and `roll`, changed in place.
 * @param {Array<object>} moves - The moves, in the order they were chained.
 * @param {function(object): number} reachedOf - How far through its clock a move is, 0 to 1.
 */
function carry(THREE, pose, moves, reachedOf) {
    const up = new THREE.Vector3(0, 1, 0);
    const facing = new THREE.Vector3();
    const right = new THREE.Vector3();
    const offset = new THREE.Vector3();
    const { position, target } = pose;
    for (const move of moves) {
        const reached = reachedOf(move);
        const start = Number(move.start) || 0;
        const end = Number(move.end ?? 1);
        const span = end - start;
        const linear = span > 0 ? (reached - start) / span : reached >= start ? 1 : 0;
        const ease = EASES[move.ease] ?? EASES["ease in and out"];
        const p = ease(Math.min(1, Math.max(0, linear)));
        const amount = Number(move.amount) || 0;

        facing.copy(target).sub(position);
        const distance = facing.length();
        facing.normalize();
        right.crossVectors(facing, up);
        if (right.lengthSq() < 1e-9) right.set(1, 0, 0);
        right.normalize();

        switch (move.move) {
            case "push in":
                position.addScaledVector(facing, Math.min(amount * p, distance - 0.01));
                break;
            case "orbit":
                offset.copy(position).sub(target).applyAxisAngle(up, THREE.MathUtils.degToRad(amount) * p);
                position.copy(target).add(offset);
                position.y += (Number(move.rise) || 0) * p;
                break;
            case "truck":
                position.addScaledVector(right, amount * p);
                target.addScaledVector(right, amount * p);
                break;
            case "crane":
                position.y += amount * p;
                break;
            case "pedestal":
                position.y += amount * p;
                target.y += amount * p;
                break;
            case "pan":
                offset.copy(target).sub(position).applyAxisAngle(up, -THREE.MathUtils.degToRad(amount) * p);
                target.copy(position).add(offset);
                break;
            case "tilt":
                offset.copy(target).sub(position).applyAxisAngle(right, THREE.MathUtils.degToRad(amount) * p);
                target.copy(position).add(offset);
                break;
            case "zoom":
                if (pose.fov > 0 && amount > 0) {
                    const goal = THREE.MathUtils.radToDeg(2 * Math.atan(12 / amount));
                    pose.fov += (goal - pose.fov) * p;
                }
                break;
            case "dolly zoom": {
                const step = Math.min(amount * p, distance - 0.01);
                position.addScaledVector(facing, step);
                if (pose.fov > 0) {
                    const half = Math.tan(THREE.MathUtils.degToRad(pose.fov) / 2) * distance / Math.max(0.01, distance - step);
                    pose.fov = THREE.MathUtils.radToDeg(2 * Math.atan(half));
                }
                break;
            }
            case "dutch":
                pose.roll += THREE.MathUtils.degToRad(amount) * p;
                break;
            default:
                break;
        }
    }
}

/**
 * Put a camera on a pose, with the viewer's orbit controls kept in step.
 *
 * @param {object} THREE - The three.js namespace.
 * @param {THREE.Camera} camera - The camera.
 * @param {object} pose - `position`, `target`, `fov` and `roll`.
 * @param {object} ctx - The runtime context, carrying the controls in a live view.
 */
function settle(THREE, camera, pose, ctx) {
    camera.position.copy(pose.position);
    if (camera.isPerspectiveCamera && pose.fov > 0 && Math.abs(camera.fov - pose.fov) > 1e-6) {
        camera.fov = Math.min(179, Math.max(0.1, pose.fov));
        camera.updateProjectionMatrix();
    }
    camera.up.set(0, 1, 0);
    camera.lookAt(pose.target);
    if (pose.roll) camera.rotateZ(-pose.roll);
    camera.userData.comfyTarget?.copy(pose.target);
    const controls = ctx?.controls;
    if (controls && !controls.dragging) {
        controls.target?.copy(pose.target);
        controls.sync?.();
    }
}

/**
 * The update that carries a camera through its moves, read from the moment alone.
 *
 * @param {object} THREE - The three.js namespace.
 * @param {THREE.Camera} camera - The camera, at its starting pose.
 * @param {Array<object>} moves - The moves, in the order they were chained.
 * @param {function(string, number, object): number} along - How far through a clock a moment is.
 * @returns {function(object): void} The update, for the runtime's update list.
 */
export function cameraMoves(THREE, camera, moves, along) {
    const startPosition = camera.position.clone();
    const startTarget = (camera.userData.comfyTarget ?? new THREE.Vector3()).clone();
    const startFov = camera.isPerspectiveCamera ? camera.fov : 0;
    const pose = { position: new THREE.Vector3(), target: new THREE.Vector3(), fov: 0, roll: 0 };
    camera.userData.wasMoves = true;
    return ({ time, ctx }) => {
        pose.position.copy(startPosition);
        pose.target.copy(startTarget);
        pose.fov = startFov;
        pose.roll = 0;
        carry(THREE, pose, moves, (move) => {
            const reached = along(move.units || "per capture", time, ctx);
            return reached > 1 ? reached % 1 : reached;
        });
        settle(THREE, camera, pose, ctx);
    };
}

/**
 * The update that cuts a camera between shots across the capture, each carried through its own moves.
 *
 * @param {object} THREE - The three.js namespace.
 * @param {THREE.Camera} camera - The camera.
 * @param {Array<object>} shots - `start` and `end` as fractions of the capture, `position`, `target`, `fov`, `moves`.
 * @param {function(string, number, object): number} along - How far through a clock a moment is.
 * @returns {function(object): void} The update, for the runtime's update list.
 */
export function cameraShots(THREE, camera, shots, along, scene = null) {
    const list = shots.filter((shot) => Array.isArray(shot?.position) && Array.isArray(shot?.target));
    const pose = { position: new THREE.Vector3(), target: new THREE.Vector3(), fov: 0, roll: 0 };
    const found = new Map();
    const named = (name) => {
        if (!scene || typeof name !== "string") return null;
        if (!found.has(name)) found.set(name, scene.getObjectByName(name) ?? null);
        return found.get(name);
    };
    const point = new THREE.Vector3();
    const origin = new THREE.Vector3();
    camera.userData.wasMoves = true;
    return ({ time, ctx }) => {
        if (!list.length) return;
        let reached = along("per capture", time, ctx);
        if (reached > 1) reached %= 1;
        let index = list.findIndex((shot) => reached < Number(shot.end));
        if (index < 0) index = list.length - 1;
        const shot = list[index];
        const start = Number(shot.start) || 0;
        const span = Math.max(1e-6, (Number(shot.end) || 1) - start);
        const local = Math.min(1, Math.max(0, (reached - start) / span));
        pose.position.fromArray(shot.position.map(Number));
        pose.target.fromArray(shot.target.map(Number));
        pose.fov = camera.isPerspectiveCamera ? Number(shot.fov) || camera.fov : 0;
        pose.roll = 0;
        const path = Array.isArray(shot.path) ? shot.path : null;
        if (path && path.length > 1) {
            const ease = EASES[path[0].ease] ?? EASES["ease in and out"];
            const along = Math.min(1, Math.max(0, ease(local)));
            let i = path.findIndex((key) => along <= Number(key.t));
            if (i <= 0) i = i < 0 ? path.length - 1 : 1;
            const a = path[i - 1];
            const b = path[i];
            const span = Math.max(1e-6, Number(b.t) - Number(a.t));
            const f = Math.min(1, Math.max(0, (along - Number(a.t)) / span));
            const before = path[Math.max(0, i - 2)];
            const after = path[Math.min(path.length - 1, i + 1)];
            pose.position.fromArray(spline(before.p, a.p, b.p, after.p, f));
            pose.target.fromArray(spline(before.target, a.target, b.target, after.target, f));
            if (pose.fov > 0) pose.fov = Number(a.fov) + (Number(b.fov) - Number(a.fov)) * f;
            pose.roll = Number(a.roll || 0) + (Number(b.roll || 0) - Number(a.roll || 0)) * f;
        }
        const rider = named(shot.ride?.name);
        if (rider && Array.isArray(shot.ride.origin)) {
            rider.updateWorldMatrix(true, false);
            rider.getWorldPosition(point).sub(origin.fromArray(shot.ride.origin.map(Number)));
            pose.position.add(point);
            pose.target.add(point);
        }
        const aimed = named(shot.aim?.name);
        if (aimed && Array.isArray(shot.aim.offset)) {
            aimed.updateWorldMatrix(true, false);
            pose.target.copy(aimed.localToWorld(point.fromArray(shot.aim.offset.map(Number))));
        }
        carry(THREE, pose, Array.isArray(shot.moves) ? shot.moves : [], () => local);
        const shake = shot.shake;
        if (shake && Number(shake.move) > 0) {
            const seconds = local * (Number(shot.seconds) || 1);
            // An impact hits at the cut and settles; the others keep wobbling.
            const scale = shake.style === "impact" ? Math.exp(-seconds * 4) : 1;
            const move = Number(shake.move) * scale;
            const turn = THREE.MathUtils.degToRad(Number(shake.turn) || 0) * scale;
            const reach = pose.position.distanceTo(pose.target);
            pose.position.x += wobble(seconds, 1) * move;
            pose.position.y += wobble(seconds, 2) * move;
            pose.position.z += wobble(seconds, 3) * move;
            pose.target.x += wobble(seconds, 4) * turn * reach;
            pose.target.y += wobble(seconds, 5) * turn * reach;
            pose.roll += wobble(seconds, 6) * turn * 0.5;
        }
        settle(THREE, camera, pose, ctx);
    };
}

/**
 * The update that carries a blockout body along its keyed path, its legs swinging while a person walks.
 *
 * @param {THREE.Object3D} body - The body's group.
 * @param {object} params - `path` keys of `t` (fraction of the capture), `p` and `turn`; `walk`; `ease`.
 * @param {function(string, number, object): number} along - How far through a clock a moment is.
 * @returns {function(object): void} The update, for the runtime's update list.
 */
export function bodyPath(body, params, along) {
    const keys = (params.path || [])
        .filter((key) => Array.isArray(key?.p) && key.p.length === 3)
        .map((key) => ({ t: Number(key.t) || 0, p: key.p.map(Number), turn: Number(key.turn) || 0 }));
    let figure = null;
    body.traverse((child) => {
        if (!figure && child.userData?.wasLegs) figure = child;
    });
    // Distance walked up to each key, so a stride is set by the moment alone.
    const walked = [0];
    for (let i = 1; i < keys.length; i++) {
        const [a, b] = [keys[i - 1].p, keys[i].p];
        walked.push(walked[i - 1] + Math.hypot(b[0] - a[0], b[2] - a[2]));
    }
    const seconds = Math.max(0.1, Number(params.seconds) || 1);
    return ({ time, ctx }) => {
        if (keys.length < 2) return;
        let reached = along("per capture", time, ctx);
        if (reached > 1) reached %= 1;
        let i = keys.findIndex((key) => reached <= key.t);
        let p;
        let turn;
        let distance;
        let moving = false;
        let speed = 0;
        if (i <= 0) {
            ({ p, turn } = keys[i < 0 ? keys.length - 1 : 0]);
            distance = walked[i < 0 ? keys.length - 1 : 0];
        } else {
            const a = keys[i - 1];
            const b = keys[i];
            let f = (reached - a.t) / Math.max(1e-9, b.t - a.t);
            if (params.ease) f = f * f * (3 - 2 * f);
            else f = ramped(f, (b.t - a.t) * seconds);
            p = a.p.map((v, k) => v + (b.p[k] - v) * f);
            const delta = ((b.turn - a.turn + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
            turn = a.turn + delta * f;
            const step = Math.hypot(b.p[0] - a.p[0], b.p[2] - a.p[2]);
            moving = step > 1e-4;
            speed = step / Math.max(1e-3, (b.t - a.t) * seconds);
            distance = walked[i - 1] + step * f;
        }
        body.position.set(p[0], p[1], p[2]);
        body.rotation.y = turn;
        if (figure && params.walk) {
            // A faster pace takes longer strides and swings the limbs further, walking into a run.
            const stride = 0.45 + 0.2 * speed;
            const reach = Math.min(1.0, 0.38 + 0.15 * speed);
            const phase = (distance / stride) * Math.PI;
            const swing = moving ? Math.sin(phase) * reach : 0;
            // The body rises over each planted foot.
            if (moving) body.position.y += Math.abs(Math.sin(phase)) * Math.min(0.06, 0.02 + 0.01 * speed);
            const [legRight, legLeft] = figure.userData.wasLegs;
            legRight.rotation.x = swing;
            legLeft.rotation.x = -swing;
            const [armRight, armLeft] = figure.userData.wasArms || [];
            if (armRight) armRight.rotation.x = -swing * 0.7;
            if (armLeft) armLeft.rotation.x = swing * 0.7;
        }
    };
}
