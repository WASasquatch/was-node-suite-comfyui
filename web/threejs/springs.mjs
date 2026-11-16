/**
 * VRM spring bones of a loaded model, from its VRMC_springBone extension or a VRM 0.x
 * secondary animation: hair and cloth chains stepped after the pose is set, with inertia,
 * stiffness, gravity, sphere or capsule colliders and an optional center node.
 */

import * as THREE from "../vendor/three/three.module.mjs";

// Length of one simulation step, in seconds.
const STEP = 1 / 60;

// Most steps one update runs.
const MAX_STEPS = 240;

// Seconds simulated at the current pose when time runs backwards or jumps far ahead.
const SETTLE = 1.5;

// A jump forward longer than this, in seconds, is settled instead of simulated.
const LONGEST_JUMP = 1.0;

// Length of the tip added past the last bone of a VRM 0.x chain, in that bone's units.
const TIP = 0.07;

// VRM 0.x bone group settings a file leaves out.
const VRM0_DEFAULTS = { stiffness: 1.0, gravityPower: 0.0, drag: 0.4, hitRadius: 0.02 };

const number = (value, fallback) => (typeof value === "number" && Number.isFinite(value) ? value : fallback);

const vector = (values, fallback) => {
    const source = Array.isArray(values) && values.length === 3 ? values : fallback;
    return new THREE.Vector3(number(source[0], 0), number(source[1], 0), number(source[2], 0));
};

// A VRM 0.x vector, stored on Unity's axes as `{x, y, z}`, on the file's glTF axes.
const unity = (value, fallback) => {
    const source = value && typeof value === "object" && !Array.isArray(value) ? value : fallback;
    return new THREE.Vector3(number(source.x, 0), number(source.y, 0), 0 - number(source.z, 0));
};

/**
 * Read the spring bones a loaded glTF or VRM file carries.
 *
 * @param {object} gltf - What GLTFLoader answered.
 * @returns {Promise<SpringBones|null>} The chains with their nodes resolved, or null for none.
 */
export async function readSprings(gltf) {
    const json = gltf?.parser?.json;
    const count = Array.isArray(json?.nodes) ? json.nodes.length : 0;
    const node = async (index) => (
        Number.isInteger(index) && index >= 0 && index < count ? gltf.parser.getDependency("node", index) : null
    );
    let chains = [];
    if (json?.extensions?.VRMC_springBone) {
        chains = await readSprings1(json.extensions.VRMC_springBone, node);
    } else if (json?.extensions?.VRM?.secondaryAnimation) {
        chains = await readSprings0(json, json.extensions.VRM.secondaryAnimation, node);
    }
    return chains.length ? new SpringBones(chains) : null;
}

/**
 * Chains of a VRMC_springBone extension.
 *
 * @param {object} extension - The extension's JSON.
 * @param {Function} node - Resolves a node index to its Object3D, or null.
 * @returns {Promise<Array<object>>} `{name, joints, colliders, center}` per chain.
 */
async function readSprings1(extension, node) {
    if (!Array.isArray(extension.springs)) return [];
    const colliders = [];
    for (const entry of Array.isArray(extension.colliders) ? extension.colliders : []) {
        const shape = entry?.shape || {};
        const body = shape.sphere || shape.capsule;
        const owner = await node(entry?.node);
        colliders.push(body && owner ? {
            node: owner,
            capsule: Boolean(shape.capsule),
            offset: vector(body.offset, [0, 0, 0]),
            tail: vector(shape.capsule?.tail, [0, 0, 0]),
            radius: Math.max(0, number(body.radius, 0)),
        } : null);
    }
    const groups = (Array.isArray(extension.colliderGroups) ? extension.colliderGroups : [])
        .map((group) => (Array.isArray(group?.colliders) ? group.colliders : [])
            .map((index) => colliders[index])
            .filter(Boolean));
    const chains = [];
    for (const spring of extension.springs) {
        const joints = [];
        for (const joint of Array.isArray(spring?.joints) ? spring.joints : []) {
            const bone = await node(joint?.node);
            if (!bone) break;
            joints.push({
                bone,
                stiffness: Math.max(0, number(joint.stiffness, 1)),
                gravityPower: Math.max(0, number(joint.gravityPower, 0)),
                gravityDir: vector(joint.gravityDir, [0, -1, 0]).normalize(),
                drag: Math.min(1, Math.max(0, number(joint.dragForce, 0.5))),
                hitRadius: Math.max(0, number(joint.hitRadius, 0)),
            });
        }
        if (joints.length < 2) continue;
        const touching = new Set();
        for (const group of Array.isArray(spring.colliderGroups) ? spring.colliderGroups : []) {
            for (const collider of groups[group] || []) touching.add(collider);
        }
        chains.push({
            name: String(spring.name || ""), joints, colliders: [...touching], center: await node(spring.center),
        });
    }
    return chains;
}

/**
 * Node chains under one VRM 0.x spring root; first children continue a chain, other children start one.
 *
 * @param {Array<object>} nodes - The file's glTF nodes.
 * @param {number} root - The root's node index.
 * @param {Set<number>} claimed - Nodes already in a chain, added to in place.
 * @returns {Array<object>} `{nodes, leaf}` per chain: `leaf` is true where it ends on a bone with
 *     no child, false where it ends on a node another chain holds.
 */
function branches(nodes, root, claimed) {
    const isBone = (index) => (
        Number.isInteger(index) && index >= 0 && index < nodes.length
        && nodes[index] && typeof nodes[index] === "object" && !("mesh" in nodes[index])
    );
    const chains = [];
    const stack = isBone(root) ? [root] : [];
    while (stack.length) {
        let index = stack.pop();
        if (claimed.has(index)) continue;
        const chain = [];
        let leaf = false;
        for (;;) {
            claimed.add(index);
            chain.push(index);
            const children = (Array.isArray(nodes[index].children) ? nodes[index].children : []).filter(isBone);
            stack.push(...children.slice(1).reverse());
            if (!children.length) {
                leaf = true;
                break;
            }
            index = children[0];
            if (claimed.has(index)) {
                chain.push(index);
                break;
            }
        }
        chains.push({ nodes: chain, leaf });
    }
    return chains;
}

/**
 * Chains of a VRM 0.x secondary animation, one per branch of each root bone.
 *
 * @param {object} json - The file's glTF JSON.
 * @param {object} secondary - The `secondaryAnimation` JSON.
 * @param {Function} node - Resolves a node index to its Object3D, or null.
 * @returns {Promise<Array<object>>} `{name, joints, colliders, center}` per chain; a chain ending on
 *     a bone with no child ends on a `{tip}` `TIP` past that bone.
 */
async function readSprings0(json, secondary, node) {
    const nodes = Array.isArray(json.nodes) ? json.nodes : [];
    const groups = [];
    for (const group of Array.isArray(secondary.colliderGroups) ? secondary.colliderGroups : []) {
        const owner = await node(group?.node);
        const members = [];
        for (const entry of owner && Array.isArray(group.colliders) ? group.colliders : []) {
            if (!entry || entry.offset == null || entry.radius == null) continue;
            members.push({
                node: owner,
                capsule: false,
                offset: unity(entry.offset, {}),
                tail: new THREE.Vector3(),
                radius: Math.max(0, number(entry.radius, 0)),
            });
        }
        groups.push(members);
    }
    const claimed = new Set();
    const chains = [];
    for (const group of Array.isArray(secondary.boneGroups) ? secondary.boneGroups : []) {
        if (!group || typeof group !== "object") continue;
        // Gravity's length is folded into its power, leaving a unit direction.
        const gravity = unity(group.gravityDir, { x: 0, y: -1, z: 0 });
        const strength = gravity.length();
        const settings = {
            stiffness: Math.max(0, number(group.stiffiness ?? group.stiffness, VRM0_DEFAULTS.stiffness)),
            gravityPower: Math.max(0, number(group.gravityPower, VRM0_DEFAULTS.gravityPower)) * strength,
            drag: Math.min(1, Math.max(0, number(group.dragForce, VRM0_DEFAULTS.drag))),
            hitRadius: Math.max(0, number(group.hitRadius, VRM0_DEFAULTS.hitRadius)),
        };
        const direction = strength > 1e-9 ? gravity.divideScalar(strength) : new THREE.Vector3(0, -1, 0);
        const touching = new Set();
        for (const index of Array.isArray(group.colliderGroups) ? group.colliderGroups : []) {
            for (const collider of groups[index] || []) touching.add(collider);
        }
        const center = await node(group.center);
        for (const root of Array.isArray(group.bones) ? group.bones : []) {
            for (const branch of branches(nodes, root, claimed)) {
                const joints = [];
                for (const index of branch.nodes) {
                    const bone = await node(index);
                    if (!bone) break;
                    joints.push({ bone, ...settings, gravityDir: direction.clone() });
                }
                const last = joints[joints.length - 1]?.bone;
                if (branch.leaf && last && joints.length === branch.nodes.length
                    && last.position.lengthSq() > 1e-18) {
                    joints.push({ tip: last.position.clone().normalize().multiplyScalar(TIP) });
                }
                if (joints.length < 2) continue;
                chains.push({ name: String(group.comment || ""), joints, colliders: [...touching], center });
            }
        }
    }
    return chains;
}

const scratch = {
    position: new THREE.Vector3(),
    scale: new THREE.Vector3(),
    rotation: new THREE.Quaternion(),
    place: new THREE.Vector3(),
    current: new THREE.Vector3(),
    previous: new THREE.Vector3(),
    inertia: new THREE.Vector3(),
    next: new THREE.Vector3(),
    head: new THREE.Vector3(),
    along: new THREE.Vector3(),
    closest: new THREE.Vector3(),
    away: new THREE.Vector3(),
    to: new THREE.Vector3(),
    turn: new THREE.Quaternion(),
    restWorld: new THREE.Matrix4(),
    inverse: new THREE.Matrix4(),
    centerWorld: new THREE.Matrix4(),
    centerInverse: new THREE.Matrix4(),
};

/**
 * The spring chains of one model, stepped after its pose is set each frame.
 */
export class SpringBones {
    /**
     * Record each joint's rest from the pose the bones hold now.
     *
     * @param {Array<object>} chains - `{name, joints, colliders, center}` per chain. A joint is
     *     `{bone, ...settings}`, or `{tip}` for an end with no node: an offset in the joint
     *     before it.
     */
    constructor(chains) {
        this.chains = chains;
        this.started = false;
        this.lastTime = 0;
        for (const chain of chains) {
            chain.center = chain.center || null;
            // The last joint is the tip that the joint before it points at.
            chain.moving = chain.joints.slice(0, -1);
            chain.moving.forEach((joint, index) => {
                const next = chain.joints[index + 1];
                joint.restRotation = joint.bone.quaternion.clone();
                joint.restMatrix = new THREE.Matrix4().compose(
                    joint.bone.position, joint.bone.quaternion, joint.bone.scale
                );
                if (next.tip) {
                    joint.childOffset = next.tip.clone();
                } else if (next.bone.parent === joint.bone) {
                    joint.childOffset = next.bone.position.clone();
                } else {
                    next.bone.updateWorldMatrix(true, false);
                    joint.childOffset = next.bone.getWorldPosition(new THREE.Vector3())
                        .applyMatrix4(joint.bone.matrixWorld.clone().invert());
                }
                if (joint.childOffset.lengthSq() < 1e-12) joint.childOffset.set(0, -0.01, 0);
                joint.axis = joint.childOffset.clone().normalize();
                joint.current = new THREE.Vector3();
                joint.previous = new THREE.Vector3();
            });
        }
    }

    /**
     * Load a chain's center frame into the scratch matrices, identity when it has none.
     *
     * @param {object} chain - The chain.
     */
    frame(chain) {
        if (chain.center) {
            chain.center.updateWorldMatrix(true, false);
            scratch.centerWorld.copy(chain.center.matrixWorld);
            scratch.centerInverse.copy(chain.center.matrixWorld).invert();
        } else {
            scratch.centerWorld.identity();
            scratch.centerInverse.identity();
        }
    }

    /** Put every joint at rest and its tail where rest puts it. */
    rest() {
        for (const chain of this.chains) {
            for (const joint of chain.moving) {
                joint.bone.quaternion.copy(joint.restRotation);
                joint.bone.updateMatrix();
            }
            this.frame(chain);
            for (const joint of chain.moving) {
                joint.bone.updateWorldMatrix(true, false);
                // Tails are kept in the chain's center frame.
                joint.current.copy(joint.childOffset).applyMatrix4(joint.bone.matrixWorld)
                    .applyMatrix4(scratch.centerInverse);
                joint.previous.copy(joint.current);
            }
        }
    }

    /**
     * Push a tail out of every collider of its chain it has entered.
     *
     * @param {THREE.Vector3} tail - The tail, moved in place.
     * @param {object} chain - The chain whose colliders apply.
     * @param {number} hit - The joint's radius in world units.
     */
    collide(tail, chain, hit) {
        for (const collider of chain.colliders) {
            const world = collider.node.matrixWorld;
            scratch.head.copy(collider.offset).applyMatrix4(world);
            scratch.closest.copy(scratch.head);
            if (collider.capsule) {
                scratch.along.copy(collider.tail).applyMatrix4(world).sub(scratch.head);
                const length = scratch.along.lengthSq();
                if (length > 1e-12) {
                    const share = scratch.away.copy(tail).sub(scratch.head).dot(scratch.along) / length;
                    scratch.closest.addScaledVector(scratch.along, Math.min(1, Math.max(0, share)));
                }
            }
            scratch.away.copy(tail).sub(scratch.closest);
            const reach = collider.radius * world.getMaxScaleOnAxis() + hit;
            const distance = scratch.away.length();
            if (distance < reach && distance > 1e-9) {
                tail.copy(scratch.closest).addScaledVector(scratch.away, reach / distance);
            }
        }
    }

    /**
     * Step every chain once.
     *
     * @param {number} delta - Seconds the step covers.
     */
    step(delta) {
        for (const chain of this.chains) {
            this.frame(chain);
            for (const joint of chain.moving) {
                const bone = joint.bone;
                const parent = bone.parent;
                parent.updateWorldMatrix(true, false);
                bone.updateMatrix();
                bone.matrixWorld.multiplyMatrices(parent.matrixWorld, bone.matrix);
                scratch.position.setFromMatrixPosition(bone.matrixWorld);
                parent.matrixWorld.decompose(scratch.place, scratch.rotation, scratch.scale);
                scratch.restWorld.multiplyMatrices(parent.matrixWorld, joint.restMatrix);
                const length = scratch.to.copy(joint.childOffset).applyMatrix4(scratch.restWorld)
                    .sub(scratch.position).length();
                scratch.current.copy(joint.current).applyMatrix4(scratch.centerWorld);
                scratch.previous.copy(joint.previous).applyMatrix4(scratch.centerWorld);
                scratch.inertia.copy(scratch.current).sub(scratch.previous).multiplyScalar(1 - joint.drag);
                scratch.next.copy(scratch.current).add(scratch.inertia)
                    .addScaledVector(scratch.to.normalize(), joint.stiffness * delta)
                    .addScaledVector(joint.gravityDir, joint.gravityPower * delta);
                scratch.next.sub(scratch.position).normalize().multiplyScalar(length).add(scratch.position);
                this.collide(scratch.next, chain, joint.hitRadius * scratch.scale.x);
                scratch.next.sub(scratch.position).normalize().multiplyScalar(length).add(scratch.position);
                joint.previous.copy(joint.current);
                joint.current.copy(scratch.next).applyMatrix4(scratch.centerInverse);
                scratch.inverse.copy(scratch.restWorld).invert();
                scratch.to.copy(scratch.next).applyMatrix4(scratch.inverse).normalize();
                scratch.turn.setFromUnitVectors(joint.axis, scratch.to);
                bone.quaternion.copy(joint.restRotation).multiply(scratch.turn);
                bone.updateMatrix();
                bone.matrixWorld.multiplyMatrices(parent.matrixWorld, bone.matrix);
            }
        }
    }

    /** Let the chains come to rest at the current pose. */
    settle() {
        this.rest();
        const steps = Math.min(MAX_STEPS, Math.ceil(SETTLE / STEP));
        for (let index = 0; index < steps; index += 1) this.step(STEP);
    }

    /**
     * Bring the chains to a moment: stepped on from the last one, or settled at the current pose
     * when time ran backwards or jumped far.
     *
     * @param {number} time - The moment, in seconds.
     */
    update(time) {
        const span = time - this.lastTime;
        if (!this.started || span < 0 || span > LONGEST_JUMP) {
            this.started = true;
            this.settle();
        } else if (span > 0) {
            const steps = Math.min(MAX_STEPS, Math.max(1, Math.ceil(span / STEP)));
            for (let index = 0; index < steps; index += 1) this.step(span / steps);
        }
        this.lastTime = time;
    }
}
