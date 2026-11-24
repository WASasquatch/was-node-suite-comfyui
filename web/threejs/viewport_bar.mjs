/**
 * The Three Viewer's viewport toolbar and overlay, built before three.js is loaded.
 */

// The widget the view is saved in.
export const VIEW_WIDGET = "view";

/**
 * One toolbar button, styled like the viewer's own.
 *
 * @param {string} label - The text on it.
 * @param {string} hint - The tooltip.
 * @returns {HTMLButtonElement} The button.
 */
function toolButton(label, hint) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.title = hint;
    button.style.cssText = "min-height:24px;padding:2px 7px;border-radius:4px;cursor:pointer;"
        + "border:1px solid var(--border-color,#555);background:var(--comfy-input-bg,#222);"
        + "color:var(--input-text,#ddd);font:11px/1.2 sans-serif";
    return button;
}

/**
 * Mark a toggle button pressed or not.
 *
 * @param {HTMLButtonElement} button - The button.
 * @param {boolean} on - Whether it reads as pressed.
 * @returns {void}
 */
export function setPressed(button, on) {
    button.setAttribute("aria-pressed", on ? "true" : "false");
    button.style.background = on ? "var(--p-primary-color,#3b6fd8)" : "var(--comfy-input-bg,#222)";
    button.style.color = on ? "#fff" : "var(--input-text,#ddd)";
}

/**
 * Build the viewport's toolbar row and its overlay.
 *
 * @returns {object} `{row, overlay, gate, buttons, viewSelect}`.
 */
export function createViewportBar() {
    const row = document.createElement("div");
    row.style.cssText = "display:flex;flex-wrap:wrap;gap:4px;align-items:center";

    const buttons = {
        orbit: toolButton("Orbit", "Drag to orbit, wheel to zoom, shift or middle drag to pan (Q)"),
        select: toolButton("Select", "Click an object to pick it; P picks its parent; Esc clears"),
        move: toolButton("Move", "Pick an object and drag the arrows (W)"),
        rotate: toolButton("Rotate", "Pick an object and drag the rings (E)"),
        scale: toolButton("Scale", "Pick an object and drag the handles (R)"),
        frame: toolButton("Frame", "Fit the picked object, or the whole scene, in view (F)"),
        grid: toolButton("Grid", "A floor grid and the three axes (G)"),
        guides: toolButton("Guides", "Rule of thirds lines over the frame"),
        gate: toolButton("Gate", "Darken what falls outside the render's frame, from the Three Render this feeds"),
        override: toolButton("Override", "Use this view and these moves instead of the nodes upstream, here and downstream"),
        capture: toolButton("Use view", "Keep the camera as it is now and turn override on"),
        clear: toolButton("Reset view", "Forget the framed camera and every move, and turn override off"),
    };
    const viewSelect = document.createElement("select");
    viewSelect.title = "Preset viewpoint";
    viewSelect.style.cssText = "min-height:24px;border-radius:4px;border:1px solid var(--border-color,#555);"
        + "background:var(--comfy-input-bg,#222);color:var(--input-text,#ddd);font:11px sans-serif";
    for (const [value, label] of [["", "View"], ["front", "Front"], ["back", "Back"], ["left", "Left"],
        ["right", "Right"], ["top", "Top"], ["camera", "Camera node"]]) {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = label;
        viewSelect.append(option);
    }
    row.append(
        buttons.orbit, buttons.select, buttons.move, buttons.rotate, buttons.scale, viewSelect,
        buttons.frame, buttons.grid, buttons.guides, buttons.gate, buttons.override, buttons.capture,
        buttons.clear,
    );

    const overlay = document.createElement("div");
    overlay.style.cssText = "position:absolute;inset:0;pointer-events:none;overflow:hidden";
    const gate = document.createElement("div");
    gate.style.cssText = "position:absolute;display:none;box-sizing:border-box";
    const thirds = [];
    for (const [side, at] of [["left", "33.333%"], ["left", "66.667%"], ["top", "33.333%"], ["top", "66.667%"]]) {
        const line = document.createElement("div");
        const vertical = side === "left";
        line.style.cssText = `position:absolute;${side}:${at};background:rgba(255,255,255,0.35);`
            + (vertical ? "top:0;bottom:0;width:1px" : "left:0;right:0;height:1px");
        line.style.display = "none";
        gate.append(line);
        thirds.push(line);
    }
    overlay.append(gate);
    return { row, overlay, gate, thirds, buttons, viewSelect };
}

/**
 * Hide the widget the view is saved in, in both node renderers.
 *
 * @param {object} node - A Three Viewer node.
 * @returns {void}
 */
export function hideViewWidget(node) {
    const widget = node?.widgets?.find((candidate) => candidate.name === VIEW_WIDGET);
    if (!widget) return;
    widget.hidden = true;
    widget.options = { ...(widget.options || {}), hidden: true };
}
