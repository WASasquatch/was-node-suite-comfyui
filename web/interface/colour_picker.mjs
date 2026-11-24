/**
 * The native colour picker node interfaces open.
 *
 * One hidden `input[type=color]` is shared by every caller. `openColourPicker` positions it,
 * opens it and reports the chosen colour as an RGB triple.
 */

const LOG_NAME = "WASNodeSuite.ColourPicker";

let colorInput = null;

/**
 * Get the one hidden colour input every picker opens.
 *
 * @returns {HTMLInputElement} The shared colour input.
 */
function getColourInput() {
  if (!colorInput) {
    colorInput = document.createElement("input");
    colorInput.type = "color";
    colorInput.style.cssText = "position:absolute;opacity:0;pointer-events:none;z-index:-999";
    document.body.appendChild(colorInput);
  }
  return colorInput;
}

/**
 * Read a `#rrggbb` colour into an RGB triple.
 *
 * @param {string} hex - Colour as written by a native colour input.
 * @returns {number[]} Three channels, 0 to 255.
 */
function hexToRgb(hex) {
  const text = String(hex ?? "").replace("#", "");
  if (text.length !== 6) return [0, 0, 0];
  return [
    parseInt(text.slice(0, 2), 16) || 0,
    parseInt(text.slice(2, 4), 16) || 0,
    parseInt(text.slice(4, 6), 16) || 0,
  ];
}

/**
 * Format a colour for a native colour input, which takes six digits and no alpha.
 *
 * @param {number[]} rgb - Red, green and blue. Any alpha is left off.
 * @returns {string} A `#rrggbb` colour.
 */
export function inputHex(rgb) {
  const values = Array.isArray(rgb) ? rgb : [0, 0, 0];
  const hex = (value) => {
    const whole = Math.max(0, Math.min(255, Math.round(Number(value) || 0)));
    return whole.toString(16).padStart(2, "0");
  };
  return `#${hex(values[0])}${hex(values[1])}${hex(values[2])}`;
}

/**
 * Open the native colour picker at a point on screen.
 *
 * @param {number} clientX - Horizontal position on screen.
 * @param {number} clientY - Vertical position on screen.
 * @param {number[]} rgb - Colour the picker opens on.
 * @param {(rgb: number[]) => void} onPicked - Called with the chosen colour.
 * @returns {void}
 */
export function openColourPicker(clientX, clientY, rgb, onPicked) {
  const input = getColourInput();
  input.value = inputHex(rgb);
  input.style.left = `${Math.round(clientX)}px`;
  input.style.top = `${Math.round(clientY)}px`;
  // Assigned rather than added, so at most one pick is ever armed.
  input.onchange = () => {
    input.onchange = null;
    try {
      onPicked(hexToRgb(input.value));
    } catch (error) {
      console.error(`[${LOG_NAME}] Failed to apply the picked colour:`, error);
    }
  };
  // The click waits a frame, which is what puts the picker at the pointer.
  requestAnimationFrame(() => input.click());
}
