/**
 * Folding a widget away on either node renderer.
 *
 * A canvas node reads `hidden`, a DOM node `options.hidden`. A folded widget's socket leaves the
 * node while nothing is wired to it. `widgets_values` carries every widget, drawn or not.
 */

// Height above its node that a folded widget's unwired socket is moved to.
const OFF_NODE_Y = -1e6;

// Widgets whose `y` is answered by `holdSocketOffNode`.
const HELD = new WeakSet();

/**
 * Whether a widget's own input socket has a link into it.
 *
 * @param {object} widget - The widget.
 * @param {object} [node] - Its node. The widget's own `node` when left out.
 * @returns {boolean} True where a link ends at the widget's socket.
 */
export function isWidgetWired(widget, node = widget?.node) {
  try {
    return node?.getSlotFromWidget?.(widget)?.link != null;
  } catch {
    return false;
  }
}

/**
 * The nearest descriptor of a property along an object's prototype chain.
 *
 * @param {object} target - The object.
 * @param {string} key - The property.
 * @returns {PropertyDescriptor|undefined} The descriptor, or undefined where nothing defines it.
 */
function descriptorOf(target, key) {
  for (let at = target; at; at = Object.getPrototypeOf(at)) {
    const found = Object.getOwnPropertyDescriptor(at, key);
    if (found) return found;
  }
  return undefined;
}

/**
 * Answer a widget's `y`, and so its socket's place, from off the node while it is folded away
 * and its socket is unwired.
 *
 * @param {object} widget - A widget being folded away.
 * @returns {void}
 */
function holdSocketOffNode(widget) {
  if (HELD.has(widget)) return;
  const current = descriptorOf(widget, "y");
  if (current && (current.get || current.set)) return;
  if (current && Object.hasOwn(widget, "y") && !current.configurable) return;

  let laidOut = Number.isFinite(widget.y) && widget.y !== OFF_NODE_Y ? widget.y : 0;
  Object.defineProperty(widget, "y", {
    configurable: true,
    // A copy of the widget starts from the frontend's own default rather than this value.
    enumerable: false,
    get() {
      if (widget.hidden && widget.node && !isWidgetWired(widget)) return OFF_NODE_Y;
      return laidOut;
    },
    set(value) {
      laidOut = value;
    },
  });
  HELD.add(widget);
}

/**
 * Draw one widget or fold it away.
 *
 * @param {object} widget - The widget to set.
 * @param {boolean} hidden - True to fold it away.
 * @returns {boolean} True where this changed what is drawn.
 */
export function setWidgetHidden(widget, hidden) {
  if (!widget) return false;
  const wanted = Boolean(hidden);
  if (wanted) holdSocketOffNode(widget);
  const options = widget.options ?? (widget.options = {});
  if (Boolean(widget.hidden) === wanted && Boolean(options.hidden) === wanted) return false;
  widget.hidden = wanted;
  options.hidden = wanted;
  return true;
}

/**
 * Fold a widget away for the life of the node, as a widget carrying state rather than a control.
 *
 * @param {object} widget - The widget to fold away.
 * @returns {void}
 */
export function keepWidgetHidden(widget) {
  if (!widget) return;
  widget.type = "hidden";
  widget.computeSize = () => [0, -4];
  widget.draw = () => {};
  setWidgetHidden(widget, true);
}
