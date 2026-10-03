/**
 * Controls Bar - UI controls for ComfyUI Viewer
 * Creates the toolbar with Edit, Clear, Fullscreen, Download buttons
 */

import { computeThemeTokens } from "../utils/theme.js";
import { createZipBlob } from "../utils/zip.js";
import { detectContentType, isMultiviewContent, getMultiviewOptions, parseMultiviewContent, stripContentMarker, getAllViews, getView, getViewDisplayName } from "../views/view_loader.js";

const CONTROLS_HEIGHT = 32;
const LIST_SEPARATOR = "\n---LIST_SEPARATOR---\n";

/**
 * Create a styled button element
 * @param {object} theme - Theme tokens
 * @param {string} text - Button text
 * @param {string} title - Button tooltip
 * @param {string} [extraStyles] - Additional CSS
 * @returns {HTMLButtonElement}
 */
function createButton(theme, text, title, extraStyles = "") {
  const btn = document.createElement("button");
  btn.textContent = text;
  btn.title = title;
  btn.style.cssText = `
    padding: 4px 12px;
    font-size: 11px;
    border: 1px solid ${theme.border};
    border-radius: 4px;
    background: ${theme.bg};
    color: ${theme.fg};
    cursor: pointer;
    font-family: sans-serif;
    ${extraStyles}
  `;
  btn.onmouseenter = () => { btn.style.background = theme.accent; btn.style.color = "#fff"; };
  btn.onmouseleave = () => { btn.style.background = theme.bg; btn.style.color = theme.fg; };
  return btn;
}

/**
 * Create view selector dropdown for multi-view content
 * @param {object} theme - Theme tokens
 * @param {object} node - ComfyUI node
 * @param {object} elements - DOM elements container
 * @param {object} callbacks - Callback functions
 * @returns {HTMLSelectElement}
 */
function createViewSelector(theme, node, elements, callbacks) {
  const { onViewChange } = callbacks;
  
  const select = document.createElement("select");
  select.style.cssText = `
    padding: 2px 6px;
    font-size: 11px;
    border: 1px solid ${theme.border};
    border-radius: 4px;
    background: ${theme.bg};
    color: ${theme.fg};
    cursor: pointer;
    font-family: sans-serif;
    display: none;
    margin-left: 8px;
  `;
  select.title = "Switch View";
  
  select.onchange = () => {
    const selectedView = select.value;
    if (onViewChange) {
      onViewChange(node, elements, selectedView);
    }
  };
  
  return select;
}

/**
 * Update view selector with available views
 * @param {HTMLSelectElement} selector - The select element
 * @param {string} content - Content to check for multi-view
 * @param {string} currentView - Currently selected view name
 */
export async function updateViewSelector(selector, content, currentView) {
  if (!selector) return;
  
  // For multi-view content, show the views the content itself declares.
  const multiviewOptions = isMultiviewContent(content) ? getMultiviewOptions(content) : [];
  if (multiviewOptions.length > 1) {
    const options = multiviewOptions;

    // Build multi-view options
    selector.innerHTML = "";
    for (const opt of options) {
      const option = document.createElement("option");
      option.value = opt.name;
      option.textContent = opt.displayName;
      if (opt.name === currentView) {
        option.selected = true;
      }
      selector.appendChild(option);
    }
    
    selector.style.display = "inline-block";
    return;
  }
  
  // Every other content offers Auto plus every registered view, so the view drawn is always
  // a choice rather than only a detection.
  const { loadAllViews } = await import("../views/view_loader.js");
  await loadAllViews();

  const allViews = getAllViews();

  if (allViews.size === 0) {
    selector.style.display = "none";
    selector.innerHTML = "";
    return;
  }

  selector.innerHTML = "";

  const autoOption = document.createElement("option");
  autoOption.value = "";
  autoOption.textContent = "Auto";
  if (!currentView) {
    autoOption.selected = true;
  }
  selector.appendChild(autoOption);

  for (const [id, view] of allViews) {
    const option = document.createElement("option");
    option.value = id;
    option.textContent = view.displayName || id;
    if (id === currentView) {
      option.selected = true;
    }
    selector.appendChild(option);
  }

  selector.style.display = "inline-block";
}

/**
 * Name the view on screen on the bar's type label.
 *
 * @param {object} elements - The bar's elements.
 * @param {string} contentType - The view the content was drawn with.
 * @returns {void}
 */
export function updateTypeLabel(elements, contentType) {
  if (!elements?.typeLabel) return;
  const selector = elements.viewSelector;
  const shown = selector?.value ? selector.selectedOptions?.[0]?.textContent : "";
  elements.typeLabel.textContent = `Type: ${shown || getViewDisplayName(contentType)}`;
}

/**
 * Create a styled checkbox element
 * @param {object} theme - Theme tokens
 * @param {string} label - Checkbox label
 * @param {string} title - Tooltip
 * @param {boolean} checked - Initial state
 * @returns {HTMLLabelElement}
 */
function createCheckbox(theme, label, title, checked = false) {
  const container = document.createElement("label");
  container.title = title;
  container.style.cssText = `
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 11px;
    color: ${theme.fg};
    cursor: pointer;
    font-family: sans-serif;
    user-select: none;
  `;
  
  const checkbox = document.createElement("input");
  checkbox.type = "checkbox";
  checkbox.checked = checked;
  checkbox.style.cssText = `
    margin: 0;
    cursor: pointer;
    accent-color: ${theme.accent};
  `;
  
  const text = document.createElement("span");
  text.textContent = label;
  
  container.appendChild(checkbox);
  container.appendChild(text);
  container.checkbox = checkbox;
  
  return container;
}

/**
 * Create the controls bar for a viewer node
 * @param {object} node - ComfyUI node
 * @param {object} elements - DOM elements container
 * @param {object} callbacks - Callback functions
 * @param {function} callbacks.getNodeContent - Get content from node
 * @param {function} callbacks.setWidgetValue - Set widget value
 * @param {function} callbacks.updateIframeContent - Update iframe content
 * @param {function} callbacks.buildIframeContent - Build iframe HTML
 * @param {object} callbacks.app - ComfyUI app reference
 * @returns {HTMLElement}
 */
export function createControlsBar(node, elements, callbacks) {
  const { getNodeContent, setWidgetValue, updateIframeContent, buildIframeContent, app } = callbacks;
  const theme = computeThemeTokens();
  
  const controls = document.createElement("div");
  controls.style.cssText = `
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 4px 8px;
    background: ${theme.bg};
    border-bottom: 1px solid ${theme.border};
    border-radius: 8px 8px 0 0;
    pointer-events: auto;
    box-sizing: border-box;
    flex-shrink: 0;
    min-height: ${CONTROLS_HEIGHT}px;
  `;

  const typeLabel = document.createElement("span");
  typeLabel.style.cssText = `
    font-size: 11px;
    color: ${theme.fg};
    opacity: 0.7;
    font-family: sans-serif;
  `;
  typeLabel.textContent = "Type: detecting...";
  controls.appendChild(typeLabel);

  // View selector dropdown for multi-view content
  const viewSelector = createViewSelector(theme, node, elements, callbacks);
  controls.appendChild(viewSelector);
  elements.viewSelector = viewSelector;

  // Spacer to push other buttons to the right
  const spacer = document.createElement("div");
  spacer.style.cssText = `flex: 1;`;
  controls.appendChild(spacer);

  const toggleAllBtn = createToggleAllButton(theme, node, elements, callbacks);
  controls.appendChild(toggleAllBtn);
  elements.toggleAllBtn = toggleAllBtn;

  const hold = createHoldControls(theme, node, elements);
  for (const part of hold) controls.appendChild(part);

  const editBtn = createEditButton(theme, node, elements, callbacks);
  controls.appendChild(editBtn);
  elements.editBtn = editBtn;

  const clearBtn = createClearButton(theme, node, elements, editBtn, callbacks);
  controls.appendChild(clearBtn);
  elements.clearBtn = clearBtn;

  const fullscreenBtn = createFullscreenButton(theme, elements);
  controls.appendChild(fullscreenBtn);
  elements.fullscreenBtn = fullscreenBtn;

  const downloadBtn = createDownloadButton(theme, node, elements, callbacks);
  controls.appendChild(downloadBtn);
  elements.downloadBtn = downloadBtn;

  elements.typeLabel = typeLabel;
  return controls;
}

// The route a held run is released through.
const PAUSE_ROUTE = "/was/interface/api/pause";

/**
 * One of the node's hidden state widgets.
 *
 * @param {object} node - The node the widget is on.
 * @param {string} name - The widget's name.
 * @returns {object|null} The widget, or null when the node has no such widget.
 */
function stateWidget(node, name) {
  return (node?.widgets ?? []).find((w) => w?.name === name) ?? null;
}

/**
 * The text the bar is holding.
 *
 * @param {object} node - The node the bar belongs to.
 * @param {object} elements - The bar's elements.
 * @returns {string} The open edit controls' text, the node's stored content otherwise, empty
 *   where there is neither.
 */
function editedText(node, elements) {
  const items = elements.listTextareas ?? [];
  if (items.length > 0) return items.map((area) => area.value).join(LIST_SEPARATOR);
  if (elements.isEditing) return elements.textarea?.value ?? "";
  const stored = stateWidget(node, "manual_content")?.value;
  return typeof stored === "string" ? stored : "";
}

/**
 * What the view on screen is offering a held run.
 *
 * @param {object} node - The node the bar belongs to.
 * @param {object} elements - The bar's elements.
 * @returns {Promise<string>} The canvas composite while a canvas is on screen, the edited
 *   text otherwise.
 */
async function heldValue(node, elements) {
  if (node.__was_viewer_kind === "canvas" && elements.contentType === "canvas") {
    const output = await getView("canvas")?.requestOutput?.(elements.iframe?.contentWindow);
    if (output) return output;
  }
  return editedText(node, elements);
}

/**
 * Hold, the seconds to hold for, and Continue.
 *
 * @param {object} theme - Theme tokens.
 * @param {object} node - The node the bar belongs to.
 * @param {object} elements - The bar's elements, which the controls are recorded on.
 * @returns {HTMLElement[]} The controls, in the order they are drawn.
 */
function createHoldControls(theme, node, elements) {
  const holdBtn = createButton(theme, "Pause Workflow",
    "Stop the run here until Continue is pressed");
  const seconds = document.createElement("input");
  seconds.type = "number";
  seconds.min = "0";
  seconds.step = "10";
  seconds.title = "Seconds before it carries on by itself. 0 waits with no limit";
  seconds.style.cssText = `width:64px;padding:3px 6px;font-size:11px;border-radius:4px;`
    + `border:1px solid ${theme.border};background:${theme.bg};color:${theme.fg};`;
  const continueBtn = createButton(theme, "Continue", "Send the content on and carry on");

  const paint = () => {
    const on = Boolean(stateWidget(node, "hold_for_edit")?.value);
    holdBtn.style.background = on ? theme.accent : theme.bg;
    holdBtn.style.color = on ? "#fff" : theme.fg;
    holdBtn.onmouseleave = () => {
      holdBtn.style.background = on ? theme.accent : theme.bg;
      holdBtn.style.color = on ? "#fff" : theme.fg;
    };
    seconds.style.display = on ? "" : "none";
    seconds.value = String(stateWidget(node, "hold_timeout")?.value ?? 600);
    const held = Boolean(node.__was_viewer_held);
    continueBtn.style.display = held ? "" : "none";
  };

  holdBtn.onclick = () => {
    const w = stateWidget(node, "hold_for_edit");
    if (w) { w.value = !w.value; w.callback?.(w.value); }
    paint();
    node.setDirtyCanvas(true, true);
  };
  seconds.onchange = () => {
    const w = stateWidget(node, "hold_timeout");
    if (w) { w.value = Math.max(0, Number(seconds.value) || 0); w.callback?.(w.value); }
  };
  continueBtn.onclick = async () => {
    continueBtn.disabled = true;
    try {
      const value = await heldValue(node, elements);
      const clientId = String(globalThis.app?.api?.clientId || globalThis.app?.api?.initialClientId || "");
      const answer = await fetch(PAUSE_ROUTE, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ node_id: String(node.id), action: "resume", value, client_id: clientId }),
      });
      if (!answer.ok) {
        const reply = await answer.json().catch(() => ({}));
        console.error("[WAS Viewer] The held run was not continued:", reply?.error ?? answer.status);
      }
    } catch (error) {
      console.error("[WAS Viewer] Failed to continue:", error);
    } finally {
      continueBtn.disabled = false;
    }
  };

  elements.holdBtn = holdBtn;
  elements.holdSeconds = seconds;
  elements.continueBtn = continueBtn;
  elements.paintHold = paint;
  node.__was_paint_hold = paint;
  paint();
  return [holdBtn, seconds, continueBtn];
}

function createToggleAllButton(theme, node, elements, callbacks) {
  const { getNodeContent, updateIframeContent } = callbacks;
  
  const toggleAllBtn = document.createElement("button");
  toggleAllBtn.textContent = "☑";
  toggleAllBtn.title = "Toggle All Checkboxes";
  toggleAllBtn.style.cssText = `
    padding: 0 6px;
    font-size: 14px;
    border: 1px solid ${theme.border};
    border-radius: 4px;
    background: ${theme.bg};
    color: ${theme.fg};
    cursor: pointer;
    font-family: sans-serif;
    display: none;
  `;
  toggleAllBtn.onmouseenter = () => { toggleAllBtn.style.background = theme.accent; toggleAllBtn.style.color = "#fff"; };
  toggleAllBtn.onmouseleave = () => { toggleAllBtn.style.background = theme.bg; toggleAllBtn.style.color = theme.fg; };
  
  toggleAllBtn.onclick = () => {
    const content = getNodeContent(node, elements);
    const metaWidget = node.widgets?.find(w => w.name === "viewer_meta");
    if (!metaWidget) return;
    
    let meta = { lastInputHash: "", excluded: [] };
    try {
      meta = JSON.parse(metaWidget.value || "{}");
      if (!Array.isArray(meta.excluded)) meta.excluded = [];
    } catch {}
    
    const items = content.split(LIST_SEPARATOR);
    const allChecked = meta.excluded.length === 0;
    
    if (allChecked) {
      meta.excluded = items.map((_, i) => i);
      toggleAllBtn.textContent = "☐";
      toggleAllBtn.title = "Check All";
    } else {
      meta.excluded = [];
      toggleAllBtn.textContent = "☑";
      toggleAllBtn.title = "Uncheck All";
    }
    
    metaWidget.value = JSON.stringify(meta);
    elements.lastContentHash = "";
    node.setDirtyCanvas?.(true, true);
    updateIframeContent(node, elements);
  };
  
  return toggleAllBtn;
}

function createEditButton(theme, node, elements, callbacks) {
  const { getNodeContent, setWidgetValue, updateIframeContent, app } = callbacks;
  
  const editBtn = createButton(theme, "Edit", "Edit content");
  
  editBtn.onclick = () => {
    if (elements.isEditing) {
      let newContent = "";
      if (elements.listTextareas && elements.listTextareas.length > 0) {
        const values = elements.listTextareas.map(ta => ta.value);
        newContent = values.join(LIST_SEPARATOR);
        elements.listEditContainer.style.display = "none";
        elements.listTextareas = [];
      } else {
        newContent = elements.textarea.value;
        elements.textarea.style.display = "none";
      }
      setWidgetValue(node, "manual_content", newContent);
      elements.lastContentHash = "";
      elements.isEditing = false;
      elements.iframe.style.display = "block";
      editBtn.textContent = "Edit";
      
      node.setDirtyCanvas?.(true, true);
      app.graph?.change?.();
      updateIframeContent(node, elements);
    } else {
      elements.isEditing = true;
      const currentContent = getNodeContent(node, elements);
      
      if (currentContent.includes(LIST_SEPARATOR)) {
        const items = currentContent.split(LIST_SEPARATOR);
        if (!elements.listEditContainer && elements.contentWrapper) {
          elements.listEditContainer = document.createElement("div");
          elements.listEditContainer.style.cssText = `
            position: absolute;
            inset: 0;
            overflow-y: auto;
            padding: 8px;
            display: flex;
            flex-direction: column;
            gap: 8px;
            background: ${theme.bg};
          `;
          elements.contentWrapper.appendChild(elements.listEditContainer);
        }
        if (!elements.listEditContainer) return;
        elements.listEditContainer.innerHTML = "";
        elements.listEditContainer.style.display = "flex";
        elements.listTextareas = [];
        
        items.forEach((item, idx) => {
          const itemContainer = document.createElement("div");
          itemContainer.style.cssText = `
            background: rgba(0,0,0,0.2);
            border: 1px solid ${theme.border};
            border-radius: 6px;
            padding: 8px;
            display: flex;
            flex-direction: column;
            gap: 4px;
          `;
          const label = document.createElement("span");
          label.textContent = `${idx + 1} / ${items.length}`;
          label.style.cssText = `
            background: ${theme.accent};
            color: #fff;
            font-size: 11px;
            font-weight: bold;
            padding: 2px 6px;
            border-radius: 4px;
            align-self: flex-start;
          `;
          const ta = document.createElement("textarea");
          ta.value = item;
          ta.style.cssText = `
            width: 100%;
            min-height: 80px;
            resize: vertical;
            background: rgba(0,0,0,0.3);
            color: ${theme.fg};
            border: 1px solid ${theme.border};
            border-radius: 4px;
            padding: 8px;
            font-family: monospace;
            font-size: 13px;
          `;
          itemContainer.appendChild(label);
          itemContainer.appendChild(ta);
          elements.listEditContainer.appendChild(itemContainer);
          elements.listTextareas.push(ta);
        });
        elements.iframe.style.display = "none";
        elements.textarea.style.display = "none";
      } else {
        elements.textarea.value = currentContent;
        elements.textarea.style.display = "block";
        elements.iframe.style.display = "none";
        if (elements.listEditContainer) elements.listEditContainer.style.display = "none";
      }
      editBtn.textContent = "Save";
      if (elements.listTextareas?.[0]) {
        elements.listTextareas[0].focus();
      } else {
        elements.textarea.focus();
      }
    }
  };
  
  return editBtn;
}

function createClearButton(theme, node, elements, editBtn, callbacks) {
  const { setWidgetValue, buildIframeContent, app } = callbacks;
  
  const clearBtn = createButton(theme, "Clear", "Clear content");
  clearBtn.onmouseenter = () => { clearBtn.style.background = "#c44"; clearBtn.style.color = "#fff"; };
  clearBtn.onmouseleave = () => { clearBtn.style.background = theme.bg; clearBtn.style.color = theme.fg; };
  
  clearBtn.onclick = () => {
    setWidgetValue(node, "manual_content", "");
    const contentWidget = node.widgets?.find((w) => w.name === "content");
    if (contentWidget) contentWidget.value = "";
    elements.lastContentHash = "";
    elements.textarea.value = "";
    if (elements.isEditing) {
      elements.isEditing = false;
      elements.textarea.style.display = "none";
      elements.iframe.style.display = "block";
      editBtn.textContent = "Edit";
    }
    node.setDirtyCanvas?.(true, true);
    app.graph?.change?.();
    
    const currentTheme = computeThemeTokens();
    const emptyHtml = buildIframeContent("<p style='opacity:0.5;text-align:center;margin-top:40px;'>No content. Click Edit to add content or connect a STRING input.</p>", "html", currentTheme);
    elements.iframe.srcdoc = emptyHtml;
    if (elements.typeLabel) elements.typeLabel.textContent = "Type: Text";
  };
  
  return clearBtn;
}

function createFullscreenButton(theme, elements) {
  const fullscreenBtn = document.createElement("button");
  fullscreenBtn.textContent = "⛶";
  fullscreenBtn.title = "Fullscreen";
  fullscreenBtn.style.cssText = `
    padding: 0 4px;
    font-size: 14px;
    border: 1px solid ${theme.border};
    border-radius: 4px;
    background: ${theme.bg};
    color: ${theme.fg};
    cursor: pointer;
    font-family: sans-serif;
  `;
  fullscreenBtn.onmouseenter = () => { fullscreenBtn.style.background = theme.accent; fullscreenBtn.style.color = "#fff"; };
  fullscreenBtn.onmouseleave = () => { fullscreenBtn.style.background = theme.bg; fullscreenBtn.style.color = theme.fg; };
  
  fullscreenBtn.onclick = () => {
    if (elements.iframe) {
      if (elements.iframe.requestFullscreen) {
        elements.iframe.requestFullscreen();
      } else if (elements.iframe.webkitRequestFullscreen) {
        elements.iframe.webkitRequestFullscreen();
      } else if (elements.iframe.mozRequestFullScreen) {
        elements.iframe.mozRequestFullScreen();
      } else if (elements.iframe.msRequestFullscreen) {
        elements.iframe.msRequestFullscreen();
      }
    }
  };
  
  return fullscreenBtn;
}

/**
 * Prepare content for download by stripping markers and ensuring valid format
 * @param {string} content - Raw content that may have markers
 * @returns {{ content: string, ext: string }} - Cleaned content and recommended extension
 */
function prepareContentForDownload(content) {
  if (!content) return { content: "", ext: "txt" };
  
  const { content: stripped, view } = stripContentMarker(content);
  
  // If content had a marker and the view indicates serialized data (like object view),
  // the stripped content should be valid JSON - verify and format it
  if (view) {
    // Object view and similar serialized content types
    if (view.id === "object" || view.id === "canvas") {
      try {
        const parsed = JSON.parse(stripped);
        return { 
          content: JSON.stringify(parsed, null, 2), 
          ext: "json" 
        };
      } catch {
        // If JSON parse fails, return stripped content as-is
        return { content: stripped, ext: "txt" };
      }
    }
  }
  
  // For non-marker content or other types, detect type and return
  const contentType = detectContentType(stripped);
  const extensions = { 
    html: "html", 
    svg: "svg", 
    markdown: "md", 
    python: "py", 
    javascript: "js", 
    css: "css", 
    text: "txt",
    object: "json",
    canvas: "json",
  };
  
  return { 
    content: stripped, 
    ext: extensions[contentType] || "txt" 
  };
}

function createDownloadButton(theme, node, elements, callbacks) {
  const { getNodeContent } = callbacks;
  
  const downloadBtn = document.createElement("button");
  downloadBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" width="16" height="16">
    <path fill="${theme.fg}" d="M5.625 15c0-.4142-.33579-.75-.75-.75s-.75.3358-.75.75h1.5Zm-.75 1h-.75.75Zm14.4-1c0-.4142-.3358-.75-.75-.75s-.75.3358-.75.75h1.5Zm-8.1664.5387c-.2547.3266-.1965.7979.1301 1.0527.3266.2547.7979.1965 1.0527-.1301l-1.1828-.9226Zm5.0828-4.0774c.2547-.3266.1965-.7979-.1301-1.0527-.3266-.2547-.7979-.1965-1.0527.1301l1.1828.9226Zm-5.0828 5c.2548.3266.7261.3848 1.0527.1301.3266-.2548.3848-.7261.1301-1.0527l-1.1828.9226Zm-2.71722-5.9226c-.25476-.3266-.72605-.3848-1.05265-.1301-.32661.2548-.38486.7261-.13011 1.0527l1.18276-.9226ZM10.95 16c0 .4142.3358.75.75.75s.75-.3358.75-.75h-1.5Zm1.5-11c0-.41421-.3358-.75-.75-.75s-.75.33579-.75.75h1.5ZM4.125 15v1h1.5v-1h-1.5Zm0 1c0 2.0531 1.62757 3.75 3.675 3.75v-1.5c-1.18343 0-2.175-.9893-2.175-2.25h-1.5ZM7.8 19.75h7.8v-1.5H7.8v1.5Zm7.8 0c2.0474 0 3.675-1.6969 3.675-3.75h-1.5c0 1.2607-.9916 2.25-2.175 2.25v1.5ZM19.275 16v-1h-1.5v1h1.5Zm-6.9836.4613 3.9-5-1.1828-.9226-3.9 5 1.1828.9226Zm0-.9226-3.90002-5-1.18276.9226 3.89998 5 1.1828-.9226ZM12.45 16V5h-1.5v11h1.5Z"/>
  </svg>`;
  downloadBtn.title = "Download";
  downloadBtn.style.cssText = `
    padding: 2px 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    border: 1px solid ${theme.border};
    border-radius: 4px;
    background: ${theme.bg};
    cursor: pointer;
  `;
  const updateDownloadSvg = (color) => {
    const path = downloadBtn.querySelector("path");
    if (path) path.setAttribute("fill", color);
  };
  downloadBtn.onmouseenter = () => { downloadBtn.style.background = theme.accent; updateDownloadSvg("#fff"); };
  downloadBtn.onmouseleave = () => { downloadBtn.style.background = theme.bg; updateDownloadSvg(theme.fg); };
  
  downloadBtn.onclick = async () => {
    let content = getNodeContent(node, elements);
    if (!content) {
      content = elements.textarea?.value || "";
    }
    if (!content) {
      alert("No content to download");
      return;
    }
    
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    
    try {
      if (content.includes(LIST_SEPARATOR)) {
        const items = content.split(LIST_SEPARATOR);
        const files = items.map((item, idx) => {
          const prepared = prepareContentForDownload(item);
          return { name: `item_${String(idx + 1).padStart(3, "0")}.${prepared.ext}`, content: prepared.content };
        });
        
        const zipBlob = await createZipBlob(files);
        const url = URL.createObjectURL(zipBlob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `content_${timestamp}.zip`;
        a.style.display = "none";
        document.body.appendChild(a);
        a.click();
        setTimeout(() => {
          document.body.removeChild(a);
          URL.revokeObjectURL(url);
        }, 100);
      } else {
        const prepared = prepareContentForDownload(content);
        const filename = `content_${timestamp}.${prepared.ext}`;
        
        const blob = new Blob([prepared.content], { type: "text/plain;charset=utf-8" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = filename;
        a.style.display = "none";
        document.body.appendChild(a);
        a.click();
        setTimeout(() => {
          document.body.removeChild(a);
          URL.revokeObjectURL(url);
        }, 100);
      }
    } catch (e) {
      console.error("[WAS Viewer] download error:", e);
    }
  };
  
  return downloadBtn;
}

/**
 * Update controls visibility based on whether the view is a UI
 * @param {object} elements - DOM elements container with editBtn, clearBtn, downloadBtn
 * @param {boolean} isUI - Whether the current view is a UI
 */
export function updateControlsForUI(elements, isUI) {
  const display = isUI ? "none" : "";
  if (elements.editBtn) elements.editBtn.style.display = display;
  if (elements.clearBtn) elements.clearBtn.style.display = display;
  if (elements.downloadBtn) elements.downloadBtn.style.display = display;
}

export { CONTROLS_HEIGHT };
