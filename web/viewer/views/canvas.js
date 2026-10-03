/**
 * Canvas View - Infinite canvas image compositor with layers, transforms, and drawing tools
 */

import { BaseView, escapeHtml } from "./base_view.js";

// How long the parent waits for a frame to answer with what its canvas holds.
const OUTPUT_TIMEOUT_MS = 20000;

class CanvasView extends BaseView {
  static id = "canvas";
  static displayName = "Canvas";
  static isUI = true;
  static priority = 95;

  static CANVAS_MARKER = "$WAS_CANVAS$";
  static OUTPUT_MARKER = "$WAS_CANVAS_OUTPUT$";

  /**
   * Canvas provides its own complete styling - opt out of base iframe styles
   */
  static usesBaseStyles() {
    return false;
  }

  static detect(content) {
    try {
      // Check for marker prefix
      let jsonContent = content;
      if (content.startsWith(this.CANVAS_MARKER)) {
        jsonContent = content.slice(this.CANVAS_MARKER.length);
      }
      const parsed = JSON.parse(jsonContent);
      if (parsed.type === "canvas_composer" && Array.isArray(parsed.images)) {
        return 200;
      }
    } catch {}
    return 0;
  }

  static render(content, theme) {
    let data;
    try {
      // Strip marker prefix if present
      let jsonContent = content;
      if (content.startsWith(this.CANVAS_MARKER)) {
        jsonContent = content.slice(this.CANVAS_MARKER.length);
      }
      data = JSON.parse(jsonContent);
    } catch {
      return `<pre>Invalid canvas data</pre>`;
    }

    // Use single quotes for attribute since JSON contains double quotes
    // Only need to escape single quotes and < > & in the JSON
    const imagesJson = JSON.stringify(data.images || [])
      .replace(/&/g, '&amp;')
      .replace(/'/g, '&#39;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
    
    // Pass saved canvas state if available
    const savedStateJson = data.savedState ? JSON.stringify(data.savedState)
      .replace(/&/g, '&amp;')
      .replace(/'/g, '&#39;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;') : '';
    
    // Pass origin from parent context since iframe can't access it due to cross-origin policy
    const origin = typeof window !== 'undefined' ? window.location.origin : '';

    return `
      <script></script>
      <div id="canvas-composer" data-images='${imagesJson}' data-origin="${origin}" data-saved-state='${savedStateJson}'>
        <div class="canvas-toolbar">
          <div class="toolbar-group tools-group">
            <button class="tool-btn active" data-tool="select" title="Select (V)">↖</button>
            <button class="tool-btn" data-tool="pan" title="Pan (H)">✋</button>
            <button class="tool-btn" data-tool="brush" title="Brush (B)">🖌</button>
          </div>
          <div class="toolbar-separator"></div>
          <div class="toolbar-group bg-group">
            <label>BG:</label>
            <input type="color" id="bg-color" value="#000000" title="Background Color">
            <label class="checkbox-label"><input type="checkbox" id="bg-transparent" checked> Transparent</label>
          </div>
          <div class="toolbar-separator"></div>
          <div class="toolbar-group layer-options">
            <label>Layer Opacity:</label>
            <input type="range" id="layer-opacity" min="0" max="100" value="100">
          </div>
          <div class="toolbar-separator"></div>
          <div class="toolbar-group export-size-group">
            <label>Size:</label>
            <input type="number" id="export-width" min="0" max="8192" value="0" placeholder="W" title="Export Width (0 = auto)">
            <span>×</span>
            <input type="number" id="export-height" min="0" max="8192" value="0" placeholder="H" title="Export Height (0 = auto)">
            <label class="checkbox-label"><input type="checkbox" id="snap-enabled"> Snap</label>
          </div>
          <div class="toolbar-spacer"></div>
          <div class="toolbar-group actions-group">
            <button id="send-output-btn" class="action-btn primary" title="Send to Node Output">📤 Send to Output</button>
            <button id="export-btn" class="action-btn" title="Download PNG">📥 Download</button>
            <button id="reset-view-btn" class="action-btn icon-only" title="Reset View">⟲</button>
          </div>
        </div>
        <div class="brush-toolbar" id="brush-toolbar" style="display:none;">
          <div class="toolbar-group">
            <label>Tool:</label>
            <select id="brush-tool">
              <option value="brush">Brush</option>
              <option value="eraser">Eraser</option>
              <option value="clone">Clone Stamp</option>
            </select>
          </div>
          <div class="toolbar-group">
            <label>Size:</label>
            <input type="range" id="brush-size" min="1" max="100" value="10">
            <span id="brush-size-val">10</span>
          </div>
          <div class="toolbar-group">
            <label>Hardness:</label>
            <input type="range" id="brush-hardness" min="0" max="100" value="100">
          </div>
          <div class="toolbar-group">
            <label>Opacity:</label>
            <input type="range" id="brush-opacity" min="1" max="100" value="100">
            <span id="brush-opacity-val">100%</span>
          </div>
          <div class="toolbar-group">
            <label>Blend:</label>
            <select id="brush-blend">
              <option value="source-over">Normal</option>
              <option value="multiply">Multiply</option>
              <option value="screen">Screen</option>
              <option value="overlay">Overlay</option>
              <option value="hard-light">Hard Light</option>
            </select>
          </div>
          <div class="toolbar-group">
            <input type="color" id="brush-color" value="#ffffff" title="Brush Color">
          </div>
          <div class="toolbar-group clone-options" id="clone-options" style="display:none;">
            <label class="checkbox-label"><input type="checkbox" id="clone-all-layers"> Sample All</label>
            <span id="clone-status" style="opacity:0.6;font-size:10px;">Alt+Click to set source</span>
          </div>
        </div>
        <div class="canvas-workspace">
          <div class="canvas-container">
            <canvas id="main-canvas"></canvas>
          </div>
          <div class="layers-drawer" id="layers-drawer">
            <div class="drawer-toggle" id="drawer-toggle" title="Toggle Layers Panel"></div>
            <div class="drawer-content">
              <div class="drawer-section layer-controls-section">
                <div class="layer-control-row">
                  <label>Opacity:</label>
                  <input type="range" id="drawer-layer-opacity" min="0" max="100" value="100">
                  <span id="drawer-opacity-val">100%</span>
                </div>
                <div class="layer-control-row">
                  <label>Blend:</label>
                  <select id="layer-blend-mode">
                    <option value="source-over">Normal</option>
                    <option value="multiply">Multiply</option>
                    <option value="screen">Screen</option>
                    <option value="overlay">Overlay</option>
                    <option value="darken">Darken</option>
                    <option value="lighten">Lighten</option>
                    <option value="color-dodge">Color Dodge</option>
                    <option value="color-burn">Color Burn</option>
                    <option value="hard-light">Hard Light</option>
                    <option value="soft-light">Soft Light</option>
                    <option value="difference">Difference</option>
                    <option value="exclusion">Exclusion</option>
                    <option value="hue">Hue</option>
                    <option value="saturation">Saturation</option>
                    <option value="color">Color</option>
                    <option value="luminosity">Luminosity</option>
                  </select>
                </div>
              </div>
              <div class="drawer-section">
                <div class="drawer-header">
                  <span>Layers</span>
                  <div class="drawer-header-buttons">
                    <button id="save-state-btn" title="Save Canvas State">💾</button>
                    <button id="load-state-btn" title="Load Canvas State">📂</button>
                    <button id="add-new-layer" title="Add New Drawing Layer">+</button>
                  </div>
                </div>
                <div id="layers-list" class="layers-list"></div>
              </div>
              <div class="drawer-section">
                <div class="drawer-header">
                  <span>Source Images</span>
                  <button id="refresh-images-btn" title="Refresh Images">🔄</button>
                </div>
                <div id="images-list" class="images-list"></div>
              </div>
              <div class="drawer-section mini-map-section">
                <div class="drawer-header">Navigator</div>
                <div class="mini-map-container">
                  <canvas id="mini-map-canvas"></canvas>
                </div>
              </div>
            </div>
          </div>
        </div>
        <div id="layer-effects-modal" class="modal-overlay" style="display:none;">
          <div class="modal-content">
            <div class="modal-header">
              <span id="modal-layer-name">Layer Effects</span>
              <button id="modal-close" class="modal-close-btn">×</button>
            </div>
            <div class="modal-body">
              <div class="effect-section">
                <div class="effect-header">
                  <label class="checkbox-label"><input type="checkbox" id="effect-stroke-enabled"> Stroke</label>
                </div>
                <div class="effect-controls" id="stroke-controls">
                  <div class="effect-row">
                    <label>Width:</label>
                    <input type="range" id="effect-stroke-width" min="1" max="50" value="2">
                    <span id="effect-stroke-width-val">2</span>px
                  </div>
                  <div class="effect-row">
                    <label>Color:</label>
                    <input type="color" id="effect-stroke-color" value="#000000">
                  </div>
                </div>
              </div>
              <div class="effect-section">
                <div class="effect-header">
                  <label class="checkbox-label"><input type="checkbox" id="effect-shadow-enabled"> Shadow</label>
                </div>
                <div class="effect-controls" id="shadow-controls">
                  <div class="effect-row">
                    <label>Blur:</label>
                    <input type="range" id="effect-shadow-blur" min="0" max="50" value="10">
                    <span id="effect-shadow-blur-val">10</span>px
                  </div>
                  <div class="effect-row">
                    <label>Offset X:</label>
                    <input type="range" id="effect-shadow-x" min="-50" max="50" value="5">
                    <span id="effect-shadow-x-val">5</span>px
                  </div>
                  <div class="effect-row">
                    <label>Offset Y:</label>
                    <input type="range" id="effect-shadow-y" min="-50" max="50" value="5">
                    <span id="effect-shadow-y-val">5</span>px
                  </div>
                  <div class="effect-row">
                    <label>Color:</label>
                    <input type="color" id="effect-shadow-color" value="#000000">
                    <label>Opacity:</label>
                    <input type="range" id="effect-shadow-opacity" min="0" max="100" value="50">
                    <span id="effect-shadow-opacity-val">50</span>%
                  </div>
                </div>
              </div>
              <div class="effect-section">
                <div class="effect-header">Transform</div>
                <div class="effect-controls">
                  <div class="effect-row">
                    <label>Rotation:</label>
                    <input type="range" id="effect-rotation" min="-180" max="180" value="0">
                    <span id="effect-rotation-val">0</span>°
                  </div>
                  <div class="effect-row">
                    <label>Scale X:</label>
                    <input type="range" id="effect-scale-x" min="10" max="300" value="100">
                    <span id="effect-scale-x-val">100</span>%
                  </div>
                  <div class="effect-row">
                    <label>Scale Y:</label>
                    <input type="range" id="effect-scale-y" min="10" max="300" value="100">
                    <span id="effect-scale-y-val">100</span>%
                  </div>
                  <div class="effect-row">
                    <label>Skew X:</label>
                    <input type="range" id="effect-skew-x" min="-100" max="100" value="0">
                    <span id="effect-skew-x-val">0</span>
                  </div>
                  <div class="effect-row">
                    <label>Skew Y:</label>
                    <input type="range" id="effect-skew-y" min="-100" max="100" value="0">
                    <span id="effect-skew-y-val">0</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  static getStyles(theme) {
    // Use CSS variables from ComfyUI theme for consistent styling
    return `
      *, *::before, *::after {
        box-sizing: border-box;
      }
      html, body {
        height: 100%;
        margin: 0;
        padding: 0;
        overflow: hidden;
      }
      #canvas-composer {
        display: flex;
        flex-direction: column;
        height: 100%;
        background: var(--theme-bg);
        color: var(--theme-fg);
        font-family: system-ui, sans-serif;
        font-size: 12px;
      }
      .canvas-toolbar {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 12px;
        background: var(--theme-panel-header, var(--theme-bg-dark));
        border-bottom: 1px solid var(--theme-border);
        flex-wrap: wrap;
        min-height: 42px;
      }
      .brush-toolbar {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 6px 12px;
        background: var(--theme-bg);
        border-bottom: 1px solid var(--theme-border);
        min-height: 36px;
      }
      .brush-toolbar .toolbar-group {
        display: flex;
        align-items: center;
        gap: 6px;
      }
      .brush-toolbar label { font-size: 11px; color: var(--theme-fg-muted); white-space: nowrap; }
      .brush-toolbar input[type="range"] { width: 60px; accent-color: var(--theme-accent); }
      .brush-toolbar select { 
        background: var(--theme-input-bg); 
        color: var(--theme-fg); 
        border: 1px solid var(--theme-input-border); 
        border-radius: 4px; 
        padding: 3px 6px;
        font-size: 11px;
      }
      .brush-toolbar select:focus { border-color: var(--theme-input-focus); outline: none; }
      .brush-toolbar span { font-size: 10px; color: var(--theme-fg-muted); min-width: 24px; }
      .brush-toolbar .checkbox-label { cursor: pointer; font-size: 11px; }
      .brush-toolbar .checkbox-label input[type="checkbox"] { accent-color: var(--theme-accent); }
      .toolbar-group {
        display: flex;
        align-items: center;
        gap: 6px;
      }
      .toolbar-separator {
        width: 1px;
        height: 24px;
        background: var(--theme-border);
        margin: 0 4px;
      }
      .toolbar-spacer {
        flex: 1;
      }
      .tool-btn {
        width: 32px;
        height: 32px;
        border: 1px solid var(--theme-input-border);
        background: var(--theme-input-bg);
        color: var(--theme-fg);
        border-radius: 6px;
        cursor: pointer;
        font-size: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.15s ease;
      }
      .tool-btn:hover { background: var(--theme-accent-bg); border-color: var(--theme-accent); }
      .tool-btn.active { background: var(--theme-accent); color: #fff; border-color: var(--theme-accent); box-shadow: 0 2px 4px var(--theme-shadow); }
      .canvas-toolbar input[type="range"] { width: 60px; accent-color: var(--theme-accent); }
      .canvas-toolbar input[type="number"] { width: 50px; background: var(--theme-input-bg); color: var(--theme-fg); border: 1px solid var(--theme-input-border); border-radius: 4px; padding: 4px; text-align: center; }
      .canvas-toolbar input[type="number"]::-webkit-inner-spin-button { opacity: 0; }
      .canvas-toolbar input[type="number"]:focus { border-color: var(--theme-input-focus); outline: none; }
      .export-size-group span { color: var(--theme-fg-muted); font-size: 11px; }
      .canvas-toolbar select, .layer-control-row select { 
        background: var(--theme-input-bg); 
        color: var(--theme-fg); 
        border: 1px solid var(--theme-input-border); 
        border-radius: 4px; 
        padding: 4px 6px;
        transform-origin: top left;
      }
      .canvas-toolbar select:focus, .layer-control-row select:focus { border-color: var(--theme-input-focus); outline: none; }
      .canvas-toolbar label { display: flex; align-items: center; gap: 4px; white-space: nowrap; font-size: 11px; color: var(--theme-fg-muted); }
      .checkbox-label { cursor: pointer; }
      .checkbox-label input[type="checkbox"] { accent-color: var(--theme-accent); }
      .action-btn {
        padding: 6px 12px;
        background: var(--theme-input-bg);
        color: var(--theme-fg);
        border: 1px solid var(--theme-input-border);
        border-radius: 6px;
        cursor: pointer;
        font-size: 12px;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 4px;
        transition: all 0.15s ease;
      }
      .action-btn:hover { background: var(--theme-accent-bg); border-color: var(--theme-accent); }
      .action-btn.primary { background: var(--theme-accent); color: #fff; border-color: var(--theme-accent); }
      .action-btn.primary:hover { background: var(--theme-accent-hover); box-shadow: 0 2px 6px var(--theme-shadow); }
      .action-btn.icon-only { padding: 6px 8px; min-width: 32px; justify-content: center; }
      .canvas-workspace {
        display: flex;
        flex: 1;
        overflow: hidden;
        position: relative;
      }
      .canvas-container {
        flex: 1;
        position: relative;
        overflow: hidden;
        background: repeating-conic-gradient(#333 0% 25%, #444 0% 50%) 50% / 20px 20px;
        min-width: 0;
        min-height: 0;
      }
      #main-canvas {
        display: block;
      }
      .layers-drawer {
        position: absolute;
        right: 0;
        top: 0;
        bottom: 0;
        width: 200px;
        background: var(--theme-panel-bg, var(--theme-bg));
        border-left: 1px solid var(--theme-border);
        display: flex;
        transition: transform 0.2s ease;
        z-index: 100;
      }
      .layers-drawer.collapsed {
        transform: translateX(200px);
      }
      .drawer-toggle {
        position: absolute;
        left: -24px;
        top: 50%;
        transform: translateY(-50%);
        width: 24px;
        height: 48px;
        background: var(--theme-panel-bg, var(--theme-bg));
        border: 1px solid var(--theme-border);
        border-right: none;
        border-radius: 6px 0 0 6px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        color: var(--theme-fg);
      }
      .drawer-toggle:hover { background: var(--theme-accent-bg); }
      .layers-drawer.collapsed .drawer-toggle { 
        transform: translateY(-50%) translateX(-200px);
      }
      .layers-drawer.collapsed .drawer-toggle::after { content: '▶'; }
      .drawer-toggle::after { content: '◀'; }
      .drawer-content {
        flex: 1;
        display: flex;
        flex-direction: column;
        overflow: hidden;
      }
      .drawer-section {
        display: flex;
        flex-direction: column;
        flex: 1;
        min-height: 0;
        border-bottom: 1px solid var(--theme-border);
      }
      .drawer-section:last-child { border-bottom: none; }
      .drawer-section.layer-controls-section {
        flex: 0 0 auto;
        padding: 8px 10px;
        display: flex;
        flex-direction: column;
        gap: 6px;
      }
      .layer-control-row {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 11px;
        position: relative;
      }
      .layer-control-row label {
        color: var(--theme-fg-muted);
        white-space: nowrap;
        min-width: 50px;
      }
      .layer-control-row select {
        flex: 1;
        padding: 4px 6px;
        font-size: 11px;
        background: var(--theme-input-bg);
        color: var(--theme-fg);
        border: 1px solid var(--theme-border);
        border-radius: 3px;
        min-width: 0;
      }
      .layer-control-row input[type="range"] {
        flex: 1;
        min-width: 0;
        accent-color: var(--theme-accent);
      }
      .layer-control-row span {
        font-size: 10px;
        min-width: 32px;
        text-align: right;
      }
      .drawer-header {
        padding: 8px 10px;
        font-weight: bold;
        font-size: 11px;
        text-transform: uppercase;
        border-bottom: 1px solid var(--theme-border);
        background: var(--theme-panel-header, var(--theme-bg-dark));
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-shrink: 0;
      }
      .drawer-header-buttons {
        display: flex;
        gap: 4px;
      }
      .drawer-header button {
        font-size: 10px;
        padding: 2px 6px;
        background: var(--theme-accent);
        color: #fff;
        border: none;
        border-radius: 3px;
        cursor: pointer;
      }
      .drawer-header button:hover { opacity: 0.8; }
      .layers-list, .images-list {
        flex: 1;
        overflow-y: auto;
        padding: 6px;
      }
      .layers-list::-webkit-scrollbar, .images-list::-webkit-scrollbar {
        width: 6px;
      }
      .layers-list::-webkit-scrollbar-thumb, .images-list::-webkit-scrollbar-thumb {
        background: var(--theme-scrollbar-thumb);
        border-radius: 3px;
      }
      .layers-list::-webkit-scrollbar-track, .images-list::-webkit-scrollbar-track {
        background: var(--theme-scrollbar-track);
      }
      .mini-map-section {
        flex: 0 !important;
        min-height: auto !important;
      }
      .mini-map-container {
        padding: 6px;
        background: #000;
      }
      #mini-map-canvas {
        width: 100%;
        height: 80px;
        display: block;
        cursor: crosshair;
      }
      .layer-item, .image-item {
        padding: 6px 8px;
        margin: 2px 0;
        background: var(--theme-bg-light);
        border-radius: 4px;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 11px;
        user-select: none;
      }
      .layer-item:hover, .image-item:hover { background: var(--theme-accent-bg); }
      .layer-item.selected { background: var(--theme-selection); outline: 1px solid var(--theme-accent); }
      .layer-item.drag-over { border-top: 2px solid var(--theme-accent); }
      .layer-item.dragging { opacity: 0.5; }
      .layer-thumb {
        width: 28px;
        height: 28px;
        background-size: contain;
        background-repeat: no-repeat;
        background-position: center;
        background-color: var(--theme-bg);
        border: 1px solid var(--theme-border);
        border-radius: 3px;
        flex-shrink: 0;
      }
      .layer-thumb.draw-layer {
        background: linear-gradient(45deg, #666 25%, transparent 25%), linear-gradient(-45deg, #666 25%, transparent 25%), linear-gradient(45deg, transparent 75%, #666 75%), linear-gradient(-45deg, transparent 75%, #666 75%);
        background-size: 8px 8px;
        background-position: 0 0, 0 4px, 4px -4px, -4px 0px;
      }
      .layer-info { flex: 1; overflow: hidden; min-width: 0; }
      .layer-name { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 11px; }
      .layer-type { font-size: 9px; color: var(--theme-fg-muted); }
      .layer-controls {
        display: flex;
        gap: 4px;
        flex-shrink: 0;
      }
      .layer-btn {
        width: 20px;
        height: 20px;
        border: none;
        background: transparent;
        color: var(--theme-fg);
        cursor: pointer;
        border-radius: 3px;
        font-size: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
      }
      .layer-btn:hover { background: var(--theme-accent-bg); }
      .layer-btn.visibility-off { opacity: 0.4; }
      .layer-btn.effects-btn { color: var(--theme-fg-muted); }
      .layer-btn.effects-btn:hover { color: var(--theme-accent); }
      .layer-btn.effects-btn.has-effects { color: var(--theme-accent); background: var(--theme-accent-bg); }
      .layer-btn.mask-add-btn { color: var(--theme-fg-disabled); font-size: 10px; }
      .layer-btn.mask-add-btn:hover { color: var(--theme-warning); }
      .layer-btn.mask-edit-btn { color: var(--theme-fg-muted); }
      .layer-btn.mask-edit-btn:hover { color: var(--theme-warning); }
      .layer-btn.mask-edit-btn.editing-mask { color: #ff6600; background: #ff660033; }
      .layer-item.has-mask { border-left: 2px solid var(--theme-warning); }
      .image-item {
        cursor: grab;
      }
      .image-item:active { cursor: grabbing; }
      
      /* Modal styles */
      .modal-overlay {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0,0,0,0.6);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 1000;
        backdrop-filter: blur(2px);
      }
      .modal-content {
        background: var(--theme-panel-bg, var(--theme-bg));
        border: 1px solid var(--theme-border);
        border-radius: 12px;
        width: 360px;
        max-width: 90vw;
        max-height: 80vh;
        overflow: hidden;
        box-shadow: 0 8px 32px var(--theme-shadow);
      }
      .modal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 16px;
        border-bottom: 1px solid var(--theme-border);
        background: var(--theme-panel-header, var(--theme-bg-dark));
        font-weight: 600;
      }
      .modal-close-btn {
        width: 28px;
        height: 28px;
        border: none;
        background: transparent;
        color: var(--theme-fg);
        font-size: 20px;
        cursor: pointer;
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
      }
      .modal-close-btn:hover { background: var(--theme-accent-bg); }
      .modal-body {
        padding: 12px 16px;
        overflow-y: auto;
        max-height: calc(80vh - 60px);
      }
      .effect-section {
        margin-bottom: 16px;
        padding-bottom: 16px;
        border-bottom: 1px solid var(--theme-border-light);
      }
      .effect-section:last-child { border-bottom: none; margin-bottom: 0; }
      .effect-header {
        font-weight: 600;
        font-size: 12px;
        margin-bottom: 10px;
        color: var(--theme-fg);
      }
      .effect-controls {
        padding-left: 8px;
      }
      .effect-row {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
        font-size: 11px;
      }
      .effect-row:last-child { margin-bottom: 0; }
      .effect-row label { min-width: 60px; color: var(--theme-fg-muted); }
      .effect-row input[type="range"] { flex: 1; accent-color: var(--theme-accent); }
      .effect-row input[type="color"] { width: 32px; height: 24px; border: 1px solid var(--theme-input-border); border-radius: 4px; cursor: pointer; }
      .effect-row span { min-width: 30px; text-align: right; font-family: monospace; }
      
      /* Transform handles cursor styles */
      .canvas-container.cursor-nwse { cursor: nwse-resize; }
      .canvas-container.cursor-nesw { cursor: nesw-resize; }
      .canvas-container.cursor-rotate { cursor: grab; }
      .canvas-container.cursor-move { cursor: move; }
    `;
  }

  static getScripts() {
    return `
      <script>
        (function() {
          try {
          
          const container = document.getElementById('canvas-composer');
          if (!container) return;
          
          // Get origin from data attribute (passed from parent context)
          const origin = container.dataset.origin || '';
          
          // Helper to convert file info to ComfyUI /view URL
          function getImageUrl(fileInfo) {
            if (typeof fileInfo === 'string') {
              // Legacy base64 URL
              return fileInfo;
            }
            // File reference - construct absolute /view URL
            const params = new URLSearchParams({
              filename: fileInfo.filename,
              subfolder: fileInfo.subfolder || '',
              type: fileInfo.type || 'temp'
            });
            return origin + '/view?' + params.toString();
          }
          
          // Load image - simple approach without crossOrigin/fetch to avoid CORS in srcdoc iframe
          function loadImage(url) {
            return new Promise((resolve, reject) => {
              const img = new Image();
              img.onload = () => resolve(img);
              img.onerror = (e) => {
                console.error('[Canvas] Failed to load image:', url, e);
                reject(e);
              };
              img.src = url;
            });
          }
          
          // Read image data from data attribute (HTML-escaped JSON)
          let imagesData = [];
          try {
            const jsonStr = container.dataset.images;
            if (jsonStr) {
              imagesData = JSON.parse(jsonStr);
            }
          } catch (e) {}
          
          const canvas = document.getElementById('main-canvas');
          if (!canvas) {
            console.error('[Canvas] Main canvas not found!');
            return;
          }
          const ctx = canvas.getContext('2d');
          
          // State
          const state = {
            tool: 'select',
            zoom: 1,
            panX: 0,
            panY: 0,
            layers: [],
            selectedLayerId: null,
            isDragging: false,
            dragStart: { x: 0, y: 0 },
            dragType: null,
            activeHandle: null,
            handleStartData: null,
            brushColor: '#ffffff',
            brushSize: 10,
            brushHardness: 100,
            brushOpacity: 100,
            brushTool: 'brush',
            brushBlend: 'source-over',
            cloneSource: null,
            cloneAllLayers: false,
            editingMask: false,
            bgColor: '#000000',
            bgTransparent: true,
            isPainting: false,
            lastPaintPos: null,
            modalLayerId: null,
            renderPending: false,
            layerListDirty: true,
            exportWidth: 0,
            exportHeight: 0,
            snapEnabled: false,
            snapThreshold: 10,
          };
          
          let layerIdCounter = 0;
          
          // Undo/Redo history - saves canvas pixel data as ImageData
          const history = {
            states: [],
            index: -1,
            maxStates: 30,
            
            save() {
              // Remove any states after current index
              this.states = this.states.slice(0, this.index + 1);
              
              // Save layer state with canvas data
              const snapshot = state.layers.map(l => {
                const saved = {
                  id: l.id,
                  type: l.type,
                  name: l.name,
                  visible: l.visible,
                  x: l.x,
                  y: l.y,
                  width: l.width,
                  height: l.height,
                  rotation: l.rotation,
                  scaleX: l.scaleX,
                  scaleY: l.scaleY,
                  skewX: l.skewX,
                  skewY: l.skewY,
                  opacity: l.opacity,
                  blendMode: l.blendMode,
                  strokeWidth: l.strokeWidth,
                  strokeColor: l.strokeColor,
                  shadow: l.shadow,
                  shadowBlur: l.shadowBlur,
                  shadowOffsetX: l.shadowOffsetX,
                  shadowOffsetY: l.shadowOffsetY,
                  shadowColor: l.shadowColor,
                  maskEnabled: l.maskEnabled,
                  imageUrl: l.imageUrl,
                  drawData: null,
                  maskData: null,
                  tainted: false
                };
                // Save draw canvas data (may fail if canvas is tainted by cross-origin images)
                if (l.drawCanvas) {
                  try {
                    saved.drawData = l.drawCtx.getImageData(0, 0, l.drawCanvas.width, l.drawCanvas.height);
                  } catch (e) {
                    // Canvas is tainted - save as data URL instead (or mark as tainted)
                    saved.tainted = true;
                    try {
                      saved.drawDataUrl = l.drawCanvas.toDataURL();
                    } catch (e2) {
                      // Can't save tainted canvas at all
                      saved.drawDataUrl = null;
                    }
                  }
                }
                // Save mask canvas data
                if (l.maskCanvas) {
                  try {
                    saved.maskData = l.maskCtx.getImageData(0, 0, l.maskCanvas.width, l.maskCanvas.height);
                  } catch (e) {
                    // Mask canvas tainted (unlikely but handle it)
                    saved.maskData = null;
                  }
                }
                return saved;
              });
              
              this.states.push(snapshot);
              if (this.states.length > this.maxStates) {
                this.states.shift();
              } else {
                this.index++;
              }
            },
            
            undo() {
              if (this.index > 0) {
                this.index--;
                this.restore();
              }
            },
            
            redo() {
              if (this.index < this.states.length - 1) {
                this.index++;
                this.restore();
              }
            },
            
            restore() {
              const snapshot = this.states[this.index];
              if (!snapshot) return;
              
              // Get IDs from snapshot
              const snapshotIds = new Set(snapshot.map(s => s.id));
              const currentIds = new Set(state.layers.map(l => l.id));
              
              // Remove layers that don't exist in snapshot
              state.layers = state.layers.filter(l => snapshotIds.has(l.id));
              
              // Process each saved layer
              snapshot.forEach(saved => {
                let layer = state.layers.find(l => l.id === saved.id);
                
                // If layer doesn't exist, recreate it
                if (!layer) {
                  if (saved.type === 'draw') {
                    layer = createDrawLayer();
                    layer.id = saved.id;
                  } else {
                    // Recreate image layer
                    layer = {
                      id: saved.id,
                      type: saved.type,
                      name: saved.name,
                      image: null,
                      imageUrl: saved.imageUrl,
                      x: saved.x,
                      y: saved.y,
                      width: saved.width,
                      height: saved.height,
                      rotation: saved.rotation || 0,
                      scaleX: saved.scaleX || 1,
                      scaleY: saved.scaleY || 1,
                      skewX: saved.skewX || 0,
                      skewY: saved.skewY || 0,
                      opacity: saved.opacity ?? 100,
                      blendMode: saved.blendMode || 'source-over',
                      visible: saved.visible !== false,
                      strokeWidth: saved.strokeWidth || 0,
                      strokeColor: saved.strokeColor || '#000000',
                      shadow: saved.shadow || false,
                      shadowBlur: saved.shadowBlur || 10,
                      shadowOffsetX: saved.shadowOffsetX || 5,
                      shadowOffsetY: saved.shadowOffsetY || 5,
                      shadowColor: saved.shadowColor || 'rgba(0,0,0,0.5)',
                      maskEnabled: saved.maskEnabled || false,
                      maskCanvas: null,
                      maskCtx: null,
                    };
                    // Reload image if needed
                    if (saved.imageUrl) {
                      const img = new Image();
                      img.crossOrigin = 'anonymous';
                      img.onload = () => {
                        layer.image = img;
                        scheduleRender();
                      };
                      img.src = saved.imageUrl;
                    }
                  }
                  state.layers.push(layer);
                }
                
                // Restore properties
                Object.keys(saved).forEach(key => {
                  if (key !== 'drawData' && key !== 'maskData' && key !== 'drawDataUrl' && key !== 'id' && key !== 'tainted') {
                    layer[key] = saved[key];
                  }
                });
                
                // Restore draw canvas data
                if (saved.drawData && layer.drawCanvas) {
                  layer.drawCtx.putImageData(saved.drawData, 0, 0);
                } else if (saved.tainted && saved.drawDataUrl && layer.drawCanvas) {
                  // Restore tainted canvas from data URL
                  const img = new Image();
                  img.onload = () => {
                    layer.drawCtx.clearRect(0, 0, layer.drawCanvas.width, layer.drawCanvas.height);
                    layer.drawCtx.drawImage(img, 0, 0);
                    scheduleRender();
                  };
                  img.src = saved.drawDataUrl;
                }
                
                // Restore mask canvas data
                if (saved.maskData && layer.maskCanvas) {
                  layer.maskCtx.putImageData(saved.maskData, 0, 0);
                }
              });
              
              // Reorder layers to match snapshot order
              const orderedLayers = [];
              snapshot.forEach(saved => {
                const layer = state.layers.find(l => l.id === saved.id);
                if (layer) orderedLayers.push(layer);
              });
              state.layers = orderedLayers;
              
              // Reset selection if selected layer no longer exists
              if (state.selectedLayerId && !state.layers.find(l => l.id === state.selectedLayerId)) {
                state.selectedLayerId = state.layers.length > 0 ? state.layers[0].id : null;
              }
              
              // Reset clone source as it may be invalid
              state.cloneSource = null;
              const cloneStatus = document.getElementById('clone-status');
              if (cloneStatus) cloneStatus.textContent = 'Alt+Click to set source';
              
              // Reset painting state
              state.isPainting = false;
              state.lastPaintPos = null;
              
              markLayerListDirty();
              scheduleRender();
            }
          };
          
          function createLayerMask(layerId) {
            const layer = state.layers.find(l => l.id === layerId);
            if (!layer || layer.maskCanvas) return;
            
            const maskCanvas = document.createElement('canvas');
            maskCanvas.width = layer.width;
            maskCanvas.height = layer.height;
            const maskCtx = maskCanvas.getContext('2d');
            // Start with fully opaque white mask (fully visible)
            maskCtx.fillStyle = 'white';
            maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
            
            layer.maskCanvas = maskCanvas;
            layer.maskCtx = maskCtx;
            layer.maskEnabled = true;
            markLayerListDirty();
            scheduleRender();
          }
          
          function deleteLayerMask(layerId) {
            const layer = state.layers.find(l => l.id === layerId);
            if (!layer || !layer.maskCanvas) return;
            
            layer.maskCanvas = null;
            layer.maskCtx = null;
            layer.maskEnabled = false;
            state.editingMask = false;
            markLayerListDirty();
            scheduleRender();
          }
          
          function toggleMaskEditing() {
            const layer = state.layers.find(l => l.id === state.selectedLayerId);
            if (!layer || !layer.maskCanvas) return;
            state.editingMask = !state.editingMask;
            markLayerListDirty();
            scheduleRender();
          }
          
          function duplicateLayer(layerId) {
            const layer = state.layers.find(l => l.id === layerId);
            if (!layer) return null;
            
            const newLayer = {
              ...layer,
              id: ++layerIdCounter,
              name: layer.name + ' Copy',
              x: layer.x + 20,
              y: layer.y + 20
            };
            
            // Clone draw canvas if it's a draw layer
            if (layer.type === 'draw' && layer.drawCanvas) {
              newLayer.drawCanvas = document.createElement('canvas');
              newLayer.drawCanvas.width = layer.drawCanvas.width;
              newLayer.drawCanvas.height = layer.drawCanvas.height;
              newLayer.drawCtx = newLayer.drawCanvas.getContext('2d');
              newLayer.drawCtx.drawImage(layer.drawCanvas, 0, 0);
              newLayer.image = newLayer.drawCanvas;
            }
            
            // Clone mask if exists
            if (layer.maskCanvas) {
              newLayer.maskCanvas = document.createElement('canvas');
              newLayer.maskCanvas.width = layer.maskCanvas.width;
              newLayer.maskCanvas.height = layer.maskCanvas.height;
              newLayer.maskCtx = newLayer.maskCanvas.getContext('2d');
              newLayer.maskCtx.drawImage(layer.maskCanvas, 0, 0);
            }
            
            state.layers.push(newLayer);
            state.selectedLayerId = newLayer.id;
            markLayerListDirty();
            history.save();
            scheduleRender();
            return newLayer;
          }
          
          async function createLayer(imageUrl, x = 100, y = 100, sourceImageIndex = -1) {
            const img = await loadImage(imageUrl);
            const layer = {
              id: ++layerIdCounter,
              type: 'image',
              name: 'Image ' + layerIdCounter,
              visible: true,
              image: img,
              imageUrl: imageUrl,
              sourceImageIndex: sourceImageIndex,
              x: x,
              y: y,
              width: img.width,
              height: img.height,
              sourceWidth: img.width,
              sourceHeight: img.height,
              trimOffsetX: 0,
              trimOffsetY: 0,
              rotation: 0,
              scaleX: 1,
              scaleY: 1,
              skewX: 0,
              skewY: 0,
              opacity: 1,
              strokeWidth: 0,
              strokeColor: '#000000',
              shadow: false,
              shadowBlur: 10,
              shadowOffsetX: 5,
              shadowOffsetY: 5,
              shadowColor: 'rgba(0,0,0,0.5)',
              blendMode: 'source-over',
            };
            state.layers.push(layer);
            markLayerListDirty();
            return layer;
          }
          
          function createDrawLayer(width = 512, height = 512) {
            const drawCanvas = document.createElement('canvas');
            drawCanvas.width = width;
            drawCanvas.height = height;
            const layer = {
              id: ++layerIdCounter,
              type: 'draw',
              name: 'Draw ' + layerIdCounter,
              visible: true,
              drawCanvas: drawCanvas,
              drawCtx: drawCanvas.getContext('2d'),
              image: drawCanvas,
              imageUrl: null,
              x: 50,
              y: 50,
              width: width,
              height: height,
              rotation: 0,
              scaleX: 1,
              scaleY: 1,
              skewX: 0,
              skewY: 0,
              opacity: 1,
              strokeWidth: 0,
              strokeColor: '#000000',
              shadow: false,
              shadowBlur: 10,
              shadowOffsetX: 5,
              shadowOffsetY: 5,
              shadowColor: 'rgba(0,0,0,0.5)',
              blendMode: 'source-over',
            };
            state.layers.push(layer);
            markLayerListDirty();
            return layer;
          }
          
          // Frame every visible layer, so a picture larger than the node opens whole rather
          // than cropped to whichever corner sat under the viewport.
          function fitToLayers(padding) {
            const margin = typeof padding === 'number' ? padding : 24;
            const shown = state.layers.filter(function (l) { return l.visible !== false; });
            if (!shown.length || !canvas.width || !canvas.height) return;
            let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
            for (const l of shown) {
              const w = (l.width || 0) * (l.scaleX == null ? 1 : l.scaleX);
              const h = (l.height || 0) * (l.scaleY == null ? 1 : l.scaleY);
              const x = l.x || 0, y = l.y || 0;
              minX = Math.min(minX, x); minY = Math.min(minY, y);
              maxX = Math.max(maxX, x + w); maxY = Math.max(maxY, y + h);
            }
            const spanX = maxX - minX, spanY = maxY - minY;
            if (!(spanX > 0) || !(spanY > 0)) return;
            const room = Math.max(1, canvas.width - margin * 2);
            const tall = Math.max(1, canvas.height - margin * 2);
            state.zoom = Math.min(room / spanX, tall / spanY, 1);
            state.panX = (canvas.width - spanX * state.zoom) / 2 - minX * state.zoom;
            state.panY = (canvas.height - spanY * state.zoom) / 2 - minY * state.zoom;
            render();
          }

          function resizeCanvas() {
            const container = canvas.parentElement;
            // Use clientWidth/Height to get actual content area without scrollbars
            const width = container.clientWidth;
            const height = container.clientHeight;
            // Set both the canvas buffer size and CSS size to match exactly
            canvas.width = width;
            canvas.height = height;
            canvas.style.width = width + 'px';
            canvas.style.height = height + 'px';
            render();
          }
          
          function scheduleRender() {
            if (state.renderPending) return;
            state.renderPending = true;
            requestAnimationFrame(() => {
              state.renderPending = false;
              render();
            });
          }
          
          // Cached offscreen canvas for compositing (avoid creating new canvas every frame)
          let cachedOffCanvas = null;
          let cachedOffCtx = null;
          // Cached temp canvas for mask compositing
          let cachedMaskCanvas = null;
          let cachedMaskCtx = null;
          let lastMiniMapUpdate = 0;
          const MINIMAP_UPDATE_INTERVAL = 100; // Update minimap at most every 100ms
          
          function render() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Background
            if (!state.bgTransparent) {
              ctx.fillStyle = state.bgColor;
              ctx.fillRect(0, 0, canvas.width, canvas.height);
            }
            
            // Calculate bounds for offscreen canvas
            let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
            for (const layer of state.layers) {
              if (!layer.visible) continue;
              const lw = layer.width * layer.scaleX;
              const lh = layer.height * layer.scaleY;
              minX = Math.min(minX, layer.x);
              minY = Math.min(minY, layer.y);
              maxX = Math.max(maxX, layer.x + lw);
              maxY = Math.max(maxY, layer.y + lh);
            }
            
            // Reuse cached offscreen canvas for layer compositing (prevents blending with checkered bg)
            const hasLayers = state.layers.some(l => l.visible);
            let offCanvas, offCtx;
            if (hasLayers && isFinite(minX)) {
              const padding = 100;
              const offWidth = Math.max(1, Math.ceil(maxX - minX + padding * 2));
              const offHeight = Math.max(1, Math.ceil(maxY - minY + padding * 2));
              
              // Reuse cached canvas if large enough, otherwise create new one
              if (!cachedOffCanvas || cachedOffCanvas.width < offWidth || cachedOffCanvas.height < offHeight) {
                cachedOffCanvas = document.createElement('canvas');
                cachedOffCanvas.width = Math.max(offWidth, 2048);
                cachedOffCanvas.height = Math.max(offHeight, 2048);
                cachedOffCtx = cachedOffCanvas.getContext('2d');
              }
              offCanvas = cachedOffCanvas;
              offCtx = cachedOffCtx;
              offCtx.setTransform(1, 0, 0, 1, 0, 0);
              offCtx.clearRect(0, 0, offCanvas.width, offCanvas.height);
              offCtx.translate(-minX + padding, -minY + padding);
            }
            
            // Render layers to offscreen canvas
            for (const layer of state.layers) {
              if (!layer.visible) continue;
              if (!offCtx) continue;
              
              offCtx.save();
              
              offCtx.translate(layer.x + layer.width * layer.scaleX * 0.5, layer.y + layer.height * layer.scaleY * 0.5);
              offCtx.rotate(layer.rotation * Math.PI / 180);
              offCtx.transform(1, layer.skewY, layer.skewX, 1, 0, 0);
              offCtx.scale(layer.scaleX, layer.scaleY);
              offCtx.translate(-layer.width * 0.5, -layer.height * 0.5);
              
              offCtx.globalAlpha = layer.opacity;
              offCtx.globalCompositeOperation = layer.blendMode || 'source-over';
              
              // Apply shadow if enabled
              if (layer.shadow) {
                offCtx.shadowBlur = layer.shadowBlur;
                offCtx.shadowOffsetX = layer.shadowOffsetX;
                offCtx.shadowOffsetY = layer.shadowOffsetY;
                offCtx.shadowColor = layer.shadowColor;
              }
              
              // Draw stroke behind image using canvas shadow trick
              if (layer.strokeWidth > 0) {
                offCtx.save();
                offCtx.shadowColor = layer.strokeColor;
                offCtx.shadowBlur = layer.strokeWidth;
                offCtx.shadowOffsetX = 0;
                offCtx.shadowOffsetY = 0;
                offCtx.drawImage(layer.image, 0, 0, layer.width, layer.height);
                offCtx.restore();
                if (layer.shadow) {
                  offCtx.shadowBlur = layer.shadowBlur;
                  offCtx.shadowOffsetX = layer.shadowOffsetX;
                  offCtx.shadowOffsetY = layer.shadowOffsetY;
                  offCtx.shadowColor = layer.shadowColor;
                } else {
                  offCtx.shadowColor = 'transparent';
                }
              }
              
              // Draw the image with mask if enabled
              if (layer.maskCanvas && layer.maskEnabled) {
                // Reuse cached mask canvas to avoid creating new canvas every frame
                if (!cachedMaskCanvas || cachedMaskCanvas.width < layer.width || cachedMaskCanvas.height < layer.height) {
                  cachedMaskCanvas = document.createElement('canvas');
                  cachedMaskCanvas.width = Math.max(layer.width, 1024);
                  cachedMaskCanvas.height = Math.max(layer.height, 1024);
                  cachedMaskCtx = cachedMaskCanvas.getContext('2d');
                }
                cachedMaskCtx.clearRect(0, 0, layer.width, layer.height);
                cachedMaskCtx.globalCompositeOperation = 'source-over';
                cachedMaskCtx.drawImage(layer.image, 0, 0, layer.width, layer.height);
                cachedMaskCtx.globalCompositeOperation = 'destination-in';
                cachedMaskCtx.drawImage(layer.maskCanvas, 0, 0);
                offCtx.drawImage(cachedMaskCanvas, 0, 0, layer.width, layer.height);
              } else {
                offCtx.drawImage(layer.image, 0, 0, layer.width, layer.height);
              }
              
              // Clear shadow state
              offCtx.shadowColor = 'transparent';
              
              // Show mask overlay when editing - white areas are visible, erased areas show red
              if (state.editingMask && layer.id === state.selectedLayerId && layer.maskCanvas) {
                offCtx.globalAlpha = 0.5;
                offCtx.drawImage(layer.maskCanvas, 0, 0, layer.width, layer.height);
                offCtx.globalAlpha = layer.opacity;
              }
              
              offCtx.restore();
            }
            
            // Draw composited layers onto main canvas
            ctx.save();
            ctx.translate(state.panX, state.panY);
            ctx.scale(state.zoom, state.zoom);
            
            if (offCanvas && hasLayers && isFinite(minX)) {
              const padding = 100;
              ctx.drawImage(offCanvas, minX - padding, minY - padding);
            }
            
            // Draw selection UI on main canvas
            for (const layer of state.layers) {
              if (!layer.visible) continue;
              
              // Selection box
              if (layer.id === state.selectedLayerId && state.tool === 'select') {
                ctx.save();
                ctx.strokeStyle = '#00aaff';
                
                ctx.translate(layer.x + layer.width * layer.scaleX * 0.5, layer.y + layer.height * layer.scaleY * 0.5);
                ctx.rotate(layer.rotation * Math.PI / 180);
                ctx.transform(1, layer.skewY, layer.skewX, 1, 0, 0);
                ctx.scale(layer.scaleX, layer.scaleY);
                ctx.translate(-layer.width * 0.5, -layer.height * 0.5);
                
                // Compensate line width and dash for layer scale so they appear consistent
                const avgScale = (layer.scaleX + layer.scaleY) / 2;
                ctx.lineWidth = 2 / state.zoom / avgScale;
                ctx.setLineDash([5 / state.zoom / avgScale, 5 / state.zoom / avgScale]);
                
                // Compensate padding for scale
                const padding = 5 / state.zoom / avgScale;
                ctx.strokeRect(-padding, -padding, layer.width + padding * 2, layer.height + padding * 2);
                
                // Corner handles for scale - compensate size for layer scale
                ctx.fillStyle = '#00aaff';
                const hsX = 8 / state.zoom / layer.scaleX;
                const hsY = 8 / state.zoom / layer.scaleY;
                const hshX = hsX * 0.5;
                const hshY = hsY * 0.5;
                ctx.fillRect(-hshX, -hshY, hsX, hsY);
                ctx.fillRect(layer.width - hshX, -hshY, hsX, hsY);
                ctx.fillRect(-hshX, layer.height - hshY, hsX, hsY);
                ctx.fillRect(layer.width - hshX, layer.height - hshY, hsX, hsY);
                
                // Rotation handle - compensate for scale
                const rotHandleOffset = 20 / state.zoom / layer.scaleY;
                ctx.setLineDash([]);
                ctx.beginPath();
                ctx.moveTo(layer.width * 0.5, 0);
                ctx.lineTo(layer.width * 0.5, -rotHandleOffset);
                ctx.stroke();
                ctx.beginPath();
                ctx.arc(layer.width * 0.5, -rotHandleOffset, Math.min(hsX, hsY) * 0.5, 0, Math.PI * 2);
                ctx.fill();
                
                ctx.restore();
              }
            }
            
            ctx.restore();
            
            // Draw export bbox overlay if size is set
            if (state.exportWidth > 0 && state.exportHeight > 0) {
              ctx.save();
              ctx.translate(state.panX, state.panY);
              ctx.scale(state.zoom, state.zoom);
              
              // Darken area outside export bbox
              ctx.fillStyle = 'rgba(0,0,0,0.4)';
              const bw = state.exportWidth;
              const bh = state.exportHeight;
              const vw = canvas.width / state.zoom;
              const vh = canvas.height / state.zoom;
              const vx = -state.panX / state.zoom;
              const vy = -state.panY / state.zoom;
              
              // Top
              ctx.fillRect(vx, vy, vw, -vy);
              // Bottom
              ctx.fillRect(vx, bh, vw, vh - bh + vy);
              // Left
              ctx.fillRect(vx, 0, -vx, bh);
              // Right
              ctx.fillRect(bw, 0, vw - bw + vx, bh);
              
              // Draw export bbox border
              ctx.strokeStyle = '#fff';
              ctx.lineWidth = 2 / state.zoom;
              ctx.setLineDash([8 / state.zoom, 4 / state.zoom]);
              ctx.strokeRect(0, 0, bw, bh);
              
              // Draw corner markers
              ctx.setLineDash([]);
              ctx.fillStyle = '#fff';
              const ms = 6 / state.zoom;
              ctx.fillRect(-ms/2, -ms/2, ms, ms);
              ctx.fillRect(bw - ms/2, -ms/2, ms, ms);
              ctx.fillRect(-ms/2, bh - ms/2, ms, ms);
              ctx.fillRect(bw - ms/2, bh - ms/2, ms, ms);
              
              ctx.restore();
            }
            
            // Draw snap guides when dragging
            if (state.isDragging && state.snapEnabled && state.snapGuides) {
              ctx.save();
              ctx.strokeStyle = '#0af';
              ctx.lineWidth = 1;
              ctx.setLineDash([4, 4]);
              
              for (const guide of state.snapGuides) {
                ctx.beginPath();
                if (guide.type === 'vertical') {
                  const sx = guide.x * state.zoom + state.panX;
                  ctx.moveTo(sx, 0);
                  ctx.lineTo(sx, canvas.height);
                } else {
                  const sy = guide.y * state.zoom + state.panY;
                  ctx.moveTo(0, sy);
                  ctx.lineTo(canvas.width, sy);
                }
                ctx.stroke();
              }
              ctx.restore();
            }
            
            if (state.layerListDirty) {
              state.layerListDirty = false;
              updateLayersList();
            }
            
            // Throttle minimap updates to avoid performance issues
            const now = performance.now();
            if (now - lastMiniMapUpdate > MINIMAP_UPDATE_INTERVAL) {
              lastMiniMapUpdate = now;
              updateMiniMap();
            }
          }
          
          // Cache minimap canvas reference and dimensions
          let miniMapCanvas = null;
          let miniMapCtx = null;
          let miniMapWidth = 0;
          let miniMapHeight = 0;
          
          function updateMiniMap() {
            if (!miniMapCanvas) {
              miniMapCanvas = document.getElementById('mini-map-canvas');
              if (!miniMapCanvas) return;
            }
            if (!miniMapCtx) {
              miniMapCtx = miniMapCanvas.getContext('2d');
              if (!miniMapCtx) return;
            }
            
            // Only recalculate dimensions occasionally (not every frame)
            const rect = miniMapCanvas.getBoundingClientRect();
            if (rect.width !== miniMapWidth || rect.height !== miniMapHeight) {
              miniMapWidth = rect.width;
              miniMapHeight = rect.height;
              miniMapCanvas.width = miniMapWidth;
              miniMapCanvas.height = miniMapHeight;
            }
            
            const mctx = miniMapCtx;
            
            mctx.fillStyle = '#222';
            mctx.fillRect(0, 0, miniMapWidth, miniMapHeight);
            
            // Calculate content bounds (include export bbox if set)
            let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
            
            // Include export bbox in bounds calculation
            if (state.exportWidth > 0 && state.exportHeight > 0) {
              minX = Math.min(minX, 0);
              minY = Math.min(minY, 0);
              maxX = Math.max(maxX, state.exportWidth);
              maxY = Math.max(maxY, state.exportHeight);
            }
            
            for (const layer of state.layers) {
              if (!layer.visible) continue;
              const lw = layer.width * layer.scaleX;
              const lh = layer.height * layer.scaleY;
              minX = Math.min(minX, layer.x);
              minY = Math.min(minY, layer.y);
              maxX = Math.max(maxX, layer.x + lw);
              maxY = Math.max(maxY, layer.y + lh);
            }
            
            if (!isFinite(minX)) {
              minX = 0; minY = 0; maxX = 512; maxY = 512;
            }
            
            // Add padding
            const pad = 100;
            minX -= pad; minY -= pad;
            maxX += pad; maxY += pad;
            
            const contentW = maxX - minX;
            const contentH = maxY - minY;
            
            // Calculate scale to fit content in mini-map
            const scale = Math.min(miniMapWidth / contentW, miniMapHeight / contentH) * 0.9;
            const offsetX = (miniMapWidth - contentW * scale) / 2;
            const offsetY = (miniMapHeight - contentH * scale) / 2;
            
            // Draw layers as rectangles
            mctx.save();
            mctx.translate(offsetX, offsetY);
            mctx.scale(scale, scale);
            mctx.translate(-minX, -minY);
            
            // Draw export bbox if set
            if (state.exportWidth > 0 && state.exportHeight > 0) {
              mctx.strokeStyle = '#f80';
              mctx.lineWidth = 2 / scale;
              mctx.setLineDash([4 / scale, 2 / scale]);
              mctx.strokeRect(0, 0, state.exportWidth, state.exportHeight);
              mctx.setLineDash([]);
            }
            
            for (const layer of state.layers) {
              if (!layer.visible) continue;
              mctx.fillStyle = layer.id === state.selectedLayerId ? '#6cf' : '#888';
              mctx.globalAlpha = 0.5;
              mctx.fillRect(layer.x, layer.y, layer.width * layer.scaleX, layer.height * layer.scaleY);
            }
            
            // Draw viewport rectangle
            const viewX = -state.panX / state.zoom;
            const viewY = -state.panY / state.zoom;
            const viewW = canvas.width / state.zoom;
            const viewH = canvas.height / state.zoom;
            
            mctx.globalAlpha = 1;
            mctx.strokeStyle = '#fff';
            mctx.lineWidth = 2 / scale;
            mctx.strokeRect(viewX, viewY, viewW, viewH);
            
            mctx.restore();
            
            // Store transform info for click handling
            miniMapCanvas._miniMapTransform = { minX, minY, scale, offsetX, offsetY };
          }
          
          function markLayerListDirty() {
            state.layerListDirty = true;
          }
          
          function calculateSnap(layer, newX, newY) {
            if (!state.snapEnabled) return { x: newX, y: newY, guides: [] };
            
            const threshold = state.snapThreshold / state.zoom;
            const guides = [];
            let snapX = newX;
            let snapY = newY;
            
            const layerW = layer.width * layer.scaleX;
            const layerH = layer.height * layer.scaleY;
            
            // Layer edges
            const layerLeft = newX;
            const layerRight = newX + layerW;
            const layerTop = newY;
            const layerBottom = newY + layerH;
            const layerCenterX = newX + layerW / 2;
            const layerCenterY = newY + layerH / 2;
            
            // Collect snap targets
            const snapTargetsX = [];
            const snapTargetsY = [];
            
            // Export bbox edges (if set)
            if (state.exportWidth > 0 && state.exportHeight > 0) {
              snapTargetsX.push({ val: 0, label: 'bbox-left' });
              snapTargetsX.push({ val: state.exportWidth, label: 'bbox-right' });
              snapTargetsX.push({ val: state.exportWidth / 2, label: 'bbox-centerX' });
              snapTargetsY.push({ val: 0, label: 'bbox-top' });
              snapTargetsY.push({ val: state.exportHeight, label: 'bbox-bottom' });
              snapTargetsY.push({ val: state.exportHeight / 2, label: 'bbox-centerY' });
            }
            
            // Other layer edges
            for (const other of state.layers) {
              if (other.id === layer.id || !other.visible) continue;
              const ow = other.width * other.scaleX;
              const oh = other.height * other.scaleY;
              snapTargetsX.push({ val: other.x, label: 'layer-left' });
              snapTargetsX.push({ val: other.x + ow, label: 'layer-right' });
              snapTargetsX.push({ val: other.x + ow / 2, label: 'layer-centerX' });
              snapTargetsY.push({ val: other.y, label: 'layer-top' });
              snapTargetsY.push({ val: other.y + oh, label: 'layer-bottom' });
              snapTargetsY.push({ val: other.y + oh / 2, label: 'layer-centerY' });
            }
            
            // Find closest snap for X (check left, right, center)
            let bestSnapX = null;
            let bestDistX = threshold;
            for (const target of snapTargetsX) {
              // Left edge
              const distL = Math.abs(layerLeft - target.val);
              if (distL < bestDistX) {
                bestDistX = distL;
                bestSnapX = { offset: target.val - layerLeft, guide: target.val };
              }
              // Right edge
              const distR = Math.abs(layerRight - target.val);
              if (distR < bestDistX) {
                bestDistX = distR;
                bestSnapX = { offset: target.val - layerRight, guide: target.val };
              }
              // Center
              const distC = Math.abs(layerCenterX - target.val);
              if (distC < bestDistX) {
                bestDistX = distC;
                bestSnapX = { offset: target.val - layerCenterX, guide: target.val };
              }
            }
            
            // Find closest snap for Y (check top, bottom, center)
            let bestSnapY = null;
            let bestDistY = threshold;
            for (const target of snapTargetsY) {
              // Top edge
              const distT = Math.abs(layerTop - target.val);
              if (distT < bestDistY) {
                bestDistY = distT;
                bestSnapY = { offset: target.val - layerTop, guide: target.val };
              }
              // Bottom edge
              const distB = Math.abs(layerBottom - target.val);
              if (distB < bestDistY) {
                bestDistY = distB;
                bestSnapY = { offset: target.val - layerBottom, guide: target.val };
              }
              // Center
              const distC = Math.abs(layerCenterY - target.val);
              if (distC < bestDistY) {
                bestDistY = distC;
                bestSnapY = { offset: target.val - layerCenterY, guide: target.val };
              }
            }
            
            // Apply snaps
            if (bestSnapX) {
              snapX = newX + bestSnapX.offset;
              guides.push({ type: 'vertical', x: bestSnapX.guide });
            }
            if (bestSnapY) {
              snapY = newY + bestSnapY.offset;
              guides.push({ type: 'horizontal', y: bestSnapY.guide });
            }
            
            return { x: snapX, y: snapY, guides };
          }
          
          function getLayerAtPoint(x, y) {
            const wx = (x - state.panX) / state.zoom;
            const wy = (y - state.panY) / state.zoom;
            
            function isPointInLayer(layer) {
              const lx = layer.x;
              const ly = layer.y;
              const lw = layer.width * layer.scaleX;
              const lh = layer.height * layer.scaleY;
              return wx >= lx && wx <= lx + lw && wy >= ly && wy <= ly + lh;
            }
            
            // Prioritize currently selected layer - allows dragging even if behind others
            if (state.selectedLayerId) {
              const selectedLayer = state.layers.find(l => l.id === state.selectedLayerId);
              if (selectedLayer && selectedLayer.visible && isPointInLayer(selectedLayer)) {
                return selectedLayer;
              }
            }
            
            // Fall back to top-to-bottom search for other layers
            for (let i = state.layers.length - 1; i >= 0; i--) {
              const layer = state.layers[i];
              if (!layer.visible) continue;
              if (isPointInLayer(layer)) {
                return layer;
              }
            }
            return null;
          }
          
          function getHandleAtPoint(x, y) {
            if (!state.selectedLayerId) return null;
            const layer = state.layers.find(l => l.id === state.selectedLayerId);
            if (!layer) return null;
            
            const wx = (x - state.panX) / state.zoom;
            const wy = (y - state.panY) / state.zoom;
            const hs = 24 / state.zoom; // Hit detection radius (larger than visual size)
            
            // Transform world point to layer's local coordinate system
            // Reverse the transform chain: translate to center, rotate, skew, scale, translate to corner
            const cx = layer.x + layer.width * layer.scaleX * 0.5;
            const cy = layer.y + layer.height * layer.scaleY * 0.5;
            
            // Translate to center
            let lx = wx - cx;
            let ly = wy - cy;
            
            // Inverse rotation
            const rad = -layer.rotation * Math.PI / 180;
            const cos = Math.cos(rad);
            const sin = Math.sin(rad);
            const rx = lx * cos - ly * sin;
            const ry = lx * sin + ly * cos;
            lx = rx;
            ly = ry;
            
            // Inverse skew (approximate for small skew values)
            const skewX = layer.skewX || 0;
            const skewY = layer.skewY || 0;
            const preLx = lx;
            lx = lx - ly * skewX;
            ly = ly - preLx * skewY;
            
            // Now lx, ly is in the rotated/skewed local space centered at layer center
            // Handle positions in local space (relative to layer center, in unscaled coords)
            const hw = layer.width * 0.5;
            const hh = layer.height * 0.5;
            
            const handles = [
              { name: 'nw', x: -hw * layer.scaleX, y: -hh * layer.scaleY },
              { name: 'ne', x: hw * layer.scaleX, y: -hh * layer.scaleY },
              { name: 'sw', x: -hw * layer.scaleX, y: hh * layer.scaleY },
              { name: 'se', x: hw * layer.scaleX, y: hh * layer.scaleY },
              { name: 'rotate', x: 0, y: -hh * layer.scaleY - 20 / state.zoom }
            ];
            
            for (const h of handles) {
              if (Math.abs(lx - h.x) < hs && Math.abs(ly - h.y) < hs) {
                return h.name;
              }
            }
            return null;
          }
          
          let draggedLayerId = null;
          
          function updateLayersList() {
            const list = document.getElementById('layers-list');
            const layerOrder = state.layers.slice().reverse();
            list.innerHTML = layerOrder.map(function(layer, displayIdx) {
              var thumbStyle = layer.imageUrl ? 'style="background-image: url(' + "'" + layer.imageUrl + "'" + ')"' : '';
              var selectedClass = layer.id === state.selectedLayerId ? 'selected' : '';
              var drawClass = layer.type === 'draw' ? 'draw-layer' : '';
              var visClass = !layer.visible ? 'visibility-off' : '';
              var visIcon = layer.visible ? '👁' : '👁‍🗨';
              var typeText = layer.type === 'draw' ? 'Drawing' : 'Image';
              var hasEffects = layer.strokeWidth > 0 || layer.shadow;
              var effectsClass = hasEffects ? 'has-effects' : '';
              var hasMask = layer.maskCanvas ? 'has-mask' : '';
              var maskEditClass = (state.editingMask && layer.id === state.selectedLayerId) ? 'editing-mask' : '';
              var maskBtn = layer.maskCanvas 
                ? '<button class="layer-btn mask-edit-btn ' + maskEditClass + '" data-id="' + layer.id + '" title="Edit Mask">🎭</button>'
                : '<button class="layer-btn mask-add-btn" data-id="' + layer.id + '" title="Add Mask">➕🎭</button>';
              return '<div class="layer-item ' + selectedClass + ' ' + hasMask + '" data-id="' + layer.id + '" data-order="' + displayIdx + '" draggable="true">' +
                '<div class="layer-thumb ' + drawClass + '" ' + thumbStyle + '></div>' +
                '<div class="layer-info">' +
                  '<div class="layer-name">' + layer.name + '</div>' +
                  '<div class="layer-type">' + typeText + '</div>' +
                '</div>' +
                '<div class="layer-controls">' +
                  maskBtn +
                  '<button class="layer-btn effects-btn ' + effectsClass + '" data-id="' + layer.id + '" title="Layer Effects">⚙</button>' +
                  '<button class="layer-btn visibility-btn ' + visClass + '" data-id="' + layer.id + '" title="Toggle Visibility">' + visIcon + '</button>' +
                  '<button class="layer-btn delete-btn" data-id="' + layer.id + '" title="Delete Layer">🗑</button>' +
                '</div>' +
              '</div>';
            }).join('');
          }
          
          // Event delegation for layers list - attach once, not on every update
          (function() {
            const list = document.getElementById('layers-list');
            
            list.addEventListener('click', (e) => {
              const item = e.target.closest('.layer-item');
              if (!item) return;
              const id = parseInt(item.dataset.id);
              
              if (e.target.closest('.effects-btn')) {
                e.stopPropagation();
                openEffectsModal(id);
              } else if (e.target.closest('.visibility-btn')) {
                e.stopPropagation();
                const layer = state.layers.find(l => l.id === id);
                if (layer) {
                  layer.visible = !layer.visible;
                  markLayerListDirty();
                  scheduleRender();
                }
              } else if (e.target.closest('.delete-btn')) {
                e.stopPropagation();
                state.layers = state.layers.filter(l => l.id !== id);
                if (state.selectedLayerId === id) state.selectedLayerId = null;
                markLayerListDirty();
                scheduleRender();
              } else if (e.target.closest('.mask-add-btn')) {
                e.stopPropagation();
                createLayerMask(id);
                state.selectedLayerId = id;
                state.editingMask = true;
                document.querySelector('[data-tool="brush"]').click();
              } else if (e.target.closest('.mask-edit-btn')) {
                e.stopPropagation();
                state.selectedLayerId = id;
                toggleMaskEditing();
                if (state.editingMask) {
                  document.querySelector('[data-tool="brush"]').click();
                }
              } else {
                state.selectedLayerId = id;
                updateLayerControls();
                markLayerListDirty();
                scheduleRender();
              }
            });
            
            list.addEventListener('dragstart', (e) => {
              const item = e.target.closest('.layer-item');
              if (!item) return;
              draggedLayerId = parseInt(item.dataset.id);
              item.classList.add('dragging');
              e.dataTransfer.effectAllowed = 'move';
            });
            
            list.addEventListener('dragend', (e) => {
              const item = e.target.closest('.layer-item');
              if (item) item.classList.remove('dragging');
              draggedLayerId = null;
              list.querySelectorAll('.layer-item').forEach(i => i.classList.remove('drag-over'));
            });
            
            list.addEventListener('dragover', (e) => {
              e.preventDefault();
              const item = e.target.closest('.layer-item');
              if (item && draggedLayerId && parseInt(item.dataset.id) !== draggedLayerId) {
                item.classList.add('drag-over');
              }
            });
            
            list.addEventListener('dragleave', (e) => {
              const item = e.target.closest('.layer-item');
              if (item) item.classList.remove('drag-over');
            });
            
            list.addEventListener('drop', (e) => {
              e.preventDefault();
              const item = e.target.closest('.layer-item');
              if (!item) return;
              item.classList.remove('drag-over');
              const targetId = parseInt(item.dataset.id);
              if (draggedLayerId && draggedLayerId !== targetId) {
                const fromIdx = state.layers.findIndex(l => l.id === draggedLayerId);
                const toIdx = state.layers.findIndex(l => l.id === targetId);
                if (fromIdx !== -1 && toIdx !== -1) {
                  const [moved] = state.layers.splice(fromIdx, 1);
                  // Layer list is displayed reversed, so swap the insert logic
                  const insertIdx = fromIdx > toIdx ? toIdx + 1 : toIdx;
                  state.layers.splice(insertIdx, 0, moved);
                  markLayerListDirty();
                  scheduleRender();
                }
              }
            });
          })();
          
          function updateImagesList() {
            const list = document.getElementById('images-list');
            list.innerHTML = imagesData.map(function(fileInfo, idx) {
              var imgUrl = getImageUrl(fileInfo);
              return '<div class="image-item" data-idx="' + idx + '" draggable="true">' +
                '<div class="layer-thumb" style="background-image: url(' + "'" + imgUrl + "'" + ')"></div>' +
                '<div class="layer-info">' +
                  '<div class="layer-name">Image ' + (idx + 1) + '</div>' +
                '</div>' +
              '</div>';
            }).join('');
            
            list.querySelectorAll('.image-item').forEach(item => {
              item.addEventListener('dragstart', (e) => {
                e.dataTransfer.setData('text/plain', item.dataset.idx);
              });
              item.addEventListener('dblclick', async () => {
                const idx = parseInt(item.dataset.idx);
                const layer = await createLayer(getImageUrl(imagesData[idx]), 50 + state.layers.length * 20, 50 + state.layers.length * 20, idx);
                state.selectedLayerId = layer.id;
                render();
              });
            });
          }
          
          function updateLayerControls() {
            const layer = state.layers.find(l => l.id === state.selectedLayerId);
            if (!layer) return;
            
            const opacityVal = Math.round(layer.opacity * 100);
            document.getElementById('layer-opacity').value = opacityVal;
            document.getElementById('drawer-layer-opacity').value = opacityVal;
            document.getElementById('drawer-opacity-val').textContent = opacityVal + '%';
            document.getElementById('layer-blend-mode').value = layer.blendMode || 'source-over';
          }
          
          function openEffectsModal(layerId) {
            const layer = state.layers.find(l => l.id === layerId);
            if (!layer) return;
            
            state.modalLayerId = layerId;
            const modal = document.getElementById('layer-effects-modal');
            document.getElementById('modal-layer-name').textContent = layer.name + ' - Effects';
            
            document.getElementById('effect-stroke-enabled').checked = layer.strokeWidth > 0;
            document.getElementById('effect-stroke-width').value = layer.strokeWidth || 2;
            document.getElementById('effect-stroke-width-val').textContent = layer.strokeWidth || 2;
            document.getElementById('effect-stroke-color').value = layer.strokeColor;
            
            document.getElementById('effect-shadow-enabled').checked = layer.shadow;
            document.getElementById('effect-shadow-blur').value = layer.shadowBlur;
            document.getElementById('effect-shadow-blur-val').textContent = layer.shadowBlur;
            document.getElementById('effect-shadow-x').value = layer.shadowOffsetX;
            document.getElementById('effect-shadow-x-val').textContent = layer.shadowOffsetX;
            document.getElementById('effect-shadow-y').value = layer.shadowOffsetY;
            document.getElementById('effect-shadow-y-val').textContent = layer.shadowOffsetY;
            let shadowOpacity = 50;
            try {
              const parts = layer.shadowColor.split(',');
              if (parts.length >= 4) {
                shadowOpacity = Math.round(parseFloat(parts[3]) * 100);
              }
            } catch(e) {}
            document.getElementById('effect-shadow-opacity').value = shadowOpacity;
            document.getElementById('effect-shadow-opacity-val').textContent = shadowOpacity;
            document.getElementById('effect-shadow-color').value = '#000000';
            
            document.getElementById('effect-rotation').value = layer.rotation;
            document.getElementById('effect-rotation-val').textContent = Math.round(layer.rotation);
            document.getElementById('effect-scale-x').value = layer.scaleX * 100;
            document.getElementById('effect-scale-x-val').textContent = Math.round(layer.scaleX * 100);
            document.getElementById('effect-scale-y').value = layer.scaleY * 100;
            document.getElementById('effect-scale-y-val').textContent = Math.round(layer.scaleY * 100);
            document.getElementById('effect-skew-x').value = layer.skewX * 100;
            document.getElementById('effect-skew-x-val').textContent = Math.round(layer.skewX * 100);
            document.getElementById('effect-skew-y').value = layer.skewY * 100;
            document.getElementById('effect-skew-y-val').textContent = Math.round(layer.skewY * 100);
            
            modal.style.display = 'flex';
          }
          
          function closeEffectsModal() {
            document.getElementById('layer-effects-modal').style.display = 'none';
            state.modalLayerId = null;
          }
          
          function updateModalLayer(property, value) {
            const layer = state.layers.find(l => l.id === state.modalLayerId);
            if (!layer) return;
            layer[property] = value;
            markLayerListDirty();
            scheduleRender();
          }
          
          // Prevent context menu on canvas for right-click panning
          canvas.addEventListener('contextmenu', (e) => e.preventDefault());
          
          // Cache canvas rect to avoid expensive getBoundingClientRect calls
          let cachedCanvasRect = null;
          let cachedCanvasRectTime = 0;
          function getCanvasRect() {
            const now = performance.now();
            if (!cachedCanvasRect || now - cachedCanvasRectTime > 100) {
              cachedCanvasRect = canvas.getBoundingClientRect();
              cachedCanvasRectTime = now;
            }
            return cachedCanvasRect;
          }
          
          // Event handlers
          canvas.addEventListener('mousedown', (e) => {
            const rect = getCanvasRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            // Right-click drag to pan (works in any tool mode)
            if (e.button === 2) {
              state.isDragging = true;
              state.dragType = 'pan';
              state.dragStart = { x: x - state.panX, y: y - state.panY };
              return;
            }
            
            if (state.tool === 'select') {
              const handle = getHandleAtPoint(x, y);
              if (handle) {
                const layer = state.layers.find(l => l.id === state.selectedLayerId);
                state.isDragging = true;
                state.dragType = 'transform';
                state.activeHandle = handle;
                state.handleStartData = {
                  x: layer.x,
                  y: layer.y,
                  width: layer.width,
                  height: layer.height,
                  scaleX: layer.scaleX,
                  scaleY: layer.scaleY,
                  skewX: layer.skewX || 0,
                  skewY: layer.skewY || 0,
                  rotation: layer.rotation,
                  mouseX: x,
                  mouseY: y
                };
              } else {
                const layer = getLayerAtPoint(x, y);
                if (layer) {
                  state.selectedLayerId = layer.id;
                  state.isDragging = true;
                  state.dragType = 'move';
                  state.dragStart = { x: x - (layer.x * state.zoom + state.panX), y: y - (layer.y * state.zoom + state.panY) };
                  markLayerListDirty();
                  updateLayerControls();
                } else {
                  state.selectedLayerId = null;
                  markLayerListDirty();
                }
              }
              scheduleRender();
            } else if (state.tool === 'pan') {
              state.isDragging = true;
              state.dragType = 'pan';
              state.dragStart = { x: x - state.panX, y: y - state.panY };
            } else if (state.tool === 'brush') {
              // Reset any stale painting state before starting new stroke
              state.isPainting = false;
              state.isDragging = false;
              
              if (state.brushTool === 'clone' && e.altKey) {
                // Alt+click sets clone source
                const wx = (x - state.panX) / state.zoom;
                const wy = (y - state.panY) / state.zoom;
                const sourceLayer = getLayerAtPoint(x, y);
                state.cloneSource = {
                  sourceX: wx,
                  sourceY: wy,
                  sourceLayer: sourceLayer,
                  startWorldX: 0,
                  startWorldY: 0
                };
                document.getElementById('clone-status').textContent = 'Source set ✓';
                return;
              }
              
              // For clone, ALWAYS record the starting WORLD position at start of each stroke
              if (state.brushTool === 'clone' && state.cloneSource) {
                const wx = (x - state.panX) / state.zoom;
                const wy = (y - state.panY) / state.zoom;
                state.cloneSource.startWorldX = wx;
                state.cloneSource.startWorldY = wy;
              }
              
              history.save();
              state.isPainting = true;
              state.lastPaintPos = { x, y };
              paintAt(x, y, true);
            }
          });
          
          canvas.addEventListener('mousemove', (e) => {
            const rect = getCanvasRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            if (state.isDragging) {
              if (state.dragType === 'move' && state.selectedLayerId) {
                const layer = state.layers.find(l => l.id === state.selectedLayerId);
                if (layer) {
                  const rawX = (x - state.dragStart.x - state.panX) / state.zoom;
                  const rawY = (y - state.dragStart.y - state.panY) / state.zoom;
                  const snap = calculateSnap(layer, rawX, rawY);
                  layer.x = snap.x;
                  layer.y = snap.y;
                  state.snapGuides = snap.guides;
                  scheduleRender();
                }
              } else if (state.dragType === 'transform' && state.selectedLayerId) {
                const layer = state.layers.find(l => l.id === state.selectedLayerId);
                if (layer && state.handleStartData) {
                  const dx = (x - state.handleStartData.mouseX) / state.zoom;
                  const dy = (y - state.handleStartData.mouseY) / state.zoom;
                  const startData = state.handleStartData;
                  const shiftKey = e.shiftKey;
                  const ctrlKey = e.ctrlKey || e.metaKey;
                  
                  if (state.activeHandle === 'se') {
                    if (ctrlKey) {
                      // Ctrl+corner: skew
                      layer.skewX = startData.skewX + dx / (layer.height * layer.scaleY);
                      layer.skewY = startData.skewY + dy / (layer.width * layer.scaleX);
                    } else {
                      let newScaleX = Math.max(0.1, startData.scaleX + dx / layer.width);
                      let newScaleY = Math.max(0.1, startData.scaleY + dy / layer.height);
                      if (shiftKey) {
                        const avgScale = (newScaleX / startData.scaleX + newScaleY / startData.scaleY) / 2;
                        newScaleX = startData.scaleX * avgScale;
                        newScaleY = startData.scaleY * avgScale;
                      }
                      layer.scaleX = newScaleX;
                      layer.scaleY = newScaleY;
                    }
                  } else if (state.activeHandle === 'nw') {
                    if (ctrlKey) {
                      layer.skewX = startData.skewX - dx / (layer.height * layer.scaleY);
                      layer.skewY = startData.skewY - dy / (layer.width * layer.scaleX);
                    } else {
                      let newScaleX = Math.max(0.1, startData.scaleX - dx / layer.width);
                      let newScaleY = Math.max(0.1, startData.scaleY - dy / layer.height);
                      if (shiftKey) {
                        const avgScale = (newScaleX / startData.scaleX + newScaleY / startData.scaleY) / 2;
                        newScaleX = startData.scaleX * avgScale;
                        newScaleY = startData.scaleY * avgScale;
                      }
                      layer.x = startData.x + (startData.scaleX - newScaleX) * layer.width;
                      layer.y = startData.y + (startData.scaleY - newScaleY) * layer.height;
                      layer.scaleX = newScaleX;
                      layer.scaleY = newScaleY;
                    }
                  } else if (state.activeHandle === 'ne') {
                    if (ctrlKey) {
                      layer.skewX = startData.skewX + dx / (layer.height * layer.scaleY);
                      layer.skewY = startData.skewY - dy / (layer.width * layer.scaleX);
                    } else {
                      let newScaleX = Math.max(0.1, startData.scaleX + dx / layer.width);
                      let newScaleY = Math.max(0.1, startData.scaleY - dy / layer.height);
                      if (shiftKey) {
                        const avgScale = (newScaleX / startData.scaleX + newScaleY / startData.scaleY) / 2;
                        newScaleX = startData.scaleX * avgScale;
                        newScaleY = startData.scaleY * avgScale;
                      }
                      layer.y = startData.y + (startData.scaleY - newScaleY) * layer.height;
                      layer.scaleX = newScaleX;
                      layer.scaleY = newScaleY;
                    }
                  } else if (state.activeHandle === 'sw') {
                    if (ctrlKey) {
                      layer.skewX = startData.skewX - dx / (layer.height * layer.scaleY);
                      layer.skewY = startData.skewY + dy / (layer.width * layer.scaleX);
                    } else {
                      let newScaleX = Math.max(0.1, startData.scaleX - dx / layer.width);
                      let newScaleY = Math.max(0.1, startData.scaleY + dy / layer.height);
                      if (shiftKey) {
                        const avgScale = (newScaleX / startData.scaleX + newScaleY / startData.scaleY) / 2;
                        newScaleX = startData.scaleX * avgScale;
                        newScaleY = startData.scaleY * avgScale;
                      }
                      layer.x = startData.x + (startData.scaleX - newScaleX) * layer.width;
                      layer.scaleX = newScaleX;
                      layer.scaleY = newScaleY;
                    }
                  } else if (state.activeHandle === 'rotate') {
                    const cx = layer.x + layer.width * layer.scaleX * 0.5;
                    const cy = layer.y + layer.height * layer.scaleY * 0.5;
                    const startAngle = Math.atan2(startData.mouseY / state.zoom - cy + state.panY / state.zoom, startData.mouseX / state.zoom - cx + state.panX / state.zoom);
                    const currentAngle = Math.atan2(y / state.zoom - cy + state.panY / state.zoom, x / state.zoom - cx + state.panX / state.zoom);
                    let newRotation = startData.rotation + (currentAngle - startAngle) * 180 / Math.PI;
                    if (shiftKey) {
                      // Snap to 45-degree increments
                      newRotation = Math.round(newRotation / 45) * 45;
                    }
                    layer.rotation = newRotation;
                  }
                  scheduleRender();
                }
              } else if (state.dragType === 'pan') {
                state.panX = x - state.dragStart.x;
                state.panY = y - state.dragStart.y;
                scheduleRender();
              }
            } else if (state.isPainting) {
              paintLine(state.lastPaintPos.x, state.lastPaintPos.y, x, y);
              state.lastPaintPos = { x, y };
            }
          });
          
          canvas.addEventListener('mouseup', () => {
            state.isDragging = false;
            state.isPainting = false;
            state.lastPaintPos = null;
            state.activeHandle = null;
            state.handleStartData = null;
            state.snapGuides = null;
            scheduleRender();
          });
          
          canvas.addEventListener('mouseleave', () => {
            state.isDragging = false;
            state.isPainting = false;
            state.snapGuides = null;
          });
          
          // Window-level mouseup to catch releases outside canvas
          window.addEventListener('mouseup', () => {
            state.isDragging = false;
            state.isPainting = false;
            state.lastPaintPos = null;
            state.activeHandle = null;
            state.handleStartData = null;
            state.snapGuides = null;
          });
          
          canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            const rect = getCanvasRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const zoomFactor = e.deltaY > 0 ? 0.9 : 1.1;
            const newZoom = Math.max(0.1, Math.min(10, state.zoom * zoomFactor));
            
            state.panX = x - (x - state.panX) * (newZoom / state.zoom);
            state.panY = y - (y - state.panY) * (newZoom / state.zoom);
            state.zoom = newZoom;
            
            scheduleRender();
          });
          
          // Drop handler
          canvas.addEventListener('dragover', (e) => e.preventDefault());
          canvas.addEventListener('drop', async (e) => {
            e.preventDefault();
            const idx = parseInt(e.dataTransfer.getData('text/plain'));
            if (!isNaN(idx) && imagesData[idx]) {
              const rect = getCanvasRect();
              const x = (e.clientX - rect.left - state.panX) / state.zoom;
              const y = (e.clientY - rect.top - state.panY) / state.zoom;
              const layer = await createLayer(getImageUrl(imagesData[idx]), x - 50, y - 50, idx);
              state.selectedLayerId = layer.id;
              render();
            }
          });
          
          function screenToLayerCoords(x, y, layer) {
            // Convert screen coords to world coords
            const wx = (x - state.panX) / state.zoom;
            const wy = (y - state.panY) / state.zoom;
            
            // Get layer center
            const cx = layer.x + layer.width * layer.scaleX * 0.5;
            const cy = layer.y + layer.height * layer.scaleY * 0.5;
            
            // Translate to layer center
            let lx = wx - cx;
            let ly = wy - cy;
            
            // Inverse rotation
            const rad = -layer.rotation * Math.PI / 180;
            const cos = Math.cos(rad);
            const sin = Math.sin(rad);
            const rx = lx * cos - ly * sin;
            const ry = lx * sin + ly * cos;
            lx = rx;
            ly = ry;
            
            // Inverse skew
            const skewX = layer.skewX || 0;
            const skewY = layer.skewY || 0;
            const preLx = lx;
            lx = lx - ly * skewX;
            ly = ly - preLx * skewY;
            
            // Inverse scale and translate to layer origin
            lx = lx / layer.scaleX + layer.width * 0.5;
            ly = ly / layer.scaleY + layer.height * 0.5;
            
            return { x: lx, y: ly };
          }
          
          function paintAt(x, y, isFirst = false) {
            const layer = state.layers.find(l => l.id === state.selectedLayerId);
            if (!layer) return;
            
            // If editing mask, paint on mask canvas
            if (state.editingMask && layer.maskCanvas) {
              const local = screenToLayerCoords(x, y, layer);
              const localX = local.x;
              const localY = local.y;
              const mctx = layer.maskCtx;
              const radius = state.brushSize * 0.5;
              const hardness = state.brushHardness / 100;
              
              // Mask uses alpha: opaque white=visible, transparent=hidden
              // Brush paints to HIDE (erase mask), Eraser paints to REVEAL (restore mask)
              if (state.brushTool === 'eraser') {
                // Eraser reveals (paints white to restore visibility)
                const gradient = mctx.createRadialGradient(localX, localY, 0, localX, localY, radius);
                gradient.addColorStop(0, 'rgba(255,255,255,1)');
                gradient.addColorStop(hardness, 'rgba(255,255,255,1)');
                gradient.addColorStop(1, 'rgba(255,255,255,0)');
                mctx.fillStyle = gradient;
                mctx.beginPath();
                mctx.arc(localX, localY, radius, 0, Math.PI * 2);
                mctx.fill();
              } else {
                // Brush hides (erases mask to make transparent)
                mctx.globalCompositeOperation = 'destination-out';
                const gradient = mctx.createRadialGradient(localX, localY, 0, localX, localY, radius);
                gradient.addColorStop(0, 'rgba(0,0,0,1)');
                gradient.addColorStop(hardness, 'rgba(0,0,0,1)');
                gradient.addColorStop(1, 'rgba(0,0,0,0)');
                mctx.fillStyle = gradient;
                mctx.beginPath();
                mctx.arc(localX, localY, radius, 0, Math.PI * 2);
                mctx.fill();
                mctx.globalCompositeOperation = 'source-over';
              }
              scheduleRender();
              return;
            }
            
            // For eraser on image layers, convert to editable canvas
            if (layer.type === 'image' && state.brushTool === 'eraser') {
              // Convert image layer to editable canvas if not already
              if (!layer.drawCanvas) {
                const drawCanvas = document.createElement('canvas');
                drawCanvas.width = layer.width;
                drawCanvas.height = layer.height;
                const dctx = drawCanvas.getContext('2d');
                dctx.drawImage(layer.image, 0, 0, layer.width, layer.height);
                layer.drawCanvas = drawCanvas;
                layer.drawCtx = dctx;
                layer.image = drawCanvas;
              }
            }
            
            // Painting requires a draw canvas
            if (!layer.drawCanvas) return;
            
            // Always get fresh context reference in case it was invalidated
            if (!layer.drawCtx || layer.drawCtx.canvas !== layer.drawCanvas) {
              layer.drawCtx = layer.drawCanvas.getContext('2d');
            }
            
            const local = screenToLayerCoords(x, y, layer);
            const localX = local.x;
            const localY = local.y;
            
            const dctx = layer.drawCtx;
            if (!dctx) return;
            
            const radius = state.brushSize * 0.5;
            
            // Apply brush opacity
            const brushAlpha = state.brushOpacity / 100;
            
            if (state.brushTool === 'eraser') {
              dctx.save();
              dctx.globalCompositeOperation = 'destination-out';
              dctx.globalAlpha = brushAlpha;
              const gradient = dctx.createRadialGradient(localX, localY, 0, localX, localY, radius);
              const hardness = state.brushHardness / 100;
              gradient.addColorStop(0, 'rgba(0,0,0,1)');
              gradient.addColorStop(hardness, 'rgba(0,0,0,1)');
              gradient.addColorStop(1, 'rgba(0,0,0,0)');
              dctx.fillStyle = gradient;
              dctx.beginPath();
              dctx.arc(localX, localY, radius, 0, Math.PI * 2);
              dctx.fill();
              dctx.restore();
            } else if (state.brushTool === 'clone' && state.cloneSource) {
              const src = state.cloneSource;
              
              // Calculate current world position
              const wx = (x - state.panX) / state.zoom;
              const wy = (y - state.panY) / state.zoom;
              
              // Offset from where we started painting (in world coords)
              const offsetX = wx - src.startWorldX;
              const offsetY = wy - src.startWorldY;
              
              // Source sample position in world coords
              const srcWorldX = src.sourceX + offsetX;
              const srcWorldY = src.sourceY + offsetY;
              
              // Create a brush stamp canvas with hardness gradient
              const stampCanvas = document.createElement('canvas');
              const stampSize = Math.ceil(radius * 2) + 2;
              stampCanvas.width = stampSize;
              stampCanvas.height = stampSize;
              const sctx = stampCanvas.getContext('2d');
              const stampCenter = stampSize / 2;
              
              // Get source layer (refresh reference in case layers array changed)
              let sourceLayer = null;
              if (src.sourceLayer && src.sourceLayer.id) {
                sourceLayer = state.layers.find(l => l.id === src.sourceLayer.id);
              }
              
              let drewSomething = false;
              
              if (state.cloneAllLayers) {
                // Create a composite of all visible layers at world coordinates
                const sampleSize = Math.max(layer.drawCanvas.width, layer.drawCanvas.height) * 2;
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = sampleSize;
                tempCanvas.height = sampleSize;
                const tctx = tempCanvas.getContext('2d');
                
                // Center the composite around the source sample point
                const compositeOffsetX = sampleSize / 2 - srcWorldX;
                const compositeOffsetY = sampleSize / 2 - srcWorldY;
                
                // Render all visible layers INCLUDING current with their transforms
                for (const l of state.layers) {
                  if (!l.visible) continue;
                  tctx.save();
                  tctx.globalAlpha = l.opacity;
                  tctx.globalCompositeOperation = l.blendMode || 'source-over';
                  tctx.translate(compositeOffsetX + l.x + l.width * l.scaleX * 0.5, 
                                 compositeOffsetY + l.y + l.height * l.scaleY * 0.5);
                  tctx.rotate(l.rotation * Math.PI / 180);
                  tctx.transform(1, l.skewY || 0, l.skewX || 0, 1, 0, 0);
                  tctx.scale(l.scaleX, l.scaleY);
                  tctx.translate(-l.width * 0.5, -l.height * 0.5);
                  tctx.drawImage(l.image, 0, 0, l.width, l.height);
                  tctx.restore();
                  drewSomething = true;
                }
                
                if (drewSomething) {
                  // Draw source onto stamp canvas
                  sctx.drawImage(tempCanvas, stampCenter - sampleSize / 2, stampCenter - sampleSize / 2);
                }
              } else if (sourceLayer && sourceLayer.image) {
                // Sample from single source layer
                const sl = sourceLayer;
                const srcLocalX = srcWorldX - sl.x;
                const srcLocalY = srcWorldY - sl.y;
                sctx.drawImage(sl.image, stampCenter - srcLocalX, stampCenter - srcLocalY);
                drewSomething = true;
              }
              
              if (drewSomething) {
                // Apply hardness gradient as mask
                sctx.globalCompositeOperation = 'destination-in';
                const hardness = state.brushHardness / 100;
                const gradient = sctx.createRadialGradient(stampCenter, stampCenter, 0, stampCenter, stampCenter, radius);
                gradient.addColorStop(0, 'rgba(0,0,0,1)');
                gradient.addColorStop(hardness, 'rgba(0,0,0,1)');
                gradient.addColorStop(1, 'rgba(0,0,0,0)');
                sctx.fillStyle = gradient;
                sctx.fillRect(0, 0, stampSize, stampSize);
                
                // Draw stamp onto layer with opacity and blend mode
                dctx.save();
                dctx.globalCompositeOperation = state.brushBlend;
                dctx.globalAlpha = brushAlpha;
                dctx.drawImage(stampCanvas, localX - stampCenter, localY - stampCenter);
                dctx.restore();
              }
            } else {
              // Regular brush
              dctx.save();
              dctx.globalCompositeOperation = state.brushBlend;
              dctx.globalAlpha = brushAlpha;
              const gradient = dctx.createRadialGradient(localX, localY, 0, localX, localY, radius);
              const hardness = state.brushHardness / 100;
              gradient.addColorStop(0, state.brushColor);
              gradient.addColorStop(hardness, state.brushColor);
              gradient.addColorStop(1, state.brushColor + '00');
              dctx.fillStyle = gradient;
              dctx.beginPath();
              dctx.arc(localX, localY, radius, 0, Math.PI * 2);
              dctx.fill();
              dctx.restore();
            }
            scheduleRender();
          }
          
          function paintLine(x1, y1, x2, y2) {
            const dist = Math.sqrt((x2-x1)**2 + (y2-y1)**2);
            const stepSize = state.brushSize * 0.25;
            const steps = Math.max(1, Math.floor(dist / stepSize));
            for (let i = 0; i <= steps; i++) {
              const t = i / steps;
              paintAt(x1 + (x2-x1)*t, y1 + (y2-y1)*t);
            }
          }
          
          // Toolbar handlers
          document.querySelectorAll('.tool-btn').forEach(btn => {
            btn.addEventListener('click', () => {
              document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
              btn.classList.add('active');
              state.tool = btn.dataset.tool;
              document.getElementById('brush-toolbar').style.display = state.tool === 'brush' ? 'flex' : 'none';
            });
          });
          
          document.getElementById('bg-color').addEventListener('input', (e) => {
            state.bgColor = e.target.value;
            scheduleRender();
          });
          
          document.getElementById('bg-transparent').addEventListener('change', (e) => {
            state.bgTransparent = e.target.checked;
            scheduleRender();
          });
          
          document.getElementById('export-width').addEventListener('input', (e) => {
            state.exportWidth = Math.max(0, parseInt(e.target.value) || 0);
            scheduleRender();
          });
          
          document.getElementById('export-height').addEventListener('input', (e) => {
            state.exportHeight = Math.max(0, parseInt(e.target.value) || 0);
            scheduleRender();
          });
          
          document.getElementById('snap-enabled').addEventListener('change', (e) => {
            state.snapEnabled = e.target.checked;
          });
          
          document.getElementById('brush-size').addEventListener('input', (e) => {
            state.brushSize = parseInt(e.target.value);
            document.getElementById('brush-size-val').textContent = state.brushSize;
          });
          
          document.getElementById('brush-hardness').addEventListener('input', (e) => {
            state.brushHardness = parseInt(e.target.value);
          });
          
          document.getElementById('brush-opacity').addEventListener('input', (e) => {
            state.brushOpacity = parseInt(e.target.value);
            document.getElementById('brush-opacity-val').textContent = state.brushOpacity + '%';
          });
          
          document.getElementById('brush-tool').addEventListener('change', (e) => {
            state.brushTool = e.target.value;
            document.getElementById('clone-options').style.display = state.brushTool === 'clone' ? 'inline' : 'none';
            if (state.brushTool === 'clone') {
              state.cloneSource = null;
              document.getElementById('clone-status').textContent = 'Alt+Click to set source';
            }
          });
          
          document.getElementById('clone-all-layers').addEventListener('change', (e) => {
            state.cloneAllLayers = e.target.checked;
          });
          
          document.getElementById('brush-blend').addEventListener('change', (e) => {
            state.brushBlend = e.target.value;
          });
          
          document.getElementById('brush-color').addEventListener('input', (e) => {
            state.brushColor = e.target.value;
          });
          
          document.getElementById('layer-opacity').addEventListener('input', (e) => {
            const layer = state.layers.find(l => l.id === state.selectedLayerId);
            if (layer) {
              layer.opacity = parseInt(e.target.value) / 100;
              document.getElementById('drawer-layer-opacity').value = e.target.value;
              document.getElementById('drawer-opacity-val').textContent = e.target.value + '%';
              scheduleRender();
            }
          });
          
          document.getElementById('drawer-layer-opacity').addEventListener('input', (e) => {
            const layer = state.layers.find(l => l.id === state.selectedLayerId);
            if (layer) {
              layer.opacity = parseInt(e.target.value) / 100;
              document.getElementById('layer-opacity').value = e.target.value;
              document.getElementById('drawer-opacity-val').textContent = e.target.value + '%';
              scheduleRender();
            }
          });
          
          document.getElementById('layer-blend-mode').addEventListener('change', (e) => {
            const layer = state.layers.find(l => l.id === state.selectedLayerId);
            if (layer) {
              layer.blendMode = e.target.value;
              scheduleRender();
            }
          });
          
          // Modal event handlers
          document.getElementById('modal-close').addEventListener('click', closeEffectsModal);
          document.getElementById('layer-effects-modal').addEventListener('click', (e) => {
            if (e.target.id === 'layer-effects-modal') closeEffectsModal();
          });
          
          document.getElementById('effect-stroke-enabled').addEventListener('change', (e) => {
            updateModalLayer('strokeWidth', e.target.checked ? parseInt(document.getElementById('effect-stroke-width').value) : 0);
          });
          document.getElementById('effect-stroke-width').addEventListener('input', (e) => {
            document.getElementById('effect-stroke-width-val').textContent = e.target.value;
            if (document.getElementById('effect-stroke-enabled').checked) {
              updateModalLayer('strokeWidth', parseInt(e.target.value));
            }
          });
          document.getElementById('effect-stroke-color').addEventListener('input', (e) => {
            updateModalLayer('strokeColor', e.target.value);
          });
          
          document.getElementById('effect-shadow-enabled').addEventListener('change', (e) => {
            updateModalLayer('shadow', e.target.checked);
          });
          document.getElementById('effect-shadow-blur').addEventListener('input', (e) => {
            document.getElementById('effect-shadow-blur-val').textContent = e.target.value;
            updateModalLayer('shadowBlur', parseInt(e.target.value));
          });
          document.getElementById('effect-shadow-x').addEventListener('input', (e) => {
            document.getElementById('effect-shadow-x-val').textContent = e.target.value;
            updateModalLayer('shadowOffsetX', parseInt(e.target.value));
          });
          document.getElementById('effect-shadow-y').addEventListener('input', (e) => {
            document.getElementById('effect-shadow-y-val').textContent = e.target.value;
            updateModalLayer('shadowOffsetY', parseInt(e.target.value));
          });
          document.getElementById('effect-shadow-opacity').addEventListener('input', (e) => {
            document.getElementById('effect-shadow-opacity-val').textContent = e.target.value;
            const opacity = parseInt(e.target.value) / 100;
            const color = document.getElementById('effect-shadow-color').value;
            const r = parseInt(color.slice(1,3), 16);
            const g = parseInt(color.slice(3,5), 16);
            const b = parseInt(color.slice(5,7), 16);
            updateModalLayer('shadowColor', 'rgba(' + r + ',' + g + ',' + b + ',' + opacity + ')');
          });
          document.getElementById('effect-shadow-color').addEventListener('input', (e) => {
            const opacity = parseInt(document.getElementById('effect-shadow-opacity').value) / 100;
            const color = e.target.value;
            const r = parseInt(color.slice(1,3), 16);
            const g = parseInt(color.slice(3,5), 16);
            const b = parseInt(color.slice(5,7), 16);
            updateModalLayer('shadowColor', 'rgba(' + r + ',' + g + ',' + b + ',' + opacity + ')');
          });
          
          document.getElementById('effect-rotation').addEventListener('input', (e) => {
            document.getElementById('effect-rotation-val').textContent = e.target.value;
            updateModalLayer('rotation', parseInt(e.target.value));
          });
          document.getElementById('effect-scale-x').addEventListener('input', (e) => {
            document.getElementById('effect-scale-x-val').textContent = e.target.value;
            updateModalLayer('scaleX', parseInt(e.target.value) / 100);
          });
          document.getElementById('effect-scale-y').addEventListener('input', (e) => {
            document.getElementById('effect-scale-y-val').textContent = e.target.value;
            updateModalLayer('scaleY', parseInt(e.target.value) / 100);
          });
          document.getElementById('effect-skew-x').addEventListener('input', (e) => {
            document.getElementById('effect-skew-x-val').textContent = e.target.value;
            updateModalLayer('skewX', parseInt(e.target.value) / 100);
          });
          document.getElementById('effect-skew-y').addEventListener('input', (e) => {
            document.getElementById('effect-skew-y-val').textContent = e.target.value;
            updateModalLayer('skewY', parseInt(e.target.value) / 100);
          });
          
          // Helper to create export canvas from current layers
          function createExportCanvas() {
            const visibleLayers = state.layers.filter(l => l.visible);
            let minX, minY, maxX, maxY, padding;
            
            // Use export size if set, otherwise auto-calculate from layers
            if (state.exportWidth > 0 && state.exportHeight > 0) {
              minX = 0;
              minY = 0;
              maxX = state.exportWidth;
              maxY = state.exportHeight;
              padding = 0;
            } else {
              minX = Infinity; minY = Infinity; maxX = -Infinity; maxY = -Infinity;
              for (const layer of visibleLayers) {
                minX = Math.min(minX, layer.x);
                minY = Math.min(minY, layer.y);
                maxX = Math.max(maxX, layer.x + layer.width * layer.scaleX);
                maxY = Math.max(maxY, layer.y + layer.height * layer.scaleY);
              }
              if (visibleLayers.length === 0) {
                minX = 0; minY = 0; maxX = 512; maxY = 512;
              }
              padding = 20;
            }
            
            const exportCanvas = document.createElement('canvas');
            exportCanvas.width = maxX - minX + padding * 2;
            exportCanvas.height = maxY - minY + padding * 2;
            const ectx = exportCanvas.getContext('2d');
            
            if (!state.bgTransparent) {
              ectx.fillStyle = state.bgColor;
              ectx.fillRect(0, 0, exportCanvas.width, exportCanvas.height);
            }
            
            ectx.translate(-minX + padding, -minY + padding);
            
            for (const layer of visibleLayers) {
              ectx.save();
              ectx.translate(layer.x + layer.width * layer.scaleX * 0.5, layer.y + layer.height * layer.scaleY * 0.5);
              ectx.rotate(layer.rotation * Math.PI / 180);
              ectx.transform(1, layer.skewY, layer.skewX, 1, 0, 0);
              ectx.scale(layer.scaleX, layer.scaleY);
              ectx.translate(-layer.width * 0.5, -layer.height * 0.5);
              ectx.globalAlpha = layer.opacity;
              
              if (layer.shadow) {
                ectx.shadowBlur = layer.shadowBlur;
                ectx.shadowOffsetX = layer.shadowOffsetX;
                ectx.shadowOffsetY = layer.shadowOffsetY;
                ectx.shadowColor = layer.shadowColor;
              }
              
              ectx.drawImage(layer.image, 0, 0, layer.width, layer.height);
              ectx.restore();
            }
            
            return exportCanvas;
          }
          
          // Prepare export data for parent window (avoids tainted canvas issue)
          function getExportData() {
            const visibleLayers = state.layers.filter(l => l.visible);
            let minX, minY, maxX, maxY, padding;
            
            // Use export size if set, otherwise auto-calculate from layers
            if (state.exportWidth > 0 && state.exportHeight > 0) {
              minX = 0;
              minY = 0;
              maxX = state.exportWidth;
              maxY = state.exportHeight;
              padding = 0;
            } else {
              minX = Infinity; minY = Infinity; maxX = -Infinity; maxY = -Infinity;
              for (const layer of visibleLayers) {
                minX = Math.min(minX, layer.x);
                minY = Math.min(minY, layer.y);
                maxX = Math.max(maxX, layer.x + layer.width * layer.scaleX);
                maxY = Math.max(maxY, layer.y + layer.height * layer.scaleY);
              }
              if (visibleLayers.length === 0) {
                minX = 0; minY = 0; maxX = 512; maxY = 512;
              }
              padding = 20;
            }
            return {
              layers: visibleLayers.map(l => ({
                imageIndex: l.sourceImageIndex,  // Index into imagesData for restoration
                imageUrl: l.imageUrl,  // Current URL for export
                x: l.x, y: l.y,
                width: l.width, height: l.height,
                scaleX: l.scaleX, scaleY: l.scaleY,
                rotation: l.rotation,
                skewX: l.skewX, skewY: l.skewY,
                opacity: l.opacity,
                blendMode: l.blendMode,
                shadow: l.shadow,
                shadowBlur: l.shadowBlur,
                shadowOffsetX: l.shadowOffsetX,
                shadowOffsetY: l.shadowOffsetY,
                shadowColor: l.shadowColor,
                isDrawLayer: l.type === 'draw',
                hasDrawCanvas: !!l.drawCanvas,
                drawDataUrl: l.drawCanvas ? (() => { try { return l.drawCanvas.toDataURL(); } catch(e) { return null; } })() : null,
                maskEnabled: l.maskEnabled || false,
                maskDataUrl: (l.maskEnabled && l.maskCanvas) ? (() => { try { return l.maskCanvas.toDataURL(); } catch(e) { return null; } })() : null
              })),
              bounds: { minX, minY, maxX, maxY, padding },
              bgTransparent: state.bgTransparent,
              bgColor: state.bgColor
            };
          }
          
          // Send to Output - sends layer data to parent for reconstruction (avoids CORS issues)
          document.getElementById('send-output-btn').addEventListener('click', () => {
            const data = getExportData();
            if (!data.layers || data.layers.length === 0) {
              alert('No visible layers to export');
              return;
            }
            
            // Send layer data to parent - parent will reconstruct canvas and set manual_content
            window.parent.postMessage({
              type: 'canvas-export-request',
              action: 'output',
              data: data,
              nodeId: window.WAS_NODE_ID
            }, '*');
            
            // Visual feedback
            const btn = document.getElementById('send-output-btn');
            const originalText = btn.textContent;
            btn.textContent = '✓ Sent!';
            btn.style.background = '#22c55e';
            setTimeout(() => {
              btn.textContent = originalText;
              btn.style.background = '';
            }, 1500);
          });
          
          // Download button - sends to parent for export
          document.getElementById('export-btn').addEventListener('click', () => {
            const data = getExportData();
            window.parent.postMessage({
              type: 'canvas-export-request',
              action: 'download',
              data: data,
              nodeId: window.WAS_NODE_ID
            }, '*');
          });
          
          document.getElementById('reset-view-btn').addEventListener('click', () => {
            state.zoom = 1;
            state.panX = 0;
            state.panY = 0;
            render();
          });
          
          // Keyboard shortcuts
          document.addEventListener('keydown', (e) => {
            if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
            
            const ctrl = e.ctrlKey || e.metaKey;
            
            // Undo: Ctrl+Z
            if (ctrl && e.key === 'z' && !e.shiftKey) {
              e.preventDefault();
              history.undo();
              return;
            }
            // Redo: Ctrl+Y or Ctrl+Shift+Z
            if ((ctrl && e.key === 'y') || (ctrl && e.shiftKey && e.key === 'z')) {
              e.preventDefault();
              history.redo();
              return;
            }
            // Duplicate: Ctrl+D
            if (ctrl && e.key === 'd') {
              e.preventDefault();
              if (state.selectedLayerId) {
                duplicateLayer(state.selectedLayerId);
              }
              return;
            }
            // Delete layer
            if (e.key === 'Delete' || e.key === 'Backspace') {
              if (state.selectedLayerId) {
                history.save();
                state.layers = state.layers.filter(l => l.id !== state.selectedLayerId);
                state.selectedLayerId = null;
                markLayerListDirty();
                render();
              }
              return;
            }
            // Tool shortcuts
            if (e.key === 'v' || e.key === 'V') {
              document.querySelector('[data-tool="select"]').click();
            } else if (e.key === 'h' || e.key === 'H') {
              document.querySelector('[data-tool="pan"]').click();
            } else if (e.key === 'b' || e.key === 'B') {
              document.querySelector('[data-tool="brush"]').click();
            } else if (e.key === 'e' || e.key === 'E') {
              // E for eraser
              document.querySelector('[data-tool="brush"]').click();
              document.getElementById('brush-tool').value = 'eraser';
              state.brushTool = 'eraser';
            }
          });
          
          // Drawer toggle
          document.getElementById('drawer-toggle').addEventListener('click', () => {
            document.getElementById('layers-drawer').classList.toggle('collapsed');
          });
          
          // Mini-map click to navigate
          // miniMapCanvas is already declared at line 1262 and initialized in updateMiniMap()
          if (!miniMapCanvas) miniMapCanvas = document.getElementById('mini-map-canvas');
          miniMapCanvas.addEventListener('mousedown', (e) => {
            const transform = miniMapCanvas._miniMapTransform;
            if (!transform) return;
            
            const rect = miniMapCanvas.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const clickY = e.clientY - rect.top;
            
            // Convert click to world coordinates
            const worldX = (clickX - transform.offsetX) / transform.scale + transform.minX;
            const worldY = (clickY - transform.offsetY) / transform.scale + transform.minY;
            
            // Center viewport on clicked point
            state.panX = -worldX * state.zoom + canvas.width / 2;
            state.panY = -worldY * state.zoom + canvas.height / 2;
            scheduleRender();
          });
          
          // Add new drawing layer
          document.getElementById('add-new-layer').addEventListener('click', () => {
            // Calculate bounding box of all existing layers
            let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
            for (const l of state.layers) {
              const lw = l.width * Math.abs(l.scaleX || 1);
              const lh = l.height * Math.abs(l.scaleY || 1);
              minX = Math.min(minX, l.x);
              minY = Math.min(minY, l.y);
              maxX = Math.max(maxX, l.x + lw);
              maxY = Math.max(maxY, l.y + lh);
            }
            // Default size if no layers exist
            let newWidth = 512, newHeight = 512, newX = 50, newY = 50;
            if (state.layers.length > 0 && isFinite(minX)) {
              newWidth = Math.max(64, Math.round(maxX - minX));
              newHeight = Math.max(64, Math.round(maxY - minY));
              newX = minX;
              newY = minY;
            }
            const layer = createDrawLayer(newWidth, newHeight);
            layer.x = newX;
            layer.y = newY;
            state.selectedLayerId = layer.id;
            
            // Save history for undo support
            history.save();
            
            // Switch to select/pointer mode directly (not brush mode)
            state.tool = 'select';
            document.querySelectorAll('.tool-btn').forEach(b => b.classList.remove('active'));
            document.querySelector('[data-tool="select"]').classList.add('active');
            document.getElementById('brush-toolbar').style.display = 'none';
            
            markLayerListDirty();
            render();
          });
          
          // Save canvas state to file
          document.getElementById('save-state-btn').addEventListener('click', () => {
            const exportData = {
              version: 1,
              layers: state.layers.map(l => {
                const layerData = {
                  type: l.type,
                  name: l.name,
                  visible: l.visible,
                  x: l.x, y: l.y,
                  width: l.width, height: l.height,
                  rotation: l.rotation,
                  scaleX: l.scaleX, scaleY: l.scaleY,
                  skewX: l.skewX, skewY: l.skewY,
                  opacity: l.opacity,
                  blendMode: l.blendMode,
                  sourceImageIndex: l.sourceImageIndex,
                  imageUrl: l.imageUrl
                };
                // Save draw layer content as data URL
                if (l.type === 'draw' && l.drawCanvas) {
                  try {
                    layerData.drawDataUrl = l.drawCanvas.toDataURL();
                  } catch (e) {
                    layerData.drawDataUrl = null;
                  }
                }
                // Save mask if exists
                if (l.maskCanvas) {
                  try {
                    layerData.maskDataUrl = l.maskCanvas.toDataURL();
                    layerData.maskEnabled = l.maskEnabled;
                  } catch (e) {}
                }
                return layerData;
              }),
              bgColor: state.bgColor,
              bgTransparent: state.bgTransparent
            };
            
            const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.download = 'canvas-state.json';
            link.href = url;
            link.click();
            URL.revokeObjectURL(url);
          });
          
          // Load canvas state from file
          document.getElementById('load-state-btn').addEventListener('click', () => {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.json';
            input.onchange = async (e) => {
              const file = e.target.files[0];
              if (!file) return;
              
              try {
                const text = await file.text();
                const savedState = JSON.parse(text);
                
                if (!savedState.layers) {
                  alert('Invalid canvas state file');
                  return;
                }
                
                // Clear existing layers
                state.layers = [];
                state.selectedLayerId = null;
                
                // Restore layers
                for (const layerData of savedState.layers) {
                  if (layerData.type === 'draw' && layerData.drawDataUrl) {
                    // Restore draw layer
                    const img = await loadImage(layerData.drawDataUrl);
                    const layer = createDrawLayer(img.width, img.height);
                    layer.drawCtx.drawImage(img, 0, 0);
                    Object.assign(layer, {
                      name: layerData.name || layer.name,
                      visible: layerData.visible !== false,
                      x: layerData.x || 0, y: layerData.y || 0,
                      rotation: layerData.rotation || 0,
                      scaleX: layerData.scaleX || 1, scaleY: layerData.scaleY || 1,
                      skewX: layerData.skewX || 0, skewY: layerData.skewY || 0,
                      opacity: layerData.opacity !== undefined ? layerData.opacity : 1,
                      blendMode: layerData.blendMode || 'source-over'
                    });
                  } else if (layerData.sourceImageIndex !== undefined && imagesData[layerData.sourceImageIndex]) {
                    // Restore image layer from source images
                    const layer = await createLayer(getImageUrl(imagesData[layerData.sourceImageIndex]), layerData.x || 0, layerData.y || 0, layerData.sourceImageIndex);
                    Object.assign(layer, {
                      name: layerData.name || layer.name,
                      visible: layerData.visible !== false,
                      rotation: layerData.rotation || 0,
                      scaleX: layerData.scaleX || 1, scaleY: layerData.scaleY || 1,
                      skewX: layerData.skewX || 0, skewY: layerData.skewY || 0,
                      opacity: layerData.opacity !== undefined ? layerData.opacity : 1,
                      blendMode: layerData.blendMode || 'source-over'
                    });
                  }
                }
                
                // Restore background settings
                if (savedState.bgColor) state.bgColor = savedState.bgColor;
                if (savedState.bgTransparent !== undefined) state.bgTransparent = savedState.bgTransparent;
                document.getElementById('bg-color').value = state.bgColor;
                document.getElementById('bg-transparent').checked = state.bgTransparent;
                
                if (state.layers.length > 0) {
                  state.selectedLayerId = state.layers[state.layers.length - 1].id;
                }
                markLayerListDirty();
                render();
              } catch (err) {
                console.error('[Canvas] Failed to load state:', err);
                alert('Failed to load canvas state: ' + err.message);
              }
            };
            input.click();
          });
          
          // Refresh source images from parent
          document.getElementById('refresh-images-btn').addEventListener('click', () => {
            window.parent.postMessage({ type: 'requestImages' }, '*');
          });
          
          // Listen for updated images from parent
          window.addEventListener('message', (e) => {
            if (e.data && e.data.type === 'updateImages' && Array.isArray(e.data.images)) {
              imagesData = e.data.images;
              updateImagesList();
              return;
            }
            // Answer the parent asking for what the canvas holds.
            if (e.data && e.data.type === 'canvas-composite-request') {
              const data = getExportData();
              const hasLayers = data.layers && data.layers.length > 0;
              window.parent.postMessage({
                type: 'canvas-composite-response',
                requestId: e.data.requestId,
                data: hasLayers ? data : null
              }, '*');
            }
          });
          
          // Initialize
          window.addEventListener('resize', resizeCanvas);
          resizeCanvas();
          updateImagesList();
          
          // Try to restore saved canvas state
          let savedState = null;
          try {
            const savedStateStr = container.dataset.savedState;
            if (savedStateStr) {
              savedState = JSON.parse(savedStateStr);
            }
          } catch (e) {
            console.warn('[Canvas] Failed to parse saved state:', e);
          }
          
          // Check if saved state matches current imagesData (same session/files)
          // If mismatch, discard saved state and start fresh
          if (savedState && savedState.layers && imagesData.length > 0) {
            const savedImageUrls = savedState.layers
              .filter(l => l.imageUrl)
              .map(l => l.imageUrl);
            const currentSubfolder = imagesData[0]?.subfolder || '';
            const savedHasDifferentSession = savedImageUrls.length > 0 && 
              !savedImageUrls.some(url => url.includes(currentSubfolder));
            
            if (savedHasDifferentSession) {
              console.warn('[Canvas] The saved layers belong to another run, so the canvas starts fresh');
              savedState = null;
            }
          }
          
          if (savedState && savedState.layers && savedState.layers.length > 0) {
            // Restore layers from saved state
            
            // Restore background settings
            if (savedState.bgTransparent !== undefined) state.bgTransparent = savedState.bgTransparent;
            if (savedState.bgColor) state.bgColor = savedState.bgColor;
            document.getElementById('bg-transparent').checked = state.bgTransparent;
            document.getElementById('bg-color').value = state.bgColor;
            
            // Restore each layer
            const restorePromises = savedState.layers.map(async (layerData, idx) => {
              try {
                if (layerData.isDrawLayer && layerData.drawDataUrl) {
                  // Restore draw layer
                  const layer = createDrawLayer(layerData.width || 512, layerData.height || 512);
                  const img = await loadImage(layerData.drawDataUrl);
                  layer.drawCtx.drawImage(img, 0, 0);
                  // Restore properties
                  Object.assign(layer, {
                    x: layerData.x || 0,
                    y: layerData.y || 0,
                    rotation: layerData.rotation || 0,
                    scaleX: layerData.scaleX || 1,
                    scaleY: layerData.scaleY || 1,
                    skewX: layerData.skewX || 0,
                    skewY: layerData.skewY || 0,
                    opacity: layerData.opacity || 1,
                    blendMode: layerData.blendMode || 'source-over',
                  });
                  return layer;
                } else if (layerData.imageIndex !== undefined && layerData.imageIndex >= 0 && imagesData[layerData.imageIndex]) {
                  // Restore image layer using fresh URL from imagesData
                  const imgIdx = layerData.imageIndex;
                  const layer = await createLayer(getImageUrl(imagesData[imgIdx]), layerData.x || 0, layerData.y || 0, imgIdx);
                  Object.assign(layer, {
                    rotation: layerData.rotation || 0,
                    scaleX: layerData.scaleX || 1,
                    scaleY: layerData.scaleY || 1,
                    skewX: layerData.skewX || 0,
                    skewY: layerData.skewY || 0,
                    opacity: layerData.opacity || 1,
                    blendMode: layerData.blendMode || 'source-over',
                    shadow: layerData.shadow || false,
                    shadowBlur: layerData.shadowBlur || 10,
                    shadowOffsetX: layerData.shadowOffsetX || 5,
                    shadowOffsetY: layerData.shadowOffsetY || 5,
                    shadowColor: layerData.shadowColor || 'rgba(0,0,0,0.5)',
                  });
                  return layer;
                }
              } catch (e) {
                console.error('[Canvas] Failed to restore layer:', e);
              }
              return null;
            });
            
            Promise.all(restorePromises).then(layers => {
              const validLayers = layers.filter(l => l);
              if (validLayers.length > 0) {
                state.selectedLayerId = validLayers[validLayers.length - 1].id;
                markLayerListDirty();
                render();
                // Save initial state for undo support
                history.save();
              } else if (imagesData.length > 0) {
                // Restore failed (e.g. temp files deleted), fall back to fresh images
                console.warn('[Canvas] The saved layers could not be restored, so the source images are loaded instead');
                createLayer(getImageUrl(imagesData[0]), 50, 50, 0).then(layer => {
                  state.selectedLayerId = layer.id;
                  markLayerListDirty();
                  render();
                  // Save initial state for undo support
                  history.save();
                }).catch(e => console.error('[Canvas] Failed to load fresh image:', e));
              } else {
                markLayerListDirty();
                render();
              }
            });
          } else if (imagesData.length > 0) {
            // Auto-add first image if available and no saved state
            createLayer(getImageUrl(imagesData[0]), 0, 0, 0).then(layer => {
              state.selectedLayerId = layer.id;
              fitToLayers();
              // Save initial state for undo support
              history.save();
            }).catch(e => console.error('[Canvas] Failed to auto-add first image:', e));
          } else {
            // No layers - save empty initial state for undo support
            history.save();
          }
          
          } catch (e) {
            console.error('[Canvas] Script error:', e);
          }
        })();
      <\/script>
    `;
  }

  /**
   * Content marker prefix for canvas view
   */
  static getContentMarker() {
    return this.CANVAS_MARKER;
  }

  /**
   * Get message types this view handles
   */
  static getMessageTypes() {
    return ["canvas-export-request", "requestImages"];
  }

  /**
   * Get view-specific state from node widgets
   */
  static getStateFromWidget(node) {
    const viewStateWidget = node.widgets?.find(w => w.name === "view_state");
    if (viewStateWidget?.value) {
      try {
        const viewState = JSON.parse(viewStateWidget.value);
        return viewState.canvasState || null;
      } catch {}
    }
    return null;
  }

  /**
   * Inject saved state into content before rendering
   */
  static injectState(content, state) {
    if (!state || !content) return content;
    
    try {
      let jsonContent = content;
      if (content.startsWith(this.CANVAS_MARKER)) {
        jsonContent = content.slice(this.CANVAS_MARKER.length);
      }
      const canvasData = JSON.parse(jsonContent);
      canvasData.savedState = state;
      return this.CANVAS_MARKER + JSON.stringify(canvasData);
    } catch (e) {
      console.warn("[Canvas View] Failed to inject saved state:", e);
      return content;
    }
  }

  /**
   * Handle messages from iframe (called by view_loader.handleViewMessage)
   */
  static handleMessage(messageType, data, node, app, iframeSource) {
    if (messageType === "requestImages") {
      return this._handleRequestImages(data, node, app, iframeSource);
    }

    if (messageType !== "canvas-export-request") return false;

    const { action, data: exportData } = data;
    this._composite(exportData).then(
      (dataUrl) => { if (dataUrl) this._finishExport(action, dataUrl, node); },
      (error) => console.error("[Canvas View] Export failed:", error)
    );
    return true;
  }

  /**
   * Ask the frame drawing the canvas for what it holds.
   *
   * @param {Window} frame - The frame the canvas is drawn in.
   * @param {number} [timeoutMs] - Milliseconds to wait for the frame's answer.
   * @returns {Promise<string|null>} The composite as a marked data URL, or null where the
   *   frame holds no layers, does not answer, or the composite cannot be read.
   */
  static requestOutput(frame, timeoutMs = OUTPUT_TIMEOUT_MS) {
    if (!frame) return Promise.resolve(null);

    const requestId = `${Date.now()}_${Math.random().toString(36).slice(2)}`;
    return new Promise((resolve) => {
      const settle = (value) => {
        window.removeEventListener("message", answer);
        clearTimeout(timer);
        resolve(value);
      };
      const answer = (event) => {
        const message = event.data;
        if (event.source !== frame) return;
        if (message?.type !== "canvas-composite-response") return;
        if (message.requestId !== requestId) return;
        if (!message.data) {
          settle(null);
          return;
        }
        this._composite(message.data).then(
          (dataUrl) => settle(dataUrl ? this.OUTPUT_MARKER + dataUrl : null),
          (error) => {
            console.error("[Canvas View] Composite failed:", error);
            settle(null);
          }
        );
      };
      window.addEventListener("message", answer);
      const timer = setTimeout(() => settle(null), timeoutMs);
      frame.postMessage({ type: "canvas-composite-request", requestId }, "*");
    });
  }

  /**
   * Draw the layers a frame reported onto one canvas here.
   *
   * @param {object} exportData - The layers, bounds and background the frame reported.
   * @returns {Promise<string|null>} The composite as a PNG data URL, or null where the canvas
   *   cannot be read.
   */
  static _composite(exportData) {
    return new Promise((resolve) => {
      const { layers, bounds, bgTransparent, bgColor } = exportData;
      const { minX, minY, maxX, maxY, padding } = bounds;

      const exportCanvas = document.createElement('canvas');
      exportCanvas.width = maxX - minX + padding * 2;
      exportCanvas.height = maxY - minY + padding * 2;
      const ctx = exportCanvas.getContext('2d');

      if (!bgTransparent) {
        ctx.fillStyle = bgColor;
        ctx.fillRect(0, 0, exportCanvas.width, exportCanvas.height);
      }

      ctx.translate(-minX + padding, -minY + padding);

      const finish = () => resolve(this._readCanvas(exportCanvas));

      if (layers.length === 0) {
        finish();
        return;
      }

      // Count total items to load (layer images + mask images)
      let totalToLoad = 0;
      let loadedCount = 0;
      const loadedImages = [];
      const loadedMasks = [];

      layers.forEach((layer) => {
        // Use drawDataUrl if available (for draw layers or image layers edited with eraser/brush)
        const imgSrc = (layer.isDrawLayer || layer.hasDrawCanvas) ? layer.drawDataUrl : layer.imageUrl;
        if (!imgSrc) return;
        totalToLoad++;
        if (layer.maskEnabled && layer.maskDataUrl) totalToLoad++;
      });

      if (totalToLoad === 0) {
        finish();
        return;
      }

      const checkComplete = () => {
        if (loadedCount === totalToLoad) {
          this._renderLayers(ctx, layers, loadedImages, loadedMasks);
          finish();
        }
      };

      layers.forEach((layer, idx) => {
        // Use drawDataUrl if available (for draw layers or image layers edited with eraser/brush)
        const imgSrc = (layer.isDrawLayer || layer.hasDrawCanvas) ? layer.drawDataUrl : layer.imageUrl;

        // Skip layers without valid image source
        if (!imgSrc) {
          console.warn('[Canvas Export] Layer', idx, 'has no valid image source, skipping');
          loadedImages[idx] = null;
          loadedMasks[idx] = null;
          return;
        }

        // Load layer image
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => {
          loadedImages[idx] = img;
          loadedCount++;
          checkComplete();
        };
        img.onerror = () => {
          console.warn('[Canvas Export] Failed to load layer', idx, layer.isDrawLayer ? '(draw layer)' : '(image layer)');
          loadedImages[idx] = null;
          loadedCount++;
          checkComplete();
        };
        img.src = imgSrc;

        // Load mask image if enabled
        if (layer.maskEnabled && layer.maskDataUrl) {
          const maskImg = new Image();
          maskImg.crossOrigin = 'anonymous';
          maskImg.onload = () => {
            loadedMasks[idx] = maskImg;
            loadedCount++;
            checkComplete();
          };
          maskImg.onerror = () => {
            console.warn('[Canvas Export] Failed to load mask for layer', idx);
            loadedMasks[idx] = null;
            loadedCount++;
            checkComplete();
          };
          maskImg.src = layer.maskDataUrl;
        } else {
          loadedMasks[idx] = null;
        }
      });
    });
  }

  /**
   * Read a canvas as a PNG data URL.
   *
   * @param {HTMLCanvasElement} canvas - The canvas to read.
   * @returns {string|null} The data URL, or null where the canvas cannot be read.
   */
  static _readCanvas(canvas) {
    try {
      return canvas.toDataURL('image/png');
    } catch (e) {
      console.error("[Canvas View] Export failed - canvas may be tainted by cross-origin images:", e);
      alert('Export failed: Canvas contains cross-origin images that cannot be exported.');
      return null;
    }
  }

  /**
   * Render layers onto canvas context
   */
  static _renderLayers(ctx, layers, loadedImages, loadedMasks = []) {
    layers.forEach((layer, idx) => {
      const img = loadedImages[idx];
      if (!img) return;
      
      ctx.save();
      ctx.translate(layer.x + layer.width * layer.scaleX * 0.5, layer.y + layer.height * layer.scaleY * 0.5);
      ctx.rotate(layer.rotation * Math.PI / 180);
      ctx.transform(1, layer.skewY, layer.skewX, 1, 0, 0);
      ctx.scale(layer.scaleX, layer.scaleY);
      ctx.translate(-layer.width * 0.5, -layer.height * 0.5);
      ctx.globalAlpha = layer.opacity;
      ctx.globalCompositeOperation = layer.blendMode || 'source-over';
      
      if (layer.shadow) {
        ctx.shadowBlur = layer.shadowBlur;
        ctx.shadowOffsetX = layer.shadowOffsetX;
        ctx.shadowOffsetY = layer.shadowOffsetY;
        ctx.shadowColor = layer.shadowColor;
      }
      
      // Apply mask if enabled
      const maskImg = loadedMasks[idx];
      if (layer.maskEnabled && maskImg) {
        // Composite image with mask using a temporary canvas
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = layer.width;
        tempCanvas.height = layer.height;
        const tctx = tempCanvas.getContext('2d');
        tctx.drawImage(img, 0, 0, layer.width, layer.height);
        tctx.globalCompositeOperation = 'destination-in';
        tctx.drawImage(maskImg, 0, 0, layer.width, layer.height);
        ctx.drawImage(tempCanvas, 0, 0, layer.width, layer.height);
      } else {
        ctx.drawImage(img, 0, 0, layer.width, layer.height);
      }
      
      ctx.restore();
    });
  }

  /**
   * Read each source image here and hand the iframe the bytes.
   *
   * The canvas is sandboxed without `allow-same-origin`, so its own requests carry a null
   * origin and `/view` answers 403. This page is same-origin and is served normally.
   *
   * @param {Array} images - File references, or strings already carrying data.
   * @param {Window} iframeSource - The canvas to send them to.
   * @returns {Promise<void>} Settled once the images have been sent.
   */
  static async _sendImagesAsData(images, iframeSource) {
    const read = async (fileInfo) => {
      if (typeof fileInfo === "string") return fileInfo;
      const params = new URLSearchParams({
        filename: fileInfo.filename,
        subfolder: fileInfo.subfolder || "",
        type: fileInfo.type || "temp",
      });
      try {
        const blob = await (await fetch("/view?" + params.toString())).blob();
        return await new Promise((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(reader.result);
          reader.onerror = reject;
          reader.readAsDataURL(blob);
        });
      } catch (error) {
        console.warn("[Canvas View] Could not read a source image:", error);
        return fileInfo;
      }
    };
    try {
      const carried = await Promise.all(images.map(read));
      iframeSource.postMessage({ type: "updateImages", images: carried }, "*");
    } catch (error) {
      console.error("[Canvas View] Failed to send the source images:", error);
    }
  }

  /**
   * Handle requestImages message from iframe
   * Note: The viewer system automatically extracts view-specific content from multiview
   * payloads and provides it via data._viewContent
   */
  static _handleRequestImages(data, node, app, iframeSource) {
    // Use view-specific content provided by the viewer system (handles multiview automatically)
    let content = data._viewContent || "";
    
    if (!content) return false;
    
    try {
      let jsonContent = content;
      if (content.startsWith(this.CANVAS_MARKER)) {
        jsonContent = content.slice(this.CANVAS_MARKER.length);
      }
      const parsed = JSON.parse(jsonContent);
      if (parsed.images && Array.isArray(parsed.images) && iframeSource) {
        this._sendImagesAsData(parsed.images, iframeSource);
        return true;
      }
    } catch (e) {
      console.warn("[Canvas View] Failed to parse content for requestImages:", e);
    }
    
    return false;
  }

  /**
   * Finish export - download or save to widget
   */
  static _finishExport(action, dataUrl, node) {
    if (action === 'download') {
      const link = document.createElement('a');
      link.download = 'canvas-export.png';
      link.href = dataUrl;
      link.click();
    } else if (action === 'output') {
      // A held run cannot read a widget written after it was queued, so the composite goes
      // to the route the hold is waiting on.
      if (node.__was_viewer_held) {
        fetch("/was/interface/api/pause", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            node_id: String(node.id), action: "resume",
            value: this.OUTPUT_MARKER + dataUrl,
            client_id: String(globalThis.app?.api?.clientId || globalThis.app?.api?.initialClientId || ""),
          }),
        }).catch((error) => console.error("[Canvas View] Failed to continue:", error));
      }
      const viewStateWidget = node.widgets?.find(w => w.name === "view_state");
      if (viewStateWidget) {
        try {
          const viewState = JSON.parse(viewStateWidget.value || "{}");
          viewState.canvas_output = this.OUTPUT_MARKER + dataUrl;
          // Note: Canvas state is no longer saved to widget to avoid quota errors
          // Use the save/load buttons in the Layers panel to export/import canvas state
          viewStateWidget.value = JSON.stringify(viewState);
        } catch (e) {
          console.error("[Canvas View] Failed to save view state:", e);
        }
      }
      
      node.setDirtyCanvas?.(true, true);
    }
  }
}

export default CanvasView;
