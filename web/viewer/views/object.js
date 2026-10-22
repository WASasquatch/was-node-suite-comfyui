/**
 * Object View - Generic Python object introspection viewer
 * Displays metrics, spectral graphs, and serialized data for:
 * - PyTorch/NumPy tensors
 * - PIL Images
 * - SafeTensors models
 * - Generic Python objects
 */

import { BaseView, escapeHtml } from "./base_view.js";

/**
 * Text written into markup, inside an element or a quoted attribute.
 * @param {unknown} value
 * @returns {string} `value` as text with `&`, `<`, `>`, `"` and `'` as entities.
 */
function markup(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

/**
 * A list of values as one line of text.
 * @param {unknown} value - An array, or a single value.
 * @param {string} [separator]
 * @returns {string}
 */
function listed(value, separator = ", ") {
  return Array.isArray(value) ? value.map((item) => String(item)).join(separator) : String(value ?? "");
}

class ObjectView extends BaseView {
  static id = "object";
  static displayName = "Object";
  static priority = 5;

  static OBJECT_MARKER = "$WAS_OBJECT$";

  static detect(content) {
    try {
      let jsonContent = content;
      if (content.startsWith(this.OBJECT_MARKER)) {
        jsonContent = content.slice(this.OBJECT_MARKER.length);
      }
      const parsed = JSON.parse(jsonContent);
      if (parsed.type === "object_viewer" && Array.isArray(parsed.objects)) {
        return 150;
      }
    } catch {}
    return 0;
  }

  static getContentMarker() {
    return this.OBJECT_MARKER;
  }

  static render(content, theme) {
    let data;
    try {
      let jsonContent = content;
      if (content.startsWith(this.OBJECT_MARKER)) {
        jsonContent = content.slice(this.OBJECT_MARKER.length);
      }
      data = JSON.parse(jsonContent);
    } catch {
      return `<pre>Invalid object data</pre>`;
    }

    const objects = data.objects || [];
    if (objects.length === 0) {
      return `<div class="object-viewer"><p>No objects to display</p></div>`;
    }

    let html = `<div class="object-viewer">`;
    
    for (const obj of objects) {
      html += this.renderObject(obj);
    }
    
    html += `</div>`;
    return html;
  }

  static renderObject(obj) {
    const id = `obj-${Math.random().toString(36).slice(2, 9)}`;
    const categoryIcon = this.getCategoryIcon(obj.category);
    const categoryClass = `category-${String(obj.category ?? "").replace(/[^\w-]/g, "")}`;
    
    let html = `
      <div class="object-card ${categoryClass}">
        <div class="object-header" data-object-toggle="${id}">
          <span class="object-icon">${categoryIcon}</span>
          <span class="object-type">${escapeHtml(obj.full_type || obj.type_name)}</span>
          <span class="object-category">${escapeHtml(String(obj.category ?? ""))}</span>
          <span class="object-toggle">▼</span>
        </div>
        <div class="object-content" id="${id}">
    `;

    // Metrics Section
    if (obj.metrics && Object.keys(obj.metrics).length > 0) {
      html += this.renderMetrics(obj.metrics, obj.category);
    }

    // Spectral/Histogram Section
    if (obj.spectral) {
      html += this.renderSpectral(obj.spectral);
    }

    // Attributes Section (for generic objects)
    if (obj.attributes && Object.keys(obj.attributes).length > 0) {
      html += this.renderAttributes(obj.attributes);
    }

    // Source Info Section
    if (obj.source_info) {
      html += this.renderSourceInfo(obj.source_info);
    }

    // Serialized Data Section
    if (obj.serialized) {
      html += this.renderSerialized(obj.serialized);
    }

    html += `
        </div>
      </div>
    `;
    
    return html;
  }

  static getCategoryIcon(category) {
    const icons = {
      tensor: "🔢",
      numpy: "📊",
      pil_image: "🖼️",
      safetensors: "📦",
      dict: "📋",
      list: "📝",
      object: "🔮",
    };
    return icons[category] || "❓";
  }

  static renderMetrics(metrics, category) {
    const id = `metrics-${Math.random().toString(36).slice(2, 9)}`;
    
    let html = `
      <div class="object-section">
        <div class="section-header" data-object-toggle="${id}">
          <span class="section-title">📊 Metrics</span>
          <span class="section-toggle">▼</span>
        </div>
        <div class="section-content" id="${id}">
          <div class="metrics-grid">
    `;

    // Render based on category for optimal display
    if (category === "tensor" || category === "numpy") {
      html += this.renderTensorMetrics(metrics);
    } else if (category === "pil_image") {
      html += this.renderImageMetrics(metrics);
    } else if (category === "safetensors") {
      html += this.renderSafetensorsMetrics(metrics);
    } else {
      html += this.renderGenericMetrics(metrics);
    }

    html += `
          </div>
        </div>
      </div>
    `;
    
    return html;
  }

  static renderTensorMetrics(metrics) {
    let html = "";
    
    // Primary info row
    if (metrics.shape) {
      html += `<div class="metric-item highlight"><span class="metric-label">Shape</span><span class="metric-value">[${markup(listed(metrics.shape))}]</span></div>`;
    }
    if (metrics.dtype) {
      html += `<div class="metric-item"><span class="metric-label">DType</span><span class="metric-value dtype">${escapeHtml(metrics.dtype)}</span></div>`;
    }
    if (metrics.resolution) {
      html += `<div class="metric-item highlight"><span class="metric-label">Resolution</span><span class="metric-value">${escapeHtml(metrics.resolution)}</span></div>`;
    }
    if (metrics.channels) {
      html += `<div class="metric-item"><span class="metric-label">Channels</span><span class="metric-value">${markup(metrics.channels)}</span></div>`;
    }
    if (metrics.image_format) {
      html += `<div class="metric-item"><span class="metric-label">Format</span><span class="metric-value">${escapeHtml(metrics.image_format)}</span></div>`;
    }
    if (metrics.memory_human) {
      html += `<div class="metric-item"><span class="metric-label">Memory</span><span class="metric-value">${escapeHtml(metrics.memory_human)}</span></div>`;
    }
    if (metrics.device) {
      html += `<div class="metric-item"><span class="metric-label">Device</span><span class="metric-value device">${escapeHtml(metrics.device)}</span></div>`;
    }
    if (metrics.numel) {
      html += `<div class="metric-item"><span class="metric-label">Elements</span><span class="metric-value">${markup(this.formatNumber(metrics.numel))}</span></div>`;
    }

    // Statistics
    if (metrics.stats) {
      html += `<div class="metric-divider"></div>`;
      html += `<div class="metric-item stat"><span class="metric-label">Min</span><span class="metric-value">${markup(this.formatNumber(metrics.stats.min))}</span></div>`;
      html += `<div class="metric-item stat"><span class="metric-label">Max</span><span class="metric-value">${markup(this.formatNumber(metrics.stats.max))}</span></div>`;
      html += `<div class="metric-item stat"><span class="metric-label">Mean</span><span class="metric-value">${markup(this.formatNumber(metrics.stats.mean))}</span></div>`;
      html += `<div class="metric-item stat"><span class="metric-label">Std</span><span class="metric-value">${markup(this.formatNumber(metrics.stats.std))}</span></div>`;
    }

    return html;
  }

  static renderImageMetrics(metrics) {
    let html = "";
    
    if (metrics.resolution) {
      html += `<div class="metric-item highlight"><span class="metric-label">Resolution</span><span class="metric-value">${escapeHtml(metrics.resolution)}</span></div>`;
    }
    if (metrics.mode) {
      html += `<div class="metric-item"><span class="metric-label">Mode</span><span class="metric-value">${escapeHtml(metrics.mode)}</span></div>`;
    }
    if (metrics.channels) {
      html += `<div class="metric-item"><span class="metric-label">Channels</span><span class="metric-value">${markup(metrics.channels)}</span></div>`;
    }
    if (metrics.bands) {
      html += `<div class="metric-item"><span class="metric-label">Bands</span><span class="metric-value">${markup(listed(metrics.bands))}</span></div>`;
    }
    if (metrics.format) {
      html += `<div class="metric-item"><span class="metric-label">Format</span><span class="metric-value">${escapeHtml(metrics.format || "N/A")}</span></div>`;
    }
    if (metrics.memory_human) {
      html += `<div class="metric-item"><span class="metric-label">Memory</span><span class="metric-value">${escapeHtml(metrics.memory_human)}</span></div>`;
    }
    if (metrics.dpi) {
      html += `<div class="metric-item"><span class="metric-label">DPI</span><span class="metric-value">${markup(listed(metrics.dpi))}</span></div>`;
    }
    if (metrics.has_icc_profile) {
      html += `<div class="metric-item"><span class="metric-label">ICC Profile</span><span class="metric-value">✓</span></div>`;
    }
    if (metrics.has_exif) {
      html += `<div class="metric-item"><span class="metric-label">EXIF</span><span class="metric-value">✓</span></div>`;
    }

    // Statistics
    if (metrics.stats) {
      html += `<div class="metric-divider"></div>`;
      html += `<div class="metric-item stat"><span class="metric-label">Min</span><span class="metric-value">${markup(metrics.stats.min)}</span></div>`;
      html += `<div class="metric-item stat"><span class="metric-label">Max</span><span class="metric-value">${markup(metrics.stats.max)}</span></div>`;
      html += `<div class="metric-item stat"><span class="metric-label">Mean</span><span class="metric-value">${markup(this.formatNumber(metrics.stats.mean))}</span></div>`;
      html += `<div class="metric-item stat"><span class="metric-label">Std</span><span class="metric-value">${markup(this.formatNumber(metrics.stats.std))}</span></div>`;
    }

    return html;
  }

  static renderSafetensorsMetrics(metrics) {
    let html = "";
    
    if (metrics.tensor_count !== undefined) {
      html += `<div class="metric-item highlight"><span class="metric-label">Tensors</span><span class="metric-value">${markup(this.formatNumber(metrics.tensor_count))}</span></div>`;
    }
    if (metrics.total_params_human) {
      html += `<div class="metric-item highlight"><span class="metric-label">Parameters</span><span class="metric-value">${escapeHtml(metrics.total_params_human)}</span></div>`;
    }

    // Tensor info table
    if (metrics.tensor_info && metrics.tensor_info.length > 0) {
      html += `<div class="metric-divider"></div>`;
      html += `<div class="tensor-table-wrapper">`;
      html += `<table class="tensor-table">`;
      html += `<thead><tr><th>Name</th><th>Shape</th><th>DType</th><th>Params</th></tr></thead>`;
      html += `<tbody>`;
      
      for (const tensor of metrics.tensor_info) {
        if (tensor.truncated) {
          html += `<tr class="truncated-row"><td colspan="4">${escapeHtml(tensor.name)}</td></tr>`;
        } else if (tensor.error) {
          html += `<tr class="error-row"><td>${escapeHtml(tensor.name)}</td><td colspan="3" class="error">${escapeHtml(tensor.error)}</td></tr>`;
        } else {
          const shape = tensor.shape ? `[${listed(tensor.shape)}]` : "?";
          const params = tensor.params ? this.formatNumber(tensor.params) : "?";
          html += `<tr><td class="tensor-name">${escapeHtml(tensor.name)}</td><td>${markup(shape)}</td><td>${escapeHtml(tensor.dtype || "?")}</td><td>${markup(params)}</td></tr>`;
        }
      }
      
      html += `</tbody></table>`;
      html += `</div>`;
    }

    return html;
  }

  static renderGenericMetrics(metrics) {
    let html = "";
    
    for (const [key, value] of Object.entries(metrics)) {
      if (key === "error") continue;
      
      let displayValue;
      if (Array.isArray(value)) {
        displayValue = value.slice(0, 10).join(", ");
        if (value.length > 10) displayValue += "...";
      } else if (typeof value === "object" && value !== null) {
        displayValue = JSON.stringify(value).slice(0, 100);
      } else {
        displayValue = String(value);
      }
      
      html += `<div class="metric-item"><span class="metric-label">${escapeHtml(this.formatLabel(key))}</span><span class="metric-value">${escapeHtml(displayValue)}</span></div>`;
    }
    
    return html;
  }

  static renderSpectral(spectral) {
    const id = `spectral-${Math.random().toString(36).slice(2, 9)}`;
    
    let html = `
      <div class="object-section spectral-section">
        <div class="section-header" data-object-toggle="${id}">
          <span class="section-title">📈 ${spectral.type === "spectral" ? "Channel Histograms" : "Histogram"}</span>
          <span class="section-toggle">▼</span>
        </div>
        <div class="section-content" id="${id}">
    `;

    if (spectral.type === "spectral" && spectral.channels) {
      // Multi-channel spectral view
      html += `<div class="spectral-container">`;
      
      for (const channel of spectral.channels) {
        const canvasId = `canvas-${Math.random().toString(36).slice(2, 9)}`;
        const color = this.getChannelColor(channel.name);
        
        html += `
          <div class="channel-histogram">
            <div class="channel-label" style="color: ${color}">${escapeHtml(channel.name)}</div>
            <canvas id="${canvasId}" class="histogram-canvas" width="200" height="60" 
              data-histogram="${markup(JSON.stringify(channel.histogram ?? []))}"
              data-color="${color}"></canvas>
          </div>
        `;
      }
      
      html += `</div>`;
    } else if (spectral.data) {
      // Single histogram
      const canvasId = `canvas-${Math.random().toString(36).slice(2, 9)}`;
      html += `
        <div class="single-histogram">
          <canvas id="${canvasId}" class="histogram-canvas wide" width="400" height="80"
            data-histogram="${markup(JSON.stringify(spectral.data ?? []))}"
            data-color="#888888"></canvas>
        </div>
      `;
    }

    html += `
        </div>
      </div>
    `;
    
    return html;
  }

  static getChannelColor(name) {
    const colors = {
      R: "#ff4444",
      G: "#44ff44", 
      B: "#4488ff",
      A: "#ffffff",
      L: "#888888",
    };
    return colors[name] || "#888888";
  }

  static renderAttributes(attributes) {
    const id = `attrs-${Math.random().toString(36).slice(2, 9)}`;
    
    let html = `
      <div class="object-section">
        <div class="section-header" data-object-toggle="${id}">
          <span class="section-title">🏷️ Attributes (${Object.keys(attributes).length})</span>
          <span class="section-toggle">▼</span>
        </div>
        <div class="section-content collapsed" id="${id}">
          <table class="attributes-table">
            <thead><tr><th>Name</th><th>Type</th><th>Value</th></tr></thead>
            <tbody>
    `;

    for (const [name, info] of Object.entries(attributes)) {
      html += `<tr>
        <td class="attr-name">${escapeHtml(name)}</td>
        <td class="attr-type">${escapeHtml(info.type)}</td>
        <td class="attr-value">${escapeHtml(String(info.value))}</td>
      </tr>`;
    }

    html += `
            </tbody>
          </table>
        </div>
      </div>
    `;
    
    return html;
  }

  static renderSourceInfo(sourceInfo) {
    const id = `source-${Math.random().toString(36).slice(2, 9)}`;
    
    let html = `
      <div class="object-section">
        <div class="section-header" data-object-toggle="${id}">
          <span class="section-title">📁 Source Info</span>
          <span class="section-toggle">▼</span>
        </div>
        <div class="section-content collapsed" id="${id}">
          <div class="source-info">
    `;

    if (sourceInfo.file) {
      html += `<div class="source-item"><span class="source-label">File:</span><span class="source-value path">${escapeHtml(sourceInfo.file)}</span></div>`;
    }
    if (sourceInfo.source_lines) {
      html += `<div class="source-item"><span class="source-label">Lines:</span><span class="source-value">${markup(sourceInfo.source_lines)}</span></div>`;
    }
    if (sourceInfo.bases && sourceInfo.bases.length > 0) {
      html += `<div class="source-item"><span class="source-label">Bases:</span><span class="source-value">${markup(listed(sourceInfo.bases, " → "))}</span></div>`;
    }

    html += `
          </div>
        </div>
      </div>
    `;
    
    return html;
  }

  static renderSerialized(serialized) {
    const id = `serial-${Math.random().toString(36).slice(2, 9)}`;
    
    let displayContent;
    try {
      const parsed = JSON.parse(serialized);
      displayContent = JSON.stringify(parsed, null, 2);
    } catch {
      displayContent = serialized;
    }

    // Truncate if too long
    const maxLen = 5000;
    let truncated = false;
    if (displayContent.length > maxLen) {
      displayContent = displayContent.slice(0, maxLen);
      truncated = true;
    }
    
    return `
      <div class="object-section">
        <div class="section-header" data-object-toggle="${id}">
          <span class="section-title">📄 Serialized Data${truncated ? " (truncated)" : ""}</span>
          <span class="section-toggle">▼</span>
        </div>
        <div class="section-content collapsed" id="${id}">
          <pre class="serialized-content">${escapeHtml(displayContent)}${truncated ? "\n\n... (truncated)" : ""}</pre>
        </div>
      </div>
    `;
  }

  static formatNumber(num) {
    if (typeof num !== "number") return String(num);
    if (Number.isInteger(num)) return num.toLocaleString();
    if (Math.abs(num) < 0.001 || Math.abs(num) > 100000) {
      return num.toExponential(3);
    }
    return num.toFixed(4);
  }

  static formatLabel(key) {
    return key
      .replace(/_/g, " ")
      .replace(/([A-Z])/g, " $1")
      .trim()
      .split(" ")
      .map(w => w.charAt(0).toUpperCase() + w.slice(1))
      .join(" ");
  }

  static getStyles(theme) {
    return `
      .object-viewer {
        padding: 12px;
        font-family: system-ui, -apple-system, sans-serif;
        font-size: 13px;
        line-height: 1.4;
      }
      
      .object-card {
        background: ${theme.bg};
        border: 1px solid ${theme.border};
        border-radius: 8px;
        margin-bottom: 12px;
        overflow: hidden;
      }
      
      .object-card.category-tensor { border-left: 3px solid #ff9800; }
      .object-card.category-numpy { border-left: 3px solid #2196f3; }
      .object-card.category-pil_image { border-left: 3px solid #4caf50; }
      .object-card.category-safetensors { border-left: 3px solid #9c27b0; }
      .object-card.category-dict { border-left: 3px solid #607d8b; }
      .object-card.category-list { border-left: 3px solid #795548; }
      .object-card.category-object { border-left: 3px solid #00bcd4; }
      
      .object-header {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 14px;
        background: linear-gradient(to right, rgba(255,255,255,0.03), transparent);
        cursor: pointer;
        user-select: none;
      }
      .object-header:hover { background: rgba(255,255,255,0.05); }
      
      .object-icon { font-size: 18px; }
      .object-type {
        font-family: 'SF Mono', Consolas, monospace;
        font-weight: 600;
        color: ${theme.accent || '#4fc3f7'};
        flex: 1;
      }
      .object-category {
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 10px;
        background: rgba(255,255,255,0.1);
        color: ${theme.fg}88;
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }
      .object-toggle {
        font-size: 10px;
        color: ${theme.fg}66;
        transition: transform 0.2s;
      }
      .object-content.collapsed + .object-header .object-toggle,
      .object-content.collapsed ~ .object-toggle { transform: rotate(-90deg); }
      
      .object-content {
        border-top: 1px solid ${theme.border};
      }
      .object-content.collapsed { display: none; }
      
      .object-section {
        border-bottom: 1px solid ${theme.border}44;
      }
      .object-section:last-child { border-bottom: none; }
      
      .section-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 8px 14px;
        background: rgba(0,0,0,0.1);
        cursor: pointer;
        user-select: none;
      }
      .section-header:hover { background: rgba(0,0,0,0.15); }
      
      .section-title {
        font-size: 12px;
        font-weight: 600;
        color: ${theme.fg}cc;
      }
      .section-toggle {
        font-size: 9px;
        color: ${theme.fg}66;
        transition: transform 0.2s;
      }
      .section-content.collapsed + .section-header .section-toggle { transform: rotate(-90deg); }
      
      .section-content {
        padding: 12px 14px;
      }
      .section-content.collapsed { display: none; }
      
      /* Metrics Grid */
      .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
        gap: 8px;
      }
      .metric-item {
        display: flex;
        flex-direction: column;
        padding: 8px 10px;
        background: rgba(255,255,255,0.03);
        border-radius: 6px;
        border: 1px solid ${theme.border}33;
      }
      .metric-item.highlight {
        background: rgba(79, 195, 247, 0.08);
        border-color: rgba(79, 195, 247, 0.3);
      }
      .metric-item.stat {
        background: rgba(156, 39, 176, 0.08);
        border-color: rgba(156, 39, 176, 0.2);
      }
      .metric-label {
        font-size: 10px;
        color: ${theme.fg}88;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 2px;
      }
      .metric-value {
        font-family: 'SF Mono', Consolas, monospace;
        font-size: 13px;
        font-weight: 500;
        color: ${theme.fg};
        word-break: break-all;
      }
      .metric-value.dtype { color: #ff9800; }
      .metric-value.device { color: #4caf50; }
      .metric-divider {
        grid-column: 1 / -1;
        height: 1px;
        background: ${theme.border}44;
        margin: 4px 0;
      }
      
      /* Tensor Table */
      .tensor-table-wrapper {
        grid-column: 1 / -1;
        max-height: 300px;
        overflow-y: auto;
        border-radius: 6px;
        border: 1px solid ${theme.border}44;
      }
      .tensor-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 11px;
      }
      .tensor-table th {
        position: sticky;
        top: 0;
        background: ${theme.bg};
        padding: 6px 8px;
        text-align: left;
        font-weight: 600;
        color: ${theme.fg}aa;
        border-bottom: 1px solid ${theme.border};
      }
      .tensor-table td {
        padding: 4px 8px;
        border-bottom: 1px solid ${theme.border}22;
        font-family: 'SF Mono', Consolas, monospace;
      }
      .tensor-table .tensor-name {
        max-width: 200px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
        color: ${theme.accent || '#4fc3f7'};
      }
      .tensor-table .truncated-row td { color: ${theme.fg}66; font-style: italic; }
      .tensor-table .error-row .error { color: #f44336; }
      
      /* Spectral/Histogram */
      .spectral-container {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
      }
      .channel-histogram {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 4px;
      }
      .channel-label {
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
      }
      .histogram-canvas {
        background: rgba(0,0,0,0.2);
        border-radius: 4px;
        border: 1px solid ${theme.border}44;
      }
      .histogram-canvas.wide {
        width: 100%;
        max-width: 400px;
      }
      .single-histogram {
        display: flex;
        justify-content: center;
      }
      
      /* Attributes Table */
      .attributes-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 12px;
      }
      .attributes-table th {
        text-align: left;
        padding: 6px 8px;
        background: rgba(0,0,0,0.15);
        color: ${theme.fg}aa;
        font-weight: 600;
      }
      .attributes-table td {
        padding: 4px 8px;
        border-bottom: 1px solid ${theme.border}22;
        vertical-align: top;
      }
      .attributes-table .attr-name {
        font-family: 'SF Mono', Consolas, monospace;
        color: ${theme.accent || '#4fc3f7'};
        white-space: nowrap;
      }
      .attributes-table .attr-type {
        font-family: 'SF Mono', Consolas, monospace;
        color: #ff9800;
        font-size: 11px;
      }
      .attributes-table .attr-value {
        font-family: 'SF Mono', Consolas, monospace;
        color: ${theme.fg}cc;
        max-width: 300px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }
      
      /* Source Info */
      .source-info {
        display: flex;
        flex-direction: column;
        gap: 6px;
      }
      .source-item {
        display: flex;
        gap: 8px;
      }
      .source-label {
        font-size: 11px;
        color: ${theme.fg}88;
        min-width: 50px;
      }
      .source-value {
        font-family: 'SF Mono', Consolas, monospace;
        font-size: 12px;
        color: ${theme.fg};
        word-break: break-all;
      }
      .source-value.path { color: #4caf50; }
      
      /* Serialized Content */
      .serialized-content {
        background: rgba(0,0,0,0.2);
        padding: 12px;
        border-radius: 6px;
        font-family: 'SF Mono', Consolas, monospace;
        font-size: 11px;
        line-height: 1.5;
        overflow-x: auto;
        white-space: pre-wrap;
        word-break: break-all;
        max-height: 400px;
        overflow-y: auto;
        margin: 0;
        color: ${theme.fg}dd;
      }
      
      /* Scrollbar */
      ::-webkit-scrollbar { width: 8px; height: 8px; }
      ::-webkit-scrollbar-track { background: transparent; }
      ::-webkit-scrollbar-thumb { background: ${theme.fg}33; border-radius: 4px; }
      ::-webkit-scrollbar-thumb:hover { background: ${theme.fg}55; }
    `;
  }

  static getScripts() {
    return `
      <script>
        document.addEventListener('click', function (event) {
          var toggle = event.target && event.target.closest ? event.target.closest('[data-object-toggle]') : null;
          if (!toggle) return;
          var el = document.getElementById(toggle.getAttribute('data-object-toggle'));
          if (el) el.classList.toggle('collapsed');
        });
        function drawHistograms() {
          var canvases = document.querySelectorAll('.histogram-canvas');
          canvases.forEach(function(canvas) {
            var ctx = canvas.getContext('2d');
            var data = JSON.parse(canvas.dataset.histogram || '[]');
            var color = canvas.dataset.color || '#888888';
            
            if (!data || data.length === 0) return;
            
            var w = canvas.width;
            var h = canvas.height;
            var max = Math.max.apply(null, data);
            var barWidth = w / data.length;
            
            ctx.clearRect(0, 0, w, h);
            
            // Draw bars
            ctx.fillStyle = color;
            ctx.globalAlpha = 0.7;
            
            for (var i = 0; i < data.length; i++) {
              var barHeight = (data[i] / max) * h;
              ctx.fillRect(i * barWidth, h - barHeight, barWidth - 0.5, barHeight);
            }
            
            // Draw outline
            ctx.globalAlpha = 1;
            ctx.strokeStyle = color;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(0, h);
            for (var i = 0; i < data.length; i++) {
              var barHeight = (data[i] / max) * h;
              ctx.lineTo(i * barWidth + barWidth / 2, h - barHeight);
            }
            ctx.lineTo(w, h);
            ctx.stroke();
          });
        }
        // Draw histograms when DOM is ready
        if (document.readyState === 'loading') {
          document.addEventListener('DOMContentLoaded', drawHistograms);
        } else {
          setTimeout(drawHistograms, 0);
        }
      <\/script>
    `;
  }
}

export default ObjectView;
