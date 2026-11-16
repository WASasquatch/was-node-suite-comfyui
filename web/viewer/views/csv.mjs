/**
 * CSV View - CSV table renderer
 */

import { BaseView, escapeHtml } from "./base_view.mjs";

class CsvView extends BaseView {
  static id = "csv";
  static displayName = "CSV";
  static priority = 50;

  static detect(content) {
    const trimmed = content.trim();
    const lines = trimmed.split("\n");
    const csvLines = lines.filter(l => l.includes(",") && !l.includes("{") && !l.includes("<"));
    
    if (csvLines.length >= 2) {
      const firstLineCommas = (lines[0].match(/,/g) || []).length;
      if (firstLineCommas >= 1) {
        const consistentCommas = csvLines.filter(l => (l.match(/,/g) || []).length === firstLineCommas).length;
        if (consistentCommas >= csvLines.length * 0.8) {
          return 5;
        }
      }
    }
    return 0;
  }

  static parseRow(line) {
    const result = [];
    let current = "";
    let inQuotes = false;
    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      if (char === '"') {
        inQuotes = !inQuotes;
      } else if (char === "," && !inQuotes) {
        result.push(current.trim());
        current = "";
      } else {
        current += char;
      }
    }
    result.push(current.trim());
    return result;
  }

  static render(content, theme) {
    const lines = content.trim().split("\n");
    if (lines.length === 0) return `<pre>${escapeHtml(content)}</pre>`;
    
    const headers = this.parseRow(lines[0]);
    const rows = lines.slice(1).map(line => this.parseRow(line));
    
    const thead = `<thead><tr>${headers.map(h => `<th>${escapeHtml(h)}</th>`).join("")}</tr></thead>`;
    const tbody = `<tbody>${rows.map(row => 
      `<tr>${row.map(cell => `<td>${escapeHtml(cell)}</td>`).join("")}</tr>`
    ).join("")}</tbody>`;
    
    return `<div class="csv-container"><table class="csv-table">${thead}${tbody}</table></div>`;
  }

  static getStyles(theme) {
    return `
      .csv-container {
        overflow-x: auto;
      }
      .csv-table {
        border-collapse: collapse;
        width: 100%;
        font-size: 13px;
      }
      .csv-table th, .csv-table td {
        border: 1px solid ${theme.border};
        padding: 8px 12px;
        text-align: left;
      }
      .csv-table th {
        background: rgba(0,0,0,0.3);
        font-weight: bold;
      }
      .csv-table tr:nth-child(even) {
        background: rgba(0,0,0,0.1);
      }
    `;
  }
}

export default CsvView;
