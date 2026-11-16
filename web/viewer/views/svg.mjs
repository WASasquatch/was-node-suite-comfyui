/**
 * SVG View - SVG content renderer
 */

import { BaseView } from "./base_view.mjs";

class SvgView extends BaseView {
  static id = "svg";
  static displayName = "SVG";
  static priority = 110;

  static detect(content) {
    const trimmed = content.trim();
    
    if (trimmed.startsWith("<svg") || 
        (trimmed.startsWith("<?xml") && trimmed.includes("<svg")) ||
        (trimmed.startsWith("<") && trimmed.includes("xmlns") && trimmed.includes("<svg"))) {
      return 100;
    }

    return 0;
  }

  static render(content, theme) {
    return `<div style="display:flex;justify-content:center;align-items:center;min-height:100%;padding:16px;">${content}</div>`;
  }

  static getStyles(theme) {
    return `
      svg {
        max-width: 100%;
        height: auto;
      }
    `;
  }
}

export default SvgView;
