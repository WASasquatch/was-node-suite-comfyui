/**
 * Text View - Plain text content renderer
 */

import { BaseView, escapeHtml } from "./base_view.mjs";

class TextView extends BaseView {
  static id = "text";
  static displayName = "Text";
  static priority = 0;

  static detect(content) {
    return 1;
  }

  static render(content, theme) {
    return `<pre>${escapeHtml(content)}</pre>`;
  }

  static getStyles(theme) {
    return "";
  }
}

export default TextView;
