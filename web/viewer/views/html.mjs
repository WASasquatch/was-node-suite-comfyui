/**
 * HTML View - HTML content renderer
 */

import { BaseView } from "./base_view.mjs";

class HtmlView extends BaseView {
  static id = "html";
  static displayName = "HTML";
  static priority = 100;

  static detect(content) {
    const trimmed = content.trim();
    
    if (trimmed.startsWith("<!DOCTYPE") || trimmed.startsWith("<html") || 
        (trimmed.startsWith("<") && (trimmed.includes("<head") || trimmed.includes("<body")))) {
      return 100;
    }

    let score = 0;
    const htmlTags = ["<div", "<span", "<p>", "<h1", "<h2", "<h3", "<table", "<ul", "<ol", 
                      "<img", "<a ", "<br", "<hr", "<em>", "<strong>", "<b>", "<i>", "<code>", "<pre>"];
    for (const tag of htmlTags) {
      if (trimmed.includes(tag)) score += 2;
    }
    if (trimmed.startsWith("<") && trimmed.endsWith(">")) score += 1;

    return score;
  }

  static render(content, theme) {
    return content;
  }

  static isFullDocument(content) {
    const trimmed = content.trim();
    return trimmed.includes("<html") || trimmed.includes("<!DOCTYPE");
  }

  static getStyles(theme) {
    return "";
  }
}

export default HtmlView;
