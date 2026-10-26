/**
 * The chat drawn on LLM Chat Session: the conversation, the reply as it streams in, and a
 * message box that sends with Enter.
 */

import { app } from "../../scripts/app.js";
import { createChatPanel } from "./interface/chat_panel.js";
import { appendInterfaceWidget } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.LLMChatSession";
const NODE_ID = "WASLLMChatSession";

const UI_WIDGET_NAME = "was_llm_chat_ui";
const UI_WIDGET_TYPE = "was_llm_chat";

app.registerExtension({
  name: EXT_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_ID) return;

    const proto = nodeType.prototype;
    if (proto.__was_llm_chat_wrapped) return;
    proto.__was_llm_chat_wrapped = true;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const panel = createChatPanel(this);
        appendInterfaceWidget(this, panel, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            panel.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the chat:`, error);
          }
          return removed;
        };

        const originalOnConfigure = this.onConfigure;
        this.onConfigure = function (...args) {
          const configured = originalOnConfigure?.apply(this, args);
          try {
            panel.refresh();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to read the saved chat:`, error);
          }
          return configured;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the chat:`, error);
      }
      return result;
    };
  },
});
