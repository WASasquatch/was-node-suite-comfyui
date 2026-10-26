/**
 * The report drawn on the LLM nodes.
 *
 * Every node listed below gets the pack's report panel: its summary, figures, facts and the
 * texts it published, such as an answer, a chunk sample or an agent's trace.
 */

import { app } from "../../scripts/app.js";
import { createReportPanel } from "./interface/report_panel.js";
import { appendInterfaceWidget, boundTextBoxes } from "./interface/widget.js";

const EXT_NAME = "WASNodeSuite.LLMReport";

const UI_WIDGET_NAME = "was_llm_report_ui";
const UI_WIDGET_TYPE = "was_llm_report";

// The widest fact name the reports write, which is what the fact column is opened at.
const LABEL_WIDTH = 96;

// Node id -> [panel height, the words drawn before the first run].
const PANELS = {
  WASLLMOpenAI: [200, "run to see the model"],
  WASLLMAnthropic: [200, "run to see the model"],
  WASLLMGemini: [200, "run to see the model"],
  WASLLMProvider: [200, "run to see the model"],
  WASLLMEmbeddingModel: [240, "run to see the model"],
  WASLLMImageEmbeddingModel: [180, "run to see the model"],
  WASLLMFromCLIP: [160, "run to see the model"],
  WASLLMGGUFLoader: [200, "run to load the model"],
  WASLLMTransformersLoader: [180, "run to load the model"],
  WASLLMUnload: [140, "run to free the models"],
  WASLLMMessage: [200, "run to see the conversation"],
  WASLLMComplete: [280, "run to see the answer"],
  WASLLMChat: [320, "run to see the reply"],
  WASLLMStructuredOutput: [300, "run to see the result"],
  WASLLMPromptTemplate: [220, "run to see the text"],
  WASLLMTokenCount: [140, "run to count"],
  WASLLMMessagesToText: [240, "run to see the text"],
  WASLLMLoadDocuments: [280, "run to read the folder"],
  WASLLMLoadDocument: [260, "run to read the file"],
  WASLLMTextToDocument: [200, "run to see the document"],
  WASLLMIndexNode: [180, "run to see the index node"],
  WASLLMToDocuments: [240, "run to see the documents"],
  WASLLMLoadWebPage: [260, "run to read the page"],
  WASLLMLoadImages: [260, "run to read the pictures"],
  WASLLMLoadTable: [280, "run to read the table"],
  WASLLMTableToDocuments: [240, "run to see the documents"],
  WASLLMImagesToDocuments: [220, "run to store the pictures"],
  WASLLMDocumentsFromDOC: [240, "run to see the documents"],
  WASLLMSplitDocuments: [300, "run to see the chunks"],
  WASLLMExtractMetadata: [300, "run to see the metadata"],
  WASLLMDocumentsToText: [240, "run to see the text"],
  WASLLMVectorIndex: [260, "run to build the index"],
  WASLLMSummaryIndex: [260, "run to build the index"],
  WASLLMTreeIndex: [260, "run to build the index"],
  WASLLMKeywordIndex: [260, "run to build the index"],
  WASLLMKnowledgeGraphIndex: [260, "run to build the graph"],
  WASLLMDocumentSummaryIndex: [260, "run to build the index"],
  WASLLMImageIndex: [260, "run to build the index"],
  WASLLMVectorStore: [140, "run to open the collection"],
  WASLLMVectorStoreIndex: [220, "run to open the index"],
  WASLLMSaveIndex: [200, "run to save the index"],
  WASLLMLoadIndex: [220, "run to load the index"],
  WASLLMIndexInsert: [240, "run to add the documents"],
  WASLLMIndexRetriever: [160, "run to see the retriever"],
  WASLLMBM25Retriever: [180, "run to see the retriever"],
  WASLLMFusionRetriever: [180, "run to see the retriever"],
  WASLLMAutoMergingRetriever: [160, "run to see the retriever"],
  WASLLMRouterRetriever: [180, "run to see the retriever"],
  WASLLMAutoRetriever: [200, "run to see the retriever"],
  WASLLMRecursiveRetriever: [180, "run to see the retriever"],
  WASLLMTransformRetriever: [140, "run to see the retriever"],
  WASLLMSQLRetriever: [160, "run to see the retriever"],
  WASLLMRetrieve: [320, "run to see what was found"],
  WASLLMPostprocessor: [140, "run to see the chain"],
  WASLLMRerank: [160, "run to see the chain"],
  WASLLMSourcesToText: [240, "run to see the text"],
  WASLLMQueryEngine: [180, "run to see the engine"],
  WASLLMSQLQueryEngine: [200, "run to see the engine"],
  WASLLMQueryTransform: [140, "run to see the engine"],
  WASLLMSubQuestionQueryEngine: [200, "run to see the engine"],
  WASLLMRouterQueryEngine: [200, "run to see the engine"],
  WASLLMRetryQueryEngine: [180, "run to see the engine"],
  WASLLMQuery: [320, "run to see the answer"],
  WASLLMChatEngine: [160, "run to see the engine"],
  WASLLMMemory: [180, "run to see the session"],
  WASLLMMemoryBlock: [140, "run to see the blocks"],
  WASLLMMemoryHistory: [300, "run to see the history"],
  WASLLMClearMemory: [120, "run to clear the session"],
  WASLLMTool: [160, "run to see the tools"],
  WASLLMQueryEngineTool: [140, "run to see the tools"],
  WASLLMRetrieverTool: [140, "run to see the tools"],
  WASLLMMCPTools: [200, "run to see the tools"],
  WASLLMAgent: [180, "run to see the agents"],
  WASLLMRunAgent: [360, "run to see the trace"],
  WASLLMEvaluate: [240, "run to see the verdict"],
  WASLLMGenerateQuestions: [280, "run to see the questions"],
  WASLLMEvaluateRetriever: [320, "run to see the scores"],
};

app.registerExtension({
  name: EXT_NAME,

  async beforeRegisterNodeDef(nodeType, nodeData) {
    const panel = PANELS[nodeData?.name];
    if (!panel) return;

    const proto = nodeType.prototype;
    if (proto.__was_llm_report_wrapped) return;
    proto.__was_llm_report_wrapped = true;
    const [height, emptyLabel] = panel;

    const originalOnNodeCreated = proto.onNodeCreated;
    proto.onNodeCreated = function () {
      const result = originalOnNodeCreated?.apply(this, arguments);
      try {
        const report = createReportPanel(this, {
          className: "was-llm-report",
          height,
          labelWidth: LABEL_WIDTH,
          emptyLabel,
          logName: EXT_NAME,
          failure: "Failed to read the LLM report:",
        });
        appendInterfaceWidget(this, report, { name: UI_WIDGET_NAME, type: UI_WIDGET_TYPE });
        boundTextBoxes(this);

        const originalOnRemoved = this.onRemoved;
        this.onRemoved = function (...args) {
          const removed = originalOnRemoved?.apply(this, args);
          try {
            report.dispose();
          } catch (error) {
            console.error(`[${EXT_NAME}] Failed to release the LLM report:`, error);
          }
          return removed;
        };
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to build the LLM report:`, error);
      }
      return result;
    };
  },
});
