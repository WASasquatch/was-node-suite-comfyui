"""Hand a document's markup to a string socket."""

from __future__ import annotations

from comfy_api.latest import io, ui

from ...modules.compat.types import DOC
from ...modules.document import container, formats
from ...modules.log import get_logger

logger = get_logger("nodes.document")

#: The markup a fragment is answered as, which is what an editor takes.
FRAGMENT = "content only"

#: The markup a whole file is answered as, with a head carrying the metadata.
WHOLE_PAGE = "whole page"

#: What the ``wrap`` widget offers, fragment first.
WRAPPING = (FRAGMENT, WHOLE_PAGE)


class DocToHTML(io.ComfyNode):
    """A document's markup on a string socket, as a fragment or a whole page."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASConvertDOCToHTML",
            display_name="Convert DOC to HTML",
            search_aliases=[
                "WASConvertDOCToHTML",
                "Convert DOC to HTML",
                "document to html",
                "doc to markup",
                "doc to string",
                "edit document",
            ],
            category="WAS Suite/Document",
            description=(
                "Put a document's markup on a string socket, as the fragment the "
                "document stores or as a whole HTML file with a head built from the "
                "metadata."
            ),
            inputs=[
                DOC.Input(
                    "doc",
                    tooltip=(
                        "The document to read, from any node with a DOC output. Nothing is "
                        "opened from disk and the document is not changed."
                    ),
                ),
                io.Combo.Input(
                    "wrap",
                    options=list(WRAPPING),
                    default=FRAGMENT,
                    tooltip=(
                        "'content only' = the markup the document stores, as Rich Text Editor "
                        "takes it; 'whole page' = wrapped in a file with a doctype and a head "
                        "carrying the metadata. Markup that already opens a page is left as it "
                        "stands."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="html",
                    tooltip=(
                        "The document's markup. Empty where the document holds none, which "
                        "a node saving it writes as an empty file."
                    ),
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, doc, wrap=FRAGMENT) -> io.NodeOutput:
        document = container.require_document(doc, "doc")
        markup = document.content or ""
        if wrap == WHOLE_PAGE:
            markup = formats.html_page(markup, document.metadata)
        title = document.metadata.title or "untitled"
        if markup:
            logger.debug(
                "read %r as %d character(s) of markup (%s)",
                title, len(markup), wrap,
            )
        else:
            logger.warning(
                "Convert DOC to HTML produced no markup, because the document %r holds "
                "none. The html output is empty, so a node that saves it writes an empty "
                "file.",
                title,
            )
        return io.NodeOutput(markup, ui=ui.PreviewText(markup))
