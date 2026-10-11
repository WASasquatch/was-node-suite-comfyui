"""Lay a document out as plain text."""

from __future__ import annotations

from comfy_api.latest import io, ui

from ...modules.compat.types import DOC
from ...modules.document import container, plaintext
from ...modules.log import get_logger

logger = get_logger("nodes.document")


class DocToPlaintext(io.ComfyNode):
    """A document's markup laid out as plain text on a string socket."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASConvertDOCToPlaintext",
            display_name="Convert DOC to Plaintext",
            search_aliases=[
                "WASConvertDOCToPlaintext",
                "Convert DOC to Plaintext",
                "document to text",
                "html to text",
                "strip html",
                "plain text",
                "doc to string",
            ],
            category="WAS Suite/Document",
            description=(
                "Turn a document into plain text that keeps its shape: paragraphs, underlined "
                "headings, indented lists, aligned tables and quotations, with markup and "
                "styling removed. Hands on a string for a prompt or a text file."
            ),
            inputs=[
                DOC.Input(
                    "doc",
                    tooltip=(
                        "The document to convert, from any node with a DOC output. Only its "
                        "content is read: the metadata and embedded pictures are not part of "
                        "the text."
                    ),
                ),
                io.Int.Input(
                    "line_width",
                    default=0,
                    min=0,
                    max=1024,
                    step=1,
                    tooltip=(
                        "Characters per line before wrapping: 0 = each paragraph on one line, "
                        "for a prompt; 72 or 80 = a text file read on its own. Long words, "
                        "tables and preformatted blocks never wrap."
                    ),
                ),
                io.Combo.Input(
                    "links",
                    options=list(plaintext.LINK_MODES),
                    tooltip=(
                        "What happens to the address behind a link. 'text and url' writes "
                        "it in brackets, 'Site (https://example.org)'; 'text only' drops it; "
                        "'footnotes' numbers it, 'Site[1]', and lists them at the end."
                    ),
                ),
                io.Combo.Input(
                    "images",
                    options=list(plaintext.IMAGE_MODES),
                    tooltip=(
                        "What stands in for a picture. 'alt text' writes its description in "
                        "square brackets, '[A grey cat]', or '[image]' where it has none; 'alt "
                        "text and source' adds the file it comes from; 'skip' leaves nothing."
                    ),
                ),
                io.Combo.Input(
                    "tables",
                    options=list(plaintext.TABLE_MODES),
                    tooltip=(
                        "How a table is written. 'aligned columns' pads cells with spaces "
                        "so the columns line up under a rule below the header row; 'tab "
                        "separated' pads nothing, for pasting into a spreadsheet."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="TEXT",
                    tooltip=(
                        "The document as plain text, with no blank line at either end and "
                        "no trailing spaces on any line. Empty where the document holds "
                        "nothing a reader would see, which a node saving it writes as an "
                        "empty file."
                    ),
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, doc, line_width, links, images, tables) -> io.NodeOutput:
        document = container.require_document(doc, "doc")
        text = plaintext.to_plaintext(
            document.content,
            links=links,
            images=images,
            tables=tables,
            width=line_width,
        )
        if text:
            logger.debug(
                "laid out %r as %d character(s) of text over %d line(s)",
                document.metadata.title or "untitled", len(text), text.count("\n") + 1,
            )
        else:
            logger.warning(
                "Convert DOC to Plaintext produced no text, because the document %r holds "
                "nothing a reader would see: %s. The TEXT output is empty, so a node that "
                "saves it writes an empty file.",
                document.metadata.title or "untitled",
                "its content is empty"
                if not document.content
                else f"its content is {len(document.content)} character(s) of markup with "
                f"nothing in it but tags",
            )
        return io.NodeOutput(text, ui=ui.PreviewText(text))
