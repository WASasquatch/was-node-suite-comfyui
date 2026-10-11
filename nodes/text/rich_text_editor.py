"""Write a document in a rich text editor and emit it as HTML."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import config, log
from ...modules.document import markup

logger = log.get_logger("nodes.text")


class RichTextEditor(io.ComfyNode):
    """A document, as HTML, held in a widget and emitted as a string."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASRichTextEditor",
            display_name="Rich Text Editor",
            search_aliases=[
                "WASRichTextEditor",
                "Rich Text Editor",
                "wysiwyg",
                "html",
                "document",
                "word processor",
                "letter",
            ],
            category="WAS Suite/Text",
            description=(
                "Write a document in a rich text editor drawn on the node, saved with the "
                "workflow, and emit it as HTML. Tokens such as [time] and [user] are "
                "replaced on the way out. Scripts and other active content are removed "
                "unless document.clean_html is set to false in config.yaml."
            ),
            inputs=[
                io.String.Input(
                    "html",
                    multiline=True,
                    default="",
                    socketless=True,
                    placeholder="The document, as HTML. The rich text editor writes here.",
                    tooltip=(
                        "The document, as HTML. The editor drawn on the node is a view onto "
                        "this box, so paste markup here and the two stay in step."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="html",
                    tooltip=(
                        "The document as HTML, for anything that takes a string. A bare < in "
                        "the text arrives as &lt;."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, html) -> io.NodeOutput:
        if not config.load_config()["document"]["clean_html"]:
            return io.NodeOutput(html)

        cleaned, removed = markup.clean(html)
        if removed:
            logger.info(
                "Rich Text Editor removed %s from its output. The document in the node is "
                "unchanged. Set document.clean_html to false in config.yaml to emit the "
                "markup exactly as written.",
                markup.describe(removed),
            )
        return io.NodeOutput(cleaned)
