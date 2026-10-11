"""Read a document's metadata onto typed sockets and show it on the node."""

from __future__ import annotations

from comfy_api.latest import io, ui

from ...modules.compat.types import DICT, DOC, LIST
from ...modules.document import container, summary
from ...modules.log import get_logger

logger = get_logger("nodes.document")


class ViewDocMetadata(io.ComfyNode):
    """A document's metadata, one field per socket, and the whole reading as text."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASViewDOCMetadata",
            display_name="View DOC Metadata",
            search_aliases=[
                "WASViewDOCMetadata",
                "View DOC Metadata",
                "document metadata",
                "document info",
                "doc metadata",
                "copyright",
                "word count",
            ],
            category="WAS Suite/Document",
            description=(
                "Read a document's metadata onto one socket per field: title, description, "
                "author, copyright, language, keywords, timestamps, custom pairs, word and "
                "character counts and embedded files. The reading is also shown on the node "
                "and handed on as one block of text. A field the document does not carry "
                "comes out empty."
            ),
            inputs=[
                DOC.Input(
                    "doc",
                    tooltip=(
                        "The document to read, from any node with a DOC output. What is "
                        "reported is what this wire carries; nothing is opened from disk."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="title",
                    tooltip=(
                        "What the document is called. Empty when it has none, which is how "
                        "a document made from plain text starts out. This is the title an "
                        "exported file carries and the one a file manager shows in its "
                        "title column."
                    ),
                ),
                io.String.Output(
                    display_name="description",
                    tooltip=(
                        "The sentence or two saying what the document is. Empty when "
                        "nothing was written. Every format a DOC exports to carries it."
                    ),
                ),
                io.String.Output(
                    display_name="author",
                    tooltip=(
                        "Who wrote the document, as free text. Empty when the document "
                        "names nobody."
                    ),
                ),
                io.String.Output(
                    display_name="copyright",
                    tooltip=(
                        "The rights statement as free text, such as '(c) 2026 A. Name, CC BY "
                        "4.0'. Empty when the document makes no claim."
                    ),
                ),
                io.String.Output(
                    display_name="language",
                    tooltip=(
                        "The language tag the document is written in, such as 'en' or "
                        "'pt-BR', which sets hyphenation and spell checking in an exported "
                        "file. Empty when the document carries no tag."
                    ),
                ),
                io.String.Output(
                    display_name="keywords",
                    tooltip=(
                        "The keywords joined with commas, which is how the document formats "
                        "and desktop search engines spell the field. Empty when the "
                        "document has none. Use the keywords_list output instead to reach "
                        "one keyword at a time."
                    ),
                ),
                LIST.Output(
                    display_name="keywords_list",
                    tooltip=(
                        "The same keywords as a list, in the order they were given, for "
                        "Text List Get, Text List Length and Text List Slice. A document "
                        "with no keywords gives an empty list, which those nodes report as "
                        "a length of zero rather than failing."
                    ),
                ),
                io.String.Output(
                    display_name="created",
                    tooltip=(
                        "When the document was first made, as UTC in the form "
                        "'2026-01-02T03:04:05Z', which sorts correctly as text. Empty when "
                        "the document carries no stamp: the time it was read is never "
                        "reported as the time it was written."
                    ),
                ),
                io.String.Output(
                    display_name="modified",
                    tooltip=(
                        "When the document's content last changed, in the same form as "
                        "created. Editing its text or embedded files stamps this; changing "
                        "only its metadata does not. Empty when the document carries no stamp."
                    ),
                ),
                io.String.Output(
                    display_name="generator",
                    tooltip=(
                        "What produced the document, 'WAS Node Suite' for one this pack "
                        "wrote. Empty for a container built by hand or by another tool that "
                        "did not fill the field in."
                    ),
                ),
                DICT.Output(
                    display_name="custom",
                    tooltip=(
                        "The author's own pairs of text, in document order, for Text "
                        "Dictionary Get, Text Dictionary Keys and Dictionary to Console. All "
                        "three export formats keep them. Empty when the document has none."
                    ),
                ),
                io.Int.Output(
                    display_name="word_count",
                    tooltip=(
                        "How many words the document's text holds, markup stripped first. "
                        "Counted from the content on every run, never read from the file. 0 "
                        "for an empty document."
                    ),
                ),
                io.Int.Output(
                    display_name="character_count",
                    tooltip=(
                        "How many characters that same text holds: spaces between words "
                        "count, each line break counts as one, indentation between tags "
                        "counts for nothing. 0 = the document came out empty."
                    ),
                ),
                io.Int.Output(
                    display_name="asset_count",
                    tooltip=(
                        "How many files are embedded in the document, such as its pictures. 0 "
                        "for a document that carries none. An entry naming a place outside "
                        "the document is not counted."
                    ),
                ),
                LIST.Output(
                    display_name="assets",
                    tooltip=(
                        "The names of those files, sorted, each one relative to the "
                        "document's own assets folder and spelled with '/' whatever machine "
                        "the document was written on. An empty list when there are none."
                    ),
                ),
                io.Boolean.Output(
                    display_name="has_metadata",
                    tooltip=(
                        "True when any of title, description, author, copyright, language or "
                        "keywords holds something; the timestamps and generator do not count. "
                        "Wire it into a switch to fill the fields in before an export."
                    ),
                ),
                io.String.Output(
                    display_name="summary",
                    tooltip=(
                        "The whole reading as text, a field to a line labelled with its "
                        "socket's name, '(not set)' where empty. The same text shows on the "
                        "node; send it to Save Text File to keep a record."
                    ),
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, doc) -> io.NodeOutput:
        document = container.require_document(doc, "doc")
        values = summary.fields(document)
        logger.debug(
            "read the metadata of %r: %d word(s), %d embedded file(s)",
            values.title or "untitled", values.word_count, values.asset_count,
        )
        return io.NodeOutput(*values, ui=ui.PreviewText(values.summary))
