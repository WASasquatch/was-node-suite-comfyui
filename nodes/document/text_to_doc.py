"""Build a document from a string of text or markup."""

from __future__ import annotations

from collections.abc import Mapping

from comfy_api.latest import io

from ...modules import config
from ...modules.compat.types import DICT, DOC
from ...modules.document import compose, markup, metadata
from ...modules.document.container import Document
from ...modules.log import get_logger

logger = get_logger("nodes.document")

#: The ``text_format`` options. ``PLAIN_TEXT`` wraps the string in paragraphs, ``HTML``
#: takes it as the document's own markup.
PLAIN_TEXT = "plain text"
HTML = "html"
TEXT_FORMATS = [PLAIN_TEXT, HTML]


class TextToDoc(io.ComfyNode):
    """A document built from one string and the metadata fields beside it."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASTextToDOC",
            display_name="Text to DOC",
            search_aliases=[
                "WASTextToDOC",
                "Text to DOC",
                "text to document",
                "document",
                "html to document",
                "wasdoc",
                "title",
                "copyright",
            ],
            category="WAS Suite/Document",
            description=(
                "Turn a string of plain text or HTML into a document carrying a title, "
                "description, author, copyright, language and keywords, ready for Save DOC or "
                "any other document node."
            ),
            inputs=[
                io.String.Input(
                    "text",
                    multiline=True,
                    default="",
                    placeholder=(
                        "The document's text. Plain text is wrapped into paragraphs; set "
                        "text_format to html to paste markup instead."
                    ),
                    tooltip=(
                        "What the document says. On text_format 'plain text' a blank line "
                        "starts a new paragraph; on 'html' it is the document's markup. Runs "
                        "of spaces and indentation collapse either way."
                    ),
                ),
                io.Combo.Input(
                    "text_format",
                    options=TEXT_FORMATS,
                    default=PLAIN_TEXT,
                    tooltip=(
                        "How the text is read. 'plain text' is for prose and shows any tag as "
                        "written; 'html' is for markup, keeping headings, lists, tables, links "
                        "and images."
                    ),
                ),
                io.String.Input(
                    "title",
                    default="",
                    optional=True,
                    tooltip=(
                        "What the document is called, such as 'Shot list, scene 4', shown by a "
                        "document viewer, a file manager and an exported file. Empty = no "
                        "title, and viewers fall back to the file name."
                    ),
                ),
                io.String.Input(
                    "description",
                    default="",
                    optional=True,
                    tooltip=(
                        "A sentence or two saying what the document is, for the reader who "
                        "finds it in six months. It is the field a document properties panel "
                        "and a search result show under the title. Left empty, the document "
                        "describes itself only by its title."
                    ),
                ),
                io.String.Input(
                    "author",
                    default="",
                    optional=True,
                    tooltip=(
                        "Who wrote the document, as a name rather than an account: 'A. Name' "
                        "or 'Studio Name'. It travels inside the document, so a copy passed on "
                        "still says who made it. Left empty, the document names nobody."
                    ),
                ),
                io.String.Input(
                    "copyright",
                    default="",
                    optional=True,
                    tooltip=(
                        "The rights statement carried with the document, as free text, such as "
                        "'(c) 2026 A. Name, CC BY 4.0'. A full sentence and a licence name are "
                        "both fine."
                    ),
                ),
                io.String.Input(
                    "language",
                    default="",
                    optional=True,
                    tooltip=(
                        "The language the text is written in, as a tag: 'en', 'en-GB', 'ja', "
                        "'pt-BR'. Sets an export's spell-check dictionary and hyphenation. "
                        "Empty = no language claimed."
                    ),
                ),
                io.String.Input(
                    "keywords",
                    default="",
                    optional=True,
                    tooltip=(
                        "Search terms, separated by commas, such as 'concept art, dragon, shot "
                        "12'. Spaces around a comma are trimmed and empty entries dropped. "
                        "Desktop search engines read this field."
                    ),
                ),
                DICT.Input(
                    "custom_metadata",
                    optional=True,
                    tooltip=(
                        "Further pairs of your own to carry in the document, from Text "
                        "Dictionary New or any other node with a DICT output."
                    ),
                ),
            ],
            outputs=[
                DOC.Output(
                    tooltip=(
                        "The whole document on one wire: its markup, the metadata filled in "
                        "here, and word and character counts. Nothing is written to disk; wire "
                        "it into View DOC Metadata or Save DOC."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        text="",
        text_format=PLAIN_TEXT,
        title="",
        description="",
        author="",
        copyright="",
        language="",
        keywords="",
        custom_metadata=None,
    ) -> io.NodeOutput:
        document = Document.build(
            _content(text, text_format),
            metadata.from_dict(
                {
                    "title": title,
                    "description": description,
                    "author": author,
                    "copyright": copyright,
                    "language": language,
                    "keywords": keywords,
                    "custom": _custom(custom_metadata),
                }
            ),
        )
        if not document.plain_text:
            logger.warning(
                "Text to DOC built a document with no readable text in it, because %s. It "
                "carries the metadata that was filled in and nothing to read, so a node that "
                "saves or exports it writes an empty document.",
                "the text input is empty"
                if not str(text or "").strip()
                else "the markup in the text input holds nothing a reader would see",
            )
        else:
            logger.debug(
                "built a document of %d word(s) and %d character(s) from %s",
                document.word_count, document.character_count, text_format,
            )
        return io.NodeOutput(document)


def _content(text: str, text_format: str) -> str:
    """The markup one run of the node produces.

    Args:
        text: The ``text`` input, as written or as another node produced it.
        text_format: :data:`PLAIN_TEXT` or :data:`HTML`.

    Returns:
        The document markup. Plain text is wrapped in paragraphs, and markup is taken as
        written with script and frame markup removed while ``document.clean_html`` is on.
    """
    written = str(text or "")
    if text_format != HTML:
        return compose.markup_from_text(written)
    if not config.load_config()["document"]["clean_html"]:
        return written
    cleaned, removed = markup.clean(written)
    if removed:
        logger.info(
            "Text to DOC removed %s from the document it built. The text input is unchanged. "
            "Set document.clean_html to false in config.yaml to carry the markup into the "
            "document exactly as written.",
            markup.describe(removed),
        )
    return cleaned


def _custom(value: object) -> Mapping:
    """The custom metadata pairs the DICT input carries.

    Args:
        value: Whatever arrived on ``custom_metadata``, or ``None`` when nothing is
            connected to it.

    Returns:
        The pairs, for the metadata record to read as text. An empty mapping where the
        input is unconnected.

    Raises:
        ValueError: The value is not a mapping of names to values, so the pairs it was
            meant to carry cannot be read, and dropping them would put out a document
            missing metadata that was wired into it.
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(
            f"the custom_metadata input takes a dictionary of names and values, and was "
            f"given {type(value).__name__}.\n"
            f"  Build one with 'Text Dictionary New', or leave the input unconnected to "
            f"carry no extra metadata."
        )
    return value
