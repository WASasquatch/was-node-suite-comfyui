"""Write a document, or a string, to a file, in a plain format or a rich one."""

from __future__ import annotations

import os
import re
from pathlib import Path

from comfy_api.latest import io, ui

from ...modules.io import rooted
from ...modules import config, deps
from ...modules.compat.types import DOC
from ...modules.document import export, formats, summary
from ...modules.document.container import Document, is_document
from ...modules.log import get_logger
from ...modules.state import history
from ...modules.util import filenames, sandbox

logger = get_logger("nodes.document")

#: Directory the node writes into unless the widget is changed, the same default the pack's
#: other save nodes carry.
DEFAULT_PATH = "./ComfyUI/output/[time(%Y-%m-%d)]"

#: How many characters of a written file the node shows before cutting the preview short.
PREVIEW_LIMIT = 4096

#: Markup a word processor cannot be given as CSS. A ``<style>`` block and a class selector
#: are read for the PDF, which lays out the HTML itself, and not for the other two, which are
#: built element by element from what each element declares for itself.
_STYLESHEET_MARKUP = re.compile(r"<\s*style[\s>]|\sclass\s*=", re.IGNORECASE)


class SaveDoc(io.ComfyNode):
    """Write a document or a string to a file.

    ``file_format`` decides the extension and what the file holds.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASSaveDOC",
            display_name="Save DOC",
            search_aliases=[
                "WASSaveDOC",
                "Save DOC",
                "save document",
                "export document",
                "wasdoc",
                "save html",
                "write file",
                "save markdown",
                "save docx",
                "save word",
                "save odt",
                "save pdf",
                "export pdf",
            ],
            category="WAS Suite/Document",
            description=(
                (
                    "Write a document, or plain text from any string output, to a file: a "
                    ".wasdoc that keeps everything, an HTML page, plain text, or a laid-out "
                    ".docx, .odt or .pdf."
                )
            ),
            inputs=[
                io.MultiType.Input(
                    "doc",
                    [DOC, io.String],
                    tooltip=(
                        "What to write. A DOC carries its markup, its metadata and every "
                        "embedded file; a string is written through unchanged, which suits "
                        "generated code. This socket takes a connection."
                    ),
                ),
                io.Combo.Input(
                    "file_format",
                    # The dependency-free formats first, then the three written through a
                    # document library. Every machine offers all of them, whether or not
                    # features.document_export is on, so a saved workflow validates
                    # wherever it is opened.
                    options=list(formats.FORMATS) + list(export.FORMATS),
                    default=formats.CONTAINER,
                    tooltip=(
                        "'.wasdoc' = loses nothing, the only format keeping metadata and "
                        "embedded files; '.html' = a whole page, metadata in its head; plain "
                        "formats = the text alone, tags removed; '.docx', '.odt', '.pdf' = "
                        "laid out, some styling lost."
                    ),
                ),
                io.Combo.Input(
                    "root",
                    options=rooted.options(),
                    tooltip=(
                        "Which folder the file lands in: ComfyUI's own 'output' or 'temp', "
                        "or any folder added under paths.allow_write in config.yaml, listed "
                        "by its own name. filename_prefix names the part below it, so "
                        "'[time(%Y-%m-%d)]/notes' files each day's under a dated folder."
                    ),
                ),
                io.String.Input(
                    "filename_prefix",
                    default="ComfyUI",
                    tooltip=(
                        "The name part of each file, before the number. Tokens are expanded "
                        "here too, so a date or a custom token can go in the name rather "
                        "than the folder."
                    ),
                ),
                io.String.Input(
                    "filename_delimiter",
                    default="_",
                    tooltip=(
                        "What sits between the name and the number: 'ComfyUI_0001.wasdoc' "
                        "with the default, 'ComfyUI0001.wasdoc' if cleared."
                    ),
                ),
                io.Int.Input(
                    "filename_number_padding",
                    default=4,
                    min=0,
                    max=9,
                    step=1,
                    tooltip=(
                        "How many digits the number is padded to with leading zeros: 4 "
                        "gives '_0001', 1 gives '_1'. 0 drops the number and rewrites the "
                        "same file every run."
                    ),
                ),
                io.String.Input(
                    "filename_suffix",
                    default="",
                    optional=True,
                    tooltip=(
                        "Extra text placed after the number and before the extension, so a "
                        "suffix of '_draft' gives 'ComfyUI_0001_draft.wasdoc'. Empty by "
                        "default."
                    ),
                ),
                io.Combo.Input(
                    "page_size",
                    options=list(export.PAGE_SIZES),
                    default="A4",
                    optional=True,
                    tooltip=(
                        "Paper the page is laid out on, for '.docx', '.odt' and '.pdf' "
                        "only. A4 is the size used everywhere but North America, where "
                        "Letter is."
                    ),
                ),
                io.Combo.Input(
                    "orientation",
                    options=list(export.ORIENTATIONS),
                    default=export.ORIENTATIONS[0],
                    optional=True,
                    tooltip=(
                        "Which way round the page is, for '.docx', '.odt' and '.pdf' only. "
                        "'landscape' swaps the width and the height, which a wide table needs; "
                        "'portrait' is the usual way up for prose."
                    ),
                ),
                io.Float.Input(
                    "margin_mm",
                    default=export.DEFAULT_MARGIN_MM,
                    min=0.0,
                    max=100.0,
                    step=0.5,
                    optional=True,
                    tooltip=(
                        "Blank edge left on all four sides in millimetres, for '.docx', "
                        "'.odt' and '.pdf' only. 25.4 is one inch, and 0 runs the text to "
                        "the very edge."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="file_path",
                    tooltip=(
                        "The full path of the file that was written, so a later node can "
                        "report it, print it, or collect it into an archive. It names the "
                        "file that actually got the bytes, numbering and all, rather than "
                        "the folder that was asked for."
                    ),
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        doc,
        file_format=formats.CONTAINER,
        root=rooted.DEFAULT,
        filename_prefix="ComfyUI",
        filename_delimiter="_",
        filename_number_padding=4,
        filename_suffix="",
        page_size="A4",
        orientation=export.ORIENTATIONS[0],
        margin_mm=export.DEFAULT_MARGIN_MM,
    ) -> io.NodeOutput:
        """Build the file's bytes, then write them to the next unused name.

        Returns:
            The absolute path of the file written, and a note of what went into it.

        Raises:
            NotADocument: ``doc`` carried neither a document nor a string.
            ValueError: ``file_format`` names no format this node writes, the document
                export group is off for one that needs it, or the library could not lay the
                document out.
            DependencyError: The library that format needs is missing or unusable.
            OSError: The folder could not be created, or the file could not be written.
            PathNotAllowed: ``path`` resolved outside every permitted write root, or
                ``filename_prefix`` named a file outside the folder.
        """
        source = formats.require_source(doc, "doc")
        if export.writes(file_format):
            extension = export.normalized(file_format)
            _require_export_group(extension)
            page = export.Page.build(page_size, orientation, margin_mm)
            data = export.export(formats.as_document(source), extension, page)
        else:
            extension = formats.normalized(file_format)
            data = formats.payload(source, extension)
        _report_losses(source, extension, data)

        below, _, leaf = (filename_prefix or "").replace("\\", "/").rpartition("/")
        directory = rooted.destination(root, below)
        filename_prefix = leaf
        if not directory.exists():
            logger.warning("the path `%s` doesn't exist! Creating it...", directory)
            try:
                os.makedirs(directory, exist_ok=True)
            except OSError as error:
                logger.error(
                    "the path `%s` could not be created! Is there write access?\n%s",
                    directory, error,
                )
                raise

        padding = int(filename_number_padding)
        filename = filenames.generate_filename(
            directory, filename_prefix, filename_delimiter, padding, extension, filename_suffix
        )
        target = sandbox.resolve_write_file(directory, filename)
        replaced = target.exists()
        with open(target, "wb") as handle:
            handle.write(data)

        if extension != formats.CONTAINER and not export.writes(extension):
            # The text-file history drives a combo that offers each entry to a text loader.
            # A container is a zip and so is a .docx or an .odt, and a PDF is not text
            # either, so only the readable formats are recorded there.
            history.update_history_text_files(str(target))
        logger.info(
            "%s %d byte(s) of %s to %s",
            "replaced" if replaced else "wrote", len(data), extension, target,
        )
        note = _note(target, extension, data, source, replaced)
        return io.NodeOutput(str(target), ui=ui.PreviewText(note))


def _require_export_group(extension: str) -> None:
    """Refuse a rich format while its feature group is set to false.

    Args:
        extension: One of :data:`modules.document.export.FORMATS`.

    Raises:
        ValueError: ``features.document_export`` is false, naming the setting, the install
            command and the formats that need neither.
    """
    if config.load_config()["features"]["document_export"]:
        return
    packages = [deps.PIP_NAMES.get(name, name) for name in export.PACKAGES.values()]
    raise ValueError(
        f"a {extension} file is written through {export.PACKAGES[extension]}, and the "
        f"{export.FEATURE} group is set to false in your config, so this pack will not "
        f"write one.\n"
        f"  That group ships off, so this is the one thing to turn on for these three\n"
        f"  formats. Put this in config.yaml and restart ComfyUI:\n"
        f"      features:\n"
        f"        document_export: true\n"
        f"  Then install the three libraries, all of them permissive-licensed and pure "
        f"python:\n"
        f"      {deps.install_command(*packages, feature=export.FEATURE)}\n"
        f"  Or pick {formats.CONTAINER} or {formats.MARKUP}, which need nothing installed "
        f"and lose nothing."
    )


def _report_losses(source: Document | str, extension: str, data: bytes) -> None:
    """Log whatever the chosen format cannot carry.

    Args:
        source: The document or string being written.
        extension: The format, from ``formats.normalized`` or ``export.normalized``.
        data: The bytes the format produced, so an empty file is reported as one.
    """
    if not data:
        logger.error(
            "there is nothing to save! %s, so an empty %s file is being written.",
            "The document holds nothing a reader would see"
            if is_document(source)
            else "The text on the doc input is empty",
            extension,
        )
    if export.writes(extension):
        _report_export_losses(source, extension)
        return
    if extension == formats.CONTAINER:
        if not is_document(source):
            logger.info(
                "Save DOC wrapped the text it was given into a document to write a %s "
                "container. It carries the text and its timestamps and no title, author or "
                "copyright statement, which a string has none of. Put Text to DOC in front "
                "of this node to fill those in, or to read the text as HTML markup.",
                formats.CONTAINER,
            )
        return
    if not is_document(source):
        return
    if source.assets:
        logger.warning(
            "Save DOC dropped the %d file(s) embedded in %r: only a %s container can hold "
            "them, and a %s file cannot. A picture in the document is now a broken link. "
            "Save the document as %s as well to keep them.",
            len(source.assets), source.metadata.title or "the document",
            formats.CONTAINER, extension, formats.CONTAINER,
        )
    if extension in formats.TEXT_FORMATS and summary.has_metadata(source.metadata):
        logger.info(
            "the title, author and the rest of the metadata in %r were not written to the "
            "%s file, which holds the text alone. %s keeps the metadata inside the file, and "
            "%s writes it into the page head.",
            source.metadata.title or "the document", extension,
            formats.CONTAINER, formats.MARKUP,
        )


def _report_export_losses(source: Document | str, extension: str) -> None:
    """Log what a conversion into one of the rich formats leaves behind.

    Args:
        source: The document or string being written.
        extension: One of :data:`modules.document.export.FORMATS`.
    """
    content = source.content if is_document(source) else ""
    if extension != export.PDF and _STYLESHEET_MARKUP.search(content):
        logger.info(
            "the document carries a <style> block or a class attribute, and a %s file is "
            "built element by element rather than laid out as HTML, so only the styling each "
            "element declares for itself reaches the file. Saving as %s instead keeps the "
            "stylesheet, since that one is laid out from the markup.",
            extension, export.PDF,
        )
    if extension != export.PDF or not is_document(source):
        return
    metadata = source.metadata
    unwritten = [
        name
        for name, value in (
            ("copyright", metadata.copyright),
            ("language", metadata.language),
            ("created and modified", metadata.created or metadata.modified),
            ("custom pairs", bool(metadata.custom)),
        )
        if value
    ]
    if unwritten:
        logger.info(
            "the %s of %r were not written into the PDF's own information fields, which hold "
            "the title, the author, the description and the keywords and have nowhere for "
            "the rest. %s and %s carry all of it, and %s keeps it as metadata.",
            ", ".join(unwritten), metadata.title or "the document",
            export.DOCX, export.ODT, formats.CONTAINER,
        )


def _note(
    target: Path, extension: str, data: bytes, source: Document | str, replaced: bool
) -> str:
    """What the node shows on itself after a write.

    Args:
        target: The file that was written.
        extension: The format it was written in.
        data: The bytes written, for the size.
        source: The document or string they came from.
        replaced: Whether a file of that name was already there.

    Returns:
        A few lines naming the file, its size and what the format put in it, then the title
        and the two counts where a document was written, then the text itself for a format
        that holds text, cut short after :data:`PREVIEW_LIMIT` characters. A container is a
        zip, so its bytes are not shown.
    """
    lines = [
        f"{'replaced' if replaced else 'wrote'} {target.name} ({_size(len(data))})",
        str(target),
        _written(extension, source),
    ]
    if is_document(source):
        lines.append(
            f"{source.metadata.title or '(no title)'}: {source.word_count} word(s), "
            f"{source.character_count} character(s)"
        )
    if extension in formats.TEXT_FORMATS or extension == formats.MARKUP:
        preview = data.decode(formats.ENCODING, "replace")
        if preview.strip():
            lines.append("")
            lines.append(
                preview if len(preview) <= PREVIEW_LIMIT
                else preview[:PREVIEW_LIMIT] + f"\n... {len(preview) - PREVIEW_LIMIT} more character(s)"
            )
    return "\n".join(lines)


def _written(extension: str, source: Document | str) -> str:
    """One line saying what the chosen format put in the file."""
    if extension == formats.CONTAINER:
        embedded = len(source.assets) if is_document(source) else 0
        return (
            f"the document container: content.html, meta.json and {embedded} embedded "
            f"file(s)"
        )
    if extension == formats.MARKUP:
        return "a whole HTML page, with the metadata in its head"
    if export.writes(extension):
        return (
            f"the document laid out through {export.PACKAGES[extension]}, a conversion "
            f"rather than a copy"
        )
    if is_document(source):
        return (
            "the document's text, with its markup removed, its entities decoded and a blank "
            "line between one block and the next"
        )
    return "the text exactly as it arrived"


def _size(count: int) -> str:
    """A byte count as a number a person reads."""
    if count < 1024:
        return f"{count} byte(s)"
    if count < 1024 * 1024:
        return f"{count / 1024:.1f} KB"
    return f"{count / (1024 * 1024):.1f} MB"
