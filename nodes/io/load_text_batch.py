"""Read a folder of text files, or the file of each wired name, as lists of texts and names."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.lists import require_values
from ...modules.compat.types import LIST
from ...modules.interface import run_result
from ...modules.io import picker, text_batch
from ...modules.util import sandbox

logger = log.get_logger("nodes.io")

#: How many names and snippets the report shows.
REPORT_ITEMS = 5


def _first(value, default=""):
    """The one value a widget input holds, read out of the list an input-list node receives."""
    if isinstance(value, list):
        value = value[0] if value else None
    return default if value is None else value


def _extension(value) -> str:
    """The extension as typed, with a leading dot added and nothing that names a folder.

    Raises:
        ValueError: The extension holds a slash.
    """
    text = str(value or "").strip()
    if "/" in text or "\\" in text:
        raise ValueError(
            f"the extension `{sandbox._shown(text)}` holds a slash. Write the ending alone, "
            f"such as `.txt`, and put folders in the names"
        )
    return f".{text}" if text and not text.startswith(".") else text


def _label(folder: str) -> str:
    """The folder as its menu names it, never an absolute path."""
    try:
        return picker.folder_label(folder)
    except sandbox.PathNotAllowed:
        return sandbox._shown(folder)


class LoadTextBatch(io.ComfyNode):
    """Read every text file a pattern matches, or the file each wired name gives."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLoadTextBatch",
            display_name="Load Text Batch",
            search_aliases=[
                "WASLoadTextBatch", "Load Text Batch", "captions", "load captions",
                "text folder", "dataset captions", "txt files",
            ],
            category="WAS Suite/IO",
            is_input_list=True,
            description=(
                "Read a folder of text files, such as a training set's captions, as one LIST "
                "and as a STRING list that runs the graph below once per file, each with its "
                "name. Wired filenames read the caption beside each image instead, in the "
                "images' order, ready to edit and save back by name."
            ),
            inputs=[
                picker.path_input(io.Combo.Input(
                    "folder",
                    options=picker.folders(),
                    tooltip=(
                        "Which folder to read. A bare 'input', 'output' or 'temp' is that "
                        "folder itself; 'training/cats [input]' is that folder below it. "
                        "Folders under paths.allow_read in config.yaml are listed under their "
                        "own names."
                    ),
                ), folder=True),
                io.String.Input(
                    "pattern",
                    default="*.txt",
                    multiline=False,
                    tooltip=(
                        "Which files to read when filenames is not wired: `*.txt` = every .txt "
                        "in the folder; `**/*.txt` = in its subfolders too; `cat_*.txt` = only "
                        "those named that way. Ignored while filenames is wired."
                    ),
                ),
                io.String.Input(
                    "extension",
                    default=".txt",
                    multiline=False,
                    tooltip=(
                        "Ending added to each wired name: `.txt` reads cat.txt for `cat`; "
                        "`.caption` reads cat.caption. An image ending on the name, as "
                        "`cat.png`, is dropped first. Ignored while filenames is not wired."
                    ),
                ),
                io.MultiType.Input(
                    "filenames",
                    [LIST, io.String],
                    optional=True,
                    tooltip=(
                        "One name per text to read, as filename_list from Load Image Batch: "
                        "`cat` reads cat.txt, `set_b/cat` reads set_b/cat.txt. A name with no "
                        "readable file gives an empty text in its place, so entry 3 always "
                        "belongs to name 3. Several lines are several names."
                    ),
                ),
            ],
            outputs=[
                LIST.Output(
                    display_name="texts",
                    tooltip=(
                        "Every text on one wire, in order, for Text List Get, Text List Slice "
                        "and Text List Length. Entry 3 here is the text of name 3 in names."
                    ),
                ),
                LIST.Output(
                    display_name="names",
                    tooltip=(
                        "The name each text came from, on one wire and in the same order: "
                        "the path below the folder without its ending, as `cat` or "
                        "`set_b/cat`, the form Image Save and Save Text File take as filenames."
                    ),
                ),
                io.String.Output(
                    display_name="text",
                    is_output_list=True,
                    tooltip=(
                        "The same texts as a STRING list, so the graph below runs once per "
                        "file: wire it into a text edit, then into Save Text File's text."
                    ),
                ),
                io.String.Output(
                    display_name="name",
                    is_output_list=True,
                    tooltip=(
                        "The name that goes with each run of text. Wire it into Save Text "
                        "File's filenames to write each edited text back under its own name."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip="How many entries both lists hold: files read, or names wired.",
                ),
                io.Int.Output(
                    display_name="missing",
                    tooltip=(
                        "0 = every wired name had a readable file; 3 = three names got an "
                        "empty text. Always 0 while filenames is not wired. The log names "
                        "each file that was unreadable or not UTF-8."
                    ),
                ),
                io.String.Output(
                    display_name="folder",
                    tooltip=(
                        "The picked folder below its root with a trailing slash: "
                        "`training/tus [input]` = `training/tus/`; a root itself = empty. Wire it "
                        "into Save Text File's filename_prefix to write back into the same folder."
                    ),
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, folder=None, pattern=None, extension=None, filenames=None):
        """Read again when a file the pattern or the extension covers is added, removed or changed."""
        del filenames
        try:
            base = picker.resolve_folder(_first(folder))
            ending = _extension(_first(extension, ".txt"))
        except ValueError:
            return float("NaN")
        return text_batch.fingerprint(base, str(_first(pattern, "*.txt")), ending)

    @classmethod
    def execute(cls, folder=None, pattern=None, extension=None, filenames=None) -> io.NodeOutput:
        """Read the folder by pattern, or the file of every wired name.

        Raises:
            PathNotAllowed: The folder, the pattern or a name leaves every folder this pack
                may read.
            ValueError: The folder is not there, the extension holds a slash, or nothing was
                read.
        """
        chosen = str(_first(folder)).strip()
        wanted = str(_first(pattern, "*.txt"))
        ending = _extension(_first(extension, ".txt"))
        if not chosen:
            raise ValueError("no folder is picked. Pick one from the folder menu, or wire a path")
        base = picker.resolve_folder(chosen)
        label = _label(chosen)
        if base is None or not base.is_dir():
            raise ValueError(
                f"`{label}` names no folder that is there. Pick another from the menu, or add "
                f"its folder to paths.allow_read in config.yaml"
            )

        names = text_batch.name_list(filenames)
        if names is None:
            batch = text_batch.read_folder(base, wanted, label)
            require_values(batch.texts, cls.nothing(label, wanted, batch))
        else:
            require_values(
                names,
                f"filenames is wired and holds no names, so Load Text Batch has nothing to "
                f"read in {label}. Wire a list of names, such as filename_list from Load "
                f"Image Batch, or disconnect it to read by pattern",
            )
            batch = text_batch.read_named(base, names, ending, label)

        logger.info(
            "Load Text Batch read %d text(s) from %s, %d missing, %d skipped",
            len(batch.texts) - batch.missing, label, batch.missing, batch.skipped,
        )
        cls.report(label, wanted if names is None else f"filenames + {ending}", batch)
        texts, found = batch.texts, batch.names
        return io.NodeOutput(
            texts, found, list(texts), list(found), len(texts), batch.missing,
            picker.folder_prefix(chosen),
        )

    @staticmethod
    def nothing(label: str, pattern: str, batch: text_batch.Batch) -> str:
        """The message for a read that found no text.

        Args:
            label: The folder's menu label.
            pattern: The pattern as written.
            batch: What the read found.

        Returns:
            A message naming the folder, the pattern and why nothing came of it.
        """
        if not batch.matched:
            return (
                f"no file in {label} matches `{pattern}`, so Load Text Batch has nothing to "
                f"hand on. Use `**/*.txt` to look in subfolders too, or pick another folder"
            )
        return (
            f"every one of the {batch.matched} file(s) `{pattern}` matched in {label} was "
            f"skipped, so Load Text Batch has nothing to hand on. The log names each one and "
            f"why: not UTF-8, unreadable, or over {text_batch.MAX_FILE_BYTES} bytes"
        )

    @staticmethod
    def report(label: str, pattern: str, batch: text_batch.Batch) -> None:
        """Publish what the read found, for the panel on this node.

        Never raises, and never changes what the node returns.

        Args:
            label: The folder's menu label.
            pattern: The pattern, or what the names were read with.
            batch: What the read found.
        """
        try:
            read = len(batch.texts) - batch.missing
            trouble = batch.missing or batch.skipped or batch.left_out
            items = [
                {"text": text_batch.snippet(text) or "(empty)", "note": name}
                for text, name in zip(batch.texts[:REPORT_ITEMS], batch.names[:REPORT_ITEMS])
            ]
            facts = {
                "folder": label,
                "pattern": pattern,
                "read": read,
                "missing": batch.missing,
                "skipped": batch.skipped,
            }
            if batch.left_out:
                facts["past limit"] = batch.left_out
            run_result.publish(
                status=run_result.WARNING if trouble else run_result.OK,
                summary=f"{read} text(s) read from {label}",
                counts={"texts": len(batch.texts)},
                facts=facts,
                items=items,
                items_total=len(batch.texts),
            )
        except Exception as error:
            logger.debug("the text batch was not reported (%s)", error)
