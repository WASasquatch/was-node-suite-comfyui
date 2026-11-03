"""Write texts to numbered text files, or each under a wired name, in a folder the pack may save into."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LIST
from ...modules.interface import file_report, run_result
from ...modules.io import rooted, text_batch
from ...modules.state import history
from ...modules.util import sandbox

logger = log.get_logger("nodes.io")


def _items(value) -> list:
    """One input's values as the node receives them: a list as it is, anything else as one."""
    return value if isinstance(value, list) else [value]


def _at(values: list, index: int):
    """The value one pass reads, the last repeating past the end of a shorter list."""
    return values[min(index, len(values) - 1)]


def _place(root: str, prefix: str) -> tuple[str, str, str]:
    """``(root, folders, name)`` a prefix names below a root, slashes forward."""
    below, _, leaf = str(prefix or "").replace("\\", "/").rpartition("/")
    return root, below, leaf


def _where(root: str, below: str) -> str:
    """A destination as its root's name and the folders below it, never an absolute path."""
    return f"{root or rooted.DEFAULT}/{below}".rstrip("/")


class SaveTextFile(io.ComfyNode):
    """Write each text to a numbered file, or to the file its wired name gives."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Save Text File",
            display_name="Save Text File",
            search_aliases=[
                "Save Text File", "write text", "text file", "save captions", "filenames",
            ],
            category="WAS Suite/IO",
            is_input_list=True,
            description=(
                "Write text to a numbered file in the folder picked with root, named from "
                "filename_prefix, the number and an optional suffix; a STRING list writes one "
                "file per text. Wired filenames write each text under its own name instead, so "
                "edited captions go back beside their images. Padding 0 drops the number and "
                "replaces the file."
            ),
            inputs=[
                io.String.Input(
                    "text",
                    multiline=True,
                    placeholder="Eg: the file contents",
                    tooltip=(
                        "File contents; STRING, as `a tabby cat`, written exactly as given. A "
                        "STRING list writes one file per text. Empty writes an empty file; with "
                        "filenames wired, an empty text leaves its file alone."
                    ),
                ),
                io.Combo.Input(
                    "root",
                    options=rooted.save_options(),
                    tooltip=(
                        "`output`, `temp` or `input` = ComfyUI's own folders, `input` being "
                        "where loaders read uploads and datasets from; any folder under "
                        "paths.allow_write or paths.allow_read in config.yaml is listed by its "
                        "own name. filename_prefix names the path below it."
                    ),
                ),
                io.String.Input(
                    "filename_prefix",
                    default="ComfyUI",
                    tooltip=(
                        "Folders and name below root: `notes/day` writes notes/day_0001.txt. "
                        "Tokens are expanded, as `[time(%Y-%m-%d)]/notes`. Wired filenames "
                        "replace the name and keep the folders: `training/cats/` and `cat` "
                        "write training/cats/cat.txt."
                    ),
                ),
                io.String.Input(
                    "filename_delimiter",
                    default="_",
                    tooltip=(
                        "What sits between the name and the number: 'ComfyUI_0001.txt' with "
                        "the default, 'ComfyUI0001.txt' if cleared."
                    ),
                ),
                io.Int.Input(
                    "filename_number_padding",
                    default=4,
                    min=0,
                    max=9,
                    step=1,
                    tooltip=(
                        "4 = '_0001'; 1 = '_1'; 0 = no number, one file replaced every run. "
                        "With filenames, 0 writes each name as given, leaving a file already "
                        "holding its text untouched; above 0 numbers it and never replaces."
                    ),
                ),
                io.String.Input(
                    "file_extension",
                    default=".txt",
                    optional=True,
                    tooltip=(
                        "Ending of the file name, leading dot included: '.txt', '.json', "
                        "'.csv'. It only names the file; the text is written exactly as it "
                        "arrives either way."
                    ),
                ),
                io.String.Input(
                    "encoding",
                    default="utf-8",
                    optional=True,
                    tooltip=(
                        "Character encoding the file is written in. 'utf-8' handles every "
                        "language and is what almost everything reads; change it only for a "
                        "program that insists on something else, such as 'latin-1'."
                    ),
                ),
                io.String.Input(
                    "filename_suffix",
                    default="",
                    optional=True,
                    tooltip=(
                        "Extra text placed after the number and before the extension, so a "
                        "suffix of '_caption' gives 'ComfyUI_0001_caption.txt'. Empty by "
                        "default."
                    ),
                ),
                io.MultiType.Input(
                    "filenames",
                    [LIST, io.String],
                    optional=True,
                    tooltip=(
                        "One name per text, as names from Load Text Batch or filename_list "
                        "from Load Image Batch: `cat` writes cat.txt, `set_b/cat` writes "
                        "set_b/cat.txt, below filename_prefix's folders. One text for every "
                        "name, or one per name. Several lines are several names."
                    ),
                ),
            ],
            outputs=[],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        text,
        root=rooted.DEFAULT,
        filename_prefix="ComfyUI",
        filename_delimiter="_",
        filename_number_padding=4,
        file_extension=".txt",
        encoding="utf-8",
        filename_suffix="",
        filenames=None,
    ) -> io.NodeOutput:
        """Write one file per pass of the longest input list, or one per wired name.

        Raises:
            OSError: A folder could not be made, or a file could not be written.
            PathNotAllowed: The root, the prefix or a name leaves every folder this pack may
                save into.
            ValueError: The texts are neither one nor one per name, or filenames holds no
                names.
        """
        settings = {
            "text": text, "root": root, "filename_prefix": filename_prefix,
            "filename_delimiter": filename_delimiter,
            "filename_number_padding": filename_number_padding,
            "file_extension": file_extension, "encoding": encoding,
            "filename_suffix": filename_suffix,
        }
        lists = {key: _items(value) for key, value in settings.items()}
        lists["text"] = ["" if item is None else str(item) for item in lists["text"]] or [""]
        names = text_batch.name_list(filenames)
        if names is None:
            passes = max(len(values) for values in lists.values())
            entries = [
                {key: _at(values, index) for key, values in lists.items()}
                for index in range(passes)
            ]
            saved = cls.numbered(entries)
        else:
            entries = cls.paired(lists, names)
            saved = cls.by_name(entries)

        if saved.paths:
            history.update_history_text_files([str(path) for path in saved.paths])
        cls.report(entries, saved, names is not None)
        return io.NodeOutput(ui={"text": [entry["text"] for entry in entries]})

    @staticmethod
    def paired(lists: dict, names: list[str]) -> list[dict]:
        """One entry per name, each with its text and the settings at its place.

        Args:
            lists: Every input as a list, the texts already strings.
            names: The wired names, in order.

        Returns:
            One mapping per name, holding a ``name`` beside the inputs.

        Raises:
            ValueError: No name arrived, or more than one text and not one per name.
        """
        texts = lists["text"]
        if not names:
            raise ValueError(
                "filenames is wired and holds no names, so Save Text File has nowhere to "
                "write. Wire one name per text, or disconnect it to number the files"
            )
        if len(texts) == 1:
            texts = texts * len(names)
        elif len(texts) != len(names):
            raise ValueError(
                f"{len(texts)} texts arrived for {len(names)} filenames, and they pair by "
                f"position, so none can be matched. Wire one text per name, in the names' "
                f"order, or a single text for all of them. To leave a name out, blank its "
                f"text instead of removing it"
            )
        entries = []
        for index, name in enumerate(names):
            entry = {key: _at(values, index) for key, values in lists.items()}
            entry["text"] = texts[index]
            entry["name"] = name
            entries.append(entry)
        return entries

    @staticmethod
    def numbered(entries: list[dict]) -> text_batch.Saved:
        """Write each entry's text to the next numbered file its prefix names.

        Args:
            entries: One mapping of the node's inputs per pass.

        Returns:
            What was written.

        Raises:
            OSError: A folder could not be made, or a file could not be written.
            PathNotAllowed: The root or the prefix leaves every save root.
        """
        paths, written, replaced = [], 0, 0
        for entry in entries:
            root, below, leaf = _place(entry["root"], entry["filename_prefix"])
            directory = rooted.save_destination(root, below)
            if entry["text"].strip() == "":
                logger.error(
                    "Save Text File was handed empty text, so it wrote an empty file. Wire the "
                    "text to save into text"
                )
            name = text_batch.next_free(
                directory, leaf, entry["filename_delimiter"],
                int(entry["filename_number_padding"]), entry["filename_suffix"] or "",
                entry["file_extension"] or "",
            )
            path, existed = text_batch.write_text(
                directory, name, entry["text"], entry["encoding"] or "utf-8",
            )
            paths.append(path)
            replaced += existed
            written += not existed
        return text_batch.Saved(paths, written, replaced, 0, 0)

    @staticmethod
    def by_name(entries: list[dict]) -> text_batch.Saved:
        """Write each entry's text under its own name, in the folders its prefix names.

        Args:
            entries: One mapping of the node's inputs per name, holding the ``name``.

        Returns:
            What was written, replaced, left unchanged and left out.

        Raises:
            OSError: A folder could not be made, or a file could not be written.
            PathNotAllowed: The root, the prefix or a name leaves every save root.
        """
        paths, claimed = [], set()
        written = replaced = unchanged = left_out = 0
        for entry in entries:
            extension = entry["file_extension"] or ""
            suffix = entry["filename_suffix"] or ""
            encoding = entry["encoding"] or "utf-8"
            name = text_batch.stem(entry["name"], extension)
            if not entry["text"].strip() or not name:
                left_out += 1
                continue
            root, below, _leaf = _place(entry["root"], entry["filename_prefix"])
            directory = rooted.save_destination(root, below)
            padding = int(entry["filename_number_padding"])
            if padding == 0:
                file = f"{name}{suffix}{extension}"
                target = sandbox.resolve_save_file(directory, file)
                key = str(target).lower()
                if key in claimed:
                    logger.warning(
                        "Save Text File was handed `%s` twice in one run; the later text "
                        "replaces the earlier", sandbox._shown(name),
                    )
                claimed.add(key)
                if text_batch.same_text(target, entry["text"], encoding):
                    unchanged += 1
                    continue
                path, existed = text_batch.write_text(directory, file, entry["text"], encoding)
            else:
                sub, _, leaf = name.rpartition("/")
                folder = sandbox.resolve_save_file(directory, sub) if sub else directory
                file = text_batch.next_free(
                    folder, leaf, entry["filename_delimiter"], padding, suffix, extension,
                )
                path, existed = text_batch.write_text(folder, file, entry["text"], encoding)
            paths.append(path)
            replaced += existed
            written += not existed
        if left_out:
            logger.info(
                "Save Text File left out %d of %d text(s) whose text or name is blank",
                left_out, len(entries),
            )
        return text_batch.Saved(paths, written, replaced, unchanged, left_out)

    @staticmethod
    def report(entries: list[dict], saved: text_batch.Saved, named: bool) -> None:
        """Publish what the run wrote, for the file report on this node.

        Never raises, and never changes what the node returns.

        Args:
            entries: The run's entries, the first naming the folder.
            saved: What was written.
            named: Whether the texts were written under wired names.
        """
        try:
            first = entries[0]
            root, below, _leaf = _place(first["root"], first["filename_prefix"])
            facts = {"written": f"{saved.written} new, {saved.replaced} replaced"}
            if named:
                facts["unchanged"] = saved.unchanged
            if saved.left_out:
                facts["left out"] = f"{saved.left_out} with a blank text or name"
            if not saved.paths and saved.unchanged:
                run_result.publish(
                    status=run_result.OK,
                    summary=(
                        f"nothing to write: {saved.unchanged} file(s) in {_where(root, below)} "
                        f"already hold their text"
                    ),
                    counts={"files": 0, "writes": 0},
                    facts={"folder": _where(root, below), **facts},
                )
                return
            file_report.publish(
                saved.paths,
                kind=(first["file_extension"] or "").lstrip(".") or "no extension",
                folder=_where(root, below),
                facts=facts,
            )
        except Exception as error:
            logger.debug("the text save was not reported (%s)", error)
