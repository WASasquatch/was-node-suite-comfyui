"""Write a dataset's captions, or its pictures and captions, into a folder the pack may save into."""

from __future__ import annotations

import os

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import DATASET
from ...modules.dataset import dataset as datasets
from ...modules.dataset import health, save
from ...modules.interface import file_report, progress, run_result
from ...modules.io import rooted

logger = log.get_logger("nodes.dataset")

#: Display name, written into the messages this node raises.
NAME = "Save Dataset"


class SaveDataset(io.ComfyNode):
    """Write a dataset's captions, or a copy of its pictures and captions, under each item's name."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASSaveDataset",
            display_name="Save Dataset",
            search_aliases=[
                "WASSaveDataset", "Save Dataset",
                "save captions",
                "write captions",
                "export dataset",
                "copy dataset",
                "training set",
                "lora dataset",
            ],
            category="WAS Suite/Dataset",
            description=(
                "Write a dataset's captions back beside its pictures, or export a copy of the "
                "set, pictures and captions, into another folder. Every file takes its item's "
                "name, so edited captions return to the pictures they describe."
            ),
            inputs=[
                DATASET.Input(
                    "dataset",
                    tooltip=(
                        "The pictures and captions to write: from Load Dataset, or Dataset "
                        "Update after the captions were edited. Each item is written under its "
                        "own name, such as `cat` or `set_b/cat`."
                    ),
                ),
                io.Combo.Input(
                    "root",
                    options=rooted.save_options(),
                    tooltip=(
                        "`input`, `output` or `temp` = ComfyUI's own folders, `input` being "
                        "where Load Dataset reads; any folder under paths.allow_write or "
                        "paths.allow_read in config.yaml is listed by its own name. folder "
                        "names the path below it."
                    ),
                ),
                io.String.Input(
                    "folder",
                    default="",
                    tooltip=(
                        "Folder below root, as `lora/cats`; empty = root itself. Tokens are "
                        "expanded, as `export/[time(%Y-%m-%d)]`. Point root and folder at the "
                        "dataset's own folder to save captions back in place."
                    ),
                ),
                io.Combo.Input(
                    "write",
                    options=list(save.WRITES),
                    tooltip=(
                        "`captions` = only the caption files, beside pictures already there; "
                        "`pictures and captions` = a copy of the set: each picture copied byte "
                        "for byte, or saved as PNG where a node replaced it, with its caption."
                    ),
                ),
                io.String.Input(
                    "caption_extension",
                    default=".txt",
                    tooltip=(
                        "The caption file's ending: `.txt` writes cat.txt beside cat.png; "
                        "`.caption` writes cat.caption. An item with no caption writes no "
                        "caption file, and nothing is ever deleted."
                    ),
                ),
                io.Combo.Input(
                    "existing",
                    options=list(save.EXISTING),
                    default="skip",
                    tooltip=(
                        "Where a file is already there: `replace` = write over it when its "
                        "content differs, leave it when the same; `skip` = leave it alone; "
                        "`number` = write under the next free name, as cat_0001.png with "
                        "cat_0001.txt."
                    ),
                ),
            ],
            outputs=[],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        dataset,
        root=rooted.DEFAULT,
        folder="",
        write=save.CAPTIONS,
        caption_extension=".txt",
        existing="skip",
    ) -> io.NodeOutput:
        """Write every item of the dataset and publish what was written.

        Raises:
            PathNotAllowed: The root, the folder or an item's name leaves every folder this
                pack may save into; nothing is written then.
            ValueError: No dataset arrived, the folder climbs out of root, or the caption
                extension is not one.
        """
        data = datasets.from_value(dataset, NAME)
        suffix = health.caption_suffix(caption_extension)
        destination = rooted.save_destination(root, str(folder or ""))
        label = _where(root, folder)
        saved = save.save(data, destination, write, suffix, existing, step=_stepper(len(data)))
        logger.info(
            "%s wrote %d new and %d replaced file(s) in %s; %d unchanged, %d skipped, "
            "%d uncaptioned", NAME, saved.written, saved.replaced, label, saved.unchanged,
            saved.skipped, saved.uncaptioned,
        )
        _report(saved, label, write, len(data))
        return io.NodeOutput()


def _where(root, folder) -> str:
    """A destination as its root's name and the folders below it, never an absolute path."""
    below = str(folder or "").strip().replace("\\", "/").strip("/")
    return f"{root or rooted.DEFAULT}/{below}".rstrip("/")


def _stepper(total: int):
    """A callable advancing the node's progress bar by one item and stopping on Cancel."""
    bar = progress.progress_bar(max(1, total))
    try:
        import comfy.model_management as management
    except ImportError:
        management = None

    def step():
        if management is not None:
            management.throw_exception_if_processing_interrupted()
        bar.update(1)

    return step


def _kinds(paths) -> str:
    """The extensions written, without their dots, as one comma separated line."""
    found = sorted({os.path.splitext(str(path))[1].lstrip(".").lower() for path in paths})
    return ", ".join(kind or "no extension" for kind in found)


def _idle(saved: save.Saved, label: str, write: str, total: int) -> tuple[str, str]:
    """The status and summary of a run that wrote nothing."""
    if not total:
        return run_result.WARNING, "the dataset is empty, so nothing was written"
    if saved.failed:
        return run_result.WARNING, (
            f"none of {saved.failed} file(s) reached {label}; the reason each one failed is in "
            f"the log"
        )
    if saved.skipped:
        return run_result.WARNING, (
            f"nothing written: {saved.skipped} file(s) in {label} were already there and "
            f"existing is `skip`. Pick `replace` to write over the ones that differ"
        )
    if saved.unchanged:
        return run_result.OK, (
            f"nothing to write: {saved.unchanged} file(s) in {label} already hold their content"
        )
    what = "carries a caption" if write == save.CAPTIONS else "holds a picture"
    return run_result.WARNING, f"no item {what}, so nothing was written"


def _report(saved: save.Saved, label: str, write: str, total: int) -> None:
    """Publish what the run wrote, for the file report on this node.

    Never raises, and never changes what the node returns.

    Args:
        saved: What was written.
        label: The folder as its root's name and the path below it.
        write: Which files were written.
        total: Items in the dataset.
    """
    try:
        if not run_result.watching():
            return
        written = f"{saved.written} new, {saved.replaced} replaced"
        if saved.renamed:
            written += f", {len(saved.renamed)} numbered"
        left = f"{saved.unchanged} unchanged, {saved.skipped} skipped"
        if saved.no_picture:
            left += f", {saved.no_picture} with no picture"
        facts = {
            "written": written,
            "left alone": left,
            "uncaptioned": f"{saved.uncaptioned} of {total}",
        }
        if not saved.paths:
            status, summary = _idle(saved, label, write, total)
            run_result.publish(
                status=status,
                summary=summary,
                counts={"files": 0, "writes": 0},
                facts={"folder": label, **facts},
            )
            return
        file_report.publish(
            saved.paths,
            intended=len(saved.paths) + saved.failed,
            kind=_kinds(saved.paths),
            folder=label,
            facts=facts,
        )
    except Exception as error:
        logger.debug("%s did not report what it wrote (%s)", NAME, error)
