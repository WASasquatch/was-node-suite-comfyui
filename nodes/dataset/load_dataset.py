"""Load a folder of pictures and the captions beside them as one dataset."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import DATASET
from ...modules.io import picker

logger = log.get_logger("nodes.dataset")

#: Sample rows the report carries.
SAMPLE_ROWS = 8

#: Characters of a caption a sample row shows.
SNIPPET_CHARS = 90


class LoadDataset(io.ComfyNode):
    """Read a folder's pictures and captions into a dataset."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLoadDataset",
            display_name="Load Dataset",
            search_aliases=[
                "WASLoadDataset", "Load Dataset", "training set", "captioned images",
                "image caption pairs", "lora dataset",
            ],
            category="WAS Suite/Dataset",
            description=(
                "Load a folder of training pictures and the caption saved beside each one as a "
                "dataset, read once and shared by every dataset node it feeds. Pictures are "
                "decoded only when a node needs their pixels."
            ),
            inputs=[
                picker.path_input(io.Combo.Input(
                    "folder",
                    options=picker.folders(),
                    tooltip=(
                        "The folder holding the pictures: 'input' is ComfyUI's input folder "
                        "itself; 'training/lara [input]' is that folder inside it."
                    ),
                ), folder=True),
                io.String.Input(
                    "pattern",
                    default="*",
                    tooltip=(
                        "Which pictures to load: `*` = every picture in the folder; `**/*` = in "
                        "subfolders too; `lara_*` = only names starting so."
                    ),
                ),
                io.String.Input(
                    "caption_extension",
                    default=".txt",
                    tooltip=(
                        "Extension of the caption beside each picture: `.txt` reads cat.txt for "
                        "cat.png. A picture with no such file is loaded uncaptioned."
                    ),
                ),
            ],
            outputs=[
                DATASET.Output(
                    display_name="dataset",
                    tooltip=(
                        "The pictures and their captions, for Dataset to Lists, Dataset Filter, "
                        "Dataset Health, Caption Statistics and Save Dataset."
                    ),
                ),
                io.Int.Output(display_name="count", tooltip="How many pictures were loaded."),
                io.Int.Output(display_name="captioned", tooltip="How many of them carry a caption."),
                io.String.Output(
                    display_name="folder",
                    tooltip=(
                        "The picked folder below its root with a trailing slash: "
                        "`training/tus [input]` = `training/tus/`; a root itself = empty. Wire it "
                        "into a save node's folder or filename_prefix to write back in place."
                    ),
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, folder="", pattern="*", caption_extension=".txt"):
        """Load again when a picture or caption under the folder is added, removed or edited."""
        from ...modules.dataset import dataset as datasets

        directory = picker.resolve_folder(folder)
        if directory is None or not directory.is_dir():
            return float("NaN")
        try:
            return datasets.fingerprint(directory, pattern, caption_extension)
        except ValueError:
            return float("NaN")

    @classmethod
    def execute(cls, folder="", pattern="*", caption_extension=".txt") -> io.NodeOutput:
        """Read the folder.

        Raises:
            PathNotAllowed: The folder or the pattern leaves the permitted read roots.
            ValueError: The folder is not there, or no picture matches the pattern.
        """
        from ...modules.dataset import dataset as datasets

        directory = picker.resolve_folder(folder)
        if directory is None or not directory.is_dir():
            raise ValueError(
                f"`{folder}` names no folder that is there. Pick another from the menu, or add "
                f"its folder to paths.allow_read in config.yaml"
            )
        loaded = datasets.load(directory, pattern, caption_extension, folder)
        if not loaded.items:
            raise ValueError(
                f"no picture in `{folder}` matches the pattern `{pattern}`, so there is no "
                f"dataset to hand on. Use `*` for every picture, or `**/*` to include subfolders"
            )
        captioned = sum(1 for item in loaded.items if item.captioned)
        logger.info("Load Dataset read %d picture(s), %d captioned, from `%s`", len(loaded), captioned, folder)
        cls._publish(loaded, folder, pattern, captioned)
        return io.NodeOutput(loaded, len(loaded), captioned, picker.folder_prefix(folder))

    @staticmethod
    def _publish(loaded, folder: str, pattern: str, captioned: int) -> None:
        """Report what was loaded, for the panel on this node. Never raises."""
        try:
            from ...modules.interface import run_result

            missing = len(loaded) - captioned
            rows = []
            for item in loaded.items[:SAMPLE_ROWS]:
                text = " ".join((item.caption or "(no caption)").split())
                if len(text) > SNIPPET_CHARS:
                    text = text[: SNIPPET_CHARS - 3] + "..."
                rows.append({"text": text, "note": item.name})
            run_result.publish(
                status=run_result.WARNING if missing else run_result.OK,
                summary=f"{len(loaded)} pictures from {folder}, {captioned} captioned",
                counts={"pictures": len(loaded), "captioned": captioned, "uncaptioned": missing},
                facts={"folder": folder, "pattern": pattern, "captions": loaded.caption_extension},
                items=rows,
                items_total=len(loaded),
            )
        except Exception as error:
            logger.debug("Load Dataset published no report (%s)", error)
