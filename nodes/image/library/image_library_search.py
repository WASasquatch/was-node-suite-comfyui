"""Find the pictures in a library folder that best match a description or a whole prompt."""

from __future__ import annotations

import time

from comfy_api.latest import io

from ....modules import log
from ....modules.constants import ALLOWED_EXT
from ....modules.image import image_library
from ....modules.interface import run_result
from ....modules.io import picker

#: Config group this module loads under.
REQUIRES = "preprocessors"

#: The node's name in messages.
NODE = "Image Library Search"

#: Most matches one search answers.
MAX_TOP_K = 64

logger = log.get_logger("nodes.image_library_search")


def _label(path: str) -> str:
    """The menu label a picture's thumbnail is read by, or ``""`` where it has none."""
    from ....modules.util import file_listing

    try:
        entry = file_listing.typed(path, ALLOWED_EXT, file_listing.ROOTS)
    except Exception as error:
        logger.debug("%s drew no thumbnail for a match (%s)", NODE, type(error).__name__)
        return ""
    return entry.label if entry is not None else ""


class ImageLibrarySearch(io.ComfyNode):
    """The pictures of a library folder closest to a prompt, best first."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageLibrarySearch",
            display_name="Image Library Search",
            search_aliases=[
                "WASImageLibrarySearch",
                "Image Library Search",
                "pose search",
                "pose library",
                "find pose",
                "reference search",
                "semantic image search",
            ],
            category="WAS Suite/Image/Library",
            description=(
                "Find the pictures in a library folder, such as a pose set, that best match a "
                "description or a whole prompt, ready for a ControlNet preprocessor. The panel "
                "browses the folders, shows the matches as the prompt changes, and pins one; "
                "pause holds the run until a match is picked."
            ),
            inputs=[
                io.String.Input(
                    "prompt", multiline=True, default="",
                    tooltip=(
                        "What to find, such as `sitting, leaning back on her hands, one knee up`, or a "
                        "whole prompt. View words such as `side view` or `from below` choose the angle."
                    ),
                ),
                picker.path_input(io.Combo.Input(
                    "folder", options=picker.folders(),
                    tooltip="The library folder, such as `poses [input]` or a folder of finished pictures in output.",
                ), folder=True),
                io.String.Input(
                    "folders", default="", advanced=True,
                    tooltip=(
                        "Folders below `folder` searched, one per line, `.` for its own pictures. "
                        "Empty = every folder. The panel's folder tree ticks them."
                    ),
                ),
                io.Combo.Input(
                    "view", options=list(image_library.VIEW_CHOICES), default="auto",
                    tooltip=(
                        "Camera view of the picture sent on: `auto` = the prompt's view words; `any` = "
                        "the closest; `front`, `side`, `diag`, `high`, `low`, `back` = that view."
                    ),
                ),
                io.String.Input(
                    "filter", default="",
                    tooltip=(
                        "Words a match must carry in its caption or path, and -words it must not: "
                        "`front -couples` keeps front views of solo poses. Empty = every picture."
                    ),
                ),
                io.Int.Input(
                    "top_k", default=8, min=1, max=MAX_TOP_K,
                    tooltip="How many matches the list outputs carry, best first: `8` = the eight closest.",
                ),
                io.Boolean.Input(
                    "one_per_group", default=True,
                    tooltip=(
                        "`true` = only the closest picture of each group, such as one angle of each pose, "
                        "so the list holds different poses; `false` = every angle."
                    ),
                ),
                io.Boolean.Input(
                    "pause", default=False,
                    tooltip=(
                        "`true` = hold the run with the matches on the panel until one is clicked and "
                        "Continue is pressed; `false` = send the pinned picture, or the closest, at once."
                    ),
                ),
                io.String.Input(
                    "pinned", default="", advanced=True,
                    tooltip=(
                        "The picture sent to `image`, as its path below `folder`, such as "
                        "`img/full_single_P061_side.png`. Empty = the closest match. Clicking a match pins it."
                    ),
                ),
                picker.path_input(io.Combo.Input(
                    "text_model", options=image_library.text_choices(),
                    default=image_library.DEFAULT_TEXT_MODEL,
                    tooltip=(
                        "Reads captions and prompts. embeddinggemma-2 = 1.5 GB, 100+ languages, the same "
                        "model as the picture model; bge-small-en = 133 MB, English; keywords only = no model."
                    ),
                ), folder=True, model=True),
                picker.path_input(io.Combo.Input(
                    "picture_model", options=image_library.picture_choices(),
                    default=image_library.DEFAULT_PICTURE_MODEL,
                    tooltip=(
                        "Reads the pictures themselves, so `hair` finds photos captioned only by file name. "
                        "embeddinggemma-2 = also the text model, loaded once; SigLIP 2 = quicker; none = captions alone."
                    ),
                ), folder=True, model=True),
                io.Float.Input(
                    "picture_weight", default=0.35, min=0.0, max=1.0, step=0.05, advanced=True,
                    tooltip=(
                        "Share of the ranking the picture model decides for captioned pictures: 0 = captions "
                        "alone; 1 = pictures alone. A picture named only by its file is ranked 0.9 by itself."
                    ),
                ),
                io.Combo.Input(
                    "captions", options=list(image_library.CAPTION_SOURCES), default="auto", advanced=True,
                    tooltip=(
                        "`auto` = a JSON manifest where a folder has one, else each picture's .txt file, "
                        "saved prompt or file name; `manifest`, `sidecar text`, `picture prompt`, "
                        "`file name` = that source alone."
                    ),
                ),
                io.Boolean.Input(
                    "recursive", default=True, advanced=True,
                    tooltip=(
                        "`true` = read every folder below `folder`, each through its own manifest where it "
                        "has one, each keeping its own index; `false` = `folder` alone."
                    ),
                ),
                io.Int.Input(
                    "max_files", default=20000, min=1, max=image_library.MAX_FILES, advanced=True,
                    tooltip="Most pictures read: 20000 = the usual; the rest are left out and counted on the panel.",
                ),
                io.Int.Input(
                    "hold_timeout", default=600, min=0, max=86400, advanced=True,
                    tooltip=(
                        "Seconds `pause` holds the run: 600 = ten minutes; 0 = no limit. A hold that runs "
                        "out sends the pinned picture, or the closest."
                    ),
                ),
                io.Int.Input(
                    "dimensions", default=0, min=0, max=4096, advanced=True,
                    tooltip=(
                        "Caption vector length compared: 0 = the model's own; 256 = shorter and quicker, "
                        "for models trained for it, such as EmbeddingGemma at 768, 512, 256 or 128."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="image",
                    tooltip="The pinned or picked match at its own size, for a pose, depth or line preprocessor.",
                ),
                io.String.Output(display_name="caption", tooltip="That match's caption."),
                io.Image.Output(
                    display_name="images", is_output_list=True,
                    tooltip="Every match as a list, best first, each picture at its own size.",
                ),
                io.String.Output(
                    display_name="captions", is_output_list=True,
                    tooltip="Each match's caption, in the same order as images.",
                ),
                io.String.Output(
                    display_name="names", is_output_list=True,
                    tooltip="Each match's file name without its extension, in the same order.",
                ),
                io.Float.Output(
                    display_name="scores", is_output_list=True,
                    tooltip=(
                        "Each match's score, higher is closer: about 0 to 1 for captions alone, or a "
                        "blend centred on 0 when picture_weight mixes in the picture model."
                    ),
                ),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def fingerprint_inputs(cls, prompt, folder, folders, view, filter, top_k, one_per_group, pause, pinned,
                           text_model, picture_model, picture_weight, captions, recursive, max_files,
                           hold_timeout, dimensions):
        """Changes when a picture, caption or manifest in the folder changes, and on every paused run."""
        if pause:
            return float("nan")
        try:
            root = picker.resolve_folder(folder)
            if root is None:
                return float("nan")
            return image_library.stamp(root, bool(recursive))
        except Exception as error:
            logger.debug("%s could not fingerprint its folder (%s)", NODE, type(error).__name__)
            return float("nan")

    @classmethod
    def execute(cls, prompt, folder, folders, view, filter, top_k, one_per_group, pause, pinned,
                text_model, picture_model, picture_weight, captions, recursive, max_files,
                hold_timeout, dimensions) -> io.NodeOutput:
        """Read and embed the folder where it changed, rank it, and send the pinned or best match.

        Args:
            prompt: A description, or a whole prompt.
            folder: A menu label, or a path wired in as text.
            folders: Folders below ``folder`` searched, one per line; empty for all.
            view: ``auto``, ``any`` or one camera view.
            filter: Words a match must carry, and ``-words`` it must not.
            top_k: Most matches.
            one_per_group: Keep only the closest picture of each group.
            pause: Hold the run until a match is picked on the panel.
            pinned: The key of the picture to send, empty for the closest.
            text_model: A text model menu entry, ``keywords only`` or a wired folder path.
            picture_model: A picture model menu entry, ``none`` or a wired folder path.
            picture_weight: Share of the ranking the picture model decides.
            captions: One of ``CAPTION_SOURCES``.
            recursive: Read the folders below ``folder``.
            max_files: Most pictures read.
            hold_timeout: Seconds ``pause`` holds the run, 0 for no limit.
            dimensions: Caption vector length compared, 0 for the model's own.

        Returns:
            The chosen picture and caption, then the matches as lists.

        Raises:
            PathNotAllowed: A wired path lies outside the readable folders.
            ValueError: The folder is missing or holds no pictures, the prompt is empty, or the
                filter keeps nothing.
        """
        from ....modules.interface import image_library as live
        from ....modules.interface import pause as hold

        if captions not in image_library.CAPTION_SOURCES:
            raise ValueError(f"{NODE} does not know the caption source `{str(captions)[:40]}`. Pick one from the menu.")
        if view not in image_library.VIEW_CHOICES:
            raise ValueError(f"{NODE} does not know the view `{str(view)[:40]}`. Pick one from the menu.")
        root = picker.resolve_folder(folder)
        label = picker.folder_label(folder)
        if root is None or not root.is_dir():
            raise ValueError(f"{NODE} cannot find the folder `{label[:120]}`. Pick one from the menu.")
        started = time.monotonic()
        found = image_library.read(root, captions, bool(recursive), int(max_files))
        chosen_folders = image_library.select(found, folders)
        if not chosen_folders.entries:
            raise ValueError(
                f"{NODE} found no pictures in `{label[:120]}`" + (" in the folders ticked" if str(folders).strip() else "")
                + ". Pick a folder holding PNG, JPEG or WebP pictures, or tick another folder on the panel."
            )
        text = image_library.text_model(text_model, NODE)
        picture = image_library.picture_model(picture_model, NODE)
        library = image_library.build(label, chosen_folders, text, picture)
        live.keep(
            live.settings_key(root, captions, bool(recursive), int(max_files), text.label,
                              picture.label if picture else image_library.NO_PICTURE_MODEL),
            found, str(folders or ""), library,
        )
        matches, kept = image_library.search(
            library, prompt, int(top_k), float(picture_weight), filter, bool(one_per_group), view,
            int(dimensions),
        )
        if not matches:
            raise ValueError(
                f"{NODE}: the filter `{str(filter)[:80]}` leaves none of the library's "
                f"{len(library.entries):,} pictures. Remove a word from filter, or clear it."
            )
        chosen = matches[0].entry
        pinned_entry = library.find(str(pinned or "").strip()) if str(pinned or "").strip() else None
        if str(pinned or "").strip() and pinned_entry is None:
            logger.warning("%s: the pinned picture is not in the folders searched, so the closest match is sent", NODE)
        chosen = pinned_entry or chosen
        figures = library.figures
        seconds = time.monotonic() - started
        made = figures["embedded captions"] + figures["embedded pictures"]
        logger.info(
            "%s ranked %d pictures from %s in %.1fs (%d vectors made)",
            NODE, len(library.entries), label, seconds, made,
        )
        lines = []
        for rank, match in enumerate(matches):
            parts = [f"{match.score:.3f}"]
            if match.text_score is not None and match.picture_score is not None:
                parts.append(f"caption {match.text_score:.3f}, picture {match.picture_score:.3f}")
            lines.append("\t".join([
                str(rank + 1), _label(match.entry.preview) or _label(match.entry.path), " / ".join(parts),
                match.entry.name, match.entry.caption, match.entry.key,
            ]))

        def report(entry, status=run_result.OK, note=""):
            run_result.publish(
                status=status,
                summary=f"{entry.name}{note}",
                counts={
                    "matches": len(matches), "searched": kept, "folders": len(chosen_folders.units),
                    "embedded": made, "seconds": round(seconds, 2),
                },
                facts={
                    "library": label,
                    "captions": found.source,
                    "ranked by": text.label + (f" and {picture.label}" if picture and float(picture_weight) > 0 else ""),
                    "sent": entry.key,
                    "caption": entry.caption,
                },
                bodies=[run_result.body("matches", "\n".join(lines))],
            )

        if pause:
            report(chosen, note=": pick a match, then Continue")
            outcome, value = hold.wait_for_resume(
                str(cls.hidden.unique_id), timeout=float(hold_timeout),
                message="Pick a match, then Continue",
            )
            picked = library.find(value.strip()) if outcome == hold.RESUMED and value.strip() else None
            if outcome != hold.RESUMED:
                logger.warning("%s %s before a match was picked, so %s was sent", NODE, outcome, chosen.name)
            chosen = picked or chosen
        report(chosen)
        images = [image_library.load_picture(match.entry.path) for match in matches]
        position = next((at for at, match in enumerate(matches) if match.entry.key == chosen.key), None)
        image = images[position] if position is not None else image_library.load_picture(chosen.path)
        return io.NodeOutput(
            image,
            chosen.caption,
            images,
            [match.entry.caption for match in matches],
            [match.entry.name for match in matches],
            [match.score for match in matches],
        )
