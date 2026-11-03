"""Score a training set from a folder or a dataset, and draw the result on the node."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat import limits
from ...modules.compat.types import DATASET, DICT, LIST
from ...modules.constants import ALLOWED_EXT
from ...modules.dataset import dataset as datasets
from ...modules.dataset import health
from ...modules.interface import progress, run_result
from ...modules.io import picker
from ...modules.util import file_listing, sandbox

logger = log.get_logger("nodes.dataset")

#: Display name, written into the messages this node raises.
NAME = "Dataset Health"

#: What :meth:`DatasetHealth.fingerprint_inputs` is handed for an input nothing is wired into.
UNWIRED = object()

#: What :meth:`DatasetHealth.fingerprint_inputs` answers for a wired dataset it was not handed.
WIRED = "dataset"

#: The report's flagged section when no picture was flagged.
NONE_FLAGGED = "no picture was flagged"

#: The report's near-duplicate section when no pair was found.
NO_PAIRS = "no near-duplicate pair was found"

#: The rows of the score table, as ``(score name, row name)``; exposure carries no score.
TABLE = (
    ("captions", "Captions"),
    ("consistency", "Consistency"),
    ("quality", "Quality"),
    (None, "Exposure"),
    ("uniqueness", "Uniqueness"),
    ("balance", "Balance"),
    ("faces", "Faces"),
)

#: Aspect ratios named in the balance figures, most common first.
MAX_RATIOS = 6

#: Frequent phrases named in the report's text.
MAX_NAMED = 5


class DatasetHealth(io.ComfyNode):
    """Measure a training set's pictures and captions and score its health."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASDatasetHealth",
            display_name="Dataset Health",
            search_aliases=[
                "WASDatasetHealth", "Dataset Health",
                "dataset check",
                "dataset statistics",
                "dataset quality",
                "training set",
                "lora dataset",
                "near duplicates",
                "duplicate images",
                "image quality",
                "face check",
                "caption coverage",
            ],
            category="WAS Suite/Dataset",
            description=(
                "Check a set of training pictures and their captions, from a folder or a "
                "dataset, before training: caption coverage and consistency, picture quality "
                "flags, near-duplicates, aspect balance and, on request, faces. Draws a score "
                "radar, the measurements and every flagged picture on the node, and hands on a "
                "report, the scores and the flagged and duplicate names."
            ),
            inputs=[
                picker.path_input(io.Combo.Input(
                    "folder",
                    options=picker.folders(),
                    tooltip=(
                        "Which folder of pictures to check. A bare 'input' is that folder "
                        "itself; 'lora/cats [input]' is that folder below it. Folders under "
                        "paths.allow_read in config.yaml are listed under their own names. "
                        "Ignored while dataset is wired."
                    ),
                ), folder=True),
                io.String.Input(
                    "pattern",
                    default="*",
                    multiline=False,
                    tooltip=(
                        "Which files to check. '*' = every picture in the folder; "
                        "'shot_*.png' = only those; '**/*' = subfolders too. Files that are "
                        "not pictures are skipped whatever the pattern says. Ignored while "
                        "dataset is wired."
                    ),
                ),
                io.String.Input(
                    "caption_extension",
                    default=".txt",
                    multiline=False,
                    tooltip=(
                        "The caption file beside each picture, under the same name: '.txt' = "
                        "cat.png reads cat.txt; '.caption' = cat.caption. A picture with no "
                        "such file counts as uncaptioned. Ignored while dataset is wired."
                    ),
                ),
                io.String.Input(
                    "subject",
                    default="[trigger]",
                    multiline=False,
                    tooltip=(
                        "What every caption should name: `[trigger]`, `ohwx woman`. Case "
                        "ignored, whole words. Captions holding it are counted; phrases "
                        "holding it leave the phrase list. Empty = no subject check."
                    ),
                ),
                io.Int.Input(
                    "target_side",
                    default=768,
                    min=0,
                    max=limits.max_resolution(),
                    step=8,
                    tooltip=(
                        "0 = no size check; 768 = a picture whose short side is under 768 "
                        "pixels is flagged small. Set it to the resolution the set trains at."
                    ),
                ),
                io.Boolean.Input(
                    "check_faces",
                    default=False,
                    tooltip=(
                        "`off` = no face check; `on` = count the faces in every picture, flag "
                        "no face, several faces or a face under 5% of the picture's height, and "
                        "add a faces score: the share with exactly one face. For a character set."
                    ),
                ),
                io.Int.Input(
                    "min_words",
                    default=5,
                    min=0,
                    max=1000,
                    tooltip=(
                        "0 = off; 5 = a caption under 5 words counts as short and lowers "
                        "consistency."
                    ),
                ),
                DATASET.Input(
                    "dataset",
                    optional=True,
                    tooltip=(
                        "Unwired = the folder is read; wired = a dataset, such as Load "
                        "Dataset's, measured as it stands, with captions and pictures edited "
                        "upstream and not yet saved. folder, pattern and caption_extension are "
                        "ignored while it is wired."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="report",
                    tooltip=(
                        "The figures as plain text: the score table, the caption and picture "
                        "measurements, the flag counts, every flagged picture with its reasons "
                        "and every near-duplicate pair."
                    ),
                ),
                DICT.Output(
                    display_name="scores",
                    tooltip=(
                        "Each score from 0 to 1, 1 = healthy: captions, consistency, quality, "
                        "uniqueness, balance, and faces while check_faces is on."
                    ),
                ),
                LIST.Output(
                    display_name="flagged",
                    tooltip=(
                        "Names of the pictures carrying any flag, most flags first: below the "
                        "folder, as 'set/cat_03.png', or the dataset's own names, as "
                        "'set/cat_03'. Empty when nothing is flagged."
                    ),
                ),
                LIST.Output(
                    display_name="duplicates",
                    tooltip=(
                        "One 'a.png | b.png' entry per near-duplicate pair, closest first: a "
                        "resized, re-saved or mirrored copy. Empty when there is none."
                    ),
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def fingerprint_inputs(
        cls,
        folder="",
        pattern="*",
        caption_extension=".txt",
        subject="[trigger]",
        target_side=768,
        check_faces=False,
        min_words=5,
        dataset=UNWIRED,
    ):
        """A digest of every picture and caption measured, so an edit measures again."""
        if dataset is not UNWIRED:
            if dataset is None:
                return WIRED
            try:
                items = datasets.from_value(dataset, NAME).items[: health.MAX_PICTURES]
            except ValueError:
                return float("NaN")
            return health.fingerprint_pictures(_pictures(items))
        try:
            directory = picker.resolve_folder(folder)
            if directory is None or not directory.is_dir():
                return float("NaN")
            suffix = health.caption_suffix(caption_extension)
            paths = health.scan(directory, pattern)[: health.MAX_PICTURES]
            return health.fingerprint(paths, suffix)
        except (sandbox.PathNotAllowed, ValueError, OSError) as error:
            logger.debug("%s measures again: %s", NAME, type(error).__name__)
            return float("NaN")

    @classmethod
    def execute(
        cls,
        folder="",
        pattern="*",
        caption_extension=".txt",
        subject="[trigger]",
        target_side=768,
        check_faces=False,
        min_words=5,
        dataset=None,
    ) -> io.NodeOutput:
        """Measure the dataset, or else the folder, and publish the figures to the node's panel.

        Returns:
            The report, the scores, the flagged names and the near-duplicate pairs.

        Raises:
            PathNotAllowed: The folder or the pattern leaves every permitted read root.
            ValueError: The dataset holds no picture; the folder is not there, holds no
                picture, or the caption extension is not one; or check_faces is on while
                features.yunet is off.
            FileNotFoundError: check_faces is on and the face detector's weights are missing.
        """
        if check_faces:
            from ...modules import config

            if not config.group_enabled("features.yunet"):
                raise ValueError(
                    "check_faces needs the face detector, and features.yunet is off in "
                    "config.yaml. Set it to true and restart ComfyUI, or turn check_faces off"
                )
        if dataset is not None:
            pictures, figures = _from_dataset(datasets.from_value(dataset, NAME))
        else:
            pictures, figures = _from_folder(folder, pattern, caption_extension)
        figures["min_words"] = int(min_words)

        result = health.measure(
            pictures,
            subject=subject,
            target_side=int(target_side),
            check_faces=bool(check_faces),
            min_words=int(min_words),
            step=_stepper(len(pictures) * (2 if check_faces else 1)),
        )
        flagged = _flagged_rows(result)
        _publish(result, flagged, figures, pictures)
        logger.info(
            "%s measured %d pictures of %s: %d flagged, %d near-duplicate pairs",
            NAME, result["pictures"], _where(figures), len(flagged), len(result["pairs"]),
        )
        return io.NodeOutput(
            _report(result, flagged, figures),
            dict(result["scores"]),
            [row["name"] for row in flagged],
            [f"{pair['a']} | {pair['b']}" for pair in result["pairs"]],
        )


def _label(folder) -> str:
    """The folder as its menu names it, never an absolute path."""
    try:
        return picker.folder_label(folder)
    except sandbox.PathNotAllowed:
        return sandbox._shown(folder)


def _from_folder(folder, pattern, caption_extension) -> tuple[list, dict]:
    """The pictures a folder holds and the figures describing where they came from.

    Args:
        folder: The folder widget's value.
        pattern: The glob pattern.
        caption_extension: The caption extension as typed.

    Returns:
        ``(pictures, figures)``: at most :data:`health.MAX_PICTURES` pictures, and the folder
        label, pattern, caption extension and match count.

    Raises:
        PathNotAllowed: The folder or the pattern leaves every permitted read root.
        ValueError: The folder is not there, holds no picture, or the extension is not one.
    """
    label = _label(folder)
    directory = picker.resolve_folder(folder)
    if directory is None or not directory.is_dir():
        raise ValueError(
            f"`{label}` names no folder that is there. Pick another from the menu, or add "
            f"its folder to paths.allow_read in config.yaml"
        )
    suffix = health.caption_suffix(caption_extension)
    found = health.scan(directory, pattern)
    if not found:
        raise ValueError(
            f"`{label}` holds no picture matching `{sandbox._shown(pattern)}`. Pick another "
            f"folder, or widen pattern: '*' reads the folder itself and '**/*' its "
            f"subfolders too"
        )
    paths = found[: health.MAX_PICTURES]
    if len(found) > len(paths):
        logger.warning(
            "%s measured the first %d of %d pictures in `%s`; narrow pattern to measure "
            "the rest", NAME, len(paths), len(found), label,
        )
    figures = {"label": label, "pattern": str(pattern), "suffix": suffix, "found": len(found)}
    return health.from_folder(paths, directory, suffix), figures


def _from_dataset(data) -> tuple[list, dict]:
    """The pictures a dataset holds and the figures describing where they came from.

    Args:
        data: The dataset.

    Returns:
        ``(pictures, figures)``: at most :data:`health.MAX_PICTURES` pictures, and the
        dataset's source and size.

    Raises:
        ValueError: The dataset holds no picture.
    """
    figures = {"label": data.source, "pattern": None, "suffix": None, "found": len(data.items)}
    if not data.items:
        raise ValueError(
            f"the dataset wired into {NAME} holds no picture, so there is nothing to measure. "
            f"Wire a dataset with pictures in it, or loosen whatever filtered it"
        )
    items = data.items[: health.MAX_PICTURES]
    if len(data.items) > len(items):
        logger.warning(
            "%s measured the first %d of %d pictures of %s; filter the dataset to measure "
            "the rest", NAME, len(items), len(data.items), _where(figures),
        )
    return _pictures(items), figures


def _pictures(items) -> list:
    """Dataset items as pictures to measure, replaced pixels standing in for their files.

    Args:
        items: :class:`datasets.Item` entries.

    Returns:
        One :class:`health.Picture` per item, in order.
    """
    pictures = []
    for item in items:
        if item.pixels is not None:
            pictures.append(health.Picture(item.name, "", item.caption, datasets.pixels(item)))
        else:
            pictures.append(health.Picture(item.name, item.image, item.caption))
    return pictures


def _where(figures: dict) -> str:
    """What a run measured, as its log line and report name it: the folder or the dataset."""
    if figures["pattern"] is not None:
        return f"`{figures['label']}`"
    return f"the dataset from `{figures['label']}`" if figures["label"] else "a dataset"


def _thumbnail_label(path: str) -> str:
    """The menu label a picture file's thumbnail is read by, or ``""`` where it has none."""
    if not path:
        return ""
    try:
        entry = file_listing.typed(path, ALLOWED_EXT, file_listing.ROOTS)
    except Exception as error:
        logger.debug("%s drew no thumbnail for a picture (%s)", NAME, type(error).__name__)
        return ""
    return entry.label if entry is not None else ""


def _thumbnails(result: dict, flagged: list, pictures: list) -> str:
    """Each picture the panel lists, its name and its thumbnail's label on one line split by a tab.

    Args:
        result: :func:`health.measure`'s answer.
        flagged: The flagged rows, most flags first.
        pictures: The pictures measured, in order.

    Returns:
        The lines, the first flagged picture and the first pair first, then the second of each,
        and so on, at most :data:`run_result.MAX_PAGE_LINES`. A picture with no label is left out.
    """
    paths = {picture.name: picture.path for picture in pictures}
    pairs = result["pairs"]
    order = []
    for index in range(max(len(flagged), len(pairs))):
        if index < len(flagged):
            order.append(flagged[index]["name"])
        if index < len(pairs):
            order += [pairs[index]["a"], pairs[index]["b"]]
    lines, seen = [], set()
    for name in order:
        if name in seen:
            continue
        seen.add(name)
        label = _thumbnail_label(paths.get(name, ""))
        if label:
            lines.append(f"{name}\t{label}")
            if len(lines) >= run_result.MAX_PAGE_LINES:
                break
    return "\n".join(lines)


def _stepper(total: int):
    """A callable advancing the node's progress bar by one picture and stopping on Cancel."""
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


def _flagged_rows(result: dict) -> list[dict]:
    """The pictures carrying any flag, most flags first, then by name."""
    rows = [row for row in result["rows"] if row["flags"]]
    return sorted(rows, key=lambda row: (-len(row["flags"]), row["name"].casefold()))


def _percent(share) -> str:
    """A 0 to 1 share written as a whole percentage."""
    return f"{round(float(share) * 100)}%"


def _plural(count: int, word: str) -> str:
    """A count and a noun, the noun taking an ``s`` unless the count is one."""
    return f"{count} {word}{'' if count == 1 else 's'}"


def _ratios(aspects: list) -> str:
    """The most common aspect ratios, width over height, each with how many pictures hold it."""
    if not aspects:
        return "none measured"
    named = ", ".join(f"{float(row['ratio']):.1f} x{row['count']}" for row in aspects[:MAX_RATIOS])
    more = len(aspects) - MAX_RATIOS
    return named + (f" and {more} more" if more > 0 else "")


def _measured(result: dict, figures: dict) -> dict:
    """What each row of the score table measured, and the figures behind it.

    Args:
        result: :func:`health.measure`'s answer.
        figures: The run's source label, pattern and caption extension (None for a dataset),
            word floor and match count.

    Returns:
        Row name to ``(value, detail)``.
    """
    total = result["pictures"]
    words = result["consistency"]
    counts = result["flag_counts"]
    subject = result["subject"]

    kind = f"a {figures['suffix']} caption" if figures["suffix"] else "a caption"
    captions = (
        f"{result['captioned']} of {total} carry {kind}"
        + (f", {result['coverage']} name {subject}" if subject else "")
    )
    short = (
        f", {words['short']} under {figures['min_words']} words" if figures["min_words"] else ""
    )
    consistency = (
        f"median caption, spread {_percent(words['words_spread'])}, "
        f"{words['identical']} identical{short}"
    )
    reasons = [
        f"{counts[flag]} {flag}" + (f" (short side under {result['target_side']})" if flag == "small" else "")
        for flag in health.FLAG_ORDER
        if counts.get(flag) and flag not in health.FACE_FLAGS and flag != "near-duplicate"
    ]
    quality = ", ".join(reasons) if reasons else "every picture is clear"
    exposure = (
        f"median brightness, spread {_percent(result['brightness_spread'])} from the 10th to "
        f"the 90th percentile, {counts.get('dark', 0)} dark, "
        f"{counts.get('blown out', 0)} blown out"
    )
    near = result["near_duplicates"]
    uniqueness = (
        f"{_plural(near, 'picture')} with a near-identical copy, in "
        f"{_plural(len(result['pairs']), 'pair')}"
        if near else "every picture is distinct"
    )
    aspects = result["aspects"]
    balance = (
        f"in the most common aspect ratio, of {_plural(len(aspects), 'ratio')}: {_ratios(aspects)}"
        if aspects else "no picture could be measured"
    )
    rows = {
        "Captions": (f"{result['captioned']}/{total}", captions),
        "Consistency": (f"{words['words_median']:g} words", consistency),
        "Quality": (f"{result['image_issues']} flagged", quality),
        "Exposure": (_percent(result["brightness_median"]), exposure),
        "Uniqueness": (str(near), uniqueness),
        "Balance": (_percent(result["aspect_top_share"]), balance),
    }
    faces = result["faces"]
    if faces is not None:
        rows["Faces"] = (
            f"{faces['one']}/{total}",
            f"{faces['one']} with one clear face, {faces['none']} with none, "
            f"{faces['several']} with several, {faces['small']} with a small face",
        )
    return rows


def _flag_line(row: dict) -> str:
    """One flagged picture as its name and its reasons."""
    return f"{row['name']}: {', '.join(row['flags'])}"


def _detail_line(row: dict) -> str:
    """One flagged picture as its name and what was measured of it."""
    if row["width"] is None:
        return f"{row['name']}: could not be read as a picture"
    parts = [f"{row['width']}x{row['height']}", f"brightness {_percent(row['brightness'])}"]
    if row["sharpness"] is not None:
        parts.append(f"sharpness {row['sharpness']:.2f} of the median")
    if row["jpeg_quality"] is not None:
        parts.append(f"JPEG quality {row['jpeg_quality']}")
    if row["faces"] is not None:
        parts.append(_plural(row["faces"], "face"))
        if row["faces"]:
            parts.append(f"main face {_percent(row['face_share'])} of the height")
    return f"{row['name']}: {', '.join(parts)}"


def _pair_line(pair: dict) -> str:
    """One near-duplicate pair as its two names and how close they are."""
    mirrored = ", mirrored" if pair["mirrored"] else ""
    return f"{pair['a']} | {pair['b']}: {_plural(pair['bits'], 'bit')} apart{mirrored}"


def _report(result: dict, flagged: list, figures: dict) -> str:
    """Every figure of a run as readable plain text.

    Args:
        result: :func:`health.measure`'s answer.
        flagged: The flagged rows, most flags first.
        figures: The run's source label, pattern and caption extension (None for a dataset),
            word floor and match count.

    Returns:
        The report, sections separated by blank lines.
    """
    total = result["pictures"]
    scores = result["scores"]
    measured = _measured(result, figures)
    if figures["pattern"] is None:
        head = f"{NAME}: {_plural(total, 'picture')} of {_where(figures)}"
    else:
        head = (
            f"{NAME}: {figures['label']}, {_plural(total, 'picture')} matching `{figures['pattern']}`"
        )
    if figures["found"] > total:
        head += f", the first {total} of {figures['found']}"

    width = max(len(name) for _key, name in TABLE)
    values = max(len(value) for value, _detail in measured.values())
    table = [f"{'':<{width}}  {'measured':<{values}}  score"]
    for key, name in TABLE:
        if name not in measured:
            continue
        value = measured[name][0]
        score = f"{round(scores[key] * 100):>5}" if key in scores else ""
        table.append(f"{name:<{width}}  {value:<{values}}  {score}".rstrip())

    words = result["consistency"]
    short = (
        f", {words['short']} under {figures['min_words']} words" if figures["min_words"] else ""
    )
    sizes = (
        f"{result['below_target']} under {result['target_side']}"
        if result["target_side"] else "no size check"
    )
    lines = [
        f"Captions: {measured['Captions'][1]}. Words: median {words['words_median']:g}, spread "
        f"{_percent(words['words_spread'])}, {words['identical']} identical{short}.",
        f"Pictures: short side median {result['short_side_median']}, {sizes}. Brightness: "
        f"median {_percent(result['brightness_median'])}, spread "
        f"{_percent(result['brightness_spread'])} from the 10th to the 90th percentile.",
        f"Aspect ratios: {_ratios(result['aspects'])}.",
    ]
    counts = result["flag_counts"]
    if counts:
        lines.append("Flags: " + ", ".join(f"{count} {flag}" for flag, count in counts.items()) + ".")

    sections = [head, "\n".join(table), "\n".join(lines)]
    sections.append(
        f"Flagged ({len(flagged)}):\n"
        + ("\n".join(_flag_line(row) for row in flagged) if flagged else NONE_FLAGGED)
    )
    pairs = result["pairs"]
    sections.append(
        f"Near-duplicates ({len(pairs)}):\n"
        + ("\n".join(_pair_line(pair) for pair in pairs) if pairs else NO_PAIRS)
    )
    phrases = result["phrases"]
    if phrases["frequent"]:
        named = ", ".join(
            f"{row['text']} ({_percent(row['share'])})" for row in phrases["frequent"][:MAX_NAMED]
        )
        sections.append(f"In more than half the captions: {named}")
    return "\n\n".join(sections)


def _summary(result: dict, flagged: list, figures: dict) -> str:
    """The one line the panel draws over the figures."""
    total = result["pictures"]
    pairs = len(result["pairs"])
    line = f"{_plural(total, 'picture')}: "
    line += f"{len(flagged)} flagged" if flagged else "none flagged"
    line += f", {_plural(pairs, 'near-duplicate pair')}" if pairs else ""
    if figures["found"] > total:
        line += f"; the first {total} of {figures['found']}"
    return line


def _publish(result: dict, flagged: list, figures: dict, pictures: list) -> None:
    """Report the scores, the measurements and the flagged pictures to the node's panel.

    Args:
        result: :func:`health.measure`'s answer.
        flagged: The flagged rows, most flags first.
        figures: The run's source label, pattern and caption extension (None for a dataset),
            word floor and match count.
        pictures: The pictures measured, in order.
    """
    try:
        if not run_result.watching():
            return
        scores = result["scores"]
        tallies = [
            {"name": name, "value": scores[name]} for name in health.SCORE_ORDER if name in scores
        ]
        facts = {
            name.lower(): f"{value}; {detail}"
            for name, (value, detail) in _measured(result, figures).items()
        }
        if figures["pattern"] is not None:
            facts["folder"] = figures["label"]
        else:
            facts["dataset"] = figures["label"] or "unnamed"
        rows = result["phrases"]["rows"]
        frequent = {row["text"] for row in result["phrases"]["frequent"]}
        bodies = [
            run_result.body(
                "flags", "\n".join(f"{count} {flag}" for flag, count in result["flag_counts"].items())
            ),
            run_result.body("flagged", "\n".join(_flag_line(row) for row in flagged)),
            run_result.body("details", "\n".join(_detail_line(row) for row in flagged)),
            run_result.body("near-duplicates", "\n".join(_pair_line(pair) for pair in result["pairs"])),
            run_result.body(
                "frequent phrases",
                "\n".join(f"{_percent(row['share'])} {row['text']}" for row in rows if row["text"] in frequent),
            ),
            run_result.body(
                "phrases",
                "\n".join(f"{_percent(row['share'])} {row['text']}" for row in rows if row["text"] not in frequent),
            ),
        ]
        if figures["pattern"] is None:
            bodies.append(run_result.body("thumbnails", _thumbnails(result, flagged, pictures)))
        low =[name for name, value in scores.items() if value < 0.5]
        run_result.publish(
            status=run_result.WARNING if low else run_result.OK,
            summary=_summary(result, flagged, figures),
            tallies=tallies,
            facts=facts,
            bodies=bodies,
        )
    except Exception as error:
        logger.debug("%s did not report its figures (%s)", NAME, error)
