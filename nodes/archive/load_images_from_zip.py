"""Read the images a zip archive holds into one batch, every one at the same size."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.archive import container
from ...modules.io import picker
from ...modules import log
from ...modules.archive import kinds, picks
from ...modules.compat import limits
from ...modules.compat.types import LIST, ZIP
from ...modules.convert.tensors import pil2tensor
from ...modules.image import sizing
from ...modules.image.draw import parse_color

logger = log.get_logger("nodes.archive")

#: What the archive menu lists.
ARCHIVE_EXTENSIONS = (container.SUFFIX,)

#: Entries the archive menu offers, and what it says when there are none.
NO_ARCHIVES = "no .zip files found"


def archive_options() -> list[str]:
    """The menu's entries, or a line saying there are none."""
    return picker.labels(ARCHIVE_EXTENSIONS) or [NO_ARCHIVES]


def archive_path(file: str) -> str:
    """The archive one menu entry names, as a path, or an empty string."""
    entry = str(file or "").strip()
    if not entry or entry == NO_ARCHIVES:
        return ""
    return picker.resolve(entry, ARCHIVE_EXTENSIONS) or ""



#: Why an entry that unpacked correctly still did not reach the batch. Read off
#: ``modules.archive.picks`` at import, so a build where those reasons have moved fails here
#: rather than in the middle of a run.
NOT_AN_IMAGE = picks.NOT_AN_IMAGE
TOO_MANY_PIXELS = picks.TOO_MANY_PIXELS

#: The pad colour used when the widget holds something that is not a colour: opaque black,
#: which is what the widget's own default spells.
FALLBACK_PAD = (0, 0, 0, 255)


class LoadImagesFromZip(io.ComfyNode):
    """Read every image in one archive that a pattern picks, as a single batch in name order."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLoadImagesFromZIP",
            display_name="Load Images from ZIP",
            search_aliases=[
                "WASLoadImagesFromZIP", "Load Images from ZIP", "Load Image ZIP",
                "zip",
                "archive",
                "unzip",
                "image batch from zip",
                "dataset archive",
                "resize batch",
            ],
            category="WAS Suite/Archive",
            description=(
                "Read the images inside a zip archive as one batch. A batch is a single "
                "tensor, so every image has to reach the same size: pick how they get there "
                "with resize_mode, width and height, and whether the batch carries "
                "transparency with channels. The file names come out alongside the pictures, "
                "in the same order."
            ),
            inputs=[
                io.Combo.Input(
                    "file",
                    options=archive_options(),
                    tooltip=(
                        "Which archive to read. The menu lists every .zip in ComfyUI's "
                        "input, output and temp folders and in any folder added under "
                        "paths.allow_read, each tagged with where it sits. Ignored while the "
                        "zip socket is connected."
                    ),
                ),
                ZIP.Input(
                    "zip",
                    optional=True,
                    tooltip=(
                        "The archive to read, from Open ZIP. Connected, it is used and "
                        "the menu is ignored, so the archive is opened and indexed once "
                        "however many nodes read it."
                    ),
                ),
                io.String.Input(
                    "pattern",
                    default="*",
                    multiline=False,
                    tooltip=(
                        "Which entries to read; STRING. No '/' matches the file name at any depth; "
                        "a '/' anchors at the archive root. Case is ignored, and non-images are "
                        "always skipped. Eg: *, *.png, frames/**/*.png"
                    ),
                ),
                io.Combo.Input(
                    "resize_mode",
                    options=list(sizing.MODES),
                    default=sizing.FIT_AND_PAD,
                    tooltip=(
                        "How mixed sizes reach width by height, since one batch holds one size. "
                        "`fit and pad`: whole image, pad_color bars. `fill and crop`: fills the "
                        "frame, ends cut. `stretch`: distorts. `crop or pad`: no resampling, "
                        "original pixels kept."
                    ),
                ),
                io.Int.Input(
                    "width",
                    default=512,
                    min=1,
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "How wide every image in the batch comes out, in pixels. Every mode "
                        "delivers exactly this width, so the batch is this wide whatever the "
                        "archive held. A multiple of 8 suits a sampler; 512 or 1024 matches "
                        "most models."
                    ),
                ),
                io.Int.Input(
                    "height",
                    default=512,
                    min=1,
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "How tall every image in the batch comes out, in pixels. Together with "
                        "width this is also what decides how many images fit in one batch: the "
                        "node loads as many as 32 megapixels holds, which is 128 at 512 by 512 "
                        "and 32 at 1024 by 1024, and says in the log when there are more."
                    ),
                ),
                io.Combo.Input(
                    "interpolation",
                    options=list(sizing.FILTER_NAMES),
                    default=sizing.DEFAULT_FILTER,
                    tooltip=(
                        "Scaling filter. `lanczos` is sharpest and slowest; `bicubic` and "
                        "`bilinear` are softer and quicker; `nearest` invents no colour, for pixel "
                        "art and label maps. Ignored in `crop or pad`."
                    ),
                ),
                io.Combo.Input(
                    "align",
                    options=list(sizing.ALIGNMENT_NAMES),
                    default=sizing.DEFAULT_ALIGNMENT,
                    tooltip=(
                        "Which part survives a crop, and which side takes the wider pad bar. `top "
                        "center` suits portraits, where a centred crop takes the forehead off. "
                        "Ignored in `stretch`."
                    ),
                ),
                io.String.Input(
                    "pad_color",
                    default="#000000",
                    multiline=False,
                    tooltip=(
                        "Fill for space the image does not cover; STRING. Any Pillow colour: "
                        "`#RRGGBB`, a name, or `#RRGGBBAA`. Empty is transparent, which only shows "
                        "while channels is RGBA. Eg: white"
                    ),
                ),
                io.Combo.Input(
                    "channels",
                    options=list(sizing.CHANNELS),
                    default="RGB",
                    tooltip=(
                        "Channels the batch carries. `RGB` is what samplers and upscalers expect; "
                        "transparency is dropped, and a transparent pixel that was scaled comes out "
                        "black. `RGBA` keeps alpha, for compositing and Image Select Channel."
                    ),
                ),
                io.Int.Input(
                    "start",
                    default=0,
                    min=0,
                    max=picks.MAX_FILES,
                    step=1,
                    tooltip=(
                        "Which matching image the batch starts at; INT, counting from 0 in sorted "
                        "name order. Leave limit at 0 to read a large archive a page at a time: the "
                        "log names the next page's index."
                    ),
                ),
                io.Int.Input(
                    "limit",
                    default=0,
                    min=0,
                    max=picks.MAX_FILES,
                    step=1,
                    tooltip=(
                        "How many images to load from start; INT. 0 loads as many as one "
                        "batch holds at the chosen size. A number above that is reduced, and the "
                        "log says so."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip=(
                        "Every image that was read, as one batch, all at width by height and "
                        "all with the same channel count. In the order the names sort, so two "
                        "runs of the same archive produce the same batch."
                    ),
                ),
                LIST.Output(
                    display_name="names",
                    tooltip=(
                        "The name each image came from, on one wire and in batch order, such "
                        "as 'frames/cat.png'. The folders inside the archive are kept, so two "
                        "files called cat.png in different folders stay apart. Read one out "
                        "with Text List Get, using the same index as the image."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip=(
                        "How many images are in the batch, which is the length of the names "
                        "list. Never 0: an archive that yields no image stops the prompt "
                        "instead, because an image batch cannot be empty."
                    ),
                ),
                io.Int.Output(
                    display_name="skipped",
                    tooltip=(
                        "How many entries did not reach the batch: not an image, an unsafe name, a "
                        "symlink, encrypted, a repeated name, damaged, or holding something other "
                        "than its extension says. The log names each one. Images left out by "
                        "start or limit are not counted."
                    ),
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls, file="", pattern="*", zip=None, resize_mode=sizing.FIT_AND_PAD, width=512, height=512,
        interpolation=sizing.DEFAULT_FILTER, align=sizing.DEFAULT_ALIGNMENT,
        pad_color="#000000", channels="RGB", start=0, limit=0,
    ):
        """Re-read when the archive on disk, or the selection, has changed."""
        return picks.fingerprint(zip if zip is not None else archive_path(file), pattern, start, limit)

    @classmethod
    def execute(
        cls, file="", pattern="*", zip=None, resize_mode=sizing.FIT_AND_PAD, width=512, height=512,
        interpolation=sizing.DEFAULT_FILTER, align=sizing.DEFAULT_ALIGNMENT,
        pad_color="#000000", channels="RGB", start=0, limit=0,
    ) -> io.NodeOutput:
        """Read the archive, decode every image the pattern picks and size them alike.

        Raises:
            NotAnArchive: ``file`` is empty, names nothing that is there, or names a file
                that is not a readable zip.
            PathNotAllowed: It resolved outside every permitted read root.
            ValueError: No entry produced an image.
            MemoryError: Neither free memory nor any scratch drive has room for the batch.
        """
        from ...modules.compat.lists import require_values
        from ...modules.image import scratch

        archive = picks.opened_archive(zip if zip is not None else archive_path(file))
        report = picks.Report()
        first, wanted = cls.window(start, limit, width, height, report)
        members, report = picks.read_matching(
            archive, pattern, kinds.IMAGE, report, first, wanted
        )
        cls.paging(report, first)
        pad = parse_color(pad_color, FALLBACK_PAD)

        advice = "Lowering limit, width or height also fits it."
        batch = scratch.allocate(
            (len(members), max(1, int(height)), max(1, int(width)), 4 if channels == "RGBA" else 3),
            node="Load Images from ZIP", advice=advice,
        )
        names: list[str] = []
        for member in members:
            try:
                source = sizing.open_bytes(member.data)
            except sizing.ImageTooLarge as error:
                report.skip(TOO_MANY_PIXELS, f"the entry {member.name!r} {error}")
                continue
            except sizing.NotAnImage as error:
                report.skip(NOT_AN_IMAGE, f"the entry {member.name!r} {error}")
                continue
            sized = sizing.fit(source, width, height, resize_mode, interpolation, align, pad)
            batch[len(names)].copy_(pil2tensor(sizing.as_channels(sized, channels))[0])
            names.append(member.name)

        for note in report.notes:
            logger.warning("%s: %s", archive.label, note)
        logger.info(
            "Load Images from ZIP read %s at %dx%d, %s: %s",
            archive.label, width, height, resize_mode, report.summary(len(names)),
        )
        require_values(names, cls.nothing(archive, pattern, report, start))
        if len(names) < len(members):
            batch = scratch.join([batch[:len(names)]], node="Load Images from ZIP", advice=advice)
        return io.NodeOutput(batch, names, len(names), report.total)

    @staticmethod
    def window(start, limit, width, height, report) -> tuple[int, int]:
        """Which matching images the batch asks for, before any of them are unpacked.

        Args:
            start: Which of them the batch starts at.
            limit: How many to take, or 0 for as many as one batch holds.
            width: Target width, which with ``height`` decides how many fit.
            height: Target height.
            report: Filled in where ``limit`` asked for more than one batch holds.

        Returns:
            ``(first, count)`` for :func:`modules.archive.picks.read_matching`, so only the
            images this batch wants are unpacked and the byte total is spent on them.
        """
        allowed = sizing.batch_limit(width, height)
        first = max(0, int(start))
        asked = int(limit)
        if asked > allowed:
            report.bound(
                f"limit asked for {asked} image(s) and {allowed} of {width} by {height} fit in "
                f"one batch, so {allowed} were loaded"
            )
        return first, allowed if asked <= 0 else min(asked, allowed)

    @staticmethod
    def paging(report, start: int) -> None:
        """Record where this batch sits among the matching images, and where the next starts.

        Args:
            report: The filled-in read report, whose ``chosen`` counts every matching image
                and whose ``reached`` says how many of them this batch got through.
            start: The index the batch started at.
        """
        if start:
            report.bound(
                f"start skipped the first {min(start, report.chosen)} of "
                f"{report.chosen} matching image(s)"
            )
        left = report.chosen - start - report.reached
        if left > 0 and report.reached:
            report.bound(
                f"{left} matching image(s) were left; set start to "
                f"{start + report.reached} to read the next batch"
            )

    @staticmethod
    def nothing(archive, pattern: str, report, start: int) -> str:
        """The message for a read that produced no image at all.

        Args:
            archive: The archive that was read.
            pattern: The pattern as the user wrote it.
            report: What the read left out.
            start: The index the batch was to start at.

        Returns:
            A message naming the archive, then whichever of four reasons applies: it holds no
            files, the pattern picked no image, start is past the last one, or every
            image picked was skipped.
        """
        offered = ", ".join(report.examples) or "nothing"
        matched = report.chosen
        if not report.examined:
            return (
                f"{archive.label} holds no files, so Load Images from ZIP has no image to "
                f"hand on and a batch cannot be empty. It opened correctly and is simply "
                f"empty."
            )
        if not report.matched:
            return (
                f"no entry in {archive.label} is picked by the pattern `{pattern}`, so Load "
                f"Images from ZIP has no image to hand on. It holds {offered}. A pattern with "
                f"no '/' in it is matched against the file's own name at any depth, so '*' or "
                f"'*.png' reads them wherever they sit. The extensions this node reads are "
                f"{kinds.extension_list(kinds.IMAGE)}."
            )
        if matched and int(start) >= matched:
            return (
                f"start is {int(start)} and only {matched} image(s) in "
                f"{archive.label} are picked by the pattern `{pattern}`, counting from 0, so "
                f"the batch would be empty. Set it to {matched - 1} or less."
            )
        return (
            f"every image the pattern `{pattern}` picked in {archive.label} was skipped, so "
            f"Load Images from ZIP has no image to hand on: {report.summary(0)}. The log names "
            f"each one. The extensions this node reads are "
            f"{kinds.extension_list(kinds.IMAGE)}."
        )
