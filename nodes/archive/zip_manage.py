"""Choose which of an archive's entries carry on, and answer the archive holding them."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.archive import container, draft, save
from ...modules.archive.selection import parse as parse_lines
from ...modules.compat.types import LIST, ZIP

logger = log.get_logger("nodes.archive")

#: Keep the named entries and drop the rest.
KEEP = "keep the chosen"

#: Drop the named entries and keep the rest.
REMOVE = "remove the chosen"

#: What the ``action`` widget offers, keep first.
ACTIONS = (KEEP, REMOVE)

#: Hold the run and tick the entries on the node.
PICK = "pick on the node"

#: Read the entries box, without holding the run.
TYPED = "the entries box"

#: What the ``selection`` widget offers, picking first.
SELECTIONS = (PICK, TYPED)

#: What the browser is told it is editing while the run is held.
HOLD_KIND = "zip_entries"


class ZipManage(io.ComfyNode):
    """Filter an archive's entries, answering an archive holding only what was chosen."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASZipManage",
            display_name="ZIP Manage",
            search_aliases=[
                'WASZipManage',
                "ZIP Manage",
                "zip manage",
                "zip filter",
                "remove from zip",
                "zip entries",
                "prune archive",
            ],
            category="WAS Suite/Archive",
            description=(
                "Choose which of an archive's entries carry on, and answer a new "
                "archive holding them. selection chooses between ticking them on the "
                "node and naming them in entries."
            ),
            inputs=[
                ZIP.Input(
                    "zip",
                    tooltip=(
                        "The archive to filter, from Open ZIP, ZIP Add or another ZIP "
                        "Manage."
                    ),
                ),
                io.String.Input(
                    "entries",
                    default="",
                    multiline=True,
                    tooltip=(
                        "Entry names, one per line, folders included, read only when selection "
                        "is 'the entries box'. 'keep the chosen' keeps these; 'remove the "
                        "chosen' drops them. A '#' line is a comment. Eg: chapters/one.html"
                    ),
                ),
                io.Combo.Input(
                    "selection",
                    options=list(SELECTIONS),
                    default=PICK,
                    tooltip=(
                        "'the entries box' = reads the names typed in entries, never holds the "
                        "run; 'pick on the node' = holds the run and keeps the entries ticked "
                        "on the node, or reads the entries box when no browser is connected."
                    ),
                ),
                io.Combo.Input(
                    "action",
                    options=list(ACTIONS),
                    tooltip=(
                        "'keep the chosen' answers an archive of the named entries alone. "
                        "'remove the chosen' answers everything except them."
                    ),
                ),
                io.Int.Input(
                    "hold_timeout",
                    default=600,
                    min=0,
                    max=86400,
                    step=1,
                    tooltip=(
                        "Seconds to hold the run while waiting for the ticks, so 600 gives "
                        "ten minutes and 0 waits with no limit. A hold that runs out keeps "
                        "every entry and says so."
                    ),
                ),
                io.Combo.Input(
                    "compression",
                    options=list(save.COMPRESSIONS),
                    tooltip=(
                        "'deflate' shrinks entries that compress; 'store' writes them as "
                        "they are, which suits pictures that are compressed already."
                    ),
                ),
            ],
            hidden=[io.Hidden.unique_id],
            outputs=[
                ZIP.Output(
                    display_name="zip",
                    tooltip=(
                        "The archive holding the chosen entries. Send it to Save ZIP, to a "
                        "loader, or to another ZIP Manage."
                    ),
                ),
                LIST.Output(
                    display_name="names",
                    tooltip="The entry names the answered archive holds, in its own order.",
                ),
                io.Int.Output(
                    display_name="entry_count",
                    tooltip="How many entries the answered archive holds.",
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls, zip=None, entries="", selection=PICK, action=KEEP, hold_timeout=600,
        compression="deflate",
    ):
        """Whether the node has to run again.

        Returns:
            ``NaN`` while picking, and the widget values otherwise.
        """
        from ...modules.interface import channel

        if selection == PICK and channel.watching():
            return float("NaN")
        return "|".join([str(entries), str(action), str(compression)])

    @classmethod
    def execute(
        cls, zip=None, entries="", selection=PICK, action=KEEP, hold_timeout=600,
        compression="deflate",
    ) -> io.NodeOutput:
        """Filter the archive and answer the one holding what was chosen.

        Raises:
            NotAnArchive: ``zip`` carries no archive.
        """
        from ...modules.interface import channel

        archive = container.require_archive(zip, "zip")
        held = [entry.name for entry in archive.files]
        picking = selection == PICK
        if picking and not channel.watching():
            logger.info(
                "no browser is connected, so ZIP Manage read the entries box rather than "
                "holding the queue for ticks that nobody could make"
            )
            picking = False
        if picking:
            keeping = cls.ticked(held, int(hold_timeout))
            built = draft.kept(archive, keeping, compression)
            names = [entry.name for entry in built.files]
            logger.info("ZIP Manage kept %d of %d entry(s)", len(names), len(held))
            return io.NodeOutput(built, names, len(names))

        chosen, repeats = parse_lines(entries)
        if repeats:
            logger.info("%d repeated entry name(s) counted once", repeats)

        # An empty box names nothing, and dropping nothing is the archive as it stands while
        # keeping nothing is an empty archive. A hold that runs out keeps everything, so an
        # empty box does too rather than emptying the archive on the way past.
        if not chosen and action == KEEP:
            logger.warning(
                "ZIP Manage was given no entry name, so every entry was kept. Tick them on "
                "the node, or name them in the entries box"
            )
            return io.NodeOutput(archive, list(held), len(held))

        wanted = set(chosen)
        missing = sorted(wanted.difference(held))
        if missing:
            logger.warning(
                "ZIP Manage was named %d entr(y/ies) the archive does not hold, which were "
                "passed over: %s",
                len(missing),
                ", ".join(missing[:8]) + (", ..." if len(missing) > 8 else ""),
            )

        keeping = [name for name in held if (name in wanted) == (action == KEEP)]
        built = draft.kept(archive, keeping, compression)
        names = [entry.name for entry in built.files]
        logger.info(
            "ZIP Manage %s %d of %d entr(y/ies)",
            "kept" if action == KEEP else "dropped to",
            len(names),
            len(held),
        )
        return io.NodeOutput(built, names, len(names))

    @classmethod
    def ticked(cls, held, timeout=0):
        """The entries chosen on the node while the run was held.

        Args:
            held: Every entry name the archive carries, in its own order.
            timeout: Seconds to hold, or 0 to hold with no limit.

        Returns:
            The names to keep. Every name when the hold ran out or answered nothing.
        """
        from ...modules.interface import pause

        outcome, value = pause.wait_for_resume(
            str(cls.hidden.unique_id),
            timeout=float(timeout),
            message="Tick the entries to keep, then resume",
            kind=HOLD_KIND,
            content=chr(10).join(held),
        )
        if outcome != pause.RESUMED:
            logger.warning(
                "ZIP Manage %s before anything was ticked, so every entry was kept",
                outcome,
            )
            return list(held)
        chosen, _ = parse_lines(value)
        if not chosen and value.strip() == "":
            logger.warning(
                "ZIP Manage was resumed with nothing chosen, so every entry was kept"
            )
            return list(held)
        return [name for name in held if name in set(chosen)]
