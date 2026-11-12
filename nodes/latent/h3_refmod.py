"""Saved MiniMax H3 references joined to an asset chain for the segments that name them."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.archive import picks
from ...modules.compat.types import H3_ASSETS, H3_REF_MODS
from ...modules.interface import preview, run_result
from ...modules.io import picker
from ...modules.latent import h3_assets, h3_conditioning, h3_refmods

logger = log.get_logger("nodes.h3_refmod")

NODE_NAME = "MiniMax H3 RefMod"

ASSETS_HINT = (
    "The assets before these, from MiniMax H3 Asset or another MiniMax H3 RefMod. These join "
    "the end of that chain."
)

REFMOD_HINT = (
    "A RefMod in models/refmods: `(none)` = only what mods carries. A file holding a look and "
    "a voice adds both, as a `<Video N>` and an `<Audio N>`."
)

SEGMENT_HINT = (
    "Which segment's prompt names them, as `1` for the opening segment or `3` for Segment 3. "
    "`0` = every segment, for a cast member the whole run references."
)

STRENGTH_HINT = (
    "`1` = as saved; `0.5` = softer, mixed toward a blurred copy; `0` = left out. Scales the "
    "strengths a RefMod Stack set too."
)

MODS_HINT = (
    "A RefMod Stack's mods output, each RefMod with the strength the stack gave it. They join "
    "after the file picked in refmod, in the stack's order."
)


def preview_of(chosen: str, picked: list) -> object:
    """A picture standing for what the node added, for its panel.

    Args:
        chosen: The refmod widget's value.
        picked: The RefMods added, in order.

    Returns:
        A ``[1, H, W, 3]`` picture, or ``None`` where none holds one.
    """
    import numpy as np
    import torch

    if chosen and chosen != h3_refmods.NONE:
        picture = h3_refmods.preview_picture(chosen)
        if picture is not None:
            return torch.from_numpy(np.array(picture)).float()[None] / 255.0
    for mod in picked:
        frame = h3_refmods.first_stored(mod)
        if frame is not None:
            return frame
    return None


class MiniMaxH3RefMod(io.ComfyNode):
    """Add saved references to a MiniMax H3 asset chain for one segment or for every one."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASMiniMaxH3RefMod",
            display_name=NODE_NAME,
            search_aliases=[
                "WASMiniMaxH3RefMod",
                NODE_NAME,
                "refmod",
                "refmods",
                "minimax h3",
                "h3 reference",
                "saved reference",
                "prompt timeline refmod",
                "asset chain",
            ],
            category="WAS Suite/Latent/Video",
            description=(
                "Saved MiniMax H3 references, RefMods, for one segment or every one, joined to "
                "the asset chain. The segment's prompt names each look and voice as "
                "<Picture N>, <Video N> or <Audio N>, and nothing is encoded again. Takes a "
                "RefMod from models/refmods, a RefMod Stack's bundle, or both."
            ),
            inputs=[
                H3_ASSETS.Input("assets", optional=True, tooltip=ASSETS_HINT),
                picker.path_input(io.Combo.Input(
                    "refmod", options=[h3_refmods.NONE, *h3_refmods.names()],
                    default=h3_refmods.NONE, tooltip=REFMOD_HINT,
                ), model=True),
                io.Int.Input(
                    "segment", default=1, min=h3_assets.EVERY_SEGMENT,
                    max=h3_conditioning.MAX_ROWS, tooltip=SEGMENT_HINT,
                ),
                io.Float.Input(
                    "strength", default=1.0, min=0.0, max=1.0, step=0.05, tooltip=STRENGTH_HINT,
                ),
                H3_REF_MODS.Input("mods", optional=True, tooltip=MODS_HINT),
            ],
            outputs=[
                H3_ASSETS.Output(
                    display_name="assets",
                    tooltip=(
                        "The chain with these RefMods at its end, for the next asset node or "
                        "for assets on MiniMax H3 Conditioning."
                    ),
                ),
                io.String.Output(
                    display_name="report",
                    tooltip="Each RefMod added, its kind, its tokens and the segment naming it.",
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, assets=None, refmod=h3_refmods.NONE, segment=1, strength=1.0,
                           mods=None):
        """Read the RefMod again once its file has changed on disk."""
        if not refmod or refmod == h3_refmods.NONE:
            return "none"
        try:
            return picks.fingerprint(h3_refmods.located(refmod)[1])
        except Exception:
            return float("nan")

    @classmethod
    def execute(cls, assets=None, refmod=h3_refmods.NONE, segment=1, strength=1.0,
                mods=None) -> io.NodeOutput:
        """Read the RefMods picked and wired, and add each to the chain.

        Raises:
            ValueError: Nothing is picked or wired, every strength is 0, a file is no RefMod, or
                assets or mods carries something else.
            PathNotAllowed: The RefMod named lies outside the RefMod folders.
            FileNotFoundError: No RefMod of that name is on this machine.
        """
        earlier = h3_assets.collect(assets, f"assets on {NODE_NAME}")
        chosen = str(refmod or "").strip()
        picked = []
        if chosen and chosen != h3_refmods.NONE:
            picked.extend((mod, 1.0) for mod in h3_refmods.load(chosen))
        picked.extend(h3_refmods.entries(mods, f"mods on {NODE_NAME}"))
        if not picked:
            raise ValueError(
                f"{NODE_NAME} has nothing to add: refmod is `(none)` and nothing is wired into "
                f"mods. Pick a RefMod from the menu, or wire a RefMod Stack's mods output"
            )

        segment = int(segment)
        where = "every segment" if segment == h3_assets.EVERY_SEGMENT else f"segment {segment}"
        added, rows, total = [], [], 0
        for mod, each in picked:
            held = float(each) * float(strength)
            if held <= 0.0:
                continue
            name = str(getattr(mod, "name", "") or "RefMod")
            role = h3_refmods.role_kind(mod.kind)
            added.append(h3_assets.Asset(role, segment, 0, None, None, name, False, mod, held))
            count = h3_refmods.tokens(mod)
            total += count
            rows.append((name, f"<{h3_refmods.TAG_WORDS[mod.kind]} N>, {count} tokens"
                         + (f", strength {held:.2f}" if held < 1.0 else "")))
        lines = [f"{name}: {what}" for name, what in rows]
        if not added:
            raise ValueError(
                f"every RefMod on {NODE_NAME} is held at strength 0, so none reaches the prompt. "
                f"Raise strength, or the strengths on the RefMod Stack"
            )

        summary = (f"{len(added)} RefMod reference{'s' if len(added) != 1 else ''} for {where}, "
                   f"{total} tokens a step")
        report = summary + "\n" + "\n".join(lines)
        logger.info("%s: %s", NODE_NAME, summary)
        picture = preview_of(chosen, [mod for mod, _ in picked])
        if picture is not None:
            preview.publish_output(picture, slot="refmod")
        facts = {"segment": where}
        if chosen and chosen != h3_refmods.NONE:
            facts["file"] = chosen
        for place, (name, what) in enumerate(rows[:6], start=1):
            facts[f"{place}. {name}"] = what
        run_result.publish(
            summary=summary,
            counts={"references": len(added), "tokens": total, "place in chain": len(earlier) + 1},
            facts=facts,
        )
        held = {"file": chosen if chosen != h3_refmods.NONE else "",
                "members": [{"kind": asset.refmod.kind, "label": asset.name[:120]} for asset in added]}
        return io.NodeOutput((*earlier, *added), report, ui={"was_refmods": [held]})
