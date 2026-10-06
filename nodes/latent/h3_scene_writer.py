"""Scene prompts for a MiniMax H3 video, written by a language model from an idea and a cast."""

from __future__ import annotations

import json

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import H3_ASSETS
from ...modules.interface import run_result
from ...modules.latent import h3_assets, h3_conditioning, h3_writer

logger = log.get_logger("nodes.h3_scene_writer")

NODE_NAME = "MiniMax H3 Scene Writer"

#: The largest seed the sampler takes.
MAX_SEED = 0xFFFFFFFFFFFFFFFF

#: Longest side a cast picture is shown to the model at, in pixels.
PICTURE_EDGE = 768


def pool_of(assets: list) -> list[tuple[str, object]]:
    """The distinct reference pictures of a chain, in chain order.

    Args:
        assets: Every asset of the chain.

    Returns:
        ``(name, picture)`` per picture, ``picture`` as ``[1, H, W, C]``.
    """
    found = {}
    for asset in assets:
        if asset.role != h3_assets.REFERENCE_PICTURE or asset.moment or asset.frames is None:
            continue
        found.setdefault(asset.name, asset.frames[:1])
    return list(found.items())


def shrunk(picture):
    """A picture no longer than ``PICTURE_EDGE`` on its longest side.

    Args:
        picture: ``[1, H, W, C]`` in ``[0, 1]``.

    Returns:
        The picture, resized by area where it was larger, as a contiguous ``[1, h, w, C]``.
    """
    import comfy.utils

    height, width = int(picture.shape[1]), int(picture.shape[2])
    scale = PICTURE_EDGE / max(height, width)
    if scale >= 1.0:
        return picture.contiguous()
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return comfy.utils.common_upscale(picture.movedim(-1, 1), *size, "area", "disabled").movedim(1, -1).contiguous()


class H3SceneWriter(io.ComfyNode):
    """A MiniMax H3 video's scene prompts, header, footer and transitions, written from an idea."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASH3SceneWriter",
            display_name=NODE_NAME,
            search_aliases=[
                "WASH3SceneWriter",
                NODE_NAME,
                "h3 scene writer",
                "write scenes",
                "story to prompts",
                "minimax h3",
                "prompt timeline",
                "llm",
                "vlm",
            ],
            category="WAS Suite/Latent/Video",
            description=(
                "Writes a whole MiniMax H3 video from an idea: a language model outlines the "
                "story, casts each scene from the reference pictures in an asset chain, writes "
                "every scene's prompt and the shared header and footer under a system prompt of "
                "rules, then reads the scenes back to choose the cuts and carries between them. "
                "The Prompt Timeline's Write scenes runs it and fills the rows."
            ),
            inputs=[
                io.Clip.Input(
                    "vlm_clip",
                    tooltip=(
                        "The language model that writes, from Load CLIP, as "
                        "`qwen3vl_8b_fp8_scaled.safetensors`. One that reads pictures, such as "
                        "Qwen3-VL or Gemma 3, describes the cast from their pictures; a text "
                        "model such as `qwen_3_8b` works from the file names."
                    ),
                ),
                io.String.Input(
                    "idea", multiline=True, default="",
                    tooltip=(
                        "What the video is about, as `The gang drive to a shrimp factory in the "
                        "bayou and unmask the monster haunting it.` Names, who speaks which "
                        "language and the beats to hit all carry into the scenes."
                    ),
                ),
                io.Int.Input(
                    "scenes", default=8, min=1, max=h3_conditioning.MAX_ROWS,
                    tooltip=f"How many scenes to write, `1` to `{h3_conditioning.MAX_ROWS}`.",
                ),
                io.Float.Input(
                    "seconds", default=8.0, min=h3_writer.SHORTEST_SCENE, max=h3_writer.MOST_SECONDS,
                    step=0.1,
                    tooltip=(
                        "About how long each scene runs, as `8.0`, at most `12`; the model may vary it "
                        "scene by scene."
                    ),
                ),
                io.String.Input(
                    "style", default=h3_writer.DEFAULT_STYLE,
                    tooltip=(
                        "The look, finishing the sentence `The target video is ...`, as `a 2D-animated "
                        "1970s Saturday-morning mystery cartoon with bold inked outlines`."
                    ),
                ),
                io.Int.Input(
                    "seed", default=0, min=0, max=MAX_SEED, control_after_generate=True,
                    tooltip="Which draw of the model's choices to take, as `0` or `42`; the same seed with the same inputs writes the same scenes.",
                ),
                H3_ASSETS.Input(
                    "assets", optional=True,
                    tooltip=(
                        "The chain whose reference pictures are the cast and places to write "
                        "for, as wired into assets on MiniMax H3 Conditioning. Without it every "
                        "subject is described in words."
                    ),
                ),
                io.String.Input(
                    "system_prompt", multiline=True, default=h3_writer.DEFAULT_SYSTEM_PROMPT, optional=True,
                    tooltip=(
                        "The rules every scene, header and footer is written under: the prompt "
                        "layout, shots, dialogue and sound. A line such as `Keep every line of "
                        "dialogue under 12 words.` adds a rule; blank writes under none."
                    ),
                ),
                io.Boolean.Input(
                    "plan_transitions", default=True, optional=True,
                    tooltip=(
                        "`true` reads the written scenes back and chooses each one's transition: "
                        "cut, carry, sound, cast or cast and sound. `false` cuts between every scene."
                    ),
                ),
                io.Float.Input(
                    "temperature", default=0.7, min=0.01, max=2.0, step=0.01, optional=True, advanced=True,
                    tooltip="How adventurous each pick is. `0.7` is balanced, `0.3` stays close to the likeliest words, `1.1` wanders.",
                ),
                io.Int.Input(
                    "top_k", default=64, min=0, max=1000, optional=True, advanced=True,
                    tooltip="Pick only among this many likeliest tokens. `0` turns the limit off.",
                ),
                io.Float.Input(
                    "top_p", default=0.95, min=0.0, max=1.0, step=0.01, optional=True, advanced=True,
                    tooltip="Pick only among the likeliest tokens whose chances add up to this. `1.0` turns it off.",
                ),
                io.Float.Input(
                    "min_p", default=0.05, min=0.0, max=1.0, step=0.01, optional=True, advanced=True,
                    tooltip="Drop any token less likely than this fraction of the likeliest one. `0` turns it off.",
                ),
                io.Float.Input(
                    "repetition_penalty", default=1.05, min=0.0, max=5.0, step=0.01, optional=True, advanced=True,
                    tooltip="Above `1.0` makes a token already used less likely again. `1.0` turns it off.",
                ),
                io.Boolean.Input(
                    "thinking", default=False, optional=True, advanced=True,
                    tooltip="`true` lets a model that reasons, such as Qwen3, think before each answer.",
                ),
                io.Boolean.Input(
                    "fast_decode", default=True, optional=True, advanced=True,
                    tooltip="`true` decodes on the fixed cache graph path where the model allows it; `false` runs as core Generate Text.",
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="scenes",
                    tooltip=(
                        "Every scene as JSON, with the header and footer: title, summary, prompt, "
                        "seconds, transition, continuity, overlap and the cast pictures it references."
                    ),
                ),
                io.String.Output(
                    display_name="prompts",
                    tooltip="Every scene's prompt, in order, divided by a line of dashes.",
                ),
                io.String.Output(display_name="header", tooltip="The text put before every scene's prompt."),
                io.String.Output(display_name="footer", tooltip="The text put after every scene's prompt."),
                io.String.Output(display_name="report", tooltip="What was written, and how long it runs."),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, vlm_clip, idea, scenes, seconds, style, seed, assets=None,
                system_prompt=h3_writer.DEFAULT_SYSTEM_PROMPT, plan_transitions=True,
                temperature=0.7, top_k=64, top_p=0.95, min_p=0.05, repetition_penalty=1.05,
                thinking=False, fast_decode=True) -> io.NodeOutput:
        """Describe the cast, outline the scenes, write the wrap and each scene, and plan the joins.

        Raises:
            ValueError: No model or no idea arrived, or the model wrote no outline.
        """
        import comfy.model_management
        import comfy.utils

        if vlm_clip is None:
            raise ValueError(
                f"{NODE_NAME} needs a language model. Wire Load CLIP with a model such as "
                f"qwen3vl_8b_fp8_scaled.safetensors into vlm_clip"
            )
        if not str(idea or "").strip():
            raise ValueError(f"{NODE_NAME} has no idea to write from. Type one into idea")
        style = str(style or "").strip().rstrip(".") or h3_writer.DEFAULT_STYLE
        pool = pool_of(h3_assets.collect(assets, f"assets on {NODE_NAME}"))
        sees = h3_writer.sees_pictures(vlm_clip)
        count = max(1, min(h3_conditioning.MAX_ROWS, int(scenes)))
        sampling = h3_writer.Sampling(float(temperature), int(top_k), float(top_p), float(min_p),
                                      float(repetition_penalty), bool(thinking), bool(fast_decode))
        write = h3_writer.make_writer(vlm_clip, int(seed), sampling, str(system_prompt or ""))
        progress = comfy.utils.ProgressBar(len(pool) + 2 + count + (1 if plan_transitions else 0))

        cast = []
        for number, (name, picture) in enumerate(pool, start=1):
            comfy.model_management.throw_exception_if_processing_interrupted()
            cast.append(h3_writer.describe(write, f"C{number}", h3_writer.name_of(name),
                                           shrunk(picture) if sees else None, sees))
            progress.update(1)

        outline = h3_writer.outline(write, str(idea), count, float(seconds), style, cast)
        progress.update(1)
        comfy.model_management.throw_exception_if_processing_interrupted()
        wrap = h3_writer.header_footer(write, outline, style)
        progress.update(1)

        known = {member.key: member for member in cast}
        written = []
        for index, scene in enumerate(outline):
            comfy.model_management.throw_exception_if_processing_interrupted()
            ids = h3_writer.referenced(scene)
            neighbours = (
                outline[index - 1]["summary"] if index > 0 else "",
                outline[index + 1]["summary"] if index + 1 < len(outline) else "",
            )
            number = index + 1
            heads, foots = number in wrap["header_scenes"], number in wrap["footer_scenes"]
            prompt = h3_writer.scene_prompt(
                write, scene, index, len(outline), style, known, ids,
                (wrap["header"] if heads else "", wrap["footer"] if foots else ""), neighbours)
            written.append({
                "title": scene["title"],
                "summary": scene["summary"],
                "prompt": prompt,
                "seconds": scene["seconds"],
                "transition": "cut",
                "continuity": "cut",
                "overlap": 0,
                "why": "",
                "references": [int(key[1:]) - 1 for key in ids],
                "wrap": h3_conditioning.WRAPS[(0 if heads and foots else 1 if heads else 2 if foots else 3)],
            })
            progress.update(1)

        if plan_transitions and len(written) > 1:
            comfy.model_management.throw_exception_if_processing_interrupted()
            planned = h3_writer.plan(write, [
                {"prompt": scene["prompt"], "seconds": scene["seconds"],
                 "pictures": len(scene["references"]), "summary": scene["summary"]}
                for scene in written
            ])
            for choice in planned:
                written[choice["scene"] - 1].update({
                    key: choice[key] for key in ("transition", "continuity", "overlap", "why")
                })
            progress.update(1)

        payload = {
            "cast": [
                {"file": name, "name": member.name, "kind": member.kind, "description": member.description}
                for (name, _), member in zip(pool, cast)
            ],
            "header": wrap["header"],
            "footer": wrap["footer"],
            "planned": bool(plan_transitions),
            "scenes": written,
        }
        text = json.dumps(payload, ensure_ascii=False)
        total = sum(scene["seconds"] for scene in written)
        joins = {}
        for scene in written[1:]:
            joins[scene["transition"]] = joins.get(scene["transition"], 0) + 1
        report = (
            f"{len(written)} scene(s), about {total:.0f}s, a cast of {len(cast)} "
            f"{'read from their pictures' if sees else 'named from their files'}"
        )
        logger.info("%s: %s", NODE_NAME, report)
        run_result.publish(
            summary=report,
            counts={"scenes": len(written), "cast": len(cast), "seconds": round(total, 1)},
            facts={
                "pictures read": "yes" if sees else "no, file names only",
                "transitions": ", ".join(f"{count} {name}" for name, count in joins.items()) or "none",
                "footer": "written" if wrap["footer"] else "empty",
            },
            bodies=run_result.body("scenes", "\n".join(
                f"{number}. {scene['title']} ({scene['seconds']:g}s, {scene['transition']})"
                for number, scene in enumerate(written, start=1)
            )),
        )
        return io.NodeOutput(
            text,
            h3_writer.SCENE_BREAK.join(scene["prompt"] for scene in written),
            wrap["header"],
            wrap["footer"],
            report,
            ui={"was_h3_scenes": [text]},
        )
