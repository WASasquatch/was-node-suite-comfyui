"""MiniMax H3 scene prompts written, rewritten and joined by a language model.

The model answers descriptions, outlines and transitions as JSON and writes each prompt as text
under a system prompt of rules; picture and subject numbers are assigned here.
"""

from __future__ import annotations

import json
import os
import re
from typing import Callable, NamedTuple

from . import h3_extend

__all__ = [
    "CAST_KINDS",
    "DEFAULT_STYLE",
    "DEFAULT_SYSTEM_PROMPT",
    "Member",
    "Sampling",
    "TRANSITIONS",
    "TRANSITION_NOTES",
    "base_prompt",
    "clean_reply",
    "describe",
    "digest",
    "header_footer",
    "json_of",
    "make_writer",
    "prompt_of",
    "name_of",
    "outline",
    "plan",
    "reference_prompt",
    "rewrite",
    "scene_prompt",
    "scene_seconds",
    "sees_pictures",
    "speakers_of",
    "split_scenes",
]

#: What a picture in the cast can be.
CAST_KINDS = ("character", "location", "object")

#: The transitions a plan chooses from, and the row settings each becomes: continuity and overlap.
TRANSITIONS = {
    "cut": ("cut", 0),
    "carry": ("carry", 22),
    "sound": (h3_extend.AUDIO_CARRY, 17),
    "cast": (h3_extend.REFERENCE_VIDEO, 0),
    "cast and sound": (h3_extend.AUDIO_REFERENCE, 17),
}

#: The transitions that reference the scene before for its cast, and what each becomes for a
#: scene with reference pictures of its own.
CAST_REFERENCES = {"cast": "cut", "cast and sound": "sound"}

#: What each transition does, in the words the model choosing them reads.
TRANSITION_NOTES = {
    "cut": "a new place, a jump in time or a new sequence. Nothing carries over: new picture, new sound.",
    "carry": "the very same shot goes on unbroken: same place, same framing, the action and camera "
             "continue from the last frame. Picture and sound both carry over.",
    "sound": "the picture cuts to a new shot while the sound and music run on across the cut, for a "
             "conversation or a moment that continues from another angle.",
    "cast": "the picture cuts to a new shot of the same people in the same place; the last moments of "
            "the scene before are shown to the model so faces, clothes and the room stay the same. "
            "The sound starts fresh.",
    "cast and sound": "as cast, and the sound and music run on across the cut.",
}

#: The style a scene is written in when none is given.
DEFAULT_STYLE = "a 2D-animated cartoon with clean inked outlines and flat cel-painted colours"

#: The rules every prompt is written under, unless the node is given its own.
DEFAULT_SYSTEM_PROMPT = """You write production-ready prompts for MiniMax H3, which generates synchronized video and audio from text.

Each requested scene is one independent prompt. Write in English except dialogue, lyrics, and visible on-screen text. Never assume textual context from another prompt. Explicitly restate any continuity-critical starting state. Every clip must state its duration and be no longer than 12 seconds.

GLOBAL HEADER / FOOTER

Use a shared HEADER or FOOTER only when multiple scenes genuinely inherit the same context.

HEADER may contain persistent visual style, character/environment rules, continuity constraints, or recurring cinematography.

FOOTER may contain persistent overall soundscape, non-diegetic music, or shared audio treatment.

Do not emit empty HEADER/FOOTER sections or attach them to scenes that do not use the shared context. Do not duplicate global information inside scenes. Scene-specific instructions override the corresponding global rule.

Never move scene-specific action, framing, timing, dialogue, or temporary sound into global sections.

PROMPT MODES

Standard T2VA and keyframe modes use:

integrated_multimodal_description:
overall_soundscape:
non_diegetic_music:

T2VA: no alignment line.

I2VA:
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

FL2VA:
How the reference pictures align with the target video \u2014 Picture 1 (from Shot 1) aligns with 0.00 seconds; Picture 2 (from Shot N) aligns with the final timestamp.

L2VA:
How the reference pictures align with the target video \u2014 <Picture 1> (from [Shot N]) aligns with the final timestamp.

For I2VA, develop naturally forward from the reference frame. For FL2VA, describe the visible transition connecting both frames. For L2VA, infer a plausible earlier state that progressively converges on the reference frame.

Full-Reference mode uses exactly:

subject_definitions:
summary:
retention_analysis:
detailed_description:
overall_soundscape:
non_diegetic_music:

subject_definitions:
Define only references that must be tracked.

Use <Subject N> for reusable visible subjects, environments, objects, costumes, poses, effects, or styles; <Picture N> for explicit frame/composition anchors; <Video N> for source motion/editing/continuation; <Audio N> only for intentionally referenced or reused audio.

If a picture only defines a subject's appearance, mention the picture inside that subject definition instead of creating a separate picture entry.

summary:
Begin with one or more applicable task types:
[reference generation], [keyframe completion], [video editing], [video continuation], [audio reuse], [audio reference].
Join multiple types with " + ". Briefly state what happens and how references are used.

retention_analysis:
One line per tracked reference.

Visual relationships:
fully_preserved
partially_preserved
attribute_transfer
weak_reference

Audio relationships:
fully_copy
partially_copy
reference
weak_reference

detailed_description:
Describe the complete audiovisual timeline using [Shot N]. Include composition, subject appearance/position, environment, light, chronological action, camera, synchronized sound, and reference usage. Use roughly 350\u2013500 words only when the scene actually benefits from that detail.

SHOTS, CAMERA, AND CONTINUITY

[Shot 1] has no timestamp.

Later cuts use:
[Shot 2] At 00:04.000, the shot cuts to ...

Timestamps must increase and remain within the clip duration.

Cut only when revealing genuinely new information. For smaller framing or angle changes, move the camera instead.

At each shot start, establish shot size/composition, subjects and positions, setting/light, action order, camera behavior, and relevant synchronized sound.

Useful camera terms include push in, pull out, zoom, pan, truck, tilt, pedestal, arc shot, tracking shot, static shot, POV, roll, and shake. State direction, amplitude, and speed when meaningful. Distinguish physical camera movement from optical zoom.

Maintain subject identity, wardrobe, lighting, spatial relationships, screen direction, and object state unless the change is visibly shown.

Describe only what can be seen or heard. Prefer concrete audiovisual instructions over literary or psychological prose.

STYLE AND SUBJECTS

When not supplied by HEADER, establish the visual medium and concrete treatment: live-action, cinematic, 2D animation, 3D CG, claymation, watercolor, vintage film, etc., plus relevant line treatment, materials, palette, contrast, lighting, lens behavior, grain, or rendering characteristics.

When a subject first appears, establish the visible identity anchors needed for that scene, then use its tag or an unambiguous description consistently.

A place, vehicle, creature, prop, effect, costume, or pose may be a subject.

Prefer no more than three principal characters simultaneously unless the requested scene requires more.

DIALOGUE

Speaker IDs are local to each independent prompt and restart at (S1). Assign IDs in order of first vocalization; a speaker keeps the same ID within that prompt.

Format:
<Subject N> (S1) performs an action and says, <d>[Language] Exact words.</d>

Inside <d>, include only the language tag and spoken words.

On the first vocal event, describe useful voice properties when relevant.

Voiceover:
says in an off-screen voiceover:
If that character is visible, explicitly state that their lips remain closed.

Use <scenetrans> only when dialogue intentionally continues across a cut. Use <cutoff> only when speech is intentionally truncated by the end of the clip.

Keep dialogue physically achievable within the duration and, unless necessary otherwise, leave a final silent action or reaction beat.

ON-SCREEN TEXT

Only visibly rendered text goes in double quotation marks, exactly as shown. Dialogue does not.

AUDIO

Place event-synchronized sounds inside the shot description where they occur.

overall_soundscape:
1\u20134 concise sentences covering ambience, physical sounds, and non-verbal human sounds. Do not repeat dialogue. Use N/A only for intentional complete silence.

non_diegetic_music:
1\u20133 concise sentences describing instrumentation, tempo, rhythm, density, dynamics, and changes over time. Do not describe intended emotion. Diegetic music belongs in the shot description. Use N/A when there is no score.

Create <Audio N> only when source audio is intentionally reused or referenced. Distinguish full copy, partial copy, and characteristic/reference use. Referencing a speaker's voice does not imply copying unrelated source dialogue.

FINAL RULES

Each prompt must be independently understandable and chronologically executable.

Do not refer to "the same character," "the previous scene," or other unavailable textual context.

Do not invent cuts, actions, dialogue, text, props, camera motion, or reference relationships that conflict with the user's staging.

Do not overload a short clip. Ensure all action, dialogue, camera movement, transitions, and the final beat can physically fit within the stated duration."""

#: Submodule names a language model carries when it reads pictures.
VISION_PARTS = ("visual", "vision_model", "vision_tower", "multi_modal_projector")

#: The most seconds a scene may run.
MOST_SECONDS = 12.0

#: Longest and shortest scene, in seconds, on the model's frame grid.
LONGEST_SCENE = ((int(MOST_SECONDS * h3_extend.FPS) - h3_extend.CLIP_LEAD) // h3_extend.CLIP_FRAMES
                 * h3_extend.CLIP_FRAMES + h3_extend.CLIP_LEAD) / h3_extend.FPS
SHORTEST_SCENE = h3_extend.snap_clip(124) / h3_extend.FPS

#: Characters, locations and objects one scene may show.
MOST_CHARACTERS = 3
MOST_OBJECTS = 2

#: The line between scenes in a block of prompts.
SCENE_BREAK = "\n\n----------\n\n"

#: Times a scene is asked for before its outline is written into the template instead.
SCENE_ATTEMPTS = 3

#: Tokens a model that thinks first is given on top of an answer's own.
THINKING_TOKENS = 2048

#: The mode a scene with reference pictures is written in, and its sections in order.
REFERENCE_MODE = ("Full-Reference mode", ("subject_definitions", "summary", "retention_analysis",
                                          "detailed_description", "overall_soundscape", "non_diegetic_music"))

#: The mode a scene without pictures is written in, and its sections in order.
TEXT_MODE = ("T2VA mode", ("integrated_multimodal_description", "overall_soundscape", "non_diegetic_music"))

#: A section name run on inside a line, as the ``non_diegetic_music:`` in ``... gulls. non_diegetic_music: N/A``.
RUN_ON_SECTION = re.compile(r"[ \t]+(?=[a-z]+(?:_[a-z]+)+:\s)")

#: A subject, picture, video or audio tag in a prompt, as ``<Picture 2>``.
TAG = re.compile(r"<(Subject|Picture|Video|Audio) (\d+)>")


class Member(NamedTuple):
    """One picture of the cast, as the writer knows it.

    Attributes:
        key: Its id in the outline, as ``C1``.
        name: A name read from its file.
        kind: An entry of :data:`CAST_KINDS`.
        description: What it looks like, as a noun phrase.
        keep: What a scene keeps of it, as a list in words.
        voice: How a character sounds, empty for anything else.
    """

    key: str
    name: str
    kind: str
    description: str
    keep: str
    voice: str


class Sampling(NamedTuple):
    """How the language model draws its words.

    Attributes:
        temperature: How adventurous each pick is.
        top_k: Picks only among this many likeliest tokens, ``0`` for no limit.
        top_p: Picks only among the likeliest tokens whose chances add up to this.
        min_p: Drops tokens less likely than this fraction of the likeliest.
        repetition_penalty: Above ``1.0`` makes a used token less likely again.
        thinking: Whether a reasoning model thinks before answering.
        fast: Whether to take the fast decode path.
    """

    temperature: float = 0.7
    top_k: int = 64
    top_p: float = 0.95
    min_p: float = 0.05
    repetition_penalty: float = 1.05
    thinking: bool = False
    fast: bool = True


def make_writer(clip, seed: int, sampling: Sampling, system: str = ""):
    """A function asking the language model one question under a system prompt.

    Args:
        clip: A ComfyUI ``CLIP`` holding a language model.
        seed: The sampling seed.
        sampling: How words are drawn.
        system: The rules every question is asked under, empty for none.

    Returns:
        ``write(prompt, image=None, max_length=1024, temperature=None, ruled=True)`` answering
        text; ``temperature`` replaces the sampling's own, ``ruled`` asks under ``system``.
    """
    import comfy.model_management
    import comfy.utils

    from ..model import text_decode

    rules = str(system or "").strip()
    honoured = honours_system(clip) if rules else False

    def quiet(*args, **kwargs):
        comfy.model_management.throw_exception_if_processing_interrupted()

    def write(prompt, image=None, max_length=1024, temperature=None, ruled=True):
        asked = str(prompt)
        extra = {}
        if ruled and rules:
            if honoured:
                extra["system_prompt"] = rules
            else:
                asked = f"Follow these rules.\n\n{rules}\n\n----------\n\n{asked}"
        tokens = clip.tokenize(asked, image=image, min_length=1, thinking=bool(sampling.thinking), **extra)
        budget = int(max_length) + (THINKING_TOKENS if sampling.thinking else 0)
        heat = sampling.temperature if temperature is None else min(float(temperature), sampling.temperature)
        # The node's own bar counts its steps; the model's per-token bar only checks for Stop.
        hook = comfy.utils.PROGRESS_BAR_HOOK
        comfy.utils.PROGRESS_BAR_HOOK = quiet
        try:
            ids, _ = text_decode.generate(
                clip, tokens, budget, fast=bool(sampling.fast), do_sample=True,
                temperature=max(0.01, float(heat)), top_k=int(sampling.top_k),
                top_p=float(sampling.top_p), min_p=float(sampling.min_p),
                repetition_penalty=float(sampling.repetition_penalty), presence_penalty=0.0,
                seed=int(seed), mtp=True,
            )
        finally:
            comfy.utils.PROGRESS_BAR_HOOK = hook
        return clip.decode(ids)

    return write


def honours_system(clip) -> bool:
    """Whether a model's chat template takes a system prompt.

    Args:
        clip: A ComfyUI ``CLIP`` holding a language model.

    Returns:
        True where a system prompt changes the tokens the model reads.
    """

    def count(tokens) -> int:
        batches = tokens.values() if isinstance(tokens, dict) else [tokens]
        return sum(len(row) for batch in batches for row in batch)

    try:
        plain = clip.tokenize("probe", min_length=1, thinking=False)
        ruled = clip.tokenize("probe", min_length=1, thinking=False, system_prompt="Answer in English only.")
    except TypeError:
        return False
    return count(plain) != count(ruled)


def name_of(file: str) -> str:
    """A readable name from a file name.

    Args:
        file: A file name or label, as ``cast/jo-zhang.png [input]``.

    Returns:
        The name, as ``Jo Zhang``.
    """
    base = os.path.basename(re.sub(r"\s*\[[^\]]*\]\s*$", "", str(file or "")))
    stem = os.path.splitext(base)[0]
    words = re.sub(r"[_\-.]+", " ", stem).split()
    return " ".join(word if word.isupper() else word.capitalize() for word in words) or "Untitled"


def sees_pictures(clip) -> bool:
    """Whether a loaded language model carries a part that reads pictures.

    Args:
        clip: A ComfyUI ``CLIP``.

    Returns:
        True for a vision language model.
    """
    model = getattr(clip, "cond_stage_model", None)
    if model is None or not hasattr(model, "named_modules"):
        return False
    return any(name.rsplit(".", 1)[-1] in VISION_PARTS for name, _ in model.named_modules())


def clean_reply(text: str) -> str:
    """A model's reply without its thinking, code fences or a heading line.

    Args:
        text: The reply.

    Returns:
        The text it wrote, trimmed.
    """
    reply = re.sub(r"<think>.*?</think>", "", str(text or ""), flags=re.S)
    reply = re.sub(r"^.*?</think>", "", reply, flags=re.S) if "</think>" in reply else reply
    reply = re.sub(r"^\s*```[a-z]*\s*\n|\n\s*```\s*$", "", reply.strip())
    return reply.strip()


def prompt_of(text: str) -> str:
    """The prompt a model wrote, without a lead-in line or the markers it was handed it in.

    Args:
        text: The reply.

    Returns:
        The text between ``<<<`` and ``>>>`` where the reply has them, else the reply from its
        first section, shot or alignment line.
    """
    reply = clean_reply(text)
    if "<<<" in reply:
        reply = reply.split("<<<")[-1].split(">>>")[0]
    else:
        reply = reply.split(">>>")[0]
        start = re.search(r"^(?:[a-z_]+:|\[Shot 1\]|For the target video|How the reference pictures)", reply, re.M)
        if start and 0 < start.start() < 200:
            reply = reply[start.start():]
    return reply.strip()


def json_of(text: str):
    """The first JSON object or array in a model's reply.

    Args:
        text: The reply, fenced or not.

    Returns:
        The parsed value.

    Raises:
        ValueError: The reply holds no JSON that parses.
    """
    reply = clean_reply(text)
    reply = re.sub(r"```(?:json)?", "", reply)
    for opening in ("{", "["):
        start = reply.find(opening)
        while start >= 0:
            closing = "}" if opening == "{" else "]"
            depth, inside, escaped = 0, False, False
            for position in range(start, len(reply)):
                char = reply[position]
                if inside:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        inside = False
                elif char == '"':
                    inside = True
                elif char == opening:
                    depth += 1
                elif char == closing:
                    depth -= 1
                    if depth == 0:
                        chunk = reply[start:position + 1]
                        try:
                            return json.loads(chunk)
                        except json.JSONDecodeError:
                            try:
                                return json.loads(re.sub(r",\s*([}\]])", r"\1", chunk))
                            except json.JSONDecodeError:
                                break
            start = reply.find(opening, start + 1)
    raise ValueError(f"the model answered without JSON: {reply.strip()[:200]!r}")


def describe(write: Callable, key: str, name: str, picture, sees: bool) -> Member:
    """One cast member described from its picture, or from its name for a model that cannot see.

    Args:
        write: As :func:`make_writer` returns it.
        key: The member's id, as ``C1``.
        name: Its name.
        picture: Its picture, ``[1, H, W, C]``.
        sees: Whether the model reads pictures.

    Returns:
        The member.
    """
    seen = "the picture" if sees else "the name alone"
    prompt = (
        f"Prepare a reference card for an animated video model from {seen}. The file is named "
        f"\"{name}\".\n\n"
        "Reply with JSON only, in this shape:\n"
        "{\"kind\": \"character\" | \"location\" | \"object\", \"description\": \"...\", "
        "\"keep\": \"...\", \"voice\": \"...\"}\n\n"
        "- kind: character for a person, creature or robot; location for a place or background "
        "with nobody in it; object for a vehicle, prop or thing.\n"
        "- description: one noun phrase of at most 40 words naming what is visible: for a "
        "character the apparent age, build, hair, face, clothing with colours and props; for a "
        "location the setting, landmarks, lighting and palette; for an object its shape, colours, "
        "markings and any visible text in double quotes.\n"
        "- keep: at most 12 words listing the features that identify it, as \"the long dark "
        "hair, black glasses and purple coat\".\n"
        "- voice: for a character, how it sounds, as \"a crisp, clever young female voice\"; "
        "otherwise an empty string.\n"
        "Describe only what is visible. Do not name real people."
    )
    reply = write(prompt, image=picture if sees else None, max_length=320, temperature=0.3, ruled=False)
    try:
        found = json_of(reply)
    except ValueError:
        found = {}
    found = found if isinstance(found, dict) else {}
    kind = str(found.get("kind", "")).strip().lower()
    kind = kind if kind in CAST_KINDS else "character"
    description = str(found.get("description", "")).strip().rstrip(".") or name
    keep = str(found.get("keep", "")).strip().rstrip(".") or f"the look of {name}"
    voice = str(found.get("voice", "")).strip().rstrip(".") if kind == "character" else ""
    return Member(key, name, kind, description, keep, voice)


def scene_seconds(value, fallback: float) -> float:
    """A scene length on the model's frame grid.

    Args:
        value: The length asked for, in seconds.
        fallback: The length used when ``value`` is not a number.

    Returns:
        Seconds between :data:`SHORTEST_SCENE` and :data:`LONGEST_SCENE`.
    """
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        seconds = float(fallback)
    frames = h3_extend.snap_clip(max(1, round(seconds * h3_extend.FPS)))
    return round(min(LONGEST_SCENE, max(SHORTEST_SCENE, frames / h3_extend.FPS)), 2)


def outline(write: Callable, idea: str, count: int, seconds: float, style: str,
            cast: list[Member]) -> list[dict]:
    """The scenes of a video, as the model outlines them.

    Args:
        write: As :func:`make_writer` returns it.
        idea: What the video is about.
        count: How many scenes.
        seconds: About how long each runs.
        style: The look of the video.
        cast: The members a scene may show.

    Returns:
        ``count`` scenes, each a dict as the reply shapes it, cast ids checked against ``cast``.

    Raises:
        ValueError: The model answered without a usable outline twice.
    """
    members = "\n".join(
        f"{member.key} ({member.kind}) {member.name}: {member.description}" for member in cast
    ) or "(no pictures; every subject is described in words)"
    # The example names a speaker the way this cast can: by picture id, or by name with no pictures.
    characters = [member.key for member in cast if member.kind == "character"]
    places = [member.key for member in cast if member.kind == "location"]
    speaker = characters[0] if characters else "..."
    shown = json.dumps(characters[:1])
    placed = json.dumps(places[0]) if places else "null"
    prompt = (
        f"Write the outline of a short video of exactly {count} scenes, as JSON only.\n\n"
        f"The idea:\n{idea.strip()}\n\n"
        f"The look: {style}.\n"
        f"Each scene runs about {min(float(seconds), LONGEST_SCENE):g} seconds, never under "
        f"{SHORTEST_SCENE:.1f} or over {LONGEST_SCENE:.1f}.\n\n"
        "The pictures available, each with an id. A recognisable character, place or prop on "
        f"screen must be one of these:\n{members}\n\n"
        "Rules:\n"
        f"- A scene shows at most {MOST_CHARACTERS} characters, at most one location and at most "
        f"{MOST_OBJECTS} objects, listed by id.\n"
        "- Lines of dialogue stay under 20 words, all of a scene's lines fit in its first two "
        "thirds, and the scene ends on a moment without speech.\n"
        "- Each line is in the language its speaker speaks; a name or the idea saying a character "
        "speaks another language sets it.\n"
        "- who is the speaker's picture id where they have a picture, otherwise their name, as "
        "\"Mara\" or \"the old fisherman\".\n"
        "- summary is one sentence of what happens.\n"
        "- sound lists ambience and sound effects; music names instruments, tempo and dynamics "
        "without mood words, or \"N/A\".\n\n"
        "Reply with JSON only, in this shape:\n"
        "{\"scenes\": [{\"title\": \"...\", \"summary\": \"...\", \"seconds\": 8, "
        f"\"location\": {placed}, \"cast\": {shown}, \"objects\": [], \"action\": \"what happens, in "
        f"order\", \"camera\": \"shot size and movement\", \"lines\": [{{\"who\": \"{speaker}\", \"language\": "
        "\"English\", \"tone\": \"excited\", \"text\": \"...\"}], \"sound\": \"...\", \"music\": "
        "\"...\"}]}"
    )
    known = {member.key: member for member in cast}
    for attempt in range(2):
        reply = write(prompt if attempt == 0 else prompt + "\n\nThe last reply did not parse. "
                      "Reply with the JSON object only.",
                      max_length=min(16000, 400 * count + 600), temperature=None if attempt == 0 else 0.4)
        try:
            found = json_of(reply)
        except ValueError:
            continue
        scenes = found.get("scenes") if isinstance(found, dict) else found
        if not isinstance(scenes, list) or not scenes:
            continue
        cleaned = []
        for index, raw in enumerate(scenes[:count]):
            raw = raw if isinstance(raw, dict) else {}
            location = raw.get("location") if raw.get("location") in known else None
            characters = [key for key in raw.get("cast", []) if key in known][:MOST_CHARACTERS]
            objects = [key for key in raw.get("objects", []) if key in known and key not in characters]
            action = str(raw.get("action", "")).strip()
            cleaned.append({
                "title": str(raw.get("title") or f"Scene {index + 1}").strip(),
                "summary": str(raw.get("summary") or action).strip(),
                "seconds": scene_seconds(raw.get("seconds"), seconds),
                "location": location,
                "cast": characters,
                "objects": objects[:MOST_OBJECTS],
                "action": action,
                "camera": str(raw.get("camera", "")).strip(),
                "lines": [line for line in raw.get("lines", []) if isinstance(line, dict) and line.get("text")],
                "sound": str(raw.get("sound", "")).strip(),
                "music": str(raw.get("music", "")).strip() or "N/A",
            })
        if cleaned:
            return cleaned
    raise ValueError(
        "the language model did not answer with a scene outline twice. Try a larger model, such "
        "as qwen3vl_8b_fp8_scaled.safetensors, or a shorter idea"
    )


def referenced(scene: dict) -> list[str]:
    """The cast ids a scene shows, location first, in the order its prompt numbers them.

    Args:
        scene: One outline scene.

    Returns:
        The ids.
    """
    ids = ([scene["location"]] if scene.get("location") else []) + list(scene.get("cast", []))
    return ids + [key for key in scene.get("objects", []) if key not in ids]


def speakers_of(scene: dict, tags: dict[str, str]) -> dict[str, str]:
    """Each speaking cast member's speaker id, in the order they first speak.

    Args:
        scene: One outline scene.
        tags: Cast id to ``<Subject N>``.

    Returns:
        Speaker, a cast id or a name, to ``S1``, ``S2`` and on.
    """
    order = {}
    for line in scene.get("lines", []):
        who = str(line.get("who") or "").strip()
        if who and who not in order:
            order[who] = f"S{len(order) + 1}"
    return order


def header_footer(write: Callable, scenes: list[dict], style: str) -> dict:
    """The header and footer every scene's prompt is wrapped in, as the model writes them.

    Args:
        write: As :func:`make_writer` returns it.
        scenes: The outline.
        style: The look of the video.

    Returns:
        ``{header, footer, header_scenes, footer_scenes}``: the texts, empty where nothing is
        shared, and the scene numbers from 1 that take each.
    """
    listed = "\n".join(
        f"{index}. {scene['title']}: {scene['summary']} Sound: {scene['sound'] or 'N/A'}. "
        f"Music: {scene['music']}."
        for index, scene in enumerate(scenes, start=1)
    )
    every = json.dumps(list(range(1, len(scenes) + 1)))
    prompt = (
        f"A video of {len(scenes)} scenes. The look: {style}.\n\n{listed}\n\n"
        "Decide the shared HEADER and FOOTER, following the rules on them, and which scenes take "
        "each. Section names go on their own line, as \"non_diegetic_music:\". Leave one empty "
        "where nothing is genuinely shared.\n\n"
        f"Reply with JSON only: {{\"header\": \"...\", \"header_scenes\": {every}, \"footer\": "
        f"\"...\", \"footer_scenes\": {every}}}, listing in each the scenes that take it."
    )

    def taking(found: dict, key: str, text: str) -> list[int]:
        if not text:
            return []
        listed = found.get(f"{key}_scenes")
        if str(listed).strip().lower() == "all":
            return list(range(1, len(scenes) + 1))
        numbers = sorted({int(item) for item in listed if str(item).strip().isdigit()
                          and 1 <= int(item) <= len(scenes)}) if isinstance(listed, list) else []
        return numbers or list(range(1, len(scenes) + 1))

    for attempt in range(2):
        reply = write(prompt, max_length=700, temperature=None if attempt == 0 else 0.4)
        try:
            found = json_of(reply)
        except ValueError:
            continue
        if isinstance(found, dict):
            header, footer = tidy(found.get("header") or ""), tidy(found.get("footer") or "")
            return {"header": header, "footer": footer, "header_scenes": taking(found, "header", header),
                    "footer_scenes": taking(found, "footer", footer)}
    return {"header": "", "footer": "", "header_scenes": [], "footer_scenes": []}


def balanced(text: str) -> bool:
    """Whether every ``<d>`` in a prompt is closed.

    Args:
        text: A prompt.

    Returns:
        True where opening and closing dialogue tags pair up.
    """
    return text.count("<d>") == text.count("</d>")


def numbered_within(text: str, limits: dict[str, int]) -> bool:
    """Whether a prompt's tags stay inside the numbers it was given.

    Args:
        text: A prompt.
        limits: Tag kind to the highest number allowed, as ``{"Picture": 3}``.

    Returns:
        True where no tag names a number past its kind's limit.
    """
    return all(1 <= int(number) <= limits.get(kind, 0) for kind, number in TAG.findall(text))


def stated(reply: str, seconds: float) -> str:
    """A written scene opening on its duration line, with no time on its first shot.

    Args:
        reply: The prompt.
        seconds: The scene's length.

    Returns:
        The prompt with ``duration_seconds:`` set to ``seconds`` as its first line.
    """
    line = f"duration_seconds: {float(seconds):g}"
    body = re.sub(r"^[ \t]*duration_seconds:[^\n]*\n*", "", reply, flags=re.M).strip()
    body = re.sub(r"\[Shot 1\]\s*At [0-9:.]+(?:\s*seconds)?\s*,?\s*(\w?)",
                  lambda match: "[Shot 1] " + match.group(1).upper(), body, count=1)
    return f"{line}\n\n{body}"


def scene_faults(reply: str, limits: dict[str, int], tagged: bool, speaking: bool,
                 sections: tuple = ()) -> tuple[list, list]:
    """What is wrong with a written scene prompt.

    Args:
        reply: The prompt.
        limits: Tag kind to the highest number given, as ``{"Picture": 3}``.
        tagged: Whether the scene has subjects to name by tag.
        speaking: Whether the scene has lines of dialogue.
        sections: The section names its mode carries; a footer or header may carry the sound ones.

    Returns:
        ``(broken, loose)``: faults that make the prompt unusable, and faults it runs with.
    """
    broken, loose = [], []
    if "[Shot 1]" not in reply:
        broken.append("it has no [Shot 1]")
    if not balanced(reply):
        broken.append("a <d> is left open")
    if not numbered_within(reply, limits):
        broken.append("it names a subject or picture number it was not given")
    shots = reply[reply.find("[Shot 1]"):] if "[Shot 1]" in reply else reply
    if tagged and "<Subject" not in shots:
        loose.append("the shots name subjects by name instead of by their <Subject N> tags")
    if speaking and "<d>" not in reply:
        broken.append("the lines of dialogue are missing; write each as <d>[Language] the words</d> after its speaker")
    if reply.count("[Shot 1]") > 1 or len(re.findall(r"^(?:integrated_multimodal_description|detailed_description):",
                                                       reply, re.M)) > 1:
        broken.append("it holds more than one version of the prompt")
    opening = re.search(r"^(?:integrated_multimodal_description|detailed_description):", reply, re.M)
    sound = re.search(r"^overall_soundscape:", reply, re.M)
    first = reply.find("[Shot 1]")
    if first >= 0 and ((opening and first < opening.start()) or (sound and first > sound.start())):
        broken.append("[Shot 1] sits outside the description section")
    missing = [name for name in sections if name not in ("overall_soundscape", "non_diegetic_music")
               and not re.search(rf"^{name}:", reply, re.M)]
    if missing:
        loose.append(f"it is missing the {', '.join(name + ':' for name in missing)} section line")
    if sound and "<d>" in reply[sound.start():]:
        broken.append("a line of dialogue sits after the sound sections; it goes in the shot where it is spoken")
    return broken, loose


def scene_prompt(write: Callable, scene: dict, index: int, count: int, style: str,
                 cast: dict[str, Member], ids: list[str], wrap: tuple, neighbours: tuple) -> str:
    """One scene's prompt, written by the model under the system prompt.

    Args:
        write: As :func:`make_writer` returns it.
        scene: The scene's outline.
        index: Its number, from 0.
        count: How many scenes the video has.
        style: The look of the video.
        cast: Every member, by id.
        ids: The members it shows, in picture order.
        wrap: ``(header, footer)`` this scene is placed between, empty where it takes none.
        neighbours: The summaries of the scenes before and after it, empty where there is none.

    Returns:
        The prompt; the outline written into the template where the model fails twice.
    """
    tags = {key: f"<Subject {number}>" for number, key in enumerate(ids, start=1)}
    speakers = speakers_of(scene, tags)

    def tagged(text: str) -> str:
        return re.sub(r"\bC(\d+)\b", lambda match: tags.get(match.group(0)) or (
            cast[match.group(0)].name if match.group(0) in cast else match.group(0)), str(text or ""))

    mode, sections = REFERENCE_MODE if ids else TEXT_MODE
    layout = f"{mode}, with the sections {', '.join(name + ':' for name in sections)}"
    if ids:
        definitions = "\n".join(definition(cast[key], number) for number, key in enumerate(ids, start=1))
        notes = "\n".join(
            f"{tags[key]} is {cast[key].name}; it keeps {cast[key].keep}"
            + (f"; it speaks in {cast[key].voice}" if cast[key].voice else "")
            for key in ids
        )
        subjects = (
            f"subject_definitions, to copy as written:\n{definitions}\n\n"
            f"For retention_analysis and the voices:\n{notes}\n\n"
            "In the shots, name each subject by its tag, as <Subject 2>, with a few words of its look "
            "the first time it appears."
        )
    else:
        subjects = "No pictures: describe every subject in words."
    def voice(who) -> str:
        who = str(who or "").strip()
        named = tags.get(who) or (cast[who].name if who in cast else who) or "A voice off-screen"
        named = "A voice off-screen" if re.fullmatch(r"C\d+", named) else named
        return f"{named} ({speakers[who]})" if who in speakers else named

    spoken = "\n".join(
        f"{voice(line.get('who'))}, {line.get('tone') or 'plainly'}, says, "
        f"<d>[{line.get('language') or 'English'}] {str(line.get('text', '')).strip()}</d>"
        for line in scene.get("lines", [])
    )
    before, after = neighbours
    description = sections[0] if not ids else "detailed_description"
    needs = [
        f"Write it in {layout}, each section name on its own line, and [Shot 1] inside the "
        f"{description}: section with no time.",
        f"Open with the line duration_seconds: {scene['seconds']:g}, and fit every shot inside it.",
    ]
    if spoken:
        needs.append(f"Write these lines into the shots exactly as given, each where it is spoken:\n{spoken}")
    if any(wrap):
        needs.append("Do not repeat what the HEADER or FOOTER gives; write it here only where this scene differs.")
    needs.append("Write one prompt only, no commentary and no second version.")
    prompt = (
        f"Scene {index + 1} of {count}.\n"
        f"The look: {style}.\n"
        + (f"The scene before: {tagged(before)}\n" if before else "")
        + (f"The scene after: {tagged(after)}\n" if after else "")
        + f"What happens: {tagged(scene['action'] or scene['summary'])}\n"
        f"Camera: {tagged(scene['camera']) or 'as the action needs'}\n"
        f"Sound: {scene['sound'] or 'N/A'}\nMusic: {scene['music']}\n\n"
        f"{subjects}\n\n"
        f"The HEADER placed before this scene's prompt:\n{wrap[0] or '(none)'}\n"
        f"The FOOTER placed after it:\n{wrap[1] or '(none)'}\n\n"
        "Requirements:\n" + "\n".join(f"{number}. {need}" for number, need in enumerate(needs, start=1))
    )
    limits = {"Subject": len(ids), "Picture": len(ids)}
    asked = prompt
    for attempt in range(SCENE_ATTEMPTS):
        reply = prompt_of(write(asked, max_length=1800, temperature=None if attempt == 0 else 0.4))
        broken, loose = scene_faults(reply, limits, bool(ids), bool(scene.get("lines")), sections)
        if not broken and (not loose or attempt == SCENE_ATTEMPTS - 1):
            return stated(reply, scene["seconds"])
        asked = (f"{prompt}\n\nThe last reply had these faults: {'; '.join(broken + loose)}. Write the whole "
                 "prompt again with them fixed.")
    fields = written_out(scene, cast, tags, speakers)
    return stated(reference_prompt(fields, [cast[key] for key in ids], style) if ids
                  else base_prompt(fields, style), scene["seconds"])


def written_out(scene: dict, cast: dict[str, Member], tags: dict[str, str],
                speakers: dict[str, str]) -> dict:
    """A scene's fields written from its outline alone.

    Args:
        scene: The scene's outline.
        cast: Every member, by id.
        tags: The scene's members, id to ``<Subject N>``.
        speakers: Cast id to speaker id.

    Returns:
        ``{summary, shots, soundscape, music}``.
    """
    seen = " ".join(f"{tags[key]}, {cast[key].description}." for key in tags)
    lines = []
    for line in scene.get("lines", []):
        who = str(line.get("who") or "").strip()
        named = tags.get(who) or (cast[who].name if who in cast else who) or "A voice off-screen"
        voice = cast[who].voice if who in cast and cast[who].voice else "a clear voice"
        speaker = f" ({speakers[who]})" if who in speakers else ""
        lines.append(
            f"{named}{speaker}, in {voice}, says, "
            f"<d>[{line.get('language') or 'English'}] {str(line['text']).strip()}</d>"
        )
    shots = " ".join(part for part in (
        "[Shot 1]", scene.get("camera", ""), seen, scene.get("action", ""), " ".join(lines),
    ) if part).strip()
    return {"summary": scene.get("summary") or scene.get("action", "") or scene["title"], "shots": shots,
            "soundscape": scene.get("sound") or "N/A", "music": scene.get("music") or "N/A"}


def definition(member: Member, number: int) -> str:
    """One subject's line of ``subject_definitions``.

    Args:
        member: The subject.
        number: Its subject and picture number.

    Returns:
        The line, as ``<Subject 2> is the lamp-room environment in <Picture 2>, featuring ...``.
    """
    tag, picture = f"<Subject {number}>", f"<Picture {number}>"
    look = member.description[:1].lower() + member.description[1:] if member.description else member.name
    if member.kind == "location":
        return f"{tag} is the {member.name} environment in {picture}, featuring {look}."
    if member.kind == "object":
        return f"{tag} is the {member.name} in {picture}, {look}."
    return f"{tag} is {member.name}, {look}, in {picture}."


def tidy(text: str) -> str:
    """A header or footer without the sections it leaves empty.

    Args:
        text: The header or footer.

    Returns:
        The text with every section line that carries nothing removed.
    """
    kept = []
    text = "\n".join(RUN_ON_SECTION.sub("\n", line) for line in str(text or "").strip().splitlines())
    for block in re.split(r"\n(?=[a-z_]+:)", text):
        name, _, body = block.partition(":")
        if re.fullmatch(r"[a-z_]+", name.strip()) and not body.strip():
            continue
        kept.append(block.strip())
    return "\n\n".join(part for part in kept if part)


def reference_prompt(fields: dict, members: list[Member], style: str) -> str:
    """A scene in the reference-to-video template, its subjects numbered in picture order.

    Args:
        fields: ``{summary, shots, soundscape, music}``.
        members: The scene's members, in picture order.
        style: The look of the video.

    Returns:
        The prompt.
    """
    definitions, retention = [], []
    for number, member in enumerate(members, start=1):
        definitions.append(definition(member, number))
        retention.append(f"<Subject {number}> (appears in [Shot 1]): fully_preserved - {member.keep} are retained.")
    return "\n\n".join([
        "subject_definitions:\n" + "\n".join(definitions),
        "summary:\n[reference generation] " + fields["summary"],
        "retention_analysis:\n" + "\n".join(retention),
        f"detailed_description:\nThe target video is {style}.\n{fields['shots']}",
        "overall_soundscape:\n" + fields["soundscape"],
        "non_diegetic_music:\n" + fields["music"],
    ])


def base_prompt(fields: dict, style: str) -> str:
    """A scene in the text-to-video template.

    Args:
        fields: ``{summary, shots, soundscape, music}``.
        style: The look of the video.

    Returns:
        The prompt.
    """
    shots = re.sub(r"^\s*\[Shot 1\]\s*", "", fields["shots"])
    return "\n\n".join([
        f"integrated_multimodal_description: [Shot 1] The video is {style}. {shots}",
        "overall_soundscape: " + fields["soundscape"],
        "non_diegetic_music: " + fields["music"],
    ])


def rewrite(write: Callable, prompt: str, directions: str = "", context: str = "") -> str:
    """A scene prompt rewritten under the system prompt, or written from directions where it is empty.

    Args:
        write: As :func:`make_writer` returns it.
        prompt: The scene's prompt, or an empty string.
        directions: What to change, or an empty string.
        context: The scenes around it and the pictures it names, or an empty string.

    Returns:
        The new prompt.

    Raises:
        ValueError: There is neither a prompt nor directions, or the model wrote nothing twice.
    """
    prompt, directions, context = (str(value or "").strip() for value in (prompt, directions, context))
    if not prompt and not directions:
        raise ValueError("there is no prompt to rewrite and no directions to write one from")
    told = directions or "none; bring it in line with the rules and fill in what they ask for"
    if prompt:
        asked = (
            "Rewrite this scene prompt so it follows the rules. Keep its story, its subjects, "
            "every <Subject N>, <Picture N>, <Video N> and <Audio N> number, and every line of "
            "dialogue word for word in its language, unless the directions change them.\n\n"
            f"Directions: {told}\n"
            + (f"\n{context}\n" if context else "")
            + f"\nThe prompt:\n<<<\n{prompt}\n>>>\n\nReply with the rewritten prompt only, no commentary."
        )
    else:
        asked = (
            "Write one scene prompt that follows the rules, from these directions.\n\n"
            f"Directions: {directions}\n"
            + (f"\n{context}\n" if context else "")
            + "\nReply with the prompt only, no commentary."
        )
    limits = {}
    for kind, number in TAG.findall(f"{prompt}\n{context}"):
        limits[kind] = max(limits.get(kind, 0), int(number))
    for attempt in range(2):
        reply = prompt_of(write(asked, max_length=2048, temperature=None if attempt == 0 else 0.4))
        if reply and balanced(reply) and numbered_within(reply, limits) and (
                "[Shot 1]" in reply or "[Shot 1]" not in prompt):
            return reply
    raise ValueError(
        "the language model did not write a usable prompt twice. Try again with another seed, or "
        "a larger model such as qwen3vl_8b_fp8_scaled.safetensors"
    )


def split_scenes(text: str) -> list[dict]:
    """Scenes from a block of prompts: JSON as the writer answers it, or prompts between dashed lines.

    Args:
        text: A JSON list of scenes or of prompts, the writer's JSON, or prompts divided by a
            line of dashes.

    Returns:
        ``[{prompt, seconds, pictures, summary}]``, ``seconds`` and ``summary`` empty where unknown.
    """
    raw = str(text or "").strip()
    found = None
    if raw[:1] in "[{":
        try:
            found = json.loads(raw)
        except json.JSONDecodeError:
            found = None
    if isinstance(found, dict):
        found = found.get("scenes")
    scenes = []
    if isinstance(found, list):
        for item in found:
            if isinstance(item, str):
                scenes.append({"prompt": item, "seconds": None, "pictures": 0, "summary": ""})
            elif isinstance(item, dict):
                scenes.append({
                    "prompt": str(item.get("prompt") or ""),
                    "seconds": item.get("seconds"),
                    "pictures": int(item.get("pictures") or len(item.get("references") or [])),
                    "summary": str(item.get("summary") or ""),
                })
        return [scene for scene in scenes if scene["prompt"].strip() or scene["summary"].strip()]
    parts = re.split(r"\n\s*-{3,}\s*\n", raw)
    return [{"prompt": part.strip(), "seconds": None, "pictures": 0, "summary": ""}
            for part in parts if part.strip()]


def digest(prompt: str, most: int = 70) -> str:
    """What a scene shows, in a sentence or two, from its prompt.

    Args:
        prompt: A scene prompt.
        most: The most words to keep.

    Returns:
        The prompt's summary section where it has one, else the start of its shots.
    """
    text = str(prompt or "")
    match = re.search(r"^summary:\s*(.+?)(?:\n\s*\n|\n[a-z_]+:|\Z)", text, re.S | re.M)
    body = match.group(1) if match else text
    if not match:
        shot = body.find("[Shot 1]")
        body = body[shot + len("[Shot 1]"):] if shot >= 0 else body
        body = re.split(r"\n[a-z_]+:", body)[0]
    body = re.sub(r"\[reference generation\]|\[[a-z +]+\]", "", body)
    body = re.sub(r"<d>\[[^\]]*\]\s*", "\"", body).replace("</d>", "\"")
    words = " ".join(body.split()).split(" ")
    return " ".join(words[:most]) + ("..." if len(words) > most else "")


def ends_speaking(prompt: str) -> bool:
    """Whether a scene's last shot still has someone talking.

    Args:
        prompt: A scene prompt.

    Returns:
        True where a line of dialogue falls in the last fifth of the prompt's shots.
    """
    text = str(prompt or "")
    shots = text.find("[Shot 1]")
    end = max(text.find("overall_soundscape"), 0) or len(text)
    body = text[shots if shots >= 0 else 0:end]
    last = body.rfind("</d>")
    return last >= 0 and last > len(body) * 0.8


#: A line of a plan, as ``7: cast and sound | the talk goes on``.
PLAN_LINE = re.compile(r"^\W*(?:scene\s*)?(\d+)[*\s]*[:.)\-][*\s<]*(cast and sound|cast|sound|carry|cut)\b\W*(.*)$",
                       re.I | re.M)


def plan(write: Callable, scenes: list[dict]) -> list[dict]:
    """How each scene follows the one before, as the model reads the scenes.

    Args:
        write: As :func:`make_writer` returns it.
        scenes: ``[{prompt, seconds, pictures, summary}]``.

    Returns:
        One ``{scene, transition, continuity, overlap, why}`` per scene from the second, scenes
        counted from 1; ``cut`` wherever the model gives nothing usable.
    """
    if len(scenes) < 2:
        return []
    listed = []
    for number, scene in enumerate(scenes, start=1):
        facts = []
        if scene.get("seconds"):
            facts.append(f"{float(scene['seconds']):.1f}s")
        facts.append(f"{scene.get('pictures') or 0} reference picture(s)")
        if ends_speaking(scene.get("prompt", "")):
            facts.append("ends mid-dialogue")
        listed.append(f"{number}. [{', '.join(facts)}] {scene.get('summary') or digest(scene.get('prompt', ''))}")
    choices = "\n".join(f"- {name}: {note}" for name, note in TRANSITION_NOTES.items())
    prompt = (
        f"You are editing a video of {len(scenes)} scenes. For every scene after the first, choose "
        "how it follows the scene before it.\n\n"
        f"The transitions:\n{choices}\n\n"
        "How to choose:\n"
        "- carry only where the scene is the same shot going on: same place, same people, no jump "
        "in time, and the camera has no reason to cut.\n"
        "- sound where a conversation, a song or a continuous moment runs on in the same place "
        "from another angle, or a scene before ends mid-dialogue.\n"
        "- cast and cast and sound only for a scene with no reference pictures of its own, where "
        "the same people go on in the same place; a scene with reference pictures keeps its cast "
        "through them.\n"
        "- cut for a new place, a jump in time, a new sequence, or credits.\n\n"
        "The scenes:\n" + "\n".join(listed) + "\n\n"
    )
    chosen = {}
    for attempt in range(3):
        missing = [number for number in range(2, len(scenes) + 1) if number not in chosen]
        if not missing:
            break
        asked = prompt + (
            f"Answer with one line for each of scenes {', '.join(str(number) for number in missing)}, "
            "in order and nothing else, each as the scene number, a colon, the transition, a bar and "
            f"a short reason drawn from those two scenes:\n{missing[0]}: <transition> | <reason>"
        )
        reply = clean_reply(write(asked, max_length=40 * len(missing) + 120,
                                  temperature=0.3 if attempt == 0 else 0.2, ruled=False))
        for number, name, why in PLAN_LINE.findall(reply):
            number = int(number)
            if number in missing and number not in chosen:
                chosen[number] = (name.lower(), why.lstrip("|-: ").strip())
    planned = []
    for number in range(2, len(scenes) + 1):
        name, why = chosen.get(number, ("cut", ""))
        if scenes[number - 1].get("pictures") and name in CAST_REFERENCES:
            name = CAST_REFERENCES[name]
        continuity, overlap = TRANSITIONS[name]
        planned.append({"scene": number, "transition": name, "continuity": continuity,
                        "overlap": overlap, "why": why})
    return planned
