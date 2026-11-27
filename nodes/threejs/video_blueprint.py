"""Measure a video shot by shot into a request a language model turns into a blockout scene."""

from __future__ import annotations

import json

import torch
from comfy_api.latest import io

from ...modules.compat.types import LLM
from ...modules.io import picker
from ...modules.model import anysplat, model_directories
from ...modules.threejs import blueprint

NODE_NAME = "Video Blueprint"

#: Cut threshold and shortest shot, as Video Split Scenes reads them.
DEFAULT_THRESHOLD = 0.4

#: Shots measured at most, as many as Three Blockout Scene takes.
MAX_SHOTS = 24

#: Longest side the pictures are handed over at, in pixels.
PICTURE_SIDE = 768

#: How the shots' cameras are solved: together, or each on its own.
SOLVES = ("each shot", "whole clip")

#: Pose model the people are found with.
POSE_MODEL = "ViTPose Base"


class VideoBlueprint(io.ComfyNode):
    """Measure a clip's cuts, camera paths and sound hits for the Three Blockout Scene Writer."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASVideoBlueprint",
            display_name=NODE_NAME,
            search_aliases=[
                "WASVideoBlueprint", "Video Blueprint", "shot reader", "video to blockout", "previs from footage",
                "camera solve", "reshoot", "shot breakdown",
            ],
            category="WAS Suite/Three",
            description=(
                "Measure a video into a blueprint for Three Blockout Scene: where it cuts, each shot's camera path "
                "and lens, and its sound hits. Hand the brief and pictures to a language model with the Three "
                "Blockout Scene Writer skill; reshoot_directions asks it to change parts of the shot."
            ),
            inputs=[
                io.Video.Input("video", tooltip="The footage to blueprint, with its sound."),
                picker.path_input(io.Combo.Input(
                    "camera_model",
                    options=anysplat.offered(),
                    tooltip="The AnySplat checkpoint that solves each shot's camera, from "
                            "ComfyUI/models/geometry_estimation; `anysplat.safetensors` is fetched on first use "
                            "with features.network on.",
                ), model=True),
                io.String.Input(
                    "reshoot_directions",
                    multiline=True,
                    default="",
                    tooltip="What to change from the original, in plain words: `make it night`, `turn shot 2 into "
                            "an orbit`, `the man becomes a woman in a red coat`. Empty = recreate it as filmed.",
                ),
                io.Float.Input(
                    "cut_threshold", default=DEFAULT_THRESHOLD, min=0.05, max=0.95, step=0.01,
                    tooltip="Share of the picture that must still follow from the frame before for the shot to "
                            "continue: 0.4 = default; 0.6 = finds softer cuts; 0.2 = only hard cuts.",
                ),
                io.Int.Input(
                    "min_shot_frames", default=6, min=1, max=10000,
                    tooltip="Fewest frames a shot holds: 6 = default; 24 = one second at 24 fps.",
                ),
                io.Int.Input(
                    "frames_per_shot", default=12, min=2, max=anysplat.MAX_PICTURES,
                    tooltip="Frames each shot's camera is solved from, evenly spaced, first and last kept: 12 = "
                            "default; 4 = faster and coarser; 32 = a finer path for long moves.",
                ),
                io.Int.Input(
                    "pictures_per_shot", default=1, min=1, max=3,
                    tooltip="Pictures of each shot handed to the language model: 1 = its middle frame; 3 = its "
                            "first, middle and last.",
                ),
                io.Combo.Input(
                    "solve", options=list(SOLVES), default=SOLVES[0],
                    tooltip="`each shot` = every shot solved on its own and joined to the one before by the "
                            "people seen on both sides of the cut; `whole clip` = every shot's frames solved "
                            "together, for shots that overlap heavily.",
                ),
                io.Boolean.Input(
                    "find_people", default=True,
                    tooltip="On = people and objects such as benches, cars and chairs are found and measured, and "
                            "the people's size sets the scale and the camera's height; off = the language model "
                            "places everything, the camera taken at eye level.",
                ),
                io.Float.Input(
                    "person_height", default=1.7, min=0.8, max=2.2, step=0.05,
                    tooltip="How tall the people in the clip stand, in metres, which sets the scale: 1.7 = adults; "
                            "1.2 = young children; 1.0 = toddlers.",
                ),
                io.Boolean.Input(
                    "environment_context", default=False,
                    tooltip="`true` = the brief also asks the language model to describe the set in words: every "
                            "object and surface, where each stands from the people and each other, and how the people "
                            "use them; `false` = the set is only placed as blocks.",
                ),
                LLM.Input(
                    "listener",
                    optional=True,
                    tooltip="A model that hears, from LLM Loader, such as Qwen2.5-Omni-7B-Q8_0.gguf with its mmproj: "
                            "it names each sound hit and writes down what is said in each shot. Unwired = hits "
                            "are given as times only.",
                ),
                LLM.Input(
                    "shot_reader",
                    optional=True,
                    tooltip="Shot Reader v2 from LLM Loader, such as shot-reader-v2-MTP.Q8_0.gguf with its mmproj: "
                            "its reading of each shot's move, zoom, angle, height, set and tags joins the brief, and "
                            "its camera height is used where no people set the scale. Unwired = the measurement "
                            "alone.",
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="brief",
                    tooltip="The request for the language model: every shot's length, its camera keys and move "
                            "in words, the sound hits and the reshoot directions.",
                ),
                io.Image.Output(
                    display_name="pictures",
                    tooltip="The pictures the brief names, shot by shot, at most 768 pixels on the long side, for "
                            "the language model's image input.",
                ),
                io.Audio.Output(
                    display_name="audio",
                    tooltip="The clip's own sound, silence where it has none, for a model that hears or to lay "
                            "under the film.",
                ),
                io.String.Output(
                    display_name="blueprint",
                    tooltip="The measurements as JSON: each shot's start, seconds, camera keys, move and "
                            "handheld flag, and the sound hits in seconds.",
                ),
                io.Int.Output(display_name="shot_count", tooltip="How many shots were found, 1 or more."),
            ],
        )

    @classmethod
    def execute(cls, video, camera_model, reshoot_directions, cut_threshold, min_shot_frames, frames_per_shot,
                pictures_per_shot, solve=SOLVES[0], find_people=True, person_height=1.7,
                environment_context=False, listener=None, shot_reader=None) -> io.NodeOutput:
        """Measure the clip.

        Raises:
            ValueError: The clip holds fewer than two frames.
            ModelUnavailable: The camera checkpoint is not on disk and could not be fetched.
        """
        import comfy.model_management
        import torch.nn.functional as F

        from ...modules import log
        from ...modules.interface import preview as published
        from ...modules.interface import run_result
        from ...modules.media import clip as clips
        from ...modules.media import scenes

        source = clips.open_clip(video, NODE_NAME)
        frames = source.frames
        count = int(frames.shape[0])
        if count < 2:
            raise ValueError(f"{NODE_NAME} needs a clip of two frames or more, and this one holds {count}.")
        rate = float(source.rate)
        device = comfy.model_management.get_torch_device()
        motion = clips.motion_for(source, None, NODE_NAME, device, clips.progress(count - 1))
        first_frames = scenes.starts(motion.cuts(float(cut_threshold)), int(min_shot_frames))
        spans = scenes.ranges(first_frames, count)[:MAX_SHOTS]
        together = solve == "whole clip" and len(spans) > 1
        each = int(frames_per_shot)
        if together:
            each = max(2, min(each, anysplat.MAX_PICTURES // len(spans)))
        picks = [torch.linspace(a, b - 1, min(each, b - a)).round().long().unique() for a, b in spans]

        pair = None
        if find_people:
            try:
                from ...modules.model import pose

                pair = pose.load(POSE_MODEL)
            except Exception as error:
                log.get_logger(__name__).info("people are not found: %s", error)

        model_directories(anysplat.FOLDER)
        model = picker.model_name(camera_model, (anysplat.FOLDER,))
        groups = [list(range(len(spans)))] if together else [[i] for i in range(len(spans))]
        shots = [blueprint.Shot(index=i + 1, start=round(a / rate, 3), seconds=round((b - a) / rate, 3))
                 for i, (a, b) in enumerate(spans)]
        people = []
        parts = [{"start": a / rate, "keys": [], "tracks": [], "objects": []} for a, _ in spans]
        things = []
        for group in groups:
            solvable = [i for i in group if picks[i].numel() >= 2]
            if not solvable:
                continue
            chosen = torch.cat([picks[i] for i in solvable])
            solved = anysplat.reconstruct(model, frames[chosen], 448)
            empty = [[] for _ in solved.cameras]
            found, boxes = blueprint.find_people(pair, solved.pictures) if pair is not None else (empty, empty)
            seen, standing, sighted = [], [], []
            for n, (index, people_here) in enumerate(zip(chosen.tolist(), found)):
                shot = next(i for i in solvable if spans[i][0] <= index < spans[i][1])
                world = []
                for person in people_here:
                    foot = blueprint.at_pixel(solved.points[n], person["foot"])
                    nose = blueprint.at_pixel(solved.points[n], person["nose"]) if person["nose"] else None
                    left = blueprint.at_pixel(solved.points[n], person["left"]) if person["left"] else None
                    right = blueprint.at_pixel(solved.points[n], person["right"]) if person["right"] else None
                    world.append({"foot": foot, "left": left, "right": right, "x": person["x"]})
                    standing.append((foot, nose))
                seen.append((index / rate, shot + 1, world))
                sighted.extend((n, shot + 1, item["label"], item["box"]) for item in boxes[n])
            frame = blueprint.frame_for(solved.cameras, solved.gaussians.means, standing, float(person_height))
            measured_things = blueprint.objects(
                [(shot, label, blueprint.measure_object(frame, solved.points[n], box)) for n, shot, label, box in sighted])
            offset = 0
            for i in solvable:
                taken = picks[i].numel()
                times = [(int(p) - spans[i][0]) / rate for p in picks[i]]
                shots[i].keys = blueprint.keys_for(frame, solved.cameras[offset:offset + taken], times)
                shots[i].words, shots[i].handheld = blueprint.describe(shots[i].keys)
                shots[i].solved = True
                shots[i].scaled = frame.measured
                offset += taken
            followed = blueprint.tracks(frame, seen)
            if together:
                people.extend(followed)
                things.extend(measured_things)
            else:
                parts[group[0]]["tracks"] = followed
                parts[group[0]]["objects"] = measured_things
            del solved
        for shot in shots:
            if not shot.keys:
                shot.keys = [{"at": 0.0, "position": [0.0, blueprint.EYE_HEIGHT, blueprint.SUBJECT_DISTANCE],
                              "look_at": [0.0, blueprint.EYE_HEIGHT, 0.0], "lens": 35}]
        if not together:
            for part, shot in zip(parts, shots):
                part["keys"] = shot.keys
            joined, people, things = blueprint.stitch(parts)
            for shot, keys in zip(shots, joined):
                shot.keys = keys
        things = blueprint.tidied(things)

        pictures = []
        for start, stop in spans:
            length = stop - start
            chosen = [start + length // 2] if int(pictures_per_shot) == 1 else \
                torch.linspace(start, stop - 1, int(pictures_per_shot)).round().long().tolist()
            pictures.extend(frames[int(i)] for i in chosen)
        batch = torch.stack(pictures).float()
        side = max(int(batch.shape[1]), int(batch.shape[2]))
        if side > PICTURE_SIDE:
            scale = PICTURE_SIDE / side
            size = (max(1, round(batch.shape[1] * scale)), max(1, round(batch.shape[2] * scale)))
            batch = F.interpolate(batch.permute(0, 3, 1, 2), size=size, mode="area").permute(0, 2, 3, 1)
        batch = batch[..., :3].clamp(0.0, 1.0).contiguous()

        audio = source.audio
        hits = blueprint.sound_hits(audio)
        if not audio or audio.get("waveform") is None:
            audio = {"waveform": torch.zeros(1, 1, max(1, int(round(count / rate * 44100)))), "sample_rate": 44100}

        if shot_reader is not None:
            from ...modules.llm import runtime, wires

            model = wires.expect(shot_reader, wires.LanguageModel, NODE_NAME, "shot_reader",
                                 "Connect LLM Loader with Shot Reader v2 and its mmproj.")
            wires.ready(model)
            with runtime.guarded():
                blueprint.read_shots(model, frames, rate, spans, shots, NODE_NAME)
            for shot in shots:
                blueprint.apply_sheet(shot)
                shot.words = blueprint.describe(shot.keys)[0]

        heard = None
        if listener is not None and source.audio and source.audio.get("waveform") is not None:
            from ...modules.llm import runtime, wires

            model = wires.expect(listener, wires.LanguageModel, NODE_NAME, "listener",
                                 "Connect LLM Loader with a model that hears, such as Qwen2.5-Omni.")
            wires.ready(model)
            with runtime.guarded():
                heard = blueprint.listen(model, source.audio, hits,
                                         [(s.start, s.start + s.seconds) for s in shots], NODE_NAME)
        text = blueprint.brief(shots, hits, int(pictures_per_shot), str(reshoot_directions or ""), people, things,
                               bool(environment_context), heard)
        measured = {
            "seconds": round(count / rate, 3),
            "shots": [{"start": s.start, "seconds": s.seconds, "camera": s.words, "handheld": s.handheld,
                       "measured": s.solved, "keys": s.keys, "shot_reader": s.sheet} for s in shots],
            "people": people,
            "objects": things,
            "sound_hits": [{"at": t, "sound": (heard[0].get(t) if heard else None)} for t, _ in hits],
            "spoken": {str(i + 1): w for i, w in (heard[1].items() if heard else [])},
        }
        published.publish_output(scenes.contact_sheet(frames, [a for a, _ in spans]))
        lines = [f"{s.index}. {s.start:.2f}-{s.start + s.seconds:.2f} s  {s.words}"
                 + (", handheld" if s.handheld else "") for s in shots]
        lines += [blueprint.describe_track(track) for track in people]
        lines += [blueprint.describe_object(item) for item in things]
        run_result.publish(
            summary=f"{len(shots)} shot{'s' if len(shots) != 1 else ''} in {count / rate:.1f} s",
            counts={"shots": len(shots), "people": len(people), "objects": len(things), "sound hits": len(hits),
                    "pictures": int(batch.shape[0])},
            bodies=run_result.body("shots", "\n".join(lines)),
        )
        return io.NodeOutput(text, batch, audio, json.dumps(measured, separators=(", ", ": ")), len(shots))
