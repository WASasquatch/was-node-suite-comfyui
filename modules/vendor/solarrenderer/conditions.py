"""Fixed sentences per condition type, and the assembly of a SolarRenderer prompt.

A prompt has six parts, in this order:
    subject_definitions, summary, retention_analysis, detailed_description, overall_soundscape, non_diegetic_music
The parts that say what each input is and how far it is followed are fixed per condition type (this file). The free
parts - what the finished video looks like - are written per clip (free_caption.py, or by hand with SKILL.md).

The sentences of the six trained condition types are the ones training used, word for word. Any other structure
(a sketch, a point cloud, a wireframe ...) is described with `custom_condition`, in the same form.
"""
import re

# condition type -> (what <V> is, what of it is followed / not given / not shown); {v} is the label
DEFINE = {
    "scene_graybox": ("an untextured grey block-out of the scene, rendered from the same camera as the target video, "
                      "giving the layout, the large geometry, the characters' bodies, their motion and the camera of every frame",
                      "(frame-by-frame layout, geometry, motion and camera): attribute_transfer - the layout, large shapes, "
                      "character bodies, motion and camera are followed exactly; small objects, materials, colours and "
                      "lighting are not given by {v} and are the target video's own; the grey look of {v} is not shown"),
    "semantic": ("a semantic segmentation of the scene, one flat colour per class (road, sidewalk, building, vegetation, "
                 "vehicle, pedestrian, sky and so on), frame by frame, from the same camera as the target video",
                 "(frame-by-frame layout of every class and camera): attribute_transfer - every region keeps its class, "
                 "outline and place; materials, colours and lighting are not given by {v}; the flat class colours are "
                 "not shown"),
    "scene_whitebox": ("an untextured grey rendering of the scene and the main character, from the same camera "
                       "as the target video, giving the layout of the scene, the character's pose and motion, and the "
                       "camera of every frame",
                       "(frame-by-frame layout, character pose and motion, camera): attribute_transfer - the layout, the "
                       "character's pose and motion and the camera are followed; materials, colours, lighting and any "
                       "added detail are the target video's own; the grey look of {v} is not shown"),
    "mask_skeleton": ("the objects of the scene as flat-coloured masks, one colour per object, with the skeletons of the "
                      "characters drawn on top, frame by frame, from the same camera as the target video",
                      "(frame-by-frame object outlines, character poses and camera): attribute_transfer - every object "
                      "keeps its outline and place, every character keeps its pose and motion; materials, colours and "
                      "lighting are not given by {v}; the flat mask colours and skeleton lines are not shown"),
    "smpl_mesh": ("a grey 3D body mesh of the main character on a plain white background, rendered from the same camera "
                  "as the target video; it gives the character's position, size, body pose and motion in every frame, "
                  "and nothing of the surroundings",
                  "(frame-by-frame body pose and motion of the main character): attribute_transfer - the main character's "
                  "position, size, pose and motion are followed; the character's clothing, face and hair, the surroundings, "
                  "the camera's view of the scene, materials, colours and lighting are not given by {v} and are the target "
                  "video's own; the grey mesh and the white background of {v} are not shown"),
    "near_depth": ("a grey shaded rendering of the game engine's depth within about 30 metres of the camera, from the "
                   "same camera as the target video; everything farther, and the sky, is black. It gives the exact "
                   "shapes, positions and motion of the nearby scene, characters and vehicles, and the camera of every "
                   "frame",
                   "(frame-by-frame near geometry, motion and camera): attribute_transfer - the shapes, positions and "
                   "motion of everything within the near range and the camera are followed; the far scene, materials, "
                   "colours and lighting are not given by {v} and are the target video's own; the grey shading and the "
                   "black background of {v} are not shown"),
}

INPUTS = ("full", "structure", "first", "text")   # structure + first frame / structure only / first frame only / text only


def custom_condition(looks_like: str, not_shown: str):
    """Fixed sentences for a structure the model was not trained on, in the form the benchmark uses, e.g.
    custom_condition("a black pencil line sketch of the scene on white paper, outlines only, like an animated storyboard",
                     "the pencil lines and white paper")"""
    return (f"{looks_like}, from the same camera as the target video",
            "(frame-by-frame layout, shapes, character poses and motion, camera): attribute_transfer - the layout, "
            "shapes, poses, motion and camera are followed exactly; materials, colours and lighting are not given by "
            "{v} and are the target video's own; " + f"{not_shown} of {{v}} is not shown")


def assemble(free: dict, inputs: str = "full", condition="scene_graybox") -> str:
    """The full prompt from the free parts (summary, detailed_description, overall_soundscape, non_diegetic_music).

    inputs: which inputs the model is given - "full" (structure + first frame), "structure", "first" or "text".
    condition: a key of DEFINE, or a (what, kept) pair from custom_condition. Labels follow the order of the inputs:
    the structure clip is <Video 1>; the first frame is <Video 2>, or <Video 1> without a structure clip.
    """
    assert inputs in INPUTS, inputs
    detail = free["detailed_description"].strip()
    sound = (free.get("overall_soundscape") or "N/A").strip() or "N/A"
    music = (free.get("non_diegetic_music") or "N/A").strip() or "N/A"
    summary = free["summary"].strip()
    if inputs == "text":  # text only: the three core fields, the style inside [Shot 1]
        body = "[Shot 1] " + re.sub(r"\[Shot 1\]\s*", "", detail, count=1)
        return f"integrated_multimodal_description: {body}\n\noverall_soundscape: {sound}\n\nnon_diegetic_music: {music}"
    what, kept = DEFINE[condition] if isinstance(condition, str) else condition
    lines, keep, tasks = [], [], []
    structure = inputs in ("full", "structure")
    first = inputs in ("full", "first")
    s_lab = "<Video 1>" if structure else None
    f_lab = ("<Video 2>" if structure else "<Video 1>") if first else None
    if structure:
        lines.append(f"{s_lab} is {what}.")
        keep.append(f"{s_lab} {kept.format(v=s_lab)}.")
        tasks.append("reference generation")
    if first:
        lines.append(f"{f_lab} is the first frame of [Shot 1], held still.")
        keep.append(f"{f_lab} ([Shot 1] first frame): fully_preserved - the composition, subjects, materials, colours "
                    f"and lighting of {f_lab} are the target video's first frame.")
        tasks.insert(0, "keyframe completion")
    lead = []
    if first:
        lead.append(f"The target video begins from {f_lab}")
    if structure:
        lead.append(f"follows the frame-by-frame structure and motion of {s_lab}" if first else
                    f"The target video follows the frame-by-frame structure and motion of {s_lab}")
    head = (" and ".join(lead) + ". ") if lead else ""
    if first:  # the opening shot starts from the first frame
        detail = re.sub(r"\[Shot 1\]\s*", f"[Shot 1] The shot begins from {f_lab}. ", detail, count=1) \
            if "[Shot 1]" in detail else f"[Shot 1] The shot begins from {f_lab}. " + detail
    return ("subject_definitions:\n" + "\n".join(lines) + "\n\n"
            f"summary:\n[{' + '.join(tasks)}] {head}{summary}\n\n"
            "retention_analysis:\n" + "\n".join(keep) + "\n\n"
            f"detailed_description:\n{detail}\n\n"
            f"overall_soundscape:\n{sound}\n\nnon_diegetic_music:\n{music}")
