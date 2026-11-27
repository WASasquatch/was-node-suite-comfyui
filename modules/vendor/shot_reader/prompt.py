"""Shot Reader v2's prompt, schema note and frame-geometry line, taken verbatim from upstream."""

import math

N_FRAMES = 16

PARAMS = {"T": 0.6564, "none_cut": 1.10}

PROMPT = """You are a previz shot reader. You see {nf} frames sampled evenly from one shot of {dur:.2f} s
(frame i is at t = {times}).

Write the shot's fact sheet: one JSON object exactly in the schema below. Measure positions in metres
in the ground frame defined below (origin under the camera in the FIRST frame, +x right, +y forward).
Use object sizes (people are about 1.7 m tall, doors 2 m, chairs 0.9 m, etc.) and perspective to
estimate distances. Give every listed subject a path with at least a key at t=0 and t={dur:.2f}
(add intermediate keys when it moves). Keep the given ids.

Subjects in the brief (id: class): {subjects}

{schema}

Reply with only the JSON object, no prose, no code fences."""

V3R_NOTE = """Fact sheet schema v3. Top-level keys in this order: schema_version (3), shot, camera, style, tags, caption, n_subjects, subjects, n_events, events, environment, n_set, set.
camera keys in this order: height_m, angle, pitch_deg, fov_deg, move, zoom, handheld, framing, subject, ambiguous.
camera.angle: low | eye | high | overhead
camera.move (the camera body only, never the lens): static | pan | tilt | push_in | pull_out | truck | crane | orbit | follow
camera.framing: wide | full | medium | close
camera.zoom.dir: in | out | none
style: live_action | cg_3d | anim_2d
events (typed only): jump | crouch | stand_up | sit | lie | fall | turn"""


def build_prompt(dur, times, subjects, geo_line):
    subs = ", ".join(subjects) if subjects else "none given (name people P01, P02, ... and objects by class)"
    return PROMPT.format(nf=N_FRAMES, dur=dur, times=", ".join(f"{t:.2f}" for t in times), subjects=subs,
                         schema=V3R_NOTE + "\n\n" + geo_line)


def _x(s):
    return f"x{s:.2f}"


def hint_line(g, P=PARAMS):
    """compact measurement -> the one prompt line"""
    head = "Frame geometry (point tracks, frame 0 to 15): "
    if not g or g.get("error") or not g.get("s_global"):
        return head + "not measurable on these frames."
    sc, T = g.get("score"), P["T"]
    if sc is None or g.get("align_failed") or math.isinf(sc):   # the scale is then only an image-search guess
        return head + "no pure zoom aligns the first and last frames (the camera body or the scene moves)."
    sg = g["s_global"]
    parts = [f"scale {_x(sg)}"]
    if g.get("s_centre"):
        parts.append(f"centre {_x(g['s_centre'])}")
    if g.get("s_periph"):
        parts.append(f"edges {_x(g['s_periph'])}")
    if sc <= T:
        ls = math.log(sg)
        z = "no zoom" if abs(ls) < math.log(P["none_cut"]) else ("zoom in" if ls > 0 else "zoom out")
        tail = f"residual after the best pure zoom {sc:.2f} px, within the {T:.2f} px a lens-only zoom leaves (lens-only: {z})."
    else:
        shown = f"{sc:.2f}" if sc < 100 else ">100"
        tail = (f"residual after the best pure zoom {shown} px, above the {T:.2f} px a lens-only zoom leaves "
                f"(the camera body or the scene moves).")
    return head + ", ".join(parts) + "; " + tail
