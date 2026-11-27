"""A video measured shot by shot for a blockout: its cuts, camera paths, lenses and sound hits.

Keys are metres, ``y`` up, each camera starting on ``+z`` looking toward its subject at the origin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

#: Camera height a shot is scaled to when the ground under it is found, in metres.
EYE_HEIGHT = 1.6

#: Subject distance a shot is scaled to when no ground is found, in metres.
SUBJECT_DISTANCE = 5.0

#: Lens a field of view across the frame's height is read as, ``mm = SENSOR / tan(fov / 2)``.
SENSOR = 12.0

#: Sound hits listed at most, the gap kept between two in seconds, and the softest kept against the loudest.
MAX_HITS = 8
HIT_GAP = 0.25
HIT_FLOOR = 0.3

#: Seconds at the start of the sound where a hit is the sound starting, not an event.
SOUND_LEAD = 0.1

#: Shot Reader's input size, height and width; the tokens its sheet takes at most; the camera travel, in metres,
#: under which a lens change is read as a zoom alone, and the residual reported for one.
READER_SIZE = (256, 448)
READER_TOKENS = 2048
LENS_ONLY_TRAVEL = 0.25
LENS_ONLY_RESIDUAL = 0.2

#: Seconds of sound played around a hit to name it, and the tokens each answer takes at most.
HIT_BEFORE = 0.25
HIT_AFTER = 0.75
HIT_TOKENS = 24
SPEECH_TOKENS = 160

#: Fastest a camera is read as moving, in metres a second; a faster solve is scaled down to it.
MAX_SPEED = 12.0

#: Farthest a shot's camera travels from where it starts, in subject distances; a longer solve is scaled to it.
MAX_REACH = 2.0

#: A standing person's nose height against their full height, which sets the scale where people are found.
NOSE_SHARE = 0.93

#: Camera heights read, in metres; a measured height outside them is held to them.
MIN_HEIGHT = 0.15
MAX_HEIGHT = 80.0

#: How sure a joint must be to be used, and the COCO joints read: nose, eyes, shoulders, ankles.
JOINT_FLOOR = 0.3
NOSE = 0
EYES = (1, 2)
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
ANKLES = (15, 16)

#: Metres apart two sightings of one kind of object in a shot may be and still be the same object, and how far
#: across a cut, nearest first.
OBJECT_GATE = 3.0
OBJECT_REJOIN = 6.0

#: Metres within which objects of different kinds are one thing seen two ways, and the objects kept at most.
OBJECT_OVERLAP = 2.0
MAX_OBJECTS = 8

#: A person seen again within this many metres, plus this many metres a second since, is the same person.
TRACK_GATE = 1.2
TRACK_SPEED = 5.0

#: Seconds before a cut a person must still be seen to be joined across it, and how much a mismatched heading and
#: a turn of the set count against a pairing, in metres per unit.
CUT_HOLD = 0.6
TURN_WEIGHT = 0.5


@dataclass
class Shot:
    """One measured shot.

    Attributes:
        index: Shot number, from 1.
        start: Seconds into the clip it starts at.
        seconds: How long it runs.
        keys: Camera keys, each ``{"at", "position", "look_at", "lens"}``, ``at`` in seconds into the shot.
        words: The camera's move in words.
        handheld: Whether the path wobbles like a held camera.
        solved: Whether the camera was measured; a shot too short to measure holds one key.
        scaled: Whether people set the shot's scale and camera height.
        sheet: Shot Reader's fact sheet for the shot, or ``None``.
    """

    index: int
    start: float
    seconds: float
    keys: list = field(default_factory=list)
    words: str = "the camera holds still"
    handheld: bool = False
    solved: bool = False
    scaled: bool = False
    sheet: dict | None = None


def lens_mm(fov: float) -> float:
    """Millimetres for a field of view across the frame's height, in degrees, held to 12 to 300."""
    return round(min(300.0, max(12.0, SENSOR / math.tan(math.radians(max(1.0, min(170.0, fov))) / 2.0))), 1)


def _unit(v):
    n = float(v.norm())
    return v / n if n > 1e-12 else v


@dataclass
class Frame:
    """The blockout frame a reconstruction is read in.

    Attributes:
        origin: The first camera's world position.
        right: World direction of the blockout's ``+x``.
        up: World direction of ``+y``.
        forward: World direction of ``-z``, the first camera's horizontal view.
        scale: Metres per world unit.
        height: The first camera's height above the ground, in metres.
        distance: The first camera's distance to its subject, in metres.
        measured: Whether people set the scale, ground and distance.
        sign: ``1`` where a camera's third rotation column points the way it looks, ``-1`` where it points back.
    """

    origin: object
    right: object
    up: object
    forward: object
    scale: float
    height: float
    distance: float
    measured: bool = False
    sign: float = -1.0

    def place(self, world) -> list[float]:
        """A world point in blockout metres, the first camera at ``(0, height, distance)``."""
        rel = world.to(self.origin.dtype) - self.origin
        return [float(rel @ self.right) * self.scale, float(rel @ self.up) * self.scale + self.height,
                self.distance - float(rel @ self.forward) * self.scale]

    def direction(self, vector) -> list[float]:
        """A world direction in the blockout's axes."""
        v = vector.to(self.origin.dtype)
        return [float(v @ self.right), float(v @ self.up), -float(v @ self.forward)]


def frame_for(cameras: list[dict], points, people: list[tuple], person_height: float = 1.7) -> Frame:
    """The blockout frame for a reconstruction.

    Args:
        cameras: ``camera_info`` per picture; the first sets the origin and the view.
        points: ``(N, 3)`` reconstructed points in the same world.
        people: ``(foot, nose)`` world points of every person found in the first picture's frame or any,
            the first picture's own listed first.
        person_height: How tall the people stand, in metres; their nose sets the scale at :data:`NOSE_SHARE` of it.

    Returns:
        The frame. With nobody found the camera is taken as eye level.
    """
    import torch

    from ..world import cameras as cams

    poses = [cams.pose(c) for c in cameras]
    origin = poses[0][0]
    pts = points.to(torch.float64).reshape(-1, 3)
    rel = pts - origin
    sign = 1.0 if float(torch.median(rel @ poses[0][1][:, 2])) > 0.0 else -1.0
    up = _unit(sum(-sign * rotation[:, 1] for _, rotation in poses))
    ahead = sign * poses[0][1][:, 2]
    forward = _unit(ahead - (ahead @ up) * up)
    right = _unit(torch.linalg.cross(forward, up))
    tall = [float((nose - foot).to(torch.float64) @ up) for foot, nose in people if nose is not None]
    tall = [t for t in tall if t > 1e-6]
    if tall:
        scale = NOSE_SHARE * float(person_height) / sorted(tall)[len(tall) // 2]
        grounds = sorted(float((foot.to(torch.float64) - origin) @ up) for foot, _ in people)
        ground = grounds[len(grounds) // 2]
        height = min(MAX_HEIGHT, max(MIN_HEIGHT, -ground * scale))
        reach = sorted(float((foot.to(torch.float64) - origin) @ forward) for foot, _ in people)
        distance = min(40.0, max(1.0, reach[len(reach) // 2] * scale))
        return Frame(origin, right, up, forward, scale, height, distance, True, sign)
    seen = rel[(rel @ ahead) > 1e-6]
    distance_units = float(torch.median(seen @ ahead)) if seen.shape[0] > 64 else 1.0
    scale = SUBJECT_DISTANCE / max(distance_units, 1e-6)
    if seen.shape[0] > 64:
        ground = float(torch.quantile(seen @ up, 0.03))
        if ground < -1e-6 and 0.5 <= EYE_HEIGHT / -ground * distance_units <= 80.0:
            scale = EYE_HEIGHT / -ground
    distance = min(40.0, max(1.0, distance_units * scale))
    return Frame(origin, right, up, forward, scale, EYE_HEIGHT, distance, False, sign)


def keys_for(frame: Frame, cameras: list[dict], times: list[float]) -> list[dict]:
    """Camera keys in the blockout's frame, one per measured picture.

    Args:
        frame: The frame the reconstruction is read in.
        cameras: ``camera_info`` per picture, in time order.
        times: Seconds into the shot each picture was taken at.

    Returns:
        Keys ``{"at", "position", "look_at", "lens"}``, ``position`` and ``look_at`` in metres.
    """
    from ..world import cameras as cams

    keys = []
    for camera, at in zip(cameras, times):
        position, rotation = cams.pose(camera)
        here = frame.place(position)
        ahead = frame.direction(frame.sign * rotation[:, 2])
        look = [here[i] + ahead[i] * frame.distance for i in range(3)]
        keys.append({"at": float(at), "position": here, "look_at": look,
                     "lens": round(lens_mm(float(camera.get("fov", 40.0))))})
    speed = max((math.dist(a["position"], b["position"]) / max(b["at"] - a["at"], 1e-3)
                 for a, b in zip(keys, keys[1:])), default=0.0)
    shrink = MAX_SPEED / speed if speed > MAX_SPEED else 1.0
    start = keys[0]["position"] if keys else [0.0, 0.0, 0.0]
    reach = max((math.dist(key["position"], start) for key in keys), default=0.0) * shrink
    if not frame.measured and reach > MAX_REACH * frame.distance:
        shrink *= MAX_REACH * frame.distance / reach
    for key in keys:
        moved = [start[i] + (key["position"][i] - start[i]) * shrink for i in range(3)]
        shift = [moved[i] - key["position"][i] for i in range(3)]
        key["position"] = [round(v, 2) for v in moved]
        key["look_at"] = [round(key["look_at"][i] + shift[i], 2) for i in range(3)]
        key["at"] = round(key["at"], 2)
    return keys


def find_people(pair, pictures, threshold: float = 0.5) -> tuple[list[list[dict]], list[list[dict]]]:
    """Every person and every object the detector names in each picture, as pixel points.

    Args:
        pair: A pose detector and pose model from the pose loader.
        pictures: ``(N, h, w, 3)`` in ``[0, 1]``, as the reconstruction saw them.
        threshold: How sure the detector has to be.

    Returns:
        ``(people, objects)`` per picture. A person is ``{"foot", "nose", "left", "right", "x"}``: pixel ``(u, v)``
        points, the shoulders ``None`` where unseen, ``x`` the foot's place across the frame, 0 to 1. An object is
        ``{"label", "box"}``, the box as ``x, y, width, height`` in pixels.
    """
    import numpy as np
    import torch

    from ..image import preprocess_models as found

    detector, poser = pair.detector, pair.poser
    device = detector.load()
    poser.load()
    height, width = int(pictures.shape[1]), int(pictures.shape[2])
    names = getattr(detector.model.config, "id2label", None) or {}
    wanted = found._subject_labels(detector.model.config, pair.layout)
    people_out, objects_out = [], []
    for frame in pictures:
        picture = (frame[..., :3].float().cpu().numpy() * 255.0).astype(np.uint8)
        inputs = detector.processor(images=picture, return_tensors="pt").to(device)
        with torch.no_grad():
            raw = detector.model(**inputs)
        result = detector.processor.post_process_object_detection(
            raw, target_sizes=torch.tensor([(height, width)], device=device), threshold=float(threshold))[0]
        labels = result["labels"].cpu().tolist()
        corners = result["boxes"].cpu()
        boxes = torch.stack([corners[:, 0], corners[:, 1], corners[:, 2] - corners[:, 0],
                             corners[:, 3] - corners[:, 1]], dim=1) if corners.numel() else corners.reshape(0, 4)
        person = [i for i, label in enumerate(labels) if label in wanted]
        objects_out.append([{"label": str(names.get(label, names.get(str(label), label))), "box": boxes[i].tolist()}
                            for i, label in enumerate(labels) if label not in wanted])
        people = []
        if person:
            chosen = boxes[person]
            points, scores = found._joints(poser, picture, chosen, pair.experts, device)
            for box, joints, sure in zip(chosen.tolist(), points, scores):

                def joint(*index):
                    kept = [joints[i] for i in index if float(sure[i]) >= JOINT_FLOOR]
                    return torch.stack(kept).mean(0).tolist() if kept else None

                foot = joint(*ANKLES) or [box[0] + box[2] / 2.0, box[1] + box[3]]
                people.append({"foot": foot, "nose": joint(NOSE) or joint(*EYES), "left": joint(LEFT_SHOULDER),
                               "right": joint(RIGHT_SHOULDER), "x": foot[0] / max(width, 1)})
        people_out.append(people)
    return people_out, objects_out


def measure_object(frame, points, box) -> dict:
    """An object's place on the floor and its width and height, from its box over one picture's point map.

    Args:
        frame: The frame the reconstruction is read in.
        points: The picture's ``(h, w, 3)`` world point map.
        box: ``x, y, width, height`` in pixels.

    Returns:
        ``{"position", "width", "height"}`` in metres, ``position`` on the floor.
    """
    x, y, w, h = box
    base = frame.place(at_pixel(points, (x + w / 2.0, y + 0.92 * h)))
    left = frame.place(at_pixel(points, (x + 0.12 * w, y + 0.85 * h)))
    right = frame.place(at_pixel(points, (x + 0.88 * w, y + 0.85 * h)))
    top = frame.place(at_pixel(points, (x + w / 2.0, y + 0.08 * h)))
    return {"position": [round(base[0], 2), 0.0, round(base[2], 2)],
            "width": round(math.dist(left[::2], right[::2]) / 0.76, 2),
            "height": round(max(0.05, top[1] - base[1]) / 0.84, 2)}


def objects(seen: list[tuple]) -> list[dict]:
    """Objects seen in several pictures gathered into one each, by name and place.

    Args:
        seen: ``(shot number, label, measured)`` per sighting, ``measured`` from :func:`measure_object`.

    Returns:
        ``{"track", "label", "position", "width", "height", "shots", "seen"}`` per object seen at least twice.
    """
    gathered: list[dict] = []
    for shot, label, measured in seen:
        near = None
        for item in gathered:
            if item["label"] == label and math.dist(item["position"][::2], measured["position"][::2]) <= OBJECT_GATE:
                near = item
                break
        if near is None:
            gathered.append({"label": label, "sum": [measured["position"][0], measured["position"][2]],
                             "position": measured["position"], "width": [measured["width"]],
                             "height": [measured["height"]], "shots": [shot], "seen": 1})
            continue
        near["seen"] += 1
        near["sum"] = [near["sum"][0] + measured["position"][0], near["sum"][1] + measured["position"][2]]
        near["position"] = [round(near["sum"][0] / near["seen"], 2), 0.0, round(near["sum"][1] / near["seen"], 2)]
        near["width"].append(measured["width"])
        near["height"].append(measured["height"])
        if shot not in near["shots"]:
            near["shots"].append(shot)
    out = []
    for item in gathered:
        if item["seen"] < 2:
            continue
        item.pop("sum")
        item["width"] = sorted(item["width"])[len(item["width"]) // 2]
        item["height"] = sorted(item["height"])[len(item["height"]) // 2]
        out.append(item)
    for number, item in enumerate(out, 1):
        item["track"] = f"O{number}"
    return out


def tidied(things: list[dict]) -> list[dict]:
    """Objects of any kind standing together kept as the most seen one, the most seen :data:`MAX_OBJECTS` kept."""
    kept: list[dict] = []
    for item in sorted(things, key=lambda t: -t.get("seen", 1)):
        if any(math.dist(item["position"][::2], k["position"][::2]) <= OBJECT_OVERLAP for k in kept):
            continue
        kept.append(item)
    kept = kept[:MAX_OBJECTS]
    for number, item in enumerate(kept, 1):
        item["track"] = f"O{number}"
    return kept


def describe_object(item: dict) -> str:
    """One measured object in words for the brief."""
    shots = ", ".join(str(s) for s in item["shots"])
    return (f"{item['track']}: a {item['label']}, about {item['width']} m wide and {item['height']} m tall, at "
            f"[{item['position'][0]}, {item['position'][2]}], in shot{'s' if len(item['shots']) > 1 else ''} {shots}")


def at_pixel(points, uv):
    """The world point under pixel ``(u, v)`` of one picture's ``(h, w, 3)`` point map."""
    h, w = int(points.shape[0]), int(points.shape[1])
    u = min(w - 1, max(0, int(round(uv[0]))))
    v = min(h - 1, max(0, int(round(uv[1]))))
    return points[v, u]


def tracks(frame: Frame, seen: list[tuple]) -> list[dict]:
    """People followed through the pictures by where they stand.

    Args:
        frame: The frame the reconstruction is read in.
        seen: ``(seconds into the clip, shot number, people)`` per picture in time order, each person
            ``{"foot", "left", "right", "x"}`` with world points.

    Returns:
        Tracks ``{"track", "path", "speed", "shots", "first"}``: ``path`` holds ``{"at", "position", "turn"}`` in
        clip seconds and metres on the floor, ``first`` says where across the frame and in which shot it was
        first seen.
    """
    out: list[dict] = []
    for when, shot, people in seen:
        taken = set()
        for person in sorted(people, key=lambda p: p["x"]):
            ground = frame.place(person["foot"])
            turn = None
            if person["left"] is not None and person["right"] is not None:
                import torch

                across = (person["right"] - person["left"]).to(frame.up.dtype)
                facing = frame.direction(torch.linalg.cross(frame.up, across))
                if math.hypot(facing[0], facing[2]) > 1e-6:
                    turn = round(math.degrees(math.atan2(facing[0], facing[2])))
            point = {"at": round(when, 2), "position": [round(ground[0], 2), 0.0, round(ground[2], 2)], "turn": turn}
            best, gap = None, None
            for index, track in enumerate(out):
                if index in taken:
                    continue
                last = track["path"][-1]
                dt = max(0.0, when - last["at"])
                far = math.dist(last["position"], point["position"])
                if far <= TRACK_GATE + TRACK_SPEED * dt and (gap is None or far < gap):
                    best, gap = index, far
            if best is None:
                side = "left" if person["x"] < 0.35 else "right" if person["x"] > 0.65 else "centre"
                out.append({"track": f"P{len(out) + 1}", "path": [point], "shots": [shot],
                            "first": f"shot {shot}, {side} of the frame"})
                taken.add(len(out) - 1)
            else:
                out[best]["path"].append(point)
                if shot not in out[best]["shots"]:
                    out[best]["shots"].append(shot)
                taken.add(best)
    out = _merged(out, max((len(people) for _, _, people in seen), default=0))
    for number, track in enumerate(out, 1):
        track["track"] = f"P{number}"
        path = track["path"]
        length = sum(math.dist(a["position"], b["position"]) for a, b in zip(path, path[1:]))
        span = path[-1]["at"] - path[0]["at"]
        track["speed"] = round(length / span, 2) if span > 0 else 0.0
    return out


def _merged(tracks: list[dict], most: int) -> list[dict]:
    """Tracks joined end to start, nearest first, down to the most people seen at once; the least seen drop."""
    tracks = [dict(t, path=list(t["path"]), shots=list(t["shots"])) for t in tracks]
    while len(tracks) > max(1, most):
        best = None
        for i, a in enumerate(tracks):
            for j, b in enumerate(tracks):
                if i == j or a["path"][-1]["at"] >= b["path"][0]["at"]:
                    continue
                far = math.dist(a["path"][-1]["position"], b["path"][0]["position"])
                if best is None or far < best[0]:
                    best = (far, i, j)
        if best is None:
            tracks.remove(min(tracks, key=lambda t: len(t["path"])))
            continue
        _, i, j = best
        a, b = tracks[i], tracks[j]
        a["path"] += b["path"]
        a["shots"] += [s for s in b["shots"] if s not in a["shots"]]
        del tracks[j]
    return sorted(tracks, key=lambda t: t["path"][0]["at"])


def _moving(path: list[dict], end: bool) -> tuple[float, float]:
    """A path's heading on the floor at its start or end, ``(dx, dz)`` per second, or zero."""
    pts = path[-3:] if end else path[:3]
    if len(pts) < 2 or pts[-1]["at"] <= pts[0]["at"]:
        return 0.0, 0.0
    span = pts[-1]["at"] - pts[0]["at"]
    return ((pts[-1]["position"][0] - pts[0]["position"][0]) / span,
            (pts[-1]["position"][2] - pts[0]["position"][2]) / span)


def _fit(pairs: list[tuple]) -> tuple[float, float, float]:
    """The turn and shift on the floor that best carry ``(source, target)`` points onto each other."""
    n = len(pairs)
    sx = sum(a[0] for a, _ in pairs) / n
    sz = sum(a[1] for a, _ in pairs) / n
    tx = sum(b[0] for _, b in pairs) / n
    tz = sum(b[1] for _, b in pairs) / n
    angle = 0.0
    if n >= 2:
        dot = sum((a[0] - sx) * (b[0] - tx) + (a[1] - sz) * (b[1] - tz) for a, b in pairs)
        cross = sum((a[0] - sx) * (b[1] - tz) - (a[1] - sz) * (b[0] - tx) for a, b in pairs)
        angle = math.atan2(cross, dot)
    c, s = math.cos(angle), math.sin(angle)
    return angle, tx - (c * sx - s * sz), tz - (s * sx + c * sz)


def _carry(point: list[float], angle: float, dx: float, dz: float) -> list[float]:
    c, s = math.cos(angle), math.sin(angle)
    x, z = point[0], point[2]
    return [round(c * x - s * z + dx, 2), point[1], round(s * x + c * z + dz, 2)]


def stitch(parts: list[dict]) -> tuple[list[list[dict]], list[dict]]:
    """Shots solved on their own joined into one set by the people seen on both sides of each cut.

    Args:
        parts: Per shot in order, ``{"start", "keys", "tracks", "objects"}`` in that shot's own frame.

    Returns:
        ``(keys per shot, tracks, objects)`` in the first shot's frame, the same person keeping one id across cuts.
        A shot with nobody to join by keeps its own frame.
    """
    import itertools

    joined: list[dict] = []
    things: list[dict] = []
    keys_out = []
    for part in parts:
        current = [dict(t, path=list(t["path"]), shots=list(t["shots"])) for t in part["tracks"]]
        alive = [t for t in joined if t["path"][-1]["at"] >= part["start"] - CUT_HOLD]
        angle = dx = dz = 0.0
        pairing = []
        if current and alive:
            best = None
            for order in itertools.permutations(range(len(alive)), min(len(alive), len(current))):
                pairs = [(current[i]["path"][0]["position"][::2], alive[j]["path"][-1]["position"][::2])
                         for i, j in enumerate(order)]
                a, x, z = _fit(pairs)
                cost = sum(math.dist(_carry([p[0], 0.0, p[1]], a, x, z)[::2], q) for p, q in pairs)
                for i, j in enumerate(order):
                    vx, vz = _moving(current[i]["path"], False)
                    wx, wz = _moving(alive[j]["path"], True)
                    c, s = math.cos(a), math.sin(a)
                    cost += TURN_WEIGHT * math.hypot(c * vx - s * vz - wx, s * vx + c * vz - wz)
                cost += TURN_WEIGHT * abs(math.degrees(a)) / 180.0
                if best is None or cost < best[0]:
                    best = (cost, a, x, z, list(enumerate(order)))
            _, angle, dx, dz, pairing = best
        keys = [dict(k, position=_carry(k["position"], angle, dx, dz), look_at=_carry(k["look_at"], angle, dx, dz))
                for k in part["keys"]]
        keys_out.append(keys)
        for track in current:
            track["path"] = [dict(p, position=_carry(p["position"], angle, dx, dz),
                                  turn=(None if p.get("turn") is None
                                        else round(p["turn"] - math.degrees(angle))))
                             for p in track["path"]]
        used: list[int] = []
        for item in part.get("objects", []):
            moved = dict(item, position=_carry(item["position"], angle, dx, dz), shots=list(item["shots"]))
            same = [t for t in things if t["label"] == moved["label"] and id(t) not in used
                    and math.dist(t["position"][::2], moved["position"][::2]) <= OBJECT_REJOIN]
            near = min(same, key=lambda t: math.dist(t["position"][::2], moved["position"][::2]), default=None)
            if near is not None:
                used.append(id(near))
            if near is None:
                things.append(moved)
                continue
            total = near["seen"] + moved["seen"]
            near["position"] = [round((near["position"][i] * near["seen"] + moved["position"][i] * moved["seen"])
                                      / total, 2) for i in range(3)]
            near["seen"] = total
            near["shots"] += [s for s in moved["shots"] if s not in near["shots"]]
        paired = dict(pairing)
        for i, track in enumerate(current):
            if i in paired:
                target = alive[paired[i]]
                target["path"] += track["path"]
                target["shots"] += [s for s in track["shots"] if s not in target["shots"]]
            else:
                joined.append(track)
    for number, track in enumerate(sorted(joined, key=lambda t: t["path"][0]["at"]), 1):
        track["track"] = f"P{number}"
        path = track["path"]
        length = sum(math.dist(a["position"], b["position"]) for a, b in zip(path, path[1:]))
        span = path[-1]["at"] - path[0]["at"]
        track["speed"] = round(length / span, 2) if span > 0 else 0.0
    for number, item in enumerate(things, 1):
        item["track"] = f"O{number}"
    return keys_out, sorted(joined, key=lambda t: t["path"][0]["at"]), things


def describe_track(track: dict) -> str:
    """One measured person in words for the brief."""
    path = track["path"]
    first, last = path[0], path[-1]
    speed = track["speed"]
    gait = "runs" if speed > 2.2 else "walks" if speed > 0.4 else "stays put"
    shots = ", ".join(str(s) for s in track["shots"])
    where = (f"from [{first['position'][0]}, {first['position'][2]}] to [{last['position'][0]}, "
             f"{last['position'][2]}]") if gait != "stays put" else f"at [{first['position'][0]}, {first['position'][2]}]"
    return (f"{track['track']}: first seen {track['first']}, in shot{'s' if len(track['shots']) > 1 else ''} {shots}, "
            f"{first['at']:.2f} to {last['at']:.2f} s, {gait} {where}" + (f" ({speed} m/s)" if gait != "stays put" else ""))


def describe(keys: list[dict]) -> tuple[str, bool]:
    """A shot's camera move in words, and whether it wobbles like a held camera.

    Args:
        keys: The shot's camera keys, in time order.

    Returns:
        ``(words, handheld)``.
    """
    if len(keys) < 2:
        return "the camera holds still", False
    first, last = keys[0], keys[-1]
    dx, dy, dz = (last["position"][i] - first["position"][i] for i in range(3))

    def heading(key):
        p, t = key["position"], key["look_at"]
        return math.degrees(math.atan2(t[0] - p[0], -(t[2] - p[2])))

    def pitch(key):
        p, t = key["position"], key["look_at"]
        return math.degrees(math.atan2(t[1] - p[1], math.hypot(t[0] - p[0], t[2] - p[2])))

    turn = heading(last) - heading(first)
    turn = (turn + 180.0) % 360.0 - 180.0
    tilt = pitch(last) - pitch(first)
    ratio = last["lens"] / max(first["lens"], 1e-6)
    parts = []
    travel = math.sqrt(dx * dx + dy * dy + dz * dz)
    if travel > 0.25:
        if abs(turn) > 25.0 and math.hypot(dx, dz) > 1.0:
            parts.append(f"orbits {'left' if turn < 0 else 'right'} {abs(turn):.0f} degrees round the subject")
        else:
            moves = []
            if abs(dz) > 0.25:
                moves.append(f"{'pushes in' if dz < 0 else 'pulls out'} {abs(dz):.1f} m")
            if abs(dx) > 0.25:
                moves.append(f"trucks {'right' if dx > 0 else 'left'} {abs(dx):.1f} m")
            if abs(dy) > 0.25:
                moves.append(f"cranes {'up' if dy > 0 else 'down'} {abs(dy):.1f} m")
            parts.append(" and ".join(moves) or f"drifts {travel:.1f} m")
            if abs(turn) > 5.0:
                parts.append(f"pans {'left' if turn < 0 else 'right'} {abs(turn):.0f} degrees")
    else:
        if abs(turn) > 5.0:
            parts.append(f"pans {'left' if turn < 0 else 'right'} {abs(turn):.0f} degrees")
        if abs(tilt) > 5.0:
            parts.append(f"tilts {'up' if tilt > 0 else 'down'} {abs(tilt):.0f} degrees")
    if ratio > 1.12:
        parts.append(f"zooms in from {first['lens']:.0f} to {last['lens']:.0f} mm")
    elif ratio < 0.89:
        parts.append(f"zooms out from {first['lens']:.0f} to {last['lens']:.0f} mm")
    wobble = 0.0
    if len(keys) >= 4:
        for a, b, c in zip(keys, keys[1:], keys[2:]):
            mid = [(a["position"][i] + c["position"][i]) / 2.0 for i in range(3)]
            wobble = max(wobble, math.dist(mid, b["position"]))
    handheld = wobble > max(0.04, 0.15 * travel / max(len(keys) - 1, 1))
    words = "the camera " + (", ".join(parts) if parts else "holds still")
    return words, handheld


def sound_hits(audio: dict | None, start: float = 0.0, length: float = 0.0) -> list[tuple[float, float]]:
    """The strongest onsets in a clip's sound, as ``(seconds, strength 0 to 1)``, in time order.

    Args:
        audio: A ComfyUI audio, or ``None``.
        start: Seconds into the audio the clip starts at.
        length: Seconds read, or 0 for the rest.

    Returns:
        Up to :data:`MAX_HITS` hits at least :data:`HIT_GAP` apart, louder than three times the median onset.
    """
    import numpy as np

    from ..media import beats

    if not audio or audio.get("waveform") is None or audio["waveform"].shape[-1] < 2048:
        return []
    part = beats.trimmed(audio, start, length) if (start or length) else audio
    samples, rate = beats.mono(part)
    if samples.size < 2048:
        return []
    log_mel, _ = beats._mel_frames(samples, rate)
    envelope = beats.onset_envelope(log_mel)
    if envelope.size < 3 or float(envelope.max()) <= 0.0:
        return []
    frame_rate = beats.ANALYSIS_RATE / beats.HOP
    floor = 3.0 * float(np.median(envelope[envelope > 0])) if np.any(envelope > 0) else 0.0
    lead = int(round(SOUND_LEAD * frame_rate))
    peaks = [i for i in range(max(1, lead), envelope.size - 1)
             if envelope[i] >= envelope[i - 1] and envelope[i] > envelope[i + 1] and envelope[i] > floor]
    peaks.sort(key=lambda i: -float(envelope[i]))
    gap = int(round(HIT_GAP * frame_rate))
    kept: list[int] = []
    for i in peaks:
        if all(abs(i - j) > gap for j in kept):
            kept.append(i)
        if len(kept) >= MAX_HITS:
            break
    top = float(envelope.max())
    return sorted((round(i / frame_rate, 2), round(float(envelope[i]) / top, 2)) for i in kept
                  if float(envelope[i]) / top >= HIT_FLOOR)



def _slice(audio: dict, start: float, end: float) -> dict:
    """A stretch of a ComfyUI audio, in seconds."""
    rate = int(audio["sample_rate"])
    first = max(0, int(start * rate))
    last = max(first + 1, int(end * rate))
    return {"waveform": audio["waveform"][..., first:last].contiguous(), "sample_rate": rate}


def listen(model, audio: dict, hits: list[tuple[float, float]], spans: list[tuple[float, float]],
           node: str) -> tuple[dict, dict]:
    """What a model that hears names at each sound hit and hears said in each shot.

    Args:
        model: A language model from the LLM family that reads sound.
        audio: The clip's sound.
        hits: ``(seconds, strength)`` per sound hit.
        spans: ``(start, end)`` seconds per shot.
        node: The asking node's name, for errors.

    Returns:
        ``(names, words)``: hit seconds to what makes the sound, and shot index to the words spoken, if any.
    """
    from ..llm import generate
    from ..llm import messages as llm_messages

    def ask(clip, question, most):
        chat = [llm_messages.message("user", question, audio=clip)]
        return generate.reply(model, chat, node, max_tokens=most).text.strip()

    names, words = {}, {}
    for when, _ in hits:
        said = ask(_slice(audio, max(0.0, when - HIT_BEFORE), when + HIT_AFTER),
                   "Name the main sound in this short clip in a few words, such as 'a car door slams'. "
                   "Answer with the words only.", HIT_TOKENS)
        if said:
            names[when] = said.splitlines()[0].strip().rstrip(".")[:80]
    for index, (start, end) in enumerate(spans):
        said = ask(_slice(audio, start, end),
                   "Write down any words spoken in this clip, exactly, each line as who says it (a man, a woman, a "
                   "child) and the words in quotes. If nobody speaks, answer: none.", SPEECH_TOKENS)
        if said and said.strip().lower().rstrip(".") not in ("none", "nothing", "no one speaks", "nobody speaks"):
            words[index] = " ".join(said.split())[:400]
    return names, words


def geometry(keys: list[dict]) -> dict:
    """The measured lens and travel of a shot, in the shape Shot Reader's frame-geometry line reads."""
    if len(keys) < 2:
        return {}
    scale = keys[-1]["lens"] / max(keys[0]["lens"], 1e-6)
    travel = math.dist(keys[0]["position"], keys[-1]["position"])
    if travel < LENS_ONLY_TRAVEL:
        return {"s_global": scale, "s_centre": scale, "s_periph": scale, "score": LENS_ONLY_RESIDUAL}
    return {"s_global": scale, "score": math.inf}


def _sheet(text: str) -> dict | None:
    """The first JSON object in a model's answer, or ``None``."""
    import json

    start = text.find("{")
    while start >= 0:
        depth = 0
        for end in range(start, len(text)):
            depth += {"{": 1, "}": -1}.get(text[end], 0)
            if depth == 0:
                try:
                    found = json.loads(text[start:end + 1])
                except ValueError:
                    break
                return found if isinstance(found, dict) else None
        start = text.find("{", start + 1)
    return None


def read_shots(model, frames, rate: float, spans: list[tuple[int, int]], shots: list[Shot], node: str) -> None:
    """Shot Reader's fact sheet for every shot, kept on each :class:`Shot` as ``sheet``.

    Args:
        model: Shot Reader v2 from LLM Loader, with its mmproj.
        frames: The clip's ``(N, H, W, 3)`` frames.
        rate: Frames a second.
        spans: ``(first, stop)`` frame per shot.
        shots: The measured shots, in the same order.
        node: The asking node's name, for errors.
    """
    import torch
    import torch.nn.functional as F

    from ..llm import generate
    from ..llm import messages as llm_messages
    from ..vendor.shot_reader import prompt as reader

    for (first, stop), shot in zip(spans, shots):
        length = stop - first
        if length < 2:
            continue
        picks = [min(stop - 1, first + int(i * length / reader.N_FRAMES)) for i in range(reader.N_FRAMES)]
        times = [(p - first) / rate for p in picks]
        batch = frames[picks][..., :3].float().permute(0, 3, 1, 2)
        batch = F.interpolate(batch, size=READER_SIZE, mode="bicubic", align_corners=False).clamp(0, 1)
        batch = batch.permute(0, 2, 3, 1).contiguous()
        text = reader.build_prompt(length / rate, times, [], reader.hint_line(geometry(shot.keys)))
        chat = [llm_messages.message("user", text, images=batch)]
        shot.sheet = _sheet(generate.reply(model, chat, node, max_tokens=READER_TOKENS).text)
        del batch
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


def apply_sheet(shot: Shot) -> None:
    """Shot Reader's camera height on a shot nobody's size scaled, and its handheld reading."""
    camera = (shot.sheet or {}).get("camera") if isinstance(shot.sheet, dict) else None
    if not isinstance(camera, dict):
        return
    if camera.get("handheld") is True:
        shot.handheld = True
    height = camera.get("height_m")
    if not shot.scaled and isinstance(height, (int, float)) and MIN_HEIGHT <= height <= MAX_HEIGHT and shot.keys:
        lift = float(height) - shot.keys[0]["position"][1]
        for key in shot.keys:
            key["position"][1] = round(key["position"][1] + lift, 2)
            key["look_at"][1] = round(key["look_at"][1] + lift, 2)


def describe_sheet(sheet: dict) -> str:
    """Shot Reader's reading of a shot in one line for the brief."""
    camera = sheet.get("camera") if isinstance(sheet.get("camera"), dict) else {}
    zoom = camera.get("zoom") if isinstance(camera.get("zoom"), dict) else {}
    said = [str(sheet.get("caption") or "").strip()]
    reading = [f"move {camera['move']}" if camera.get("move") else "",
               f"zoom {zoom['dir']}" if zoom.get("dir") not in (None, "none") else "",
               f"{camera['angle']} angle" if camera.get("angle") else "",
               f"camera {camera['height_m']} m up" if isinstance(camera.get("height_m"), (int, float)) else "",
               "handheld" if camera.get("handheld") is True else "",
               f"{camera['framing']} framing" if camera.get("framing") else ""]
    tags = sheet.get("tags") if isinstance(sheet.get("tags"), list) else []
    env = sheet.get("environment") if isinstance(sheet.get("environment"), dict) else {}
    line = " ".join(s for s in said if s)
    line += " (" + ", ".join(r for r in reading if r) + ")" if any(reading) else ""
    if env.get("desc"):
        line += f" Set: {env['desc']}."
    if tags:
        line += " Tags: " + ", ".join(str(t) for t in tags[:8]) + "."
    return line.strip()


def brief(shots: list[Shot], hits: list[tuple[float, float]], pictures_per_shot: int, directions: str = "",
          people: list[dict] | None = None, things: list[dict] | None = None, environment: bool = False,
          heard: tuple[dict, dict] | None = None) -> str:
    """The request a language model turns into a blockout scene.

    Args:
        shots: The measured shots, in order.
        hits: Sound hits as ``(seconds into the clip, strength)``.
        pictures_per_shot: Pictures handed over for each shot, in shot order.
        directions: What to change from the original, or empty to keep it.
        people: Measured people from :func:`tracks`, or none.
        things: Measured objects from :func:`objects`, or none.
        environment: Also ask for the set, its objects, the people's use of them and where each stands, in words.
        heard: ``(names, words)`` from :func:`listen`, or none.

    Returns:
        Plain text naming every shot's length and camera move, the people measured and the sound hits.
    """
    total = sum(s.seconds for s in shots)
    cuts = [f"{s.start:.2f} s" for s in shots[1:]]
    directions = (directions or "").strip()
    lines = [
        f"Recreate this video as a blockout scene. It runs {total:.2f} s in {len(shots)} shot"
        f"{'s' if len(shots) != 1 else ''}" + (f", cut at {', '.join(cuts)}." if cuts else "."),
    ]
    if directions:
        lines.append(f"Change it as asked: {directions} Write the look, setting, sound, people and props as changed.")
    lines += [
        f"The pictures are the shots in order, {pictures_per_shot} of each. Place the props and people as "
        "they stand in them; name and describe each person and thing as seen.",
        "Each shot's camera was measured: write every shot as \"rig\": \"measured\" and the measured "
        "camera, its seconds and its shake are filled in. Write another rig only for a shot whose camera the "
        "reshoot directions change. The first camera stands on the +z side looking toward -z, the subject "
        "near the origin; place the props as the pictures show them from there.",
    ]
    if people:
        lines.append("People were measured: write each person once with \"track\" set to their id below, a "
                     "name and a description from the pictures; where they stand, walk and face is filled in.")
        lines.extend(describe_track(track) for track in people)
    else:
        lines.append("Place the people as the pictures show them.")
    if things:
        lines.append("Objects were measured: write each as a prop with \"track\" set to its id below and the type "
                     "that fits it; its place is filled in. Add what else the pictures show, such as trees, walls "
                     "and buildings, where it stands.")
        lines.extend(describe_object(item) for item in things)
    if environment:
        lines.append("Describe the environment in words as well: in \"setting\", name every object and surface the "
                     "pictures show, its material and state, and where each stands from the people and from each "
                     "other; in each shot's \"action\", say how the people use, touch or pass the things around "
                     "them.")
    names, words = heard or ({}, {})
    for shot in shots:
        held = ", handheld" if shot.handheld else ""
        lines.append(f"[Shot {shot.index}] {shot.start:.2f} to {shot.start + shot.seconds:.2f} s "
                     f"({shot.seconds:.2f} s): {shot.words}{held}.")
        read = describe_sheet(shot.sheet) if shot.sheet else ""
        if read:
            lines.append(f"  Shot Reader: {read}")
        if words.get(shot.index - 1):
            lines.append(f"  Heard spoken: {words[shot.index - 1]} Give these lines as the shot's \"dialogue\".")
    if hits:
        loud = max(hits, key=lambda h: h[1])[0]
        listed = ", ".join(f"{t:.2f} s" + (f" {names[t]}" if t in names else "") + (" (the loudest)" if t == loud else "")
                           for t, _ in hits)
        lines.append(f"Sound hits: {listed}. Give each one a shot's \"sounds\" entry, its \"at\" counted from "
                     "that shot's start, " + ("in the words heard." if names else
                                             "naming what makes it where the pictures show the cause."))
    if directions:
        lines.append(f"Reshoot directions: {directions} Change only what these ask for; keep everything "
                     "else as measured.")
    return "\n".join(lines)
