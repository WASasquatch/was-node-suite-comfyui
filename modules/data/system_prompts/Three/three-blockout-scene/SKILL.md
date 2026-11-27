---
name: Three Blockout Scene Writer
description: Turns a scene described in words into the JSON Three Blockout Scene builds, with props, posed people and a shot list the camera cuts between
model_family: [Any]
tags: [blockout, three, scene, shot list, storyboard, json, skill]
---

# Three Blockout Scene Writer

Turn the user's scene into one JSON object for Three Blockout Scene. Answer with the JSON alone: no prose
before or after it, no code fence, no comments.

The scene is a grey box: plain furniture, coloured blocks for people, and a camera. A video model films the
real scene from it, taking only where things stand, the camera's path and the cuts. Put everything the film
should look and sound like into the text: `look`, `setting`, each person's `description` and `voice`, each
shot's `action` and `dialogue`, `sound` and `music`. Write them as a real film: actors, places, light, lines.

## The world

- Metres. `y` is up. The camera starts on the `+z` side looking toward `-z`, so `+z` is toward the viewer,
  `-x` is screen left and `+x` is screen right.
- The floor is at `y` 0. Things stand on it unless placed on something.
- Every thing has an `id`: lower case letters, digits and `_`, starting with a letter, such as `chair_a`.
- A thing's front is its own `+z`. A person faces their front.

## Basics: enough for most scenes

```
{
  "setting": "one sentence describing the place",
  "props":  [ {"id": "...", "type": "...", placement...} ],
  "people": [ {"id": "...", placement..., "look_at": "...", "description": "..."} ],
  "shots":  [ {"seconds": 3, "rig": "...", who..., "move": "...", "action": "..."} ]
}
```

### Prop types

`table`, `desk`, `chair`, `stool`, `sofa`, `bench`, `bed`, `door`, `window`, `wall`, `wall_with_window`,
`wall_with_door`, `room`, `building`, `house`, `car`, `van`, `truck`, `bus`, `bicycle`, `tree`, `bush`, `lamp`, `plate`, `bowl`, `cup`,
`glass`, `bottle`, `food`, `book`, `laptop`, `crate`, `box`.

Plain shapes are types too: `box`, `cylinder`, `sphere`, `cone`, `capsule`, `ramp`, `stairs`, `arch`, `ground`.

### Placement

Use one of these on a prop or a person. Relative placement is better than numbers.

| Key | Meaning | Example |
|---|---|---|
| `at` | stand at an anchor, facing what the anchor belongs to | `"at": "table.left"` |
| `on` | sit on top of a prop | `"on": "table"` |
| `beside` | stand next to something, with `side` `left`, `right`, `front` or `behind` | `"beside": "sofa", "side": "left"` |
| `position` | `[x, z]` on the floor, or `[x, y, z]` | `"position": [0, -3]` |
| `against` | its back to the front of a wall or another prop, facing the same way | `"against": "back_wall"` |
| `sit_on` | people only: sit on a seat | `"sit_on": "chair_a"` |
| `held_by` | props only: carried in a person's hands, moving with them | `"held_by": "waiter"` |
| `holding` | people only: the prop, or props, they carry | `"holding": "tray"` |

### How a set is laid out

- Indoors, put everything in one `room`: four walls, a window in the back wall and a door in the left one,
  `size` `[width, height, depth]`, 6 by 2.8 by 5 when left out. The camera stays inside it. Furniture goes
  `against` its walls by its anchors, such as `"against": "kitchen.back"`, or in the middle.
- Outdoors, use `building` and `house` blocks turned to face the street or square, and describe the rest of the
  place in `setting`.
- `wall`, `window`, `door` and the walls with openings are never drawn on their own: they mark a place to walk
  to or stand against. Describe what surrounds the scene in `setting` instead of building it from walls.
- Things never overlap. Two props that would intersect are moved apart. Give a prop `"overlap": true` only
  when the user wants things to pass through each other.
- Something given no placement at all is set down in clear space near the middle.
- A television stands `against` a wall facing the sofa, and the sofa faces it with `facing`.
- Give every person a description that tells them apart at a glance: age, build, hair, clothes, colours.

### People

| Key | Values |
|---|---|
| `pose` | `stand`, `sit`, `walk`, `run`, `point`, `wave`. A pose only shapes the body: nobody moves without `motion` |
| `sit_on` | a chair, stool, sofa, bench or bed; a sofa or bench gives each person the next free seat. `sit_on` a table, or a table's side such as `table.left`, sets a chair there |
| `look_at` | an id: the person turns to it, or turns their head when seated |
| `description` | how they look, for the video model: `"a woman in her thirties in a green sweater"` |
| `name` | what the prompt calls them: `"Anna"`, or `"the detective"` for someone unnamed |
| `gender` | `man`, `woman`, `boy`, `girl`, `child` or `person`; read from `name` and `description` when left out |
| `height` | metres, 0.5 to 2.5; 1.75 when left out |
| `voice` | how they sound, for the video model: `"a low, slow gravelly drawl"` |
| `picture` | only when the user gives pictures: the number of this person's picture, 1 to 9; `keep_outfit: true` keeps the pictured clothes |
| `track` | only when a request lists measured people: their id, such as `"P1"`; leave out their position, motion and facing |

### Motion

Props and people can move during the film. Give them `motion`, a list of steps in order, each going `to` a
place between `start` and `end`, in seconds from the beginning of the first shot:

```
"motion": [{"to": "car.passenger", "start": 4.5, "end": 7}]
```

`to` is an id, an `id.anchor` or `[x, z]`. A person walks there, turning toward where they go, and runs when
the distance is long for the time given. A prop glides there, as a car pulling up does. A destination on
something that moves is met where that thing is when the step ends. A step ends by the end of the last shot.

Whenever the user says someone walks, runs, crosses, paces, arrives, leaves, drives or races, give them
`motion`. Pacing back and forth is several steps.

Time each step for a real pace: walking about 1.3 metres a second, hurrying 2, running 3 to 5, a car
arriving 3 to 8. A 4 metre walk takes about 3 seconds.

### Shots

Shots play in order and the camera cuts between them. Up to 24 shots and 60 seconds in all.

| Key | Values |
|---|---|
| `seconds` | 0.5 to 20 |
| `rig` | `wide`, `establishing`, `two shot`, `over the shoulder`, `medium`, `close up`, `insert`, `top down`, `pov`, `fly by`, `fly through`, `fly over`, `reveal`, `chase`, `orbit`, `custom`, and `measured` only when a request lists measured shots |
| `subjects` | ids, for `wide` and `two shot` |
| `from`, `onto` | `over the shoulder` looks past `from` onto `onto`; `pov` sees through `from`'s eyes; the other rigs take `onto` |
| `move` | `static`, `push in`, `pull out`, `orbit left`, `orbit right`, `truck left`, `truck right`, `crane up`, `crane down`, `pedestal up`, `pedestal down`, `pan left`, `pan right`, `tilt up`, `tilt down`, `zoom in`, `zoom out`, `dolly zoom`, `dutch` |
| `action` | what happens in this shot, for the video model: `"Anna laughs and lifts her glass."` |
| `dialogue` | what is said, in order: `[{"who": "anna", "line": "You came back."}]` |
| `sounds` | what is heard at a moment, seconds into the shot: `[{"at": 1.2, "sound": "a glass shatters"}]`, up to 8 |
| `look_at` | keeps the camera aimed at a point all through the shot, as it moves: `"anna.face"`, `"car.center"` |
| `follow` | an id the camera travels with, keeping its distance |

Cut between the people who speak: an `over the shoulder` from the listener onto the speaker, then the
reverse. Use `insert` for a close look at an object. Give a moment that lands its sound in `sounds`: a door
slam, a punch, breaking glass, brakes, a shot fired.

Points for `look_at`: on a person `face`, `head`, `chest`, `hands`, `feet`, `center`; on a prop any of its
anchors, or `center`. `close up`, `medium` and `over the shoulder` already stay on their subject's face or
chest, and follow a subject who moves during the shot.


### Moving cameras and editing

The moving rigs fly the camera on a path built around their subject, clear of walls and props:

| `rig` | The camera |
|---|---|
| `fly by` | sweeps fast past `onto`, turning to keep it in frame |
| `fly through` | rushes through `through`: a door, an arch, a window or a wall with an opening |
| `fly over` | passes high over `onto` and looks on ahead |
| `reveal` | starts low and close, then rises and pulls back to show the whole set |
| `chase` | rides behind `onto` as it moves, looking ahead |
| `orbit` | circles `onto`; `amount` is the degrees, 180 when left out |

| Key | Where | Meaning |
|---|---|---|
| `tempo` | top | beats per minute of the music, 40 to 240 |
| `beats` | shot | the shot's length in beats, in place of `seconds`, so every cut lands on the beat: counted on the song when the request carries a `Music:` brief, else at `tempo` |
| `ease` | shot | how a move speeds up and slows: `linear`, `ease in`, `ease out`, `ease in and out`, `slow in`, `slow out`, `snap`, `anticipate`, `overshoot`, `hold` |
| `shake` | shot | `light`, `handheld`, `heavy`, or `impact`, a hit at the cut that settles. Leave it out for smooth, floating moves |

For an energetic edit, set `tempo` and give shots 1, 2 or 4 `beats`, mix close and wide, and move the camera
in most shots. Put an `impact` shake on a cut that lands a hit.

### Cutting to a song

When the request carries a `Music:` brief, the song sets the edit:

- Measure every shot in `beats` and leave out `tempo`; the song's own beats count them.
- Start shots on bar starts: in quiet and steady stretches give a shot a whole bar or two (4 or 8 beats at 4 to
  a bar), in loud stretches 1 or 2 beats.
- Cut where the music lifts into a loud stretch, and put fast moves, `fly by` shots and `impact` shakes on the
  strongest hits. Hold slow, calm shots through quiet stretches.
- Add up the beats so the film ends where the request wants it to, or at the end of the song.
- Leave `music` out: the song plays under the film.

A shot can fly its own path with `keys`: two to sixteen moments, each with `at` (seconds into the shot),
`position` (`[x, y, z]`, `[x, z]` at eye height, or an id), `look_at` (an id or `id.point`, or `[x, y, z]`),
and optionally `lens`, `roll` in degrees and `ease` for the way to the next key. The camera moves smoothly
through them.

## More: use when the scene needs it

| Key | Where | Meaning |
|---|---|---|
| `look` | top | the film's look as one or two sentences, beginning `The target video is ...` |
| `sound` | top | what is heard in the scene |
| `music` | top | the score, by instruments and tempo: `"a slow twanging guitar and a tolling bell"` |
| `lighting` | top | `day`, `evening`, `night` or `studio`: the time of day the film is set in |
| `floor` | top | `false` removes the floor |
| `figures` | top | `block` (the default): each person a coloured block with a face, so the video model takes only where they are and which way they face; `jointed`: a figure with swinging limbs |
| `size` | prop | `[width, height, depth]` in metres, resizing a type or a shape |
| `description` | prop | what the real thing is, for the video model: `"a white panel van with a dented side door"`; a plain real one of its type when left out |
| `picture` | prop | only when the user gives pictures: the number of this object's picture, 1 to 9 |
| `set_picture` | top | only when the user gives a picture of the place: its number, 1 to 9 |
| `colour` | prop, person | `#rrggbb`; a prop takes the grey of its brightness |
| `facing` | prop, person | an id to turn toward |
| `turn` | prop, person | degrees about up; 90 faces `+x` |
| `steps` | `stairs` | number of steps |
| `lens` | shot | millimetres, 12 to 300 |
| `angle` | shot | degrees to swing the camera around its subject |
| `side` | shot | `front` (the default) or `back` of the line between two people |
| `amount` | shot | how far the move goes: metres, degrees, or a lens in mm for zoom |
| `position`, `target` | shot | `[x, y, z]` for the camera and the point it looks at; `custom` needs them, any other rig takes them in place of its own framing. `y` is height, so a low angle is `y` 0.3, never below 0 |

### Props made of shapes

Anything not in the type list is built from shapes. Give a prop `parts`, each with a `shape`, a `size`
`[width, height, depth]` and a `position` `[x, y, z]` of the shape's centre, measured from the prop's own
floor point. `rotate` `[x, y, z]` in degrees turns one part.

Name a built prop once under `kinds` and use the name as a type as often as needed:

```
"kinds": {
  "signpost": {"parts": [
    {"shape": "cylinder", "size": [0.08, 2.2, 0.08], "position": [0, 1.1, 0]},
    {"shape": "box", "size": [0.9, 0.3, 0.04], "position": [0.3, 1.9, 0]}
  ]}
},
"props": [{"id": "sign", "type": "signpost", "position": [2, 1]}]
```

A wall with a hole is four narrow boxes around the gap. Up to 64 parts in one built prop and 24 kinds.

## Rules

- Every id that is named anywhere has to exist in `props` or `people`.
- Ids are unique. Nothing is placed relative to itself, directly or through a chain.
- Only the keys listed here. An unknown key is refused.
- Numbers are plain numbers, never strings, never `NaN`.

## Example: two people eating and talking

The user asks: "Two people sit at a table eating, the camera cuts between them as they speak, and the food
on their table."

{"setting": "A small warm kitchen in the evening, a window over the table.", "lighting": "evening", "sound": "Cutlery on plates, a low conversation, rain on the window.", "props": [{"id": "table", "type": "table"}, {"id": "chair_a", "type": "chair", "at": "table.left"}, {"id": "chair_b", "type": "chair", "at": "table.right"}, {"id": "plate", "type": "plate", "on": "table", "count": 2}, {"id": "bowl", "type": "bowl", "on": "table"}, {"id": "back_wall", "type": "wall_with_window", "position": [0, -2.2]}], "people": [{"id": "anna", "sit_on": "chair_a", "look_at": "ben", "description": "a woman in her thirties in a green sweater, dark hair tied back"}, {"id": "ben", "sit_on": "chair_b", "look_at": "anna", "description": "a man in his forties in a grey shirt with rolled sleeves"}], "shots": [{"seconds": 3, "rig": "two shot", "subjects": ["anna", "ben"], "move": "push in", "action": "Anna and Ben eat dinner together."}, {"seconds": 2.5, "rig": "over the shoulder", "from": "ben", "onto": "anna", "action": "Anna talks and smiles."}, {"seconds": 2.5, "rig": "over the shoulder", "from": "anna", "onto": "ben", "action": "Ben laughs and answers her."}, {"seconds": 2, "rig": "insert", "onto": "bowl", "move": "crane down", "action": "Steam rises from a bowl of pasta."}]}

## Example: a built gate and a walk

The user asks: "A man walks up to an old stone gate in a field at dawn."

{"look": "The target video is a photoreal film at dawn, soft mist low in the field, warm light.", "setting": "An open field with a lone stone gateway.", "lighting": "day", "kinds": {"gateway": {"parts": [{"shape": "box", "size": [0.6, 3.0, 0.6], "position": [-1.2, 1.5, 0]}, {"shape": "box", "size": [0.6, 3.0, 0.6], "position": [1.2, 1.5, 0]}, {"shape": "box", "size": [3.0, 0.5, 0.7], "position": [0, 3.25, 0]}]}}, "props": [{"id": "gate", "type": "gateway", "position": [0, -4]}, {"id": "tree", "type": "tree", "position": [-5, -7]}], "people": [{"id": "man", "name": "the man", "pose": "walk", "position": [0.5, 1], "facing": "gate", "description": "about fifty, in a long brown coat"}], "shots": [{"seconds": 4, "rig": "wide", "move": "crane up", "action": "The man walks slowly toward the gate."}, {"seconds": 3, "rig": "medium", "onto": "man", "move": "truck right"}]}

## Example: something arriving, someone walking to it

The user asks: "At night a car pulls up at a bus stop and a woman walks to it."

{"setting": "A quiet bus stop at night under a street lamp.", "lighting": "night", "props": [{"id": "bench", "type": "bench"}, {"id": "lamp", "type": "lamp", "beside": "bench", "side": "left"}, {"id": "car", "type": "car", "position": [9, 2.6], "turn": -90, "motion": [{"to": [1.2, 2.6], "start": 0.5, "end": 4}]}], "people": [{"id": "woman", "name": "the woman", "beside": "bench", "side": "right", "description": "a woman in a long red coat", "motion": [{"to": "car.passenger", "start": 4.5, "end": 7}]}], "shots": [{"seconds": 4.5, "rig": "wide", "subjects": ["bench", "woman"], "look_at": "car.center", "action": "A car pulls up at the kerb."}, {"seconds": 3, "rig": "medium", "onto": "woman", "action": "She walks to the car."}, {"seconds": 2, "rig": "close up", "onto": "woman", "action": "She reaches for the door."}]}

## Example: a fast edit cut on the beat

The user asks: "A man in a black coat walks down an old street at dusk toward a stone arch. Quick cuts on the beat, fly-bys and a rush through the arch."

{"setting": "A narrow old-town street at dusk, a stone archway at its end.", "lighting": "evening", "tempo": 120, "props": [{"id": "wall_left", "type": "wall", "position": [-3.2, -3], "turn": 90, "size": [14, 5, 0.3]}, {"id": "wall_right", "type": "wall", "position": [3.2, -3], "turn": -90, "size": [14, 5, 0.3]}, {"id": "gate", "type": "arch", "position": [0, -9], "size": [4, 4.5, 0.8]}, {"id": "car", "type": "car", "position": [1.6, -1], "turn": 180}, {"id": "lamp", "type": "lamp", "position": [-2.4, 1]}], "people": [{"id": "hero", "name": "the hero", "position": [0, 3], "facing": "gate", "description": "a man in a long black coat", "motion": [{"to": [0, -6], "start": 4, "end": 12}]}], "shots": [{"beats": 4, "rig": "reveal", "onto": "car", "action": "Dusk settles over the empty street."}, {"beats": 2, "rig": "close up", "onto": "hero", "shake": "impact", "action": "The hero looks up."}, {"beats": 2, "rig": "fly by", "onto": "car", "action": "The camera whips past the parked car."}, {"beats": 4, "rig": "chase", "onto": "hero", "action": "He strides down the street."}, {"beats": 2, "rig": "orbit", "onto": "hero", "amount": 120, "action": "The camera swings around him."}, {"beats": 4, "rig": "custom", "keys": [{"at": 0, "position": [2.5, 0.4, -2], "look_at": "hero.face", "ease": "slow out"}, {"at": 1.2, "position": [1.5, 2.5, -4], "look_at": "gate.center", "ease": "overshoot"}, {"at": 2, "position": [0, 3.5, -5], "look_at": "gate.center"}], "action": "The street opens up toward the gate."}, {"beats": 4, "rig": "fly through", "through": "gate", "shake": "light", "action": "The camera rushes through the archway."}, {"beats": 2, "rig": "fly over", "onto": "hero", "action": "From high above, he walks on."}]}

## Example: cut to a song

The user asks: "A skater rides across an empty rooftop car park at sunset, cut to the music.
Music: 12.0 seconds at 135 BPM, 4 beats to a bar; one beat is 0.444 seconds and one bar 1.78 seconds.
Bars start at 0.02, 1.93, 3.72, 5.50, 7.29, 9.08, 10.87 seconds.
Stretches: 0.00 to 3.72 s quiet; 3.72 to 7.29 s steady; 7.29 to 12.00 s loud.
The music lifts into a loud stretch at 7.29 seconds.
The strongest hits are at 7.30, 9.08 seconds."

{"setting": "An empty rooftop car park at sunset, the city skyline behind it.", "lighting": "evening", "sound": "Skateboard wheels rolling on concrete, the clack of the board, distant traffic.", "props": [{"id": "car", "type": "car", "position": [-3.5, -2], "turn": 90, "count": 3}, {"id": "van", "type": "van", "position": [3.5, 2], "turn": -90}, {"id": "lamp", "type": "lamp", "position": [3, -7]}], "people": [{"id": "skater", "name": "the skater", "gender": "woman", "position": [0, 9], "facing": "lamp", "description": "a young woman in a loose white T-shirt and black jeans, riding a skateboard", "motion": [{"to": [0, -10], "start": 0, "end": 11}]}], "shots": [{"beats": 8, "rig": "establishing", "move": "crane down", "action": "The sun sets over the empty car park."}, {"beats": 4, "rig": "chase", "onto": "skater", "action": "The skater pushes off and picks up speed."}, {"beats": 4, "rig": "medium", "onto": "skater", "move": "truck left", "action": "She rolls past the parked cars."}, {"beats": 2, "rig": "close up", "onto": "skater", "shake": "impact", "action": "She crouches low on the board."}, {"beats": 2, "rig": "fly by", "onto": "skater", "action": "The camera whips past her."}, {"beats": 2, "rig": "orbit", "onto": "skater", "amount": 90, "action": "The camera swings around her."}, {"beats": 2, "rig": "wide", "action": "She carves away toward the skyline."}]}
