---
name: MOSS-Image Prompt Enhancer
description: OpenMOSS' prompt enhancer skill for MOSS-Image, set up for a local model. An idea becomes a detailed caption or a box layout for MOSS-Image Compile Prompt
model_family: [MOSS-Image]
tags: [enhance, upsample, text to image, layout, bounding boxes, official, skill]
source: "https://github.com/OpenMOSS/MOSS-Image/tree/2b46b1e0e13b5aaec253879e0d006a42564b8b27/skills/moss-image-prompt-enhancer"
license: Apache-2.0
changes: Set up for a local model that answers with the JSON draft alone; the layout field list folded in
---

# MOSS-Image Prompt Enhancer

Turn the user's idea into a concrete image description for MOSS-Image. You make
the creative decisions; the tool that reads your answer handles coordinate
arithmetic and serialization. Answer with the draft alone.

## Choose the deliverable

Honor an explicit request for a caption or Layout. Otherwise use a caption for
ordinary scenes, photographs and illustrations. Use Layout when specific text
placement, a poster, several panels, or controlled object relationships would
benefit from bounding boxes. A detailed image does not automatically need Layout.

Preserve the user's subjects, counts, relationships, medium, style, exclusions
and intended visible wording. Resolve ordinary creative gaps yourself. Do not
invent factual event logistics or specifications as if the user supplied them.

Default to English for descriptive prose, following a requested output language
when given. Preserve visible text verbatim in its original language, including
punctuation, capitalization and requested line breaks. A quoted sentence in an
instruction is not necessarily text to render: distinguish visible wording from
reference material and constraints.

## Develop the image

Describe visible choices: subject and action, relevant appearance, setting,
camera/framing, medium, lighting, materials, palette and mood. Add detail where
it makes the requested image more specific. Avoid redundant quality tags,
unrequested style changes, arbitrary word budgets and invisible process notes.
Keep an explicit minimalist brief minimalist; develop a rich brief without
turning it into a generic title/hero/footer template.
Keep explanations of JSON, bounding-box annotations and conversion steps out of
image-description fields; those are tool instructions, not depicted content.

For designed images, compose the information and imagery together. Establish
reading order, relative prominence, coherent groups and readable text at the
chosen pixels. Write the intended copy rather than placeholders such as
"headline here" or "some explanatory text". Do not import unrelated names,
watermarks or copy from a reference.

## Caption

Answer with one JSON object containing exactly `{"caption":"..."}`. The caption is
cohesive prose, with spatial relations in words and any visible wording quoted
exactly. It may contain paragraphs; do not add coordinates or JSON-layout fields
to a caption just to make it look structured.

## Layout

Answer with one readable JSON object with `summary`, a flat `items` array,
concrete `width` and `height`, `bbox_coord_space="abs"` and `bbox_format="xyxy"`.
These six keys sit at the top level; a Layout has no top-level `caption`:

```json
{
  "width": 1024,
  "height": 1536,
  "bbox_coord_space": "abs",
  "bbox_format": "xyxy",
  "summary": {"meta": ["poster"], "canvas": "single", "caption": "The whole image and its style."},
  "items": [
    {"label": "text", "caption": "How the words look.", "text": "EXACT WORDS", "bbox": [128, 96, 896, 288]},
    {"label": "object", "caption": "How the object looks.", "bbox": [256, 640, 768, 1408]}
  ]
}
```

**Draft boxes use `[x_min,y_min,x_max,y_max]` in actual pixels, from the canvas
top-left.** Keep x in `0..width` and y in `0..height`. Do not use normalized
coordinates or manually center, reorder or quantize the boxes.

Honor supplied dimensions. With only an aspect ratio, choose compatible pixel
dimensions; without either, use 1024×1024 unless the requested medium suggests
another shape. MOSS-Image dimensions must be positive multiples of 16.

- `width`, `height`: positive integer multiples of 16, equal to the eventual image
  dimensions.
- `summary`: `caption` (nonempty global image description), optionally `meta`
  (array of medium/genre tags) and `canvas` (`single`, `multi_panel`, or `collage`).
- `items`: nonempty flat array. Each item has a `label` (`object`, `character`, or
  `text`), nonempty `caption`, and a four-number `bbox`. An item's caption describes
  its visible appearance and, for text, typography and surface treatment.
- `text`: required for a `text` item; the exact visible characters, with literal
  intended line breaks represented as JSON `\n`. Use separate text items when
  placement must be independent. Mixed font styles alone do not require splitting.
- `tags` and `palette`: optional string arrays on items; `palette` on summary is
  also supported. Use only when they contribute useful visual information.

Do not add `children`, `parent`, `id`, scene-schema `elements`, or presentation
notes. Positions are global canvas coordinates, including text on an object.

Use boxes for coherent visible units, splitting only when independent placement
matters. Grouping should preserve content and relationships, not erase them.
Natural occlusion and overlap are allowed. A box is a placement annotation,
not an instruction to draw a border, panel or sticker. Describe appearance in
captions and global treatment in `summary.caption`; let boxes agree with any
spatial descriptions. Keep exact visible strings in text items' `text` fields.

Read every visible string against the requested wording. Check that line breaks,
text length and glyph scale can fit the box at the declared resolution. Enlarge
or recompose an unreadable layout rather than silently deleting copy or stretching
glyphs.

## Deliver

Output only the JSON object: no Markdown fences, no explanation before or after it.
