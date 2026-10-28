# Pixel Layout draft

The public MOSS-Image Layout vocabulary is `summary` plus a flat `items` array.
Use this one schema for new drafts. The source is readable JSON; the compiler
handles the model-facing compact form. There are no nested/local coordinate
systems, percentage boxes or editor IDs.

## Fields

- `width`, `height`: positive integer multiples of 16, equal to the eventual image
  dimensions. They describe the draft canvas and are omitted from the compiled
  prompt. Do not rescale boxes merely by changing these fields.
- `bbox_coord_space`: exactly `"abs"`.
- `bbox_format`: exactly `"xyxy"`.
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
notes. The compiler rejects unrecognized fields rather than silently discarding
creative intent. Positions are global canvas coordinates, including text on an
object. Several items may share or overlap a box when the intended image does.

## Example

A 1536×1024 landscape campaign image, with a ceramic cup and exact Chinese copy:

```json
{
  "width": 1536,
  "height": 1024,
  "bbox_coord_space": "abs",
  "bbox_format": "xyxy",
  "summary": {
    "meta": ["product campaign", "studio photography"],
    "canvas": "single",
    "caption": "A warm studio campaign for a handmade ceramic cup, with an ivory backdrop, soft side lighting and restrained dark-green typography. Keep the ceramic texture tactile and the composition calm."
  },
  "items": [
    {
      "label": "object",
      "caption": "An ivory ceramic cup with a rounded handle, fine speckled glaze and a soft contact shadow, photographed at a three-quarter angle.",
      "bbox": [176, 224, 800, 864]
    },
    {
      "label": "text",
      "caption": "Large dark-green Chinese title in crisp, moderately heavy lettering, with generous spacing and no backing panel.",
      "text": "把日常，过成诗。",
      "bbox": [928, 272, 1408, 448]
    },
    {
      "label": "text",
      "caption": "Smaller dark-green Chinese supporting copy, comfortably legible on two lines.",
      "text": "一杯温暖\n一刻从容",
      "bbox": [928, 512, 1376, 656]
    }
  ]
}
```

For the cup, conversion yields `[-288,-592,352,32]`: its original top/bottom
coordinates minus H/2, then left/right coordinates minus W/2, in `yxyx` order.
The model should never calculate these values itself. For fractional draft edges,
the compiler floors left/top and ceils right/bottom before subtracting the center.

## Composition checks

Read every visible string against the requested wording. Check that line breaks,
text length and glyph scale can fit the box at the declared resolution. A short
Chinese title needs a different footprint from a paragraph of Latin text. Enlarge
or recompose an unreadable layout rather than silently deleting copy or stretching
glyphs. For a knowledge graphic, develop actual explanations and explanatory
imagery; do not substitute empty modules. For a poster, keep its intended visual
identity and information hierarchy rather than forcing infographic organization.

When revising an existing draft, reread it, preserve unrequested content and change
the specific affected captions or boxes. Keep a single coordinate convention
throughout. Recompile after every geometry or canvas edit.
