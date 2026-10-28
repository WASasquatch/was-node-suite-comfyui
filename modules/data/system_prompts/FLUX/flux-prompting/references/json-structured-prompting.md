---
name: json-structured-prompting
description: Using JSON format for complex scene composition
---

# JSON Structured Prompting

For complex scenes with multiple elements, spatial relationships, or production automation, use JSON-structured prompts.

## When to Use

- Multiple characters with distinct attributes
- Precise spatial positioning
- Complex scene composition
- Reproducible, template-based prompts
- Programmatic prompt generation
- Production workflows with variable substitution

## Basic Structure

```json
{
  "scene": {
    "setting": "description of environment",
    "time": "time of day/period",
    "mood": "atmospheric quality"
  },
  "subjects": [
    {
      "type": "person/object/animal",
      "description": "detailed description",
      "position": "location in frame",
      "action": "what they're doing"
    }
  ],
  "style": {
    "medium": "photography/painting/illustration",
    "technique": "specific style details",
    "reference": "artist or style reference"
  },
  "technical": {
    "camera": "camera and lens",
    "lighting": "lighting setup",
    "composition": "framing details"
  },
  "colors": ["#hex1", "#hex2"]
}
```

## Single Subject Example

```json
{
  "scene": {
    "setting": "cozy home office with bookshelves",
    "time": "late afternoon",
    "mood": "focused, peaceful"
  },
  "subjects": [
    {
      "type": "person",
      "description": "woman in her 30s, dark curly hair in loose bun, wearing casual cream sweater",
      "position": "seated at desk, center frame",
      "action": "typing on laptop, slight smile of concentration"
    }
  ],
  "style": {
    "medium": "photography",
    "technique": "lifestyle editorial",
    "reference": "kinfolk magazine aesthetic"
  },
  "technical": {
    "camera": "Sony A7III with 50mm f/1.8",
    "lighting": "soft natural window light from left",
    "composition": "medium shot, rule of thirds"
  }
}
```

## Multi-Character Scene

```json
{
  "scene": {
    "setting": "Victorian-era drawing room with ornate wallpaper and antique furniture",
    "time": "evening, candlelit",
    "mood": "tense, mysterious"
  },
  "subjects": [
    {
      "id": "detective",
      "type": "person",
      "description": "tall man in his 50s, sharp features, grey at temples, wearing brown tweed suit",
      "position": "standing center-left, facing right",
      "action": "examining a letter with magnifying glass, intense focus"
    },
    {
      "id": "lady",
      "type": "person",
      "description": "elegant woman in her 40s, auburn hair in Victorian updo, emerald green evening dress",
      "position": "seated on chaise lounge, right side",
      "action": "watching the detective with concealed anxiety, hands clasped"
    },
    {
      "id": "butler",
      "type": "person",
      "description": "elderly man in formal butler attire, stoic expression",
      "position": "background, near doorway",
      "action": "standing at attention, observing"
    }
  ],
  "style": {
    "medium": "oil painting",
    "technique": "classical realism with dramatic lighting",
    "reference": "Victorian narrative painting, John Singer Sargent"
  },
  "technical": {
    "lighting": "warm candlelight as key, cool moonlight through window as fill",
    "composition": "triangular arrangement of figures, detective at apex"
  }
}
```

## Product Scene with Colors

```json
{
  "scene": {
    "setting": "minimalist product photography studio",
    "mood": "clean, premium, aspirational"
  },
  "subjects": [
    {
      "type": "product",
      "description": "sleek wireless earbuds in charging case",
      "position": "center, slightly angled",
      "details": "matte finish, subtle branding"
    }
  ],
  "style": {
    "medium": "commercial photography",
    "technique": "high-end product shot",
    "reference": "Apple product photography"
  },
  "technical": {
    "camera": "Phase One with 120mm macro",
    "lighting": "large softbox overhead, subtle fill from below",
    "composition": "centered, hero product shot"
  },
  "colors": {
    "product": "#1A1A2E",
    "accent": "#E94560",
    "background": "#FFFFFF"
  }
}
```

## Converting JSON to Natural Language

Flatten your JSON into flowing prose for the actual prompt:

### From JSON
```json
{
  "subjects": [
    {
      "type": "person",
      "description": "elderly craftsman with weathered hands",
      "position": "seated at workbench",
      "action": "carefully carving wood"
    }
  ],
  "scene": { "setting": "traditional workshop", "time": "morning" },
  "technical": { "lighting": "natural window light from right" }
}
```

### To Prompt
```
An elderly craftsman with weathered hands seated at his workbench in a
traditional workshop, carefully carving wood with focused precision.
Morning natural light streams through the window from the right,
illuminating the wood shavings and tools scattered across the worn surface.
```

## Template Variables

Use JSON structure for template-based generation:

```json
{
  "template": "product_hero",
  "variables": {
    "product_name": "{{PRODUCT_NAME}}",
    "product_color": "{{PRODUCT_COLOR}}",
    "brand_primary": "{{BRAND_HEX_1}}",
    "brand_secondary": "{{BRAND_HEX_2}}",
    "background_style": "{{BG_STYLE}}"
  },
  "prompt_template": "Professional product photography of {{PRODUCT_NAME}} in {{PRODUCT_COLOR}}, brand colors {{BRAND_HEX_1}} and {{BRAND_HEX_2}} accents, {{BG_STYLE}} background, commercial quality"
}
```

## Spatial Relationships

Define explicit spatial relationships:

```json
{
  "composition": {
    "layout": "triangular",
    "focal_point": "center-left intersection",
    "depth_layers": [
      {
        "layer": "foreground",
        "elements": ["flowers in vase"],
        "focus": "soft blur"
      },
      {
        "layer": "midground",
        "elements": ["main subject"],
        "focus": "sharp"
      },
      {
        "layer": "background",
        "elements": ["window", "garden view"],
        "focus": "soft blur"
      }
    ]
  }
}
```

## Best Practices

1. **Use IDs for References** - Give subjects IDs when they interact
2. **Separate Concerns** - Keep scene, subjects, style, and technical distinct
3. **Be Consistent** - Use the same terminology throughout
4. **Include All Details** - Don't assume, specify everything
5. **Flatten for Execution** - Convert to natural language before sending to model
6. **Version Templates** - Track template versions for reproducibility
