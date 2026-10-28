---
name: hex-color-prompting
description: Using hex color codes for precise color specification
---

# Hex Color Prompting

FLUX supports hex color codes (#RRGGBB) for precise color specification, essential for brand consistency and exact color matching.

## Syntax

Include hex codes directly in your prompt with descriptive names:

```
A modern living room with walls painted in #2C3E50 (dark blue-gray),
accent pillows in #E74C3C (vibrant red), and a #F39C12 (warm amber)
throw blanket on a #ECF0F1 (off-white) sofa
```

## Signal Keywords

Use these keywords to indicate color specification:

```
color #02eb3c
hex #edfa3c
in #FF5733
using color code #3498DB
```

## Best Practices

### 1. Always Pair with Description

Never use hex codes alone - include the color name:

```
Good: #FF6B6B (coral pink)
Bad: #FF6B6B
```

### 2. Associate with Specific Objects

Clearly connect colors to their targets:

```
A product shot featuring a smartphone with a #1DA1F2 (Twitter blue) case,
resting on a #14171A (near black) matte surface
```

### 3. Limit Color Palette

3-5 colors typically work best. Too many can confuse the model:

```
Color palette for the scene: #2ECC71 (emerald green), #3498DB (sky blue),
#F1C40F (sunflower yellow), #FFFFFF (pure white)
```

## Use Cases

### Brand Colors

```
Corporate office reception with brand colors prominently featured:
walls in #0066CC (company blue), accent furniture in #FF6600 (company orange),
logo displayed in #FFFFFF (white) against the blue backdrop
```

### Interior Design

```
Scandinavian minimalist bedroom with #F5F5F5 (warm white) walls,
#8B4513 (saddle brown) wooden headboard and nightstands,
#708090 (slate gray) linen bedding, and #DAA520 (goldenrod) accent lamp
```

### Fashion

```
Editorial fashion photo: model wearing #000000 (black) cashmere turtleneck,
#FF4500 (orange-red) wide-leg wool pants, #C0C0C0 (silver) geometric earrings,
against a #F0F0F0 (light gray) studio backdrop
```

### Product Design

```
Premium headphones product shot: #1C1C1E (space gray) aluminum body,
#F5F5F7 (silver) mesh ear cups, #FF9500 (iOS orange) accent ring around controls
```

### Digital Art

```
Synthwave cityscape: #FF00FF (magenta) and #00FFFF (cyan) neon signs,
#1A1A2E (deep navy) night sky, #E94560 (hot pink) setting sun on horizon,
#16213E (dark blue) building silhouettes, rain-slicked streets reflecting lights
```

### Data Visualization

```
Infographic showing market share: segments in #2ECC71 (green) for growth,
#E74C3C (red) for decline, #3498DB (blue) for stable, #95A5A6 (gray) for other,
clean #FFFFFF (white) background
```

## Gradient Colors

Specify gradients with start and end colors:

```
Abstract background starting with color #02eb3c (bright green) and
finishing with color #edfa3c (lime yellow), smooth horizontal gradient
```

```
Sunset sky gradient from #FF6B6B (coral) at horizon through
#FFA07A (light salmon) to #87CEEB (sky blue) at top
```

## Color Harmony Patterns

### Complementary (Opposite on color wheel)
```
Scene using complementary colors: #3498DB (blue) dominant with
#E67E22 (orange) accents for visual pop
```

### Analogous (Adjacent colors)
```
Harmonious palette using analogous colors: #9B59B6 (purple),
#8E44AD (deep purple), #3498DB (blue) - flowing naturally together
```

### Triadic (Evenly spaced)
```
Vibrant triadic scheme: #E74C3C (red), #F1C40F (yellow),
#3498DB (blue) - balanced and dynamic
```

### Monochromatic (Single hue variations)
```
Sophisticated monochromatic blue: #1A5276 (dark navy), #2980B9 (medium blue),
#85C1E9 (light blue), #D4E6F1 (pale blue) - elegant depth
```

## Combining with JSON Structured Prompts

```json
{
  "scene": {
    "setting": "modern tech startup office",
    "mood": "innovative, energetic"
  },
  "colors": {
    "primary": "#6C5CE7 (electric purple)",
    "secondary": "#00CEC9 (teal)",
    "accent": "#FD79A8 (pink)",
    "neutral": "#DFE6E9 (light gray)",
    "dark": "#2D3436 (charcoal)"
  },
  "application": {
    "walls": "neutral #DFE6E9",
    "furniture": "dark #2D3436",
    "accent_pieces": "primary #6C5CE7",
    "plants": "secondary #00CEC9 pots"
  }
}
```

## Common Brand Color References

For reference only - always verify current brand guidelines:

```
# Social Media
Twitter/X Blue: #1DA1F2
Facebook Blue: #1877F2
Instagram Gradient: #833AB4 to #FD1D1D
LinkedIn Blue: #0A66C2

# Tech
Apple Gray: #1C1C1E
Google Blue: #4285F4
Microsoft Blue: #00A4EF
Amazon Orange: #FF9900

# Design
Figma Purple: #A259FF
Dribbble Pink: #EA4C89
Behance Blue: #1769FF
```

## Troubleshooting

### Color Not Accurate
- Add the color name alongside hex
- Specify the exact object the color applies to
- Use fewer total colors in the prompt

### Color Bleeding
- Clearly delineate which objects get which colors
- Use spatial descriptions: "the LEFT chair in #color"

### Muddy Colors
- Check hex code accuracy
- Specify lighting that won't shift colors
- Use "maintaining exact color #XXXXXX" for emphasis
