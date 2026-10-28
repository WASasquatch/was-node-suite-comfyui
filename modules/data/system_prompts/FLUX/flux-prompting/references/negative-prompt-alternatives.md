---
name: negative-prompt-alternatives
description: Positive alternatives to negative prompts
---

# Negative Prompt Alternatives

FLUX does not support negative prompts. This guide provides positive alternatives for common negative prompt patterns.

## Why No Negative Prompts?

Negative prompts can actually make models focus MORE on unwanted elements. Instead, describe exactly what you DO want - this gives clearer direction and better results.

## Replacement Strategy

For any unwanted element:
1. Identify what you don't want
2. Ask: "What would be there instead?"
3. Describe the positive alternative

## Common Replacements

### People/Crowds

| Instead of | Use |
|-----------|-----|
| "no people" | "empty", "deserted", "solitary", "abandoned" |
| "no crowds" | "quiet", "peaceful", "secluded", "private" |
| "without background people" | "isolated subject", "clean background", "solo figure" |

**Example:**
```
Bad: A beach scene, no people
Good: A deserted beach at dawn, pristine untouched sand, solitary seagull
```

### Skin/Appearance

| Instead of | Use |
|-----------|-----|
| "no makeup" | "natural skin", "bare face", "fresh-faced" |
| "no blemishes" | "clear skin", "smooth complexion", "healthy glow" |
| "no wrinkles" | "youthful skin", "smooth features" |

**Example:**
```
Bad: Portrait of woman, no makeup, no blemishes
Good: Portrait of a woman with natural clear skin, fresh-faced with a healthy glow
```

### Accessories

| Instead of | Use |
|-----------|-----|
| "no glasses" | "visible eyes", "unobstructed gaze", "clear eye contact" |
| "no hat" | "bare head", "visible hair", "uncovered head" |
| "no jewelry" | "minimal accessories", "understated", "unadorned" |

**Example:**
```
Bad: Man portrait, no glasses, no hat
Good: Portrait of a man with clear direct gaze, wind-swept visible hair
```

### Colors

| Instead of | Use |
|-----------|-----|
| "no color" | "monochrome", "black and white", "grayscale" |
| "not colorful" | "muted tones", "subdued palette", "desaturated" |
| "no bright colors" | "neutral tones", "earth tones", "soft pastels" |

**Example:**
```
Bad: Landscape photo, no bright colors
Good: Landscape in muted earth tones, soft morning light, desaturated palette
```

### Text/Watermarks

| Instead of | Use |
|-----------|-----|
| "no text" | "clean surfaces", "unmarked", "text-free" |
| "no watermark" | "pristine image", "clean composition" |
| "no logos" | "unbranded", "plain", "logo-free surface" |

**Example:**
```
Bad: Product photo, no watermark, no text
Good: Clean product photography with pristine unmarked surfaces, minimal unbranded design
```

### Style/Era

| Instead of | Use |
|-----------|-----|
| "not modern" | "traditional", "classical", "vintage", "historical" |
| "no CGI look" | "photorealistic", "authentic", "natural", "organic" |
| "not cartoonish" | "realistic", "lifelike", "naturalistic" |

**Example:**
```
Bad: Building design, not modern, no futuristic elements
Good: Traditional Victorian architecture with classical ornate details and period-accurate features
```

### Quality/Artifacts

| Instead of | Use |
|-----------|-----|
| "no blur" | "sharp focus", "crisp details", "tack-sharp" |
| "no noise" | "clean image", "smooth gradients", "low ISO" |
| "no artifacts" | "pristine quality", "clean render", "flawless" |

**Example:**
```
Bad: Portrait, no blur, no noise
Good: Tack-sharp portrait with pristine image quality, smooth skin tones, crisp details
```

### Objects

| Instead of | Use |
|-----------|-----|
| "no cars" | "pedestrian area", "car-free zone", "walking street" |
| "no buildings" | "open landscape", "natural scenery", "wilderness" |
| "no furniture" | "empty room", "bare space", "minimalist interior" |

**Example:**
```
Bad: Street scene, no cars, no modern buildings
Good: Historic cobblestone walking street lined with traditional stone buildings from the 1800s
```

### Weather/Environment

| Instead of | Use |
|-----------|-----|
| "no rain" | "clear sky", "dry weather", "sunny day" |
| "no clouds" | "clear blue sky", "cloudless", "perfect visibility" |
| "not dark" | "well-lit", "bright", "daylight", "illuminated" |

**Example:**
```
Bad: Outdoor portrait, no rain, no clouds, not dark
Good: Outdoor portrait under clear blue sky on a bright sunny day, perfect natural lighting
```

### Composition

| Instead of | Use |
|-----------|-----|
| "no distractions" | "clean composition", "focused framing", "minimal elements" |
| "nothing in background" | "solid background", "isolated subject", "clean backdrop" |
| "no clutter" | "organized", "tidy", "minimal", "streamlined" |

**Example:**
```
Bad: Product shot, no distractions, nothing in background
Good: Product on clean white seamless backdrop, isolated subject, minimal focused composition
```

## Complex Replacement Examples

### Original Negative-Heavy Prompt
```
Portrait of a woman, no glasses, no makeup, no wrinkles, no blemishes,
no bright colors, no distracting background, no harsh lighting
```

### Positive Rewrite
```
Portrait of a youthful woman with clear natural skin and visible bright eyes,
fresh-faced with a healthy glow, wearing muted earth tones against a soft
blurred neutral background, gentle diffused lighting creating soft shadows
```

### Original Negative-Heavy Prompt
```
Landscape photo, no people, no buildings, no power lines, no modern elements,
no overcast sky, no dead trees
```

### Positive Rewrite
```
Pristine wilderness landscape with lush green living forest, clear blue sky,
untouched natural scenery stretching to the horizon, peaceful solitude with
only birdsong and wind, golden hour sunlight filtering through healthy foliage
```

## Quick Reference Card

| Unwanted | Positive Alternative |
|----------|---------------------|
| No people | Empty, solitary, deserted |
| No makeup | Natural, fresh-faced, bare |
| No text | Clean, unmarked, pristine |
| No blur | Sharp, crisp, tack-sharp |
| No modern | Traditional, vintage, classical |
| No dark | Bright, well-lit, luminous |
| No busy | Minimal, clean, focused |
| No artificial | Natural, organic, authentic |
