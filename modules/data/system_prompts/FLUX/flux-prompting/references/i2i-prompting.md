---
name: i2i-prompting
description: Image-to-image editing prompts with FLUX models
---

# Image-to-Image (I2I) Prompting

Guide to effective image-to-image editing with FLUX models.

## Overview

All FLUX.2 models support image-to-image editing via reference images:
- **FLUX.2 [klein]**: Up to 4 reference images - fast editing
- **FLUX.2 [pro]**: Up to 8 reference images - balanced quality/speed
- **FLUX.2 [max]**: Up to 8-10 reference images - highest quality editing
- **FLUX.2 [flex]**: Up to 8 reference images - best for typography edits

Simply provide your source image as a reference and describe the desired changes. The model understands image context and can modify specific elements while preserving others.

> **Note:** FLUX.2 models are recommended for image editing. They provide better results than the older FLUX.1 Kontext models.

## Providing Images

**Preferred: Use URLs directly** - simpler and more convenient than base64.

When you have an image URL, pass it directly to `input_image`:

```json
{
  "prompt": "Change the background to a beach sunset",
  "input_image": "https://example.com/photo.jpg"
}
```

The API fetches URLs automatically. Both URL and base64 work, but URLs are recommended when available.

## Basic Edit Patterns

### Simple Modifications
Direct, single-change instructions:

```
Change the car color to red
```

```
Make the sky a dramatic sunset
```

```
Add snow to the ground
```

### Attribute Changes
Modifying specific characteristics:

```
Change her hair color to platinum blonde
```

```
Make the building taller
```

```
Age the person to appear 20 years older
```

## Controlled Editing

### Explicit Preservation
When you need to keep specific elements unchanged:

```
Change the background to a beach scene while keeping the subject's
pose, clothing, and expression exactly the same
```

```
Transform the daytime photo to nighttime, maintaining the exact
composition, colors of the subject's outfit, and lighting direction
```

### Style Preservation
Preventing unwanted style shifts:

```
Change the season to autumn with falling leaves, but maintain
the photograph's realistic style and color grading
```

```
Add rain effects to the scene while preserving the painting's
impressionist brushwork and color palette
```

## Transformation Types

### Environmental Changes

#### Time of Day
```
Convert to golden hour lighting with warm tones and long shadows,
keeping all other elements identical
```

```
Transform to blue hour with city lights beginning to glow,
maintaining the exact composition
```

#### Weather
```
Add heavy rain with wet reflections on surfaces, dark overcast sky
```

```
Create a foggy atmosphere with reduced visibility, mysterious mood
```

#### Season
```
Transform to winter with snow covering surfaces, bare trees,
cold blue color cast
```

```
Change to spring with cherry blossoms, fresh green leaves,
soft warm lighting
```

### Style Transfer

#### Artistic Movements
```
Transform into Art Nouveau style with flowing organic lines,
decorative patterns, and muted earth tones
```

```
Convert to Pop Art style with bold primary colors, halftone dots,
and high contrast graphic treatment
```

#### Artist References
```
Reimagine in the style of Monet with visible brushstrokes,
soft focus, and impressionist color harmony
```

```
Transform to match Edward Hopper's style with dramatic lighting,
urban isolation feeling, and muted palette
```

#### Medium Conversion
```
Convert this photograph to a detailed pencil sketch with
careful shading and visible line work
```

```
Transform into a watercolor painting with soft edges,
transparent washes, and paper texture visible
```

### Subject Modifications

#### Clothing Changes
```
Change the outfit to a formal black suit with white shirt and red tie
```

```
Replace the casual clothes with traditional Japanese kimono in blue floral pattern
```

#### Expression Changes
```
Change the expression to a warm genuine smile
```

```
Make the expression more serious and contemplative
```

#### Age Modifications
```
Age the subject to appear as a wise elderly person with grey hair and wrinkles
```

```
Make the subject appear younger, around 25 years old
```

### Object Editing

#### Addition
```
Add a vintage leather briefcase in the subject's left hand
```

```
Place a steaming cup of coffee on the table
```

#### Removal
```
Remove the background people, replace with empty street
```

```
Remove the text/logo from the shirt, replace with solid color
```

#### Replacement
```
Replace the modern car with a 1960s vintage Mustang in cherry red
```

```
Swap the coffee mug for an ornate teacup with floral pattern
```

## Text Editing

### Adding Text
```
Add a neon sign reading "OPEN 24 HOURS" in the window,
glowing red letters with blue outline
```

```
Include a wooden sign with hand-painted text "Welcome Home"
mounted above the door
```

### Modifying Text
```
Change the store sign to read "BAKER'S DOZEN" in the same style
```

```
Update the poster text to "SUMMER SALE 2025" maintaining the design
```

## Complex Multi-Step Edits

For dramatic transformations, consider breaking into steps:

### Step-by-Step Approach
Instead of:
```
Transform this modern office into a Victorian library with completely
different furniture, add a fireplace, change the lighting to candlelit,
and age the photograph
```

Try sequential edits:
1. `Change the furniture style to Victorian antique pieces`
2. `Add a stone fireplace on the right wall`
3. `Transform lighting to warm candlelit atmosphere`
4. `Apply vintage photograph aesthetic with sepia tones`

## Common Pitfalls

### Avoid Vague Instructions
```
Bad: Make it look better
Good: Increase contrast, add warm color grading, sharpen details
```

### Be Specific About Scope
```
Bad: Change the background
Good: Replace the office background with a tropical beach at sunset,
      maintaining the subject's exact position and lighting direction
```

### Explicit Style Preservation
```
Bad: Make it nighttime
Good: Transform to nighttime while maintaining the photorealistic style,
      add appropriate artificial lighting sources
```

## Best Practices

1. **Start Simple** - Begin with single-element changes
2. **Be Explicit** - State what should change AND what should stay
3. **Reference Context** - Mention existing elements when relevant
4. **Iterate** - Refine through multiple small edits rather than one large one
5. **Preserve Deliberately** - Always specify style/composition preservation needs
