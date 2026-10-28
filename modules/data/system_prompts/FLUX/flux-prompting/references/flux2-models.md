---
name: flux2-models
description: Prompting guidelines specific to FLUX.2 model family
---

# FLUX.2 Model Family

Complete guide to FLUX.2 variants and their optimal prompting strategies.

> **Key Feature:** All FLUX.2 models natively support both text-to-image generation AND image-to-image editing via reference images. There's no need to use legacy FLUX.1 Kontext models for editing tasks.

## Model Overview

| Model | Parameters | Best For | Speed | Reference Images |
|-------|------------|----------|-------|------------------|
| [klein] | 4B/9B | Fast iterations, previews, quick edits | Fastest | Up to 4 |
| [max] | - | Highest quality generation & editing | Slowest | Up to 8-10 |
| [pro] | - | Production balanced | Medium | Up to 8 |
| [flex] | - | Typography, text rendering | Medium | Up to 8 |
| [dev] | - | Local development | Varies | Varies |

## FLUX.2 [klein] - Fast Generation

Best for rapid prototyping, previews, and high-volume generation.

### Characteristics
- 4B or 9B parameter versions available
- Sub-second generation times
- Optimized for speed over maximum detail
- **No prompt upsampling** - be descriptive yourself
- Supports up to 4 reference images

### Prompting Style: Narrative Prose

Klein responds best to descriptive, narrative-style prompts with emphasis on lighting and atmosphere.

### Example Prompt
```
A cozy coffee shop interior bathed in warm afternoon light, steam rising lazily
from ceramic cups, worn leather armchairs arranged around small wooden tables,
bookshelves lining exposed brick walls, the soft atmosphere of a quiet afternoon
with dust motes floating in sunbeams through tall windows
```

### Tips for [klein]
- Write like a novelist describing a scene
- Front-load your subject (word order critical)
- Emphasize lighting descriptions heavily
- Keep prompts moderately detailed (40-70 words)

## FLUX.2 [max] - Highest Quality

Premium model for final production assets and maximum detail.

### Characteristics
- Highest detail and coherence
- Best editing consistency
- Vast world knowledge
- Includes grounding search (real-time web data)
- Strongest prompt following
- Supports up to 8 reference images (API), 10 (playground)

### Prompting Style: Technical + Descriptive

[max] excels with detailed technical specifications combined with descriptive prose.

### Example Prompt
```
Portrait of a weathered fisherman, age 70, deep wrinkles telling stories of
decades at sea, salt-and-pepper beard with streaks of white, wearing a navy
cable-knit sweater with visible wool texture. Shot on Hasselblad X2D with
90mm f/2.8 lens at f/4, golden hour natural light from the left creating
strong rim lighting, shallow depth of field with soft bokeh from harbor
lights behind, Kodak Portra 400 color science with natural grain
```

### Tips for [max]
- Include camera and lens specifications for photorealism
- Specify film stock or digital sensor characteristics
- Use technical photography terms (aperture, focal length)
- Leverage grounding search for current events: "news photo of [recent event]"

## FLUX.2 [pro] - Production Balanced

Optimal balance of quality and speed for production workflows.

### Characteristics
- Good quality-to-speed ratio
- Reliable, consistent output
- Suitable for batch processing
- Supports prompt upsampling
- Supports up to 8 reference images

### Prompting Style: Balanced Detail

Standard detailed prompts work well without excessive technical specification.

### Example Prompt
```
A modern minimalist living room with floor-to-ceiling windows overlooking
a city skyline at dusk, clean white furniture with subtle textures, a single
statement plant in the corner, warm ambient lighting from hidden sources,
architectural photography style with clean lines and balanced composition
```

### Tips for [pro]
- Balance specificity with generation speed
- Good for template-based prompt systems
- Enable prompt upsampling for enhanced results
- Consistent quality for production pipelines

## FLUX.2 [flex] - Typography Specialist

Optimized for text rendering and typographic content.

### Characteristics
- Superior text rendering quality
- Handles multiple text elements
- Adjustable steps (1-50) and guidance (1.5-10)
- Best for signage, posters, UI mockups
- Supports up to 8 reference images

### Prompting Style: Typography-Focused

Always quote text and specify font characteristics explicitly.

### Example Prompt
```
A modern minimalist poster design with the headline "DESIGN SUMMIT 2025"
in bold condensed sans-serif typography centered in the upper third,
subtitle "Innovation Meets Creativity" in lighter weight below,
date "MARCH 15-17" in small caps at the bottom, all text in white
on a gradient background transitioning from deep purple #4A0080 to
coral #FF6B6B, clean geometric accent lines, professional print quality
```

### Tips for [flex]
- Always quote exact text: `"Your Text Here"`
- Specify font style: serif, sans-serif, script, display, monospace
- Describe text hierarchy: headline, subhead, body
- Include placement: centered, left-aligned, upper third
- Adjust steps (higher = better quality) and guidance (higher = stricter)

## FLUX.2 [dev] - Local Development

For local development, testing, and non-commercial use.

### Characteristics
- Open weights on Hugging Face
- Runs locally (~13GB VRAM recommended)
- Full customization available
- Free for non-commercial use
- Base variants available (undistilled) for fine-tuning

### Prompting Style: Standard

Same prompting patterns as [pro] work well.

### Tips for [dev]
- Use for development and testing before production
- Experiment with prompt variations
- Good for fine-tuning experiments
- Check license for commercial use restrictions

## Image-to-Image Editing with FLUX.2

All FLUX.2 models support image editing via reference images. This replaces the need for legacy FLUX.1 Kontext models.

### How It Works

1. Provide your source image(s) as reference images
2. Describe the desired changes in your prompt
3. The model preserves context while applying edits

### Example: Style Transfer
```
Reference: [your source image]
Prompt: Transform this image into a watercolor painting style,
maintaining the exact composition and subject positioning
```

### Example: Object Modification
```
Reference: [your source image]
Prompt: Change the car color to red while keeping everything else identical
```

### Example: Character Consistency
```
Reference: [character reference image]
Prompt: The same person from the reference image walking through
a busy Tokyo street at night, neon lights reflecting on wet pavement
```

### Model Selection for Editing

| Use Case | Recommended Model |
|----------|-------------------|
| Quick iterations/previews | FLUX.2 [klein] |
| Production quality edits | FLUX.2 [pro] |
| Maximum quality/complex edits | FLUX.2 [max] |
| Text/typography edits | FLUX.2 [flex] |
