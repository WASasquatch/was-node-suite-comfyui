---
name: flux1-models
description: Prompting guidelines for FLUX.1 model family
---

# FLUX.1 Model Family

> **Tip:** FLUX.2 models are the latest generation and recommended for most use cases. FLUX.1 models are still available for specific needs.

Guide to FLUX.1 models and their specialized capabilities.

## Model Overview

| Model | Purpose | Notes |
|-------|---------|-------|
| FLUX1.1 [pro] | Text-to-image | FLUX.2 [pro] offers improved results |
| FLUX.1 Kontext | Image-to-image | FLUX.2 with references recommended |
| FLUX.1 Kontext Max | Image-to-image | FLUX.2 [max] with references recommended |
| FLUX.1 Fill | Inpainting | Useful for specific inpainting tasks |

## FLUX1.1 [pro]

Fast and reliable text-to-image generation.

### Characteristics
- Strong prompt adherence
- Production-grade architecture
- Consistent, reliable results
- Scalable for high-volume
- Pricing: $0.04 per image

### Prompting Style

Standard descriptive prompts with clear subject and style specification.

### Example Prompt
```
A golden retriever puppy playing in autumn leaves, warm afternoon sunlight,
shallow depth of field with bokeh background, joyful expression, professional
pet photography style
```

## FLUX.1 Kontext - Image Editing

> **Recommendation:** FLUX.2 models with reference images provide improved editing results.

Context-aware image-to-image editing model for transformations and modifications.

### Characteristics
- Understands image context
- Preserves unedited regions
- Style transfer capabilities
- Object modification
- Basic to complex transformations

### Prompting Strategies

#### Basic Edits
Simple, direct instructions work best:

```
Change the color of the car to red
```

```
Make the sky sunset orange and pink
```

#### Controlled Edits
Be explicit about what to preserve:

```
Change the setting to nighttime while maintaining the exact same
composition and the painting's artistic style
```

#### Complex Transformations
For dramatic changes, be specific about preservation:

```
Transform the modern office into a Victorian library with the same
furniture arrangement. Keep the window positions and overall room
proportions identical.
```

#### Style Transfer
Reference specific artistic movements:

```
Transform this photograph into a Bauhaus art style with geometric
shapes and primary colors, maintaining the original composition
and subject positioning
```

#### Text Editing
Describe text placement and integration:

```
Add the text "OPEN" as a neon sign in the window, red glowing letters
with slight reflection on the glass, matching the nighttime atmosphere
```

### Tips for Kontext
- Be explicit about what should NOT change
- Start with simpler edits, build complexity
- Specify style preservation when needed
- Use for incremental refinement

## FLUX.1 Kontext Max

Advanced multi-reference editing for complex compositions.

### Characteristics
- Handles up to 10 reference images
- Best editing consistency across references
- Complex scene composition
- Character consistency maintenance
- Rate limit: 6 concurrent requests

### Multi-Reference Prompting

#### Natural Language References
Describe relationships between images naturally:

```
The person from image 1 is sitting in the cafe from image 2,
wearing the outfit from image 3, with the lighting style of image 4
```

#### Explicit Indexing
Reference images by number for precision:

```
Replace the top half of the person in image 1 with the clothing
from image 2, maintaining the pose and background
```

### Tips for Kontext Max
- Plan your reference images carefully
- Use natural language for relationships
- Specify which elements come from which image
- Leverage for character consistency across scenes

## FLUX.1 Fill - Inpainting

Specialized tool for object removal and area completion.

### Characteristics
- Clean object removal
- Intelligent background completion
- Texture-aware filling
- Seamless blending

### Use Cases
- Remove unwanted objects from photos
- Complete partial images
- Replace specific regions
- Clean up image artifacts

### Prompting for Fill

Describe what should fill the masked area:

```
Fill with continuation of the brick wall texture and ivy
```

```
Complete with matching ocean waves and sandy beach
```

### Tips for Fill
- Provide context about surrounding areas
- Specify texture and pattern continuation
- Describe lighting consistency
- Use for cleanup and removal tasks
