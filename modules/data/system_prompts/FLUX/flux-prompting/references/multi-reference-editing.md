---
name: multi-reference-editing
description: Using multiple reference images for complex compositions
---

# Multi-Reference Image Editing

Guide to using multiple reference images for character consistency, style transfer, and complex compositions.

## Overview

FLUX.2 models support multiple reference images for advanced editing:

- **FLUX.2 [klein]**: Up to 4 reference images - fast editing
- **FLUX.2 [pro]**: Up to 8 via API - balanced quality/speed
- **FLUX.2 [max]**: Up to 8 via API, 10 in playground - highest quality
- **FLUX.2 [flex]**: Up to 8 via API - best for typography

> **Note:** FLUX.2 models are recommended over FLUX.1 Kontext Max for better results.

## Providing Images

**Preferred: Use URLs directly** - simpler and more convenient than base64.

Pass image URLs directly to `input_image`, `input_image_2`, etc.:

```json
{
  "prompt": "Person from image 1 wearing outfit from image 2",
  "input_image": "https://example.com/person.jpg",
  "input_image_2": "https://example.com/outfit.jpg"
}
```

The API fetches URLs automatically. Both URL and base64 work, but URLs are recommended when available.

## Reference Methods

### Natural Language Description

Describe relationships between images naturally:

```
The person from image 1 is sitting at the cafe table from image 2,
wearing the outfit from image 3, with the warm lighting style of image 4
```

### Explicit Indexing

Reference images by number for precision:

```
Replace the background of image 1 with the landscape from image 2,
maintaining the subject's exact position and lighting
```

```
Combine the face from image 1 with the hairstyle from image 2
on the body pose from image 3
```

## Use Cases

### Character Consistency

Maintain the same character across multiple scenes:

```
Input: Reference image of character
Prompt: The character from image 1 walking through a busy Tokyo street
at night, neon lights reflecting on wet pavement
```

For sequential consistency:

```
The same person from image 1, now seated at a desk in a modern office,
same clothing and hairstyle, different environment
```

### Style Transfer

Apply the style of one image to another:

```
Transform image 1 into the artistic style of image 2,
maintaining the original composition and subject
```

```
Apply the color grading and mood from image 2 to the scene in image 1
```

### Pose Guidance

Use a reference for body positioning:

```
The person from image 1 in the exact pose shown in image 2,
placed in the environment from image 3
```

### Object Composition

Combine elements from multiple images:

```
Place the product from image 1 on the table setting from image 2,
using the lighting style from image 3
```

### Background Replacement

```
Keep the subject from image 1 exactly as shown, replace the background
with the beach scene from image 2, match the lighting naturally
```

## Multi-Character Scenes

### Two Characters

```
Image 1 (person A) and image 2 (person B) having a conversation
at a coffee shop table, person A on the left gesturing, person B
on the right listening intently
```

### Group Composition

```
The three people from images 1, 2, and 3 standing together for a
group photo, arranged left to right in that order, friendly poses,
outdoor park setting
```

## Attribute Mixing

### Selective Attribute Transfer

```
The face and expression from image 1, the hairstyle from image 2,
wearing the outfit from image 3, in the pose from image 4
```

### Partial Transfer

```
Apply only the color palette from image 2 to image 1,
keeping all other aspects (style, composition, lighting) unchanged
```

## Collage Method

Use a collage input for layout guidance:

```
Arrange the scene using the layout shown in the collage input:
- Person from image 1 in the left position
- Object from image 2 in the center position
- Background element from image 3 filling the right side
```

## Best Practices

### 1. Clear Image Roles

Specify what each reference provides:

```
Image 1: face/identity reference
Image 2: pose/body reference
Image 3: style/aesthetic reference
Image 4: environment/background reference
```

### 2. Quality References

- Use high-quality, clear reference images
- Ensure good lighting in references
- Avoid heavily processed or filtered images

### 3. Consistent Lighting

When combining elements:

```
...ensure the lighting direction matches across all elements,
with main light source from the upper left
```

### 4. Resolution Awareness

For [pro] API with 9MP total limit:

- At 1MP output: up to 8 reference images comfortably
- Calculate: input images + output = total MP

### 5. Explicit Relationships

Don't assume - specify exactly how elements relate:

```
Vague: The person and the background together
Better: The person from image 1 standing in the foreground,
the beach from image 2 visible behind them at a distance
```

## Complex Composition Example

```
Create a scene combining:
- The woman from image 1 (keep exact face, expression, hair)
- Wearing the vintage dress from image 2 (exact pattern and cut)
- In the pose from image 3 (seated position, arm placement)
- Set in the library from image 4 (bookshelves, furniture)
- Using the warm lighting style from image 5 (golden hour quality)

Position her in the center of frame, medium shot, looking slightly
to the right with a thoughtful expression.
```

## Troubleshooting

### Elements Not Transferring

- Be more specific about which element from which image
- Use explicit indexing ("from image 1")
- Reduce the number of references and complexity

### Inconsistent Blending

- Specify lighting consistency
- Describe how elements should interact
- Use style references to unify the composition

### Identity Drift

- Emphasize key identifying features
- Use phrases like "maintaining exact likeness"
- Provide multiple angles of the same subject if available
