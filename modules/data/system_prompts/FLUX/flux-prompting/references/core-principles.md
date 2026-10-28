---
name: core-principles
description: Universal prompting principles that apply to all FLUX models
---

# Core FLUX Prompting Principles

These principles apply to all FLUX models and form the foundation of effective prompting.

## 1. Positive Descriptions Only

FLUX does NOT support negative prompts. Always describe what you WANT, not what you don't want.

### Wrong Approach
```
a portrait of a woman, no glasses, no hat, no makeup
```

### Correct Approach
```
a portrait of a woman with natural skin, clear face, bare head, visible eyes
```

See [negative-prompt-alternatives.md](negative-prompt-alternatives.md) for comprehensive replacement strategies.

## 2. Prompt Structure Formula

Build prompts using this structure for consistent results:

```
[Subject] + [Action/Pose] + [Style/Medium] + [Context/Setting] + [Lighting] + [Technical Details]
```

### Example

```
A young woman with flowing auburn hair (subject)
dancing gracefully in mid-leap (action)
in the style of classical oil painting (style)
in a moonlit garden with roses (context)
soft diffused moonlight with subtle rim lighting (lighting)
medium shot, shallow depth of field (technical)
```

## 3. Specificity Matters

More specific prompts yield dramatically better results.

### Vague (Poor Results)
```
a cat sitting
```

### Specific (Excellent Results)
```
A fluffy orange tabby cat with bright green eyes sitting regally on a vintage
velvet armchair, afternoon sunlight streaming through lace curtains, warm
golden hour lighting, shallow depth of field, shot on medium format film
```

## 4. Natural Language Works Best

Write prompts as descriptive prose rather than keyword lists.

### Keyword Style (Less Effective)
```
woman, portrait, beautiful, blonde, studio, professional, 8k, detailed
```

### Prose Style (More Effective)
```
A professional studio portrait of a beautiful blonde woman in her thirties,
captured with soft studio lighting that accentuates her features, rendered
in stunning detail with natural skin texture and subtle catchlights in her eyes
```

## 5. Lighting is Critical

Always specify lighting - it has the single greatest impact on image quality.

### Natural Lighting
- Golden hour - warm, soft, directional
- Overcast - soft, diffused, even
- Harsh midday - high contrast, strong shadows
- Dappled forest light - specular, organic patterns

### Studio Lighting
- Softbox - even, professional
- Rim light - edge definition, separation
- Butterfly lighting - beauty, glamour
- Rembrandt lighting - dramatic, classic portraits

### Atmospheric Lighting
- Volumetric fog - depth, mystery
- God rays - dramatic, spiritual
- Neon glow - urban, cyberpunk
- Candlelight - warm, intimate

### Mood-Based Lighting
- Dramatic shadows - tension, noir
- High key - bright, airy, clean
- Low key - moody, mysterious
- Chiaroscuro - strong contrast, painterly

## 6. Word Order Matters

FLUX prioritizes elements that appear earlier in the prompt. Front-load important elements.

### Less Effective
```
A forest background with soft lighting where a knight in shining armor stands
```

### More Effective
```
A knight in shining armor stands in a forest, soft dappled lighting filtering
through the canopy
```

## 7. Medium Prompt Length

Optimal prompt length is typically 30-80 words (FLUX can handle up to 512 tokens).

- Too short: Lacks direction, generic results
- Too long: Can become unfocused
- Sweet spot: Enough detail to guide, not so much it confuses

## 8. Iterative Refinement

Build prompts iteratively:

1. Start with core subject and action
2. Add style and medium
3. Specify lighting and atmosphere
4. Include technical details
5. Refine based on results

Change one element at a time to understand what affects your output.
