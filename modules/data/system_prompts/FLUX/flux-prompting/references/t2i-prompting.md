---
name: t2i-prompting
description: Text-to-image prompting patterns and techniques
---

# Text-to-Image (T2I) Prompting

Comprehensive guide to crafting effective text-to-image prompts for FLUX models.

## Prompt Structure Framework

### Basic Formula
```
[Subject] + [Action] + [Style] + [Context] + [Lighting] + [Technical]
```

### Expanded Framework
```
[Main Subject] - who/what is the focus
[Attributes] - characteristics, details, clothing
[Action/Pose] - what they're doing
[Environment] - where, setting, background
[Style/Medium] - artistic approach
[Lighting] - light source, quality, mood
[Composition] - framing, camera angle
[Technical] - camera, lens, film stock
```

## Subject Types

### People/Portraits
```
A distinguished professor in his 60s with silver hair and round spectacles,
wearing a tweed jacket with leather elbow patches, deep-set thoughtful eyes,
slight smile suggesting hidden wisdom
```

### Animals
```
A majestic snow leopard with piercing blue-grey eyes, thick spotted fur
dusted with snowflakes, powerful muscular build, alert posture on a
rocky outcrop
```

### Objects/Products
```
A vintage Leica M3 camera with worn brass edges showing decades of use,
black leather covering with patina, sitting on weathered wooden table
```

### Landscapes
```
A dramatic fjord at dawn, steep granite cliffs rising from mirror-still
water, wisps of morning mist, distant snow-capped peaks catching first
golden light
```

### Architecture
```
A brutalist concrete apartment building in late afternoon light, geometric
shadows creating abstract patterns, warm sunlight contrasting with cool
grey concrete
```

## Style Categories

### Photorealistic

#### Modern Digital
```
shot on Sony A7IV, clean and sharp, high dynamic range, professional color grading
```

#### Film Photography
```
shot on Kodak Portra 400, natural film grain, organic colors, slight warmth
```

#### Vintage Digital (2000s)
```
early digital camera aesthetic, slight noise, flash photography, candid feel
```

#### 80s Film
```
80s film photography, film grain, warm color cast, soft focus, nostalgic
```

### Artistic Styles

#### Oil Painting
```
classical oil painting style, visible brushstrokes, rich colors, dramatic lighting
```

#### Watercolor
```
delicate watercolor painting, soft edges, transparent washes, paper texture visible
```

#### Digital Art
```
polished digital illustration, clean lines, vibrant colors, professional concept art
```

#### Anime/Manga
```
anime style, large expressive eyes, clean linework, cel shading, vibrant palette
```

## Lighting Patterns

### Portrait Lighting
```
Rembrandt lighting - 45 degree key light creating triangle shadow on cheek
Butterfly lighting - overhead key creating shadow under nose
Split lighting - 90 degree side light, half face in shadow
Loop lighting - slight angle creating small nose shadow
```

### Natural Lighting
```
Golden hour - warm, soft, directional light 1 hour before sunset
Blue hour - cool, ambient light just after sunset
Overcast - soft, even, diffused lighting
Harsh midday - strong contrast, defined shadows
```

### Atmospheric
```
Volumetric light - visible light rays through fog/dust
Rim lighting - backlight creating edge glow
Practical lighting - visible light sources in scene
Neon glow - colorful artificial urban lighting
```

## Camera and Lens Simulation

### Camera Bodies
```
Shot on Hasselblad X2D - medium format, exceptional detail
Shot on Canon 5D Mark IV - professional DSLR quality
Shot on Leica M10 - rangefinder character, smooth tonality
Shot on iPhone 15 Pro - computational photography look
```

### Lens Characteristics
```
85mm f/1.4 - classic portrait, creamy bokeh
24mm f/2.8 - wide angle, environmental
50mm f/1.2 - natural perspective, shallow DOF
135mm f/2 - compressed perspective, smooth background
Macro lens - extreme close-up detail
Tilt-shift lens - miniature effect or architectural correction
```

### Technical Settings
```
f/1.4 - extremely shallow depth of field
f/2.8 - moderate background blur
f/8 - sharp throughout, landscape
f/16 - maximum sharpness, long exposure
ISO 100 - clean, no noise
ISO 3200 - visible grain, low light
```

## Composition Techniques

### Framing
```
extreme close-up - filling frame with detail
close-up - head and shoulders
medium shot - waist up
full shot - entire body
wide shot - subject in environment
establishing shot - location focus
```

### Angles
```
eye level - natural, relatable
low angle - powerful, imposing
high angle - diminished, overview
Dutch angle - tension, unease
bird's eye - pattern, layout
worm's eye - dramatic upward view
```

### Composition Rules
```
rule of thirds - subject at intersection points
centered composition - symmetry, stability
leading lines - guiding eye to subject
frame within frame - natural framing elements
negative space - minimalist, breathing room
```

## Complete Example Prompts

### Editorial Portrait
```
A fashion editorial portrait of a young woman with striking features and
high cheekbones, wearing an avant-garde geometric collar in silver, dramatic
side lighting creating strong shadows, shot on Hasselblad with 100mm lens
at f/2.8, studio background with subtle gradient, high fashion magazine style
```

### Product Photography
```
A premium wireless headphone product shot, matte black finish with rose gold
accents, floating at slight angle against pure white background, soft even
lighting eliminating harsh shadows, reflection visible on glossy surface below,
commercial catalog style, ultra sharp focus throughout
```

### Landscape
```
A misty morning in ancient redwood forest, towering trees disappearing into
fog above, ferns covering forest floor in layers of green, single shaft of
golden sunlight breaking through canopy, shot on large format camera, rich
detail in bark textures, Ansel Adams inspired black and white with deep tones
```

### Architectural
```
Modern minimalist beach house at golden hour, floor-to-ceiling glass walls
reflecting sunset colors, clean white concrete and natural wood, infinity
pool merging with ocean horizon, architectural photography style, wide angle
showing full structure, warm evening light
```
