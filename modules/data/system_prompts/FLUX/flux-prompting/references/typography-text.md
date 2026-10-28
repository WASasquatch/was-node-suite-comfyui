---
name: typography-text
description: Prompting for text rendering and typography in FLUX
---

# Typography and Text Prompting

Guide to rendering text in FLUX images. Use FLUX.2 [flex] for best typography results.

## Basic Syntax

Always quote the exact text you want rendered:

```
A coffee shop chalkboard sign displaying "TODAY'S SPECIAL" in decorative script
```

## Core Rules

### 1. Use Quotation Marks

```
Correct: A poster with "HELLO WORLD" in bold letters
Wrong: A poster with HELLO WORLD in bold letters
```

### 2. Specify Font Style

```
"ADVENTURE" in bold sans-serif typography
"Welcome" in elegant cursive script
"CHAPTER ONE" in classic serif typeface
"CODE" in monospace terminal font
"SALE!" in decorative display lettering
```

### 3. Describe Size Hierarchy

```
Large headline "BREAKING NEWS" above smaller subtext "Details inside"
```

### 4. Indicate Placement

```
"OPEN" sign centered in storefront window
"EXIT" text positioned above doorway
"Page 1" in bottom right corner
```

### 5. Front-Load Text

Place text descriptions early in the prompt for better accuracy:

```
Good: A sign reading "FRESH BREAD" in a bakery window...
Less Good: A bakery window with a sign that says "FRESH BREAD"...
```

## Font Style Categories

### Sans-Serif (Modern/Clean)
```
"MINIMAL" in clean geometric sans-serif, Swiss modernist style
"TECH SUMMIT" in bold condensed grotesque typeface
"future" in thin uppercase sans-serif, contemporary design
```

### Serif (Classic/Elegant)
```
"The New Yorker" in traditional serif typeface, editorial masthead
"LUXURY" in high-contrast Didone serif with thin/thick strokes
"Wisdom" in old-style serif with subtle bracketed serifs
```

### Script/Cursive (Decorative)
```
"With Love" in flowing calligraphic script with flourishes
"Signature" in connected brush script, casual elegance
"Romance" in formal copperplate script, wedding invitation style
```

### Display/Decorative
```
"ROCK CONCERT" in distressed vintage concert poster lettering
"CIRCUS" in ornate Victorian display type with decorative elements
"RETRO" in 1970s rounded bubble letters
```

### Handwritten
```
"Note to self" in casual handwritten style, slightly imperfect
"Thanks!" in quick marker pen handwriting
"ideas" in sketchy pencil handwriting
```

### Monospace
```
"CODE_COMPLETE" in terminal monospace, developer aesthetic
"SYSTEM" in typewriter monospace, vintage tech
"DEBUG" in LCD-style digital monospace
```

## Text Effects

### Neon Signs
```
Glowing neon sign spelling "OPEN 24/7" in pink neon tubes with
blue outline, slight glow and reflection, night scene
```

### Metallic/3D
```
"GOLD" in three-dimensional metallic gold letters with realistic
reflections and subtle shadows, luxury aesthetic
```

### Embossed/Debossed
```
"PREMIUM" embossed into leather surface, subtle shadows showing
the raised letterforms
```

### Outlined
```
"MODERN" in outline-only letters, no fill, thin white stroke
on dark background
```

### Gradient Text
```
"SUMMER" with gradient fill from #FF6B6B (coral) at top to
#4ECDC4 (teal) at bottom
```

## Multi-Text Compositions

### Poster Design
```
Event poster with "SUMMER FEST 2025" as large headline in bold
condensed sans-serif at top, "JULY 15-17" as medium subheading
in regular weight, "Central Park, NYC" as small body text at
bottom, all in white text on #FF6B35 (sunset orange) background
```

### Book Cover
```
Book cover design: "THE GREAT GATSBY" in elegant art deco gold
lettering centered in upper third, author name "F. SCOTT FITZGERALD"
in smaller gold caps below, #1A1A2E (midnight blue) background
with geometric gold accents
```

### Magazine Cover
```
Fashion magazine cover with "VOGUE" in classic serif masthead at top,
cover line "SPRING COLLECTION" in bold sans-serif, "The New Rules of Style"
in lighter weight italic, all in white against dramatic portrait
```

### Signage
```
Vintage diner sign: "MEL'S DINER" in red neon script lettering,
"OPEN" below in separate green neon block letters, chrome border,
1950s Americana aesthetic
```

### Business Card
```
Minimalist business card with "JOHN SMITH" in medium weight sans-serif,
"Creative Director" in lighter weight below, contact details in small
type at bottom, #2C3E50 (dark blue) text on white background
```

## Text Placement Strategies

### Centered Composition
```
Centered text layout: "WELCOME" in large caps at center,
perfectly balanced with equal margins
```

### Left-Aligned
```
Left-aligned text block: "Company Name" as header,
"Tagline goes here" below, flush left alignment
```

### Text on Path
```
"GOING IN CIRCLES" text following a circular path around
the center of the design
```

### Text Overlay
```
"ADVENTURE AWAITS" in bold white text overlaid on landscape
photograph, positioned in lower third with slight shadow for readability
```

## Technical Considerations for [flex]

### Steps Parameter
- Higher steps (30-50) = better text quality
- Lower steps (10-20) = faster, lower quality

### Guidance Parameter
- Higher guidance (6-10) = stricter prompt following
- Lower guidance (1.5-4) = more creative interpretation

### Recommended Settings
```
For clean typography: steps=50, guidance=7
For artistic text: steps=30, guidance=4
```

## Troubleshooting

### Misspelled Words
- Keep text short (1-4 words work best)
- Use common words when possible
- Repeat the exact text in the prompt

### Illegible Text
- Specify larger text size
- Use simpler fonts (sans-serif)
- Ensure high contrast with background
- Use [flex] model

### Wrong Font Style
Be more specific:
```
Instead of: "text in a nice font"
Use: "text in bold geometric sans-serif similar to Futura"
```

### Text Not Appearing
- Front-load text description in prompt
- Put text in quotes
- Specify exact placement
- Reduce other prompt complexity
