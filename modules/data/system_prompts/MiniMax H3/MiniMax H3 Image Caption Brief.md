---
name: MiniMax H3 Image Caption Brief
description: The request side of the MiniMax H3 Image Prompt Enhancer skill, for LLM Query Engine's qa_prompt. The skill's retrieved notes fill {context_str} and the idea fills {query_str}
model_family: [MiniMax H3 Image]
tags: [enhance, text to image, qa_prompt, json]
---

Notes for this request, retrieved from the skill's documents. Apply the ones that fit, together with every rule in the system prompt.

---------------------

{context_str}

---------------------

Idea: {query_str}

First settle the aspect ratio from the idea as given, remembering that a ratio is width to height and that a poster or cover is taller than wide. Then, while thinking, write a numbered plan: for a photograph or depicted scene, the sentences under each of the ten coverage items in order; for a designed page, one sentence per region or element. Count them, and add more until there are at least 17 sentences and 280 words, each sentence one idea of visible content. A scene stays under about 380 words; a dense design may run longer. Quote every requested word letter for letter, keeping its capitals and punctuation, and keep every named style, artist and photographer word for word. The caption never mentions the ratio or the canvas shape, and holds no point sizes, percentages or measurements the picture would not show. Then answer with the JSON object alone, its two keys expanded_caption and aspect_ratio.
