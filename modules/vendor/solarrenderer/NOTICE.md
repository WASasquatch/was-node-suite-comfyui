# SolarRenderer prompt sentences, vendored

The fixed sentences of the six-part prompt that **SolarRenderer: Few-Step Generative Reference
Rendering** (Fang, Qian, Lu, Huang, Jiang, Wang, Song, Jiang, Shou, 2026) was trained on, and the
function that assembles a prompt from them.

- Upstream: <https://github.com/Enderfga/SolarRenderer>
- Taken at commit `0ca4f5e45b062cc9e9aee506de46aabdc22e4c3d` (2026-10-31)
- Licence: Apache-2.0, retained verbatim in `LICENSE` beside this file

**No weights are bundled.** The two SolarRenderer LoRAs are published at
<https://huggingface.co/Enderfga/SolarRenderer> under the MiniMax H3 Community License and are
placed by hand in `ComfyUI/models/loras`.

## Files taken from upstream

| File | Upstream path | Changed |
|---|---|---|
| `conditions.py` | `skills/solarrenderer-caption/conditions.py` | unchanged, byte for byte |

## Every change made

None. `__init__.py` is the pack's own.

## Upstream NOTICE

```
SolarRenderer
Copyright 2026 SolarRenderer authors

The code in this repository is licensed under the Apache License, Version 2.0 (see LICENSE). It builds on the SolarWM
training framework (https://github.com/Junchao-cs/SolarWM, Apache-2.0).

MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, Copyright (c) 2026 MiniMax. All Rights Reserved.

SolarRenderer runs on MiniMax-H3, which is not distributed here; obtain it from its publisher
(https://huggingface.co/MiniMaxAI/MiniMax-H3) under its own license and Acceptable Use Policy.
```
