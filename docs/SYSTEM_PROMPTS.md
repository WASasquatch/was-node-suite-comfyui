# System prompt library

The folder **Load System Prompt** reads and **Save System Prompt** writes, the files it holds, and the prompts the pack ships.

## Where it is

| Install | Folder |
|---|---|
| Windows portable | `ComfyUI_windows_portable\ComfyUI\user\was-node-suite\system_prompts` |
| Any other install | `ComfyUI/user/was-node-suite/system_prompts` |
| `WAS_CONFIG_DIR` set | `system_prompts` inside that folder |

Folders inside it, eight levels deep, are the folder tree on Load System Prompt. A file or folder
whose name starts with `.` is not listed.

## A file

| | |
|---|---|
| Extensions | `.txt`, `.md` |
| Encoding | UTF-8 |
| Largest file | 256 KB. A larger one is not listed, and its path is logged |
| Files listed | 2000 at most |

Front matter between two `---` lines opens the file. Everything after the closing line is the
system prompt, and the front matter never reaches the model.

```
---
name: Detailed Caption
description: One dense paragraph for training captions
model_family: [Qwen, Gemma]
tags: [caption, vision]
---
You describe pictures for an image model in one plain paragraph.
```

| Key | Holds | When it is missing |
|---|---|---|
| `name` | The name in the library | The file name without its extension |
| `description` | One line under the name | No line is drawn |
| `model_family` | One family, a list, or names separated by commas. `models` and `family` are read the same way | The prompt is listed under **No family** |
| `tags` | Words the search matches | Only the name, description, folder and family are searched |

Front matter that is not a YAML mapping, or that does not close within the first 16 KB, is part of
the prompt. Save System Prompt keeps every other key when it writes a file that has front matter.

## Skill folders

A folder holding `SKILL.md` is one skill. The library lists that file under the folder's name,
unless its front matter names it, and lists nothing else inside the folder.

| | |
|---|---|
| System prompt | The text of `SKILL.md` after its front matter |
| Documents | Every other file in the folder and its subfolders, eight levels deep, one document each, on Load System Prompt's `documents` output |
| File types read | `.md`, `.markdown`, `.txt`, `.rst`, `.yaml`, `.yml`, `.json`, `.jsonl`, `.toml`, `.ini`, `.cfg`, `.csv`, `.tsv`, `.xml`, `.html`, `.htm`, `.css`, `.py`, `.js`, `.mjs`, `.ts`, `.sh`, `.ps1`, `.bat`, `.sql`, `.lua`, `.r` |
| Passed over | Pictures and other files, names starting with `.`, `__pycache__`, `node_modules`, links and junctions, files that are not UTF-8 |
| Largest file | 1 MB |
| Files per skill | 200, 8 MB together |
| Document metadata | `file_name`, `relative_path` such as `references/layout.md`, `skill`, and `section`, the first folder of the path |

The `documents` output carries nothing for a plain prompt, or while `features.llm` is off. Scripts
in a skill are read as text and never run.

## Importing a folder

**Import** on Load System Prompt opens a folder picker and writes the folder into the library
under its own name, then loads the first skill it holds on the node.

| Folder | What is written |
|---|---|
| Holding `SKILL.md` at its top | The skill, with every file of a type a skill reads |
| Holding skills in subfolders | Each skill with its files, and `.txt` and `.md` files elsewhere |
| Holding no `SKILL.md` | Its `.txt` and `.md` files, as prompts |

| | |
|---|---|
| Files per import | 400 |
| Text per import | 16 MB |
| A folder of the same name | Replaced only when the dialog is answered yes. Files of the same path are written over and the rest of the library's copy stays |

## Built-in prompts

The pack ships prompt enhancers for image and video models. They are listed under `Built-in` in
the library, one folder per model family, and are read-only.

| | |
|---|---|
| Where they are | `ComfyUI/custom_nodes/was-node-suite-comfyui/modules/data/system_prompts` |
| Key on Load System Prompt | `Built-in/` then the folder and file, such as `Built-in/FLUX/FLUX.2 Prompt Upsampler.md` |
| Editing one | **Copy** on the built-in's preview writes it into the library at the same path without `Built-in/`, such as `FLUX/FLUX.2 Prompt Upsampler.md`, and loads the copy on the node. A copy whose file already holds other text is replaced only when the dialog is answered yes |
| A folder of your own named `Built-in` | Not listed. Rename it |
| Origin and licence | Front matter `source`, `license` and, where the text was changed, `changes`. `NOTICE.md` beside the prompts lists every project, commit and change |

Each one is the system prompt; the idea, the edit request or the picture goes in the model's own
prompt and image inputs, as on Fast Generate Text.

### Sending the request

Most take the idea as it is typed. These want something more:

| Built-in | Send |
|---|---|
| Any edit, image-to-video or Image to Prompt built-in | The picture on the generating node's `image` input, from a model that reads pictures such as Qwen3-VL or Qwen3.5 |
| FLUX 3 Klein Upsampler, Layout Upsampler | Optionally a first line `TARGET CANVAS: 16:9` naming the aspect ratio |
| FLUX 3 Klein Edit Upsampler | `Reference images follow in order (the first image, the second image, ...).` then `User request: <request>` then `Output the caption.`, each on its own line, as the fine-tuned upsampler was trained on |
| FLUX 3 Klein Layout Upsampler, Layout Edit Upsampler | `ORIGINAL PROMPT:` and the request, then the boxes to keep as a JSON list of `{"id", "bbox", "desc"}` rows, boxes `[y_min, x_min, y_max, x_max]` from 0 to 1000 |
| LTX-Video Image to Video Enhancer | `user_prompt: <request>` then `image_caption: <a caption of the first frame>` on the next line |
| Fibo 2 Refine JSON | `[EXISTING JSON]:` and the caption, then `[EDIT INSTRUCTIONS]:` and the change |
| ERNIE Image Prompt Enhancer | `{"prompt": "<request>", "width": 1024, "height": 1024}` |
| Krea 2 Prompt Expansion, MOSS-Image Recaption | `thinking` on, on a model that reasons |
| MiniMax H3 Image to Video, First and Last Frame, Last Frame | The clip length in seconds in the request, such as `8 seconds` |
| MiniMax H3 Reference to Video | Each reference by its label, such as `Picture 1: my dog. Audio 1: my voice.`, then what happens |

| Built-in | Its answer goes to |
|---|---|
| Fibo 2 | Text Encode Fibo 2 |
| MOSS-Image Prompt Enhancer | MOSS-Image Compile Prompt, which turns a layout into the prompt MOSS-Image reads |
| MiniMax H3 | MiniMax H3 Conditioning |
| Any other | The model's text encoder |

### Settings the makers use

| Built-ins | Model the makers run | Sampling |
|---|---|---|
| FLUX.2 | Mistral Small 3.2 24B | temperature 0.15, 512 tokens |
| FLUX 3 Klein | `qwen3.5_9b_flux3_prompt_upsampler`, from Load CLIP with type `flux3` | sampling `off`, `thinking` off, 3584 tokens |
| LTX-2 | Gemma 3 12B | temperature 0.7, 512 tokens |
| LTX-2.5 | Gemma 4 E2B | sampling `off`, 600 tokens |
| Wan 2.2, Wan First and Last Frame | Qwen2.5 14B, Qwen2.5-VL 7B for pictures | 512 tokens |
| Kandinsky 6 | Qwen3.5 9B | temperature 0.3, repetition penalty 1.1, 1500 tokens, `thinking` off |
| Fibo 2 | a vision model | temperature 0.2 |
| ERNIE Image | ERNIE's own `ernie-image-prompt-enhancer`, which ignores a system prompt | temperature 0.6 |

### The built-ins

**ERNIE Image**

| Prompt | Writes | From |
|---|---|---|
| ERNIE Image Prompt Enhancer | A short request and the image size become one rich Chinese visual description | Baidu, changed |

**FLUX**

| Prompt | Writes | From |
|---|---|---|
| FLUX 3 Klein Edit Upsampler (Any LLM) | A request and its reference pictures become one detailed edit instruction | Black Forest Labs |
| FLUX 3 Klein Edit Upsampler | A request and its reference pictures become one detailed edit instruction | Black Forest Labs |
| FLUX 3 Klein Layout Edit Upsampler (Any LLM) | An edit request becomes an instruction plus a JSON table of source and target boxes | Black Forest Labs |
| FLUX 3 Klein Layout Edit Upsampler | An edit request becomes an instruction plus a JSON table of source and target boxes | Black Forest Labs |
| FLUX 3 Klein Layout Upsampler (Any LLM) | A short idea becomes a caption plus a JSON table of bounding boxes | Black Forest Labs |
| FLUX 3 Klein Layout Upsampler | A short idea becomes a caption plus a JSON table of bounding boxes | Black Forest Labs |
| FLUX 3 Klein Upsampler (Any LLM) | A short idea becomes the detailed caption FLUX 3 Image Klein was trained on | Black Forest Labs |
| FLUX 3 Klein Upsampler | A short idea becomes the detailed caption FLUX 3 Image Klein was trained on | Black Forest Labs |
| FLUX.1 Kontext Edit Instruction | Turns a rough edit request into one precise FLUX.1 Kontext instruction that names what changes and what must stay the same | the pack |
| FLUX.1 Prompt Enhancer | Expands a short idea into one natural-language prompt for FLUX.1 [dev], [schnell] and FLUX.1 Krea [dev], subject first and lighting stated | the pack |
| FLUX.2 Edit Upsampler | A request and its reference pictures become one concise edit instruction | Black Forest Labs |
| FLUX.2 JSON Prompt Writer | Turns an idea into a structured JSON prompt for FLUX.2 [dev], with the scene, each subject, camera, light and palette in their own fields | the pack |
| FLUX.2 Klein Edit Instruction | Turns a rough edit request into one short FLUX.2 [klein] edit prompt that names each reference picture's role and the change | the pack |
| FLUX.2 Klein Prompt Enhancer | Expands a short idea into one rich prose prompt for FLUX.2 [klein] 4B and 9B, which take the prompt exactly as written | the pack |
| FLUX.2 Prompt Upsampler | A short idea becomes a detailed prompt with every visible word quoted | Black Forest Labs |
| FLUX Prompting (skill) | One skill for every FLUX model, from FLUX.1 to FLUX 3 Klein. Its documents are Black Forest Labs' own prompting rules, for the LLM index nodes to look up | the pack |

**Fibo**

| Prompt | Writes | From |
|---|---|---|
| Fibo 2 Edit | A picture and an edit request become the JSON caption of the edited picture, with its edit_instruction | Bria, changed |
| Fibo 2 Image to JSON | A picture becomes the JSON caption that describes it | Bria, changed |
| Fibo 2 Inspire | A picture and a request become the JSON caption of a new scene, the picture a visual reference only | Bria, changed |
| Fibo 2 Masked Edit | A picture with its masked regions greyed and a request become the caption of the repainted picture | Bria, changed |
| Fibo 2 Multi-Image Edit | Two to four pictures and a request become one JSON caption naming each picture as <image_N> | Bria, changed |
| Fibo 2 Prompt to JSON | A short idea becomes the JSON caption Fibo 2.0 reads | Bria, changed |
| Fibo 2 Refine JSON | A JSON caption and an instruction become the revised caption. Send the caption, then the instruction | Bria, changed |

**General**

| Prompt | Writes | From |
|---|---|---|
| Image Prompt Enhancer | Expands a short idea into one detailed natural-language prompt for any text-to-image model that reads sentences | the pack |
| Image to Prompt | Describes an attached picture as a text-to-image prompt that would recreate it, for a vision model | the pack |
| Prompt Translator | Translates an image or video prompt into English, keeping its meaning, its order and any visible text in its own language | the pack |
| Video Prompt Enhancer | Expands a short idea into one detailed text-to-video prompt with motion, camera and timing, for any video model that reads sentences | the pack |

**Hunyuan**

| Prompt | Writes | From |
|---|---|---|
| HunyuanVideo 1.5 Image to Video | Turns a request and its first frame into a HunyuanVideo 1.5 prompt that describes only the movement, step by step, with no invented camera moves | the pack |
| HunyuanVideo 1.5 Text to Video | Expands a short idea into one detailed English HunyuanVideo 1.5 prompt, ordered from the whole scene down to the camera and style, sized to a five second clip | the pack |

**Kandinsky**

| Prompt | Writes | From |
|---|---|---|
| Kandinsky 6 Image to Video with Sound | A request and its first frame become one paragraph of picture, speech and sound, the frame taken as ground truth | Kandinsky Lab |
| Kandinsky 6 Text to Video with Sound | A short request becomes one paragraph of picture with its speech in <S> tags and its sound in an <AUDCAP> sentence | Kandinsky Lab |

**Krea**

| Prompt | Writes | From |
|---|---|---|
| Krea 2 Prompt Expansion | A short idea becomes one faithful, detailed paragraph. Turn thinking on | Krea |

**LTX**

| Prompt | Writes | From |
|---|---|---|
| LTX-2 First and Last Frame | Writes an LTX-2 or LTX-2.5 prompt for a clip between a given first and last frame, describing the motion, camera and sound that join them | the pack |
| LTX-2 Image to Video Enhancer | A request and its first frame become a prompt describing only what moves and sounds | Lightricks |
| LTX-2 Text to Video Enhancer | A short idea becomes a detailed video prompt with its sound and dialogue woven in | Lightricks |
| LTX-2.5 Image to Video Enhancer | A request and its first frame become a training-style caption that opens on that exact frame | Lightricks |
| LTX-2.5 Text to Video Enhancer | A short idea becomes a caption in the style LTX-2.5 was trained on, shots and sound included | Lightricks |
| LTX-Video Image to Video Enhancer | A request and a description of the first frame become one shot description, the image first. Send the request, then the frame's caption | Lightricks |
| LTX-Video Text to Video Enhancer | A short idea becomes one chronological shot description of at most 150 words | Lightricks |

**MOSS-Image**

| Prompt | Writes | From |
|---|---|---|
| MOSS-Image Recaption | A picture becomes long, medium and short captions plus its watermark boxes, as JSON | OpenMOSS, changed |
| MOSS-Image Tag Writer | Turns an idea into a MOSS-Image tag sequence in the order the model was trained on, quality first and general tags last | the pack |
| MOSS-Image Prompt Enhancer (skill) | OpenMOSS' prompt enhancer skill for MOSS-Image, set up for a local model. An idea becomes a detailed caption or a box layout for MOSS-Image Compile Prompt | OpenMOSS, changed |

**MiniMax H3**

| Prompt | Writes | From |
|---|---|---|
| MiniMax H3 First and Last Frame | Writes one MiniMax H3 prompt that moves from a first frame to a last frame, with the alignment line and the three fields H3 reads | the pack |
| MiniMax H3 Image to Video | Writes one MiniMax H3 prompt that starts from a first frame and develops it forward, with the alignment line and the three fields H3 reads | the pack |
| MiniMax H3 Last Frame | Writes one MiniMax H3 prompt that builds up to a given last frame, with the alignment line and the three fields H3 reads | the pack |
| MiniMax H3 Reference to Video | Writes one MiniMax H3 full-reference prompt, the six sections H3 reads when pictures, videos or audio define the people, places and sound | the pack |
| MiniMax H3 Text to Video | Writes one MiniMax H3 text-to-video prompt with picture, dialogue and sound, in the three fields H3 reads, from a short idea and a length in seconds | the pack |

**Qwen-Image**

| Prompt | Writes | From |
|---|---|---|
| Qwen-Image 2.1 Edit Instruction | Turns an edit request and its pictures into one Qwen-Image 2.1 instruction that leads with the operation and edits only what was named, at full strength | the pack |
| Qwen-Image 2.1 Prompt Writer | Expands any brief into the long, positional English description Qwen-Image 2.1 reads best, around 400 words that walk the whole frame | the pack |
| Qwen-Image 2512 Prompt Rewriter (Chinese) | The Qwen team's own Chinese rewriting prompt for Qwen-Image 2512, with separate checklists for portraits, text-heavy designs and general scenes | Qwen team |
| Qwen-Image 2512 Prompt Rewriter | Portraits, text-heavy designs and general scenes each get their own checklist | Qwen team |
| Qwen-Image Prompt Rewriter (Chinese) | A short idea becomes a complete Chinese prompt, visible text quoted | Qwen team |
| Qwen-Image Prompt Rewriter | A short idea becomes a complete prompt under 200 words, visible text quoted | Qwen team |
| Qwen-Image-Edit 2509 Instruction Rewriter | A rough edit request and its pictures become one precise instruction | Qwen team, changed |
| Qwen-Image-Edit 2511 Instruction Rewriter | A rough edit request and its pictures become one precise instruction | Qwen team, changed |

**SDXL Tags**

| Prompt | Writes | From |
|---|---|---|
| Illustrious and NoobAI Tag Writer | Turns an idea into a Danbooru tag prompt in the order Illustrious XL, NoobAI-XL and Animagine XL were trained on, with their quality and year tags | the pack |
| Pony Diffusion V6 Tag Writer | Turns an idea into a Pony Diffusion V6 XL prompt that opens with the full score string, then a plain description and the tags | the pack |

**Wan**

| Prompt | Writes | From |
|---|---|---|
| Wan 2.2 Image to Video Extender (Chinese) | A request and its first frame become a Chinese prompt about the motion and the camera | Wan team |
| Wan 2.2 Image to Video Extender | A request and its first frame become an English prompt about the motion and the camera | Wan team |
| Wan 2.2 Image to Video Without a Prompt (Chinese) | A first frame alone becomes a Chinese prompt for how it moves | Wan team |
| Wan 2.2 Image to Video Without a Prompt | A first frame alone becomes an English prompt for how it moves | Wan team |
| Wan 2.2 Text to Video Extender (Chinese) | A short idea becomes a cinematic Chinese prompt with time, light, shot size and composition | Wan team, changed |
| Wan 2.2 Text to Video Extender | A short idea becomes a cinematic English prompt with time, light, shot size and composition | Wan team, changed |
| Wan First and Last Frame Extender (Chinese) | A request and both frames become a Chinese prompt naming the change between them | Wan team |
| Wan First and Last Frame Extender | A request and both frames become an English prompt naming the change between them | Wan team, changed |

**Z-Image**

| Prompt | Writes | From |
|---|---|---|
| Z-Image Prompt Enhancer (Chinese) | A request becomes a faithful, concrete visual description with every visible word quoted | Tongyi-MAI, changed |

## Troubleshooting

| Seen | Fix |
|---|---|
| A file just added is missing | Press the refresh button at the top of the library on the node |
| A file is never listed | The extension is `.txt` or `.md`, the file is under 256 KB, and no folder above it starts with `.` |
| `is not in the system prompt library` | The workflow names a prompt this library does not hold. Pick another on the node, or place the file at the path the message names |
| `is not UTF-8 text` | Save the file again as UTF-8 |
| `already in the system prompt library with different text` | Turn `overwrite` on, or type another name |
| A file added to a skill folder is not listed | It is one of the skill's documents, on the `documents` output, not a prompt of its own. Move it out of the folder to list it |
| A built-in is missing from the menu | Press the refresh button at the top of the library. A workflow from a newer version of the pack can name a built-in this version lacks: update the pack |
| `holds the pack's own prompts, so nothing is saved into it` | Save System Prompt and Import do not write into `Built-in`. Type another folder |
| A folder of your own named `Built-in` is not listed | Rename it. That name is where the pack's own prompts are listed |
| `is already in the system prompt library with different text` on Copy | An earlier copy was edited. Answer yes to replace it with the built-in text, or no to keep the edit |
| An answer carries `<think>` text or a JSON wrapper | Turn `thinking` off on the generating node, or use a built-in that answers in plain text |
| `documents` is empty for a skill | `features.llm` is off, or the skill's other files are of a type not read |
| `holds no .txt or .md prompt and no SKILL.md` | The folder picked holds no file the library takes. Pick the folder holding `SKILL.md` itself |
