# System prompt library

The folder **Load System Prompt** reads and **Save System Prompt** writes, and the files it holds.

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

## Troubleshooting

| Seen | Fix |
|---|---|
| A file just added is missing | Press the refresh button at the top of the library on the node |
| A file is never listed | The extension is `.txt` or `.md`, the file is under 256 KB, and no folder above it starts with `.` |
| `is not in the system prompt library` | The workflow names a prompt this library does not hold. Pick another on the node, or place the file at the path the message names |
| `is not UTF-8 text` | Save the file again as UTF-8 |
| `already in the system prompt library with different text` | Turn `overwrite` on, or type another name |
| A file added to a skill folder is not listed | It is one of the skill's documents, on the `documents` output, not a prompt of its own. Move it out of the folder to list it |
| `documents` is empty for a skill | `features.llm` is off, or the skill's other files are of a type not read |
| `holds no .txt or .md prompt and no SKILL.md` | The folder picked holds no file the library takes. Pick the folder holding `SKILL.md` itself |
