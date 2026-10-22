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

A folder holding `SKILL.md` lists that file under the folder's name, unless its front matter
names it. Every other `.txt` and `.md` file in the folder is listed as a prompt of its own.

## Troubleshooting

| Seen | Fix |
|---|---|
| A file just added is missing | Press the refresh button at the top of the library on the node |
| A file is never listed | The extension is `.txt` or `.md`, the file is under 256 KB, and no folder above it starts with `.` |
| `is not in the system prompt library` | The workflow names a prompt this library does not hold. Pick another on the node, or place the file at the path the message names |
| `is not UTF-8 text` | Save the file again as UTF-8 |
| `already in the system prompt library with different text` | Turn `overwrite` on, or type another name |
