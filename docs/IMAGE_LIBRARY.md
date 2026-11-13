# Picture libraries

Image Library Search reads a folder of reference pictures and a caption for each, and finds the
closest ones to a prompt. This page lists where libraries go, the pose sets it reads, how captions
are found, and the models it uses.

## Where a library goes

| Folder | Holds |
|---|---|
| `ComfyUI/input/poses/<set>` | One pose set per folder. Picking `poses [input]` with `recursive` on searches every set at once |
| `ComfyUI/input/costumes/<set>` | One costume set per folder, searched the same way |
| Any folder under `ComfyUI/input` or `ComfyUI/output` | Any pictures. `output [output]` with `recursive` on searches finished renders by the prompts saved in them |
| A folder listed in `paths.allow_read` in `config.yaml` | The same, for a library kept outside ComfyUI |
| `ComfyUI/user/was-node-suite/image_library` | The index of every folder read, one file per folder and model, written by the pack |

On the Windows portable build the first row is
`ComfyUI_windows_portable\ComfyUI\input\poses\<set>`.

## Pose sets

The pose pictures in these sets are POSE-DB-01 by ギア ([@giaaibusi](https://x.com/giaaibusi)),
[ai-pose-gallery.pages.dev](https://ai-pose-gallery.pages.dev/): 288 poses of a faceless grey
mannequin, each from five camera angles, 1,439 pictures. The pack ships none of them, and each
source states its own terms of use.

| Source | Pictures | Captions | Fetch into `ComfyUI/input/poses` |
|---|---|---|---|
| [RuiC-PoseAtlas](https://github.com/HRuiCcc/RuiC-PoseAtlas) | PNG, 900 × 1200, and JPEG thumbnails | English, Chinese and Japanese, in `data/poses.json` | `git clone --depth 1 https://github.com/HRuiCcc/RuiC-PoseAtlas` |
| [pose-gallery](https://github.com/aaa8221953/pose-gallery) | WebP, and JPEG thumbnails | Chinese in `pose-index.json`, Japanese in `data/pose.json` | `git clone --depth 1 https://github.com/aaa8221953/pose-gallery` |
| [ai-pose-gallery-zh](https://github.com/anhao/ai-pose-gallery-zh) | PNG line art and depth renders | Chinese and Japanese, in `data/` | `git clone --depth 1 https://github.com/anhao/ai-pose-gallery-zh` |
| [ComfyUI-PoseAtlas](https://github.com/mugicha-AI/ComfyUI-PoseAtlas) | Fetched by its own download script into `web/atlas` | As RuiC-PoseAtlas | Add `ComfyUI/custom_nodes/ComfyUI-PoseAtlas/web/atlas` to `paths.allow_read` |

Every set uses the same picture names, such as `full_sexy_P001_front`. RuiC-PoseAtlas's
`data/poses.json` copied into another set's `data` folder gives that set English captions.

The `Couples` series of POSE-DB-01 is explicit. With English captions, `-couples` in Image
Library Search's `filter` leaves it out.

## Costume sets

COSTUME-DB-01 by ギア ([@giaaibusi](https://x.com/giaaibusi)),
[ai-costume-gallery.pages.dev](https://ai-costume-gallery.pages.dev/): 1,407 outfits with no one
wearing them, in eight categories from spring to swimwear. The pack ships none of them.

| Source | Pictures | Captions | Fetch into `ComfyUI/input/costumes` |
|---|---|---|---|
| [costume-gallery](https://github.com/aaa8221953/costume-gallery) | WebP, and WebP thumbnails | English keywords, such as `black leather jacket grey knit blue denim jeans`, in `costumes-index.json` | `git clone --depth 1 https://github.com/aaa8221953/costume-gallery` |

## Captions

| `captions` | Each picture's caption is |
|---|---|
| `auto` | The manifest's record where a folder holds one, otherwise the first of the three below that the picture has |
| `manifest` | The record a JSON manifest in the folder or its `data` folder holds for it. Pictures no manifest names are left out |
| `sidecar text` | `<picture name>.txt` or `<picture name>.caption` beside it |
| `picture prompt` | The prompt ComfyUI or AUTOMATIC1111 saved in it |
| `file name` | Its file name, with separators as spaces and a trailing counter dropped |

A manifest is a JSON list of records, or an object whose longest list is the records:

| In a record | Read as |
|---|---|
| A text ending `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.gif` or `.tiff` | A picture, relative to the folder or to the manifest. A key or path holding `thumb`, `preview` or `small` marks the thumbnail the panel draws. A picture missing on disk is looked for as `.webp`, `.png`, `.jpg` and `.jpeg` |
| `name`, `names`, `title`, `caption`, `description`, `prompt`, `keywords`, `tags` | The start of the caption |
| Any other text | `key: text`, after the start |
| `{"en": …, "zh": …}` | Its English text, else its first |
| A list of records | One picture each, sharing the record's fields and grouped as one pose |
| `id`, `no`, `search`, `url`, sizes, and codes such as `P001` | Nothing |

A top-level list of `{"key": …, "names": …}` names the values of the record field it is named
after: `categories` names `category`, `angles` names `angle`. Where several manifests name the
same pictures, the one placing the most full-size pictures with English captions is read.

## Camera views

A picture's view is the word its file name ends in. A view word in the prompt picks that view of
the matched pose; it never picks the pose.

| View | File name ends | Prompt words |
|---|---|---|
| Front | `_front` | `front view`, `from the front`, `front-facing`, `facing the camera` |
| Side | `_side`, `_left`, `_right`, `_profile` | `side view`, `from the side`, `profile`, `side-on` |
| Three-quarter | `_diag`, `_diagonal`, `_three_quarter` | `three-quarter`, `3/4`, `diagonal`, `angled view` |
| High | `_high`, `_top` | `high angle`, `from above`, `bird's eye`, `overhead`, `top-down` |
| Low | `_low` | `low angle`, `from below`, `worm's eye` |
| Back | `_back`, `_rear` | `from behind`, `back view`, `rear view`, `facing away` |

## Models

Text and picture models are Hugging Face folders in `ComfyUI/models/LLM/<owner>/<name>`, such as
`ComfyUI/models/LLM/google/embeddinggemma-2`. A model in the table not on disk is downloaded there
on first use while `features.network` is on; with it off, download the model's files from its page
into that folder. Only `model.safetensors` weights are read. Every model runs on the packages
ComfyUI itself installs.

| Model | Download | Reads | Licence |
|---|--:|---|---|
| [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2) | 1.49 GB | Captions in over 100 languages and the pictures themselves, in one space; the default `text_model` and `picture_model`, loaded once for both | Apache-2.0 |
| [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5) | 133 MB | English captions | MIT |
| [`unsloth/embeddinggemma-300m`](https://huggingface.co/unsloth/embeddinggemma-300m) | 1.21 GB | Captions in over 100 languages; `dimensions` 768, 512, 256 or 128. Needs transformers 4.56 or newer | Gemma Terms of Use |
| [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) | 471 MB | Captions in English, Chinese, Japanese and about 100 more | MIT |
| [`Alibaba-NLP/gte-modernbert-base`](https://huggingface.co/Alibaba-NLP/gte-modernbert-base) | 298 MB | English captions | Apache-2.0 |
| [`Snowflake/snowflake-arctic-embed-m-v1.5`](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v1.5) | 436 MB | English captions | Apache-2.0 |
| [`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5) | 438 MB | English captions | MIT |
| [`BAAI/bge-large-en-v1.5`](https://huggingface.co/BAAI/bge-large-en-v1.5) | 1.34 GB | English captions | MIT |
| [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | 1.19 GB | Captions in over 100 languages | Apache-2.0 |
| [`intfloat/multilingual-e5-large`](https://huggingface.co/intfloat/multilingual-e5-large) | 2.24 GB | Captions in English, Chinese, Japanese and about 100 more | MIT |
| [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 91 MB | English captions | Apache-2.0 |
| [`mixedbread-ai/mxbai-embed-large-v1`](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1) | 670 MB | English captions | Apache-2.0 |
| [`google/siglip2-base-patch16-224`](https://huggingface.co/google/siglip2-base-patch16-224) | 1.5 GB | The pictures themselves, beside English captions | Apache-2.0 |
| [`google/siglip2-so400m-patch14-384`](https://huggingface.co/google/siglip2-so400m-patch14-384) | 4.54 GB | The pictures themselves, beside English captions | Apache-2.0 |
| [`openai/clip-vit-large-patch14`](https://huggingface.co/openai/clip-vit-large-patch14) | 1.71 GB | The pictures themselves, beside English captions | None stated on its model page |
| [`laion/CLIP-ViT-H-14-laion2B-s32B-b79K`](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K) | 3.95 GB | The pictures themselves, beside English captions | MIT |
| `keywords only` | none | Shared words, in any language | |

A picture captioned only by its file name is ranked nine tenths by the picture model and one tenth
by its name; a picture with a manifest record, a `.txt` file or a saved prompt is ranked by
`picture_weight`.

## Troubleshooting

| What happens | Cause | Fix |
|---|---|---|
| `not in models/LLM, and features.network is off` | The model is not on disk | Set `features.network: true`, or put the model's folder in `ComfyUI/models/LLM` |
| Fewer pictures than the set holds | The manifest names files not in the folder | Fetch the set's picture folders, not only its manifest |
| The right pose at the wrong angle | The prompt names no view | Add a view word from the table above, or set the view on the panel's bar |
| A folder on the panel reads `not indexed` and its pictures never match | Its index has not been made with the models chosen | Run the node once; every ticked folder is indexed |
| A search for what a photo shows, such as `hair`, finds no photo | `picture_model` is `none`, and the photos are captioned by their file names | Choose a picture model |
| The panel's matches do not change as the prompt is typed | The prompt arrives through a link | The panel shows the last run's matches; run the node to search again |
| Power Preprocessor's `openpose` answers a black frame on a mannequin | Its person detector finds no person in line art | Use `depth_map` on mannequin pictures, or wire the picture itself to a line art ControlNet |
