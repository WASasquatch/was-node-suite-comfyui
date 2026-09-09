# **WAS** Node Suite v3 &nbsp; ![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom_Node-blue) ![License](https://img.shields.io/badge/License-MIT-green) [![Donate](https://img.shields.io/badge/Donate-PayPal-blue.svg)](https://paypal.me/ThompsonJordan?country.x=US&locale.x=en_US) 

<img src="was-node-suite-v3.png" width="600">

WAS Node Suite has been going since 2023 on Civitai and on GitHub, and was among the
first packs to put hundreds of nodes into users' hands. WAS-NS has over a million downloads, 
and is used by thousands of users daily. It has been MIT since the first commit: use it, change it, 
teach with it, or run it in paid services.

The pack contains **463 nodes for ComfyUI**, across images, filters and colour, masking, 
text and prompts, logic and flow, numbers, latents and sampling, files, animation and video. 

### See [`NODES.md`](NODES.md) for reference.
### Consider [donating to the project](https://paypal.me/ThompsonJordan?country.x=US&locale.x=en_US) to help me afford caffeine.

---

# Installation

**ComfyUI Manager**, search for `WAS Node Suite v3` and install. This is the recommended route.

**Manually**, clone into `custom_nodes`:

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/WASasquatch/was-node-suite-comfyui.git
```

Requires **ComfyUI 0.14.0 or newer** and **Python 3.10+**.

That is the whole install. **Nothing is installed, downloaded or built**, now or on any later
start. The pack never runs pip on its own.

**The first start takes a second or two longer than the rest.** Your `config.yaml`, the state
database and the wildcard, LUT and view-extension folders are written under
`<ComfyUI user dir>/was-node-suite/`, and python compiles the pack to bytecode. On the install
this was measured on that is 1.4 seconds the first time and around 0.25 seconds on every start
after. An update recompiles, so it happens once more each time you pull.

One optional group wants packages and ships off: `document_export`, which lets **Save DOC**
write `.docx`, `.odt` and `.pdf`.

```sh
# portable ComfyUI, from the ComfyUI_windows_portable directory
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\was-node-suite-comfyui\requirements\document_export.txt
```

Then set `document_export: true` under `features:` in `config.yaml`.
[`docs/CONFIG.md`](docs/CONFIG.md#features) has every group, what it needs and what it gates.

---

# What changed since v2

463 nodes across a package of source files. The pack itself needs no packages, and nothing is
fetched from a git URL. What it bundles ships in the repository with its licence beside it,
listed in [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md).

| | v2 | v3 |
|---|---|---|
| Nodes | 220 | 463 |
| Default packages installed | 20 | 0 |
| Installed from a git URL | 3 | 0 |
| Third-party carried in the repository | SAM and BLIP, 75 files of python | 128 files: browser libraries, fonts, eight face cascades, two sets of weights and one network, each with its licence |
| Optional node groups | none | 22 keys in `config.yaml`, 8 of them off out of the box. Per-node disable group. |

216 of the 220 node ids are unchanged. Four are retired, and the section below says what opens
in their place. Face detection, gradient maps, background removal, frame interpolation,
seamless textures, colour matching, levels, palettes, masks and deconvolution all run on torch,
on ComfyUI's own device, so OpenCV, numba, rembg, timm, scipy, scikit-image, scikit-learn and
matplotlib are not installed. Everything else a node reaches for either ships with ComfyUI or
belongs to an optional group.

## 247 new nodes

| Area | New | |
|---|--:|---|
| **Three.js scenes** | 43 | build a 3D scene out of wires, draw it on the node, render or path trace it to an image batch, write it out as a page that runs on its own |
| **Files, archives and documents** | 28 | zip archives on the wire, documents in six formats, whole folders through a graph, the content viewer |
| **Logic and flow** | 24 | boolean reduction, condition chains, typed switches, for and while loops |
| **Layers** | 20 | a layer stack on the wire, with effects, arrangement and a canvas drawn on the node |
| **Bounds** | 10 | a region as a value: measure it, draw it, crop to it and paste the result back |
| **HDR and linear light** | 9 | decode above white, carry range through filters, read and write EXR and DNG |
| **Affine sampling** | 6 | scale and offset a latent through a mask, on its own or from inside a running sampler |
| **Everything else** | 107 | image transforms, LUTs and colour, masking, numbers, text lists and dictionaries, samplers, LoRA stacks, animation, the terminology pantry and the style library |

## Nodes that show their work

194 nodes draw a panel on themselves: both sides of a picture, a histogram, what a loader read
off disk, a colour ramp, a 3D view, a text editor, a contact sheet you click to pick a look.

---

# Opening a workflow saved before v3

No node id is renamed. Two nodes are now a different node under the same menu name, two ids
are gone and another node does the job of each, and ComfyUI offers to swap all four in.
Widgets, slot counts, three menu labels and three results changed, and a workflow saved
before v3 is carried across as it loads.

**Four retired ids.** Each one opens listed in ComfyUI's missing-node dialog. Tick it, press
the replace button, and the node comes back in place. Leave the dialog and it stays missing.
The old name still finds its replacement in the Add Node search.

| Saved as | Comes back as | With |
|---|---|---|
| `Image Crop Face` | the same name, on a replacement running the same eight cascades | wiring, crop padding, cascade choice, flip setting |
| `Image Gradient Map` | the same name, on a replacement running the same gradients | wiring and settings |
| `Load Lora` | **Lora Loader (Advanced)**, same five inputs and three outputs | model and CLIP wired, all three outputs wired, LoRA choice, both strengths |
| `Number to Text` | **Number to String**, same NUMBER, INT and FLOAT input, same STRING output | both wires. Neither node has a widget, so nothing is retyped |

The two face and gradient nodes ran through OpenCV, which the pack no longer installs. Their
replacements run on ComfyUI's own device and need nothing installed.

**Widgets and slots, carried across for you.** Thirty-six dropdowns reading `true` and `false`
are now checkboxes, and fourteen nodes that stopped at four to eight slots now declare
twenty-four or twenty-six, drawing one empty slot below the last one filled. Saved settings are
read back onto the new widgets and everything wired stays wired. The three batchers each gain a
`count` output below the one they had. `Image Save`'s `prefix_as_filename` is now
`overwrite_mode`, ticked where it read `true`.

**Three renamed labels.** `KSampler` is now `KSampler (Seed Socket)`, `Seed` is
`Seed (Number Outputs)` and `Save Video` is `Save Video (Advanced)`, so none shares a name with
a core ComfyUI node. Only the label changed, and the old names still find them in search.

**Three changed results.**

| Node | What changed | What to do |
|---|---|---|
| `Image Blending Mode` | 26 blend modes instead of 14, in linear light on the GPU, keeping values above white. `add` now adds the two layers rather than painting `image_b` over `image_a` | A graph using `add` comes out brighter. Switch it to `normal` for what it did before |
| `Image Style Filter` | 37 looks, every graded one finished with a halation. The 26 period looks keep the colour their name has always meant | Nothing. A saved workflow opens on the same style |
| `Mask Erode Region` | Holds the frame edge instead of treating outside the frame as unset | A mask that touched an edge comes out wider. One with a clear margin is unchanged |

---

# Nodes

**463 nodes** across 47 categories. See [`NODES.md`](NODES.md), which also groups them by the
`config.yaml` switch that gates them: [feature gates](NODES.md#feature-gates).

**28 of them are deprecated**, which ComfyUI marks in the Add Node menu. Each one opens its
description with the node to use instead.
All of them sit in the nine `legacy` groups in `config.yaml`, four of which are off until you
turn them on: [`docs/CONFIG.md`](docs/CONFIG.md#legacy) has the group table.

[**Content Viewer**](#content-viewer) renders whatever is wired into it, in the graph, and takes
view extensions of its own, such as the OpenReel video editor.

---
# Pushing texture into a generation while it is still forming

**Latent Affine** multiplies a latent and adds an offset to it, but only where a mask says
to. The mask is procedural grain, a repeating shape, a reading of the latent's own detail or
edges, or one you wire in: 25 patterns in all, each with its own settings on **Affine
Options**.

The same transform runs *during* sampling on **Affine Sampler**, **KSampler Affine Advanced**
and **Custom Sampler Affine Advanced**, on a curve **Affine Schedule** defines.

Every stock sampler carries it except `dpm_fast`, `dpm_adaptive`, `uni_pc` and `uni_pc_bh2`,
which sample normally.

On a video latent, `temporal_mode` sets whether the mask holds still, is redrawn each frame or
slides. The four content-aware patterns are read off the frame the model has resolved, so they
follow the subject and ignore it.

`affine_acts_on` on the Affine samplers is the setting worth understanding:

| Value | Multiplies |
|---|---|
| `content` | Only the picture the model has resolved so far. |
| `latent` | The whole latent, the sampler's own noise included. |

`latent` also multiplies the sampler's own noise draw, which prints as grain fixed in the
frame: 19.4 units of noise per unit of detail at 95% noise, against none for `content`. On a
model that holds a high noise level most of the way, such as MiniMax H3 at `shift = 12.0`,
prefer `content` and start the schedule later than the default.

**Affine Pattern Noise** puts the same transform on the starting draw instead of on a latent
part way through, so the noise carries the pattern from the first step. It replaces
**RandomNoise** on any custom sampler and takes the same **Affine Options**. The four patterns
read off a picture are not offered, since a starting draw has no picture to read.

[`NODES.md`](NODES.md) under **WAS Suite/Latent/Transform** and **WAS Suite/Sampling**. Graphs:
[`affine-krea2.json`](docs/workflows/affine-krea2.json) for a still image and
[`affine-minimax-h3-example.json`](docs/workflows/affine-minimax-h3-example.json) for the
affine running inside a sampler on a video latent.

---
# Filling out a video scene from the starting noise

**Temporal Noise Hold** replaces **RandomNoise** on any custom sampler, carrying each video
frame's noise pattern into the next rather than drawing every frame fresh. On MiniMax H3 the
same prompt and seed give a bare alley at `hold 2.0` and parked cars, shop doors and a lit
background street at `25.0`.

At `0.0` it matches **RandomNoise** exactly. Ancestral and SDE samplers draw fresh noise every
step and undo it.

[`NODES.md`](NODES.md) under **WAS Suite/Sampling**.

---
# Judging one render against another

**Compare Video** plays two videos on the node under a divider that drags left and right, both
from one clock, so the same frame is shown on each side. Wire a render into each socket.

[`NODES.md`](NODES.md) under **WAS Suite/Animation**.

---
# Deciding whether part of a graph runs at all

**Execution Gate** passes a value on while its switch is on. Switched off, the branch feeding
it is never evaluated and every node after it stops, so a slow sampler upstream and a Save
downstream both go unrun.

`bypass_downstream` sets those nodes to bypass on the canvas instead, where `open` is the
gate's own switch rather than something wired in. They are drawn as bypassed and never reach
the run. Left off, the nodes are stopped from the run and ComfyUI draws the first of them as
failed.

[`NODES.md`](NODES.md) under **WAS Suite/Logic**.

---
# Running a whole folder through a graph, keeping the filenames

**Load Image Batch** and **Load Image Sequence** answer a paired `image_list` and
`filename_list`. Wire the first into the graph and the second into **Image Save**'s
`filename_prefix`, and every result is written under the name it came in with. The two lists
stay in step.

[`NODES.md`](NODES.md) under **WAS Suite/IO**. Graph:
[`folder-round-trip.json`](docs/workflows/folder-round-trip.json).

---
# Your own prompt terminology, and a style library

`__animals__` in a prompt is replaced with a random word from the Noodle Soup Prompts pantry:
around 17,500 words across 84 terminologies. Browse it, add terminologies of your own, save
prompt pairs as styles, and move either store between machines as JSON or an AUTOMATIC1111
`styles.csv`. Both live in `was_state.db` beside your `config.yaml`.

[`NODES.md`](NODES.md) under **WAS Suite/Text/Terminology** and **WAS Suite/Text/Styles**.
Graphs: [`noodle-soup-pick.json`](docs/workflows/noodle-soup-pick.json) and
[`prompt-library.json`](docs/workflows/prompt-library.json).

---
# Content Viewer

<img src="docs/images/content-viewer.jpg" width="800">

### Wire anything into it and look at it.

Markdown, HTML, SVG, documents, code, JSON, CSV, logs and an image canvas, rendered in the node
and passed on unchanged. Two nodes, in [`NODES.md`](NODES.md) under **WAS Suite/View**.

**More views are installable.** Drop an extension's `.zip` into
`<ComfyUI user dir>/was-node-suite/viewer-extensions/` and set `install_extensions: true` under
`viewer:` in `config.yaml`. Two exist already:
[Image Search](https://github.com/WASasquatch/ComfyUI_Viewer_Image_Search_Extension) and
[OpenReel Video](https://github.com/WASasquatch/ComfyUI_Viewer_OpenReel_Extension).

---

# Three.js scenes

### Build a 3D scene out of wires, look at it on the canvas, and render it.

43 nodes for geometry, materials, lights, cameras, model loading, rendering and path tracing.
Off out of the box: set `threejs: true` under `features:` in `config.yaml`.

[`NODES.md`](NODES.md) under **WAS Suite/Three**. Five graphs in
[`docs/workflows/`](docs/workflows).

---

# Model weights

Nothing is downloaded unless you ask for it. `features.network` is `false` out of the box, so a node needing weights it cannot find says so and names the key rather than reaching for the network.

**Let it download.** Set `network: true` under `features:` in `config.yaml`.

## See **[docs/MODELS.md](docs/MODELS.md)** for more information.

---

# Configuration

Written for you on first start at `<ComfyUI user dir>/was-node-suite/config.yaml`, as a copy of
[`config.example.yaml`](config.example.yaml) with every key at its default and the comment that
explains it. Edit it there as it will survive updates.

## [`docs/CONFIG.md`](docs/CONFIG.md)

---

## Workflows

Runnable workflows live in [`docs/workflows/`](docs/workflows). Open one with **Workflow, Open**.
They name pictures and, in `hdr.json`, a checkpoint that will not be on your machine: pick your
own from each node's menu.

| | |
|---|---|
| [Configuration](docs/CONFIG.md) | Every setting, the feature groups that gate optional nodes, and how a group's packages are installed |
| [Model weights](docs/MODELS.md) | Which nodes need weights and where they go |

---

# Third-party code in the pack

The suite is MIT, see [`LICENSE`](LICENSE). It bundles 128 third-party files: three.js and
three-gpu-pathtracer, HugeRTE, Prism.js, KaTeX and Mermaid, the EMA-VFI network, the DejaVu and
Liberation fonts, eight face cascades and two sets of weights. Each keeps its own licence text
in the folder it ships in, and none of it is ever fetched. Every licence permits redistribution.

Versions, copyright lines, which licence covers what and a byte-level account of every bundled
copy: [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md).

The software licences place no conditions on what you make with the suite: documents, captions
and images you produce are yours. Open RAIL++-M, covering one 8 KB embedding, is the exception.
It claims no rights in output but its use restrictions reach any use of that output, including
the intrinsic maps **Power Preprocessor** answers with.

---

# Contributing

Bug reports and feature requests are welcome as issues. For a change to a node, open an issue first, so its inputs and outputs are settled before the code is written. See [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md).

<a href="https://github.com/WASasquatch/was-node-suite-comfyui/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=WASasquatch/was-node-suite-comfyui" />
</a>
