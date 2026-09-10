# **WAS** Node Suite v3 &nbsp; ![ComfyUI](https://img.shields.io/badge/ComfyUI-Custom_Node-blue) [![Open Manager](https://img.shields.io/badge/Open_Manager-Compatible-yellow)](https://github.com/WASasquatch/open-manager-comfyui) ![License](https://img.shields.io/badge/License-MIT-green) [![Donate](https://img.shields.io/badge/Donate-PayPal-blue.svg)](https://paypal.me/ThompsonJordan?country.x=US&locale.x=en_US) 

<img src="was-node-suite-v3.png" width="600">

WAS Node Suite has been going since 2023 on Civitai and on GitHub, and was among the
first packs to put hundreds of nodes into users' hands. WAS-NS has over a million downloads, 
and is used by thousands of users daily. It has been MIT since the first commit: use it, change it, 
teach with it, or run it in paid services.

The pack contains **467 nodes for ComfyUI**, across images, filters and colour, masking, 
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
`<ComfyUI user dir>/was-node-suite/`, and python compiles the pack to bytecode. An update
recompiles, so it happens once more each time you pull.

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

467 nodes across a package of source files. The pack itself needs no packages, and nothing is
fetched from a git URL. What it bundles ships in the repository with its licence beside it,
listed in [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md).

| | v2 | v3 |
|---|---|---|
| Nodes | 220 | 467 |
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

## 251 new nodes

| Area | New | Area | New |
|---|--:|---|--:|
| Three.js scenes | 43 | HDR and linear light | 9 |
| Files, archives and documents | 28 | Affine sampling | 8 |
| Logic and flow | 25 | Everything else | 108 |
| Layers | 20 | | |
| Bounds | 10 | | |

What each area does: [`FEATURES.md`](FEATURES.md).

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

**467 nodes** across 47 categories. [`NODES.md`](NODES.md) carries every input, output and
tooltip, and groups them by the `config.yaml` switch that gates them:
[feature gates](NODES.md#feature-gates).

**28 are deprecated**, which ComfyUI marks in the Add Node menu, and each names its
replacement. They sit in the nine `legacy` groups in `config.yaml`, four of which start off:
[`docs/CONFIG.md`](docs/CONFIG.md#legacy).

---
# Features

[`FEATURES.md`](FEATURES.md) covers what the pack does, by area: masking, layers, bounds,
filters and optics, transforms, colour and LUTs, HDR, instruments, logic and flow, numbers,
text and dictionaries, prompt terminology and styles, files and archives, animation, models
and LoRA, the affine transform, noise shaping, video comparison, execution gating, the
Content Viewer, Three.js scenes and graph plumbing.

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
