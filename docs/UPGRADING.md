# Upgrading from v2

What v3 changes for an install and for a workflow saved with v2. Every node's inputs and outputs:
[`NODES.md`](../NODES.md).

## What changed since v2

779 nodes across a package of source files. The pack itself needs no packages, and nothing is
fetched from a git URL. What it bundles ships in the repository with its licence beside it,
listed in [`THIRD_PARTY.md`](THIRD_PARTY.md).

| | v2 | v3 |
|---|---|---|
| Nodes | 220 | 779 |
| Default packages installed | 20 | 0 |
| Installed from a git URL | 3 | 0 |
| Third-party carried in the repository | SAM and BLIP, 75 files of python | 176 files: browser libraries, fonts, eight face cascades, three sets of weights, five networks and one prompt, each with its licence |
| Optional node groups | none | 26 keys in `config.yaml`, 10 of them off out of the box. Per-node disable group. |

213 of the 220 node ids are unchanged. Four are retired onto replacements, three are removed
outright, and the section below says what opens in their place. Face detection, gradient maps, background removal, frame interpolation,
seamless textures, colour matching, levels, palettes, masks and deconvolution all run on torch,
on ComfyUI's own device, so OpenCV, numba, rembg, timm, scipy, scikit-image, scikit-learn and
matplotlib are not installed. Everything else a node reaches for either ships with ComfyUI or
belongs to an optional group.

What each area v3 adds: [`FEATURES.md`](../FEATURES.md).

---

## Opening a workflow saved before v3

No node id is renamed. Two nodes are now a different node under the same menu name, two ids
are gone and another node does the job of each, and ComfyUI offers to swap all four in. Five
ids are removed with nothing in their place. Widgets, slot counts, three menu labels and six
results changed, and a workflow saved before v3 is carried across as it loads.

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

**Five removed ids.** Each opens as a red box where it stood, with its links dropped. There
is no replacement for any of them; delete the node.

| Saved as | What it did | What to do |
|---|---|---|
| `Image Send HTTP` | Posted an image batch to an address | Write the batch with **Image Save** |
| `Text Random Prompt` | Read a prompt from lexica.art | Write the prompt, or use **Prompt Library** |
| `True Random.org Number Generator` | Drew a seed from RANDOM.ORG | Use **Random Number** or **Seed** |
| `WASDownloadImage` | Read an image from a web address | Save the file, then use **Load Image** |
| `WASNoodleSoupPantryRefresh` | Downloaded the Noodle Soup pantry | Nothing. The pantry ships with the pack and seeds the database on first start |

The pack now makes no outbound request of its own from any node.

**Widgets and slots, carried across for you.** Thirty-six dropdowns reading `true` and `false`
are now checkboxes, and fourteen nodes that stopped at four to eight slots now declare
twenty-four or twenty-six, drawing one empty slot below the last one filled. Saved settings are
read back onto the new widgets and everything wired stays wired. The three batchers each gain a
`count` output below the one they had. `Image Save`'s `prefix_as_filename` is now
`overwrite_mode`, ticked where it read `true`. `Image Save`'s `show_previews` is now the
`gallery` menu in the same place: `node gallery` where it read `true`, and `interface gallery`
where it read `false`. `Logic Boolean`'s 0 to 1 slider is a checkbox,
ticked where it read above 0.5.

**Three Custom Material takes a module file.** Its `javascript` box is gone. The node now picks
a `.js` or `.txt` module from ComfyUI's `input` folder or a folder under `paths.allow_read`, and
the module's header names the textures it wants, which the node's slots then read as. A body
typed into an older workflow is dropped rather than read onto the menu, so open one, put the
body in a file and pick it. Code no longer travels inside a workflow, so a graph from someone
else cannot bring javascript with it. [`CONFIG.md`](CONFIG.md#threejs) has the setting
that permits a module to run at all.

**Three renamed labels.** `KSampler` is now `KSampler (Seed Socket)`, `Seed` is
`Seed (Number Outputs)` and `Save Video` is `Save Video (Advanced)`, so none shares a name with
a core ComfyUI node. Only the label changed, and the old names still find them in search.

**Six changed results.**

| Node | What changed | What to do |
|---|---|---|
| `Image Blending Mode` | 26 blend modes instead of 14, in linear light on the GPU, keeping values above white. `add` now adds the two layers rather than painting `image_b` over `image_a` | A graph using `add` comes out brighter. Switch it to `normal` for what it did before |
| `Image Style Filter` | 37 looks, every graded one finished with a halation. The 26 period looks keep the colour their name has always meant | Nothing. A saved workflow opens on the same style |
| `Image Film Grain` | Adds grain that follows the picture's brightness and keeps its colour, instead of blending a grey, sharpened noise layer over it. `density`, `intensity`, `highlights` and `supersample_factor` are replaced by `strength`, `grain_size`, `color` and `seed` | Nothing. A saved workflow opens on the new defaults, which give fine film grain |
| `Logic Boolean` | Its input takes a wired INT, FLOAT, NUMBER or text as well as the checkbox, and reads a number as true unless it is 0, where it rounded at 0.5. The FLOAT output gives 1.0 or 0.0 instead of the slider's value | A FLOAT between 0 and 0.5 wired into it now reads true. Put **Logic Compare Numbers** in front of it, `a > b` against 0.5, for the old answer |
| `Mask Erode Region` | Holds the frame edge instead of treating outside the frame as unset | A mask that touched an edge comes out wider. One with a clear margin is unchanged |
| `Image SSAO (Ambient Occlusion)` | Traces rays across the map as real relief instead of averaging neighbouring pixels, so the shading follows the surface and no longer reacts to the image's own colour. The `depth_images` input is now `height_maps`, `tile_size` is gone, and `height_scale`, `ray_count`, `step_count` and `angle_bias` set the relief and the tracing. The whole body is float now, so linear light above 1.0 is shaded rather than clipped, and a `precision` widget writes 8, 16 or 32 bit for EXR Save | Re-wire the depth map into `height_maps`, which opens unconnected. The other settings are read back onto the new widgets. Highlight masking now starts off, so the node shades as soon as it is dropped, and `strength` at 0 leaves the image untouched rather than black |
