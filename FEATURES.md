# Features

What the pack does. Every entry names the nodes and what the area is for.
[`NODES.md`](NODES.md) carries every input, output and tooltip.

| | |
|---|---|
| Nodes | **468** across **48** categories |
| Deprecated | **28**, each naming its replacement |
| Gated | Nine `legacy` groups and several feature groups in `config.yaml`: [`docs/CONFIG.md`](docs/CONFIG.md) |
| Panels | 194 nodes draw their own readout, picture or editor on the canvas |
| Graphs | Runnable examples in [`docs/workflows/`](docs/workflows), linked per area below |

---

## Nodes that show their work

195 nodes draw a panel on themselves rather than answering only through a socket: both sides of
a picture, a histogram, a waveform, what a loader read off disk, a colour ramp, a colour wheel,
a 3D view, a text editor, a contact sheet that picks a look on click.

For seeing whether a node did what was wanted without wiring a preview to find out.

---

## Masking and regions

34 nodes. **CLIPSeg Masking** and **SAM Image Mask** make a mask from a prompt or from points.
**Mask Crop Region**, **Mask Dominant Region**, **Mask Minority Region** and **Mask Arbitrary
Region** isolate part of one. **Mask Grow**, **Mask Feather**, **Mask Dilate Region**, **Mask
Erode Region**, **Mask Smooth Region**, **Mask Fill Holes** and **Mask Guided Filter** shape it.
**Masks Add**, **Masks Subtract**, **Masks Combine Regions** and **Mask Invert** combine them,
and **Mask Statistics** measures one.

For building a mask inside the graph rather than painting one by hand.

[`NODES.md`](NODES.md) under **WAS Suite/Image/Masking**. Graphs:
[`mask-bounds.json`](docs/workflows/mask-bounds.json),
[`mask-rect-area.json`](docs/workflows/mask-rect-area.json).

---

## A layer stack on the wire

22 nodes. **Layers from Image Batch** and **Layer Edit** build a stack, **Layer Order**,
**Layer Align**, **Layer Fit** and **Layers Arrange** place it, **Layer Glow**, **Layer
Shadow**, **Layer Bevel**, **Layer Stroke** and **Layer Overlay** style it, and **Layers
Merge** flattens it. **Layers Canvas** draws the stack on the node.

For composition that would otherwise mean a round trip through an image editor.

[`NODES.md`](NODES.md) under **WAS Suite/Image/Layers**. Graph:
[`layers.json`](docs/workflows/layers.json).

---

## A layered PSD or TIFF, written and read

2 nodes. **Layers Save** writes a stack as a `.psd` or a layered `.tif`, and **Layers Load**
reads one back in. Each layer keeps its name, its place on the canvas, its opacity, its blend
mode and whether it was hidden, at 8 bit, 16 bit or 32 bit float. Both layouts are written
here, so nothing is installed for them.

For handing a composite to Photoshop, Affinity Photo, GIMP or Krita and taking the result
back, and for delivering an editable file instead of a flat picture.

The group starts on. Set `features.photoshop` to false in `config.yaml` to leave both nodes
out.

[`NODES.md`](NODES.md) under **WAS Suite/Image/Layers**. Graph:
[`layers-psd.json`](docs/workflows/layers-psd.json).

---

## A region as a value

16 nodes. **Image Bounds**, **Mask to Bounds** and **Bounding Boxes to Bounds** produce a
bounds value. **Bounded Image Crop**, **Bounded Image Blend** and their masked variants act on
one, **Inset Image Bounds** and **Bounding Boxes Filter** adjust it, **Draw Image Bounds**
shows it, and **Bounds to Mask**, **Bounds to Numbers** and **Bounds to Crop Data** convert it.

For measuring a region once, then cropping to it, working on it and pasting the result back.

[`NODES.md`](NODES.md) under **WAS Suite/Image/Bound**. Graph:
[`mask-bounds.json`](docs/workflows/mask-bounds.json).

---

## Filters, optics and processing

49 nodes. **Image Style Filter** carries 37 looks. **Image Bloom Filter**, **Image Chromatic
Aberration**, **Image Lens Distortion**, **Image Vignette**, **Image fDOF Filter**, **Image Film
Grain** and **Image Monitor Effects Filter** are optical. **Image SSAO (Ambient Occlusion)**
shades from a height map in 8, 16 or 32 bit and **Image SSDO (Direct Occlusion)** from depth.
**Vivid Sharpen**, **Image Lucy Sharpen**, **Image High Pass Filter**, **Image Median Filter**
and **Image Guided Filter** work on detail.

**Image Crop Face (YuNet)**, **Image Paste Face**, **Image Crop Region**, **Image Paste Crop**,
**Image Seamless Texture**, **Image Tiled**, **Image Draw Text**, **Image Pixelate**, **Image
Select Color**, **Image Remove Color** and **Create Grid Image** cover the processing side.

For grading and finishing a render inside the graph.

[`NODES.md`](NODES.md) under **WAS Suite/Image/Filter** and **WAS Suite/Image/Process**. Graphs:
[`image-style-filter.json`](docs/workflows/image-style-filter.json),
[`ssao-height-map.json`](docs/workflows/ssao-height-map.json).

---

## One node that measures a picture

4 nodes. **Power Preprocessor** answers 25 questions about an image: depth, surface
direction, body and animal pose, what every pixel is, edges, drawn lines, straight runs, the
paint and the light it was lit by, and the frame with its noise or its darkness taken out.
Picking the question redraws the node so only what that question reads is on it. **HDR
Reconstruct**, **Image Remove Background** and its model loader sit beside it.

Five answers need no model and most fetch a checkpoint on first use. **Marigold v2** is the
exception: pick it from the model menu on `depth_map`, `normal_map` or `albedo` and the node
reads a transformer off one socket and finds that map's adapter, decoder and prompt
embedding by name in ComfyUI's own model folders. One transformer serves all three maps, it
takes a single step, and it is the sharpest of the three. See
[`docs/MODELS.md`](docs/MODELS.md).

For feeding a ControlNet, and for relighting, defocus, parallax, masking and stylising.

[`NODES.md`](NODES.md) under **WAS Suite/Image/Preprocess**. Graph:
[`preprocessors.json`](docs/workflows/preprocessors.json),
[`marigold-v2.json`](docs/workflows/marigold-v2.json).

---

## Geometry and transforms

13 nodes. **Image Resize**, **Image Rotate (Advanced)**, **Image Perspective**, **Image Flip**,
**Image Transpose**, **Image Padding** and **Image Displacement Warp** move pixels. **Image
Tile Extract (Grid)**, **Image Tile Extract (Quadrants)**, **Image Tile Shuffle** and **Image
Stitch (Advanced)** split an image and put it back. **Tiled Image Upscale (With Model)** runs an
upscale model over overlapping tiles and cross-fades them, to any magnification rather than the
model's own, and its `precision` runs the model in half precision where the model declares that
safe.

For tiled work and for fitting an image to a target without leaving the canvas.

[`NODES.md`](NODES.md) under **WAS Suite/Image/Transform** and **WAS Suite/Image/Upscaling**.

---

## Colour, levels and LUTs

15 nodes. **Image Levels Adjustment**, **Image Curves**, **Image Auto Levels**, **Image Color
Balance**, **Image White Balance**, **Image Shadows and Highlights** and **Image Rotate Hue**
grade. **Image Color Match** matches one image to another and **Image Temporal Equalize**
matches across a batch. **Load LUT**, **Apply LUT**, **LUT Blender**, **LUT from Reference**
and **Save LUT (.cube)** carry a look as a file.

For matching shots to each other, and for moving a grade between graphs as a `.cube`.

[`NODES.md`](NODES.md) under **WAS Suite/Image/Adjustment** and **WAS Suite/Image/LUT**. Graph:
[`image-optics-and-grade.json`](docs/workflows/image-optics-and-grade.json).

---

## HDR and linear light

9 nodes. **Linear Light** and **Images to Linear** move an image out of picture codes.
**HDR Reconstruct** recovers range above white, **Image Dequantise** smooths banding, **Image
Tone Map** brings it back down, and **EXR Load**, **EXR Save** and **DNG Save** read and write
formats that hold it. **HDR VAE Decode** decodes without clipping.

**Image SSAO (Ambient Occlusion)** works in the same light: it shades without clipping
and its `precision` widget writes 8, 16 or 32 bit for EXR Save.

For work where values above white have to survive the filter chain.

[`NODES.md`](NODES.md) under **WAS Suite/Image/HDR**. Graphs:
[`hdr.json`](docs/workflows/hdr.json),
[`ssao-height-map.json`](docs/workflows/ssao-height-map.json).

---

## Instruments

**Image Histogram Chart**, **Image Waveform**, **Image Statistics**, **Image Color Palette** and
**Image Compare (Advanced)** measure a picture. **Latent Statistics** and **Latent Power
Spectrum** measure a latent. **Compare Video** plays two renders under a divider that drags.

For deciding whether a change improved anything, from a number rather than an impression.

[`NODES.md`](NODES.md) under **WAS Suite/Image/Analyze**, **WAS Suite/Latent** and
**WAS Suite/Animation**. Graph:
[`analysis-readouts.json`](docs/workflows/analysis-readouts.json).

---

## Logic and flow

35 nodes. **For Loop Open** and **For Loop Close**, **While Loop Open** and **While Loop
Close** iterate, and **Collect to List** gathers the results. **Condition Chain**, **Boolean
Reduce**, **Compare**, **Logic Comparison AND**, **Logic Comparison OR**,
**Logic Comparison XOR** and **Logic NOT** decide.
**Tensor Switch**, **Tensor Index Switch**, **Model Switch**, **Any Input Switch** and **Any
Switch (First Connected)** route, evaluating only the branch they pick. **Execution Gate**
stops a branch running at all. **Pause** holds a run for a look.

For graphs that vary per iteration, or that skip work a condition rules out.

[`NODES.md`](NODES.md) under **WAS Suite/Logic** and its groups. Graph:
[`loops-for-loop.json`](docs/workflows/loops-for-loop.json).

---

## Numbers

23 nodes. **Number Expression** evaluates a formula, **Number Operation** and **Number
Easing** shape a value, **Number Counter** and **Number Range** step one per run, and
**Number List Statistics** summarises many. **Image Size to Number**, **Latent Size to
Number**, **Image Aspect Ratio** and **Resolution Selector (Advanced)** read a size.
**Curve to Numbers** turns a drawn curve into values.

For driving a setting from something the graph worked out rather than a typed constant.

[`NODES.md`](NODES.md) under **WAS Suite/Number** and **WAS Suite/Number/Operations**.

---

## Text, lists and dictionaries

41 nodes. **Text Multiline**, **Text Multiline (Code Compatible)** and **Rich Text Editor**
author text. **Text List**, **Text Split to List**, **Text List Slice**, **Text List Get** and
**Text List to Numbers** work on lists. **Text Dictionary New**, **Text Dictionary Get**, **Text
Dictionary Keys**, **Text Dictionary Items** and **Text Dictionary Update** work on key and
value pairs. **Prompt Parse**, **Prompt Tag Cleanup**, **Text Parse A1111 Embeddings**, **Text
Add Tokens** and **Text Parse Tokens** handle prompt syntax. **Text Load Line From File**,
**Text Random Line** and **Load Text File** read from disk.

For prompt sets, per-run variation, and carrying structured values between nodes as text.

[`NODES.md`](NODES.md) under **WAS Suite/Text** and its groups. Graph:
[`prompt-lists.json`](docs/workflows/prompt-lists.json).

---

## Prompt terminology and a style library

`__animals__` in a prompt is replaced with a random word from the Noodle Soup Prompts pantry:
17,518 words across 82 terminologies, which ship with the pack and are read into the database on
first start. Terminologies and saved prompt pairs can be added, browsed, and moved between
machines as JSON or an AUTOMATIC1111 `styles.csv`. Both live in `was_state.db` beside
`config.yaml`.

For prompt variation without editing the prompt, and for carrying a style set across installs.

[`NODES.md`](NODES.md) under **WAS Suite/Text/Terminology** and **WAS Suite/Text/Styles**.
Graphs: [`noodle-soup-pick.json`](docs/workflows/noodle-soup-pick.json),
[`prompt-library.json`](docs/workflows/prompt-library.json).

---

## Files, folders and archives

32 nodes. **Load Image Batch** and **Load Image Sequence** answer a paired `image_list` and
`filename_list`, so wiring the second into **Image Save**'s `filename_prefix` writes every
result under the name it came in with. **Directory Listing**, **Path Exists** and **Download
Image** reach the filesystem and the network.

**Open ZIP**, **ZIP Add**, **Save ZIP**, **Zip Extract**, **ZIP Manage** and the three
`Load ... from ZIP` nodes put an archive on the wire. **Load Document**, **Save DOC**, **Text
to DOC**, **Convert DOC to HTML**, **Convert DOC to Plaintext** and **View DOC Metadata** do
the same for documents in six formats.

For batch work where output has to be matched back to input, and for graphs that read or write
an archive or a document directly.

[`NODES.md`](NODES.md) under **WAS Suite/IO**, **WAS Suite/Archive** and
**WAS Suite/Document**. Graphs:
[`folder-round-trip.json`](docs/workflows/folder-round-trip.json),
[`zip-archives.json`](docs/workflows/zip-archives.json),
[`document-export.json`](docs/workflows/document-export.json).

---

## Animation and video

11 nodes. **Load Video (Advanced)**, **Video Dump Frames** and **Video Frame Sample
(Advanced)** read a clip. **Create Video from Path**, **Write to Video**, **Write to GIF** and
**Save Video (Advanced)** write one. **EMA-VFI Frame Interpolation** adds frames, **Video Super
Resolution (PS-SR)** enlarges them, **Camera Motion Trajectory from Images** measures the move,
and **Create Morph Image** blends between stills.

For getting frames in and out of a graph, and for finishing a clip after sampling.

[`NODES.md`](NODES.md) under **WAS Suite/Animation** and **WAS Suite/IO**.

---

## A clip extended one segment at a time

**H3 Extend Window** opens one segment of a MiniMax H3 clip for a sampler. The first segment
samples an empty latent. Every segment after it picks up from the clip so far: `carry` copies
the clip's last frames into the window and masks them, so the sampler holds those frames and
generates only what follows, and `reference` decodes them and encodes them again as a video
reference. `refresh` decodes the carried frames, softens the fine detail they gained and
encodes them again, metered against the clip's opening frames rather than against a fixed
amount, and `handoff` starts the next segment on the last frame alone. `cut`, or an overlap
of `0`, samples a new scene from an empty latent with nothing carried, and H3 Extend Append
joins it after the clip's last full 17 frame block. `drift_control` takes the contrast and
fine detail the carried frames have gained back out again in latent space, each segment, and
never sharpens.
**H3 Extend Append** joins each sampled segment onto the clip and drops the carried head, so
no rendered frame is written twice. Overlaps and lengths snap to the model's frame grid, and
both nodes carry the audio with the video. A `carry` or `refresh` segment holds the soundtrack
for the whole audio steps its overlap covers, whatever length that overlap is, and
`audio_release` opens the mask back up over the last of those steps so the held take meets the
new frames without a step in level.

**MiniMax H3 Conditioning** writes the whole run on one node. Each row is one segment with its
own prompt, its own frame count, its own overlap and its own continuity, and a new row appears
as the last one is filled. A row's `continuity_N` is `as set` until it is changed, leaving H3
Extend Window's own setting to drive the run, and any other choice overrides it for that
segment alone. That is how one run holds a continuous scene, a cut that keeps the cast on
`reference`, and a cut to somewhere new on an overlap of `0`. `t2va` takes no pictures,
`i2va` opens on a first frame, `fl2va` closes on a last one and `fl2va_batched` runs every
segment between a neighbouring pair out of one batch, and only the pictures the mode reads
are drawn. Every prompt is encoded together before any sampling starts, so the text encoder
is loaded once rather than between segments.

`ref2va` builds every segment on references: up to 9 pictures, 3 clips each with its own
soundtrack, and 3 sounds, named `<Picture 1>`, `<Video 1>` and `<Audio 1>` in the prompts. Every
row carries the same references, so a cast, a location or a voice holds across carried shots
and cuts alike, and the `report` lists each tag with the size it was encoded at. A soundtrack
takes its `<Audio>` number ahead of the standalone sounds, and any audio needs `audio_vae`.
`ref_image_size` sets how large every reference picture is encoded: `match` fits it to the
clip's area, `256` to `2048` cap its longest side, which keeps very large pictures light, and
`max` takes it up to a 2048 pixel short edge for the closest likeness at several times the
sampling cost. Every picture lands between 256 and 5760 pixels a side.

`prompt_header` and `prompt_footer` go before and after every segment's prompt, so whatever
holds across the whole run is written once and each row carries only what that segment does.
A blank line separates them, and either left blank adds nothing.

`aspect_ratio` picks the canvas shape and `megapixels` its area, so a size is chosen the way
a resolution picker does. `custom` takes the shape from the pictures the mode reads, and 16:9
where it reads none. A `width` or `height` above `0` sets that side itself and the other is
worked out from the area, which makes the shape whatever those give. Sides round to a
multiple of 32.

The nodes run inside a loop. **While Loop Open** and **While Loop Close** carry the clip from
one segment to the next, `segments` sets the loop's count, and the clip is decoded once after
the last segment.

For growing a clip past the length a single sampling pass covers, and for directing it as it
grows.

[`NODES.md`](NODES.md) under **WAS Suite/Latent/Video** and **WAS Suite/Logic/Loop**. Graphs:
[`minimax-h3-extend-loop.json`](docs/workflows/minimax-h3-extend-loop.json),
[`minimax-h3-ref2va-extend-loop.json`](docs/workflows/minimax-h3-ref2va-extend-loop.json),
[`minimax-h3-flf-pair-loop.json`](docs/workflows/minimax-h3-flf-pair-loop.json).

---

## Texture pushed into a generation while it is still forming

**Latent Affine** multiplies a latent and adds an offset to it where a mask says to. The mask is
procedural grain, a repeating shape, a reading of the latent's own detail or edges, or one wired
in: 26 patterns, each with its settings on **Affine Options**.

The same transform runs during sampling on **Affine Sampler**, **KSampler Affine Advanced** and
**Custom Sampler Affine Advanced**, on a curve **Affine Schedule** defines. `affine_acts_on`
chooses between the picture the model has resolved and the whole latent. On a video model,
prefer `content`.

For adding grain, fabric and surface detail that a prompt will not reach.

[`NODES.md`](NODES.md) under **WAS Suite/Latent/Transform** and **WAS Suite/Sampling**. Graphs:
[`affine-krea2.json`](docs/workflows/affine-krea2.json),
[`affine-minimax-h3-example.json`](docs/workflows/affine-minimax-h3-example.json).

---

## Starting noise shaped before the first step

**Affine Pattern Noise** applies the affine transform to the starting draw, so the noise carries
a pattern's structure from the first step. **Temporal Noise Hold** carries each video frame's
noise into the next rather than drawing every frame fresh, which returns a denser, more detailed
scene the further it carries. Both replace **RandomNoise** on any custom sampler.

**WAS Latent Detail Boost**, **Blend Latents** and **Latent Batch (Advanced)** work on a latent
directly, and **SPEED Sampler** and **KSampler Cycle** are samplers of their own.

For steering a generation from its initialisation rather than correcting it later.

[`NODES.md`](NODES.md) under **WAS Suite/Sampling** and **WAS Suite/Latent**.

---

## Models and LoRA

16 nodes. **Power LoRA Loader** stacks LoRAs on one node, **Power LoRA Merger** bakes a stack
into a model, and **Apply Reweighted LoRA** reweights one. **BLIP Analyze Image** captions,
**MiDaS Depth Approximation** estimates depth, **Image Remove Background** cuts a subject out,
and **CLIPSeg Model Loader** and **SAM Model Loader** feed the masking nodes. **Model Info**
reports what a loaded model is.

Weights ship with the pack or download on first use to a stated folder:
[`docs/MODELS.md`](docs/MODELS.md).

[`NODES.md`](NODES.md) under **WAS Suite/LoRA**, **WAS Suite/Loaders** and
**WAS Suite/Image/AI**. Graph:
[`face-detection.json`](docs/workflows/face-detection.json).

---

## One render judged against another

**Compare Video** plays two videos on the node under a divider that drags left and right, both
from one clock. **Image Compare (Advanced)** does the same for stills, with metrics.

For settling whether a change to a graph improved anything.

[`NODES.md`](NODES.md) under **WAS Suite/Animation** and **WAS Suite/Image/Analyze**.

---

## Parts of a graph that do not run

**Execution Gate** passes a value on while its switch is on. Switched off, the branch feeding it
is never evaluated and every node after it stops. `bypass_downstream` draws those nodes as
bypassed on the canvas instead.

For skipping an expensive sampler, or a save, on a condition the graph works out.

[`NODES.md`](NODES.md) under **WAS Suite/Logic**.

---

## Content Viewer

<img src="docs/images/content-viewer.jpg" alt="Content Viewer" width="800">

Markdown, HTML, SVG, documents, code, JSON, CSV, logs and an image canvas, rendered in the node
and passed on unchanged. For inspecting what a graph is carrying without leaving the canvas.

Further views install as an extension `.zip`, configured under `viewer:` in `config.yaml`. Two
exist: [Image Search](https://github.com/WASasquatch/ComfyUI_Viewer_Image_Search_Extension) and
[OpenReel Video](https://github.com/WASasquatch/ComfyUI_Viewer_OpenReel_Extension).

[`NODES.md`](NODES.md) under **WAS Suite/View**. Graph:
[`content-viewer.json`](docs/workflows/content-viewer.json).

---

## Three.js scenes

43 nodes for geometry, materials, lights, cameras, model loading, rendering and path tracing,
with the scene drawn on the canvas. For building and rendering a 3D scene inside a graph.

Off out of the box: set `threejs: true` under `features:` in `config.yaml`.

[`NODES.md`](NODES.md) under **WAS Suite/Three**. Five graphs in
[`docs/workflows/`](docs/workflows).

---

## Graph plumbing

**Bus Node** and **Bus Node (Dynamic)** carry many wires as one. **Fast Groups** toggles groups
of nodes. **Free Memory** releases VRAM mid-graph. **Display Any**, **Text to Console** and
**Debug Number to Console** print what is on a wire. **Image History Loader** and **Text File
History Loader** reopen what a previous run produced. **App Workflow** turns a graph into a
form.

For keeping a large graph readable, and for seeing what is on a wire without wiring a preview.

[`NODES.md`](NODES.md) under **WAS Suite/Utilities**, **WAS Suite/Debug**,
**WAS Suite/History** and **WAS Suite/Workflow**. Graphs:
[`app-workflow.json`](docs/workflows/app-workflow.json),
[`comfyui-interop.json`](docs/workflows/comfyui-interop.json).
