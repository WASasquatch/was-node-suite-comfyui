# Features

What the pack does. Every entry names the nodes and what the area is for.
[`NODES.md`](NODES.md) carries every input, output and tooltip.

| | |
|---|---|
| Nodes | **509** across **48** categories |
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

51 nodes. **Image Style Filter** carries 37 looks. **Image Bloom Filter**, **Image Chromatic
Aberration**, **Image Lens Distortion**, **Image Vignette**, **Image fDOF Filter**, **Image Film
Grain** and **Image Monitor Effects Filter** are optical. **Image SSAO (Ambient Occlusion)**
shades from a height map in 8, 16 or 32 bit and **Image SSDO (Direct Occlusion)** from depth.
**Vivid Sharpen**, **Image Lucy Sharpen**, **Image High Pass Filter**, **Image Median Filter**
and **Image Guided Filter** work on detail. **Image Morphology** erodes, dilates, opens and
closes with a square kernel, matching core Apply Morphology on any kernel size or batch length.
**Image Quantize** reduces each frame to a palette of its own, matching core Quantize Image with
the frames of a batch worked on at once.

**Image Crop Face (YuNet)**, **Image Paste Face**, **Image Crop Region**, **Image Paste Crop**,
**Image Seamless Texture**, **Image Tiled**, **Image Draw Text**, **Image Pixelate**, **Image
Select Color**, **Image Remove Color** and **Create Grid Image** cover the processing side.

For grading and finishing a render inside the graph.

[`NODES.md`](NODES.md) under **WAS Suite/Image/Filter** and **WAS Suite/Image/Process**. Graphs:
[`image-style-filter.json`](docs/workflows/image-style-filter.json),
[`ssao-height-map.json`](docs/workflows/ssao-height-map.json),
[`image-morphology-and-resize.json`](docs/workflows/image-morphology-and-resize.json).

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

42 nodes. **Text Multiline**, **Text Multiline (Code Compatible)** and **Rich Text Editor**
author text. **Text List**, **Text Split to List**, **Text List Slice**, **Text List Get** and
**Text List to Numbers** work on lists. **Text Dictionary New**, **Text Dictionary Get**, **Text
Dictionary Keys**, **Text Dictionary Items** and **Text Dictionary Update** work on key and
value pairs. **Prompt Parse**, **Prompt Tag Cleanup**, **Text Parse A1111 Embeddings**, **Text
Add Tokens** and **Text Parse Tokens** handle prompt syntax. **Text Load Line From File**,
**Text Random Line** and **Load Text File** read from disk. **Fast Generate Text** writes text
with the language model in a loaded CLIP, as core Generate Text does, on ComfyUI's graph captured
decode, and shows tokens per second on the node.

For prompt sets, per-run variation, and carrying structured values between nodes as text.

[`NODES.md`](NODES.md) under **WAS Suite/Text** and its groups. Graphs:
[`prompt-lists.json`](docs/workflows/prompt-lists.json),
[`fast-generate-text.json`](docs/workflows/fast-generate-text.json).

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

33 nodes. **Load Image Batch** and **Load Image Sequence** answer a paired `image_list` and
`filename_list`, so wiring the second into **Image Save**'s `filename_prefix` writes every
result under the name it came in with. **Image Load**, **Load Image Batch** and **Load Image
Sequence** read a 16-bit PNG at full precision, and a file marked linear with no curve applied.
**Image Save** encodes the files of a batch in parallel.
**Fast Save Animated WEBP** writes a batch as one animated WebP with its frames encoded in
parallel, lossless frames pixel for pixel. **Directory Listing** and **Path Exists** reach the
filesystem.

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

12 nodes. **Load Video (Advanced)**, **Video Dump Frames** and **Video Frame Sample
(Advanced)** read a clip. **Create Video from Path**, **Write to Video**, **Write to GIF** and
**Save Video (Advanced)** write one. **EMA-VFI Frame Interpolation** adds frames, **Video Super
Resolution (PS-SR)** enlarges them, **Camera Motion Trajectory from Images** measures the move,
and **Create Morph Image** blends between stills. **Audio Beats** finds a song's beats, bar
lines, tempo and per-frame loudness.

For getting frames in and out of a graph, for finishing a clip after sampling, and for timing
one to music.

[`NODES.md`](NODES.md) under **WAS Suite/Animation** and **WAS Suite/IO**.

---

## A video built one segment at a time

**MiniMax H3 Conditioning** writes every segment of a MiniMax H3 video on one node. Each row is
one segment with its own prompt, length, transition, overlap, source, header and footer choice,
model, hold, sound and seed, and a new row appears as the last one is filled. `prompt_header`
and `prompt_footer` wrap the rows; a section a row writes itself, as `overall_soundscape:`,
replaces theirs for that row, and `header_footer_N` takes either, both or neither. `t2va` takes
no pictures, `i2va` opens on a first frame, `fl2va` closes on a last one, `fl2va_batched` runs
every segment between a neighbouring pair out of one batch, and `ref2va` builds every segment on
shared references: up to 9 pictures, 3 clips each with its own soundtrack, and 3 sounds, named
`<Picture 1>`, `<Video 1>` and `<Audio 1>` in the prompts, at the size `ref_image_size` sets.
`aspect_ratio` and `megapixels` pick the canvas, a `width` or `height` above `0` sets that side,
and `loop` closes the last scene on the video's first frame. With `model_fl2va` and
`model_ref2va` both wired, each row's `model_N` picks the one that samples it: `auto` is ref2va
where the segment references anything and fl2va otherwise. Every prompt is encoded before
sampling starts.

**MiniMax H3 Asset** gives a segment its own picture, clip or sound: the frame it opens or closes
on, a keyframe pinned at any frame of it, or a reference its prompt names as `<Picture N>`,
`<Video N>` or `<Audio N>`, from ComfyUI's folders, any `paths.allow_read` folder or a wired
input. `(the video being made)` takes one frame of the video itself, for a later segment to
reference. Asset nodes chain, each into the next one's `assets`, and the last goes into `assets`
on the conditioning node.

**H3 Extend Window** opens each segment for a sampler and **H3 Extend Append** joins the result
onto the clip so far, dropping what was carried so no frame is written twice. The transition into
a segment is one of `carry` (the same shot goes on, picture and sound held), `refresh` (carried
with fresh noise, as much as `renewal` sets), `handoff` (a cut that opens on the last frame),
`reference (video)` (a cut that keeps the cast by referencing the last frames),
`reference (sample)` (a cut referencing stills from the whole clip), `cut` (a new scene, nothing
carried), `carry (audio only)` (a cut whose sound runs on, in whole 17 frame clips) and
`carry (audio) + reference (video)` (a cut that keeps both). A bridged cut that references the
scene before opens its prompt on a shot of that scene and cuts to its own shots where the bridge
ends. `sound_N` sets each segment's sound apart from its picture: `auto` as the transition does,
`carry` across any cut, `fresh` under a carried shot. `strength_N` loosens how firmly a segment
holds its pinned frames and references, and `seed_N` gives a segment a seed of its own. A cut
trims the 5 frames the model renders past its last whole clip. `drift_control` takes back the
contrast and detail carried frames gain, and `audio_release` opens a held soundtrack back up
where it meets new frames. With a `taeh3` decoder in `models/vae_approx`, `live_preview` draws
each segment as it samples.

**H3 Decode Video** decodes the finished clip one scene at a time into a frame cache on disk, each
scene that opens on a cut on its own, and **Save Video** writes it straight from the cache.
**Video Cache** keeps any long frame batch on disk the same way, and **Load Video Cache** opens a
cache again after a restart. **H3 Save Clip** writes a finished clip with every scene boundary,
and **H3 Load Clip** reads it back whole or as it stood after any scene, so a later scene renders
again without the ones before it. **H3 Repair Window** and **H3 Repair Splice** redraw a stretch
of a finished clip, picture, sound or both, and write it back with everything else untouched.
**H3 Soundtrack** lays one long track, a music bed or a recorded dialogue, under the whole video
so it runs on unbroken across carries and cuts. **H3 Control** drives each segment with its own
stretch of one control video through the H3 Fun ControlNet-Union patch, whose weights the pack
does not ship. **MiniMax H3 Clip Select** takes one clip of a run with its own empty latent, for a
loop that samples every clip from fresh.

**H3 Low VRAM** samples long clips in less memory with the same result, **H3 Tiles** runs every
model call in overlapping tiles across the frame and windows along the clip, and **H3 Tiled
Sampler** refines a long or upscaled clip that way with KSampler's settings. All three keep a
joined clip's scene cuts.

For growing a video past the length one sampling pass covers, scene by scene, and for going back
to any part of it.

[`NODES.md`](NODES.md) under **WAS Suite/Latent/Video**, **WAS Suite/Sampling**,
**WAS Suite/IO** and **WAS Suite/Logic/Loop**. Graphs:
[`minimax-h3-prompt-timeline.json`](docs/workflows/minimax-h3-prompt-timeline.json),
[`minimax-h3-extend-loop.json`](docs/workflows/minimax-h3-extend-loop.json),
[`minimax-h3-ref2va-extend-loop.json`](docs/workflows/minimax-h3-ref2va-extend-loop.json),
[`minimax-h3-flf-pair-loop.json`](docs/workflows/minimax-h3-flf-pair-loop.json),
[`minimax-h3-scene-loop.json`](docs/workflows/minimax-h3-scene-loop.json),
[`minimax-h3-extend-loop-audio-carry.json`](docs/workflows/minimax-h3-extend-loop-audio-carry.json).

---

## The Prompt Timeline

**The Prompt Timeline** is the editor window MiniMax H3 Conditioning opens from its **Open Prompt
Timeline** button: the run's scenes along a time ruler with tracks for pinned frames and
references, a media bin of every picture, clip and sound the pack may read, the chosen scene's
settings, and a monitor that plays each segment as it samples and the finished video once the run
saves it. Dragging sets a scene's length, its overlap, its order, a keyframe's frame and which
scene a reference belongs to. **Reference frame** makes the frame under the playhead a picture
reference in any scene. Every edit writes the node's rows and a chain of MiniMax H3 Asset nodes, so
the graph runs the same with the window closed, and one undo takes back one gesture. The **Show
the Open Prompt Timeline button** setting, under WAS Node Suite, hides the button.

With a language model wired into `vlm_clip`, the window writes too. **Write scenes** turns a
description of the video into every scene's prompt, a shared header and footer, and a transition
between each pair of scenes, casting each scene from the reference pictures in the asset chain.
**Rewrite** rewrites one scene's prompt to directions, or writes it from them where it is empty.
**Plan transitions** reads the scenes as they stand and picks each cut and carry. The **LLM
settings** tab holds the system prompt every one of them is written under and how the model draws
its words. Each job is a queued prompt of its own, so it waits behind any run, and the model is
never loaded by a render. **MiniMax H3 Scene Writer**, **MiniMax H3 Prompt Rewrite** and **MiniMax
H3 Plan Transitions** are the nodes those jobs run, and work on a graph of their own as well.

For laying out a whole video by eye, and for drafting it from a few sentences.

[`docs/H3_COND_TIMELINE.md`](docs/H3_COND_TIMELINE.md) for the window, tab by tab.
[`NODES.md`](NODES.md) under **WAS Suite/Latent/Video**. Graph:
[`minimax-h3-prompt-timeline.json`](docs/workflows/minimax-h3-prompt-timeline.json).

---

## Fast motion held for a refining pass

**H3 De-RoPE Stretch** finds where a MiniMax H3 clip moves fastest from its own latent, shows
those frames several times over, and encodes the stretched clip with its audio slowed to
match, pitch kept, as the start latent for any sampler. `mode` picks how much is held:
`balanced`, `wide`, `economy`, or `manual` with the threshold, peak hold, bridge and ramp.
`audio_mode` picks how much of the audio the pass re-renders the same way, and `denoise` carries
the strength to the sampler. With the sampler's model wired through it, `low_vram` runs each
model block over the stretched clip in slices and keeps only the weights that fit beside it on the
card, streaming the rest, for the same output at a lower memory peak.

**H3 De-RoPE Recover** takes the held frames back out of the decoded result and passes the source
audio out beside them, in time with the recovered frames.

The clip comes in as frames, as a latent, or both. For a clip generated from text, H3 De-RoPE
Stretch takes the sampler's finished output, reads its motion and its audio from it, and needs no
decode and encode to measure it.

For re-rendering fast action at a slowed pace and putting it back on the clip's own timing, in a
finished H3 clip, in the output of an earlier pass, or in a clip generated from text.

[`NODES.md`](NODES.md) under **WAS Suite/Latent/Video**. Graphs:
[`minimax-h3-derope.json`](docs/workflows/minimax-h3-derope.json) for a finished clip,
[`minimax-h3-derope-t2v.json`](docs/workflows/minimax-h3-derope-t2v.json) for a clip generated from text.

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
directly, and **SPEED Sampler** and **KSampler Cycle** are samplers of their own. **Fast
KSampler** samples exactly as KSampler does with a cheaper live preview: the preview decoder
stays loaded, the preview is decoded at the size it is shown, and `preview_every` skips steps.

**CNS Model Patch** is colored noise sampling on any model: every sampler that adds noise each
step, such as `euler_ancestral`, `dpmpp_2m_sde`, `er_sde`, RES4LYF's samplers or RES4SHO's
`hfx_stochastic`, moves that noise toward the frequency bands the image has not finished yet
instead of spreading it evenly. Each run measures how far every band had come at each step and
the next run of that model at that size uses the measurement, so it needs no setup for any
model. `auto` uses the published settings, one set for runs with CFG above 1 and one without;
`manual` sets the divider, power, tilt and energy by hand. Graphs:
[`cns-klein9b.json`](docs/workflows/cns-klein9b.json) and
[`cns-krea2.json`](docs/workflows/cns-krea2.json).

For steering a generation from its initialisation rather than correcting it later.

[`NODES.md`](NODES.md) under **WAS Suite/Sampling** and **WAS Suite/Latent**. Graph:
[`fast-samplers-and-savers.json`](docs/workflows/fast-samplers-and-savers.json).

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
bypassed on the canvas instead. **Execution Gate Controlboard** lists every Execution Gate and
Any Gate in the workflow, subgraphs included, with a switch on each and **All open** and **All
closed** beside them. A gate whose `open` comes from a subgraph input or a Boolean node is
switched there; one whose `open` the graph computes is listed as `wired` and left to it.

For skipping an expensive sampler, or a save, on a condition the graph works out, and for turning
a large workflow's branches on and off from one node.

[`NODES.md`](NODES.md) under **WAS Suite/Logic**. Graphs:
[`execution-gate.json`](docs/workflows/execution-gate.json),
[`execution-gate-controlboard.json`](docs/workflows/execution-gate-controlboard.json).

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
