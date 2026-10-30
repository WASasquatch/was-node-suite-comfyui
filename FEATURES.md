# Features

What the pack does. Every entry names the nodes and what the area is for.
[`NODES.md`](NODES.md) carries every input, output and tooltip.

| | |
|---|---|
| Nodes | **659** across **66** categories |
| Deprecated | **28**, each naming its replacement |
| Gated | Nine `legacy` groups and several feature groups in `config.yaml`: [`docs/CONFIG.md`](docs/CONFIG.md) |
| Panels | 362 nodes draw their own readout, picture or editor on the canvas |
| Graphs | Runnable examples in [`docs/workflows/`](docs/workflows), linked per area below |

---

## Nodes that show their work

195 nodes draw a panel on themselves rather than answering only through a socket: both sides of
a picture, a histogram, a waveform, what a loader read off disk, a colour ramp, a colour wheel,
a 3D view, a text editor, a contact sheet that picks a look on click.

For seeing whether a node did what was wanted without wiring a preview to find out.

---

## Masking and regions

36 nodes. **CLIPSeg Masking** and **SAM Image Mask** make a mask from a prompt or from points,
and **Image Skin Mask** and **Image Simple Skin Mask** from the people in a picture.
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

## Skin masks

**Image Skin Mask** and **Image Simple Skin Mask** mask the bare skin of every person in a
picture and leave out clothes, hair, background, eyes, brows and the inside of the mouth, with
an edge that follows the picture. Image Skin Mask reads each body part with Meta's Sapiens2
network and holds bare torsos, highlights and tattoos on one or two people close up. Image
Simple Skin Mask reads a small clothing model, is the lighter of the two, and holds small
figures in a crowd or a wide shot. In both, each person's own skin colour settles
what the network takes for clothing. Both keep all skin, the face alone or the body alone, and
draw the picture with its skin tinted.

For retouching, grading or inpainting skin without touching what is around it.

[`NODES.md`](NODES.md) under **WAS Suite/Image/Masking**. Graph:
[`image-skin-mask.json`](docs/workflows/image-skin-mask.json). The models, and the terms the
Sapiens2 weights come with: [`docs/MODELS.md`](docs/MODELS.md#skin-masks).

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

4 nodes. **Power Preprocessor** answers 22 questions about an image: depth, surface
direction, body and animal pose, what every pixel is, edges, drawn lines, straight runs, the
paint and the light it was lit by, and the frame with its noise or its darkness taken out.
Picking the question redraws the node so only what that question reads is on it. **HDR
Reconstruct**, **Image Remove Background** and its model loader sit beside it.

Five answers need no model and most fetch a checkpoint on first use. **Marigold v2** is the
exception: pick it from the model menu on `depth_map`, `normal_map` or `albedo` and the node
wires a Load Diffusion Model on its transformer, finds that map's adapter, decoder and prompt
embedding by name in ComfyUI's own model folders, and names any file still to download in
ComfyUI's missing models list. One transformer serves all three maps, it takes a single
step, and it is the sharpest of the three. See [`docs/MODELS.md`](docs/MODELS.md).

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

36 nodes. **For Loop Open** and **For Loop Close**, **While Loop Open** and **While Loop
Close** iterate, and **Collect to List** gathers the results. **Condition Chain**, **Boolean
Reduce**, **Compare**, **Logic Comparison AND**, **Logic Comparison OR**,
**Logic Comparison XOR** and **Logic NOT** decide.
**Tensor Switch**, **Tensor Index Switch**, **Model Switch**, **Any Input Switch** and **Any
Switch (First Connected)** route, evaluating only the branch they pick. **Execution Gate**
stops a branch running at all. **Pause** holds a run for a look. **Combo Primitive** copies the
dropdown it is connected to and sets it from one place, **Combo Selector** names a model file,
sampler or scheduler, and **Text to Combo** chooses by name, each on a wire any dropdown takes.

For graphs that vary per iteration, or that skip work a condition rules out.

[`NODES.md`](NODES.md) under **WAS Suite/Logic** and its groups. Graph:
[`loops-for-loop.json`](docs/workflows/loops-for-loop.json).

---

## Numbers

25 nodes. **Number Expression** evaluates a formula, **Number Operation** and **Number
Easing** shape a value, **Number Counter** and **Number Range** step one per run, and
**Number List Statistics** summarises many. **Number Sequence** builds a list from a start and
end or from typed values, shaped by a curve, repeated as a sawtooth, a wave or a staircase, and
drawn on the node with its points dragged into place. **Number List Plot** draws up to four
lists on the node. **Image Size to Number**, **Latent Size to
Number** and **Image Aspect Ratio** read a size. **Resolution Selector (Advanced)** picks one
from a shape and an area in megapixels, or from one edge in pixels, with every shape listed
portrait and landscape and set by dragging the corner of the frame it draws.
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
decode, takes a system prompt, hands a reasoning model's thinking out on its own socket, and shows
on the node how long loading, reading the prompt and writing each took.

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

## A system prompt library

System prompts and skill files are `.txt` and `.md` files in
`ComfyUI/user/was-node-suite/system_prompts`, in folders of any depth, each carrying its name,
model family and description as front matter. **Load System Prompt** draws the library on the node,
searched and filtered by folder and model family, with the chosen prompt's text, and hands that
text to a model's `system_prompt` input. **Save System Prompt** files a prompt into it.

A folder holding `SKILL.md` is a skill, and **Import** on the node adds one, or a folder of
prompts, from a folder on disk; **Import ZIP** adds the same from a zip, a skill's pictures and
videos with it. A zip holding a program, a link, a path out of its folder, a zip bomb or a picture
or video that is not what its name says is refused whole. The skill's `SKILL.md` is the system
prompt, and its other text files, such as `references/layout.md`, leave Load System Prompt on its
`documents` output for an LLM index, so a local model looks up what the skill references while it
answers.

The library ships with built-in prompt enhancers under **Built-in**, read-only, one folder per
model family: FLUX.1 to FLUX 3 Klein, Krea 2, LTX-Video and LTX-2, MiniMax H3, Wan, HunyuanVideo,
Kandinsky 6, Qwen-Image, Z-Image, ERNIE Image, MOSS-Image, Fibo and the tag-based SDXL models. Each
turns a short idea, a picture or an edit request into the prompt that model reads, and where the
model's makers publish an enhancer prompt, it is theirs, unchanged or with its changes listed.
**Copy** on a built-in writes an editable copy into the library and loads it on the node.

For keeping the instructions that steer Qwen, Gemma and other language models in one place,
picking one per node instead of pasting it, running skills written for Claude Code and Codex on
a local model, and expanding a few words into the prompt a particular image or video model
expects.

[`NODES.md`](NODES.md) under **WAS Suite/Text/System Prompts**. Files, front matter, skills and
the built-in prompts: [`docs/SYSTEM_PROMPTS.md`](docs/SYSTEM_PROMPTS.md). Graphs:
[`system-prompt-library.json`](docs/workflows/system-prompt-library.json),
[`llm-skill-prompt-enhancer.json`](docs/workflows/llm-skill-prompt-enhancer.json),
[`built-in-prompt-enhancers.json`](docs/workflows/built-in-prompt-enhancers.json).

---

## MOSS-Image captions and layouts written by a model

**MOSS-Image Compile Prompt** turns the caption or box layout a language model drafts, as the
MOSS-Image prompt enhancer skill has it write them, into the prompt MOSS-Image reads. A layout's
boxes are placed on the size being generated, and its summary tags, canvas and box tags are kept.
The panel lists the caption and every box.

For prompt enhancement that places subjects and lettering where the model is told to, on the
model already loaded for the generation.

[`NODES.md`](NODES.md) under **WAS Suite/Text**. Graph:
[`llm-skill-prompt-enhancer.json`](docs/workflows/llm-skill-prompt-enhancer.json).

---

## Qwen3 32B and other language models as a CLIP

**Language Model Loader** loads a language model from `ComfyUI/models/text_encoders` as a CLIP
that writes text, for core Generate Text, **Fast Generate Text** and **LLM From CLIP**. It reads
Qwen3 14B and 32B as one safetensors file in Hugging Face key names, in bfloat16 or quantized in
ComfyUI's own formats, and loads every other language model as ComfyUI reads it. Its panel shows
the model, its size, its layers and how its weights are stored. A prompt encoder with no output
layer, such as MiniMax H3's Qwen3-VL 32B, is refused by name.

For prompt writing, chat and retrieval on a larger model than the text encoders image models
ship with.

[`NODES.md`](NODES.md) under **WAS Suite/Loaders**. Files:
[`docs/MODELS.md`](docs/MODELS.md#language-models). Graph:
[`language-model-loader.json`](docs/workflows/language-model-loader.json).

---

## Llama-Index in ComfyUI

The nodes under **WAS Suite/LLM** are llama-index on the canvas: each part of a llama-index
pipeline, from the model to the agent, is a node, wired and queued like any other and run on the
models a graph wires in:

- **Models.** **LLM From CLIP** uses the language model inside a core **Load CLIP** or a
  **Language Model Loader**, such as Qwen3 or Gemma, with pictures. **LLM GGUF Loader** runs a GGUF file, and **LLM Transformers
  Loader** a Hugging Face model folder. **LLM OpenAI**, **LLM Anthropic**, **LLM Google Gemini**
  and **LLM Provider** reach hosted models and local servers such as Ollama and LM Studio, with
  keys read from the environment. **LLM Embedding Model** embeds text, and **LLM Unload** frees
  the models between steps.
- **Writing.** **LLM Complete** answers a prompt, the text a CLIP Text Encode can take. **LLM
  Chat** and **LLM Message** hold a conversation with pictures and sound. **LLM Structured
  Output** fills a schema as JSON. **LLM Prompt Template**, **LLM Token Count** and **LLM
  Messages to Text** handle the text around them.
- **Documents.** **LLM Load Documents**, **LLM Load Document**, **LLM Load Web Page**, **LLM
  Documents From DOC** and **LLM Text to Document** read text in, from notes and code to PDFs,
  Word documents, e-books and slide decks. **LLM To Documents** turns whatever a graph holds into
  documents: text, a list of text, pictures, frames spread across a video, a DOC, a conversation,
  found sources or table rows. **LLM Split Documents** cuts documents into chunks eight ways,
  **LLM Extract Metadata** labels the chunks with titles, summaries, questions, keywords and the
  names they mention, and **LLM Documents to Text** joins them for a prompt.
- **Pictures.** **LLM Load Images** reads a folder of renders with the prompt saved in each file,
  and **LLM Images to Documents** takes frames from the graph, either one described by a vision
  model on request. **LLM Image Embedding Model** and **LLM Image Index** make them searchable by
  words or by another picture, and **LLM Sources to Images** hands the pictures found back as an
  image batch.
- **Tables.** **LLM Load Table** reads CSV, Excel, JSON and SQLite files into tables that are
  only ever read, **LLM SQL Query Engine** answers questions about them by writing SQL, and **LLM
  Table to Documents** turns rows into documents for an index. **LLM SQL Retriever** finds the
  rows that answer a question, for a retriever tool or a recursive retriever.
- **Indexes and retrieval.** Vector, summary, tree, keyword, knowledge graph, document summary
  and image indexes, saved and loaded by name, and brought up to date with **LLM Index Insert**,
  which skips unchanged documents and replaces changed ones. **LLM Vector Store** keeps a vector
  index in a Qdrant, Chroma, LanceDB or FAISS collection on this machine, or on a Qdrant, Chroma
  or Postgres server, reopened with **LLM Vector Store Index**. **LLM Index Retriever**, **LLM
  BM25 Retriever**, **LLM Fusion Retriever**, **LLM Auto-Merging Retriever**, **LLM Router
  Retriever** and **LLM Auto Retriever**, which writes its own metadata filters, find chunks.
  **LLM Transform Retriever** rewrites the question first, as a made-up answer or a simpler
  question, and **LLM Index Node** with **LLM Recursive Retriever** searches short summaries
  first, then the retriever each one points to. **LLM Postprocessor** and **LLM Rerank** filter
  and reorder what is found before **LLM Retrieve** or **LLM Sources to Text** hands it on.
- **Answers and chat.** **LLM Query Engine** answers from retrieved chunks under a system
  prompt, such as a skill from Load System Prompt whose files are the indexed documents, with
  citations, showing a vision model the pictures found, and
  **LLM Query Transform**, **LLM Sub-Question Query Engine**, **LLM Router Query Engine** and
  **LLM Retry Query Engine** reshape, route or retry each question for **LLM Query**. **LLM Chat Engine** and **LLM Chat Session**
  hold a chat over them on a chat panel that saves each conversation under a name, with a menu
  to start, open, rename and delete them. **LLM Memory** adds static, facts and vector blocks,
  **LLM Memory History** reads a conversation back and **LLM Clear Memory** deletes one.
- **Agents.** **LLM Tool**, **LLM Query Engine Tool** and **LLM Retriever Tool** give **LLM
  Agent** a calculator, dates, Wikipedia, web pages, web search, a document and the indexes,
  **LLM MCP Tools** adds the tools of any MCP server named in config.yaml, and
  **LLM Run Agent** runs one agent or a team that hands work to each other, with every tool call
  in its trace.
- **Evaluation.** **LLM Evaluate** scores an answer eight ways, one of them against a second answer, **LLM Generate Questions** writes
  test questions from documents, and **LLM Evaluate Retriever** measures a retriever on them.

For prompt enhancement on the model already loaded for a generation, questions answered from a
folder of notes or a spreadsheet, finding past renders by what they show, and agents that look
things up before they answer.

The group is `features.llm`, off until its requirements are installed. [`NODES.md`](NODES.md)
under **WAS Suite/LLM**. Keys, folders and servers: [`docs/LLM.md`](docs/LLM.md). Graphs:
[`llm-chat-with-notes.json`](docs/workflows/llm-chat-with-notes.json),
[`llm-research-agent.json`](docs/workflows/llm-research-agent.json),
[`llm-picture-search.json`](docs/workflows/llm-picture-search.json),
[`llm-table-questions.json`](docs/workflows/llm-table-questions.json),
[`llm-vector-store.json`](docs/workflows/llm-vector-store.json),
[`llm-retrievers.json`](docs/workflows/llm-retrievers.json),
[`llm-to-documents.json`](docs/workflows/llm-to-documents.json),
[`llm-skill-prompt-enhancer.json`](docs/workflows/llm-skill-prompt-enhancer.json).

---

## Files, folders and archives

33 nodes. **Load Image Batch** and **Load Image Sequence** answer a paired `image_list` and
`filename_list`. Wired into **Image Save**'s `filenames`, the second saves every result under
the name it came in with, and text wired into `captions` is written beside each image under
the same name, `cat.png` with `cat.txt`, a captioned training set in one node. Both take one
value per image from a list output or a `LIST`, or one value for the whole batch. **Image
Save** also saves into ComfyUI's `input` folder and every folder `paths.allow_read` or
`paths.allow_write` lists, so a dataset is edited where its loaders read it.
**Image Load**, **Load Image Batch** and **Load Image
Sequence** read a 16-bit PNG at full precision, and a file marked linear with no curve applied.
**Image Save** encodes the files of a batch in parallel.
**Fast Save Animated WEBP** writes a batch as one animated WebP with its frames encoded in
parallel, lossless frames pixel for pixel. **Directory Listing** and **Path Exists** reach the
filesystem. Every file and folder menu a loader reads from also takes a path wired in as text,
such as `E:\images` from a String node, inside the folders the menu lists; **Image Load**,
**Load Video** and **Load Video Upload** take it on their `path` socket and keep their upload
button.

**Open ZIP**, **ZIP Add**, **Save ZIP**, **Zip Extract**, **ZIP Manage** and the three
`Load ... from ZIP` nodes put an archive on the wire. **Load Document**, **Save DOC**, **Text
to DOC**, **Convert DOC to HTML**, **Convert DOC to Plaintext** and **View DOC Metadata** do
the same for documents in six formats.

For batch work where output has to be matched back to input, and for graphs that read or write
an archive or a document directly.

[`NODES.md`](NODES.md) under **WAS Suite/IO**, **WAS Suite/Archive** and
**WAS Suite/Document**. Graphs:
[`folder-round-trip.json`](docs/workflows/folder-round-trip.json),
[`image-save-captions.json`](docs/workflows/image-save-captions.json),
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
replaces theirs for that row, and `header_footer_N` takes either, both or neither. A tag written
`<Subject 1: He-Man>` gives `<Subject 1>` a name the prompts can use in its place
([named tags](docs/H3_COND_TIMELINE.md#named-tags)). `t2va` takes
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
carried), `carry (audio only)` (a cut whose sound runs on, in whole 17 frame clips),
`carry (audio) + reference (video)` (a cut that keeps both) and `carry + reference (sample)` (the
same shot goes on and references stills from the scene it continues, so its set holds whatever
the prompt describes). A bridged cut that references the
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
[`minimax-h3-extend-loop-audio-carry.json`](docs/workflows/minimax-h3-extend-loop-audio-carry.json),
[`minimax-h3-extend-loop-derope.json`](docs/workflows/minimax-h3-extend-loop-derope.json)
with each segment de-roped before it joins.

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

## MiniMax H3 attention made sparse

**H3 Sparse Attention** runs MiniMax H3's attention block-sparse with Sol-Attn, SLA or VSA,
layer by layer, while the layers H3's video suffers most for stay dense on the model's own
attention backend. Presets set which layers stay dense, and a written plan gives any layer a
setting of its own. Sparse layers compute in INT8 and soften fine detail.

For faster, lighter H3 sampling where the speed is worth some softness.

[`NODES.md`](NODES.md) under **WAS Suite/Sampling**. Graph:
[`minimax-h3-sparse-attention.json`](docs/workflows/minimax-h3-sparse-attention.json).

---

## Pika H3 Flex

**Pika H3 Flex LoRA** turns a MiniMax H3 fl2va or ref2va model into Pika H3 Flex: the matching
Pika adapter in bypass or merge mode, the release's video and audio shifts and its fp32 AdaLN.
**Pika H3 Flex Sampling** gives the release's six-step sigmas and core's re-noise sampler from a
knot list, which also writes a refining pass or a ladder that climbs back. **Pika H3 Flex
Canvas** gives the trained 480p and 768p sizes at eleven aspects and the frame count for 5 to
15 seconds. **H3 Hires Latent** encodes upscaled frames beside a clip's own sound for a second
pass at the larger size.

For Pika H3 Flex video with sound in six model calls, from text, frames or references.

[`NODES.md`](NODES.md) under **WAS Suite/Sampling** and **WAS Suite/Latent/Video**.

## MiniMax H3 in four to eight steps

**H3 PTD Patch** gives MiniMax H3 a Parametric Trajectory Distillation file's per-step adapters
and output heads. The file is picked in ComfyUI's own **Load Model Patch**, and a stock sampler
on euler, scheduler `simple` and cfg 1 then finishes a clip with sound in the steps the file was
trained for, which the node hands to the scheduler. It reads the Ref2VA release on a ref2va
transformer and the fl2va release on an fl2va one, in bf16 or int8, on pruned and quantized
transformers, with other LoRAs still applied. Its panel shows the release, the steps, the
trained sigmas and what it checked about the transformer. A sampler given sigmas other than the
file's steps, such as the other release's, stops before its first step and names the right ones.
The adapters stay in system memory and reach the GPU one block at a time, tens of megabytes on
top of what the plain transformer needs. A PTD file can also carry a LoRA, held once and applied
at every step beside the step adapters, so a distilled style or adapter ships as one file.
**H3 PTD Sigmas** reads those steps back off any model made from the patch, so an extend loop
that switches between fl2va and Ref2VA samples every segment on its own file's steps.

For finished H3 clips at a fraction of the usual steps.

[`NODES.md`](NODES.md) under **WAS Suite/Sampling**. Files:
[`docs/MODELS.md`](docs/MODELS.md#minimax-h3). Graph:
[`minimax-h3-ptd.json`](docs/workflows/minimax-h3-ptd.json).

---

## Kandinsky 6 video with sound

Kandinsky 6 runs on ComfyUI's own nodes: **Load Diffusion Model** loads the transformer,
**DualCLIPLoader** with type `kandinsky5` and **CLIP Text Encode** read the prompt, **KSampler**
samples the picture and the sound together, and **VAE Decode (Tiled)** and **VAE Decode Audio**
turn them into frames and a soundtrack. **Empty Kandinsky 6 Latent** starts a clip, **Kandinsky 6 Image To
Video** starts one on a picture, **Kandinsky 6 Audio VAE Loader** loads the sound's decoder, and
**Kandinsky 6 Text Encode** takes what is seen and what is heard in separate boxes. Pro and Lite,
distilled and not, as released or as the int8 ConvRot and W6A8 checkpoints the pack publishes.

For text-to-video and image-to-video with a soundtrack generated alongside the picture.

[`NODES.md`](NODES.md) under **WAS Suite/Latent/Video** and **WAS Suite/Loaders**. Weights and
settings: [`docs/MODELS.md`](docs/MODELS.md#kandinsky-6). Graphs:
[`kandinsky6-lite-distill.json`](docs/workflows/kandinsky6-lite-distill.json),
[`kandinsky6-pro-distill.json`](docs/workflows/kandinsky6-pro-distill.json).

---

## Fibo 2.0 images and edits

Bria's Fibo 2.0 runs in ComfyUI: **Load Diffusion Model** loads Dev or Edit Turbo, **Load CLIP**
loads Qwen3-VL 4B, and **Text Encode Fibo 2** turns a structured JSON caption into the prompt and
negative, with up to five images to edit and a mask to repaint one. Captions that ask for text are
shown to the model as pictures of that text, as in training. **Fibo 2 Scheduler** gives the
released sampling schedule for the image size, and **Generate Fibo 2 Prompt** writes the caption
from a short idea, from pictures, or by revising an earlier caption, on the same Qwen3-VL. The
weights are Comfy-Org's, and Comfy-Org's Load CLIP type `fibo2` and its Fibo 2.0 model work with
these nodes too.

For photographs and designs that follow a long description closely, legible lettering, and edits
that change one thing and keep the rest.

[`NODES.md`](NODES.md) under **WAS Suite/Conditioning**, **WAS Suite/Sampling** and **WAS
Suite/Text**. Weights and settings: [`docs/MODELS.md`](docs/MODELS.md#fibo-20). Graph:
[`fibo2-generate-edit.json`](docs/workflows/fibo2-generate-edit.json).

---

## Fast motion held for a refining pass

**H3 De-RoPE Stretch** finds where a MiniMax H3 clip moves fastest from its own latent, shows
those frames several times over, and encodes the stretched clip with its audio slowed to
match, pitch kept, as the start latent for a refining pass. `mode` picks how much is held:
`balanced`, `wide`, `economy`, or `manual` with the threshold, peak hold, bridge and ramp.
`audio_mode` picks how much of the audio the pass re-renders the same way, and `denoise` carries
the strength on. A prompt wired through it comes out with its pinned frames moved to where those
frames land in the stretched clip, so a first or last frame still opens or closes it. With the
model wired through it, `low_vram` runs each model block in slices for a lower memory peak.

**H3 De-RoPE Refine** samples the stretched latent in one pass, or in windows as long as H3's
longest pass where the stretch runs past it, each carrying the clip it shares with the window
before. A SAMPLER, such as an affine-wrapped one, can stand in for the named sampler.

**H3 De-RoPE Recover** takes the held frames back out, keeping the first showing of each or the
one nearest the source frame, as frames with the source audio beside them for Create Video, or,
with the video VAE wired, as an H3 latent carrying the source audio.

The clip comes in as frames, as a latent, or both. For a clip generated from text, H3 De-RoPE
Stretch takes the sampler's finished output and reads its motion and its audio from it. Inside a
Prompt Timeline loop, each segment is de-roped between its sampler and H3 Extend Append, refined
by the model that sampled it: the frames it carried stay as they were sampled, its last clip is
never held, and the next segment continues from the de-roped result.

For re-rendering fast action at a slowed pace and putting it back on the clip's own timing, in a
finished H3 clip, in a clip generated from text, or in every segment of a long video.

[`NODES.md`](NODES.md) under **WAS Suite/Latent/Video**. Graphs:
[`minimax-h3-derope.json`](docs/workflows/minimax-h3-derope.json) for a finished clip,
[`minimax-h3-derope-t2v.json`](docs/workflows/minimax-h3-derope-t2v.json) for a clip generated from text,
[`minimax-h3-extend-loop-derope.json`](docs/workflows/minimax-h3-extend-loop-derope.json) for
every segment of a Prompt Timeline video.

---

## Texture pushed into a generation while it is still forming

**Latent Affine** multiplies a latent and adds an offset to it where a mask says to. The mask is
procedural grain, a repeating shape, a reading of the latent's own detail or edges, or one wired
in: 26 patterns, each with its settings on **Affine Options**.

The same transform runs during sampling on **Affine Sampler**, **KSampler Affine Advanced** and
**Custom Sampler Affine Advanced**, on a curve **Affine Schedule** defines. `affine_acts_on`
chooses the picture the model has resolved, the noise alone or the whole latent, and
`affine_keep_colour` holds the overall or regional colour while the multiplier scales detail.
On a video model, prefer `content`.

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

**Latent Detail Boost**, **Blend Latents** and **Latent Batch (Advanced)** work on a latent
directly. Latent Detail Boost treats the video half of a MiniMax H3 latent and passes its sound
through unchanged. **SPEED Sampler** and **KSampler Cycle** are samplers of their own. **Fast
KSampler** samples exactly as KSampler does with a cheaper live preview: the preview decoder
stays loaded, the preview is decoded at the size it is shown, and `preview_every` skips steps.

**CNS Model Patch** is colored noise sampling on any model: every sampler that adds noise each
step, such as `euler_ancestral`, `dpmpp_2m_sde`, `er_sde`, RES4LYF's samplers or RES4SHO's
`hfx_stochastic`, moves that noise toward the frequency bands the image has not finished yet
instead of spreading it evenly. Each run measures how far every band had come at each step and
the next run of that model at that size uses the measurement, so it needs no setup for any
model. `auto` uses the published settings, one set for runs with CFG above 1 and one without;
`manual` sets the divider, power, tilt and energy by hand. Its panel shows how many noise draws
the run coloured, on which sampler, or warns when the sampler adds no noise to colour. Graphs:
[`cns-klein9b.json`](docs/workflows/cns-klein9b.json) and
[`cns-krea2.json`](docs/workflows/cns-krea2.json).

For steering a generation from its initialisation rather than correcting it later.

[`NODES.md`](NODES.md) under **WAS Suite/Sampling** and **WAS Suite/Latent**. Graph:
[`fast-samplers-and-savers.json`](docs/workflows/fast-samplers-and-savers.json).

---

## The distilled FLUX 3 Image Klein on its base model

**Flux 3 Flow Map Patch** gives the FLUX 3 Image Klein base model the distilled checkpoint's
target-time branch. With one of the distillation LoRAs from
[WAS/flux3_klein](https://huggingface.co/WAS/flux3_klein) on the base, it samples as the
distilled model does from 16 steps: the distilled picture from the same seed at CFG 1, and
more contrast and colour at CFG 2.5 to 3.5, which the distilled model does not take.

For the distilled model's look with CFG, on the base model.

[`NODES.md`](NODES.md) under **WAS Suite/Sampling**. Graph:
[`flux3-klein-distill-lora.json`](docs/workflows/flux3-klein-distill-lora.json). Files:
[`docs/MODELS.md`](docs/MODELS.md#flux-3-image-klein).

---

## A large frame sampled in tiles

**Tiled Diffusion Model Patch** makes a model denoise the latent as overlapping tiles inside any
sampler, each tile a picture the model's native size, blended where they meet. `start_percent`
and `end_percent` are shares of the model's own noise schedule: the steps before `start_percent`
lay out the whole frame, and the tiles take over from there. A shift set on the model before the
patch, by **Flow Shift** or ModelSamplingFlux, lays out only those whole-frame steps, and the
tiles run at the model's own shift.

For detail across a frame larger than the model's native size, with one composition and no
repeated subjects.

[`NODES.md`](NODES.md) under **WAS Suite/Sampling**. Graphs:
[`flux3-klein-tiled-diffusion.json`](docs/workflows/flux3-klein-tiled-diffusion.json),
[`moss-image-high-res.json`](docs/workflows/moss-image-high-res.json). Files:
[`docs/MODELS.md`](docs/MODELS.md#flux-3-image-klein),
[`docs/MODELS.md`](docs/MODELS.md#moss-image).

---

## A flow model's shift and schedule

**Flow Shift** sets the shift of a flow model, such as FLUX 3 Image Klein, MOSS-Image, FLUX or
SD3, as a plain factor, or grows it with the square root of the picture's area. **Shifted Tangent
Scheduler** makes the two-stage arctangent schedule of RES4LYF's `bong_tangent` in the model's
own time, so the shift moves where the steps fall, with every setting of the curve exposed. The
same schedule is `shifted_tangent` in every KSampler's scheduler menu.

For more steps on the layout of a large frame, and a tangent schedule that follows the model's
shift.

[`NODES.md`](NODES.md) under **WAS Suite/Sampling**. Graph:
[`flux3-klein-tiled-diffusion.json`](docs/workflows/flux3-klein-tiled-diffusion.json).

---

## Fewer model passes where sampling is smooth, one more where detail forms

**Spectrum Forecast** runs some of a sampler's steps through the whole transformer and forecasts
the rest from them, between `start_percent` and `end_percent` of the model's noise schedule. With
`taper` on, the full steps sit closer together toward both ends of that window. MOSS-Image,
FLUX.1, Fibo 2, Qwen-Image, Wan and MiniMax H3 forecast their last block; other models forecast
their output. Forecast steps soften fine detail.

**Adaptive Detail** tells the model a slightly lower noise level on the steps where detail forms,
and the picture comes out with more fine texture in the composition its plain steps make. `detail`
sets how much, and the push stops at `end_percent`, leaving the last steps to clean it up. `extra`
gives the steps with the most fine-detail error a corrective second pass, picked during the run.
It works on samplers that call the model once per step, such as euler, and after Spectrum Forecast
on the same model, where it brings back the detail the forecasts soften.

Each node's panel shows the run's steps: which ran in full or were forecast, and which were
refined.

For faster sampling, and for more fine detail from the same steps without moving the picture.

[`NODES.md`](NODES.md) under **WAS Suite/Sampling**. Graph:
[`spectrum-and-adaptive-detail.json`](docs/workflows/spectrum-and-adaptive-detail.json).

---

## MOSS-Image without banding, and past its native size

**Model Block Restore** puts a range of a model's transformer blocks back to another checkpoint of
the same model, with the LoRAs on the model still applied. Blocks 32 to 41 from MOSS-Image Base
remove the vertical banding MOSS-Image SFT draws across skies and smooth backgrounds.

**RoPE Extension Patch** scales a model's RoPE on every call longer than its native side.
MOSS-Image widens its RoPE frequencies and FLUX compresses its positions. Tiles at native size
from Tiled Diffusion Model Patch stay unscaled.

For MOSS-Image SFT pictures with clean gradients, and for frames longer than the model's native
side.

[`NODES.md`](NODES.md) under **WAS Suite/Sampling**. Graph:
[`moss-image-high-res.json`](docs/workflows/moss-image-high-res.json). Files:
[`docs/MODELS.md`](docs/MODELS.md#moss-image).

---

## Models and LoRA

16 nodes. **Power LoRA Loader** stacks LoRAs on one node, **Power LoRA Merger** bakes a stack
into a model, and **Apply Reweighted LoRA** reweights one. A list on Power LoRA Loader's
strength or on one row's weight, such as Number Sequence's, sets it at every sampling step, for
a LoRA that fades out once the layout is set, comes in only for the detail, or hands over to
another LoRA partway through. Graph:
[`number-sequence-lora-schedule.json`](docs/workflows/number-sequence-lora-schedule.json). Power LoRA Loader
hands its rows on a LoRA stack wire, and Power LoRA Merger merges a stack from that wire, from the loader's model
output or from another pack's stack nodes in place of its own rows. The pack's LoRA nodes leave
out any part of a LoRA that does not fit the model, with one line in the console naming it: a
LoRA trained on a pruned MiniMax H3 applies to an unpruned one without its time modulation, and
the reverse. **BLIP Analyze Image** captions,
**MiDaS Depth Approximation** estimates depth, **Image Remove Background** cuts a subject out
and hands on its mask for compositing, and **CLIPSeg Model Loader** and **SAM Model Loader** feed the masking nodes. **Model Info**
reports what a loaded model is.

Weights ship with the pack or download on first use to a stated folder:
[`docs/MODELS.md`](docs/MODELS.md).

[`NODES.md`](NODES.md) under **WAS Suite/LoRA**, **WAS Suite/Loaders** and
**WAS Suite/Image/AI**. Graphs:
[`face-detection.json`](docs/workflows/face-detection.json),
[`power-lora-stack.json`](docs/workflows/power-lora-stack.json).

---

## A model quantized in ComfyUI's own formats

**Quantize Model** turns a loaded diffusion model into int8 convrot, w6a8, w4a8, int8, fp8, mxfp8
or nvfp4, with any LoRAs applied before it merged in, and writes the file Load Diffusion Model
reads. Set to `like Comfy-Org`, it makes the file Comfy-Org publishes for that model and format.
Its panel shows the size before and after, what every other format would come to, and the layers
that lost the most.

For a smaller, faster copy of any finetune or merge, laid out as the official releases are.

[`NODES.md`](NODES.md) under **WAS Suite/Utilities**. Formats and releases:
[`docs/MODELS.md`](docs/MODELS.md#quantized-models).

---

## One render judged against another

**Compare Video** plays two videos on the node under a divider that drags left and right, both
from one clock. **Image Compare (Advanced)** does the same for stills, with metrics, showing
both at full resolution however far the graph is zoomed in.

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

The group starts on. Set `features.threejs` to false in `config.yaml` to leave the 43 nodes
out.

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
