# Configuring WAS Node Suite

Your config file is written on first start and lives outside the pack. Updating or
reinstalling leaves it alone. `config.example.yaml` is the template.

| Order | Location | Notes |
|---|---|---|
| 1 | `$WAS_CONFIG` | A full path to a config file. A value that is not a file is reported and skipped. |
| 2 | `<user_dir>/was-node-suite/config.yaml` | The normal place. `<user_dir>` is ComfyUI's own `user/` directory. |
| 3 | `config.yaml` beside this pack's own `__init__.py` | Read where the pack is installed, in `ComfyUI/custom_nodes/was-node-suite-comfyui/`. Lost on a clean reinstall. |
| 4 | built-in defaults | Used for any key the file above does not set. `config.example.yaml` lists every key at its default. |

`config.json` is accepted in place of `config.yaml` and is never rewritten.

`was_state.db` is a SQLite file. Two ComfyUI instances pointed at one config directory share
it safely. The four `.json` files beside it are read on first start and never written to, so
they stay as a copy of the state at the moment it moved into the database.

| Variable | Effect |
|---|---|
| `WAS_CONFIG` | Full path to the config file to read, ahead of every other candidate. |
| `WAS_CONFIG_DIR` | Relocates the whole `was-node-suite` directory, config file and writable state together. |

| File or directory | Holds |
|---|---|
| `config.yaml` | This configuration. |
| `was_state.db` | Custom text tokens, node cursors, the image, text-file and output-image history, the style library and the Noodle Soup Prompts terminology. |
| `was_suite_settings.json` | Read into `was_state.db` on first start and then left alone. |
| `was_history.json` | Read into `was_state.db` on first start and then left alone. |
| `styles.json` | Imported into the style library each time the file changes. Never written to. |
| `nsp_pantry.json` | Read into `was_state.db` on first start and then left alone. |
| `wildcards/` | Wildcard `.txt` files, unless `paths.wildcards` names another directory. |
| `cache/` | Files written by `Cache Node`, when `legacy.cache` is on. |

Each release brings your file up to date from the template: new keys arrive at their default,
retired keys are removed, and the file as it was is kept beside it as `config.yaml.bak`. Your
values are kept. A `features` group still holding the default from the release that wrote your
file is moved to this release's default, and every move is named in the log.

| What your `features:` group holds | What the rewrite does |
|---|---|
| The default it was given | Brings it forward to this release's default, and names it in the log under `turned on` or `turned off`. Where the two defaults are the same, which is most groups most of the time, that is no change and there is nothing to log |
| Any other value | Keeps it exactly as it is, and names it in the log under `kept` |
| Nothing. The key is not in the file | The key arrives from the template at this release's default, and is named in the log under `added` |
| A value for a group the file's revision does not carry, written in by hand | Keeps it exactly as it is, and logs nothing about it. There is no default to recognise it against |

---

## `version`

| Key | Type | Default |
|---|---|---|
| `version` | int | `13` |

Written by the pack. Editing it changes which release's defaults your file is compared
against.

---

## `logging`

| Key | Type | Default | Does |
|---|---|---|---|
| `logging.level` | `debug` \| `info` \| `warning` \| `error` | `info` | Lowest level the pack's own loggers emit. `debug` adds dependency probes, model resolution, cache eviction, per-module load times and the node ids a disabled group holds back. |
| `logging.rich` | bool | `true` | Colourised output, when `rich` is installed. Falls back to a plain stream handler when it is not, or when this is `false`. |
| `logging.quotes` | bool | `false` | Accepted and has no effect: no node or loader prints a quote. |

---

## `paths`

| Key | Type | Default | Does |
|---|---|---|---|
| `paths.wildcards` | path or `null` | `null` | Directory of wildcard `.txt` files. `null` means `<user_dir>/was-node-suite/wildcards`. |
| `paths.styles` | path or `null` | `null` | A style library to import. A `.json` file or an Automatic1111 `styles.csv` is read into the library each time the file changes, and is never written to. A style the library holds that this file has never named is kept. `null` means `<user_dir>/was-node-suite/styles.json`. |
| `paths.luts` | path or `null` | `null` | Directory of `.cube` colour lookup tables, and where **Save LUT** writes. `null` means `<user_dir>/was-node-suite/luts`. ComfyUI's `models/LUT` is read after this directory, whichever it is. |
| `paths.allow_read` | list of paths | `[]` | Extra directories nodes may read from. See [Containment](#containment). |
| `paths.allow_write` | list of paths | `[]` | Extra directories nodes may write to. See [Containment](#containment). |

Two folders have no key. **Image Draw Text** reads typefaces from
`<user_dir>/was-node-suite/fonts`, alongside the nine that ship with the pack, and a `.ttf` or
`.otf` dropped there joins the `font` menu after a restart. On a portable Windows install that
is `ComfyUI_windows_portable\ComfyUI\user\was-node-suite\fonts`. **App Workflow** lists every
`*.app.json` under `<user_dir>/default/workflows/`, including subfolders.

`~` is expanded in every path setting. A relative path resolves against the process working
directory, which for ComfyUI is not a stable place; prefer absolute paths.

A directory added to `allow_read` is offered in every file and folder menu under its own name,
and its files carry that name in brackets: `D:/photos` appears as `photos`, holding
`sunset.png [photos]`. A directory added to `allow_write` is offered in every `root` menu the
same way. Folder menus list three levels deep and stop at 2000 entries. Restart ComfyUI after
changing either list.

### Containment

A node input naming a file or directory is resolved against these roots before the
filesystem is touched. A path landing outside all of them is refused and nothing is read or
written.

| | Read | Write |
|---|---|---|
| ComfyUI `input/` | yes | **no**, a write there is moved to `temp/`, below |
| ComfyUI `output/` | yes | yes |
| ComfyUI `temp/` | yes | yes |
| The pack's own directory, `<user_dir>/was-node-suite/`, or `$WAS_CONFIG_DIR` | yes | yes |
| Anything in `paths.allow_read` | yes | no |
| Anything in `paths.allow_write` | no | yes |

Subdirectories of a permitted root are permitted. A write that resolves inside `input/` is
rewritten to the same relative path under `temp/` and logged with both paths.

Both sides of a comparison are fully resolved first, which covers `..` segments and symlinks: a
symlink inside a root pointing outside it does not widen the boundary. Comparison is
case-folded on Windows. Every entry must be an existing directory.

A refusal names the path, every permitted root and the key that would permit it.

---

## `text`

| Key | Type | Default | Does |
|---|---|---|---|
| `text.strip_comments` | bool | `true` | Reserved for comment stripping in multiline text. Comment handling does not consult it: `Text Multiline` always drops lines whose first non-blank character is `#`, and `Text Multiline (Code Compatible)` never does. Setting it to `false` changes nothing. |

---

## `document`

| Key | Type | Default | Does |
|---|---|---|---|
| `document.clean_html` | bool | `true` | Removes script and frame markup from the HTML a document node emits. Exactly what goes and what stays is below. |

Removes `<script>` and `<iframe>` elements with their contents, the `<object>` and `<embed>`
tags while keeping their fallback, every attribute whose name starts with `on`, and every
`javascript:` link. Nothing else is touched, and the box on the node is never rewritten.

It cleans up markup written on your own machine. It is not a hardened sanitiser for untrusted
pages.

| Removed | Kept from it |
|---|---|
| `<script>` elements | nothing, the tags and everything between them go |
| `<iframe>` elements | nothing, the tags and everything between them go |
| `<object>` and `<embed>` tags | everything between them, which is the fallback a reader is shown when the embed does not load |
| Every attribute whose name starts with `on`, such as `onclick` or `onerror` | the element and its text |
| `javascript:` in an `href`, `src`, `action`, `formaction`, `data`, `poster`, `cite`, `ping`, `background`, `longdesc`, `srcset` or `xlink:href` | the element and its text, so a scripted link becomes plain words |

The **Rich Text Editor** parses what it is given into a document and writes that document
back out, so markup is normalised on the way through. None of the rewrites below is what
`document.clean_html` does, none of them can be switched off, and they reach every document
the editor is given.

| What happens | What you get back |
|---|---|
| A top-level node the schema will not take as a body child is wrapped in a paragraph | `Hello` reads back `<p>Hello</p>`. So does an `<iframe>`, `<object>`, `<textarea>`, `<input>`, `<math>` or an element nobody has heard of, standing on its own |
| An element the schema will not allow where it stands is unwrapped, moved, or given the parent it needs | `<td>a</td>` reads back `<p>a</p>`; `<li>a</li>` reads back `<ul><li>a</li></ul>`; `<table><tr><td>x</td></tr></table>` gains a `<tbody>`; a `<body>` inside a `<div>` is unwrapped |
| Five characters are written as entities: `<`, `>` and `&` in text, and those three plus `"` and a backtick in an attribute value | `<p>AT&T</p>` reads back `<p>AT&amp;T</p>`; `<p>5 < 6</p>` reads back `<p>5 &lt; 6</p>`; `title='a"b'` reads back `title="a&quot;b"`. An apostrophe is not one of the five |
| Every other entity is decoded to the character it names, and written back as that character | `&copy; &eacute; &#169;` reads back `© é ©`; `&nbsp;` reads back as a literal non-breaking space. `&amp;amp;` survives: it decodes to `&amp;` and re-encodes to `&amp;amp;` |
| Whitespace in text collapses | A tab or a newline in text becomes a space and a run of spaces becomes one: `<p>a    b</p>` reads back `<p>a b</p>`. Whitespace between blocks is dropped rather than collapsed, and whitespace at the two ends of the document is trimmed. Whitespace inside `<pre>` is left alone |
| HTML tag and attribute names come back lowercase and every attribute value comes back double quoted | `<DIV CLASS=x>a</DIV>` reads back `<div class="x">a</div>`. SVG keeps its camel case: `<linearGradient>` stays as written |
| A void element loses its XHTML slash | `<p>a<BR/>b</p>` reads back `<p>a<br>b</p>` |
| A `style` attribute is reparsed and reserialised | `style="font-weight:700; COLOR:#ABCDEF"` reads back `style="font-weight: bold; color: #abcdef;"`; a `url()` is requoted with single quotes, whatever quotes it was written with |
| A `style` declaration is dropped if it holds `javascript:`, `vbscript:` or `data:image/svg+xml` in a `url()`, or is `behavior`, or holds `expression()` or a CSS comment; a `style` attribute left empty by that goes with it | `style="color:red;background:url('javascript:go()')"` reads back `style="color: red;"`; `style="background:url('javascript:go()')"` loses the attribute |
| An `on*` attribute on a descendant of `<svg>` is deleted | `<svg><circle onload="go()" r="4"></circle></svg>` reads back `<svg><circle r="4"></circle></svg>`; a nested `<a onclick>` and a `<div onclick>` inside `<foreignObject>` go the same way |
| A boolean attribute is rewritten as `name="name"`, and an attribute the schema gives a default for is added | `disabled=""` reads back `disabled="disabled"`, a bare `checked` reads back `checked="checked"`, and `<input name="a">` gains `type="text"` |
| A document, head or metadata element is dropped | `<!DOCTYPE html><html><head><title>t</title></head><body><p>x</p></body></html>` reads back `<p>x</p>`. A `<style>` block goes the same way wherever it sits, and so do `<link>` and `<meta>` |
| A CDATA section or an XML processing instruction is not read as markup, which is what HTML does | `<![CDATA[raw & <stuff>]]>` reads back `<![CDATA[raw & <stuff]]><p>]]&gt;</p>`; `<?xml version="1.0"?>` reads back `<!--?xml version="1.0"?-->` |
| An empty block gains a `<br>`, and a document that is nothing but one empty paragraph comes back as the empty string | `<p></p>` reads back `<p><br></p>`, `<div></div>` reads back `<div><br></div>`, and `<p>&nbsp;</p>` on its own reads back as nothing at all |
| An internal `data-mce-*` attribute is removed | `<p data-mce-style="color:red">a</p>` reads back `<p>a</p>` |

---

## `history`

| Key | Type | Default | Does |
|---|---|---|---|
| `history.display_limit` | int | `36` | How many recent entries a History node's combo lists, and how many saved files `Image Save` reports back to the frontend. The stored history is not trimmed; this is a display cap. |

---

## `video`

| Key | Type | Default | Does |
|---|---|---|---|
| `video.extra_codecs` | map of code to container extension | `{}` | Registers extra codec codes on the video writer and overrides the container extension of a code already there. For example `{libx265: .mkv}`, or `{avc1: .mp4}` to change where AVC1 is muxed. |

There is nothing to set for ffmpeg. Encoding runs in-process through `av`, which ComfyUI
already requires.

---

## `interface`

| Key | Type | Default | Does |
|---|---|---|---|
| `interface.preview_max_edge` | int | `1024` | Longest edge, in pixels, of the picture a node publishes to its own interface. `0` publishes it at the size the node received it, up to the channel's own ceiling of 8 MB a frame. |

A 3840x2160 frame costs 7.9 MB and 315 ms to hold and encode, and 1.4 MB and 136 ms at 1024.

---

## `viewer`

| Key | Type | Default | Does |
|---|---|---|---|
| `viewer.install_extensions` | bool | `false` | Unpacks `.zip` view extensions dropped in `<user_dir>/was-node-suite/viewer-extensions/` at startup, and pip-installs whatever they require with the python that runs ComfyUI. Also copies in view extensions cloned into `custom_nodes` as `ComfyUI_Viewer_*`. |

---

## `dependencies`

| Key | Type | Default | Does |
|---|---|---|---|
| `dependencies.install_missing` | bool | `false` | Installs the requirements file of a switched-on feature group at startup when a package in it is absent. `false` logs the command to run instead. |

Off out of the box, so a fresh install brings in nothing. A switched-on group missing a package
is named in the log with the command that installs it, and the pack runs pip only once this is
set to `true`.

Set to `true`, a group with a requirements file installs it at startup when the group is on and
something it needs is absent. It only ever adds: pip is asked what it would do first, and a plan
that would replace an installed version is refused, printed in full and left for you to run.

`requirements/<group>.txt` is the file for a group that needs a package, named exactly like the
features key. A group that needs none has no file there, so `ls requirements/` is the whole
list. Today that is `document_export` alone.

---

## `features`

| Group | Extra install | Default | Gates |
|---|---|---|---|
| `features.blip` | none | **on** | `BLIP Analyze Image`, `BLIP Model Loader` |
| `features.clipseg` | none | **on** | `CLIPSeg Masking`, `CLIPSeg Batch Masking`, `CLIPSeg Model Loader` |
| `features.sam` | none | **on** | `SAM Image Mask`, `SAM Model Loader`, `SAM Parameters`, `SAM Parameters Combine` |
| `features.midas` | none | **on** | `MiDaS Depth Approximation`, `MiDaS Model Loader` |
| `features.diffusers` | none. The two nodes load through ComfyUI's own diffusers-format loader | **on** | `Diffusers Model Loader`, `Diffusers Hub Model Down-Loader` |
| `features.network` | none | off | `Download Image`, `Image Send HTTP`, `True Random.org Number Generator`, `Text Random Prompt` |
| `features.yunet` | none. The detector runs in torch on weights that ship with the pack | **on** | `YuNet Model Loader`, `Image Crop Face (YuNet)` |
| `features.document_export` | `python-docx`, `odfdo`, `xhtml2pdf`<br>`pip install -r requirements/document_export.txt` | off | No nodes. Lets `Save DOC` write `.docx`, `.odt` and `.pdf` |
| `features.pssr` | four packages, and a 22 GB checkout placed by hand<br>see [`docs/MODELS.md`](MODELS.md) | off | `Video Super Resolution (PS-SR)` |
| `features.preprocessors` | none. Every answer runs in torch on weights the pack publishes | **on** | `Power Preprocessor`, `Image Remove Background`, `Image Remove Background Model Loader`, `HDR Reconstruct` |
| `features.photoshop` | none. The PSD and TIFF layouts are written and read by the pack | **on** | `Layers Save`, `Layers Load` |
| `features.threejs` | none. three.js ships with the pack, and the browser fetches it only when a scene runs | **on** | The 43 `Three` nodes |
| `features.extras` | none | **on** | The 27 WAS_Extras nodes |
| `features.viewer` | none | **on** | `Content Viewer` and `CV Canvas Compose Batch`, the two nodes shared with the ComfyUI_Viewer pack |

### Model weights

| Group | Model folder | Repository used when nothing is on disk |
|---|---|---|
| `features.blip` | `models/blip` | `Salesforce/blip-image-captioning-base` (captions), `Salesforce/blip-vqa-base` (questions) |
| `features.clipseg` | `models/clipseg` | `CIDAS/clipseg-rd64-refined` |
| `features.sam` | `models/sams`, then `models/sam` | `facebook/sam-vit-huge` / `-large` / `-base`, by `model_size` |
| `features.midas` | `models/midas` | `Intel/dpt-large` (DPT_Large), `Intel/dpt-hybrid-midas` (DPT_Hybrid), `Intel/dpt-swinv2-tiny-256` (DPT_Small) |

---

## `legacy`

| Group | Default | Gates | Superseded by |
|---|---|---|---|
| `legacy.loaders` | **on** | `Checkpoint Loader (Advanced)`, `Checkpoint Loader (Simple, Advanced)`, `Lora Loader (Advanced)`, `unCLIP Checkpoint Loader (Advanced)`, `Upscale Model Loader (Advanced)` | ComfyUI's own `Load Checkpoint`, `Load LoRA`, `unCLIP Checkpoint Loader`, `Load Upscale Model` |
| `legacy.text_type` | **on** | `String to Text`, `Text to String` | Nothing: every text socket is a plain `STRING` already |
| `legacy.core_dupes` | **on** | `Image to Latent Mask`, `Convert Masks to Images`, `Seed (Number Outputs)` | Core `ImageToMask` and `ImageColorToMask`, core `MaskToImage`, ComfyUI's own `Seed` |
| `legacy.dupes` | off | `CLIPSEG2`, listed as **CLIPSeg Tiled Masking** | `CLIPSeg Masking` |
| `legacy.sampling` | **on** | `KSampler (Seed Socket)` | ComfyUI's `KSampler` with a `Seed` node |
| `legacy.switches` | **on** | `CLIP Input Switch`, `CLIP Vision Input Switch`, `Conditioning Input Switch`, `Control Net Model Input Switch`, `Image Input Switch`, `Latent Input Switch`, `Model Input Switch`, `Number Input Switch`, `Text Input Switch`, `Upscale Model Switch`, `VAE Input Switch` | `Tensor Switch` for images, masks and latents; `Model Switch` for models, VAEs and text encoders; `Any Input Switch` for everything else |
| `legacy.cache` | off | `Cache Node`, `Load Cache` | Nothing. They write `.latent`, `.image` and `.conditioning` files into `cache/` beside `config.yaml` and read them back |
| `legacy.debug` | off | `Export API`, `Samples Passthrough (Stat System)` | ComfyUI's **Workflow > Export (API)** menu item; ComfyUI's own system stats endpoint |
| `legacy.superseded` | off | `MiDaS Mask Image` | `Image Remove Background`, `CLIPSeg Masking`, `SAM Image Mask` |

---

## `nodes`

| Key | Type | Default |
|---|---|---|
| `nodes.enable` | list of node ids | `[]` |
| `nodes.disable` | list of node ids | `[]` |

`disable` beats `enable`, and both beat the group flags. Names are node ids, which are frozen,
so an override keeps working across a renamed menu label.
