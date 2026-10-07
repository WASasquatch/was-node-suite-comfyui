# Security

## Reporting

Report a suspected vulnerability through
[GitHub Security Advisories](https://github.com/WASasquatch/was-node-suite-comfyui/security/advisories/new),
or open an issue where the problem is already public. A report naming the file, the line and an
input that reaches it is fixed fastest.

## What this pack does not do

| | |
|---|---|
| Start a process | No `subprocess`, `multiprocessing` or `pty`, and no `os.system`, `os.popen`, `os.exec*`, `os.spawn*`, `os.posix_spawn`, `os.startfile`, `os.fork` or `asyncio.create_subprocess_*`, in any import form |
| Install a package | Nothing runs `pip`. A feature group that needs a library names it in the log with the command to run |
| Execute supplied code on the server | No `eval`, no `exec`, no `compile` of runtime input. Supplied JavaScript runs in the browser in two places, both below: Three.js scenes while `threejs.allow_scripts` is on, and the Content Viewer's sandboxed frame |
| Load arbitrary pickles | Every `torch.load` passes `weights_only=True`, and every `comfy.utils.load_torch_file` passes `safe_load=True` |
| Require any package | The default node set installs nothing and `requirements.txt` is empty. `requirements-optional.txt` points at `requirements/document_export.txt`, which names three packages for the `document_export` group. Nothing installs them: the group prints the command for a person to run |

## Network

`features.network` is `false` out of the box, and while it is off nothing in this pack makes a
request beyond ComfyUI's own server. A node that cannot find its weights names the file and the
key instead.

With it on, two requests are possible. `huggingface_hub` fetches model weights a node needs and
does not find on disk, from Hugging Face only. Three Texture URL lets the browser load the
`http` or `https` address the node was given. Every other address the browser is handed is the
pack's own asset route, a `data:` URL or a `blob:` URL, and the Three.js runtime turns any other
address a scene or a model names into an empty file.

No Python code here is an HTTP client: nothing imports `requests`, `httpx`, `urllib.request`,
`http.client`, `urllib3` or `websockets`, and nothing opens a socket. Three related names do
appear, and none of them reaches the network. `aiohttp.web` registers this pack's own inbound
routes in twelve modules, and its client half is never imported. `urllib.parse` reads strings in
two files: it decodes a `data:` URI in one and unescapes the file names a 3D model refers to in
the other. `socket.gethostname()` fills the `[hostname]` token.

The PDF export hands its document to xhtml2pdf with every picture already inside it as a `data:`
URL. Stylesheet imports, `url()` references and `<link>` tags are removed first, every file
request is answered with an empty inline file, and xhtml2pdf's resource policy refuses network
and local reads wherever the installed version has one.

## Three.js scenes

**Custom Update**, **Custom Object**, **Custom Material**, **Custom Geometry** and **Script
Module** put javascript into the scene they build, and the browser runs it. A scene carrying
code is refused while `threejs.allow_scripts` is `false`, which is the default, at every point a
descriptor leaves the server: handed to the browser, rendered, or written into a page by **Three
Compile**. The check reads the whole descriptor, so it covers a descriptor built any way at all.
See [docs/CONFIG.md](docs/CONFIG.md#threejs).

Frames for a render are drawn in the browser, so the node files a job and waits. A job is filed
only for a prompt queued from a ComfyUI session and is offered only to that session. A frame is
refused before it is read when the request declares no size or a size over the bound, and each
frame it takes back is bounded by the frame count the job asked for and by 64 MB.

## Files

A path naming another machine is refused before anything resolves it: a UNC path such as
`\\server\share`, the same with forward slashes, the `\\?\UNC\` and `\\.\UNC\` forms, and the NT
prefix `\??\`, checked after `~` is expanded. A link along a path is read without being followed,
and one pointing at another machine is refused the same way. ComfyUI's own path helpers resolve
what they are given, so the pack calls them only through wrappers that refuse first.

A glob pattern, a file name prefix, and a name or folder typed below a folder the node has
already chosen are refused before anything is listed or resolved when they name another
machine, carry a drive, start at a root, or hold a segment of dots and spaces alone such as
`..`. Whatever they match stays inside that folder.

A node input naming a file or directory is resolved and checked against an allowlist before it
is opened, listed or written, whatever value arrives: a menu offers labels, and a linked input
or an expanded App Workflow can still deliver any string. ComfyUI's `input/`, `output/` and
`temp/`, the pack's state directory `ComfyUI/user/was-node-suite`, and anything in
`paths.allow_read` or `paths.allow_write` are permitted; everything else is refused. Both sides
of the comparison are resolved, which covers `..` segments and symlinks. Model files are read
from ComfyUI's model folders, saved workflows from its workflows folder and fonts from the font
catalog, each by a name that cannot leave its folder. A glob pattern holding `**` lists through
links the user placed inside a permitted folder; every match it returns is checked again before
it is read.

No node writes the pack's settings. `config.yaml`, `config.json`, the config file `WAS_CONFIG`
names, and the `viewer-extensions/` folder are refused as write targets, even inside the state
directory. [docs/CONFIG.md](docs/CONFIG.md#containment) has the table.

## HTTP routes

Every route the pack registers takes a key. None takes a path.

| Route | Takes | Answers from |
|---|---|---|
| `GET /was/interface/api/preview` | node id, slot, side, frame | an in-memory store |
| `POST /was/interface/api/preview/subscribe` | node id | an in-memory store |
| `POST /was/interface/api/preview/discard` | mask editor upload names | deletes those files: names matching `clipspace-…-<digits>.png`, directly in `input/`, at most eight a call |
| `GET /was/interface/api/run_result`, `/run_result_page` | node id | an in-memory store |
| `GET /was/interface/api/text_lines` | menu label | the file listing, then the allowlist |
| `GET /was/interface/api/font` | font name | the font catalog |
| `GET /was/interface/api/file_listing` | suffixes, a count | names, sizes and times of files in the permitted folders, 5000 at most |
| `GET /was/interface/api/file_thumbnail` | menu label, a size | the file listing, then the allowlist; a picture of the file or a video's first frame, 512 pixels a side at most |
| `GET /was/interface/api/segment_preview`, `/segment_preview/index` | node id, segment | an in-memory store of preview frames, 384 MB at most |
| `GET /was/interface/api/nsp_pantry` | search text | the state database |
| `GET /was/interface/api/video_probe` | menu label | another machine refused, then the allowlist |
| `GET`, `POST /was/interface/api/pause` | node id, session id, value | an in-memory hold; a held run is released only by the session that queued it |
| `GET /was/interface/api/app_exposure` | saved workflow name | the workflows folder, a name leaving it refused before it is read |
| `GET /was/threejs/api/asset` | key | an in-memory store |
| `GET`, `POST /was/threejs/api/render` | session id, token, frames | an in-memory job queue |
| `GET /was/threejs/api/module` | menu label | the file listing, then the allowlist; refused while `threejs.allow_scripts` is off |

## The content viewer

The Content Viewer draws whatever a node holds, and a workflow carries that content, so the
frame it is drawn in never holds this page's origin.

| What is framed | Origin |
|---|---|
| Content built from a node, as `srcdoc` or a blob URL | None. `allow-same-origin` is stripped, whatever the view asked for |
| An app a view extension serves from its own endpoint | This origin, and only for an extension the user installed: by hand, or from `viewer-extensions/` with `viewer.install_extensions` on, which is off by default |

A frame reaches the viewer only through `postMessage`. The viewer acts on a message only when it
comes from a frame the viewer created, and only for that frame's own node, so framed content
cannot read another node's pictures, change its settings or release its held run. A document
inside a frame carries a policy limiting what it may load and connect to.

The rich text editor shows a document in a frame of this page's origin, under a content policy
that allows no scripts, plugins, frames, forms or connections, and loads images, fonts, media
and stylesheets only from this origin or from `data:` and `blob:` URLs.

## Vendored code

Six directories under `web/` hold third-party browser libraries, kept as upstream publishes
them. `web/vendor/MANIFEST.json` records a SHA-256 for all 92 of those files, alongside the npm
package and version each directory was taken from. Every digest recomputes from the file beside
it, and the manifest names no file the tree does not hold.

To verify a copy independently, fetch the package named in `MANIFEST.json` from npm and compare
digests. `NOTICE.md` in each directory names the upstream file behind every file it holds and
what, if anything, was changed in it.

| Directory | Package | Licence |
|---|---|---|
| `web/vendor/three` | `three@0.185.0` | MIT |
| `web/vendor/hugerte` | `hugerte@1.0.12` | MIT, with DOMPurify 3.4.11 under MPL-2.0 or Apache-2.0 |
| `web/vendor/pathtracer` | `three-gpu-pathtracer@0.0.24`, `three-mesh-bvh@0.9.14` | MIT |
| `web/viewer/views/code_scripts` | `prismjs@1.29.0` | MIT |
| `web/viewer/views/markdown_scripts` | `mermaid@10.9.5`, `katex@0.16.9` | MIT, with DOMPurify 3.2.4 under MPL-2.0 or Apache-2.0 |
| `web/viewer/fonts` | `katex@0.16.9` | SIL Open Font License 1.1 |

[docs/THIRD_PARTY.md](docs/THIRD_PARTY.md) carries the full list, including the Python code
under `modules/vendor/`.

## Image parsers

EXR Load and Layers Load read OpenEXR, PSD and TIFF with readers written here rather than taken
from a library. Image Load and the batch, sequence, grid and morph loaders read TIFF and the
other common formats with Pillow, and Image Load reads a PNG's header chunks itself. A size in a
header is checked before anything is allocated from it, a header that describes more data than
the file could hold is refused, and data shorter than its header promises is refused rather than
padded.

| Ceiling | Value | What it bounds |
|---|---|---|
| Longest side | 30000 pixels | The EXR data window, the PSD canvas and each PSD layer, and the TIFF picture |
| Pixels | 268435456 | The same four, as an area, so two permitted sides cannot multiply into a refused one |
| Samples | 1073741824 | An EXR's pixels times its channels, and a TIFF's pixels times its samples per pixel |
| Samples | 268435456 | The floats every PSD layer is decoded into, four a pixel, across the document, counted before any layer is decoded |
| Channels | 1024 | The EXR channel list |
| Channels | 56 | One PSD document or layer |
| Samples per pixel | 1024 | One TIFF pixel |
| Bits per sample | 8, 16 or 32 | The TIFF picture |
| Expansion | 2048 bytes per byte held | What a file of a given size may describe, in all three readers |
| Directory bytes | 2 times the file | Every value one TIFF directory copies, together |
| Chunk | 2147483647 bytes, and no more than the rest of the file | One PNG chunk read for its header |

A packed block is expanded with a byte ceiling, never past the size its header gives, so a
deflate bomb yields no more than the block it claims to be. A file is read whole into memory, so
the ceilings are what keep a small file from asking for a large allocation. The expansion figure
sits above the 1032 bytes per byte deflate can produce, so it refuses no real picture.

## Bundled data

The Noodle Soup Prompts terminology ships as a snapshot, `modules/data/nsp_pantry.pack`:
zlib-compressed JSON holding 82 terms and 17,518 entries. A fresh start reads it into the
state database and makes no request. The snapshot is refreshed and re-bundled with a release,
never at runtime.

## Static analysis

The following are false or outdated positives by Comfy Registry's pattern-matching scanners. Each is listed with what the code is doing, so
a reviewer can confirm it without reading the whole tree.

| Reported as | Where | What it is |
|---|---|---|
| Socket `connect` | `web/was_app_workflow.js` | LiteGraph's own method for wiring one node's output slot to another node's input, called on a node the graph looked up by id |
| Socket `bind` | 7 of this pack's own files under `web/`, and `web/interface/mask_paint.js` | `Function.prototype.bind`, the standard way to fix a callback's `this`, in those 7, which are every one of the pack's own browser files that calls it. `mask_paint.js` defines a method of its own named `bind`. The vendored libraries call it throughout |
| Socket `connect`, `bind` | `web/vendor/three`, `web/vendor/hugerte` | The same two JavaScript methods, in third-party code for Three.js and HugeRTE |
| Socket `bind` | `web/viewer/views/code_scripts/prism.min.txt`, `web/viewer/views/markdown_scripts/mermaid.min.txt` | The same JavaScript method, in Prism and Mermaid. They are syntax highlighting and diagram drawing, read as text and inlined into the sandboxed frame a view builds, which has no access to this page |
| Network operation, database connection | `modules/state/store.py` | `sqlite3.connect`, opening a local database file. Reported twice, under both rule names |
| Dynamic import | `__init__.py` | Walking this pack's own `nodes/` package and importing each valid module it finds |
| Dynamic import | `modules/deps.py` | Resolving an optional dependency by name, so a missing one reports itself instead of raising on import |
| Dynamic import | `prestartup_script.py` | Loading this pack's own `modules/viewer/install.py` before ComfyUI starts |
| Dynamic import | `modules/viewer/parsers`, `modules/viewer/extension_nodes` | Loading view extensions from the user's `viewer-extensions/` folder, which holds only what the user put there or installed with `viewer.install_extensions` on |
| Environment access | `modules/config/paths.py`, `prestartup_script.py` | Reading eight variables and writing none: `WAS_CONFIG` and `WAS_CONFIG_DIR` relocate the config file, `LOCALAPPDATA` and `XDG_DATA_HOME` locate user data, `HF_HOME`, `HF_HUB_CACHE` and `HUGGINGFACE_HUB_CACHE` find weights already on disk, and `PSSR_ROOT` finds one model. Every read in `modules/` goes through one function, `paths.env_value`. Python's `Path.home()` and `expanduser()` read the user's home directory |
| Sensitive file access | `modules/image/exr.py`, `modules/image/psd.py` | Reading the image file chosen in the node, resolved through the allowlist above whatever value arrives. See the section above for what bounds the parse |

JavaScript files are reported under Python rule names by some scanners without proper filtering enabled. `connect` and `bind`, etc,
are ordinary JavaScript methods and carry no network meaning.

This page is itself matched. Naming an API to explain it puts that name in the file, and
version 3.2.2 was reported for a socket pattern at this document's own table row above. The
rows describe code held elsewhere in the tree; nothing here executes.
