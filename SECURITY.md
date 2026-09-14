# Security

## Reporting

Report a suspected vulnerability through
[GitHub Security Advisories](https://github.com/WASasquatch/was-node-suite-comfyui/security/advisories/new),
or open an issue where the problem is already public. A report naming the file, the line and an
input that reaches it is fixed fastest.

## What this pack does not do

| | |
|---|---|
| Start a process | No `subprocess`, `os.system`, `os.popen`, `os.exec*` or `os.spawn*` anywhere |
| Install a package | Nothing runs `pip`. A feature group that needs a library names it in the log with the command to run |
| Execute supplied code | No `eval`, no `exec`, no `compile` of runtime input |
| Load arbitrary pickles | Every `torch.load` passes `weights_only=True` |
| Require any package | The default node set installs nothing. `requirements.txt` is empty |

## Network

No node makes a request of its own. The nodes that did are retired, and the Noodle Soup
Prompts terminology that was once downloaded ships with the pack and seeds the database on
first start.

The only outbound traffic is `huggingface_hub` fetching model weights, and only when a node
needs a file that is not already on disk. `features.network` is `false` out of the box, and
while it is off a node that cannot find its weights names the file and the key rather than
reaching for the network.

There is no HTTP client in this pack: nothing here calls `requests`, `urllib` or a socket.

## Three.js scenes

**Custom Update**, **Custom Object**, **Custom Material**, **Custom Geometry** and **Script
Module** put javascript into the scene they build, and the browser runs it. A scene carrying
code is refused while `threejs.allow_scripts` is `false`, which is the default. The check reads
the whole descriptor at the point it is handed to the browser, so it covers a descriptor built
any way at all. See [docs/CONFIG.md](docs/CONFIG.md#threejs).

Frames for a render are drawn in the browser, so the node files a job and waits. A job is
offered only to the ComfyUI session its prompt was queued from, and each frame it takes back
is bounded by the frame count the job asked for and by 64 MB.

## Files

A path naming another machine, such as `\\server\share`, is refused before it is resolved:
resolving one reaches that host over the network. A node input naming a file or directory is
then resolved against an allowlist before the filesystem is touched. ComfyUI's `input/`, `output/` and `temp/`, this pack's own directory, and anything
in `paths.allow_read` or `paths.allow_write` are permitted; everything else is refused. Both
sides of the comparison are fully resolved first, which covers `..` segments and symlinks.
[docs/CONFIG.md](docs/CONFIG.md#containment) has the table.

The pack's HTTP routes take a key, never a path. A route serving a font looks the name up in a
catalog; one serving a rendered asset reads an in-memory store; one reading text resolves a
menu label through the file listing and then through the allowlist above. None joins a value
from a request onto a directory.

## The content viewer

The Content Viewer draws whatever a node holds, and a workflow carries that content, so the
frame it is drawn in never holds this page's origin.

| What is framed | Origin |
|---|---|
| Content built from a node, as `srcdoc` or a blob URL | None. `allow-same-origin` is stripped, whatever the view asked for |
| An app a view extension serves from its own endpoint | This origin, and only by the extension that was installed manually by the user outside this pack |

A script in framed content therefore cannot read or call anything on the page around it. The
frame talks to the viewer through `postMessage`, which needs no shared origin, and a document
inside it carries a policy limiting what it may load and connect to.

## Vendored code

Three directories under `web/` hold third-party browser libraries, kept as upstream publishes
them. `web/vendor/MANIFEST.json` records a SHA-256 for all 81 of those files, alongside the npm
package and version each directory was taken from. The digests are verified on every release,
and any file that is added, removed or edited without the manifest being rebuilt stops that
release.

To verify a copy independently, fetch the package named in `MANIFEST.json` from npm and compare
digests. `NOTICE.md` in each directory lists the upstream path of every file and marks the ones
whose import specifiers were rewritten.

| Directory | Package | Licence |
|---|---|---|
| `web/vendor/three` | `three@0.185.0` | MIT |
| `web/vendor/hugerte` | `hugerte@1.0.12` | MIT, with DOMPurify 3.4.11 under MPL-2.0 or Apache-2.0 |
| `web/vendor/pathtracer` | `three-gpu-pathtracer@0.0.24`, `three-mesh-bvh@0.9.14` | MIT |
| `web/viewer/views/code_scripts` | `prismjs` | MIT |
| `web/viewer/views/markdown_scripts` | `mermaid@10.9.5`, `katex@0.16.9` | MIT, with DOMPurify 3.2.4 under MPL-2.0 or Apache-2.0 |

[docs/THIRD_PARTY.md](docs/THIRD_PARTY.md) carries the full list, including the Python code
under `modules/vendor/`.

## Image parsers

The OpenEXR, PSD and TIFF readers are written here rather than taken from a library, so a
crafted file is a risk they have to answer for themselves. A size in a header is checked before
anything is allocated from it, and a header that describes more data than the file could hold
is refused.

| Ceiling | Value | What it bounds |
|---|---|---|
| Longest side | 30000 pixels | The EXR data window, the PSD canvas and each PSD layer, and the TIFF picture |
| Pixels | 268435456 | The same four, as an area, so two permitted sides cannot multiply into a refused one |
| Samples | 268435456 | Every PSD layer channel added together across the document |
| Samples | 1073741824 | A TIFF's pixels times its samples per pixel, which is the largest picture the area ceiling allows carrying four samples |
| Channels | 1024 | The EXR channel list |
| Samples per pixel | 1024 | One TIFF pixel |
| Bits per sample | 8, 16 or 32 | The TIFF picture |
| Expansion | 2048 bytes per byte held | What a file of a given size may describe |

A packed block is expanded with a byte ceiling rather than in full, so a deflate bomb yields no
more than the block it claims to be. A file is read whole into memory, so the ceilings are what
keeps a small file from asking for a large allocation.

The expansion figure sits above the 1032 bytes per byte deflate can actually produce, so it
refuses only a header no file of that size could be describing and never a real picture. The
area and sample ceilings are what bound the largest allocation a permitted file can ask for.

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
| Socket `connect` | `web/was_app_workflow.js` | `graph.getNodeById(id).connect(...)`, a LiteGraph wire between two nodes |
| Socket `bind` | 7 files under `web/` | `Function.prototype.bind` |
| Socket `connect`, `bind` | `web/vendor/three`, `web/vendor/hugerte` | The same two JavaScript methods, in third-party code for Three.js and HugerTE (Path Tracer) |
| Socket `bind` | `web/viewer/views/code_scripts/prism.min.txt`, `web/viewer/views/markdown_scripts/mermaid.min.txt` | The same JavaScript method, in Prism and Mermaid. They are syntax highlighting and diagram drawing, read as text and inlined into the sandboxed frame a view builds, which has no access to this page |
| Network operation, database connection | `modules/state/store.py` | `sqlite3.connect`, opening a local database file. Reported twice, under both rule names |
| Dynamic import | `__init__.py` | Walking this pack's own `nodes/` package and importing each valid module it finds |
| Dynamic import | `modules/deps.py` | Resolving an optional dependency by name, so a missing one reports itself instead of raising on import |
| Dynamic import | `modules/viewer/parsers`, `modules/viewer/extension_nodes` | Loading view extensions a user has **manually installed**; does not act on its own |
| Environment access | `modules/config/paths.py`, `prestartup_script.py` | Reading eight variables and writing none: `WAS_CONFIG` and `WAS_CONFIG_DIR` relocate the config file, `LOCALAPPDATA` and `XDG_DATA_HOME` locate user data, `HF_HOME`, `HF_HUB_CACHE` and `HUGGINGFACE_HUB_CACHE` find weights already on disk, and `PSSR_ROOT` finds one model. Every read in `modules/` goes through one function, `paths.env_value` |
| Sensitive file access | `modules/image/exr.py`, `modules/image/psd.py` | Reading the image file chosen in the node. The value is a menu label, not a path, and it is resolved through the allowlist before the file is opened. See the section below for what bounds the parse |

JavaScript files are reported under Python rule names by some scanners without proper filtering enabled. `connect` and `bind`, etc,
are ordinary JavaScript methods and carry no network meaning.
