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
| Execute supplied code on the server | No `eval`, no `exec`, no `compile` of runtime input. Supplied JavaScript runs in the browser in three places, all below: Three.js scenes while `threejs.allow_scripts` is on, the page Three Compile writes under the same setting, and the Content Viewer's sandboxed frame while `viewer.allow_scripts` is on, which it is not by default |
| Load arbitrary pickles | Every `torch.load` passes `weights_only=True`, and Load Cache, which reads a file a person picks, refuses torch older than 2.6. Every `comfy.utils.load_torch_file` passes `safe_load=True`. Diffusers Hub Model Down-Loader fetches and loads `.safetensors` weights only, and BLIP and CLIPSeg load `model.safetensors` only. The checkpoint and Diffusers model loaders hand a model-folder file to ComfyUI's own loading code, as ComfyUI's loader nodes do |
| Require any package | The default node set installs nothing, and `requirements.txt` names no package. `requirements-optional.txt` points at `requirements/document_export.txt`, which names three packages for the `document_export` group, and at `requirements/llm.txt`, which names the packages of the `llm` group. Nothing installs them: each group prints the command for a person to run |

## Number inputs

Every number input is held to the range its node declares, whether it is typed, linked from
another node or posted to the API. A value outside that range, or a float that is not finite,
stops the run before the node does any work.

Image Resize, Image Padding, Image Transpose, Image Perspective, the layer nodes and Bounded
Image Blend refuse an output past 30000 pixels a side or 268435456 pixels before allocating it,
and name the input to lower. An empty Kandinsky 6 latent and a MiniMax H3 segment are refused
past 2147483648 values, Number Range and Number Sequence make at most 10000 values, Power LoRA
Loader takes at most 10000 values in one strength or weight list, and every LUT a node builds
is at most 256 a side. Bounding Box Editor reads at most 262144 characters of layout and 64
boxes, and blocks its masks output past 134217728 values.

## Network

`features.network` is `false` out of the box, and while it is off nothing in this pack makes a
request to another machine. Beyond ComfyUI's own server, it then makes requests only to loopback
addresses: a language model server LLM Provider or LLM Embedding Model names (Ollama, LM Studio,
vLLM, a llama.cpp server, or an `llm.endpoints` entry whose address is loopback), and an MCP or
vector store server at a loopback address that config.yaml names. An MCP server named by a
`command` is a program started on this machine, and what that program reaches is its own. A node
that cannot find its weights names the file and the key instead.

With it on, `huggingface_hub` and transformers fetch files a node needs and does not find on
disk, from Hugging Face only, and never send a stored Hugging Face token. Most nodes fetch fixed
files from fixed repositories. BLIP Model Loader, CLIPSeg Model Loader and Diffusers Hub Model
Down-Loader fetch the repository the workflow names into the model folder: its configuration and
its `.safetensors` weights. LLM Embedding Model, LLM Rerank and LLM Transformers Loader fetch
only a repository on their own fixed list, its configuration, tokenizer and `.safetensors`
weights, into `models/LLM`, and load no model carrying its own code. Three Texture URL lets the browser load the `http` or `https`
address the node was given. The browser learns whether `features.network` is on from what the
server sends with a scene, never from the scene itself. Every other address the browser is
handed is the pack's own asset route, a `data:` URL, a `blob:` URL or one of the ten fixed
Marigold v2 downloads below, and the Three.js runtime turns any other address a scene or a model
names into an empty file. Content shown in the Content Viewer reaches no other address while
`viewer.allow_scripts` is off, whatever `features.network` is set to: it runs no script of its
own and cannot move its frame.

Choosing `Marigold v2` on Power Preprocessor names any of its ten files the model folders lack
in ComfyUI's missing models list, each with its fixed address under
`https://huggingface.co/Comfy-Org/marigold-v2-0/resolve/main/`. ComfyUI's frontend asks Hugging
Face for each listed file's size, whatever `features.network` is set to, and downloads one only
when its Download button is pressed, as it does for any workflow that lists its models. No other
outside address appears in the pack's own browser code.

Two modules are HTTP clients, both loaded only with `features.llm` on and both refusing every
request while `features.network` is off. `modules/llm/documents.py` fetches a web page for LLM
Load Web Page and for the web page tool an agent calls, through `http.client` and
`socket.create_connection`: `http` and `https` only, refusing a host any of whose addresses is
loopback, private, link-local, multicast, reserved or unspecified, checking again after each of
at most five redirects, connecting only to an address it checked, and reading at most 20 MB.
`modules/llm/tools.py` calls three fixed services for the agent tools through `httpx`,
following a redirect only to the same host over `https`, at most three times:
Wikipedia's search and page summary API, Tavily's search API with the key in
`TAVILY_API_KEY`, and Brave Search's API with the key in `BRAVE_API_KEY`. The hosted language
and embedding models reach their providers through those providers' own libraries, as
[Language models](#language-models) lists.

No other Python code here is an HTTP client: nothing else imports `requests`, `httpx`,
`urllib.request`, `http.client`, `urllib3` or `websockets`, and nothing else opens a socket.
Three related names do appear, and none of them reaches the network. `aiohttp.web` registers
this pack's own inbound routes in seventeen modules, and its client half is never used.
`urllib.parse` reads strings in ten files: it decodes a `data:` URI and a picture's name in
the document exporter, unescapes the file names a 3D model refers to, splits the address Three
Texture URL is given, and splits the addresses the LLM nodes fetch or call and the servers
`llm.endpoints`, `llm.mcp_servers` and `llm.vector_stores` name.
`socket.gethostname()` fills the `[hostname]` token. Tokens are not expanded in the address
Three Texture URL or LLM Load Web Page is given. Text wired into either from another node arrives
with its tokens already expanded, so with `features.network` on, a workflow can put this
computer's name or login into an address it fetches.

The PDF export hands its document to xhtml2pdf with every picture already inside it as a `data:`
URL. Stylesheet imports, `url()` references and `<link>` tags are removed first, every file
request is answered with an empty inline file, and xhtml2pdf's resource policy refuses network
and local reads wherever the installed version has one.

## Language models

Every LLM node loads only while `features.llm` is on, and it is off out of the box.

| | |
|---|---|
| Hosted models | LLM OpenAI, LLM Anthropic, LLM Google Gemini, LLM Provider's hosted presets and an `llm.endpoints` entry at an address other than loopback refuse to run while `features.network` is off. Each preset sends to one fixed address with the key from one fixed environment variable; a workflow supplies neither |
| Keys | Read from the environment ComfyUI starts in, never from a widget. No key is logged, put in a run result or named in an error: error text passes through a filter that replaces anything shaped like a key and the exact value of every key, token and password the pack has read. A keyless OpenAI-compatible server, and every server LLM Provider reaches other than OpenAI, is sent the `OpenAI-Organization`, `OpenAI-Project` and `OPENAI_CUSTOM_HEADERS` headers empty, and Ollama is sent `Bearer not-needed` in place of `OLLAMA_API_KEY` when its entry names no key. A keyless OpenAI-compatible server is sent the placeholder `not-needed`, never `OPENAI_API_KEY` |
| A model nobody wired | Inside every LLM node that runs a model, llama-index's global default models are replaced by ones that raise, so no step falls back to a hosted model the graph did not connect |
| Logs | llama-index's loggers, and those of httpx, httpcore and the MCP client, are held at WARNING, so the prompts, documents, answers, search queries and request addresses they would log at INFO stay out of the log |
| Files an agent reads | The document tool reads only the one file picked on its menu, through the file listing and the sandbox. The calculator evaluates arithmetic through the pack's expression parser, never `eval` |
| Saved indexes and conversations | Written only under `ComfyUI/user/was-node-suite/llm/`, by a name of letters, digits, spaces, dots, dashes and underscores up to 64 characters, each path resolved through the sandbox. Saving an index replaces only the files the pack wrote in a folder that carries its marker |
| MCP servers | LLM MCP Tools runs or reaches only a server `llm.mcp_servers` in config.yaml names, picked by that name; a workflow carries no command, argument, address or token. A command is started with a minimal environment plus the variables its entry sets or names in `env_from`. An address other than loopback refuses while `features.network` is off. Errors and panels name a server by its name and host, never the path, query or credentials of its address. A server's tools then act with whatever access that program has |
| Vector stores | A local store is the folder `ComfyUI/user/was-node-suite/llm/vector_stores`, written by Qdrant, or by Chroma, LanceDB or FAISS while `features.vector_stores` is on. Chroma is opened with its telemetry off. A server is reached only when `llm.vector_stores` in config.yaml names it, and one other than loopback refuses while `features.network` is off. Building into a collection first removes the chunks of the documents being added again, and nothing else |
| Tables and documents | LLM Load Table reads a JSON or JSON Lines table of at most 64 MB, at most 512 columns of a sheet and at most `max_rows` rows of it, and copies a SQLite file's tables through a read-only connection; every query an answer writes passes a read-only authorizer and stops at `timeout_seconds`. A PDF's decoded pages are released as each is read. A json split reads a document of at most 16 million characters and refuses one whose key paths would write more than 64 million |
| Agents | An agent calls only the tools wired to it. Text it reads can steer which tool it calls and with what, so with `features.network` on, whatever an agent can read can reach an address its web tools fetch |
| Picture documents | A picture document stores a JPEG copy at most 384 pixels a side, its folder's menu label and its path below that folder, never an absolute path. LLM Sources to Images reads the file back only through the folder menu and the read allowlist, and a picture is embedded only from its stored copy |

## Text tokens

`[hostname]` and `[user]` expand to this computer's name and the login running ComfyUI, kept to
letters, digits, `.`, `_` and `-`, without leading dots, and to 64 characters. Built-in and
custom tokens expand in every text input except the ones the token module lists as literal:
token definitions, repository ids, colours, state a node's own frontend writes, listing keys,
stored system prompts, the addresses Three Texture URL and LLM Load Web Page are given, and the
model, tool, agent, conversation, table and collection names the LLM nodes look up.

A custom token's value is placed into any text input that names it, so anyone who can queue a
prompt can read it, and a token is no place for a secret. Text Add Tokens and Text Add Token by
Input log the names of the custom tokens defined, never their values.

## Three.js scenes

**Custom Update**, **Custom Object**, **Custom Material**, **Custom Geometry** and **Script
Module** put javascript into the scene they build, and the browser runs it. A scene carrying
code is refused while `threejs.allow_scripts` is `false`, which is the default, at every point a
descriptor leaves the server: handed to the browser, rendered, or written into a page by **Three
Compile**. A scene naming an address outside the server, or marked to fetch from the network, is
refused at the same points while `features.network` is off. The check reads the whole
descriptor, so it covers a descriptor built any way at all. See
[docs/CONFIG.md](docs/CONFIG.md#threejs).

Frames for a render are drawn in the browser, so the node files a job and waits. A job is filed
only for a prompt queued with a client id and is offered only to a request carrying that id. A
frame is refused when its token, read from the first 512 bytes of the request, names no waiting
job, and before it is read when the request declares no size or a size over the bound. Each
frame a job takes back is bounded by the frame count it asked for and by 64 MB.

## Files

A path naming another machine is refused before anything resolves it: a UNC path such as
`\\server\share`, the same with forward slashes, the `\\?\UNC\` and `\\.\UNC\` forms, and the NT
prefix `\??\`, checked after `~` is expanded. A link along a path is read without being followed,
from the root down, and one pointing at another machine is refused the same way. ComfyUI's
`get_annotated_filepath` and `get_save_image_path` resolve what they are given, so the pack calls
them only through wrappers that refuse first; a subfolder in a file name prefix is checked before
ComfyUI creates it. Model files are found with ComfyUI's `get_full_path`, which keeps a name
inside its model folder.

The pack teaches ComfyUI's Load Model Patch node one more kind of file: a safetensors file whose
header, read up to 64 MB and never past the file's size, names a Parametric Trajectory
Distillation release. It loads with `safe_load`, and every other file goes to ComfyUI's own
loader unchanged.

A glob pattern, a file name prefix, a frame cache name, and a name or folder typed below a
folder the node has already chosen are refused before anything is listed or resolved when they
name another machine, carry a drive, start at a root, or hold a segment of dots and spaces alone
such as `..`. On Windows they are refused as well when a segment holds `:`, which names a stream
of another file, or names a device such as `NUL` or `COM1`, and so is any path a node reads or
writes. A frame cache name is refused when it holds a folder separator. Whatever they match stays
inside that folder.

A refusal repeats the value as it was given, cut to 240 characters, and names the permitted
folders as `input`, `output`, `temp`, `ComfyUI/user/was-node-suite` and the number of folders
`paths.allow_read` or `paths.allow_write` lists. It never names a resolved path, a link's target
or a folder's full path, and its text is the same whether anything is at the refused path or not.
Path Exists answers a refused path as not existing, with `resolved` empty.

A node input naming a file or directory is resolved and checked against an allowlist before it
is opened, listed or written, whatever value arrives: a menu offers labels, and a linked input
or an expanded App Workflow can still deliver any string. ComfyUI's `input/`, `output/` and
`temp/`, the pack's state directory `ComfyUI/user/was-node-suite`, and anything in
`paths.allow_read` or `paths.allow_write` are permitted; everything else is refused. Both sides
of the comparison are resolved, which covers `..` segments and symlinks. Model files are read
from ComfyUI's model folders, saved workflows from a user's workflows folder (the App Workflow
node and its menu read the default user's; the route reads the asking user's) and fonts from the
font catalog, each by a name that cannot leave its folder. A glob pattern holding `**` lists through
links the user placed inside a permitted folder; every match it returns is checked again before
it is read. The file listing and the pack's folder and file menus follow no symlink or Windows
junction out of the folder they list. A folder in `paths.allow_read` is listed, previewed and read by the routes below for
any client that can reach ComfyUI, the same as `input/`.

A video, sound or picture the pack opens with FFmpeg, through PyAV, opens only as a single-file
container, stream or picture from local disk. A playlist, a concatenation list and any network
protocol are refused.

No node reads or writes the pack's settings. In the state directory, every file whose name
begins `config.yaml`, `config.json`, `was_state.db`, `was_suite_config.json`,
`was_suite_settings.json` or `was_history.json` is refused as a read and a write target. So are
the config file in use and the one `WAS_CONFIG` names, with every file beside them whose name
begins with theirs, and the `viewer-extensions/` folder as a write target. Names are compared
after links are resolved, without regard to case and to trailing dots and spaces. Custom tokens
are the exception: Text Add Tokens and Text Add Token by Input store them in the state database.
[docs/CONFIG.md](docs/CONFIG.md#containment) has the table.

Power LoRA Merger records each source in the merged file's header by its name inside the LoRA
folder, and leaves out any value from the first source's header that names a path on disk.
In `obfuscate` mode the header keeps nothing from the sources but what model and network the
file is for.

A frame cache picked from a folder no node writes to, such as `input/` or a `paths.allow_read`
folder, is read and never added to or deleted.

At startup the pack writes into its state directory, into the config file in use with a `.bak`
beside it when it brings that file up to date, and into
`web/viewer/views/extension_views.json` when the installed view extensions change. With
`viewer.install_extensions` on, it also writes the files of the extensions it installs, never
over a file the pack ships.

### Deleting files

Files are deleted, moved or renamed over in these modules and no others. No folder is removed
with its contents except a checkpoint's download folder, which the fetch made inside the model
folder, and the temporary folder the ODT export makes for itself.

| Module | What it removes |
|---|---|
| `modules/media/frame_cache.py` | A frame cache's own `cache.wasframes`, `frames.bin` and `audio.npy` once a save has written it, then its folder if that is left empty. Only a numbered cache folder such as `clip_00001` inside a folder nodes write to, never a permitted root itself. The manifest is replaced through a `.part` file |
| `modules/media/video.py` | The video Write to Video adds a frame to, replaced by the copy holding that frame |
| `modules/media/temporal_upscale.py` | The files in `temp/` a failed Video Temporal Upscale run was writing |
| `modules/archive/save.py` | The `.part` file Zip Save writes, renamed over the archive once complete or removed when the write fails, and a leftover `.part` of the same name |
| `modules/latent/h3_clip_file.py` | The `.part` file H3 Save Clip writes, renamed over the clip once complete |
| `modules/interface/preview.py` | Mask editor uploads in `input/clipspace/` written in the last ten minutes, as the route table below states |
| `modules/interface/compare_files.py` | Files it kept for the compare panels in `ComfyUI/user/was-node-suite/compare`, named by 32 hex digits and `.mp4`, `.png` or `.wav`, least recently used first once the folder holds more than 2 GB or 2000 of them, and its own `.part` files there older than an hour. A file is kept through a `.part` file renamed over its name |
| `modules/viewer/parsers/canvas_parser.py` | The `.part` file the Content Viewer writes a picture into, renamed to `input/was_viewer/<md5>.png` once complete |
| `modules/model/h3_latent_upscaler.py`, `modules/model/flow_networks.py` | The download folder a checkpoint lands in, once the checkpoint is moved out of it |
| `modules/llm/storage.py` | In `ComfyUI/user/was-node-suite/llm/indexes/<name>/`, the llama-index files and marker it wrote there, before a save under the same name, and only in a folder carrying its marker. In `ComfyUI/user/was-node-suite/llm/sessions/`, a session's `<name>.json` and the `<name>.<block>.vectors.json` files it lists, while LLM Clear Memory's `confirm` or LLM Chat Session's `reset` is on, or when the chat panel deletes that conversation. A rename moves the same files to the new name and refuses a name already saved. Each file is written through a `.part` file renamed over its name |

Some nodes replace a file of the same name: Zip Extract, Save LUT and the save nodes'
overwrite settings replace files inside the folder they write to, and Power LoRA Merger replaces
only a LoRA it wrote itself.

## HTTP routes

Each route takes a key, a menu label looked up in the file listing, or a name inside one fixed
folder that is refused if it would leave it. None takes a path to open.

| Route | Takes | Answers from |
|---|---|---|
| `GET /was/interface/api/preview` | node id, slot, side, frame, workflow, channel, client id | an in-memory store, served only to the client id that queued the run, or to any client for a run queued without one |
| `POST /was/interface/api/preview/subscribe` | node id | an in-memory store |
| `POST /was/interface/api/preview/discard` | a list of mask editor upload names | deletes those files: names matching `clipspace-…-<digits>.png`, directly in `input/clipspace/`, written in the last ten minutes, at most eight a call |
| `GET /was/interface/api/run_result`, `/run_result_page` | node id, workflow, client id | an in-memory store, served only to the client id that queued the run, or to any client for a run queued without one |
| `GET /was/interface/api/text_lines` | menu label | the file listing, then the allowlist |
| `GET /was/interface/api/font` | font name | the font catalog |
| `GET /was/interface/api/file_listing` | suffixes, a count | names, sizes and times of files in the permitted folders, 5000 at most. Symlinks and Windows junctions are not followed |
| `GET /was/interface/api/file_thumbnail` | menu label, a size | the file listing, then the allowlist; a picture of the file or a video's first frame, 512 pixels a side at most. A source over 64 megapixels is refused before it is decoded |
| `GET /was/interface/api/segment_preview`, `/segment_preview/index` | node id, segment, client id | an in-memory store of preview frames, 384 MB at most, served to the client that queued the run, or to any client for a run queued without one |
| `GET /was/interface/api/nsp_pantry` | search text | the state database |
| `GET /was/interface/api/system_prompts` | library key | the `system_prompts` folder in the state directory; a key its own listing does not hold reads nothing. 2000 entries and 256 KB of text at most, read off the server's event loop |
| `GET`, `POST /was/interface/api/llm_sessions` | conversation name, a new name | the `llm/sessions` folder in the state directory, registered only while `features.llm` is on. GET lists 500 conversations at most, or reads one by name; POST renames or deletes one by name. A name is 1 to 64 letters, digits, spaces, dots, dashes and underscores and is refused otherwise. Read and changed off the server's event loop |
| `GET /was/interface/api/compare_file` | a kept file's name | the `compare` folder in the state directory; a name other than 32 hex digits and `.mp4`, `.png` or `.wav`, or a link, reads nothing |
| `GET /was/interface/api/video_probe` | menu label | the file listing of video files, then the allowlist; read off the server's event loop |
| `GET`, `POST /was/interface/api/pause` | node id, client id, value | an in-memory hold. GET lists only the holds queued under the asking client id, and POST releases a hold only for that id |
| `GET /was/interface/api/app_exposure` | saved workflow name | the asking user's workflows folder, `ComfyUI/user/<user>/workflows`, where `--multi-user` takes the user from the `Comfy-User` header and an unknown user reads nothing. A name leaving it is refused before it is read. A file over 16 MB, a subgraph holding itself, nesting past 32 levels or more than 20000 nodes is refused with fixed text, and the work runs off the server's event loop |
| `GET /was/threejs/api/asset` | key | an in-memory store |
| `GET`, `POST /was/threejs/api/render` | client id, token, frames | an in-memory job queue |
| `GET /was/threejs/api/module` | menu label | the file listing, then the allowlist; refused while `threejs.allow_scripts` is off. A module that cannot be read is answered with fixed text naming no path |

Messages the pack sends over ComfyUI's socket about a run (tile captions, segment previews, held
runs, render progress notes) go only to the client id that queued it. A run queued without one
is announced to every tab, as ComfyUI's own messages are.

A client id is not a secret: ComfyUI lists them on its `/queue` route. The routes that answer by
client id keep one tab from reading or releasing another's run; they authenticate no one.

## The content viewer

The Content Viewer draws whatever a node holds, and a workflow carries that content, so the
frame it is drawn in never holds this page's origin.

| What is framed | Origin |
|---|---|
| Content built from a node, as `srcdoc` or a blob URL | None. `allow-same-origin` is stripped, whatever the view asked for |
| An app a view extension serves from its own endpoint | This origin, and only for an extension the user installed: by hand, or with `viewer.install_extensions` on, which is off by default, from `viewer-extensions/` or a `custom_nodes/ComfyUI_Viewer_*` checkout |

A frame reaches the viewer only through `postMessage`. The viewer acts on a message only when it
comes from a frame the viewer created, with that frame's own origin, and only for that frame's
own node and the view it is showing, so framed content cannot read another node's pictures,
change its settings or release its held run. A canvas frame is handed only the pictures the
Content Viewer wrote for it, and the page draws only `data:` and `blob:` pictures and those from
it. A document inside a frame carries a policy limiting what it may load and connect to.

While `viewer.allow_scripts` is off, the default, a document built from a node is placed with its
scripts, event handlers, forms, `<base>` and refresh markup removed, under a policy that runs only
the viewer's own scripts, each carrying a nonce the document cannot know. It can neither run
supplied code nor move its own frame, and a link in it opens in a new tab only after the address
is confirmed. With the setting on, the document runs its own scripts and can navigate its own
frame. The browser learns the setting from a Content Viewer node's run, never from the workflow,
and holds it off until a run says otherwise.

The rich text editor shows a document in a frame of this page's origin, under a content policy
that allows no scripts, plugins, frames, forms or connections, and loads images, fonts, media
and stylesheets only from this origin or from `data:` and `blob:` URLs.

Text a workflow carries reaches a context menu with every `<` replaced, and is drawn as text
whatever the frontend's version.

## Vendored code

Six directories under `web/` hold third-party browser libraries. `web/vendor/MANIFEST.json`
records a SHA-256 for all 92 of those files, alongside the npm package and version each
directory was taken from. Every digest recomputes from the file beside it, and the manifest
names no file the tree does not hold.

`NOTICE.md` in each directory names the upstream file behind every file it holds and what, if
anything, was changed in it: three.js and the path tracer have their import specifiers
rewritten, and the KaTeX stylesheet has its fonts inlined. To verify a copy independently, fetch
the package named in `MANIFEST.json` from npm and compare digests for the files `NOTICE.md` marks
unchanged.

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
padded. Load Video Cache reads a frame cache's manifest the same way: before any frame is read,
the frame size is held to the ceilings below and every run of frames the manifest lists must lie
inside `frames.bin`. A 16-bit PNG's declared size is held to the same ceilings before FFmpeg
decodes it, and a `.cube` LUT's declared size before its samples are read.

| Ceiling | Value | What it bounds |
|---|---|---|
| Longest side | 30000 pixels | The EXR data window, the PSD canvas and each PSD layer, the TIFF picture, a frame cache's frames, and a 16-bit PNG |
| Pixels | 268435456 | The same six, as an area, so two permitted sides cannot multiply into a refused one |
| Channels | 4 | A frame cache's frames |
| Samples | 1073741824 | An EXR's pixels times its channels, and a TIFF's pixels times its samples per pixel |
| Samples | 268435456 | The floats every PSD layer is decoded into, four a pixel, across the document, counted before any layer is decoded |
| Channels | 1024 | The EXR channel list |
| Channels | 56 | One PSD document or layer |
| Samples per pixel | 1024 | One TIFF pixel |
| Bits per sample | 8, 16 or 32 | The TIFF picture |
| Edge | 256 | A `.cube` LUT's `LUT_3D_SIZE` |
| Expansion | 2048 bytes per byte held | What a file of a given size may describe, in all three readers |
| Directory bytes | 2 times the file | Every value one TIFF directory copies, together |
| Chunk | 2147483647 bytes, and no more than the rest of the file | One PNG chunk read for its header |

A packed block is expanded with a byte ceiling, never past the size its header gives, so a
deflate bomb yields no more than the block it claims to be. A file is read whole into memory, so
the ceilings are what keep a small file from asking for a large allocation. The expansion figure
sits above the 1032 bytes per byte deflate can produce, so it refuses no real picture.

A Word document's property parts are read only when under 4 MB and holding no document type
declaration. A latent VAEEncode (Bundle Latent) carries in a
workflow is read only when its text is under 64 MB and inflates to no more than 256 MB.

## Bundled data

The Noodle Soup Prompts terminology ships as a snapshot, `modules/data/nsp_pantry.pack`:
zlib-compressed JSON holding 82 terms and 17,518 entries. The first Noodle Soup parse on an
empty database reads it into the state database and makes no request. The snapshot is refreshed
and re-bundled with a release, never at runtime.

## Static analysis

The following are false or outdated positives by Comfy Registry's pattern-matching scanners.
Each is listed with what the code is doing, so a reviewer can confirm it without reading the
whole tree.

| Reported as | Where | What it is |
|---|---|---|
| Socket `connect` | `web/was_app_workflow.js`, `web/interface/h3_asset_chain.js` | LiteGraph's own method for wiring one node's output slot to another node's input, called on a node the graph looked up by id |
| Socket `bind` | 5 of this pack's own files under `web/`, and `web/was_mask_brush.js`, `web/was_mask_rect_area.js` | `Function.prototype.bind`, the standard way to fix a callback's `this`, in those 5, which are every one of the pack's own browser files that uses it. The two mask files call the `bind` method `web/interface/mask_paint.js` defines. The vendored libraries call it throughout |
| Socket `connect`, `bind` | `web/vendor/three`, `web/vendor/hugerte` | The same two JavaScript methods, in third-party code for Three.js and HugeRTE |
| Socket `bind` | `web/viewer/views/code_scripts/prism.min.txt`, `web/viewer/views/markdown_scripts/mermaid.min.txt` | The same JavaScript method, in Prism and Mermaid. They are syntax highlighting and diagram drawing, read as text and inlined into the sandboxed frame a view builds, which has no access to this page |
| Network operation, database connection | `modules/state/store.py` | `sqlite3.connect`, opening a local database file. Reported twice, under both rule names |
| Dynamic import | `__init__.py` | Walking this pack's own `nodes/` package and importing each valid module it finds |
| Dynamic import | `modules/deps.py` | Resolving an optional dependency by name, so a missing one reports itself instead of raising on import |
| Dynamic import | `prestartup_script.py` | Loading this pack's own `modules/viewer/install.py` before ComfyUI starts |
| Dynamic import | `modules/viewer/parsers`, `modules/viewer/extension_nodes` | Loading the pack's own built-in parsers, and view extension modules placed there by hand or, with `viewer.install_extensions` on, installed from `viewer-extensions/*.zip` or a `custom_nodes/ComfyUI_Viewer_*` checkout |
| Dynamic import | `modules/model/pssr.py` | With `features.pssr` on, which is off by default, importing the PS-SR checkout the user places in `models/PS-SR` or names with `PSSR_ROOT`. That code and the `diffsynth` package it uses run as published, outside this pack's checks |
| Environment access | `modules/config/paths.py`, `prestartup_script.py` | Reading eight variables and writing none: `WAS_CONFIG` and `WAS_CONFIG_DIR` relocate the config file, `LOCALAPPDATA` and `XDG_DATA_HOME` locate user data, `HF_HOME`, `HF_HUB_CACHE` and `HUGGINGFACE_HUB_CACHE` find weights already on disk, and `PSSR_ROOT` finds one model. Every read in `modules/` goes through one function, `paths.env_value`. Python's `Path.home()` and `expanduser()` read the user's home directory, and `tempfile.gettempdir()` reads `TMPDIR`, `TEMP` and `TMP` to find the scratch folder |
| Sensitive file access | `modules/image/exr.py`, `modules/image/psd.py` | Reading the image file chosen in the node, resolved through the allowlist above whatever value arrives. See the section above for what bounds the parse |
| Blacklisted URL | `modules/model/flow_networks.py` | The two Google Drive pages where FlowSeek's authors publish its checkpoints, quoted as text in the message a missing checkpoint raises. Nothing opens them: the pack fetches FlowSeek's weights from Hugging Face only, with `features.network` on, and no other shipped file names Google Drive |

JavaScript files are reported under Python rule names by some scanners without proper filtering
enabled. `connect` and `bind` are ordinary JavaScript methods and carry no network meaning.

This page is itself matched. Naming an API to explain it puts that name in the file, and
version 3.2.2 was reported for a socket pattern at this document's own table row above. The
rows describe code held elsewhere in the tree; nothing here executes.
