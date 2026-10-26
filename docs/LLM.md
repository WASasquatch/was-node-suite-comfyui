# LLM nodes

Install, keys, model folders, saved work and servers for the nodes under **WAS Suite/LLM**.
Every node, with its inputs and outputs, is in [`NODES.md`](../NODES.md).

## Turning them on

| Step | What to do |
|---|---|
| Install | `pip install -r requirements/llm.txt` with ComfyUI's own python. On the portable build, from `ComfyUI_windows_portable`: `python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\was-node-suite-comfyui\requirements\llm.txt` |
| Enable | `features.llm: true` in `ComfyUI/user/was-node-suite/config.yaml`, then restart ComfyUI |
| Hosted models | `features.network: true` as well. With it off, every hosted model and web tool refuses to run, and models on this machine still run |
| GGUF models | A `llama-cpp-python` build matching the machine's CUDA version, from that project's prebuilt wheels. The build PyPI installs runs on the CPU only |

## Keys

Keys are read from the environment ComfyUI starts in, never typed into a node, so a shared
workflow carries none. Set the variable, then restart ComfyUI.

| Node or preset | Variable | Address |
|---|---|---|
| LLM OpenAI, OpenAI embeddings | `OPENAI_API_KEY` | `https://api.openai.com/v1` |
| LLM Anthropic | `ANTHROPIC_API_KEY` | `https://api.anthropic.com` |
| LLM Google Gemini, Gemini embeddings | `GOOGLE_API_KEY` or `GEMINI_API_KEY` | `https://generativelanguage.googleapis.com` |
| LLM Provider: Groq | `GROQ_API_KEY` | `https://api.groq.com/openai/v1` |
| LLM Provider: DeepSeek | `DEEPSEEK_API_KEY` | `https://api.deepseek.com` |
| LLM Provider: xAI | `XAI_API_KEY` | `https://api.x.ai/v1` |
| LLM Provider: Mistral | `MISTRAL_API_KEY` | `https://api.mistral.ai/v1` |
| LLM Provider: OpenRouter | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1` |
| LLM Provider: Together AI | `TOGETHER_API_KEY` | `https://api.together.ai/v1` |
| LLM Provider: Fireworks AI | `FIREWORKS_API_KEY` | `https://api.fireworks.ai/inference/v1` |
| LLM Provider: Perplexity | `PERPLEXITY_API_KEY` | `https://api.perplexity.ai` |
| LLM Provider: Cerebras | `CEREBRAS_API_KEY` | `https://api.cerebras.ai/v1` |
| LLM Provider: NVIDIA | `NVIDIA_API_KEY` | `https://integrate.api.nvidia.com/v1` |
| LLM Tool: web search (Tavily) | `TAVILY_API_KEY` | `https://api.tavily.com/search` |
| LLM Tool: web search (Brave) | `BRAVE_API_KEY` | `https://api.search.brave.com/res/v1/web/search` |
| A server in `llm.endpoints` | the variable its `key_env` names | its `url` |

## Servers on this machine

**LLM Provider** and **LLM Embedding Model** reach these with `features.network` off.

| Preset | Address |
|---|---|
| Ollama | `http://127.0.0.1:11434` |
| LM Studio | `http://127.0.0.1:1234/v1` |
| vLLM | `http://127.0.0.1:8000/v1` |
| llama.cpp server | `http://127.0.0.1:8080/v1` |

`llm.endpoints` in `config.yaml` adds servers by name, offered as `endpoint: <name>`. An address
other than this machine needs `features.network`.

```yaml
llm:
  endpoints:
    lan-ollama: {url: "http://192.168.1.20:11434", api: ollama}
    my-vllm: {url: "http://127.0.0.1:8000/v1", api: openai, key_env: VLLM_KEY}
```

| Field | Value |
|---|---|
| `url` | The server's `http` or `https` address |
| `api` | `openai` for an OpenAI-compatible server, `ollama` for Ollama's own API |
| `key_env` | The environment variable holding the server's key; left out, the server is sent no key |

## Model folders

| Model | Folder |
|---|---|
| Language models for LLM From CLIP | `models/text_encoders`, loaded by core **Load CLIP** |
| GGUF language and embedding models | `models/LLM` and `models/text_encoders` |
| Hugging Face folders: language, embedding and reranker models | `models/LLM`, one folder per model holding its `config.json` |
| Picture models for LLM Image Embedding Model | `models/LLM`, one CLIP or SigLIP folder holding its `config.json` |

A suggested repository on **LLM Transformers Loader**, **LLM Embedding Model**, **LLM Image
Embedding Model** or **LLM Rerank** downloads into `models/LLM/<owner>/<name>` the first time it runs, with `features.network` on.
Only configuration, tokenizer and `.safetensors` files are fetched, and no model carrying its own
code is loaded.

## Saved indexes and conversations

| What | Where | Written by | Removed by |
|---|---|---|---|
| An index | `ComfyUI/user/was-node-suite/llm/indexes/<name>/` | LLM Save Index | A save under the same name replaces its files |
| A vector store collection on this machine | `ComfyUI/user/was-node-suite/llm/vector_stores/` | LLM Vector Index with a `vector_store` connected | Building a document again replaces its chunks |
| A conversation | `ComfyUI/user/was-node-suite/llm/sessions/<name>.json` | LLM Chat Session under its `conversation` name, or a node wired to an LLM Memory with a session name, on every run | Delete in the chat panel's menu, LLM Clear Memory with `confirm` on, or LLM Chat Session with `reset` on |
| Vector memory of a conversation | `ComfyUI/user/was-node-suite/llm/sessions/<name>.<block>.vectors.json` | The same | The same |
| The pictures of an image index | Inside its index folder, a JPEG copy of each at most 384 pixels a side, with its folder and path below that folder | LLM Save Index | A save under the same name |

A name holds letters, digits, spaces, dots, dashes and underscores, up to 64 characters. A
message sent again unchanged to a saved conversation is answered from it, not sent to the model.

## The chat panel

| Control | What it does |
|---|---|
| Conversation menu | Lists the saved conversations, newest first, with their message counts and saved times. Picking one shows it and sends the next message to it |
| New | Starts a conversation. The first message sent names it from its opening words |
| Rename | Renames the open conversation and its vector memory files. A name already saved is refused |
| Delete | Deletes the open conversation and its vector memory files, after a confirmation |
| `reset on` | `reset` is on, so every run deletes the conversation before its message |

A `conversation` name on LLM Chat Session comes before the session name of a wired LLM Memory.
With neither, the chat lasts one run.

## Pictures

| What | Value |
|---|---|
| Files LLM Load Images reads | `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.gif`, `.tif`, `.tiff` |
| Prompts read from a file | The positive prompt and model a ComfyUI PNG saved, or the prompt and model of an A1111 `parameters` chunk |
| Suggested picture models | `google/siglip2-base-patch16-224`, `google/siglip2-so400m-patch14-384`, `openai/clip-vit-large-patch14`, `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` |
| LLM Sources to Images with `full_size` on | The picture file at its own size while it is still in its folder, else the stored copy |
| A query engine over an image index | Shows the pictures found to a model that reads pictures; any other model is given their text |
| Frames LLM To Documents takes from a video | `video_frames`, spread evenly from the first frame to the last, each named by its time, as `clip 00:12.5` |

## MCP servers

**LLM MCP Tools** offers the servers `llm.mcp_servers` in `config.yaml` names. Restart ComfyUI
after adding one.

```yaml
llm:
  mcp_servers:
    notes: {command: npx, args: [-y, "@modelcontextprotocol/server-filesystem", "C:/notes"]}
    github: {command: "C:/tools/github-mcp-server.exe", args: [stdio], env_from: [GITHUB_TOKEN]}
    tracker: {url: "https://mcp.example.com/mcp", bearer_env: TRACKER_TOKEN}
```

| Field | Value |
|---|---|
| `command` | The program run on this machine, speaking MCP over its standard streams |
| `args` | Its arguments, as a list |
| `env` | Variables set for the program, as a mapping |
| `env_from` | Names of variables in the environment ComfyUI starts from, passed on to the program |
| `url` | The `http` or `https` address of a server; an address other than this machine needs `features.network` |
| `headers` | Headers sent to an address, as a mapping |
| `bearer_env` | The variable holding a token sent as `Authorization: Bearer` |

## Vector stores

**LLM Vector Store** keeps a vector index in a collection rather than in the index's own files.
Building into a collection replaces the chunks of each document it already holds.

| Store | Where | Needs |
|---|---|---|
| `qdrant` | `ComfyUI/user/was-node-suite/llm/vector_stores/qdrant` | `features.llm` |
| `chroma` | `ComfyUI/user/was-node-suite/llm/vector_stores/chroma` | `features.vector_stores` |
| `lancedb` | `ComfyUI/user/was-node-suite/llm/vector_stores/lancedb` | `features.vector_stores` |
| `faiss` | `ComfyUI/user/was-node-suite/llm/vector_stores/faiss/<collection>` | `features.vector_stores` |
| `server: <name>` | A server `llm.vector_stores` in `config.yaml` names; an address other than this machine needs `features.network` | `features.vector_stores` for Chroma and Postgres |

`features.vector_stores` is installed with `pip install -r requirements/vector_stores.txt`, run the
same way as the LLM requirements, and turned on in `config.yaml`.

```yaml
llm:
  vector_stores:
    team: {url: "https://qdrant.example.com:6333", api_key_env: QDRANT_API_KEY}
    chroma-box: {kind: chroma, url: "http://192.168.1.30:8000", api_key_env: CHROMA_TOKEN}
    warehouse: {kind: postgres, url: "postgresql://comfy@db.example.com:5432/vectors", password_env: PG_PASSWORD}
```

| Field | Value |
|---|---|
| `kind` | `qdrant`, `chroma` or `postgres`; left out, `qdrant` |
| `url` | `http` or `https` for Qdrant and Chroma; `postgresql://user@host:port/database` for Postgres |
| `api_key_env` | Qdrant and Chroma: the variable holding the server's key |
| `password_env` | Postgres: the variable holding the user's password |

| What | Value |
|---|---|
| Collection names | 1 to 64 letters, digits, dashes and underscores; on Postgres, 1 to 63 lower case letters, digits and underscores, starting with a letter |
| A Postgres collection | The table `data_<collection>` in the named database, which needs the `vector` extension |
| Reopening | **LLM Vector Store Index** with the embedding model that filled the collection, not **LLM Save Index** and **LLM Load Index** |

## Tables

| What | Value |
|---|---|
| Files LLM Load Table reads | `.csv`, `.tsv`, `.xlsx`, `.json` (a list of objects), `.jsonl`, `.sqlite`, `.sqlite3`, `.db` |
| Table and column names | Lower case letters, digits and underscores, taken from the file and its header row |
| Column types | Whole numbers, numbers or text, read from the values; an empty cell is empty |
| A SQLite file | Every table in it is copied, read through a read-only connection |
| Changes | None. Questions are answered with reading queries only, and a query past `timeout_seconds` is stopped |

## Troubleshooting

| What you see | Cause | Fix |
|---|---|---|
| No LLM nodes in the menu | `features.llm` is off | Turn it on and restart ComfyUI |
| `llama_index.core is not installed` | The requirements are not in ComfyUI's python | Run the install line above with ComfyUI's own python |
| `sends text to a server on another machine, and features.network is off` | A hosted model, a web page or a web tool, with network off | Turn on `features.network`, or use a model on this machine |
| `Set OPENAI_API_KEY in the environment ComfyUI starts from` | The variable is not set where ComfyUI runs | Set it and restart ComfyUI |
| `Nothing answered at the model's address` | A local server such as Ollama is not running | Start it and load the model |
| `You have no credits remaining` | The provider account has no credit | Add credit with the provider |
| GGUF runs slowly and its panel says CPU only | `llama-cpp-python` came from PyPI | Install a CUDA build |
| `This step needs an embedding model and none was connected` | A step needed embeddings and none was wired | Connect LLM Embedding Model to the node's `embedding` input |
| A chat answers a repeated message with its last reply | An unchanged message to a saved conversation is answered from it | Change the message to ask again |
| The conversation menu is empty after a restart | `features.llm` was turned off, or nothing has been saved yet | Turn it on and restart ComfyUI, or send a message |
| `Conversation name ... cannot be used` | The `conversation` name holds a character outside the set allowed | Rename it with letters, digits, spaces, dots, dashes or underscores |
| A named time zone is refused by the date tool | No zone database on this machine | Install the requirements again; they include the zone names |
| `config.yaml names no MCP server` | `llm.mcp_servers` is empty | Add a server as shown above and restart ComfyUI |
| `could not list the tools of the MCP server` | Its command is not installed, or nothing answers at its address | Run the command in a terminal to see its error, or start the server |
| `holds no text, and it indexes text only` | Pictures were wired to a text index | Use LLM Image Index, or connect a vision model to `describe` on LLM Load Images |
| `The image index ... needs it to search` | LLM Load Index has no picture model | Connect LLM Image Embedding Model to its `image_embedding` input |
| `.xlsx` files are refused | openpyxl is missing | Install the requirements again |
| `a JSON table is read up to 64 MB` | The JSON or JSON Lines table is larger | Save it as CSV |
| `would write more than 64 million characters from one JSON document` | A deeply nested JSON document in `json` split mode | Split it with another splitter, such as `sentence` |
| `This index lives in the vector store` | LLM Save Index was given an index kept in a vector store | Reopen it with LLM Vector Store and LLM Vector Store Index instead |
| `holds no vectors yet` | LLM Vector Store Index opened an empty collection | Fill it with LLM Vector Index, its `vector_store` input connected |
| The store menu offers only `qdrant` | `features.vector_stores` is off | Install `requirements/vector_stores.txt`, turn it on and restart ComfyUI |
| `reads its secret from ... which is not set` | The server's `api_key_env` or `password_env` variable is not set where ComfyUI runs | Set it and restart ComfyUI |
| `type "vector" does not exist` | The Postgres database has no `vector` extension | Run `CREATE EXTENSION vector;` in that database |
| `cannot turn a ... into documents` | LLM To Documents was given a kind it does not read | Connect text, a list of text, an IMAGE, a VIDEO, a DOC, a conversation, sources or tables |
