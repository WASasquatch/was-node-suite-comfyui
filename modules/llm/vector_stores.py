"""Vector stores outside an index's own files: Qdrant, Chroma, LanceDB and FAISS folders, and servers.

A local store is a folder under ``<config dir>/llm/vector_stores/<backend>``. A server is named in
``llm.vector_stores`` in config.yaml and picked by that name.
"""

from __future__ import annotations

import atexit
import os
import re
import threading
from dataclasses import dataclass, field
from urllib.parse import quote, unquote, urlsplit

from .. import deps, log
from ..util import sandbox
from . import FEATURE, runtime, storage

__all__ = [
    "BACKENDS",
    "EXTRA_FEATURE",
    "SERVER_KINDS",
    "SERVER_PREFIX",
    "VectorStore",
    "check_collection",
    "count",
    "open_store",
    "options",
    "servers",
]

logger = log.get_logger("llm.vector_stores")

#: Backends offered as a folder on this machine, Qdrant first.
BACKENDS = ("qdrant", "chroma", "lancedb", "faiss")

#: Backends a ``llm.vector_stores`` server entry may name in ``kind``.
SERVER_KINDS = ("qdrant", "chroma", "postgres")

#: The feature group whose packages every backend but Qdrant reads.
EXTRA_FEATURE = "features.vector_stores"

#: Label prefix of a server named in ``llm.vector_stores``.
SERVER_PREFIX = "server: "

#: The folder under ``<config dir>/llm`` local stores live in.
FOLDER = "vector_stores"

#: Servers read from the configuration at most.
MAX_SERVERS = 32

#: Seconds a server request may take.
TIMEOUT = 30

#: A collection name; Postgres keeps to lower case letters, digits and underscores.
_COLLECTION = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]{0,63}")
_TABLE = re.compile(r"[a-z][a-z0-9_]{0,62}")

#: An environment variable name.
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")

_lock = threading.Lock()
_clients: dict = {}


@dataclass(frozen=True, eq=False)
class VectorStore:
    """A collection in a vector store, on the ``WAS_LLM_VECTOR_STORE`` wire.

    Attributes:
        kind: ``qdrant``, ``chroma``, ``lancedb``, ``faiss`` or ``postgres``.
        collection: The collection's name.
        label: What panels call it, such as ``qdrant/documents``.
        local: True when it is on this machine.
        settings: Where it lives: ``folder`` for a local store, else the server's own fields.
    """

    kind: str
    collection: str
    label: str
    local: bool
    settings: dict = field(default_factory=dict)
    _held: dict = field(default_factory=dict)

    def models(self) -> tuple:
        return ()

    def store(self, dimensions: int = 0):
        """The llama-index vector store over the collection, made on first use.

        Args:
            dimensions: The vector length, which FAISS and Postgres need to make a new collection.
        """
        if "store" not in self._held:
            self._held["store"] = _make(self, int(dimensions))
        return self._held["store"]

    @property
    def keeps_files(self) -> bool:
        """True for FAISS, whose collection is its folder's files, written after every change."""
        return self.kind == "faiss"

    def open_index(self, embedding, create: bool = False):
        """A vector index over what the collection holds.

        Args:
            embedding: The :class:`~modules.llm.wires.EmbeddingModel` it searches with.
            create: Make an empty collection where none exists yet.

        Returns:
            The llama-index ``VectorStoreIndex``.
        """
        from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage

        if self.keeps_files:
            folder = self.settings["folder"]
            if os.path.isfile(os.path.join(folder, "docstore.json")):
                faiss_store = deps.require("llama_index.vector_stores.faiss", feature="features.vector_stores")
                loaded = faiss_store.FaissMapVectorStore.from_persist_dir(folder)
                self._held["store"] = loaded
                context = StorageContext.from_defaults(vector_store=loaded, persist_dir=folder)
                return load_index_from_storage(context, embed_model=embedding.model)
            if not create:
                return None
            context = StorageContext.from_defaults(vector_store=self.store(embedding.dimensions))
            return VectorStoreIndex(nodes=[], storage_context=context, embed_model=embedding.model)
        return VectorStoreIndex.from_vector_store(self.store(embedding.dimensions), embed_model=embedding.model)

    def delete_document(self, index, document: str) -> None:
        """Remove every chunk of one document from the collection.

        Args:
            index: The vector index over the collection.
            document: The document's id.
        """
        if self.kind == "faiss":
            # FAISS deletes by chunk id only, so a document's chunks are named one by one.
            held = index.docstore.get_ref_doc_info(document)
            if held is not None and held.node_ids:
                index.vector_store.delete_nodes(list(held.node_ids))
        elif self.kind == "lancedb":
            lancedb = deps.require("lancedb", feature="features.vector_stores")
            table = lancedb.connect(self.settings["folder"]).open_table(self.collection)
            quoted = str(document).replace("'", "''")
            table.delete(f"doc_id = '{quoted}'")
            return
        index.delete_ref_doc(document, delete_from_docstore=True)

    def written(self, index) -> None:
        """Keep what a write changed: FAISS writes its folder, every other store already holds it."""
        if self.keeps_files:
            target = sandbox.resolve_write_file(storage.root(), "/".join((FOLDER, self.kind, self.collection)))
            target.mkdir(parents=True, exist_ok=True)
            index.storage_context.persist(persist_dir=str(target))


def servers() -> dict[str, dict]:
    """Every usable ``llm.vector_stores`` entry, by name: ``kind``, ``url`` and its key or password variable."""
    from ..config import load_config

    try:
        block = load_config().get("llm", {}).get("vector_stores", {})
    except AttributeError:
        return {}
    if not isinstance(block, dict):
        return {}
    found: dict[str, dict] = {}
    for raw, entry in list(block.items())[:MAX_SERVERS]:
        name = str(raw).strip()[:64]
        if not name or not isinstance(entry, dict):
            continue
        kind = str(entry.get("kind") or "qdrant").strip().lower()
        url = str(entry.get("url") or "").strip().rstrip("/")
        try:
            parts = urlsplit(url)
        except ValueError:
            continue
        schemes = ("postgresql", "postgres") if kind == "postgres" else ("http", "https")
        secret = str(entry.get("password_env" if kind == "postgres" else "api_key_env") or "").strip()
        if kind not in SERVER_KINDS or parts.scheme not in schemes or not parts.hostname or (secret and not _ENV_NAME.fullmatch(secret)):
            logger.warning("llm.vector_stores entry %r was skipped: its kind, url or key variable cannot be used", name)
            continue
        found[name] = {"kind": kind, "url": url, "secret_env": secret}
    return found


def options() -> list[str]:
    """What the store menu offers: Qdrant, the other local backends while their group is on, then servers."""
    from ..config import group_enabled

    local = list(BACKENDS) if group_enabled(EXTRA_FEATURE) else ["qdrant"]
    return [*local, *(SERVER_PREFIX + name for name in servers())]


def check_collection(name, node: str, kind: str = "qdrant") -> str:
    """A collection name, trimmed, once it fits what the backend accepts.

    Raises:
        ValueError: The name holds anything else.
    """
    text = str(name or "").strip()
    if kind == "postgres":
        if _TABLE.fullmatch(text):
            return text
        raise ValueError(
            f"{node}: collection {text[:80]!r} cannot be used on Postgres. Use 1 to 63 lower case letters, "
            f"digits and underscores, starting with a letter, such as `studio_notes`."
        )
    if _COLLECTION.fullmatch(text):
        return text
    raise ValueError(
        f"{node}: collection {text[:80]!r} cannot be used. Use 1 to 64 letters, digits, dashes and "
        f"underscores, starting with a letter or digit, such as `studio_notes`."
    )


def _folder(kind: str, *parts: str) -> str:
    """A local backend's folder under the pack's config directory, made where missing."""
    target = sandbox.resolve_write_file(storage.root(), "/".join((FOLDER, kind, *parts)))
    target.mkdir(parents=True, exist_ok=True)
    return str(target)


def _client(key: tuple, build):
    """A client kept for the life of ComfyUI, one per backend and place."""
    with _lock:
        if key not in _clients:
            if not _clients:
                atexit.register(_close_clients)
            _clients[key] = build()
        return _clients[key]


def _close_clients() -> None:
    """Close every kept client while the interpreter can still run their cleanup."""
    with _lock:
        held = list(_clients.values())
        _clients.clear()
    for client in held:
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception as error:
                logger.debug("a vector store client did not close (%s)", error)


def _secret(entry: dict, node: str) -> str | None:
    """The key or password a server entry names, or ``None`` where it names none.

    Raises:
        RuntimeError: The entry names a variable that is not set.
    """
    variable = entry.get("secret_env") or ""
    if not variable:
        return None
    value = os.environ.get(variable)
    if not value:
        raise RuntimeError(
            f"{node}: the vector store server reads its secret from {variable}, which is not set in the "
            f"environment ComfyUI starts from. Set it and restart ComfyUI."
        )
    runtime.hide(value)
    return value


def open_store(where: str, collection, node: str) -> VectorStore:
    """A collection in a local backend or on a configured server, created when first written.

    Args:
        where: An entry of :func:`options`.
        collection: The collection's name.
        node: The node's display name, for messages.

    Raises:
        ValueError: The name, the backend or the server cannot be used.
        RuntimeError: A server on another machine with ``features.network`` off, or its secret unset.
    """
    from ..config import group_enabled

    choice = str(where or "").strip()
    if choice in BACKENDS:
        if choice != "qdrant" and not group_enabled(EXTRA_FEATURE):
            raise ValueError(
                f"{node}: {choice} needs features.vector_stores on in config.yaml and its packages: "
                f"{deps.install_command(feature='features.vector_stores')}. Turn it on and restart ComfyUI, or pick qdrant."
            )
        name = check_collection(collection, node, choice)
        settings = {"folder": _folder(choice, name) if choice == "faiss" else _folder(choice)}
        return VectorStore(kind=choice, collection=name, label=f"{choice}/{name}", local=True, settings=settings)
    entry = servers().get(choice[len(SERVER_PREFIX):]) if choice.startswith(SERVER_PREFIX) else None
    if entry is None:
        raise ValueError(
            f"{node} was given `{choice[:80]}`, which is neither a local store nor a server llm.vector_stores in "
            f"config.yaml names. Pick one from the menu."
        )
    if entry["kind"] != "qdrant" and not group_enabled(EXTRA_FEATURE):
        raise ValueError(
            f"{node}: a {entry['kind']} server needs features.vector_stores on in config.yaml and its packages: "
            f"{deps.install_command(feature='features.vector_stores')}."
        )
    local = runtime.is_loopback(entry["url"]) or (urlsplit(entry["url"]).hostname or "") in ("localhost", "127.0.0.1", "::1")
    if not local:
        runtime.require_network(f"the vector store at {urlsplit(entry['url']).hostname}")
    name = check_collection(collection, node, entry["kind"])
    server = choice[len(SERVER_PREFIX):]
    return VectorStore(
        kind=entry["kind"], collection=name, label=f"{server}/{name}", local=local,
        settings={**entry, "secret": _secret(entry, node), "server": server},
    )


def _postgres(settings: dict) -> dict:
    """A Postgres server's ``host``, ``port``, ``database``, ``user`` and ``password``, the password hidden from errors."""
    parts = urlsplit(settings["url"])
    password = settings.get("secret") or unquote(parts.password or "")
    runtime.hide(password)
    return {
        "host": parts.hostname, "port": int(parts.port or 5432),
        "database": (parts.path or "/").lstrip("/") or "postgres",
        "user": unquote(parts.username or "postgres"), "password": password,
    }


def _make(wire: VectorStore, dimensions: int):
    """The llama-index vector store a wire names."""
    kind, name, settings = wire.kind, wire.collection, wire.settings
    if kind == "qdrant":
        qdrant = deps.require("qdrant_client", feature=FEATURE)
        store = deps.require("llama_index.vector_stores.qdrant", feature=FEATURE)
        if "folder" in settings:
            client = _client(("qdrant", settings["folder"]), lambda: qdrant.QdrantClient(path=settings["folder"]))
        else:
            client = qdrant.QdrantClient(url=settings["url"], api_key=settings.get("secret"), timeout=TIMEOUT)
        wire._held["client"] = client
        return store.QdrantVectorStore(collection_name=name, client=client)
    if kind == "chroma":
        chromadb = deps.require("chromadb", feature="features.vector_stores")
        store = deps.require("llama_index.vector_stores.chroma", feature="features.vector_stores")
        quiet = chromadb.Settings(anonymized_telemetry=False)
        if "folder" in settings:
            client = _client(("chroma", settings["folder"]), lambda: chromadb.PersistentClient(path=settings["folder"], settings=quiet))
        else:
            parts = urlsplit(settings["url"])
            headers = {"x-chroma-token": settings["secret"]} if settings.get("secret") else None
            client = chromadb.HttpClient(
                host=parts.hostname, port=parts.port or (443 if parts.scheme == "https" else 8000),
                ssl=parts.scheme == "https", headers=headers, settings=quiet,
            )
        collection = client.get_or_create_collection(name, metadata={"hnsw:space": "cosine"})
        wire._held["client"] = collection
        return store.ChromaVectorStore(chroma_collection=collection)
    if kind == "lancedb":
        deps.require("pandas", feature="features.vector_stores")
        store = deps.require("llama_index.vector_stores.lancedb", feature="features.vector_stores")
        lancedb = deps.require("lancedb", feature="features.vector_stores")
        held = lancedb.connect(settings["folder"]).table_names()
        return store.LanceDBVectorStore(uri=settings["folder"], table_name=name, mode="append" if name in held else "create")
    if kind == "faiss":
        faiss = deps.require("faiss", feature="features.vector_stores")
        store = deps.require("llama_index.vector_stores.faiss", feature="features.vector_stores")
        if dimensions <= 0:
            raise ValueError("A new FAISS collection needs the embedding model's vector length; run the embedding model first.")
        return store.FaissMapVectorStore(faiss_index=faiss.IndexIDMap2(faiss.IndexFlatIP(int(dimensions))))
    if kind == "postgres":
        store = deps.require("llama_index.vector_stores.postgres", feature="features.vector_stores")
        if dimensions <= 0:
            raise ValueError("A Postgres collection needs the embedding model's vector length; run the embedding model first.")
        fields = _postgres(settings)
        host = f"[{fields['host']}]" if ":" in fields["host"] else fields["host"]
        login = quote(fields["user"], safe="") + (f":{quote(fields['password'], safe='')}" if fields["password"] else "")
        place = f"{login}@{host}:{fields['port']}/{quote(fields['database'], safe='')}"
        return store.PGVectorStore.from_params(
            connection_string=f"postgresql+psycopg2://{place}",
            async_connection_string=f"postgresql+asyncpg://{place}",
            table_name=name, embed_dim=int(dimensions),
        )
    raise ValueError(f"{kind!r} is not a vector store this pack opens.")


def count(store: VectorStore) -> int:
    """How many vectors the collection holds, 0 while it does not exist or cannot be read."""
    try:
        if store.kind == "qdrant":
            made = store.store()
            client = store._held.get("client") or made.client
            if not client.collection_exists(store.collection):
                return 0
            return int(client.count(store.collection, exact=True).count)
        if store.kind == "chroma":
            store.store()
            return int(store._held["client"].count())
        if store.kind == "lancedb":
            lancedb = deps.require("lancedb", feature="features.vector_stores")
            database = lancedb.connect(store.settings["folder"])
            names = database.table_names() if hasattr(database, "table_names") else []
            return int(database.open_table(store.collection).count_rows()) if store.collection in names else 0
        if store.kind == "faiss":
            held = store._held.get("store")
            if held is not None:
                return int(held._faiss_index.ntotal)
            path = os.path.join(store.settings["folder"], "default__vector_store.json")
            if not os.path.isfile(path):
                return 0
            faiss = deps.require("faiss", feature="features.vector_stores")
            return int(faiss.read_index(path).ntotal)
        if store.kind == "postgres":
            psycopg2 = deps.require("psycopg2", feature="features.vector_stores")
            fields = _postgres(store.settings)
            table = f"data_{store.collection}"
            connection = psycopg2.connect(
                host=fields["host"], port=fields["port"], dbname=fields["database"], user=fields["user"],
                password=fields["password"], connect_timeout=TIMEOUT,
            )
            try:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT to_regclass(%s)", (f"public.{table}",))
                    if cursor.fetchone()[0] is None:
                        return 0
                    cursor.execute(f'SELECT count(*) FROM public."{table}"')
                    return int(cursor.fetchone()[0])
            finally:
                connection.close()
    except Exception as error:
        logger.debug("the collection %s could not be counted (%s)", store.label, error)
    return 0
