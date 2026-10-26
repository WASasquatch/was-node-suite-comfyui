"""MCP servers as agent tools: the servers ``llm.mcp_servers`` in config.yaml names, and their tools.

A server is a command speaking over its standard streams, or an ``http`` or ``https`` address,
picked by its configured name.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from urllib.parse import urlsplit

from .. import deps, log
from . import FEATURE, runtime

__all__ = [
    "MAX_SERVERS",
    "NO_SERVERS",
    "Server",
    "fetch_tools",
    "options",
    "result_text",
    "server",
    "servers",
]

logger = log.get_logger("llm.mcp")

#: Servers read from the configuration at most.
MAX_SERVERS = 64

#: Characters of a server name kept.
MAX_NAME = 64

#: Arguments one command takes at most, and characters of one.
MAX_ARGS = 64
MAX_ARG_CHARS = 4096

#: Environment variables and headers one server is given at most.
MAX_ENV = 64

#: Tools one server hands an agent at most.
MAX_TOOLS = 256

#: Characters of a tool description kept.
MAX_DESCRIPTION_CHARS = 1024

#: Characters of one tool result handed the model at most.
MAX_RESULT_CHARS = 20000

#: The menu entry while the configuration names no server, never a valid name.
NO_SERVERS = "(no MCP servers in config.yaml)"

#: An environment variable or header name.
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")
_HEADER = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]{0,63}")

#: A tool name an agent can call.
_TOOL_NAME = re.compile(r"[^A-Za-z0-9_]+")


@dataclass(frozen=True)
class Server:
    """One server the configuration names.

    Attributes:
        name: Its name in ``llm.mcp_servers``.
        command: The program run for a standard streams server, ``""`` for an address.
        args: Its arguments.
        env: ``(name, value)`` pairs set for the program, beside a minimal environment.
        env_from: Names of ComfyUI's own environment variables passed on to the program.
        url: The address of an ``http`` or ``https`` server, ``""`` for a command.
        headers: ``(header, value)`` pairs sent to an address.
        bearer_env: The environment variable whose value is sent as a bearer token.
    """

    name: str
    command: str = ""
    args: tuple = ()
    env: tuple = ()
    env_from: tuple = ()
    url: str = ""
    headers: tuple = ()
    bearer_env: str = ""

    @property
    def local(self) -> bool:
        """True for a command, or an address on this machine."""
        return not self.url or runtime.is_loopback(self.url)

    @property
    def label(self) -> str:
        """What a panel calls the server: its name and what it runs or reaches."""
        if self.url:
            return f"{self.name} at {_host(self.url)}"
        return f"{self.name}: {os.path.basename(self.command)[:60]}"


def _host(url: str) -> str:
    """An address's host and port, without its path, query or credentials."""
    try:
        parts = urlsplit(url)
        return f"{parts.hostname or 'an address'}{f':{parts.port}' if parts.port else ''}"
    except ValueError:
        return "an address"


def _address_parts(url: str) -> tuple:
    """An address in full, its path with its query, and its credentials, each one hidden from errors."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return (url,)
    tail = parts.path + (f"?{parts.query}" if parts.query else "")
    return (url, tail if len(tail) > 1 else "", parts.password or "", parts.username or "")


def _pairs(raw, name_rule) -> tuple:
    """``(name, text)`` pairs from a configuration mapping, each name checked."""
    if not isinstance(raw, dict):
        return ()
    kept = []
    for key, value in list(raw.items())[:MAX_ENV]:
        if name_rule.fullmatch(str(key)) and isinstance(value, (str, int, float, bool)):
            kept.append((str(key), str(value)[:MAX_ARG_CHARS]))
    return tuple(kept)


def _entry(name: str, raw) -> Server | None:
    """A server from one ``llm.mcp_servers`` entry, or ``None`` when it is not usable."""
    if not isinstance(raw, dict):
        return None
    command = str(raw.get("command") or "").strip()
    url = str(raw.get("url") or "").strip()
    if bool(command) == bool(url):
        return None
    if url:
        try:
            parts = urlsplit(url)
        except ValueError:
            return None
        if parts.scheme not in ("http", "https") or not parts.hostname:
            return None
    args = raw.get("args") or []
    if not isinstance(args, list) or len(args) > MAX_ARGS:
        return None
    env_from = raw.get("env_from") or []
    if not isinstance(env_from, list):
        return None
    bearer = str(raw.get("bearer_env") or "").strip()
    if bearer and not _ENV_NAME.fullmatch(bearer):
        return None
    return Server(
        name=name,
        command=command[:MAX_ARG_CHARS],
        args=tuple(str(item)[:MAX_ARG_CHARS] for item in args),
        env=_pairs(raw.get("env"), _ENV_NAME),
        env_from=tuple(str(item) for item in env_from[:MAX_ENV] if _ENV_NAME.fullmatch(str(item))),
        url=url[:2048],
        headers=_pairs(raw.get("headers"), _HEADER),
        bearer_env=bearer,
    )


def servers() -> dict[str, Server]:
    """Every usable ``llm.mcp_servers`` entry in the loaded configuration, by name."""
    from ..config import load_config

    try:
        block = load_config().get("llm", {}).get("mcp_servers", {})
    except AttributeError:
        return {}
    if not isinstance(block, dict):
        return {}
    found: dict[str, Server] = {}
    for raw_name, raw in list(block.items())[:MAX_SERVERS]:
        name = str(raw_name).strip()[:MAX_NAME]
        made = _entry(name, raw) if name else None
        if made is None:
            logger.warning(
                "llm.mcp_servers entry %r was skipped: it needs either a command or an http or https url", name,
            )
            continue
        found[name] = made
    return found


def options() -> list[str]:
    """What the server menu offers: each configured name, or :data:`NO_SERVERS` alone."""
    return list(servers()) or [NO_SERVERS]


def server(name, node: str) -> Server:
    """The configured server a menu entry names.

    Raises:
        ValueError: The configuration names no server by that name.
        RuntimeError: The server is an address on another machine and ``features.network`` is off.
    """
    text = str(name or "").strip()
    found = servers().get(text)
    if found is None:
        if text == NO_SERVERS or not text:
            raise ValueError(
                f"{node}: config.yaml names no MCP server. Add one under llm.mcp_servers, such as "
                f"`files: {{command: npx, args: [-y, \"@modelcontextprotocol/server-filesystem\", \"C:/notes\"]}}`, "
                f"then restart ComfyUI."
            )
        raise ValueError(
            f"{node} was given `{text[:80]}`, which llm.mcp_servers in config.yaml does not name. Pick "
            f"a server from the menu."
        )
    if not found.local:
        runtime.require_network(f"the MCP server {found.name}")
    return found


def _client(found: Server, timeout: float):
    """The llama-index MCP client for a server."""
    mcp = deps.require("llama_index.tools.mcp", feature=FEATURE)
    if found.url:
        headers = dict(found.headers)
        if found.bearer_env:
            token = os.environ.get(found.bearer_env, "")
            if not token:
                raise RuntimeError(
                    f"The MCP server {found.name} sends a token from {found.bearer_env}, which is not set "
                    f"in the environment ComfyUI starts from. Set it and restart ComfyUI."
                )
            headers["Authorization"] = f"Bearer {token}"
        runtime.hide(*headers.values(), *_address_parts(found.url))
        return mcp.BasicMCPClient(found.url, timeout=int(timeout), headers=headers or None)
    env = dict(found.env)
    for variable in found.env_from:
        if variable in os.environ:
            env[variable] = os.environ[variable]
    runtime.hide(*env.values())
    return mcp.BasicMCPClient(found.command, args=list(found.args), env=env, timeout=int(timeout))


def _tool_name(prefix: str, name: str) -> str:
    """A tool name an agent can call: letters, digits and underscores, starting with a letter."""
    joined = "_".join(part for part in (prefix, name) if part)
    cleaned = _TOOL_NAME.sub("_", joined).strip("_")
    if not cleaned or not cleaned[0].isalpha():
        cleaned = f"tool_{cleaned}"
    return cleaned[:64]


async def fetch_tools(found: Server, wanted: list[str], prefix: str, timeout: float, node: str) -> list:
    """The tools a server offers, as llama-index function tools an agent calls.

    Args:
        found: The server.
        wanted: Tool names to keep, empty for all.
        prefix: Placed before every tool name, such as ``files``.
        timeout: Seconds one request may take.
        node: The node's display name, for messages.

    Returns:
        ``FunctionTool`` objects, each calling the server when the agent calls it.

    Raises:
        RuntimeError: The server could not be started or reached, or named tools it does not offer.
    """
    mcp = deps.require("llama_index.tools.mcp", feature=FEATURE)
    client = _client(found, timeout)
    spec = mcp.McpToolSpec(client=client, allowed_tools=list(wanted) or None)
    try:
        tools = await spec.to_tool_list_async()
    except Exception as error:
        if type(error).__name__ == "InterruptProcessingException":
            raise
        raise RuntimeError(
            f"{node} could not list the tools of the MCP server {found.label} "
            f"({runtime.describe_error(error)}). Check that its command runs, or that it answers at its address."
        ) from None
    offered = {tool.metadata.name for tool in tools}
    missing = [name for name in wanted if name not in offered]
    if missing:
        raise RuntimeError(
            f"{node}: the MCP server {found.name} offers no tool named {', '.join(missing[:8])}. Leave "
            f"allowed_tools blank to see the ones it offers on the panel."
        )
    from llama_index.core.tools import FunctionTool

    renamed = []
    for tool in tools[:MAX_TOOLS]:
        metadata = replace(
            tool.metadata,
            name=_tool_name(prefix, tool.metadata.name),
            description=str(tool.metadata.description or "")[:MAX_DESCRIPTION_CHARS],
        )
        renamed.append(FunctionTool(async_fn=_as_text(tool.async_fn), metadata=metadata))
    return renamed


def _as_text(call):
    """An MCP tool call that answers the text of the server's result in place of the result object."""

    async def answered(*args, **kwargs):
        try:
            return result_text(await call(*args, **kwargs))
        except Exception as error:
            if type(error).__name__ == "InterruptProcessingException":
                raise
            return f"Error: {runtime.describe_error(error)}"

    return answered


def result_text(result) -> str:
    """The words of an MCP tool result: its text parts, or its structured content as JSON.

    Args:
        result: An MCP ``CallToolResult``, or anything else, which reads as its text.

    Returns:
        The text, ``Error: `` before it for a result the server marked as an error, cut to
        :data:`MAX_RESULT_CHARS`.
    """
    import json

    parts = []
    for item in getattr(result, "content", None) or ():
        text = getattr(item, "text", None)
        if isinstance(text, str):
            parts.append(text)
        elif getattr(item, "type", "") == "image":
            parts.append("[a picture the server returned]")
        elif getattr(item, "type", "") == "resource":
            inner = getattr(getattr(item, "resource", None), "text", None)
            if isinstance(inner, str):
                parts.append(inner)
    written = "\n".join(parts).strip()
    if not written:
        structured = getattr(result, "structured_content", None) or getattr(result, "structuredContent", None)
        if structured is not None:
            try:
                written = json.dumps(structured, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                written = str(structured)
        elif not hasattr(result, "content"):
            written = str(result)
    if getattr(result, "is_error", False) or getattr(result, "isError", False):
        written = f"Error: {written or 'the server reported a failure'}"
    return written[:MAX_RESULT_CHARS]
