"""LLM Tool's web search (Monid), with every request mocked: nothing here reaches the network."""

import asyncio
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Pytest imports the pack's own __init__.py for the folder's package setup, and that file loads
# only inside ComfyUI. An empty module under the name pytest looks for stands in for it.
sys.modules.setdefault(ROOT.name if ROOT.name.isidentifier() else "__init__", ModuleType("__init__"))

from modules.llm import runtime, tools  # noqa: E402

KEY = "monid_test1234567890_abcdefghijkl"


@pytest.fixture
def client(monkeypatch):
    """The httpx client the tools call, mocked; each test queues its answers on `send`."""
    monkeypatch.setenv("MONID_API_KEY", KEY)
    monkeypatch.setattr(tools, "_function_tool", lambda fn, name, description: fn)
    monkeypatch.setattr(runtime, "require_network", lambda what: None)
    monkeypatch.setattr(runtime, "check_interrupt", lambda: None)
    monkeypatch.setattr(tools.asyncio, "sleep", AsyncMock())
    mocked = Mock(send=AsyncMock())
    mocked.build_request.return_value.url = SimpleNamespace(host="api.monid.ai", scheme="https")
    mocked.__aenter__ = AsyncMock(return_value=mocked)
    mocked.__aexit__ = AsyncMock(return_value=False)
    httpx = SimpleNamespace(AsyncClient=lambda **kwargs: mocked)
    monkeypatch.setattr(tools.deps, "require", lambda *args, **kwargs: httpx)
    return mocked


def answer(status, payload):
    async def chunks():
        yield json.dumps(payload).encode("utf-8")

    return SimpleNamespace(status_code=status, next_request=None, aiter_bytes=chunks, aclose=AsyncMock())


def search(query, **settings):
    tool, = tools.monid_tools(**settings)
    return asyncio.run(tool(query))


def test_search_sends_the_query_and_the_count(client):
    client.send.return_value = answer(200, {"status": "COMPLETED", "output": {"results": [
        {"title": "ComfyUI", "url": "https://example.com/docs", "description": "Node documentation", "relevance": "high"},
    ]}})

    assert search("节点 docs", results=3) == "Web results:\n- ComfyUI\n  https://example.com/docs\n  Node documentation"
    (method, url), sent = client.build_request.call_args
    assert (method, url) == ("POST", "https://api.monid.ai/v1/run")
    assert sent["headers"]["Authorization"] == f"Bearer {KEY}"
    assert sent["headers"]["Content-Type"] == "application/json; charset=utf-8"
    assert "节点".encode("utf-8") in sent["content"]
    assert json.loads(sent["content"]) == {
        "provider": "context.dev", "endpoint": "/web/search", "input": {"body": {"query": "节点 docs", "numResults": 3}},
    }


def test_search_still_going_is_read_back(client):
    client.send.side_effect = [
        answer(202, {"runId": "run/1"}),
        answer(200, {"runId": "run/1", "status": "RUNNING"}),
        answer(200, {"runId": "run/1", "status": "COMPLETED", "output": {"results": []}}),
    ]

    assert search("ComfyUI") == "The web search found nothing. Try other words."
    reads = client.build_request.call_args_list[1:]
    assert [call.args for call in reads] == [("GET", "https://api.monid.ai/v1/runs/run%2F1")] * 2
    assert all(call.kwargs["headers"] == {"User-Agent": tools.USER_AGENT, "Accept": "application/json",
                                          "Authorization": f"Bearer {KEY}"} for call in reads)


def test_search_gives_up_after_the_wait(client, monkeypatch):
    monkeypatch.setattr(tools, "MONID_WAIT", 0.0)
    client.send.return_value = answer(202, {"runId": "run-1"})

    assert search("ComfyUI") == "Error: The web search failed: the search was still going after 0 seconds"
    assert client.send.call_count == 1


@pytest.mark.parametrize("payload, said", [
    ({"status": "COMPLETED", "providerResponse": {"httpStatus": 404, "error": {"message": "nothing"}}},
     "the service answered HTTP 404"),
    ({"runId": "run-1", "status": "FAILED"}, "the search ended FAILED"),
])
def test_failed_search_is_an_error_for_the_model(client, payload, said):
    client.send.return_value = answer(200, payload)

    assert search("ComfyUI") == f"Error: The web search failed: {said}"


def test_key_never_reaches_the_model(client):
    client.send.side_effect = RuntimeError(f"connection reset, Authorization: Bearer {KEY}")

    text = search("ComfyUI")
    assert text.startswith("Error: The web search failed: connection reset")
    assert KEY not in text
    assert KEY not in runtime.describe_error(f"refused key {KEY}")
    assert runtime._SECRET.sub("[key hidden]", f"key {KEY} refused") == "key [key hidden] refused"


def test_missing_key_is_named_and_nothing_is_sent(client, monkeypatch):
    monkeypatch.delenv("MONID_API_KEY")

    with pytest.raises(ValueError, match="MONID_API_KEY is not set"):
        tools.monid_tools()
    client.send.assert_not_called()


def test_network_off_refuses_the_search(client, monkeypatch):
    def refuse(what):
        raise RuntimeError(f"{what} sends text to a server on another machine, and features.network is off.")

    monkeypatch.setattr(runtime, "require_network", refuse)

    assert search("ComfyUI").startswith("The Monid search tool sends text to a server on another machine")
    client.send.assert_not_called()


def test_brave_search_is_unchanged_with_a_monid_key(client, monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "brave-test-key-123456")
    client.send.return_value = answer(200, {"web": {"results": []}})

    tool, = tools.brave_tools()
    assert asyncio.run(tool("ComfyUI")) == "The web search found nothing. Try other words."
    (method, url), sent = client.build_request.call_args
    assert (method, url) == ("GET", "https://api.search.brave.com/res/v1/web/search")
    assert sent["params"] == {"q": "ComfyUI", "count": 5} and sent["json"] is None
