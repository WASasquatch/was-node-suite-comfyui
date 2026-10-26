"""Fetch one web page onto an LLM documents wire."""

from __future__ import annotations

from urllib.parse import urlsplit

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM_DOCUMENTS
from ...modules.interface import run_result
from ...modules.llm import documents as loading
from ...modules.llm import runtime, wires

REQUIRES = "llm"

logger = log.get_logger("llm.documents")


class LLMLoadWebPage(io.ComfyNode):
    """One public web page's text, as a document."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMLoadWebPage",
            display_name="LLM Load Web Page",
            search_aliases=[
                "WASLLMLoadWebPage",
                "LLM Load Web Page",
                "SimpleWebPageReader",
                "BeautifulSoupWebReader",
                "fetch url",
                "web page",
                "scrape",
                "rag web",
            ],
            category="WAS Suite/LLM/Documents",
            description=(
                "Fetch one public web page and turn its text into a document, for splitting, "
                "indexing or a prompt. Needs features.network on; addresses on this machine or a "
                "private network are refused. The panel shows the page title and a sample of the "
                "text."
            ),
            inputs=[
                io.String.Input(
                    "url",
                    default="",
                    tooltip=(
                        "The page to read, such as `https://en.wikipedia.org/wiki/Lighthouse`. "
                        "http and https only; the address is sent exactly as typed."
                    ),
                ),
                io.Boolean.Input(
                    "readable_text",
                    default=True,
                    tooltip=(
                        "`true` = the page's main text, menus, headers and footers left out; "
                        "`false` = all of the page's text, link addresses included."
                    ),
                ),
                io.Int.Input(
                    "timeout",
                    default=20,
                    min=1,
                    max=120,
                    tooltip="Seconds the fetch may take, redirects included. `20` suits most sites.",
                ),
                io.Int.Input(
                    "max_kb",
                    default=2048,
                    min=16,
                    max=20480,
                    tooltip=(
                        "Most kilobytes of the page read; the rest is cut. `2048` holds a long "
                        "article; `20480` is the ceiling."
                    ),
                ),
                io.Boolean.Input(
                    "keep_html",
                    default=False,
                    tooltip=(
                        "`false` = the page as text; `true` = its markup, scripts and styles "
                        "removed, for LLM Split Documents set to html."
                    ),
                ),
                LLM_DOCUMENTS.Input(
                    "documents",
                    optional=True,
                    tooltip=(
                        "Documents from another loader, such as another LLM Load Web Page. This "
                        "page is added after them."
                    ),
                ),
            ],
            outputs=[
                LLM_DOCUMENTS.Output(
                    display_name="documents",
                    tooltip=(
                        "The page as one document, naming its address, title and when it was "
                        "fetched, for LLM Split Documents or an LLM index."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, url, readable_text, timeout, max_kb, keep_html, documents=None) -> io.NodeOutput:
        """Fetch the page and read its text.

        Args:
            url: The address.
            readable_text: Read the main text only.
            timeout: Seconds the fetch may take.
            max_kb: Most kilobytes read.
            keep_html: Keep the page's markup.
            documents: Documents to add this page after.

        Returns:
            The documents.

        Raises:
            RuntimeError: features.network is off, or the server could not be reached.
            ValueError: The address is refused, or the page holds no text.
        """
        runtime.require_network("LLM Load Web Page")
        previous = None
        if documents is not None:
            previous = wires.expect(
                documents, wires.Documents, "LLM Load Web Page", "documents", "an LLM document loader"
            )
        fetched = loading.fetch(url, float(timeout), int(max_kb) * 1024)
        made, title = loading.page_documents(fetched, bool(readable_text), bool(keep_html))
        host = urlsplit(fetched.url).hostname or ""
        added = wires.Documents(tuple(made), host)
        result = previous.plus(added) if previous is not None else added
        logger.info(
            "LLM Load Web Page read %d byte(s) from %s into %d character(s)",
            len(fetched.data), host, len(made[0].text),
        )
        facts = {"host": host, "status": fetched.status, "type": fetched.content_type or "unnamed"}
        if title:
            facts["title"] = title
        if fetched.redirects:
            facts["redirects"] = fetched.redirects
        if fetched.truncated:
            facts["cut at"] = f"{max_kb} KB"
        loading.publish(
            f"read {host}" + (" (cut at max_kb)" if fetched.truncated else ""),
            result,
            made,
            facts=facts,
            status=run_result.WARNING if fetched.truncated else run_result.OK,
        )
        return io.NodeOutput(result)
