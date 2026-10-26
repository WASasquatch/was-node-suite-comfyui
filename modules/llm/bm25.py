"""Okapi BM25 keyword ranking over chunks, as a llama-index retriever.

Text is split into words, lowercased, cleared of stopwords from the NLTK lists bundled with
llama-index and stemmed. :func:`retriever` builds the retriever, :func:`corpus` gathers chunks.
"""

from __future__ import annotations

import math
import os
import re
import time
from collections import Counter

from .. import deps, log
from . import FEATURE, runtime

__all__ = ["DEFAULT_B", "DEFAULT_K1", "LANGUAGES", "Okapi", "analyzer", "corpus", "retriever", "stopwords"]

logger = log.get_logger("llm.retrieve")

#: Stopword lists bundled with llama-index's NLTK data, plus ``none`` for no list.
LANGUAGES = (
    "english", "none", "albanian", "arabic", "azerbaijani", "basque", "belarusian", "bengali",
    "catalan", "chinese", "danish", "dutch", "finnish", "french", "german", "greek", "hebrew",
    "hinglish", "hungarian", "indonesian", "italian", "kazakh", "nepali", "norwegian",
    "portuguese", "romanian", "russian", "slovene", "spanish", "swedish", "tajik", "tamil",
    "turkish", "uzbek",
)

#: Languages NLTK's Snowball stemmer reads. English uses the Porter stemmer.
SNOWBALL = frozenset({
    "arabic", "danish", "dutch", "finnish", "french", "german", "hungarian", "italian",
    "norwegian", "portuguese", "romanian", "russian", "spanish", "swedish",
})

#: Term frequency saturation.
DEFAULT_K1 = 1.5

#: Document length normalisation, 0 = none, 1 = full.
DEFAULT_B = 0.75

#: One word: a run of letters, digits or underscores.
WORD = re.compile(r"\w+")

#: Documents tokenised between two interrupt checks.
CHECK_EVERY = 256

_stopwords: dict[str, frozenset] = {}
_retriever_class = None


def stopwords(language: str) -> frozenset:
    """The stopword list for a language, read from disk without downloading.

    Args:
        language: One of :data:`LANGUAGES`.

    Returns:
        Lowercase words. Empty for ``none``.

    Raises:
        ValueError: The language is not one of :data:`LANGUAGES`.
        RuntimeError: No stopword data is installed for it.
    """
    if language == "none":
        return frozenset()
    if language not in LANGUAGES:
        raise ValueError(f"`{language}` is not a stopword language. Pick one of: {', '.join(LANGUAGES)}.")
    if language in _stopwords:
        return _stopwords[language]
    deps.require("nltk", feature=FEATURE)
    import nltk
    from nltk.corpus.reader import WordListCorpusReader

    bundled = _bundled()
    # NLTK opens corpora only under the folders on its search path.
    if bundled not in nltk.data.path:
        nltk.data.path.append(bundled)
    roots = [bundled] + [str(path) for path in nltk.data.path if str(path) != bundled]
    for root in roots:
        try:
            folder = nltk.data.find("corpora/stopwords", paths=[root])
        except LookupError:
            continue
        reader = WordListCorpusReader(folder, [language])
        try:
            words = frozenset(word.strip().lower() for word in reader.words(language) if word.strip())
        except (OSError, LookupError):
            continue
        _stopwords[language] = words
        return words
    raise RuntimeError(
        f"No {language} stopword list is installed. Set stopwords to `none`, or reinstall "
        f"llama-index-core, which carries the lists."
    )


def _bundled() -> str:
    """The NLTK data folder inside llama-index."""
    core = runtime.core()
    return os.path.join(os.path.dirname(os.path.abspath(core.__file__)), "_static", "nltk_cache")


def analyzer(language: str = "english", stemming: bool = True):
    """A function turning text into the terms BM25 counts.

    Args:
        language: The stopword list, one of :data:`LANGUAGES`. It also picks the stemmer.
        stemming: Whether words are reduced to their stems, ``running`` to ``run``.

    Returns:
        A callable taking text and returning a list of terms in order.
    """
    stop = stopwords(language)
    stem = _stemmer(language) if stemming else None
    stems: dict[str, str] = {}

    def analyse(text: str) -> list[str]:
        terms = []
        for word in WORD.findall(str(text or "").lower()):
            if word in stop:
                continue
            if stem is not None:
                found = stems.get(word)
                if found is None:
                    found = stems[word] = stem(word)
                word = found
            if word:
                terms.append(word)
        return terms

    return analyse


def _stemmer(language: str):
    """The stem function for a language: Snowball where it has one, Porter otherwise."""
    deps.require("nltk", feature=FEATURE)
    from nltk.stem import PorterStemmer, SnowballStemmer

    if language in SNOWBALL:
        return SnowballStemmer(language).stem
    return PorterStemmer().stem


class Okapi:
    """Okapi BM25 scores over a fixed set of tokenised documents.

    Attributes:
        size: Documents held.
        average: Mean document length in terms.
        vocabulary: Distinct terms across every document.
    """

    def __init__(self, documents: list[list[str]], k1: float = DEFAULT_K1, b: float = DEFAULT_B):
        """Count every document's terms.

        Args:
            documents: Each document's terms, as :func:`analyzer` returns them.
            k1: Term frequency saturation, ``0`` counts presence alone.
            b: Length normalisation, ``0`` = none, ``1`` = full.
        """
        import numpy as np

        self.k1 = float(k1)
        self.b = float(b)
        self.size = len(documents)
        lengths = np.array([len(terms) for terms in documents], dtype=np.float64)
        self.average = float(lengths.mean()) if self.size else 0.0
        scale = lengths / self.average if self.average > 0 else np.ones_like(lengths)
        self._norm = self.k1 * (1.0 - self.b + self.b * scale)
        gathered: dict[str, tuple[list[int], list[int]]] = {}
        for number, terms in enumerate(documents):
            for term, count in Counter(terms).items():
                rows, counts = gathered.setdefault(term, ([], []))
                rows.append(number)
                counts.append(count)
        self._postings = {
            term: (np.array(rows, dtype=np.int64), np.array(counts, dtype=np.float64))
            for term, (rows, counts) in gathered.items()
        }
        self.vocabulary = len(self._postings)

    def idf(self, term: str) -> float:
        """Inverse document frequency, ``log(1 + (N - df + 0.5) / (df + 0.5))``, 0 for an unseen term."""
        posting = self._postings.get(term)
        if posting is None:
            return 0.0
        df = len(posting[0])
        return math.log(1.0 + (self.size - df + 0.5) / (df + 0.5))

    def scores(self, terms: list[str]):
        """Every document's score for a query.

        Args:
            terms: The query's terms. A term given twice counts twice.

        Returns:
            A float64 numpy array, one score per document, 0 where no term matches.
        """
        import numpy as np

        scores = np.zeros(self.size, dtype=np.float64)
        for term, repeats in Counter(terms).items():
            posting = self._postings.get(term)
            if posting is None:
                continue
            rows, counts = posting
            weight = self.idf(term) * repeats
            scores[rows] += weight * counts * (self.k1 + 1.0) / (counts + self._norm[rows])
        return scores


def corpus(documents=None, index=None) -> list:
    """The chunks a BM25 retriever ranks, from a documents wire, an index wire or both.

    Args:
        documents: A :class:`~modules.llm.wires.Documents`, or ``None``.
        index: A :class:`~modules.llm.wires.Index`, whose docstore is read, or ``None``.

    Returns:
        llama-index nodes, each once, documents first. Chunks that only group smaller chunks
        and document summaries are left out.
    """
    runtime.core()
    from llama_index.core.schema import NodeRelationship

    nodes, seen = [], set()

    def keep(node) -> None:
        if node is None or node.node_id in seen:
            return
        if NodeRelationship.CHILD in (getattr(node, "relationships", None) or {}):
            return
        seen.add(node.node_id)
        nodes.append(node)

    for node in getattr(documents, "items", ()) or ():
        keep(node)
    if index is not None:
        built = index.index
        summaries = set(getattr(getattr(built, "index_struct", None), "summary_id_to_node_ids", {}) or {})
        docstore = getattr(built, "docstore", None)
        for node_id, node in (getattr(docstore, "docs", None) or {}).items():
            if node_id not in summaries:
                keep(node)
    return nodes


def retriever(nodes, top_k: int = 10, language: str = "english", stemming: bool = True,
              k1: float = DEFAULT_K1, b: float = DEFAULT_B):
    """A llama-index retriever ranking nodes by BM25 against the question.

    Args:
        nodes: The chunks to rank.
        top_k: The most chunks one retrieval returns.
        language: The stopword list, one of :data:`LANGUAGES`.
        stemming: Whether words are stemmed.
        k1: Term frequency saturation.
        b: Length normalisation.

    Returns:
        A ``BaseRetriever`` whose results are copies of the nodes, best first, each scored.
    """
    started = time.perf_counter()
    analyse = analyzer(language, stemming)
    terms = []
    for number, node in enumerate(nodes):
        if number % CHECK_EVERY == 0:
            runtime.check_interrupt()
        terms.append(analyse(node.get_content()))
    okapi = Okapi(terms, k1=k1, b=b)
    built = _class()(list(nodes), okapi, analyse, int(top_k))
    built.seconds = time.perf_counter() - started
    logger.info(
        "BM25 counted %d chunk(s), %d term(s), in %.2fs", okapi.size, okapi.vocabulary, built.seconds
    )
    return built


def _class():
    """The BM25 retriever class, defined on first use."""
    global _retriever_class
    if _retriever_class is not None:
        return _retriever_class
    runtime.core()
    from llama_index.core.base.base_retriever import BaseRetriever
    from llama_index.core.schema import NodeWithScore

    class BM25Retriever(BaseRetriever):
        """Ranks a fixed set of chunks by Okapi BM25."""

        def __init__(self, nodes, okapi, analyse, top_k):
            super().__init__()
            self._nodes = nodes
            self._okapi = okapi
            self._analyse = analyse
            self.top_k = top_k
            self.seconds = 0.0

        @property
        def okapi(self):
            """The scorer."""
            return self._okapi

        def _retrieve(self, query_bundle):
            import numpy as np

            terms = self._analyse(query_bundle.query_str)
            if not terms or not self._nodes:
                return []
            scores = self._okapi.scores(terms)
            count = min(self.top_k, len(scores))
            best = np.argpartition(-scores, count - 1)[:count] if count < len(scores) else np.arange(len(scores))
            best = best[np.argsort(-scores[best], kind="stable")]
            found = []
            for row in best:
                score = float(scores[row])
                if score <= 0.0:
                    break
                node = self._nodes[int(row)]
                copy = node.model_copy(update={"metadata": dict(node.metadata)})
                found.append(NodeWithScore(node=copy, score=score))
            return found

    _retriever_class = BM25Retriever
    return _retriever_class
