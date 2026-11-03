"""Caption set measurements: length consistency, phrase frequency, subject coverage and flags.

Captions are strings, one per picture; ``None`` reads as empty. A subject matches as a whole
word with case ignored, and ``[trigger]`` matches literally.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter

from .tags import strip_emphasis

__all__ = [
    "FILLER",
    "FREQUENT_SHARE",
    "MIN_FREQUENT_CAPTIONS",
    "consistency",
    "coverage",
    "flags",
    "phrases",
]

#: Share of the captions a phrase has to pass to be listed as frequent.
FREQUENT_SHARE = 0.5

#: Fewest captions for which a phrase is listed as frequent.
MIN_FREQUENT_CAPTIONS = 5

#: Longest phrase counted, in words.
MAX_PHRASE_WORDS = 3

#: Fewest characters of a phrase of one word.
MIN_WORD_CHARS = 3

#: Fewest comma-separated pieces in a caption read as tags.
MIN_TAGS = 3

#: Most words a caption's pieces average for it to read as tags.
MAX_TAG_WORDS = 4

#: English words a counted phrase neither starts nor ends on: articles, pronouns, prepositions,
#: conjunctions, auxiliaries, quantifiers and the stock verbs and nouns of a written caption.
FILLER = frozenset("""
a an the this that these those such another other others same own
i me my mine myself we us our ours you your yours he him his himself she her hers herself
it its itself they them their theirs themselves one ones someone something anyone anything
who whom whose which what where when why how whoever whatever
and or but nor so yet if then than as because while although though whether either neither
of in on at to for with without from by into onto upon over under above below beneath
across along around about against among amid between through throughout during before after
behind beside besides beyond near nearby toward towards within inside outside off out up down
via per like unlike past
is are was were be been being am isn't aren't wasn't weren't
has have had having do does did doing done can could will would shall should may might must
not no yes very really quite rather too also just only even still almost nearly mostly
some any each every all both few many much more most less least several various lots lot
here there now again once ever never always often sometimes
two three four five six seven eight nine ten first second
image images photo photos photograph picture pictures shot frame scene
shows show showing shown depicts depict depicting depicted features feature featuring
appears appear appearing seen visible captured displayed presented
wearing wears worn wear holding holds hold looking looks look
""".split())

#: A run of whitespace, line breaks included.
_SPACE = re.compile(r"\s+")

#: A break between runs of words in a prose caption: sentence punctuation, brackets, slashes.
_BREAK = re.compile(r"[.,;:!?()\[\]{}\"“”|/\\]+")

#: One word: letters and digits, with apostrophes, hyphens and underscores inside it.
_WORD = re.compile(r"[^\W_](?:[\w'-]*[^\W_])?")

#: A bracketed placeholder inside a subject, such as ``[trigger]``.
_PLACEHOLDER = re.compile(r"\[[^\]]*\]")


def consistency(texts, min_words: int = 5, max_words: int = 0) -> dict:
    """Measure how even a set of captions is in length and how many repeat or fall short.

    Args:
        texts: The captions, one per picture. ``None`` and blank strings count as empty.
        min_words: A caption under this many words counts as short. 0 counts none.
        max_words: A caption over this many words counts as long. 0 counts none.

    Returns:
        ``captions`` (non-empty count), ``empty``, ``words_median``, ``words_spread``
        (standard deviation over the mean), ``short``, ``long``, ``identical`` (captions
        repeating an earlier one) and ``score`` from 0 to 1, 1 being even.
    """
    captions = [_stripped(text) for text in texts]
    written = [text for text in captions if text]
    words = [len(text.split()) for text in written]
    mean = statistics.fmean(words) if words else 0.0
    spread = round(statistics.pstdev(words) / mean, 3) if len(words) > 1 and mean else 0.0
    identical = sum(count - 1 for count in Counter(written).values())
    short = sum(count < min_words for count in words) if min_words > 0 else 0
    long = sum(count > max_words for count in words) if max_words > 0 else 0
    score = 0.0
    if written:
        evenness = 1.0 - min(1.0, spread) * 0.5
        clean = 1.0 - (identical + short) / len(written)
        score = round(max(0.0, min(1.0, evenness * clean)), 3)
    return {
        "captions": len(written),
        "empty": len(captions) - len(written),
        "words_median": float(statistics.median(words)) if words else 0.0,
        "words_spread": spread,
        "short": short,
        "long": long,
        "identical": identical,
        "score": score,
    }


def phrases(texts, subject: str = "", top: int = 40) -> dict:
    """Count the share of captions mentioning each phrase or tag, most mentioned first.

    Args:
        texts: The captions, one per picture. ``None`` and blank strings are skipped.
        subject: Phrases naming it are left out. Empty leaves every phrase in.
        top: Most rows returned. 0 or less returns every row.

    Returns:
        ``style`` (``tags`` or ``prose``; ``prose`` when there are no captions), ``rows`` as
        ``{"text", "count", "share"}`` for phrases in two or more captions, ``frequent`` (rows
        past :data:`FREQUENT_SHARE`, given :data:`MIN_FREQUENT_CAPTIONS` captions or more) and
        ``captions``, the non-empty count shares are taken over.
    """
    written = [text for text in (_stripped(text) for text in texts) if text]
    total = len(written)
    if not total:
        return {"style": "prose", "rows": [], "frequent": [], "captions": 0}
    tagged = _tag_style(written)
    needle = _needle(subject)
    names = _subject_words(subject)
    counts: Counter = Counter()
    for text in written:
        counts.update(_tags(text, needle) if tagged else _runs(text, needle))
    counts = Counter({text: count for text, count in counts.items() if count >= 2})
    dropped = {text for text in counts if names.intersection(text.split())}
    if not tagged:
        dropped |= _contained(counts, dropped)
    ordered = sorted(
        (text for text in counts if text not in dropped), key=lambda text: (-counts[text], text)
    )
    if top > 0:
        ordered = ordered[:top]
    rows = [
        {"text": text, "count": counts[text], "share": round(counts[text] / total, 3)}
        for text in ordered
    ]
    frequent = (
        [row for row in rows if row["count"] / total > FREQUENT_SHARE]
        if total >= MIN_FREQUENT_CAPTIONS else []
    )
    return {
        "style": "tags" if tagged else "prose",
        "rows": rows,
        "frequent": frequent,
        "captions": total,
    }


def coverage(texts, subject: str) -> int:
    """Count the captions that contain the subject.

    Args:
        texts: The captions, one per picture.
        subject: What to look for, case ignored, as a whole word. Empty counts nothing.

    Returns:
        How many captions contain it.
    """
    needle = _needle(subject)
    if not needle:
        return 0
    return sum(_mentions(_folded(_stripped(text)), needle) for text in texts)


def flags(texts, names, subject: str = "", min_words: int = 5, max_words: int = 0) -> list:
    """List every caption worth a second look, with the reasons.

    Args:
        texts: The captions, one per picture.
        names: A name per caption, by position. A missing or blank name is replaced by the
            caption's position, counted from 0.
        subject: A caption without it is flagged ``no subject``. Empty checks nothing.
        min_words: A caption under this many words is flagged ``short``. 0 checks nothing.
        max_words: A caption over this many words is flagged ``long``. 0 checks nothing.

    Returns:
        ``(name, reasons)`` pairs in caption order, reasons drawn from ``empty``, ``short``,
        ``long``, ``identical`` and ``no subject``.
    """
    labels = list(names or [])
    needle = _needle(subject)
    seen = set()
    out = []
    for index, raw in enumerate(texts):
        text = _stripped(raw)
        label = labels[index] if index < len(labels) else None
        label = str(label).strip() if label is not None and str(label).strip() else str(index)
        if not text:
            out.append((label, ["empty"]))
            continue
        count = len(text.split())
        reasons = []
        if min_words > 0 and count < min_words:
            reasons.append("short")
        if max_words > 0 and count > max_words:
            reasons.append("long")
        if text in seen:
            reasons.append("identical")
        seen.add(text)
        if needle and not _mentions(_folded(text), needle):
            reasons.append("no subject")
        if reasons:
            out.append((label, reasons))
    return out


def _stripped(text) -> str:
    """A caption as a string with surrounding blank space removed; ``None`` reads as empty."""
    return "" if text is None else str(text).strip()


def _folded(text: str) -> str:
    """Lower-cased text with curly apostrophes made straight and whitespace runs made one space."""
    return _SPACE.sub(" ", text.casefold().replace("’", "'")).strip()


def _needle(subject) -> str:
    """The subject as it is searched for, or an empty string for none."""
    return _folded(_stripped(subject))


def _is_word_char(char: str) -> bool:
    """Whether a character belongs to a word."""
    return char.isalnum() or char == "_"


def _matches(text: str, needle: str) -> list:
    """Every place a subject stands as a whole word in folded text.

    Args:
        text: Folded text.
        needle: The folded subject.

    Returns:
        ``(start, end)`` spans in order, none overlapping.
    """
    spans = []
    start = text.find(needle)
    head = _is_word_char(needle[0])
    tail = _is_word_char(needle[-1])
    while start >= 0:
        end = start + len(needle)
        before = start == 0 or not head or not _is_word_char(text[start - 1])
        after = end == len(text) or not tail or not _is_word_char(text[end])
        if before and after:
            spans.append((start, end))
            start = text.find(needle, end)
        else:
            start = text.find(needle, start + 1)
    return spans


def _mentions(text: str, needle: str) -> bool:
    """Whether folded text holds the subject as a whole word."""
    return bool(_matches(text, needle))


def _subject_words(subject) -> set:
    """The subject's own words, placeholders and filler words left out."""
    words = _WORD.findall(_PLACEHOLDER.sub(" ", _needle(subject)))
    return {word for word in words if word not in FILLER}


def _tag_style(texts: list) -> bool:
    """Whether most captions read as comma-separated tags rather than sentences."""
    tagged = 0
    for text in texts:
        pieces = [piece for piece in text.split(",") if piece.strip()]
        if len(pieces) >= MIN_TAGS:
            if sum(len(piece.split()) for piece in pieces) / len(pieces) <= MAX_TAG_WORDS:
                tagged += 1
    return tagged * 2 > len(texts)


def _tags(text: str, needle: str) -> set:
    """The distinct tags of one caption, folded, with emphasis and underscores removed.

    Args:
        text: One caption.
        needle: The folded subject; a tag holding it is left out. Empty leaves every tag in.

    Returns:
        Each tag once.
    """
    found = set()
    for piece in text.split(","):
        if needle and _mentions(_folded(piece), needle):
            continue
        tag = _folded(strip_emphasis(piece).replace("_", " ")).strip(" .")
        if tag:
            found.add(tag)
    return found


def _runs(text: str, needle: str) -> set:
    """The runs of one to three words in a prose caption that neither start nor end on filler.

    Args:
        text: One caption.
        needle: The folded subject, cut out of the text as a break. Empty cuts nothing.

    Returns:
        Each run once, its words joined by single spaces.
    """
    found = set()
    for line in text.splitlines():
        folded = _folded(line)
        pieces = []
        last = 0
        for start, end in _matches(folded, needle) if needle else []:
            pieces.append(folded[last:start])
            last = end
            # A possessive on the subject is cut with it.
            if folded.startswith("'s", end) and not _is_word_char(folded[end + 2:end + 3] or " "):
                last = end + 2
        pieces.append(folded[last:])
        for piece in pieces:
            for clause in _BREAK.split(piece):
                words = _WORD.findall(clause)
                for size in range(1, MAX_PHRASE_WORDS + 1):
                    for first in range(len(words) - size + 1):
                        run = words[first:first + size]
                        if run[0] in FILLER or run[-1] in FILLER:
                            continue
                        if size == 1 and len(run[0]) < MIN_WORD_CHARS:
                            continue
                        found.add(" ".join(run))
    return found


def _contained(counts: Counter, dropped: set) -> set:
    """The phrases that appear in no caption without a longer phrase around them.

    Args:
        counts: Captions mentioning each phrase.
        dropped: Phrases already left out, which hide nothing inside them.

    Returns:
        The phrases mentioned in exactly as many captions as some longer phrase containing them.
    """
    widest: Counter = Counter()
    for longer, count in counts.items():
        if longer in dropped:
            continue
        words = longer.split()
        for size in range(1, len(words)):
            for first in range(len(words) - size + 1):
                inner = " ".join(words[first:first + size])
                widest[inner] = max(widest[inner], count)
    return {text for text, count in counts.items() if widest[text] >= count}
