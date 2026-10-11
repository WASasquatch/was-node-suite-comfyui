"""Score two strings against each other and report the words behind the score."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat.types import NUMBER


class TextCompare(io.ComfyNode):
    """Compare two strings by edit distance and collect the words that drove the result."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Text Compare",
            display_name="Text Compare",
            search_aliases=["Text Compare", "similarity", "difference", "levenshtein", "diff"],
            category="WAS Suite/Text/Search",
            description=(
                "Compare two strings, passing both through unchanged with an exact-match "
                "flag, a score, and the words behind that score. `similarity` scores 1.0 for "
                "identical text, falling toward 0.0, and lists the shared words; `difference` "
                "lists the parts of text_a that changed, on its own scale that can pass 1.0."
            ),
            inputs=[
                io.String.Input(
                    "text_a",
                    multiline=True,
                    placeholder="Eg: a cat",
                    tooltip=(
                        "First text to compare; STRING. Empty boxes count as identical."
                    ),
                ),
                io.String.Input(
                    "text_b",
                    multiline=True,
                    placeholder="Eg: a dog",
                    tooltip=(
                        "Second text to compare; STRING. Both pass through unchanged on "
                        "TEXT_A_PASS and TEXT_B_PASS."
                    ),
                ),
                io.Combo.Input(
                    "mode",
                    options=["similarity", "difference"],
                    tooltip=(
                        "Which measure to report. `similarity` scores how alike the two texts "
                        "are and lists the words they share; `difference` lists the parts of "
                        "text_a that changed."
                    ),
                ),
                io.Float.Input(
                    "tolerance",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How loosely a word counts as shared in COMPARISON_TEXT, in "
                        "`similarity` mode only; no score changes. 0.0 to 0.99 = only words "
                        "in both texts; 1.0 = also words one character apart, so 'colour' "
                        "matches 'color'."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="TEXT_A_PASS",
                    tooltip=(
                        "text_a unchanged, so the node can sit in the middle of a prompt "
                        "chain instead of on a branch."
                    ),
                ),
                io.String.Output(
                    display_name="TEXT_B_PASS",
                    tooltip="text_b unchanged.",
                ),
                io.Boolean.Output(
                    display_name="BOOLEAN",
                    tooltip=(
                        "True only when the two texts are identical character for character. "
                        "Neither the mode nor the tolerance affects it."
                    ),
                ),
                NUMBER.Output(
                    display_name="SCORE_NUMBER",
                    tooltip=(
                        "The score, for the NUMBER inputs of the suite's own maths and logic "
                        "nodes. In `similarity` mode 1.0 means identical and 0.0 means nothing "
                        "in common; `difference` mode uses its own scale, which can go past "
                        "1.0."
                    ),
                ),
                io.String.Output(
                    display_name="COMPARISON_TEXT",
                    tooltip=(
                        "The words behind the comparison, space-separated. In `similarity` "
                        "mode the words the two texts have in common, and once tolerance is "
                        "raised the near matches from both. In `difference` mode the parts of "
                        "text_a that changed."
                    ),
                ),
                io.Float.Output(
                    display_name="SCORE_FLOAT",
                    tooltip="The same score as a decimal, for a core FLOAT input.",
                ),
                io.Int.Output(
                    display_name="SCORE_INT",
                    tooltip=(
                        "The score with everything after the decimal point dropped, so in "
                        "`similarity` mode it is 1 only for identical text and 0 for "
                        "everything else."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, text_a="", text_b="", mode="similarity", tolerance=0.0) -> io.NodeOutput:
        boolean = 1 if text_a == text_b else 0
        sim = cls.string_compare(text_a, text_b, tolerance, mode == "difference")
        score = float(sim[0])
        sim_result = " ".join(sim[1][::-1])
        sim_result = " ".join(sim_result.split())

        return io.NodeOutput(
            text_a, text_b, bool(boolean), score, sim_result, float(score), int(score)
        )

    @staticmethod
    def string_compare(str1, str2, threshold=1.0, difference_mode=False):
        """``(score, words)`` for one pair of strings.

        Args:
            str1: Left string; the words returned are taken from it.
            str2: Right string.
            threshold: Edit distance at or below which a cell counts as a match while the
                similarity table is filled. Higher values collect more words.
            difference_mode: Score and collect the differing words rather than the shared
                ones.

        Returns:
            ``(score, words)``. The score is 1 for identical strings and falls toward 0 as
            they diverge.

        Raises:
            ZeroDivisionError: ``difference_mode`` with both strings empty.
        """
        m = len(str1)
        n = len(str2)
        if difference_mode:
            dp = [[0 for x in range(n + 1)] for x in range(m + 1)]
            for i in range(m + 1):
                for j in range(n + 1):
                    if i == 0:
                        dp[i][j] = j
                    elif j == 0:
                        dp[i][j] = i
                    elif str1[i - 1] == str2[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1]
                    else:
                        dp[i][j] = 1 + min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1])
            diff_indices = []
            i, j = m, n
            while i > 0 and j > 0:
                if str1[i - 1] == str2[j - 1]:
                    i -= 1
                    j -= 1
                else:
                    diff_indices.append(i - 1)
                    i, j = min((i, j - 1), (i - 1, j))
            diff_indices.reverse()
            words = []
            start_idx = 0
            for i in diff_indices:
                if str1[i] == " ":
                    words.append(str1[start_idx:i])
                    start_idx = i + 1
            words.append(str1[start_idx:m])
            difference_score = 1 - ((dp[m][n] - len(words)) / max(m, n))
            return (difference_score, words[::-1])

        dp = [[0 for x in range(n + 1)] for x in range(m + 1)]
        similar_words = set()
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0:
                    dp[i][j] = j
                elif j == 0:
                    dp[i][j] = i
                elif str1[i - 1] == str2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                    if i > 1 and j > 1 and str1[i - 2] == " " and str2[j - 2] == " ":
                        word1_start = i - 2
                        word2_start = j - 2
                        while word1_start > 0 and str1[word1_start - 1] != " ":
                            word1_start -= 1
                        while word2_start > 0 and str2[word2_start - 1] != " ":
                            word2_start -= 1
                        word1 = str1[word1_start:i - 1]
                        word2 = str2[word2_start:j - 1]
                        if word1 in str2 or word2 in str1:
                            if word1 not in similar_words:
                                similar_words.add(word1)
                            if word2 not in similar_words:
                                similar_words.add(word2)
                else:
                    dp[i][j] = 1 + min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1])
                if dp[i][j] <= threshold and i > 0 and j > 0:
                    word1_start = max(0, i - dp[i][j])
                    word2_start = max(0, j - dp[i][j])
                    word1_end = i
                    word2_end = j
                    while word1_start > 0 and str1[word1_start - 1] != " ":
                        word1_start -= 1
                    while word2_start > 0 and str2[word2_start - 1] != " ":
                        word2_start -= 1
                    while word1_end < m and str1[word1_end] != " ":
                        word1_end += 1
                    while word2_end < n and str2[word2_end] != " ":
                        word2_end += 1
                    word1 = str1[word1_start:word1_end]
                    word2 = str2[word2_start:word2_end]
                    if word1 in str2 or word2 in str1:
                        if word1 not in similar_words:
                            similar_words.add(word1)
                        if word2 not in similar_words:
                            similar_words.add(word2)
        if max(m, n) == 0:
            similarity_score = 1
        else:
            similarity_score = 1 - (dp[m][n] / max(m, n))
        return (similarity_score, list(similar_words))
