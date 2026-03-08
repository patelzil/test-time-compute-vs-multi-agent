"""Patch evaluation: compare generated patches against SWE-bench gold patches."""

import re


def _extract_files(patch: str) -> set[str]:
    """Extract file paths from a unified diff."""
    return set(re.findall(r"^[+-]{3} [ab]/(.+)$", patch, re.MULTILINE))


def _normalize(patch: str) -> str:
    """Normalize whitespace for comparison."""
    return "\n".join(line.rstrip() for line in patch.strip().splitlines())


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance (line-level) between two strings."""
    a_lines, b_lines = a.splitlines(), b.splitlines()
    m, n = len(a_lines), len(b_lines)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a_lines[i - 1] == b_lines[j - 1] else 1 + min(dp[j], dp[j - 1], prev)
            prev = temp
    return dp[n]


def score(generated: str, gold: str) -> dict:
    """Score a generated patch against the gold patch.

    Returns dict with exact_match, diff_similarity (0-1), and files_correct.
    """
    gen_norm = _normalize(generated)
    gold_norm = _normalize(gold)

    exact = gen_norm == gold_norm

    max_len = max(len(gen_norm.splitlines()), len(gold_norm.splitlines()), 1)
    dist = _edit_distance(gen_norm, gold_norm)
    similarity = round(1 - dist / max_len, 3)

    gen_files = _extract_files(generated)
    gold_files = _extract_files(gold)
    files_correct = bool(gen_files and gen_files == gold_files)

    return {
        "exact_match": exact,
        "diff_similarity": max(similarity, 0.0),
        "files_correct": files_correct,
        "gold_files": sorted(gold_files),
        "generated_files": sorted(gen_files),
    }
