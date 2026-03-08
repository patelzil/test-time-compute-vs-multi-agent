"""Single-model strategy: GPT-5.4 with extreme reasoning (test-time compute)."""

import time
from openai import OpenAI

SYSTEM_PROMPT = """You are an expert software engineer. You will be given a bug report from a real GitHub issue.
Your task is to generate a unified diff patch that fixes the bug.

Rules:
- Output ONLY the unified diff patch, nothing else
- Use the standard unified diff format (--- a/file, +++ b/file, @@ hunks)
- Be precise — only change what's necessary to fix the bug
- Do not add unrelated changes, comments, or refactoring"""


def run(problem_statement: str, repo: str, hints: str) -> dict:
    """Run GPT-5.4 with max reasoning effort on a single SWE-bench instance."""
    client = OpenAI()
    user_msg = f"Repository: {repo}\n\n## Bug Report\n{problem_statement}"
    if hints:
        user_msg += f"\n\n## Additional Context\n{hints}"

    start = time.time()
    try:
        resp = client.responses.create(
            model="gpt-5.4",
            instructions=SYSTEM_PROMPT,
            input=user_msg,
            reasoning={"effort": "high"},
            max_output_tokens=16384,
        )
        patch = resp.output_text.strip()
        tokens = resp.usage.total_tokens if resp.usage else 0
        return {
            "patch": patch,
            "latency_s": round(time.time() - start, 2),
            "tokens_used": tokens,
            "error": None,
        }
    except Exception as e:
        return {
            "patch": "",
            "latency_s": round(time.time() - start, 2),
            "tokens_used": 0,
            "error": str(e),
        }
