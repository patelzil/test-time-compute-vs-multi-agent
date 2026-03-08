"""Multi-agent strategy: 3-agent GPT-5.4 pipeline (planner → analyzer → patcher).

Uses the same model as the single-model strategy but with no extended reasoning,
split across 3 specialized agents. This isolates the variable: same model,
different inference strategy.
"""

import time
from openai import OpenAI

MODEL = "gpt-5.4"
MAX_OUTPUT_TOKENS = 4096

PLANNER_SYSTEM = """You are a senior software engineer triaging a bug report.
Given the bug report and repository name, output a structured diagnosis:
1. Root cause hypothesis
2. Affected files and components
3. Proposed fix strategy (high-level)
Be concise and precise. No code yet — just the plan."""

ANALYZER_SYSTEM = """You are a code analysis specialist.
Given a bug report and a planner's diagnosis, output a detailed code-level analysis:
1. Specific functions/methods that need changes
2. Line-level description of what to modify
3. Edge cases to handle
4. Potential regression risks
Be specific enough that a patch writer can produce the diff without ambiguity."""

PATCHER_SYSTEM = """You are an expert patch writer.
Given a bug report, diagnosis, and code analysis, generate a unified diff patch.

Rules:
- Output ONLY the unified diff patch, nothing else
- Use standard unified diff format (--- a/file, +++ b/file, @@ hunks)
- Be precise — only change what's necessary
- Do not add unrelated changes"""


def _call(client: OpenAI, system: str, user_msg: str) -> tuple[str, float, int]:
    """Make a single GPT-5.4 call with no extended reasoning. Returns (content, latency, tokens)."""
    start = time.time()
    resp = client.responses.create(
        model=MODEL,
        instructions=system,
        input=user_msg,
        reasoning={"effort": "none"},
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    content = resp.output_text
    tokens = resp.usage.total_tokens if resp.usage else 0
    return content, round(time.time() - start, 2), tokens


def run(problem_statement: str, repo: str, hints: str) -> dict:
    """Run the 3-agent pipeline on a single SWE-bench instance."""
    client = OpenAI()
    agent_latencies = []
    total_tokens = 0
    start = time.time()

    base_context = f"Repository: {repo}\n\n## Bug Report\n{problem_statement}"
    if hints:
        base_context += f"\n\n## Additional Context\n{hints}"

    try:
        # Agent 1: Planner
        plan, lat, tok = _call(client, PLANNER_SYSTEM, base_context)
        agent_latencies.append(lat)
        total_tokens += tok

        # Agent 2: Analyzer
        analyzer_input = f"{base_context}\n\n## Planner Diagnosis\n{plan}"
        analysis, lat, tok = _call(client, ANALYZER_SYSTEM, analyzer_input)
        agent_latencies.append(lat)
        total_tokens += tok

        # Agent 3: Patcher
        patcher_input = f"{base_context}\n\n## Diagnosis\n{plan}\n\n## Code Analysis\n{analysis}"
        patch, lat, tok = _call(client, PATCHER_SYSTEM, patcher_input)
        agent_latencies.append(lat)
        total_tokens += tok

        return {
            "patch": patch.strip(),
            "latency_s": round(time.time() - start, 2),
            "tokens_used": total_tokens,
            "agent_latencies": agent_latencies,
            "error": None,
        }
    except Exception as e:
        return {
            "patch": "",
            "latency_s": round(time.time() - start, 2),
            "tokens_used": total_tokens,
            "agent_latencies": agent_latencies,
            "error": str(e),
        }
