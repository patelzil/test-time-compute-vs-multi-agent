# Test-Time Compute vs Multi-Agent Orchestration

A minimal comparison harness that tests the contrarian hypothesis: **one model thinking longer often beats a multi-agent pipeline**.

## The Experiment

| Strategy | Description | Model |
|----------|-------------|-------|
| **Single Model** | One API call with maximum reasoning effort | GPT-5.4 (`reasoning.effort=high`) |
| **Multi-Agent Pipeline** | 3 sequential specialized agents: Planner → Analyzer → Patcher | GPT-5.4 (`reasoning.effort=none`) |

**Same model, different inference strategy** — this isolates the variable. The single-model approach spends its compute budget on deeper thinking; the multi-agent approach spreads it across specialized roles.

Both strategies receive identical inputs from 9 real GitHub bugs ([SWE-bench_Lite](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite)) and must produce a unified diff patch.

## Why This Matters

The default instinct in AI engineering is to decompose hard problems into multi-agent pipelines. But inference-time scaling — letting one model dynamically allocate thinking where needed — may outperform pre-allocated specialization for many tasks.

**Mental model**: Pre-training compute bought general intelligence once. Inference compute buys task-specific reasoning every time. Multi-agent architectures pre-allocate that budget across roles; test-time compute lets one model allocate thinking dynamically.

## Results

**Single-model reasoning wins 7 out of 9 tasks.**

### Head-to-Head

| Bug | Repo | Single Model | Multi-Agent | Winner |
|-----|------|:------------:|:-----------:|:------:|
| `astropy-12907` | astropy | **0.545** · 32s | 0.017 · 72s | Single |
| `django-11039` | django | **0.133** · 191s | 0.079 · 30s | Single |
| `django-13230` | django | **0.067** · 112s | 0.010 · 62s | Single |
| `matplotlib-23314` | matplotlib | **0.750** · 79s | 0.116 · 44s | Single |
| `pylint-7993` | pylint | 0.000 · 284s | **0.037** · 54s | Multi |
| `pytest-5692` | pytest | **0.125** · 483s | 0.086 · 49s | Single |
| `scikit-learn-13496` | scikit-learn | **0.412** · 164s | 0.092 · 54s | Single |
| `sympy-13146` | sympy | 0.000 · 236s | **0.015** · 72s | Multi |
| `sympy-20442` | sympy | **0.182** · 166s | 0.100 · 59s | Single |

*Scores are diff similarity (0–1) against the gold patch. Higher = closer to the correct fix.*

### Summary

| Metric | Single Model | Multi-Agent |
|--------|:------------:|:-----------:|
| **Avg diff similarity** | **0.246** | 0.061 |
| **Correct files targeted** | **7 / 9** (78%) | 0 / 9 (0%) |
| **Wins** | **7** | 2 |
| Avg latency | 194s | 55s |
| Avg tokens | 10,661 | 8,786 |

The single-model approach produces **4× higher patch similarity** and targets the correct file 78% of the time vs 0% for multi-agent. The multi-agent pipeline is ~3.5× faster but consistently edits the wrong files (it tends to also generate test files the gold patch doesn't include).

The two multi-agent wins (`pylint-7993`, `sympy-13146`) are the cases where the single model produced an empty patch or targeted the wrong file entirely — suggesting that for the hardest bugs, the single model's deeper reasoning can lead it down a wrong path with no recovery, while the multi-agent pipeline at least produces *something*.

> Reproduce with `python harness.py`. Results save to `results/`.

## Failure Mode Analysis

### Single-model failure modes
- Over-reasoning: patches that change too much
- Format drift: model ignores diff format instructions under heavy reasoning
- Timeout risk at high effort on complex bugs

### Multi-agent failure modes
- **Context degradation**: Each agent only sees a summary of the previous agent's work
- **Error compounding**: Planner misdiagnosis cascades through analyzer and patcher
- **Serialization overhead**: 3 sequential API calls with cumulative latency
- **Coordination gaps**: No shared state — the patcher can't ask the planner to clarify

## Quick Start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Add your API keys to .env
cp .env.example .env
# Edit .env with your keys

python harness.py             # Full run (9 tasks)
python harness.py --tasks 3   # Quick test
```

## Project Structure

```
harness.py              # Main entry point — loads tasks, runs strategies, prints results
strategies/
  single_model.py       # GPT-5.4 extreme reasoning (one call, max thinking)
  multi_agent.py        # 3-agent GPT-5.4 pipeline, no reasoning (planner → analyzer → patcher)
evaluate.py             # Patch scoring (exact match, diff similarity, file targeting)
results/                # JSON output from each run
```

## Limitations

- **No execution-based evaluation**: We compare patches against gold diffs rather than running test suites in Docker (which is how the full SWE-bench harness works). This means we may undercount valid alternative patches.
- **9 tasks**: Small sample size — useful for demonstrating methodology, not for drawing statistical conclusions.
- **No repo context**: Neither strategy sees the actual codebase, only the bug report. Real SWE-bench agents typically have access to the full repository.

## Key Takeaway

When multi-agent wins, it's because the task genuinely benefits from decomposition (parallel work, heterogeneous model strengths, human-in-the-loop checkpoints). When it loses, it's because the overhead of serialization, context loss, and error compounding exceeds the benefit of specialization.

The right question isn't "single model vs multi-agent" — it's "where does the inference budget create the most value for this specific task?"
