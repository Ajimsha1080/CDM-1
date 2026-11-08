# SWE-bench Lite Evaluation Harness for CDM-1

This directory provides the real, reproducible SWE-bench Lite evaluation runner for the **CDM-1 Agent Product**.

---

## Architecture & Evaluation Configurations

The benchmark runner tests four distinct architectural configurations:

| Configuration | Layer 1 (LLM) | Layer 2 (Cognitive Memory) | Layer 3 (Policy Gate) | Layer 4 (Engineering Tools) | Description |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`llm_only`** | Yes | No | No | No | Baseline: direct one-shot git patch generation from problem statement without loop or tools. |
| **`llm_tools`** | Yes | No | No | Yes | Autonomous tool-use loop without CDM policy gate or memory writeback. |
| **`llm_tools_gate`** | Yes | No | Yes | Yes | Tool loop constrained by CDM Policy Gate (risk thresholds and action-loop prevention). |
| **`llm_tools_gate_memory`** | Yes | Yes | Yes | Yes | Full 4-layer CDM-1: Tool loop + Policy Gate + Episodic Memory retrieval & World State writeback. |

---

## Prerequisites

Real evaluation requires containerized environments and foundation model API access. The harness strictly prohibits fabricated or mocked results:

1. **Docker Engine**: A running Docker daemon is mandatory. Repository checkouts and the official SWE-bench evaluation harness run strictly inside isolated containers.
2. **Anthropic API Key**: `ANTHROPIC_API_KEY` environment variable must be exported.
3. **Official Dependencies**:
   ```bash
   pip install swebench datasets
   ```

If either `ANTHROPIC_API_KEY` or `Docker` is missing, the runner halts immediately with exit code 1.

---

## Execution Commands

### 1. Baseline: Direct LLM Patch Generation
```bash
python research/swe_eval/runner.py --limit 10 --config llm_only --model claude-3-5-sonnet-20241022
```

### 2. LLM + Sandboxed Engineering Tools
```bash
python research/swe_eval/runner.py --limit 10 --config llm_tools --model claude-3-5-sonnet-20241022
```

### 3. LLM + Tools + CDM Policy Gate
```bash
python research/swe_eval/runner.py --limit 10 --config llm_tools_gate --model claude-3-5-sonnet-20241022
```

### 4. Full CDM-1 Architecture (LLM + Tools + Gate + Episodic Memory)
```bash
python research/swe_eval/runner.py --limit 10 --config llm_tools_gate_memory --model claude-3-5-sonnet-20241022
```

---

## Benchmark Results (Real Runs Only)

In accordance with CDM-1 release integrity rules:
> *Never claim a result unless you ran it and can show the real output. Results come from real CI runs only.*

| Configuration | Tasks | Resolved | Resolve Rate (%) | Avg Cost / Task ($) | Avg Wall Time (s) | Gate Blocks |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `llm_only` | 10 | *Pending CI Run* | *Pending* | ~$0.027 | ~15s | 0 |
| `llm_tools` | 10 | *Pending CI Run* | *Pending* | ~$0.250 | ~120s | 0 |
| `llm_tools_gate` | 10 | *Pending CI Run* | *Pending* | ~$0.210 | ~110s | *Tracked in CI* |
| `llm_tools_gate_memory` | 10 | *Pending CI Run* | *Pending* | ~$0.310 | ~145s | *Tracked in CI* |

*Note: If `llm_tools_gate` or `llm_tools_gate_memory` does not beat `llm_only` on resolution rate in a given run, report that outcome plainly.*

---

## Full 300-Task SWE-bench Lite Cost & Time Projections

Projections based on Claude 3.5 Sonnet pricing ($3.00 / 1M prompt tokens, $15.00 / 1M completion tokens):

| Configuration | Avg Tokens / Task | Projected Cost / Task | Total 300-Task Cost | Serial Wall Time |
| :--- | :---: | :---: | :---: | :---: |
| **`llm_only`** | ~5,000 | ~$0.027 | **~$8.10 USD** | ~1.25 hours |
| **`llm_tools`** | ~28,000 | ~$0.250 | **~$75.00 USD** | ~10.0 hours |
| **`llm_tools_gate`** | ~24,000 | ~$0.210 | **~$63.00 USD** | ~9.2 hours |
| **`llm_tools_gate_memory`** | ~35,000 | ~$0.310 | **~$93.00 USD** | ~12.1 hours |

---

## Output Artifacts

For each run, audit logs are saved under:
```
research/swe_eval/results/<config>/<task_id>_run<K>.json
research/swe_eval/results/<config>/all_predictions.jsonl
```
Each JSON log preserves:
- Exact task identifier and git base commit
- Step-by-step tool invocation sequence
- Token count and accrued USD cost
- Policy gate intercept decisions (blocks vs approvals)
- Final unified git diff patch
- Official `swebench` evaluation harness verdict
