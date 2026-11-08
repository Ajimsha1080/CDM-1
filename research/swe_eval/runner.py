"""SWE-bench Lite Runner for CDM-1 Agent Architecture.

Executes official SWE-bench Lite benchmarks across four configurations:
1. llm_only
2. llm_tools
3. llm_tools_gate
4. llm_tools_gate_memory

Enforces strict provenance and evaluation rules:
- Verifies ANTHROPIC_API_KEY and Docker availability upfront.
- Halts and exits with code 1 if prerequisites are absent (zero simulated or fabricated results).
- Executes evaluations strictly via official `swebench` Docker harness.
- Records fine-grained per-task audit logs (steps, tokens, cost in USD, wall time, gate blocks, final outcome).
"""

import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import time
import json
import uuid
import logging
import argparse
import subprocess
from typing import Dict, Any, List, Optional, Tuple

from research.swe_eval.configs import (
    EvalConfig,
    EvalConfigName,
    CONFIG_DEFINITIONS,
    build_configured_loop,
)
from cdm.llm.base import LLMMessage
from cdm.llm.factory import get_llm_provider
from cdm.llm.anthropic_provider import AnthropicProvider

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cdm.swe_eval")


def check_prerequisites(require_api_key: bool = True, require_docker: bool = True) -> Tuple[bool, List[str]]:
    """Validates that real prerequisites (API Key and Docker) are present.
    Returns (is_ready, list_of_missing_reasons).
    """
    missing = []

    # 1. Check Anthropic API Key
    if require_api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            missing.append(
                "ANTHROPIC_API_KEY environment variable is NOT set. "
                "Real SWE-bench evaluations require a valid Anthropic API key to query Claude foundation models."
            )

    # 2. Check Docker Daemon
    if require_docker:
        docker_ok = False
        try:
            res = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=3.0,
                shell=False,
            )
            docker_ok = (res.returncode == 0)
        except Exception:
            docker_ok = False

        if not docker_ok:
            missing.append(
                "Docker daemon is NOT accessible or not running. "
                "SWE-bench Lite requires active Docker containers to build reproducible task environments "
                "and execute the official evaluation testbed."
            )

    return (len(missing) == 0, missing)


def load_swebench_tasks(dataset_name: str = "princeton-nlp/SWE-bench_Lite", limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Loads benchmark task instances using official swebench or datasets package."""
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, split="test")
        tasks = [dict(item) for item in ds]
        if limit and limit > 0:
            tasks = tasks[:limit]
        return tasks
    except Exception as e:
        logger.warning(f"Could not load tasks via datasets package: {e}")
        # Check if local task manifest exists
        cache_file = Path("research/swe_eval/tasks_cache.json")
        if cache_file.is_file():
            with open(cache_file, "r", encoding="utf-8") as f:
                tasks = json.load(f)
                if limit and limit > 0:
                    tasks = tasks[:limit]
                return tasks
        raise RuntimeError(
            f"Failed to load SWE-bench tasks from '{dataset_name}': {e}. "
            "Please ensure datasets or swebench is installed and internet connectivity is available."
        )


def evaluate_with_official_harness(
    predictions_file: Path,
    dataset_name: str = "princeton-nlp/SWE-bench_Lite",
    run_id: str = "cdm_eval",
) -> Dict[str, Any]:
    """Invokes the official swebench Docker evaluation harness.
    Strictly forbids writing custom pass/fail evaluators.
    """
    try:
        from swebench.harness.run_evaluation import main as swebench_eval_main
        # Run official swebench harness
        eval_args = [
            "--dataset_name", dataset_name,
            "--predictions_path", str(predictions_file),
            "--run_id", run_id,
            "--max_workers", "1",
        ]
        logger.info(f"Invoking official swebench evaluation harness: {eval_args}")
        # Execute official harness
        # swebench.harness evaluates containerized test runs and outputs grading JSON
        res = subprocess.run(
            [sys.executable, "-m", "swebench.harness.run_evaluation"] + eval_args,
            capture_output=True,
            text=True,
            timeout=1800.0,
        )
        return {
            "harness_exit_code": res.returncode,
            "stdout": res.stdout,
            "stderr": res.stderr,
        }
    except ImportError:
        raise RuntimeError(
            "The official 'swebench' package is required for evaluation. "
            "Install with: pip install swebench"
        )


def run_task_instance(
    task: Dict[str, Any],
    config: EvalConfig,
    model_name: str,
    output_dir: Path,
    run_idx: int = 1,
) -> Dict[str, Any]:
    """Runs a single task instance under the specified configuration and records audit metrics."""
    task_id = task.get("instance_id", "unknown_task")
    repo = task.get("repo", "")
    base_commit = task.get("base_commit", "")
    problem_statement = task.get("problem_statement", "")

    logger.info(f"--- Running Task [{task_id}] (Config: {config.name.value}, Run: {run_idx}) ---")
    start_time = time.time()

    task_workspace = output_dir / "workspaces" / f"{task_id}_{config.name.value}"
    os.makedirs(task_workspace, exist_ok=True)

    # 1. Initialize environment (clone / checkout repo inside workspace / container)
    # SWE-bench uses official docker containers; for agent tool operations we attach the sandbox_root
    task_log: Dict[str, Any] = {
        "task_id": task_id,
        "repo": repo,
        "base_commit": base_commit,
        "config": config.name.value,
        "run_index": run_idx,
        "model": model_name,
        "start_time": start_time,
        "steps": 0,
        "tokens": 0,
        "cost_usd": 0.0,
        "gate_blocks": 0,
        "human_approvals": 0,
        "patch": "",
        "resolved": False,
        "status": "in_progress",
    }

    try:
        if config.name == EvalConfigName.LLM_ONLY:
            # Single-shot prompt directly asking for unified diff
            prompt = (
                f"You are an expert software engineer resolving an issue in repository '{repo}'.\n"
                f"Base Commit: {base_commit}\n\n"
                f"Problem Statement:\n{problem_statement}\n\n"
                "Provide a unified git diff patch resolving the issue. Output only the git patch enclosed in ```diff ... ``` blocks."
            )
            provider = get_llm_provider()
            llm_resp = provider.complete(
                messages=[LLMMessage(role="user", content=prompt)],
                tenant_id=f"swe_{task_id}",
            )
            raw = llm_resp.raw_text or ""
            patch = ""
            if "```diff" in raw:
                patch = raw.split("```diff")[1].split("```")[0].strip()
            elif "```" in raw:
                patch = raw.split("```")[1].split("```")[0].strip()
            else:
                patch = raw.strip()

            task_log["steps"] = 1
            task_log["tokens"] = llm_resp.input_tokens + llm_resp.output_tokens
            task_log["cost_usd"] = llm_resp.cost_usd
            task_log["patch"] = patch
            task_log["status"] = "completed"

        else:
            # Loop-based agent evaluation
            loop = build_configured_loop(
                config=config,
                sandbox_root=str(task_workspace),
            )
            goal = f"Fix issue {task_id} in {repo}"
            run_state = loop.create_run(
                goal=goal,
                task_description=problem_statement,
                tenant_id=f"swe_{task_id}",
                step_budget=config.step_budget,
                cost_budget_usd=config.cost_budget_usd,
                max_context_tokens=config.max_context_tokens,
            )

            # Run agent loop
            run_state = loop.run_until_pause_or_finish(run_state.run_id, max_steps=config.max_steps)

            # Collect metrics from run_state
            task_log["steps"] = run_state.step_count
            task_log["tokens"] = run_state.accumulated_tokens
            task_log["cost_usd"] = run_state.accumulated_cost_usd
            task_log["gate_blocks"] = sum(
                1 for h in run_state.history if h.get("gate_decision") in ("block", "blocked_loop")
            )
            task_log["human_approvals"] = sum(
                1 for h in run_state.history if h.get("gate_decision") == "human_approved"
            )

            # Extract generated patch
            git_diff_tool = GitDiffTool(sandbox_root=str(task_workspace))
            diff_res = git_diff_tool.execute({})
            task_log["patch"] = diff_res.output.get("diff", "") if diff_res.success else ""
            task_log["status"] = run_state.status

    except Exception as e:
        logger.error(f"Error executing task {task_id}: {e}", exc_info=True)
        task_log["status"] = "error"
        task_log["error"] = str(e)

    task_log["wall_time_seconds"] = round(time.time() - start_time, 2)
    return task_log


def print_summary_table(results: List[Dict[str, Any]], config_name: str) -> None:
    """Prints formatted summary table of benchmark results."""
    print("\n" + "=" * 95)
    print(f" SWE-BENCH LITE RESULTS SUMMARY — Config: {config_name}")
    print("=" * 95)
    print(f"{'Task ID':<25} | {'Steps':<6} | {'Tokens':<8} | {'Cost ($)':<8} | {'Time (s)':<8} | {'Blocks':<6} | {'Status':<12}")
    print("-" * 95)

    total_tokens = 0
    total_cost = 0.0
    total_time = 0.0
    total_blocks = 0
    resolved_count = 0

    for r in results:
        task_id = r.get("task_id", "")[:24]
        steps = r.get("steps", 0)
        tokens = r.get("tokens", 0)
        cost = r.get("cost_usd", 0.0)
        t_sec = r.get("wall_time_seconds", 0.0)
        blocks = r.get("gate_blocks", 0)
        status = r.get("status", "unknown")
        resolved = r.get("resolved", False)

        total_tokens += tokens
        total_cost += cost
        total_time += t_sec
        total_blocks += blocks
        if resolved:
            resolved_count += 1

        print(f"{task_id:<25} | {steps:<6} | {tokens:<8} | {cost:<8.4f} | {t_sec:<8.1f} | {blocks:<6} | {status:<12}")

    print("-" * 95)
    n = len(results) if results else 1
    print(f"Total Tasks Evaluated : {len(results)}")
    print(f"Resolved Count / Rate : {resolved_count} / {len(results)} ({resolved_count / n * 100:.1f}%)")
    print(f"Total Accrued Cost    : ${total_cost:.4f}")
    print(f"Avg Cost per Task     : ${total_cost / n:.4f}")
    print(f"Avg Wall Time per Task: {total_time / n:.1f}s")
    print(f"Total Gate Blocks     : {total_blocks}")
    print("=" * 95 + "\n")


def estimate_full_run_cost(avg_cost_per_task: float, avg_time_per_task: float, task_count: int = 300) -> None:
    """Prints cost and time estimates for full 300-task SWE-bench Lite suite."""
    est_cost = avg_cost_per_task * task_count
    est_hours = (avg_time_per_task * task_count) / 3600.0
    print("\n" + "#" * 60)
    print(" FULL 300-TASK SWE-BENCH LITE PROJECTION")
    print("#" * 60)
    print(f"Projected Total Tasks : {task_count}")
    print(f"Estimated Total Cost  : ${est_cost:.2f} USD")
    print(f"Estimated Total Time  : {est_hours:.2f} hours (serial)")
    print("#" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="SWE-bench Lite Evaluation Runner for CDM-1")
    parser.add_argument("--limit", type=int, default=10, help="Number of benchmark tasks to run (e.g. 10)")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per task (default: 1)")
    parser.add_argument("--model", type=str, default="claude-3-5-sonnet-20241022", help="Model identifier")
    parser.add_argument(
        "--config",
        type=str,
        default="llm_tools_gate_memory",
        choices=["llm_only", "llm_tools", "llm_tools_gate", "llm_tools_gate_memory"],
        help="Evaluation configuration variant",
    )
    parser.add_argument("--output-dir", type=str, default="research/swe_eval/results", help="Directory for logs and patches")
    parser.add_argument("--dataset", type=str, default="princeton-nlp/SWE-bench_Lite", help="SWE-bench dataset name")
    parser.add_argument("--require-prerequisites", action="store_true", default=True, help="Enforce real API key and Docker")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" CDM-1: SWE-BENCH LITE EVALUATION HARNESS")
    print(f" Model  : {args.model}")
    print(f" Config : {args.config}")
    print(f" Limit  : {args.limit} tasks (runs per task: {args.runs})")
    print("=" * 70 + "\n")

    # Step 1: Strict Prerequisites Check
    is_ready, missing_reasons = check_prerequisites(require_api_key=True, require_docker=True)
    if not is_ready:
        print("\n" + "!" * 70)
        print(" ERROR: Cannot proceed with real SWE-bench evaluation.")
        print(" Missing required environment prerequisites:")
        for idx, reason in enumerate(missing_reasons, 1):
            print(f"   [{idx}] {reason}")
        print("\n CDM-1 Evaluation Policy:")
        print(" - Fabricated, simulated, or hand-written benchmark numbers are strictly forbidden.")
        print(" - To perform real evaluations, set ANTHROPIC_API_KEY and start Docker Desktop / daemon.")
        print("!" * 70 + "\n")
        sys.exit(1)

    # Step 2: Load Tasks
    tasks = load_swebench_tasks(dataset_name=args.dataset, limit=args.limit)
    out_dir = Path(args.output_dir)
    os.makedirs(out_dir / args.config, exist_ok=True)

    cfg = CONFIG_DEFINITIONS[args.config]
    results = []

    for task in tasks:
        for r_idx in range(1, args.runs + 1):
            log = run_task_instance(
                task=task,
                config=cfg,
                model_name=args.model,
                output_dir=out_dir,
                run_idx=r_idx,
            )
            results.append(log)
            # Write per-task log
            log_path = out_dir / args.config / f"{log['task_id']}_run{r_idx}.json"
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(log, f, indent=2)

    # Step 3: Grade predictions using official swebench harness
    predictions_file = out_dir / args.config / "all_predictions.jsonl"
    with open(predictions_file, "w", encoding="utf-8") as f:
        for res in results:
            entry = {
                "instance_id": res["task_id"],
                "model_name_or_path": args.model,
                "model_patch": res["patch"],
            }
            f.write(json.dumps(entry) + "\n")

    logger.info(f"Wrote predictions to {predictions_file}. Running official harness evaluation...")
    eval_res = evaluate_with_official_harness(predictions_file, dataset_name=args.dataset)
    logger.info(f"Official evaluation complete: {eval_res}")

    # Step 4: Summary Table
    print_summary_table(results, args.config)


if __name__ == "__main__":
    main()
