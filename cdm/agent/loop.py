"""CDM-1 Control Loop: 4-Layer Agent Product Architecture.
Layer 1: External / Local LLM
Layer 2: CDM Cognitive Engine (context, memory, world state, reasoning, prediction)
Layer 3: Decision & Planning Engine (goals, constraints, risk, policy gate)
Layer 4: Execution & Feedback (sandboxed tools, observations, outcome verification, learning)
Loop: Execution results flow back into Layer 2 (memory & world state) after every action.
Runs are strictly resumable from persistent storage.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import os
import time
import json
import uuid
import sqlite3
import threading
import logging
from pydantic import BaseModel, Field

from cdm.llm.base import BaseLLMProvider, LLMMessage, LLMResponse
from cdm.llm.factory import get_llm_provider
from cdm.cognitive.engine import CognitiveEngine
from cdm.cognitive.state import WorldState, WorldEntity, GoalNode, Plan
from cdm.gate.policy_gate import PolicyGate, ProposedAction, GateDecisionType, GateDecision
from cdm.execution.tools.base import BaseTool, ToolResult
from cdm.execution.tools.shell_tool import SandboxedShellTool
from cdm.execution.tools.file_tool import FileReadTool, FileWriteTool, FileEditTool
from cdm.execution.tools.http_tool import HttpCallTool
from cdm.execution.tools.engineering_tools import (
    RepoSearchTool,
    EnhancedFileReadTool,
    EnhancedFileEditTool,
    FileCreateTool,
    RunTestsTool,
    GitDiffTool,
    GitStatusTool,
    ListDirectoryTool,
)
from cdm.execution.verifier import OutcomeVerifier, VerificationSpec, VerificationResult
from cdm.execution.audit import AuditTrailManager

logger = logging.getLogger("cdm.agent.loop")


class AgentActionProposal(BaseModel):
    tool: str = Field(..., description="Name of tool to execute or 'complete_task'")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Arguments for tool")
    reason: str = Field(..., description="Reasoning and intent behind proposed action")
    claim_goal_met: bool = Field(default=False, description="Whether LLM believes the task goal is fully achieved")


class AgentRunState(BaseModel):
    run_id: str
    tenant_id: str
    session_id: str
    goal: str
    task_description: str
    status: str = "running" # 'running', 'waiting_for_approval', 'completed', 'failed', 'rejected', 'step_budget_exceeded', 'cost_budget_exceeded', 'loop_detected', 'inconclusive', 'paused'
    step_count: int = 0
    step_budget: int = 10
    cost_budget_usd: float = 1.0
    accumulated_cost_usd: float = 0.0
    accumulated_tokens: int = 0
    max_context_tokens: int = 8000
    tool_call_count: int = 0
    history: List[Dict[str, Any]] = Field(default_factory=list)
    pending_action: Optional[Dict[str, Any]] = None
    pending_reason: Optional[str] = None
    verification_spec: Optional[Dict[str, Any]] = None
    verification_status: Optional[str] = None
    tenant_policies: List[str] = Field(default_factory=list)
    prohibited_actions: List[str] = Field(default_factory=list)
    active_plan: Optional[Dict[str, Any]] = None
    current_plan_step: int = 0
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class AgentStore:
    """Thread-safe persistent SQLite storage for resumable agent runs."""

    def __init__(self, db_path: str = "experiments/agent_runs.sqlite"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS agent_runs (
                        run_id TEXT PRIMARY KEY,
                        tenant_id TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        goal TEXT NOT NULL,
                        state_json TEXT NOT NULL,
                        updated_at REAL NOT NULL
                    );
                    """
                )
                conn.commit()
            finally:
                conn.close()

    def save_run(self, run: AgentRunState) -> None:
        run.updated_at = time.time()
        payload = json.dumps(run.model_dump())
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO agent_runs (run_id, tenant_id, session_id, status, goal, state_json, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    (run.run_id, run.tenant_id, run.session_id, run.status, run.goal, payload, run.updated_at),
                )
                conn.commit()
            finally:
                conn.close()

    def load_run(self, run_id: str) -> Optional[AgentRunState]:
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                cur.execute("SELECT state_json FROM agent_runs WHERE run_id = ?;", (run_id,))
                row = cur.fetchone()
                if not row:
                    return None
                data = json.loads(row[0])
                return AgentRunState.model_validate(data)
            finally:
                conn.close()


class AgentControlLoop:
    """CDM-1 Four-Layer Agent Control Loop."""

    def __init__(
        self,
        llm_provider: Optional[BaseLLMProvider] = None,
        cognitive_engine: Optional[CognitiveEngine] = None,
        policy_gate: Optional[PolicyGate] = None,
        tools: Optional[Dict[str, BaseTool]] = None,
        store: Optional[AgentStore] = None,
        audit_manager: Optional[AuditTrailManager] = None,
        sandbox_root: Optional[str] = None,
    ):
        self.sandbox_root = sandbox_root or os.getcwd()
        self.llm = llm_provider or get_llm_provider()
        self.cognitive_engine = cognitive_engine or CognitiveEngine()
        self.policy_gate = policy_gate or PolicyGate()
        self.store = store or AgentStore()
        self.audit = audit_manager or AuditTrailManager()

        # Register default execution tools (including engineering tools)
        self.tools: Dict[str, BaseTool] = tools or {
            "read_file": EnhancedFileReadTool(sandbox_root=self.sandbox_root),
            "write_file": FileWriteTool(sandbox_root=self.sandbox_root),
            "create_file": FileCreateTool(sandbox_root=self.sandbox_root),
            "edit_file": EnhancedFileEditTool(sandbox_root=self.sandbox_root),
            "repo_search": RepoSearchTool(sandbox_root=self.sandbox_root),
            "run_tests": RunTestsTool(sandbox_root=self.sandbox_root),
            "git_diff": GitDiffTool(sandbox_root=self.sandbox_root),
            "git_status": GitStatusTool(sandbox_root=self.sandbox_root),
            "list_directory": ListDirectoryTool(sandbox_root=self.sandbox_root),
            "shell_exec": SandboxedShellTool(working_dir=self.sandbox_root),
            "http_request": HttpCallTool(),
        }

    def register_tool(self, tool: BaseTool) -> None:
        self.tools[tool.name] = tool

    def _build_history_context(self, run: AgentRunState) -> str:
        """Constructs a token-bounded history context, summarizing older steps when exceeding budget."""
        if not run.history:
            return "No actions taken yet."

        # Keep last 4 steps detailed, summarize earlier steps concisely
        keep_recent = 4
        earlier = run.history[:-keep_recent] if len(run.history) > keep_recent else []
        recent = run.history[-keep_recent:] if len(run.history) > keep_recent else run.history

        parts = []
        if earlier:
            parts.append(f"--- Compact Summary of Earlier Steps 1 to {earlier[-1].get('step')} ({len(earlier)} actions) ---")
            for h in earlier:
                tool = h.get("tool", "unknown")
                gate = h.get("gate_decision", "allow")
                succ = "SUCCESS" if h.get("success") else ("BLOCKED" if "block" in str(gate) else "FAIL")
                out_snippet = str(h.get("output") or h.get("error") or "")[:100].replace("\n", " ")
                parts.append(f"- Step {h.get('step')}: [{tool}] -> {succ} ({out_snippet})")
            parts.append("--- End of Compact Summary ---\n")

        parts.append("Recent Execution Steps:")
        for h in recent:
            out_snippet = str(h.get("output") or h.get("error") or "")[:300].replace("\n", " ")
            parts.append(
                f"Step {h.get('step')}: Action '{h.get('tool')}' (args: {h.get('arguments', {})}) -> "
                f"Gate: {h.get('gate_decision')} | Success: {h.get('success')} | Outcome: {out_snippet}"
            )
        return "\n".join(parts)

    def _is_action_loop(self, history: List[Dict[str, Any]], proposed_tool: str, proposed_args: Dict[str, Any]) -> bool:
        """Detects whether identical tool and arguments have been repeated without progress (loop detection)."""
        if len(history) < 2:
            return False
        # If the last 2 completed actions match proposed tool and identical arguments:
        norm_proposed_args = json.dumps(proposed_args or {}, sort_keys=True)
        for h in history[-2:]:
            if h.get("tool") != proposed_tool:
                return False
            if json.dumps(h.get("arguments") or {}, sort_keys=True) != norm_proposed_args:
                return False
        return True

    def resume_run(
        self,
        run_id: str,
        additional_steps: Optional[int] = None,
        additional_cost_budget: Optional[float] = None,
        resume_loop: bool = True,
    ) -> AgentRunState:
        """Resumes an interrupted, paused, or budget-exhausted run from persistent storage."""
        run = self.store.load_run(run_id)
        if not run:
            raise ValueError(f"Agent run '{run_id}' not found.")

        if additional_steps is not None and additional_steps > 0:
            run.step_budget += additional_steps
        if additional_cost_budget is not None and additional_cost_budget > 0:
            run.cost_budget_usd += additional_cost_budget

        if run.status in ("step_budget_exceeded", "cost_budget_exceeded", "paused", "loop_detected"):
            run.status = "running"

        self.store.save_run(run)

        if resume_loop and run.status == "running":
            return self.run_until_pause_or_finish(run_id)
        return run

    def pause_run(self, run_id: str) -> AgentRunState:
        """Pauses a running agent run and persists its state."""
        run = self.store.load_run(run_id)
        if not run:
            raise ValueError(f"Agent run '{run_id}' not found.")
        run.status = "paused"
        self.store.save_run(run)
        return run

    def create_run(
        self,
        goal: str,
        task_description: str,
        tenant_id: str = "default_tenant",
        step_budget: int = 10,
        cost_budget_usd: float = 1.0,
        max_context_tokens: int = 8000,
        tenant_policies: Optional[List[str]] = None,
        prohibited_actions: Optional[List[str]] = None,
        verification_spec: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> AgentRunState:
        """Initializes a new persistent agent run."""
        run_id = f"run_{uuid.uuid4().hex}"
        sess_id = session_id or f"sess_{uuid.uuid4().hex[:8]}"

        state = AgentRunState(
            run_id=run_id,
            tenant_id=tenant_id,
            session_id=sess_id,
            goal=goal,
            task_description=task_description,
            step_budget=step_budget,
            cost_budget_usd=cost_budget_usd,
            max_context_tokens=max_context_tokens,
            tenant_policies=tenant_policies or [],
            prohibited_actions=prohibited_actions or [],
            verification_spec=verification_spec,
        )
        self.store.save_run(state)
        return state

    def step(self, run_id: str) -> AgentRunState:
        """Executes a single step of the control loop."""
        run = self.store.load_run(run_id)
        if not run:
            raise ValueError(f"Agent run '{run_id}' not found.")

        if run.status != "running":
            return run

        # 1. Budget checks
        if run.step_count >= run.step_budget:
            run.status = "step_budget_exceeded"
            self.store.save_run(run)
            return run

        if run.accumulated_cost_usd >= run.cost_budget_usd:
            run.status = "cost_budget_exceeded"
            self.store.save_run(run)
            return run

        # Metacognitive recurring failure check
        consecutive_failures = 0
        for h in reversed(run.history):
            if h.get("success") is False:
                consecutive_failures += 1
            else:
                break
        if consecutive_failures >= 3:
            step_num = run.step_count + 1
            logger.warning(f"Run {run.run_id}: Metacognitive recurring failure threshold exceeded ({consecutive_failures} failures).")
            run.status = "waiting_for_approval"
            run.pending_reason = f"Metacognitive intervention: Systemic failure detected ({consecutive_failures} consecutive action failures). Requiring human review before continuing."
            run.history.append({
                "step": step_num,
                "tool": "metacognitive_intervention",
                "gate_decision": "suspend_systemic_failure",
                "success": False,
                "error": f"Metacognitive intervention: {consecutive_failures} consecutive execution failures detected. Execution suspended for human review.",
            })
            run.step_count += 1
            self.store.save_run(run)
            return run

        # =====================================================================
        # Layer 2: Authoritative Unified Cognitive Processing
        # =====================================================================
        candidate_tools = list(self.tools.keys()) + ["complete_task"]
        execute_planning = (run.active_plan is None)

        cog_result = self.cognitive_engine.process(
            text=f"Task: {run.task_description}. Goal: {run.goal}",
            structured_features={"step": run.step_count + 1, "cost_usd": run.accumulated_cost_usd, "accumulated_cost": run.accumulated_cost_usd},
            candidates=candidate_tools[:8],
            prohibited_actions=run.prohibited_actions,
            task="agent_control_step",
            goals=[run.goal] if run.goal else None,
            constraints=run.tenant_policies + run.prohibited_actions,
            session_id=run.session_id,
            tenant_id=run.tenant_id,
            execute_planning=execute_planning,
        )

        if cog_result.abstain and cog_result.abstention_reason:
            step_num = run.step_count + 1
            logger.warning(f"Run {run.run_id}: Cognitive engine abstention: {cog_result.abstention_reason}")
            run.status = "waiting_for_approval"
            run.pending_reason = cog_result.abstention_reason
            gate_dec = "fail_closed_ranking_error" if "Decision core failure" in str(cog_result.abstention_reason) else "abstain"
            run.history.append({
                "step": step_num,
                "tool": "abstain",
                "arguments": {},
                "gate_decision": gate_dec,
                "success": False,
                "error": f"Cognitive runtime failed closed: {cog_result.abstention_reason}",
            })
            run.step_count += 1
            self.store.save_run(run)
            return run

        perception = cog_result.shared_state.perception if cog_result.shared_state else self.cognitive_engine.perception_engine.process(text=run.task_description)
        context = cog_result.shared_state.context if cog_result.shared_state else None
        world_model = self.cognitive_engine.get_isolated_world_model(run.tenant_id, run.session_id)
        working_memory = self.cognitive_engine.get_isolated_working_memory(run.tenant_id, run.session_id)
        working_state = cog_result.shared_state.working_memory if cog_result.shared_state else working_memory.snapshot()
        world_state = cog_result.shared_state.world_state if cog_result.shared_state else world_model.get_state()
        reasoning_trace = cog_result.reasoning_evidence
        causal_consequences = cog_result.causal_analysis or {}
        candidate_scores = cog_result.candidate_scores or {}
        pred_risk = cog_result.risk or 0.05
        cdm_prov = cog_result.metadata.get("provenance")
        cdm_model_id = cog_result.metadata.get("model_id", "CDM-1-General")
        cdm_model_version = cog_result.metadata.get("model_version", "1.0.0")
        semantic_rules = cog_result.shared_state.semantic_rules if cog_result.shared_state else []
        relevant_episodes = cog_result.shared_state.episodic_context if cog_result.shared_state else []

        # Sequential Planning Engine invocation & Active Plan Tracking
        if not run.active_plan and cog_result.selected_plan and cog_result.selected_plan.steps:
            run.active_plan = cog_result.selected_plan.model_dump()
            run.current_plan_step = 0
            active_plan = cog_result.selected_plan
        elif run.active_plan:
            active_plan = Plan.model_validate(run.active_plan)
            curr_step_idx = run.current_plan_step
            if curr_step_idx < len(active_plan.steps):
                curr_step = active_plan.steps[curr_step_idx]
                preconds, preconds_met = self.cognitive_engine.planning_engine._evaluate_preconditions(
                    action=curr_step.action_name,
                    current_world_state=world_state,
                    forbidden_actions=set(run.prohibited_actions),
                )
                if not preconds_met:
                    logger.info(f"Run {run.run_id}: Preconditions for step '{curr_step.action_name}' invalid. Replanning...")
                    active_plan = self.cognitive_engine.planning_engine.generate_plan(
                        target_goal=working_state.active_goals[0] if working_state.active_goals else GoalNode(goal_id="g1", description=run.goal),
                        available_actions=candidate_tools,
                        current_world_state=world_state,
                        reasoning_trace=reasoning_trace,
                        prohibited_actions=run.prohibited_actions,
                        tenant_id=run.tenant_id,
                        task="replan",
                    )
                    if active_plan and active_plan.steps:
                        run.active_plan = active_plan.model_dump()
                        run.current_plan_step = 0
        else:
            active_plan = None

        # Build prompt context
        memory_notes = []
        for ep in relevant_episodes:
            memory_notes.append(f"- Past action '{getattr(ep, 'action_taken', 'action')}' produced outcome '{getattr(ep, 'observed_outcome', '')}' (reward: {getattr(ep, 'reward_signal', 0.0)})")

        rule_notes = [f"- Policy Rule: {r.get('rule_id', '')} ({r.get('condition_description', '')})" for r in semantic_rules[:3]]
        deduction_notes = [f"- Deduction: {s.conclusion}" for s in reasoning_trace.steps[:3]]
        plan_note = f"Recommended Plan: Step {active_plan.steps[run.current_plan_step].step_number} -> {active_plan.steps[run.current_plan_step].action_name}" if (active_plan and active_plan.steps and run.current_plan_step < len(active_plan.steps)) else None

        tools_desc = "\n".join([f"- {name}: {t.description}" for name, t in self.tools.items()])

        system_prompt = (
            "You are an autonomous AI Agent operating within the CDM-1 Control Loop.\n"
            f"Available tools:\n{tools_desc}\n"
            "- complete_task: Declare that the task goal is fully achieved.\n\n"
            "Output RFC-8259 JSON strictly adhering to schema:\n"
            "{\n"
            '  "tool": "tool_name_here",\n'
            '  "arguments": { ... },\n'
            '  "reason": "Detailed justification",\n'
            '  "claim_goal_met": false\n'
            "}"
        )

        history_context = self._build_history_context(run)

        user_prompt_parts = [
            f"Task Goal: {run.goal}",
            f"Description: {run.task_description}",
            f"Step Count: {run.step_count + 1} / {run.step_budget}",
        ]
        if perception.extracted_entities:
            user_prompt_parts.append("Entities Detected: " + ", ".join([f"{e.name} ({e.entity_type})" for e in perception.extracted_entities]))
        if rule_notes:
            user_prompt_parts.append("Active Semantic Rules:\n" + "\n".join(rule_notes))
        if deduction_notes:
            user_prompt_parts.append("Reasoning Deductions:\n" + "\n".join(deduction_notes))
        if plan_note:
            user_prompt_parts.append("Cognitive Planner Recommendation: " + plan_note)
        if memory_notes:
            user_prompt_parts.append("Context from Episodic Memory:\n" + "\n".join(memory_notes))

        user_prompt_parts.append(history_context)
        user_prompt_parts.append("Propose the next optimal action.")
        user_prompt = "\n\n".join(user_prompt_parts)

        # =====================================================================
        # Layer 1: Foundation Model Proposal with Automatic Retries
        # =====================================================================
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]
        max_retries = 2
        attempt = 0
        proposed = None
        last_error = None
        current_messages = list(messages)
        last_llm_resp = None

        while attempt <= max_retries:
            attempt += 1
            llm_resp = self.llm.complete(messages=current_messages, schema=AgentActionProposal, tenant_id=run.tenant_id)
            last_llm_resp = llm_resp
            run.accumulated_tokens += (llm_resp.input_tokens + llm_resp.output_tokens)
            run.accumulated_cost_usd += llm_resp.cost_usd

            if llm_resp.parsed_json:
                try:
                    proposed = AgentActionProposal.model_validate(llm_resp.parsed_json)
                    break
                except Exception as e:
                    last_error = f"Schema validation error: {e}"
            else:
                raw_text = (llm_resp.raw_text or "").strip()
                try:
                    if "```" in raw_text:
                        block = raw_text.split("```")[1]
                        if block.startswith("json"):
                            block = block[4:]
                        raw_text = block.strip()
                    parsed = json.loads(raw_text)
                    proposed = AgentActionProposal.model_validate(parsed)
                    break
                except Exception as e:
                    last_error = f"JSON parse error: {e}"

            if attempt <= max_retries:
                logger.warning(f"Malformed LLM response on attempt {attempt}: {last_error}. Retrying...")
                current_messages.append(LLMMessage(role="assistant", content=(llm_resp.raw_text or "")[:400]))
                current_messages.append(LLMMessage(
                    role="user",
                    content=f"Your previous output was invalid ({last_error}). Output strictly valid RFC-8259 JSON matching AgentActionProposal.",
                ))

        if proposed is None:
            # Phase 6: Safe LLM response handling. Never fall back to shell_exec!
            step_num = run.step_count + 1
            run.status = "failed"
            run.history.append({
                "step": step_num,
                "tool": "abstain",
                "gate_decision": "abstain_malformed_llm_output",
                "success": False,
                "error": f"LLM output could not be validated after {max_retries} retries: {last_error}. Agent safely abstained from arbitrary tool execution.",
            })
            run.step_count += 1
            self.store.save_run(run)
            return run

        # Check PolicyGate on proposed action first to catch policy blocks (dangerous patterns, prohibited actions)
        action_item = ProposedAction(tool=proposed.tool, args=proposed.arguments, reason=proposed.reason)
        gate_res = self.policy_gate.evaluate(
            action=action_item,
            tenant_id=run.tenant_id,
            world_state=world_state,
            tenant_policies=run.tenant_policies,
            prohibited_actions=run.prohibited_actions,
        )

        step_num = run.step_count + 1

        if gate_res.decision == GateDecisionType.BLOCK:
            logger.warning(f"Run {run.run_id}: Step {step_num} BLOCKED by Policy Gate: {gate_res.reason}")
            self.audit.record_step(
                run_id=run.run_id,
                step_number=step_num,
                tenant_id=run.tenant_id,
                proposed_action={"tool": proposed.tool, "arguments": proposed.arguments, "reason": proposed.reason},
                gate_decision=gate_res.model_dump(),
                execution_result=None,
                verification_result={"verified": False, "reason": "Blocked by policy gate"},
            )
            run.history.append({
                "step": step_num,
                "tool": proposed.tool,
                "arguments": proposed.arguments,
                "gate_decision": "block",
                "success": False,
                "error": f"Policy Gate Blocked Action: {gate_res.reason}",
            })
            run.step_count += 1
            self.store.save_run(run)
            return run

        # =====================================================================
        # Phase 1 & 2: Task-Relevant Candidate Generation & Pre-Ranking Validation
        # =====================================================================
        planned_action_name = None
        plan_step_params = {}
        if run.active_plan:
            plan_obj = Plan.model_validate(run.active_plan)
            if run.current_plan_step < len(plan_obj.steps):
                planned_step_obj = plan_obj.steps[run.current_plan_step]
                planned_action_name = planned_step_obj.action_name
                plan_step_params = planned_step_obj.parameters or {}

        # 1. Task-relevant candidate generation (replaces indiscriminate all-tool ranking)
        raw_candidates = []
        raw_candidates.append({
            "tool": proposed.tool,
            "arguments": proposed.arguments or {},
            "reason": proposed.reason,
            "source": "proposed",
        })

        if planned_action_name and planned_action_name != proposed.tool:
            clean_plan_args = {k: v for k, v in plan_step_params.items() if k != "target_goal"}
            raw_candidates.append({
                "tool": planned_action_name,
                "arguments": clean_plan_args,
                "reason": f"Executing planned step {run.current_plan_step + 1} ({planned_action_name}) per validated sequential plan.",
                "source": "plan",
            })

        task_text = f"{run.goal} {run.task_description}".lower()
        if "search" in task_text or "find" in task_text or "grep" in task_text:
            if "repo_search" in self.tools and all(c["tool"] != "repo_search" for c in raw_candidates):
                query_val = (
                    proposed.arguments.get("path")
                    or proposed.arguments.get("query")
                    or (run.goal if "search" in run.goal.lower() else run.task_description)
                )
                raw_candidates.append({
                    "tool": "repo_search",
                    "arguments": {"query": str(query_val)},
                    "reason": "Task-relevant repository search candidate derived from task intent.",
                    "source": "task_relevance",
                })

        # 2. Pre-ranking validation: Tool registration, authorization, arguments, schema, and preconditions
        eligible_candidates = []
        validation_rejections = {}
        for cand in raw_candidates:
            c_tool = cand["tool"]
            c_args = cand["arguments"]
            c_reason = cand.get("reason", "")

            # A. Registration validation
            if c_tool not in self.tools and c_tool != "complete_task":
                validation_rejections[c_tool] = ("rejected_unregistered_tool", f"Tool '{c_tool}' is not in the registered tool interface.")
                continue

            # B. Authorization validation (prohibited actions & policy gate)
            if c_tool in run.prohibited_actions:
                validation_rejections[c_tool] = ("block", f"Tool '{c_tool}' is prohibited by tenant execution policy.")
                continue

            action_item = ProposedAction(tool=c_tool, args=c_args, reason=c_reason)
            gate_eval = self.policy_gate.evaluate(
                action=action_item,
                tenant_id=run.tenant_id,
                world_state=world_state,
                tenant_policies=run.tenant_policies,
                prohibited_actions=run.prohibited_actions,
            )
            if gate_eval.decision == GateDecisionType.BLOCK:
                validation_rejections[c_tool] = ("block", f"Policy Gate Blocked Action: {gate_eval.reason}")
                continue

            # C. Arguments & Schema validation (candidate must have valid arguments)
            if c_tool in self.tools and c_tool != "complete_task":
                tool_obj = self.tools[c_tool]
                schema_err = tool_obj.validate_args(c_args) if hasattr(tool_obj, "validate_args") else None
                if schema_err:
                    validation_rejections[c_tool] = ("rejected_schema_validation", f"Tool argument schema validation failed: {schema_err}")
                    continue

            # D. Dynamic world state preconditions validation
            if c_tool in self.tools and c_tool != "complete_task":
                preconds, preconds_met = self.cognitive_engine.planning_engine._evaluate_preconditions(
                    action=c_tool,
                    current_world_state=world_state,
                    forbidden_actions=set(run.prohibited_actions),
                )
                if not preconds_met:
                    validation_rejections[c_tool] = ("rejected_world_state_preconditions", f"Action '{c_tool}' rejected: preconditions not satisfied under current world state metrics.")
                    continue

            eligible_candidates.append(cand)

        core_obj = getattr(self.cognitive_engine, "decision_core", None)
        is_neural = bool(core_obj is not None and getattr(core_obj, "is_fallback", False) is not True)
        cand_scores = cog_result.candidate_scores or {}
        heur_scores = getattr(cog_result, "heuristic_candidate_scores", {}) or {}

        # 3. Abstain when no valid candidate exists
        if not eligible_candidates:
            step_num = run.step_count + 1
            if proposed.tool in validation_rejections:
                gate_decision, err_msg = validation_rejections[proposed.tool]
            else:
                gate_decision = "abstain_no_valid_candidates"
                err_msg = "No valid candidate actions satisfy tool registration, authorization, argument schema, and dynamic world state preconditions."

            logger.warning(f"Run {run.run_id}: Step {step_num} candidate validation rejected all options: {err_msg}")
            dec_prov = {
                "candidate_identifier": proposed.tool,
                "candidate_validation_result": gate_decision,
                "decision_model_version": cdm_model_version,
                "used_neural_inference": is_neural,
                "neural_score": None,
                "heuristic_score": None,
                "risk_estimate": round(pred_risk, 4),
                "confidence": round(cog_result.confidence, 4) if cog_result.confidence is not None else None,
                "policy_result": gate_decision,
                "final_selected_action": None,
                "reason_for_rejection_or_abstention": err_msg,
            }
            run.history.append({
                "step": step_num,
                "tool": proposed.tool,
                "arguments": proposed.arguments,
                "gate_decision": gate_decision,
                "success": False,
                "error": err_msg,
                "model_version": cdm_model_version,
                "decision_provenance": dec_prov,
            })
            run.step_count += 1
            run.status = "waiting_for_approval"
            run.pending_reason = err_msg

            if working_state.active_goals and gate_decision in ("rejected_schema_validation", "rejected_world_state_preconditions"):
                replan_candidates = [a for a in candidate_tools if a != proposed.tool]
                replan_res = self.cognitive_engine.planning_engine.generate_plan(
                    target_goal=working_state.active_goals[0],
                    available_actions=replan_candidates if replan_candidates else candidate_tools,
                    current_world_state=world_state,
                    reasoning_trace=reasoning_trace,
                    prohibited_actions=run.prohibited_actions,
                    tenant_id=run.tenant_id,
                    task="replan_after_candidate_validation_failure",
                )
                if replan_res and replan_res.steps:
                    run.active_plan = replan_res.model_dump()
                    run.current_plan_step = 0
            self.store.save_run(run)
            return run

        # 4. Authoritative Decision Ranking (uses single authoritative result from cog_result, ZERO duplicate inference)
        alternatives_evaluated = []
        selected_tool_name = eligible_candidates[0]["tool"]
        selected_tool_args = eligible_candidates[0]["arguments"]
        selected_reason = eligible_candidates[0]["reason"]
        
        # Determine base candidate scores
        c0_name = selected_tool_name
        raw_c0_neural = cand_scores.get(c0_name)
        if is_neural and raw_c0_neural is not None:
            selected_neural_score = float(raw_c0_neural)
            selected_heuristic_score = None
            used_neural_inference = True
        else:
            selected_neural_score = None
            raw_c0_heur = heur_scores.get(c0_name)
            selected_heuristic_score = float(raw_c0_heur) if raw_c0_heur is not None else 0.0
            used_neural_inference = False

        if len(eligible_candidates) > 1:
            try:
                ranked_candidates = []
                for cand in eligible_candidates:
                    c_name = cand["tool"]
                    raw_neural = cand_scores.get(c_name)
                    if is_neural and raw_neural is not None:
                        c_neural = float(raw_neural)
                        c_heuristic = None
                        c_used_neural = True
                        c_effective_score = c_neural
                    else:
                        c_neural = None  # Explicitly None: missing model scores are never labeled with artificial defaults
                        raw_heur = heur_scores.get(c_name)
                        c_heuristic = float(raw_heur) if raw_heur is not None else 0.0
                        c_used_neural = False
                        c_effective_score = c_heuristic

                    c_risk = pred_risk
                    if c_name in causal_consequences and isinstance(causal_consequences[c_name], dict):
                        c_risk = max(c_risk, causal_consequences[c_name].get("expected_risk", 0.0))

                    # Plan alignment bonus: prioritize the planned step of validated plan
                    c_plan_bonus = 0.35 if (planned_action_name and c_name == planned_action_name) else 0.0
                    c_proposed_bonus = 0.05 if (c_name == proposed.tool) else 0.0
                    composite_score = (c_effective_score * (cog_result.confidence or 0.85)) - (c_risk * 0.4) + c_plan_bonus + c_proposed_bonus
                    ranked_candidates.append((composite_score, cand, c_neural, c_heuristic, c_risk, c_used_neural))
                    alternatives_evaluated.append({
                        "action": c_name,
                        "composite_score": round(composite_score, 4),
                        "used_neural_inference": c_used_neural,
                        "neural_score": round(c_neural, 4) if c_neural is not None else None,
                        "heuristic_score": round(c_heuristic, 4) if c_heuristic is not None else None,
                        "risk": round(c_risk, 4),
                        "in_active_plan": (c_name == planned_action_name),
                        "candidate_validation_result": "valid",
                    })

                ranked_candidates.sort(key=lambda x: x[0], reverse=True)
                top_composite, top_cand, top_neural, top_heur, top_c_risk, top_used_neural = ranked_candidates[0]
                selected_tool_name = top_cand["tool"]
                selected_tool_args = top_cand["arguments"]
                selected_reason = top_cand["reason"]
                selected_neural_score = top_neural
                selected_heuristic_score = top_heur
                used_neural_inference = top_used_neural
            except Exception as e:
                logger.error(f"Candidate ranking exception: {e}")
                step_num = run.step_count + 1
                fail_prov = {
                    "candidate_identifier": proposed.tool,
                    "candidate_validation_result": "fail_closed_ranking_error",
                    "decision_model_version": cdm_model_version,
                    "used_neural_inference": False,
                    "neural_score": None,
                    "heuristic_score": None,
                    "risk_estimate": round(pred_risk, 4),
                    "confidence": round(cog_result.confidence, 4) if cog_result.confidence is not None else None,
                    "policy_result": "fail_closed_ranking_error",
                    "final_selected_action": None,
                    "reason_for_rejection_or_abstention": f"Decision ranking failed closed: {e}",
                }
                run.history.append({
                    "step": step_num,
                    "tool": proposed.tool,
                    "arguments": proposed.arguments,
                    "gate_decision": "fail_closed_ranking_error",
                    "success": False,
                    "error": f"Decision ranking failed closed: {e}",
                    "alternatives_evaluated": alternatives_evaluated,
                    "model_version": cdm_model_version,
                    "decision_provenance": fail_prov,
                })
                run.status = "waiting_for_approval"
                run.pending_reason = f"Candidate ranking error: {e}"
                run.step_count += 1
                self.store.save_run(run)
                return run

        # Check if tool is complete_task (Phase 3)
        if selected_tool_name == "complete_task":
            # Independent ground truth verification
            if run.verification_spec:
                v_spec = VerificationSpec.model_validate(run.verification_spec)
                v_res = OutcomeVerifier.verify(spec=v_spec, sandbox_root=self.sandbox_root)
                if not v_res.verified:
                    # Refute LLM's claim of completion!
                    run.history.append({
                        "step": run.step_count + 1,
                        "tool": "complete_task",
                        "gate_decision": "rejected",
                        "success": False,
                        "error": f"Task verification failed: {v_res.failure_reason}",
                        "provenance": v_res.provenance,
                    })
                    run.step_count += 1
                    # Trigger replanning after verification failure
                    if working_state.active_goals:
                        replan_res = self.cognitive_engine.planning_engine.generate_plan(
                            target_goal=working_state.active_goals[0],
                            available_actions=candidate_tools,
                            current_world_state=world_state,
                            reasoning_trace=reasoning_trace,
                            prohibited_actions=run.prohibited_actions,
                            tenant_id=run.tenant_id,
                            task="replan_after_verification_failure",
                        )
                        if replan_res and replan_res.steps:
                            run.active_plan = replan_res.model_dump()
                            run.current_plan_step = 0
                    self.store.save_run(run)
                    return run

                # Confirmed goal achievement!
                run.status = "completed"
                run.verification_status = "verified_complete"
                if hasattr(self.cognitive_engine, "feedback_collector"):
                    self.cognitive_engine.feedback_collector.record_feedback(
                        decision_id=f"dec_{run.run_id}_goal",
                        predicted_decision={"tool": "complete_task", "claim_goal_met": True},
                        predicted_outcome={"expected_success": True},
                        actual_outcome={"goal_verified": True, "details": v_res.details},
                        is_success=True,
                        tenant_id=run.tenant_id,
                        review_status="verified",
                    )
                run.history.append({
                    "step": run.step_count + 1,
                    "tool": "complete_task",
                    "gate_decision": "allow",
                    "success": True,
                    "output": "Task goal verified and completed.",
                    "provenance": v_res.provenance,
                })
                run.step_count += 1
                self.store.save_run(run)
                return run
            else:
                run.status = "completed"
                run.verification_status = "unverified"
                run.history.append({
                    "step": run.step_count + 1,
                    "tool": "complete_task",
                    "gate_decision": "allow",
                    "success": True,
                    "output": "Task completed without independent verification (status: unverified).",
                })
                run.step_count += 1
                self.store.save_run(run)
                return run

        # Loop Detection: intercept repeated identical actions
        if self._is_action_loop(run.history, selected_tool_name, selected_tool_args):
            step_num = run.step_count + 1
            logger.warning(f"Run {run.run_id}: Step {step_num} loop detected for action '{selected_tool_name}'. Halting repeated action.")
            run.history.append({
                "step": step_num,
                "tool": selected_tool_name,
                "arguments": selected_tool_args,
                "gate_decision": "blocked_loop",
                "success": False,
                "error": f"Action loop detected: action '{selected_tool_name}' with identical arguments repeated 3 times without progress.",
            })
            run.step_count += 1
            loop_count = sum(1 for h in run.history[-2:] if h.get("gate_decision") == "blocked_loop")
            if loop_count >= 2:
                run.status = "loop_detected"
            self.store.save_run(run)
            return run

        # =====================================================================
        # Layer 3: Policy Gate & Risk Evaluation
        # =====================================================================
        action_item = ProposedAction(tool=selected_tool_name, args=selected_tool_args, reason=selected_reason)
        gate_res = self.policy_gate.evaluate(
            action=action_item,
            tenant_id=run.tenant_id,
            world_state=world_state,
            tenant_policies=run.tenant_policies,
            prohibited_actions=run.prohibited_actions,
        )

        step_num = run.step_count + 1

        if gate_res.decision == GateDecisionType.BLOCK:
            logger.warning(f"Run {run.run_id}: Step {step_num} BLOCKED by Policy Gate: {gate_res.reason}")
            self.audit.record_step(
                run_id=run.run_id,
                step_number=step_num,
                tenant_id=run.tenant_id,
                proposed_action={"tool": selected_tool_name, "arguments": selected_tool_args, "reason": selected_reason},
                gate_decision=gate_res.model_dump(),
                execution_result=None,
                verification_result={"verified": False, "reason": "Blocked by policy gate"},
            )
            block_prov = {
                "candidate_identifier": selected_tool_name,
                "candidate_validation_result": "blocked_by_policy",
                "decision_model_version": cdm_model_version,
                "used_neural_inference": used_neural_inference,
                "neural_score": selected_neural_score,
                "heuristic_score": selected_heuristic_score,
                "risk_estimate": round(pred_risk, 4),
                "confidence": round(cog_result.confidence, 4) if cog_result.confidence is not None else None,
                "policy_result": "block",
                "final_selected_action": None,
                "reason_for_rejection_or_abstention": f"Policy Gate Blocked Action: {gate_res.reason}",
            }
            run.history.append({
                "step": step_num,
                "tool": selected_tool_name,
                "arguments": selected_tool_args,
                "gate_decision": "block",
                "success": False,
                "error": f"Policy Gate Blocked Action: {gate_res.reason}",
                "alternatives_evaluated": alternatives_evaluated,
                "model_version": cdm_model_version,
                "decision_provenance": block_prov,
            })
            run.step_count += 1
            self.store.save_run(run)
            return run

        if gate_res.decision == GateDecisionType.REQUIRE_HUMAN_APPROVAL:
            logger.info(f"Run {run.run_id}: Step {step_num} suspended for HUMAN APPROVAL.")
            run.status = "waiting_for_approval"
            run.pending_action = {"tool": selected_tool_name, "arguments": selected_tool_args, "reason": selected_reason}
            run.pending_reason = gate_res.reason
            self.store.save_run(run)
            return run

        # =====================================================================
        # Layer 4: Execution & Verification
        # =====================================================================
        tool = self.tools.get(selected_tool_name)
        if not tool:
            exec_res = ToolResult(success=False, error=f"Tool '{selected_tool_name}' is not registered.", tool_name=selected_tool_name)
        else:
            exec_res = tool.execute(args=selected_tool_args, tenant_id=run.tenant_id)
            run.tool_call_count += 1

        # Phase 3: Action-level postcondition verification
        action_verif_res = None
        if selected_tool_name in ("create_file", "write_file", "edit_file") and "path" in selected_tool_args:
            action_spec = VerificationSpec(check_type="file_exists", target=selected_tool_args["path"], scope="action")
            action_verif_res = OutcomeVerifier.verify(spec=action_spec, sandbox_root=self.sandbox_root)
        elif selected_tool_name == "run_tests":
            action_spec = VerificationSpec(check_type="exit_code_zero", target="exit_code", scope="action")
            action_verif_res = OutcomeVerifier.verify(spec=action_spec, execution_output=exec_res.output, sandbox_root=self.sandbox_root)

        # Plan progress & replanning on failure (Phase 2)
        if exec_res.success:
            run.current_plan_step += 1
            # Check if ground truth verifier confirms goal met after this action
            if run.verification_spec:
                v_spec = VerificationSpec.model_validate(run.verification_spec)
                goal_res = OutcomeVerifier.verify(spec=v_spec, execution_output=exec_res.output, sandbox_root=self.sandbox_root)
                if goal_res.verified:
                    run.status = "completed"
                    run.verification_status = "verified_complete"
        else:
            # Action failed: trigger replanning!
            logger.info(f"Run {run.run_id}: Step {step_num} action '{selected_tool_name}' failed. Triggering replanning...")
            if working_state.active_goals:
                replan_candidates = [a for a in candidate_tools if a != selected_tool_name]
                replan_res = self.cognitive_engine.planning_engine.generate_plan(
                    target_goal=working_state.active_goals[0],
                    available_actions=replan_candidates if replan_candidates else candidate_tools,
                    current_world_state=world_state,
                    reasoning_trace=reasoning_trace,
                    prohibited_actions=run.prohibited_actions,
                    tenant_id=run.tenant_id,
                    task="replan_after_action_failure",
                )
                if replan_res and replan_res.steps:
                    run.active_plan = replan_res.model_dump()
                    run.current_plan_step = 0

        # Phase 4: Verification-Driven Learning
        is_goal_verified = (run.status == "completed")
        if is_goal_verified:
            step_reward = 1.0
            step_review_status = "verified"
        elif not exec_res.success:
            step_reward = -0.5
            step_review_status = "rejected_for_training"
        else:
            # Tool succeeded but goal outcome is unverified / in-progress!
            step_reward = 0.0
            step_review_status = "unverified"

        # Update episodic memory & world model
        try:
            self.cognitive_engine.episodic_memory.record_episode(
                session_id=run.session_id,
                goal=run.goal,
                action_taken=f"{selected_tool_name}({selected_tool_args})",
                observed_outcome=str(exec_res.output if exec_res.success else exec_res.error)[:512],
                reward_signal=step_reward,
                tenant_id=run.tenant_id,
            )
            world_model.record_transition(
                session_id=run.session_id,
                action_name=selected_tool_name,
                action_args=selected_tool_args,
                outcome_observed={"success": exec_res.success, "output": str(exec_res.output if exec_res.success else exec_res.error)[:256]},
                tenant_id=run.tenant_id,
            )
            world_model.record_observed_fact(f"action_executed_{selected_tool_name}", True)
            working_memory.write_scratchpad(f"step_{step_num}_action", selected_tool_name)
            working_memory.write_scratchpad(f"step_{step_num}_success", exec_res.success)

            if hasattr(self.cognitive_engine, "feedback_collector"):
                self.cognitive_engine.feedback_collector.record_feedback(
                    decision_id=f"dec_{run.run_id}_{step_num}",
                    predicted_decision={"tool": selected_tool_name, "args": selected_tool_args, "reason": selected_reason},
                    predicted_outcome={"expected_risk": gate_res.risk_score, "predicted_action": selected_tool_name},
                    actual_outcome={"success": exec_res.success, "goal_verified": is_goal_verified, "action_verified": action_verif_res.verified if action_verif_res else None},
                    is_success=is_goal_verified,
                    tenant_id=run.tenant_id,
                    review_status=step_review_status,
                )
        except Exception as e:
            logger.debug(f"Feedback writeback warning: {e}")

        # Audit persistence with signed provenance record
        self.audit.record_step(
            run_id=run.run_id,
            step_number=step_num,
            tenant_id=run.tenant_id,
            proposed_action={"tool": selected_tool_name, "arguments": selected_tool_args, "reason": selected_reason},
            gate_decision=gate_res.model_dump(),
            execution_result=exec_res.model_dump(),
            verification_result=action_verif_res.model_dump() if action_verif_res else None,
        )

        exec_outcome_state = OutcomeVerifier.classify_execution_state(
            attempted=True,
            executed=exec_res.success,
            action_verified=(action_verif_res.verified if action_verif_res else False),
            goal_verified=is_goal_verified,
        ).value

        allow_prov = {
            "candidate_identifier": selected_tool_name,
            "candidate_validation_result": "valid",
            "decision_model_version": cdm_model_version,
            "used_neural_inference": used_neural_inference,
            "neural_score": round(selected_neural_score, 4) if selected_neural_score is not None else None,
            "heuristic_score": round(selected_heuristic_score, 4) if selected_heuristic_score is not None else None,
            "risk_estimate": round(pred_risk, 4),
            "confidence": round(cog_result.confidence, 4) if cog_result.confidence is not None else None,
            "policy_result": "allow",
            "final_selected_action": selected_tool_name,
            "reason_for_rejection_or_abstention": None,
        }

        run.history.append({
            "step": step_num,
            "tool": selected_tool_name,
            "arguments": selected_tool_args,
            "gate_decision": "allow",
            "success": exec_res.success,
            "output": exec_res.output,
            "error": exec_res.error,
            "execution_state": exec_outcome_state,
            "action_verified": action_verif_res.verified if action_verif_res else None,
            "alternatives_evaluated": alternatives_evaluated,
            "neural_score": selected_neural_score,
            "heuristic_score": selected_heuristic_score,
            "used_neural_inference": used_neural_inference,
            "model_version": cdm_model_version,
            "provenance": cdm_prov,
            "decision_provenance": allow_prov,
        })
        run.step_count += 1
        self.store.save_run(run)
        return run

    def run_until_pause_or_finish(self, run_id: str, max_steps: Optional[int] = None) -> AgentRunState:
        """Executes the loop continuously until completion, pause for approval, or budget exhaust."""
        steps_taken = 0
        limit = max_steps or 20

        while steps_taken < limit:
            run = self.store.load_run(run_id)
            if not run or run.status != "running":
                return run
            self.step(run_id)
            steps_taken += 1

        return self.store.load_run(run_id)

    def approve_pending(self, run_id: str, resume_loop: bool = True) -> AgentRunState:
        """Approves a suspended action, executes it, and optionally resumes the loop."""
        run = self.store.load_run(run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found.")

        if run.status != "waiting_for_approval" or not run.pending_action:
            raise ValueError(f"Run {run_id} is not waiting for human approval.")

        proposed = AgentActionProposal.model_validate(run.pending_action)
        step_num = run.step_count + 1

        # Re-check policy: explicitly prohibited actions cannot be executed even with human approval
        if proposed.tool in run.prohibited_actions:
            run.status = "rejected"
            run.pending_action = None
            run.pending_reason = None
            run.history.append({
                "step": step_num,
                "tool": proposed.tool,
                "arguments": proposed.arguments,
                "gate_decision": "blocked_prohibited",
                "success": False,
                "error": f"Tool '{proposed.tool}' is explicitly prohibited by policy.",
            })
            run.step_count += 1
            self.store.save_run(run)
            return run

        # Execute approved action
        tool = self.tools.get(proposed.tool)
        if not tool:
            exec_res = ToolResult(success=False, error=f"Tool '{proposed.tool}' is not registered.", tool_name=proposed.tool)
        else:
            exec_res = tool.execute(args=proposed.arguments, tenant_id=run.tenant_id)
            run.tool_call_count += 1

        # Action-level postcondition verification
        action_verif_res = None
        if proposed.tool in ("create_file", "write_file", "edit_file") and "path" in proposed.arguments:
            action_spec = VerificationSpec(check_type="file_exists", target=proposed.arguments["path"], scope="action")
            action_verif_res = OutcomeVerifier.verify(spec=action_spec, sandbox_root=self.sandbox_root)
        elif proposed.tool == "run_tests":
            action_spec = VerificationSpec(check_type="exit_code_zero", target="exit_code", scope="action")
            action_verif_res = OutcomeVerifier.verify(spec=action_spec, execution_output=exec_res.output, sandbox_root=self.sandbox_root)

        # Verifier
        verif_res = None
        if run.verification_spec:
            v_spec = VerificationSpec.model_validate(run.verification_spec)
            verif_res = OutcomeVerifier.verify(spec=v_spec, execution_output=exec_res.output, sandbox_root=self.sandbox_root)
            if getattr(v_spec, "scope", "goal") == "goal" and verif_res.verified:
                run.status = "completed"
                run.verification_status = "verified_complete"

        # Phase 4: Verification-Driven Learning for approved execution
        is_goal_verified = (run.status == "completed")
        if is_goal_verified:
            step_reward = 1.0
            step_review_status = "verified"
        elif not exec_res.success:
            step_reward = -0.5
            step_review_status = "rejected_for_training"
        else:
            step_reward = 0.0
            step_review_status = "unverified"

        # Audit record
        self.audit.record_step(
            run_id=run.run_id,
            step_number=step_num,
            tenant_id=run.tenant_id,
            proposed_action=proposed.model_dump(),
            gate_decision={"decision": "human_approved", "risk_score": 0.0},
            execution_result=exec_res.model_dump(),
            verification_result=verif_res.model_dump() if verif_res else None,
        )

        # Feedback to episodic memory & world model
        try:
            self.cognitive_engine.episodic_memory.record_episode(
                session_id=run.session_id,
                goal=run.goal,
                action_taken=f"{proposed.tool}({proposed.arguments})",
                observed_outcome=str(exec_res.output if exec_res.success else exec_res.error)[:512],
                reward_signal=step_reward,
                tenant_id=run.tenant_id,
            )
            world_model = self.cognitive_engine.get_isolated_world_model(tenant_id=run.tenant_id, session_id=run.session_id)
            world_model.record_transition(
                session_id=run.session_id,
                action_name=proposed.tool,
                action_args=proposed.arguments,
                outcome_observed={"success": exec_res.success, "output": str(exec_res.output if exec_res.success else exec_res.error)[:256]},
                tenant_id=run.tenant_id,
            )
            if hasattr(self.cognitive_engine, "feedback_collector"):
                self.cognitive_engine.feedback_collector.record_feedback(
                    decision_id=f"dec_{run.run_id}_{step_num}",
                    predicted_decision={"tool": proposed.tool, "args": proposed.arguments},
                    predicted_outcome={"human_approved": True},
                    actual_outcome={"success": exec_res.success, "goal_verified": is_goal_verified, "verified": verif_res.verified if verif_res else None},
                    is_success=is_goal_verified,
                    tenant_id=run.tenant_id,
                    review_status=step_review_status,
                )
        except Exception as e:
            logger.debug(f"Approval feedback writeback skipped: {e}")

        run.history.append({
            "step": step_num,
            "tool": proposed.tool,
            "arguments": proposed.arguments,
            "gate_decision": "human_approved",
            "success": exec_res.success,
            "output": exec_res.output,
            "error": exec_res.error,
        })
        run.step_count += 1
        run.pending_action = None
        run.pending_reason = None
        run.status = "completed" if is_goal_verified else "running"
        self.store.save_run(run)

        # Resume loop if requested and still running
        if resume_loop and run.status == "running":
            return self.run_until_pause_or_finish(run_id)
        return run

    def reject_pending(self, run_id: str, rejection_reason: str = "Rejected by human operator") -> AgentRunState:
        """Rejects a suspended action and records the rejection."""
        run = self.store.load_run(run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found.")

        if run.status != "waiting_for_approval" or not run.pending_action:
            raise ValueError(f"Run {run_id} is not waiting for human approval.")

        proposed = AgentActionProposal.model_validate(run.pending_action)
        step_num = run.step_count + 1

        self.audit.record_step(
            run_id=run.run_id,
            step_number=step_num,
            tenant_id=run.tenant_id,
            proposed_action=proposed.model_dump(),
            gate_decision={"decision": "human_rejected", "reason": rejection_reason},
            execution_result=None,
            verification_result=None,
        )

        run.history.append({
            "step": step_num,
            "tool": proposed.tool,
            "arguments": proposed.arguments,
            "gate_decision": "human_rejected",
            "success": False,
            "error": f"Action rejected: {rejection_reason}",
        })
        run.step_count += 1
        run.pending_action = None
        run.pending_reason = None
        run.status = "rejected"
        self.store.save_run(run)
        return run
