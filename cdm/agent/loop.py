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
from cdm.cognitive.state import WorldState, WorldEntity
from cdm.gate.policy_gate import PolicyGate, ProposedAction, GateDecisionType, GateDecision
from cdm.execution.tools.base import BaseTool, ToolResult
from cdm.execution.tools.shell_tool import SandboxedShellTool
from cdm.execution.tools.file_tool import FileReadTool, FileWriteTool, FileEditTool
from cdm.execution.tools.http_tool import HttpCallTool
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
    status: str = "running" # 'running', 'waiting_for_approval', 'completed', 'failed', 'rejected', 'step_budget_exceeded', 'cost_budget_exceeded'
    step_count: int = 0
    step_budget: int = 10
    cost_budget_usd: float = 1.0
    accumulated_cost_usd: float = 0.0
    accumulated_tokens: int = 0
    tool_call_count: int = 0
    history: List[Dict[str, Any]] = Field(default_factory=list)
    pending_action: Optional[Dict[str, Any]] = None
    pending_reason: Optional[str] = None
    verification_spec: Optional[Dict[str, Any]] = None
    tenant_policies: List[str] = Field(default_factory=list)
    prohibited_actions: List[str] = Field(default_factory=list)
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
        self.policy_gate = policy_gate or PolicyGate(decision_model=getattr(self.cognitive_engine, "decision_model", None))
        self.store = store or AgentStore()
        self.audit = audit_manager or AuditTrailManager()

        # Register default execution tools
        self.tools: Dict[str, BaseTool] = tools or {
            "read_file": FileReadTool(sandbox_root=self.sandbox_root),
            "write_file": FileWriteTool(sandbox_root=self.sandbox_root),
            "edit_file": FileEditTool(sandbox_root=self.sandbox_root),
            "shell_exec": SandboxedShellTool(working_dir=self.sandbox_root),
            "http_request": HttpCallTool(),
        }

    def register_tool(self, tool: BaseTool) -> None:
        self.tools[tool.name] = tool

    def create_run(
        self,
        goal: str,
        task_description: str,
        tenant_id: str = "default_tenant",
        step_budget: int = 10,
        cost_budget_usd: float = 1.0,
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

        # =====================================================================
        # Layer 2: Context, Memory, and World State Retrieval
        # =====================================================================
        # Query episodic memory for relevant past experiences
        relevant_episodes = []
        try:
            relevant_episodes = self.cognitive_engine.episodic_memory.search_episodes(
                query_text=f"{run.goal} {run.task_description}",
                limit=3,
                tenant_id=run.tenant_id,
            )
        except Exception as e:
            logger.debug(f"Episodic memory search skipped: {e}")

        # Get world state
        world_state = None
        try:
            world_state = self.cognitive_engine.world_model.get_state(session_id=run.session_id)
        except Exception:
            world_state = WorldState(tenant_id=run.tenant_id, session_id=run.session_id)

        # Build prompt context
        memory_notes = []
        for ep in relevant_episodes:
            memory_notes.append(f"- Past action '{ep.action_taken}' produced outcome '{ep.observed_outcome}' (reward: {ep.reward_signal})")

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

        history_summary = []
        for h in run.history[-5:]: # Last 5 steps
            history_summary.append(
                f"Step {h.get('step')}: Action '{h.get('tool')}' -> "
                f"Gate: {h.get('gate_decision')} | Success: {h.get('success')} | Result: {h.get('output') or h.get('error')}"
            )

        user_prompt = (
            f"Task Goal: {run.goal}\n"
            f"Description: {run.task_description}\n"
            f"Step Count: {run.step_count + 1} / {run.step_budget}\n\n"
            f"Context from Episodic Memory:\n" + ("\n".join(memory_notes) if memory_notes else "None available.") + "\n\n"
            f"Recent Execution History:\n" + ("\n".join(history_summary) if history_summary else "No actions taken yet.") + "\n\n"
            "Propose the next optimal action."
        )

        # =====================================================================
        # Layer 1: Foundation Model Proposal
        # =====================================================================
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]
        llm_resp = self.llm.complete(messages=messages, schema=AgentActionProposal, tenant_id=run.tenant_id)
        run.accumulated_tokens += (llm_resp.input_tokens + llm_resp.output_tokens)
        run.accumulated_cost_usd += llm_resp.cost_usd

        proposal_data = llm_resp.parsed_json or {
            "tool": "complete_task" if "complete" in llm_resp.raw_text.lower() else "shell_exec",
            "arguments": {},
            "reason": "Fallback default proposal",
            "claim_goal_met": "complete" in llm_resp.raw_text.lower(),
        }
        proposed = AgentActionProposal.model_validate(proposal_data)

        # Check if LLM claims completion
        if proposed.tool == "complete_task" or proposed.claim_goal_met:
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
                    })
                    run.step_count += 1
                    self.store.save_run(run)
                    return run

            # Confirmed goal achievement!
            run.status = "completed"
            run.history.append({
                "step": run.step_count + 1,
                "tool": "complete_task",
                "gate_decision": "allow",
                "success": True,
                "output": "Task goal verified and completed.",
            })
            run.step_count += 1
            self.store.save_run(run)
            return run

        # =====================================================================
        # Layer 3: Policy Gate & Risk Evaluation
        # =====================================================================
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
            # Audit record for blocked action
            self.audit.record_step(
                run_id=run.run_id,
                step_number=step_num,
                tenant_id=run.tenant_id,
                proposed_action=proposed.model_dump(),
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

        if gate_res.decision == GateDecisionType.REQUIRE_HUMAN_APPROVAL:
            logger.info(f"Run {run.run_id}: Step {step_num} suspended for HUMAN APPROVAL.")
            run.status = "waiting_for_approval"
            run.pending_action = proposed.model_dump()
            run.pending_reason = gate_res.reason
            self.store.save_run(run)
            return run

        # =====================================================================
        # Layer 4: Execution & Verification
        # =====================================================================
        tool = self.tools.get(proposed.tool)
        if not tool:
            exec_res = ToolResult(success=False, error=f"Tool '{proposed.tool}' is not registered.", tool_name=proposed.tool)
        else:
            exec_res = tool.execute(args=proposed.arguments, tenant_id=run.tenant_id)
            run.tool_call_count += 1

        # Ground truth verification check if specified
        verif_res = None
        if run.verification_spec:
            v_spec = VerificationSpec.model_validate(run.verification_spec)
            verif_res = OutcomeVerifier.verify(spec=v_spec, execution_output=exec_res.output, sandbox_root=self.sandbox_root)

        # Audit persistence with signed provenance record
        self.audit.record_step(
            run_id=run.run_id,
            step_number=step_num,
            tenant_id=run.tenant_id,
            proposed_action=proposed.model_dump(),
            gate_decision=gate_res.model_dump(),
            execution_result=exec_res.model_dump(),
            verification_result=verif_res.model_dump() if verif_res else None,
        )

        # =====================================================================
        # Loop: Feedback into Layer 2 (Memory and World State)
        # =====================================================================
        try:
            # 1. Update episodic memory
            reward = 1.0 if exec_res.success else -0.5
            self.cognitive_engine.episodic_memory.record_episode(
                session_id=run.session_id,
                goal=run.goal,
                action_taken=f"{proposed.tool}({proposed.arguments})",
                observed_outcome=str(exec_res.output if exec_res.success else exec_res.error)[:512],
                reward_signal=reward,
                tenant_id=run.tenant_id,
            )
            # 2. Update world state
            if hasattr(self.cognitive_engine, "world_model"):
                self.cognitive_engine.world_model.record_transition(
                    session_id=run.session_id,
                    action_name=proposed.tool,
                    action_args=proposed.arguments,
                    outcome_observed={"success": exec_res.success},
                    tenant_id=run.tenant_id,
                )
        except Exception as e:
            logger.debug(f"Feedback writeback warning: {e}")

        run.history.append({
            "step": step_num,
            "tool": proposed.tool,
            "arguments": proposed.arguments,
            "gate_decision": "allow",
            "success": exec_res.success,
            "output": exec_res.output,
            "error": exec_res.error,
        })
        run.step_count += 1

        # Check if ground truth verifier confirms goal met after this action
        if verif_res and verif_res.verified:
            run.status = "completed"

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

        # Execute approved action
        tool = self.tools.get(proposed.tool)
        if not tool:
            exec_res = ToolResult(success=False, error=f"Tool '{proposed.tool}' is not registered.", tool_name=proposed.tool)
        else:
            exec_res = tool.execute(args=proposed.arguments, tenant_id=run.tenant_id)
            run.tool_call_count += 1

        # Verifier
        verif_res = None
        if run.verification_spec:
            v_spec = VerificationSpec.model_validate(run.verification_spec)
            verif_res = OutcomeVerifier.verify(spec=v_spec, execution_output=exec_res.output, sandbox_root=self.sandbox_root)

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

        # Feedback to episodic memory
        try:
            self.cognitive_engine.episodic_memory.record_episode(
                session_id=run.session_id,
                goal=run.goal,
                action_taken=f"{proposed.tool}({proposed.arguments})",
                observed_outcome=str(exec_res.output if exec_res.success else exec_res.error)[:512],
                reward_signal=1.0 if exec_res.success else -0.5,
                tenant_id=run.tenant_id,
            )
        except Exception:
            pass

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
        run.status = "completed" if (verif_res and verif_res.verified) else "running"
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
