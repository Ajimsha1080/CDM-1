"use client";

import React, { useState, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import {
  cdmApi,
  AgentRunResponse,
  CreateAgentRunRequest,
} from "../../lib/cdm-api";
import {
  Bot,
  Play,
  CheckCircle2,
  AlertTriangle,
  ShieldAlert,
  Check,
  X,
  Plus,
  Coins,
  Footprints,
  Clock,
  Terminal,
  RefreshCw,
  Sliders,
  Sparkles,
} from "lucide-react";

function AgentRunsContent() {
  const searchParams = useSearchParams();
  const [runs, setRuns] = useState<AgentRunResponse[]>([]);
  const [selectedRun, setSelectedRun] = useState<AgentRunResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState(false);
  const [isRealtime, setIsRealtime] = useState(true);

  // New Run Form State
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [goal, setGoal] = useState("Investigate and remediate elevated latency in authentication service");
  const [taskDescription, setTaskDescription] = useState("Inspect recent logs, identify problematic database queries, and apply rate limiting if necessary.");
  const [stepBudget, setStepBudget] = useState(10);
  const [costBudget, setCostBudget] = useState(1.0);
  const [autoRun, setAutoRun] = useState(true);

  // Rejection reason state
  const [rejectReason, setRejectReason] = useState("");
  const [showRejectInput, setShowRejectInput] = useState(false);

  const fetchRuns = async (silent = false) => {
    if (!silent) setLoading(true);
    try {
      const data = await cdmApi.listAgentRuns(50);
      setRuns(data);
      if (data.length > 0) {
        if (!selectedRun) {
          const queryRunId = searchParams.get("run");
          if (queryRunId) {
            const found = data.find((r) => r.run_id === queryRunId);
            if (found) setSelectedRun(found);
            else setSelectedRun(data[0]);
          } else {
            setSelectedRun(data[0]);
          }
        } else {
          // Keep current selectedRun synced with latest backend status
          const updatedSelected = data.find((r) => r.run_id === selectedRun.run_id);
          if (updatedSelected && (updatedSelected.step_count !== selectedRun.step_count || updatedSelected.status !== selectedRun.status)) {
            const fullDetails = await cdmApi.getAgentRun(selectedRun.run_id);
            setSelectedRun(fullDetails);
          }
        }
      }
      setError(null);
    } catch (err: any) {
      if (!silent) setError(err.message || "Failed to load agent runs");
    } finally {
      if (!silent) setLoading(false);
    }
  };

  useEffect(() => {
    fetchRuns();
  }, [searchParams]);

  // Real-time automatic polling loop (every 3 seconds)
  useEffect(() => {
    if (!isRealtime) return;
    const interval = setInterval(() => {
      fetchRuns(true);
    }, 3000);
    return () => clearInterval(interval);
  }, [isRealtime, selectedRun?.run_id]);

  const handleSelectRun = async (runId: string) => {
    try {
      const run = await cdmApi.getAgentRun(runId);
      setSelectedRun(run);
    } catch (err: any) {
      setError(err.message || "Failed to load run details");
    }
  };

  const handleCreateRun = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!goal.trim()) return;

    setCreating(true);
    setError(null);

    try {
      const newRun = await cdmApi.createAgentRun({
        goal: goal.trim(),
        task_description: taskDescription.trim(),
        step_budget: stepBudget,
        cost_budget_usd: costBudget,
        auto_run: autoRun,
      });

      setShowCreateModal(false);
      setSelectedRun(newRun);
      await fetchRuns();
    } catch (err: any) {
      setError(err.message || "Failed to create agent run");
    } finally {
      setCreating(false);
    }
  };

  const handleApprove = async () => {
    if (!selectedRun) return;
    setActionLoading(true);
    try {
      const updated = await cdmApi.approveAgentRun(selectedRun.run_id);
      setSelectedRun(updated);
      await fetchRuns();
    } catch (err: any) {
      setError(err.message || "Approval failed");
    } finally {
      setActionLoading(false);
    }
  };

  const handleReject = async () => {
    if (!selectedRun) return;
    setActionLoading(true);
    try {
      const updated = await cdmApi.rejectAgentRun(
        selectedRun.run_id,
        rejectReason.trim() || "Rejected by human supervisor"
      );
      setSelectedRun(updated);
      setShowRejectInput(false);
      await fetchRuns();
    } catch (err: any) {
      setError(err.message || "Rejection failed");
    } finally {
      setActionLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto space-y-6 pb-12">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#E8E5DC] pb-4">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-semibold text-[#1C1917] tracking-tight">
              Autonomous Agent Lifecycle & Human Gate
            </h1>
            <span className="text-[11px] font-mono font-medium px-2 py-0.5 rounded-full luxury-pill-neutral">
              POST /v1/agent/runs
            </span>
          </div>
          <p className="text-xs text-[#78716C] mt-1.5 leading-relaxed">
            Monitor autonomous cognitive agent cycles, step budgets, cost constraints, and human-in-the-loop sign-offs.
          </p>
        </div>

        <div className="flex items-center gap-2.5">
          {/* Real-time streaming toggle */}
          <button
            type="button"
            onClick={() => setIsRealtime(!isRealtime)}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs font-mono font-medium transition cursor-pointer ${
              isRealtime
                ? "bg-[#F0FDF4] border-[#BBF7D0] text-[#166534]"
                : "bg-white border-[#E8E5DC] text-[#78716C] hover:bg-[#FAF9F5]"
            }`}
            title="Toggle real-time background polling (every 3 seconds)"
          >
            <span
              className={`w-2 h-2 rounded-full ${
                isRealtime ? "bg-[#16A34A] animate-pulse" : "bg-[#A8A29E]"
              }`}
            />
            <span>{isRealtime ? "Live Streaming: 3s" : "Live Paused"}</span>
          </button>

          <button
            onClick={() => fetchRuns(false)}
            className="p-2 rounded-lg border border-[#E8E5DC] bg-white hover:bg-[#F7F5EE] text-[#44403C] transition cursor-pointer"
            title="Force refresh runs"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} strokeWidth={1.75} />
          </button>
          <button
            onClick={() => setShowCreateModal(true)}
            className="flex items-center gap-2 px-3.5 py-2 rounded-lg luxury-btn text-xs font-semibold cursor-pointer"
          >
            <Plus className="w-3.5 h-3.5" strokeWidth={1.75} />
            <span>Launch Agent Run</span>
          </button>
        </div>
      </div>

      {error && (
        <div className="p-3.5 luxury-pill-rose rounded-xl text-xs flex items-start gap-2.5">
          <AlertTriangle className="w-4 h-4 text-[#9F1239] flex-shrink-0 mt-0.5" strokeWidth={1.75} />
          <span className="font-mono">{error}</span>
        </div>
      )}

      {/* Main Grid: Left Runs List, Right Run Detail */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Runs List (4 cols) */}
        <div className="lg:col-span-4 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-mono uppercase text-[#78716C] tracking-wider font-semibold">
              Execution Runs ({runs.length})
            </h3>
            <span className="text-[10px] text-[#A8A29E] font-mono">Real-time status</span>
          </div>

          <div className="space-y-2.5 max-h-[700px] overflow-y-auto pr-1">
            {runs.length === 0 ? (
              <div className="p-8 border border-dashed border-[#E8E5DC] bg-white/50 rounded-xl text-center text-xs text-[#78716C]">
                No agent runs recorded yet. Click Launch Agent Run to begin.
              </div>
            ) : (
              runs.map((r) => {
                const isSelected = selectedRun?.run_id === r.run_id;
                const isPaused = r.status === "paused_approval_required";
                return (
                  <div
                    key={r.run_id}
                    onClick={() => handleSelectRun(r.run_id)}
                    className={`p-3.5 rounded-xl border cursor-pointer transition space-y-2.5 ${
                      isSelected
                        ? "bg-[#FFFFFF] border-[#1C1917] shadow-sm ring-1 ring-[#1C1917]"
                        : "bg-white hover:bg-[#FAF9F5] border-[#E8E5DC]"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-xs font-semibold text-[#1C1917]">
                        {r.run_id.slice(0, 10)}...
                      </span>
                      <span
                        className={`text-[10px] font-mono font-medium px-2 py-0.5 rounded-full ${
                          r.status === "completed"
                            ? "luxury-pill-green"
                            : isPaused
                            ? "luxury-pill-amber font-bold animate-pulse"
                            : "luxury-pill-neutral"
                        }`}
                      >
                        {r.status}
                      </span>
                    </div>

                    <p className="text-xs text-[#44403C] line-clamp-2 leading-relaxed">{r.goal}</p>

                    <div className="flex items-center justify-between text-[11px] font-mono text-[#78716C] pt-2 border-t border-[#F5F3ED]">
                      <span className="flex items-center gap-1">
                        <Footprints className="w-3 h-3 text-[#A8A29E]" strokeWidth={1.75} />
                        Steps: {r.step_count}/{r.step_budget}
                      </span>
                      <span className="flex items-center gap-1 font-semibold text-[#1C1917]">
                        <Coins className="w-3 h-3 text-[#A8A29E]" strokeWidth={1.75} />
                        ${r.accumulated_cost_usd.toFixed(4)}
                      </span>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* Selected Run Details (8 cols) */}
        <div className="lg:col-span-8 space-y-4">
          {!selectedRun ? (
            <div className="p-16 border border-dashed border-[#E8E5DC] bg-white rounded-xl text-center text-[#78716C] text-xs">
              Select or launch an agent run to inspect execution state and live tool traces.
            </div>
          ) : (
            <div className="space-y-4">
              {/* Human Approval Alert Banner if paused */}
              {selectedRun.status === "paused_approval_required" && (
                <div className="p-4 bg-[#FFFBEB] border border-[#FDE68A] rounded-xl space-y-3 shadow-sm">
                  <div className="flex items-start gap-3">
                    <ShieldAlert className="w-5 h-5 text-[#92400E] flex-shrink-0 mt-0.5" strokeWidth={1.75} />
                    <div className="space-y-1 text-xs">
                      <h4 className="font-bold text-[#92400E] text-sm">
                        Human Authorization Gate Required
                      </h4>
                      <p className="text-[#B45309] leading-relaxed">
                        {selectedRun.pending_reason || "This operational action exceeds risk thresholds and requires explicit human supervisor sign-off before proceeding."}
                      </p>
                      {selectedRun.pending_action && (
                        <div className="p-2.5 bg-white rounded-lg border border-[#FDE68A] font-mono text-[11px] text-[#92400E] mt-2">
                          <span className="font-bold">Pending Action: </span>
                          <span className="font-semibold text-[#1C1917]">{selectedRun.pending_action.action}</span>
                          {" "}(Estimated Risk: <span className="font-bold text-[#9F1239]">{(selectedRun.pending_action.estimated_risk * 100).toFixed(0)}%</span>)
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center justify-end gap-2 pt-2 border-t border-[#FDE68A]">
                    {!showRejectInput ? (
                      <>
                        <button
                          type="button"
                          onClick={() => setShowRejectInput(true)}
                          disabled={actionLoading}
                          className="px-3.5 py-1.5 rounded-lg border border-[#FECDD3] bg-white hover:bg-[#FFF1F2] text-[#9F1239] text-xs font-semibold flex items-center gap-1.5 transition cursor-pointer"
                        >
                          <X className="w-3.5 h-3.5" strokeWidth={1.75} />
                          Reject Action
                        </button>
                        <button
                          type="button"
                          onClick={handleApprove}
                          disabled={actionLoading}
                          className="px-4 py-1.5 rounded-lg bg-[#166534] hover:bg-[#14532D] text-white text-xs font-semibold flex items-center gap-1.5 transition shadow-sm cursor-pointer"
                        >
                          <Check className="w-3.5 h-3.5" strokeWidth={1.75} />
                          Approve and Resume Loop
                        </button>
                      </>
                    ) : (
                      <div className="flex items-center gap-2 w-full">
                        <input
                          type="text"
                          value={rejectReason}
                          onChange={(e) => setRejectReason(e.target.value)}
                          placeholder="Reason for rejecting this action..."
                          className="flex-1 text-xs px-3 py-1.5 rounded-lg luxury-input text-[#1C1917]"
                        />
                        <button
                          type="button"
                          onClick={handleReject}
                          disabled={actionLoading}
                          className="px-3 py-1.5 rounded-lg bg-[#9F1239] hover:bg-[#881337] text-white text-xs font-semibold cursor-pointer"
                        >
                          Confirm Rejection
                        </button>
                        <button
                          type="button"
                          onClick={() => setShowRejectInput(false)}
                          className="px-2.5 py-1.5 text-xs text-[#78716C] hover:text-[#1C1917] cursor-pointer"
                        >
                          Cancel
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Run Metrics Bar */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="luxury-card p-3.5 rounded-xl">
                  <span className="text-[11px] font-mono text-[#78716C]">Step Budget</span>
                  <div className="text-base font-bold text-[#1C1917] font-mono mt-1">
                    {selectedRun.step_count} <span className="text-xs text-[#A8A29E] font-normal">/ {selectedRun.step_budget}</span>
                  </div>
                </div>

                <div className="luxury-card p-3.5 rounded-xl">
                  <span className="text-[11px] font-mono text-[#78716C]">Accumulated Cost</span>
                  <div className="text-base font-bold text-[#166534] font-mono mt-1">
                    ${selectedRun.accumulated_cost_usd.toFixed(4)}
                    <span className="text-[10px] text-[#78716C] ml-1 font-normal">
                      / ${selectedRun.cost_budget_usd.toFixed(2)}
                    </span>
                  </div>
                </div>

                <div className="luxury-card p-3.5 rounded-xl">
                  <span className="text-[11px] font-mono text-[#78716C]">Tool Invocations</span>
                  <div className="text-base font-bold text-[#1C1917] font-mono mt-1">
                    {selectedRun.tool_call_count}
                  </div>
                </div>

                <div className="luxury-card p-3.5 rounded-xl">
                  <span className="text-[11px] font-mono text-[#78716C]">Run Status</span>
                  <div className="text-base font-bold text-[#1C1917] font-mono mt-1 truncate">
                    {selectedRun.status}
                  </div>
                </div>
              </div>

              {/* Goal Description */}
              <div className="luxury-card p-4 rounded-xl space-y-2">
                <span className="text-[11px] font-mono text-[#78716C] uppercase tracking-wider font-semibold">Primary Goal</span>
                <p className="text-xs text-[#1C1917] leading-relaxed font-semibold">{selectedRun.goal}</p>
                {selectedRun.task_description && (
                  <p className="text-xs text-[#78716C] leading-relaxed pt-1 border-t border-[#F5F3ED]">{selectedRun.task_description}</p>
                )}
              </div>

              {/* Execution Steps Timeline */}
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <h4 className="text-xs font-mono uppercase text-[#78716C] tracking-wider font-semibold">
                    Execution Steps ({selectedRun.history.length})
                  </h4>
                  <span className="text-[11px] text-[#A8A29E] font-mono">Sequential verification</span>
                </div>

                <div className="space-y-3">
                  {selectedRun.history.length === 0 ? (
                    <div className="p-6 text-center text-[#78716C] text-xs border border-dashed border-[#E8E5DC] bg-white rounded-xl">
                      No steps executed yet.
                    </div>
                  ) : (
                    selectedRun.history.map((step, idx) => (
                      <div
                        key={idx}
                        className="luxury-card p-4 rounded-xl space-y-2.5 text-xs"
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <span className="px-2 py-0.5 rounded-md bg-[#FAF9F5] border border-[#E8E5DC] text-[#44403C] font-mono text-[10px] font-semibold">
                              Step #{step.step_index}
                            </span>
                            <span className="font-mono font-bold text-[#1C1917]">
                              {step.action_name}
                            </span>
                          </div>
                          <span className="text-[11px] font-mono text-[#78716C]">
                            Risk: <span className="font-semibold text-[#1C1917]">{(step.risk_assessed * 100).toFixed(0)}%</span>
                          </span>
                        </div>

                        {step.observation && (
                          <div className="p-2.5 rounded-lg bg-[#FAF9F5] border border-[#E8E5DC] text-[#44403C] font-mono text-[11px] leading-relaxed">
                            {step.observation}
                          </div>
                        )}

                        {step.tool_result && (
                          <div className="p-2.5 rounded-lg bg-[#FAF9F5] border border-[#E8E5DC] text-[#166534] font-mono text-[11px] overflow-x-auto">
                            <pre>{JSON.stringify(step.tool_result, null, 2)}</pre>
                          </div>
                        )}
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Modal: Create Agent Run */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/40 backdrop-blur-xs z-50 flex items-center justify-center p-4">
          <form
            onSubmit={handleCreateRun}
            className="bg-white border border-[#E8E5DC] rounded-2xl p-6 max-w-lg w-full shadow-2xl space-y-4"
          >
            <div className="flex items-center justify-between border-b border-[#E8E5DC] pb-3">
              <h3 className="font-semibold text-sm text-[#1C1917] flex items-center gap-2">
                <Bot className="w-4 h-4 text-[#1C1917]" strokeWidth={1.75} />
                Initialize Autonomous Agent Run
              </h3>
              <button
                type="button"
                onClick={() => setShowCreateModal(false)}
                className="text-[#A8A29E] hover:text-[#1C1917] text-sm cursor-pointer"
              >
                ✕
              </button>
            </div>

            <div>
              <label className="block text-xs font-semibold text-[#1C1917] mb-1">
                Goal / Primary Objective
              </label>
              <textarea
                rows={2}
                value={goal}
                onChange={(e) => setGoal(e.target.value)}
                required
                className="w-full text-xs p-2.5 luxury-input"
              />
            </div>

            <div>
              <label className="block text-xs font-semibold text-[#1C1917] mb-1">
                Task Instructions & Operational Context
              </label>
              <textarea
                rows={3}
                value={taskDescription}
                onChange={(e) => setTaskDescription(e.target.value)}
                className="w-full text-xs p-2.5 luxury-input"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-semibold text-[#1C1917] mb-1">
                  Step Budget (1-50)
                </label>
                <input
                  type="number"
                  min={1}
                  max={50}
                  value={stepBudget}
                  onChange={(e) => setStepBudget(parseInt(e.target.value) || 10)}
                  className="w-full text-xs px-3 py-2 luxury-input font-mono"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-[#1C1917] mb-1">
                  Cost Budget ($USD)
                </label>
                <input
                  type="number"
                  step="0.05"
                  min={0.01}
                  max={50.0}
                  value={costBudget}
                  onChange={(e) => setCostBudget(parseFloat(e.target.value) || 1.0)}
                  className="w-full text-xs px-3 py-2 luxury-input font-mono"
                />
              </div>
            </div>

            <div className="flex items-center gap-2 pt-2">
              <input
                type="checkbox"
                id="autoRunCheck"
                checked={autoRun}
                onChange={(e) => setAutoRun(e.target.checked)}
                className="rounded border-[#E8E5DC] text-[#1C1917] focus:ring-0"
              />
              <label htmlFor="autoRunCheck" className="text-xs text-[#44403C]">
                Auto-run loop until pause condition or completion
              </label>
            </div>

            <div className="flex justify-end gap-2 pt-4 border-t border-[#E8E5DC]">
              <button
                type="button"
                onClick={() => setShowCreateModal(false)}
                className="px-3 py-1.5 text-xs text-[#78716C] hover:text-[#1C1917] cursor-pointer"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={creating}
                className="px-4 py-2 text-xs font-semibold luxury-btn rounded-lg cursor-pointer disabled:opacity-60"
              >
                {creating ? "Launching..." : "Launch Agent"}
              </button>
            </div>
          </form>
        </div>
      )}
    </div>
  );
}

export default function AgentRunsPage() {
  return (
    <Suspense fallback={<div className="p-8 text-center text-xs font-mono text-[#78716C]">Loading Agent Lifecycle...</div>}>
      <AgentRunsContent />
    </Suspense>
  );
}
