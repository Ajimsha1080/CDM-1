"use client";

import React, { useState } from "react";
import { CognitiveProcessResponse } from "../lib/cdm-api";
import {
  CheckCircle2,
  AlertCircle,
  Eye,
  Database,
  Brain,
  ShieldAlert,
  GitCommit,
  CheckCheck,
  Scale,
  Code2,
  Copy,
  Check,
  ChevronRight,
  Sparkles,
} from "lucide-react";

interface TraceVisualizerProps {
  result: CognitiveProcessResponse | null;
  loading?: boolean;
}

export function TraceVisualizer({ result, loading }: TraceVisualizerProps) {
  const [copiedCode, setCopiedCode] = useState<string | null>(null);

  const handleCopyCode = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedCode(id);
    setTimeout(() => setCopiedCode(null), 2000);
  };

  if (loading) {
    return (
      <div className="luxury-card p-12 rounded-xl text-center space-y-4 animate-pulse">
        <div className="inline-flex p-3 rounded-full bg-[#FAF9F5] text-[#1C1917] border border-[#E8E5DC]">
          <Brain className="w-6 h-6 animate-spin text-[#1C1917]" strokeWidth={1.75} />
        </div>
        <div className="space-y-1">
          <h4 className="text-sm font-semibold text-[#1C1917] tracking-tight">
            Executing Multi-Stage Cognitive Pipeline
          </h4>
          <p className="text-xs text-[#78716C] font-mono">
            Perception → Working Memory → Reasoning Gate → Plan Generation → Calibration
          </p>
        </div>
      </div>
    );
  }

  if (!result) {
    return (
      <div className="luxury-card p-16 rounded-xl text-center space-y-3">
        <Brain className="w-10 h-10 mx-auto text-[#A8A29E]" strokeWidth={1.75} />
        <h4 className="text-sm font-semibold text-[#1C1917]">Awaiting Cognitive Scenario</h4>
        <p className="text-xs text-[#78716C] max-w-sm mx-auto leading-relaxed">
          Configure task narrative, structured features, and constraints on the left, then click Execute to trace live pipeline stages.
        </p>
      </div>
    );
  }

  const {
    reasoning_evidence,
    world_state_summary,
    selected_plan,
    abstain,
    abstention_reason,
    execution_recommendation,
    confidence,
    calibrated_confidence,
    priority,
    risk,
    embedding_mode,
  } = result;

  const ruleActivations = reasoning_evidence?.rule_activations || [];
  const riskFactors = reasoning_evidence?.risk_factors || [];
  const safetyViolations = reasoning_evidence?.safety_violations || [];
  const perceivedEntities = reasoning_evidence?.perceived_entities || [];

  return (
    <div className="space-y-4">
      {/* Overview Top Metric Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {/* Decision */}
        <div className="luxury-card p-4 rounded-xl space-y-1">
          <span className="text-[10px] text-[#78716C] uppercase font-mono tracking-wider font-semibold">
            Synthesized Decision
          </span>
          <div className="text-base font-bold text-[#1C1917] truncate font-mono">
            {typeof result.decision === "object" ? JSON.stringify(result.decision) : String(result.decision)}
          </div>
          <div className="text-[10px] text-[#78716C] font-mono">
            Mode: <span className="font-semibold text-[#1C1917]">{embedding_mode || "neural"}</span>
          </div>
        </div>

        {/* Confidence */}
        <div className="luxury-card p-4 rounded-xl space-y-1">
          <span className="text-[10px] text-[#78716C] uppercase font-mono tracking-wider font-semibold">
            Calibrated Confidence
          </span>
          <div className="text-base font-bold text-[#166534] font-mono">
            {(calibrated_confidence * 100).toFixed(1)}%
          </div>
          <div className="text-[10px] text-[#78716C] font-mono">
            Raw model: {(confidence * 100).toFixed(1)}%
          </div>
        </div>

        {/* Risk Level */}
        <div className="luxury-card p-4 rounded-xl space-y-1">
          <span className="text-[10px] text-[#78716C] uppercase font-mono tracking-wider font-semibold">
            Assessed Risk
          </span>
          <div
            className={`text-base font-bold font-mono ${
              risk > 0.6 ? "text-[#9F1239]" : risk > 0.3 ? "text-[#92400E]" : "text-[#166534]"
            }`}
          >
            {(risk * 100).toFixed(1)}%
          </div>
          <div className="text-[10px] text-[#78716C] font-mono uppercase">
            Priority: [{priority}]
          </div>
        </div>

        {/* Gate State */}
        <div className="luxury-card p-4 rounded-xl space-y-1">
          <span className="text-[10px] text-[#78716C] uppercase font-mono tracking-wider font-semibold">
            Execution Status
          </span>
          <div
            className={`text-base font-bold font-mono ${
              result.execution_status === "rejected"
                ? "text-[#9F1239]"
                : result.execution_status === "needs_approval"
                ? "text-[#D97706]"
                : abstain
                ? "text-[#92400E]"
                : "text-[#166534]"
            }`}
          >
            {(result.execution_status || (abstain ? "ABSTAINED" : "AUTHORIZED")).toUpperCase().replace("_", " ")}
          </div>
          <div className="text-[10px] text-[#78716C] font-mono">
            Policy: {result.policy_check?.policy_status ? result.policy_check.policy_status.toUpperCase() : (abstain ? "BLOCKED" : "PASSED")}
          </div>
        </div>
      </div>

      {/* Human Approval Required Banner if pending approval */}
      {result.approval_requirements?.required && !abstain && (
        <div className="p-4 bg-[#FFFBEB] border border-[#FDE68A] rounded-xl flex items-start gap-3 shadow-xs">
          <AlertCircle className="w-5 h-5 text-[#B45309] flex-shrink-0 mt-0.5" strokeWidth={1.75} />
          <div className="text-xs space-y-1 font-sans">
            <span className="font-semibold text-[#92400E]">
              Human Supervisor Approval Required &bull; Role: [{result.approval_requirements.approver_role || "compliance_officer"}]
            </span>
            <p className="text-[#B45309] leading-relaxed">
              {result.approval_requirements.reason || `Action '${result.proposed_decision || result.decision}' requires explicit authorization before execution.`}
            </p>
            <p className="text-[#78716C] font-mono text-[11px] pt-1">
              Execution Status: <span className="font-semibold text-[#D97706]">NON-EXECUTABLE (Awaiting Approval)</span> &bull; Proposed Decision: <span className="font-semibold text-[#1C1917]">{String(result.proposed_decision || result.decision)}</span>
            </p>
          </div>
        </div>
      )}

      {/* Abstention Warning Banner if triggered */}
      {abstain && (
        <div className="p-4 bg-[#FFFBEB] border border-[#FDE68A] rounded-xl flex items-start gap-3 shadow-xs">
          <AlertCircle className="w-5 h-5 text-[#92400E] flex-shrink-0 mt-0.5" strokeWidth={1.75} />
          <div className="text-xs space-y-1 font-sans">
            <span className="font-semibold text-[#92400E]">
              Cognitive Safety Abstention Triggered
            </span>
            <p className="text-[#B45309] leading-relaxed">
              Reason: {abstention_reason || "Uncertainty calibration or policy risk thresholds exceeded."}
            </p>
            <p className="text-[#92400E] font-mono text-[11px] pt-1">
              Action Recommendation: <span className="font-semibold">{execution_recommendation}</span>
            </p>
          </div>
        </div>
      )}

      {/* Pipeline Stages Breakdown */}
      <div className="space-y-3">
        <h4 className="text-xs font-mono uppercase text-[#78716C] tracking-wider font-semibold">
          Cognitive Pipeline Trace Stages
        </h4>

        {/* Stage 1: Perception */}
        <div className="luxury-card p-4 rounded-xl space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs font-semibold text-[#1C1917]">
              <Eye className="w-4 h-4 text-[#1C1917]" strokeWidth={1.75} />
              <span>1. Perception & Entity Grounding</span>
            </div>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded-full luxury-pill-neutral">
              Backbone: {embedding_mode || "ModernBERT-base"}
            </span>
          </div>
          <div className="text-xs text-[#44403C] space-y-1.5">
            <div className="text-[11px] text-[#78716C]">Ground truth perceived entities:</div>
            {perceivedEntities.length > 0 ? (
              <div className="flex flex-wrap gap-1.5">
                {perceivedEntities.map((e, idx) => (
                  <span
                    key={idx}
                    className="px-2.5 py-0.5 rounded-md bg-[#FAF9F5] text-[#1C1917] text-[11px] font-mono border border-[#E8E5DC]"
                  >
                    {e}
                  </span>
                ))}
              </div>
            ) : (
              <span className="text-xs text-[#A8A29E] italic">No named entities flagged.</span>
            )}
          </div>
        </div>

        {/* Stage 2: Memory Retrieval */}
        <div className="luxury-card p-4 rounded-xl space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs font-semibold text-[#1C1917]">
              <Database className="w-4 h-4 text-[#1C1917]" strokeWidth={1.75} />
              <span>2. Memory Graph & Prior Knowledge</span>
            </div>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded-full luxury-pill-neutral">
              Active Rules: {ruleActivations.length}
            </span>
          </div>
          <div className="space-y-1.5 text-xs">
            {ruleActivations.length > 0 ? (
              ruleActivations.map((rule, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between bg-[#FAF9F5] p-2.5 rounded-lg border border-[#E8E5DC]"
                >
                  <span className="font-mono text-[#1C1917] font-semibold">{rule.rule_id}</span>
                  <span className="text-[#78716C] text-[11px]">{rule.matched_text || "Pattern Match"}</span>
                  <span className="font-mono text-[#78716C] text-[11px]">wt: {rule.weight}</span>
                </div>
              ))
            ) : (
              <div className="text-xs text-[#A8A29E] italic">No domain rule conflicts retrieved.</div>
            )}
          </div>
        </div>

        {/* Stage 3: Reasoning & Safety */}
        <div className="luxury-card p-4 rounded-xl space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-xs font-semibold text-[#1C1917]">
              <ShieldAlert className="w-4 h-4 text-[#1C1917]" strokeWidth={1.75} />
              <span>3. Safety & Governance Constraint Verification</span>
            </div>
            <span className={`text-[10px] font-mono px-2 py-0.5 rounded-full font-semibold ${
              safetyViolations.length > 0 ? "luxury-pill-rose" : "luxury-pill-green"
            }`}>
              Violations: {safetyViolations.length}
            </span>
          </div>
          {safetyViolations.length > 0 ? (
            <div className="space-y-1">
              {safetyViolations.map((v, i) => (
                <div key={i} className="text-xs text-[#9F1239] flex items-center gap-1.5 font-mono">
                  <AlertCircle className="w-3.5 h-3.5 text-[#9F1239]" strokeWidth={1.75} />
                  <span>{v}</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-xs text-[#166534] flex items-center gap-1.5">
              <CheckCircle2 className="w-3.5 h-3.5" strokeWidth={1.75} />
              <span>All active governance safety constraints verified satisfied.</span>
            </div>
          )}
          {riskFactors.length > 0 && (
            <div className="pt-1 text-xs">
              <span className="text-[#78716C]">Risk Contributors: </span>
              <span className="text-[#92400E] font-mono font-medium">{riskFactors.join(", ")}</span>
            </div>
          )}
        </div>

        {/* Stage 4: Synthesized Operational Plan */}
        {selected_plan && (
          <div className="luxury-card p-4 rounded-xl space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-xs font-semibold text-[#1C1917]">
                <GitCommit className="w-4 h-4 text-[#1C1917]" strokeWidth={1.75} />
                <span>4. Synthesized Multi-Step Plan</span>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-full luxury-pill-green font-semibold">
                Steps: {selected_plan.steps?.length || 0}
              </span>
            </div>
            <div className="space-y-2 text-xs">
              {selected_plan.steps?.map((step, idx) => (
                <div
                  key={idx}
                  className="p-2.5 bg-[#FAF9F5] border border-[#E8E5DC] rounded-lg flex items-center justify-between"
                >
                  <div className="flex items-center gap-2.5">
                    <span className="font-mono text-[#78716C] text-[11px]">#{idx + 1}</span>
                    <span className="font-mono font-semibold text-[#1C1917]">{step.action}</span>
                    {step.requires_approval && (
                      <span className="text-[10px] px-2 py-0.5 rounded-full luxury-pill-amber font-mono font-semibold">
                        HUMAN SIGN-OFF REQUIRED
                      </span>
                    )}
                  </div>
                  {step.estimated_risk !== undefined && (
                    <span className="text-[11px] font-mono text-[#78716C]">
                      risk: <span className="font-semibold text-[#1C1917]">{(step.estimated_risk * 100).toFixed(0)}%</span>
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
