"use client";

import React, { useState, useEffect, useRef, Suspense } from "react";
import {
  cdmApi,
  CognitiveProcessResponse,
  CognitiveProcessRequest,
} from "../../lib/cdm-api";
import {
  MessageSquareText,
  Send,
  Sparkles,
  RotateCcw,
  CheckCircle2,
  AlertCircle,
  Brain,
  ShieldCheck,
  ShieldAlert,
  GitCommit,
  SlidersHorizontal,
  ChevronDown,
  ChevronUp,
  Plus,
  Copy,
  Check,
  Layers,
  ArrowRight,
  TrendingUp,
  Scale,
  Activity,
  CornerDownRight,
  HelpCircle,
  FileText,
} from "lucide-react";

interface ChatTurn {
  id: string;
  role: "user" | "assistant";
  timestamp: string;
  question: string;
  context?: {
    goals?: string[];
    constraints?: string[];
    candidates?: string[];
    prohibited?: string[];
    features?: Record<string, any>;
  };
  response?: CognitiveProcessResponse;
  loading?: boolean;
  error?: string;
}

const CATEGORY_PROMPTS = [
  {
    category: "General Problem Solving",
    prompt: "Our payment gateway is experiencing sudden 504 Gateway Timeouts under elevated checkout load. What is the optimal operational response to restore transaction flow?",
    candidates: ["act_cordon_and_drain", "act_burst_autoscale", "act_restart_pods", "act_failover_gateway"],
    prohibited: ["act_restart_pods"],
    goals: ["Restore payment processing within 60s", "Zero transaction data loss"],
    constraints: ["Comply with SLA 99.99%"],
  },
  {
    category: "Decision-Making & Comparison",
    prompt: "Compare AWS multi-region read replica deployment vs. local in-memory Redis cluster for mitigating database connection pool exhaustion.",
    candidates: ["act_deploy_redis_cluster", "act_deploy_read_replica", "act_increase_connection_limit"],
    prohibited: ["act_increase_connection_limit"],
    goals: ["Maintain query latency < 15ms"],
    constraints: ["Budget cap $2,000/mo"],
  },
  {
    category: "Risk Assessment",
    prompt: "Evaluate the security, financial loss, and regulatory compliance risk of approving an expedited $125,000 offshore wire transfer to a newly registered vendor.",
    candidates: ["act_approve_wire", "act_require_dual_mfa", "act_hold_for_compliance", "act_reject_transfer"],
    prohibited: ["act_approve_wire"],
    goals: ["Prevent fraudulent settlement", "Maintain client SLA"],
    constraints: ["AML Directive 4.2 enforcement"],
  },
  {
    category: "Planning & Resource Allocation",
    prompt: "Synthesize a 4-step remediation plan to isolate potentially compromised Kubernetes nodes and safely rotate TLS certificates without dropping live ingress traffic.",
    candidates: ["act_drain_node", "act_rotate_ingress_certs", "act_verify_mtls", "act_reboot_node"],
    prohibited: ["act_reboot_node"],
    goals: ["Zero downtime certificate renewal", "Isolate compromised workers"],
    constraints: ["Maintenance window not required for non-destructive actions"],
  },
];

const FOLLOW_UP_SUGGESTIONS = [
  "Why did you select this option?",
  "What happens if my risk budget decreases by 50%?",
  "Compare this decision against the runner-up alternative.",
  "Revise the operational plan with a strict zero-downtime constraint.",
];

function CognitiveQAPageContent() {
  const [sessionId, setSessionId] = useState<string>("");
  const [inputQuestion, setInputQuestion] = useState("");
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  // Advanced contextual drawer
  const [showDrawer, setShowDrawer] = useState(false);
  const [candidates, setCandidates] = useState<string[]>([]);
  const [newCandidate, setNewCandidate] = useState("");
  const [prohibited, setProhibited] = useState<string[]>([]);
  const [newProhibited, setNewProhibited] = useState("");
  const [goals, setGoals] = useState<string[]>([]);
  const [newGoal, setNewGoal] = useState("");
  const [constraints, setConstraints] = useState<string[]>([]);
  const [newConstraint, setNewConstraint] = useState("");
  const [featuresJson, setFeaturesJson] = useState('{\n  "urgency_level": 2,\n  "domain": 1\n}');

  const chatBottomRef = useRef<HTMLDivElement>(null);

  // Initialize unique session for multi-turn conversational memory
  useEffect(() => {
    initNewSession();
  }, []);

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [turns]);

  const initNewSession = () => {
    const newId = "qa_session_" + Date.now().toString(36);
    setSessionId(newId);
    setTurns([]);
    setInputQuestion("");
    resetContextInputs();
  };

  const resetContextInputs = () => {
    setCandidates([]);
    setProhibited([]);
    setGoals([]);
    setConstraints([]);
    setFeaturesJson('{\n  "urgency_level": 2,\n  "domain": 1\n}');
  };

  const handleApplyPreset = (preset: typeof CATEGORY_PROMPTS[0]) => {
    setInputQuestion(preset.prompt);
    setCandidates(preset.candidates || []);
    setProhibited(preset.prohibited || []);
    setGoals(preset.goals || []);
    setConstraints(preset.constraints || []);
    setShowDrawer(true);
  };

  const handleSubmit = async (overridePrompt?: string) => {
    const textToSubmit = (overridePrompt || inputQuestion).trim();
    if (!textToSubmit || submitting) return;

    let parsedFeatures: Record<string, any> = {};
    try {
      if (featuresJson.trim()) {
        parsedFeatures = JSON.parse(featuresJson);
      }
    } catch {
      // Fallback to empty if parse fails
      parsedFeatures = {};
    }

    const currentContext = {
      goals: goals.length > 0 ? [...goals] : undefined,
      constraints: constraints.length > 0 ? [...constraints] : undefined,
      candidates: candidates.length > 0 ? [...candidates] : undefined,
      prohibited: prohibited.length > 0 ? [...prohibited] : undefined,
      features: Object.keys(parsedFeatures).length > 0 ? parsedFeatures : undefined,
    };

    const turnId = Date.now().toString(36);
    const userTurn: ChatTurn = {
      id: turnId,
      role: "user",
      timestamp: new Date().toLocaleTimeString(),
      question: textToSubmit,
      context: currentContext,
    };

    const assistantTurnId = "ast_" + turnId;
    const pendingAssistantTurn: ChatTurn = {
      id: assistantTurnId,
      role: "assistant",
      timestamp: new Date().toLocaleTimeString(),
      question: textToSubmit,
      loading: true,
    };

    setTurns((prev) => [...prev, userTurn, pendingAssistantTurn]);
    setInputQuestion("");
    setSubmitting(true);

    try {
      const payload: CognitiveProcessRequest = {
        text: textToSubmit,
        structured_features: parsedFeatures,
        task: candidates.length > 0 ? "choice" : "general_qa",
        candidates: candidates.length > 0 ? candidates : undefined,
        prohibited_actions: prohibited.length > 0 ? prohibited : undefined,
        goals: goals.length > 0 ? goals : undefined,
        constraints: constraints.length > 0 ? constraints : undefined,
        session_id: sessionId,
      };

      // Streaming execution with live cognitive trace updates
      let finalResultReceived = false;
      const liveTrace: Array<{ stage_name: string; duration_ms: number; execution_status: string }> = [];

      await cdmApi.processCognitiveStream(
        payload,
        (evt) => {
          if (evt.event === "stage_start") {
            liveTrace.push({ stage_name: evt.stage, duration_ms: 0, execution_status: "running" });
            setTurns((prev) =>
              prev.map((t) =>
                t.id === assistantTurnId
                  ? {
                      ...t,
                      response: {
                        ...(t.response || ({} as any)),
                        cognitive_trace: [...liveTrace],
                      },
                    }
                  : t
              )
            );
          } else if (evt.event === "stage_complete") {
            const idx = liveTrace.findIndex((s) => s.stage_name === evt.stage);
            if (idx >= 0) {
              liveTrace[idx] = { stage_name: evt.stage, duration_ms: evt.duration_ms, execution_status: evt.status };
            } else {
              liveTrace.push({ stage_name: evt.stage, duration_ms: evt.duration_ms, execution_status: evt.status });
            }
            setTurns((prev) =>
              prev.map((t) =>
                t.id === assistantTurnId
                  ? {
                      ...t,
                      response: {
                        ...(t.response || ({} as any)),
                        cognitive_trace: [...liveTrace],
                      },
                    }
                  : t
              )
            );
          } else if (evt.event === "final_result") {
            finalResultReceived = true;
            setTurns((prev) =>
              prev.map((t) =>
                t.id === assistantTurnId
                  ? {
                      ...t,
                      loading: false,
                      response: {
                        ...evt,
                        execution_id: evt.request_id || "req_stream",
                        decision: evt.proposed_decision,
                        cognitive_trace: [...liveTrace],
                      },
                    }
                  : t
              )
            );
          }
        },
        async (streamErr) => {
          // If SSE streaming fails, fall back cleanly to synchronous API
          try {
            const syncRes = await cdmApi.processCognitive(payload);
            setTurns((prev) =>
              prev.map((t) =>
                t.id === assistantTurnId
                  ? { ...t, loading: false, response: syncRes }
                  : t
              )
            );
          } catch (syncErr: any) {
            setTurns((prev) =>
              prev.map((t) =>
                t.id === assistantTurnId
                  ? { ...t, loading: false, error: syncErr.message || "Cognitive execution error." }
                  : t
              )
            );
          }
        },
        () => {
          if (!finalResultReceived) {
            setTurns((prev) =>
              prev.map((t) =>
                t.id === assistantTurnId ? { ...t, loading: false } : t
              )
            );
          }
        }
      );
    } catch (err: any) {
      setTurns((prev) =>
        prev.map((t) =>
          t.id === assistantTurnId
            ? {
                ...t,
                loading: false,
                error: err.message || "Cognitive Engine execution error.",
              }
            : t
        )
      );
    } finally {
      setSubmitting(false);
    }
  };

  const handleCopyJson = (obj: any, id: string) => {
    navigator.clipboard.writeText(JSON.stringify(obj, null, 2));
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6 pb-12 flex flex-col h-[calc(100vh-7rem)]">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-[#E8E5DC] pb-4 flex-shrink-0">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-semibold text-[#1C1917] tracking-tight">
              Cognitive Question-Answering
            </h1>
            <span className="text-[11px] font-mono font-medium px-2 py-0.5 rounded-full luxury-pill-neutral">
              Session-Persistent
            </span>
          </div>
          <p className="text-xs text-[#78716C] mt-1 leading-relaxed">
            Multi-turn problem solving, candidate comparisons, risk assessments, and plan generation executed by CDM-1 backend.
          </p>
        </div>

        <div className="flex items-center gap-2.5">
          <div className="text-[11px] font-mono text-[#78716C] bg-white px-2.5 py-1 rounded-lg border border-[#E8E5DC] shadow-2xs">
            ID: <span className="font-semibold text-[#1C1917]">{sessionId.slice(0, 16)}</span>
          </div>
          <button
            onClick={initNewSession}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[#E8E5DC] bg-white hover:bg-[#F7F5EE] text-[#44403C] text-xs font-medium transition cursor-pointer"
            title="Start new conversation with clean working memory"
          >
            <RotateCcw className="w-3.5 h-3.5" strokeWidth={1.75} />
            <span>New Chat</span>
          </button>
        </div>
      </div>

      {/* Main Conversation Feed Area */}
      <div className="flex-1 overflow-y-auto space-y-5 pr-2">
        {turns.length === 0 ? (
          <div className="py-8 space-y-6 max-w-3xl mx-auto">
            <div className="luxury-card p-8 rounded-2xl text-center space-y-3">
              <div className="inline-flex p-3 rounded-full bg-[#FAF9F5] border border-[#E8E5DC] text-[#1C1917]">
                <MessageSquareText className="w-6 h-6" strokeWidth={1.75} />
              </div>
              <h3 className="text-base font-semibold text-[#1C1917]">
                How can CDM-1 assist your cognitive workflow today?
              </h3>
              <p className="text-xs text-[#78716C] max-w-lg mx-auto leading-relaxed">
                Ask operational questions, request risk evaluations, compare candidate actions, or challenge decisions. Every response is verified through real neural inference and rule engines.
              </p>
            </div>

            {/* Ready-made category starters */}
            <div className="space-y-2.5">
              <span className="text-[11px] font-mono uppercase text-[#78716C] font-semibold tracking-wider">
                Explore Common Cognitive Question Types:
              </span>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {CATEGORY_PROMPTS.map((cat, idx) => (
                  <div
                    key={idx}
                    onClick={() => handleApplyPreset(cat)}
                    className="luxury-card p-3.5 rounded-xl border border-[#E8E5DC] hover:border-[#1C1917] transition cursor-pointer space-y-1.5 group"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold text-[#1C1917] group-hover:underline">
                        {cat.category}
                      </span>
                      <ArrowRight className="w-3.5 h-3.5 text-[#A8A29E] group-hover:text-[#1C1917] transition" strokeWidth={1.75} />
                    </div>
                    <p className="text-xs text-[#78716C] line-clamp-2 leading-relaxed">
                      {cat.prompt}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : (
          turns.map((turn) => {
            if (turn.role === "user") {
              return (
                <div key={turn.id} className="flex flex-col items-end space-y-1.5 max-w-3xl ml-auto">
                  <div className="flex items-center gap-2 text-[10px] font-mono text-[#A8A29E]">
                    <span>You</span>
                    <span>•</span>
                    <span>{turn.timestamp}</span>
                  </div>
                  <div className="p-4 rounded-2xl bg-[#1C1917] text-white text-xs leading-relaxed shadow-sm space-y-2">
                    <p className="font-sans whitespace-pre-wrap">{turn.question}</p>

                    {/* Context Pill summary if provided */}
                    {turn.context && (
                      <div className="pt-2 border-t border-white/10 flex flex-wrap gap-1.5 text-[10px] font-mono">
                        {turn.context.candidates && (
                          <span className="px-2 py-0.5 rounded bg-white/10 text-white/90">
                            {turn.context.candidates.length} candidates
                          </span>
                        )}
                        {turn.context.prohibited && (
                          <span className="px-2 py-0.5 rounded bg-[#9F1239]/40 text-rose-200">
                            {turn.context.prohibited.length} prohibited
                          </span>
                        )}
                        {turn.context.goals && (
                          <span className="px-2 py-0.5 rounded bg-white/10 text-white/90">
                            {turn.context.goals.length} goals
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              );
            }

            // Assistant Turn
            return (
              <div key={turn.id} className="flex flex-col space-y-2 max-w-4xl mr-auto w-full">
                <div className="flex items-center gap-2 text-[10px] font-mono text-[#78716C]">
                  <span className="font-semibold text-[#1C1917] flex items-center gap-1">
                    <Brain className="w-3 h-3 text-[#1C1917]" strokeWidth={1.75} />
                    CDM-1 Cognitive Engine
                  </span>
                  <span>•</span>
                  <span>{turn.timestamp}</span>
                </div>

                {turn.loading ? (
                  <div className="luxury-card p-6 rounded-2xl space-y-3 animate-pulse border border-[#E8E5DC]">
                    <div className="flex items-center gap-2.5 text-xs text-[#78716C] font-mono">
                      <Brain className="w-4 h-4 animate-spin text-[#1C1917]" strokeWidth={1.75} />
                      <span>Executing Neural Multi-Stage Reasoning Loop...</span>
                    </div>
                    <div className="h-2 w-48 bg-[#FAF9F5] rounded-full" />
                  </div>
                ) : turn.error ? (
                  <div className="p-4 luxury-pill-rose rounded-xl text-xs flex items-start gap-2.5">
                    <AlertCircle className="w-4 h-4 text-[#9F1239] flex-shrink-0 mt-0.5" strokeWidth={1.75} />
                    <div>
                      <div className="font-semibold">Cognitive Processing Exception</div>
                      <div className="font-mono mt-0.5">{turn.error}</div>
                    </div>
                  </div>
                ) : turn.response ? (
                  <div className="luxury-card rounded-2xl border border-[#E8E5DC] p-5 space-y-4 shadow-xs">
                    {/* 1. Natural Language Synthesized Answer */}
                    <div className="space-y-2">
                      <div className="flex items-center justify-between">
                        <span className="text-[10px] font-mono uppercase text-[#78716C] font-semibold tracking-wider flex items-center gap-1.5">
                          <span>Synthesized Cognitive Answer</span>
                          {turn.response.answer_type && (
                            <span className="px-1.5 py-0.5 rounded bg-[#FAF9F5] border border-[#E8E5DC] text-[9px] text-[#44403C]">
                              [{turn.response.answer_type.toUpperCase()}]
                            </span>
                          )}
                        </span>
                        {turn.response.safety_status && (
                          <span
                            className={`text-[10px] font-mono font-semibold uppercase px-2 py-0.5 rounded-full ${
                              turn.response.safety_status === "verified"
                                ? "luxury-pill-success"
                                : turn.response.safety_status === "needs_approval"
                                ? "luxury-pill-warning"
                                : "luxury-pill-danger"
                            }`}
                          >
                            {turn.response.safety_status.replace("_", " ")}
                          </span>
                        )}
                      </div>
                      <div className="text-sm font-medium text-[#1C1917] leading-relaxed bg-[#FAF9F5] p-3.5 rounded-xl border border-[#E8E5DC]">
                        {turn.response.answer ||
                          turn.response.execution_recommendation ||
                          `Optimal recommended action: ${String(turn.response.decision)}`}
                      </div>
                      {turn.response.explanation && (
                        <p className="text-xs text-[#57534E] leading-relaxed px-1">
                          {turn.response.explanation}
                        </p>
                      )}
                    </div>

                    {/* 2. Cognitive Metrics Bar */}
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 pt-2 border-t border-[#F5F3ED]">
                      <div className="p-2.5 rounded-lg bg-[#FAF9F5] border border-[#E8E5DC]">
                        <span className="text-[10px] font-mono text-[#78716C] block">Calibrated Confidence</span>
                        <span className="text-sm font-bold font-mono text-[#166534]">
                          {(turn.response.calibrated_confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="p-2.5 rounded-lg bg-[#FAF9F5] border border-[#E8E5DC]">
                        <span className="text-[10px] font-mono text-[#78716C] block">Assessed Risk</span>
                        <span
                          className={`text-sm font-bold font-mono ${
                            turn.response.risk > 0.6 ? "text-[#9F1239]" : "text-[#166534]"
                          }`}
                        >
                          {(turn.response.risk * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="p-2.5 rounded-lg bg-[#FAF9F5] border border-[#E8E5DC]">
                        <span className="text-[10px] font-mono text-[#78716C] block">Safety Gate</span>
                        <span
                          className={`text-sm font-bold font-mono ${
                            turn.response.abstain ? "text-[#92400E]" : "text-[#166534]"
                          }`}
                        >
                          {turn.response.abstain ? "ABSTAINED" : "AUTHORIZED"}
                        </span>
                      </div>
                      <div className="p-2.5 rounded-lg bg-[#FAF9F5] border border-[#E8E5DC]">
                        <span className="text-[10px] font-mono text-[#78716C] block">Inference Latency</span>
                        <span className="text-sm font-bold font-mono text-[#1C1917]">
                          {turn.response.latency_ms} ms
                        </span>
                      </div>
                    </div>

                    {/* Abstention Warning if triggered */}
                    {turn.response.abstain && (
                      <div className="p-3.5 bg-[#FFFBEB] border border-[#FDE68A] rounded-xl flex items-start gap-2.5">
                        <AlertCircle className="w-4 h-4 text-[#92400E] flex-shrink-0 mt-0.5" strokeWidth={1.75} />
                        <div className="text-xs space-y-0.5">
                          <span className="font-semibold text-[#92400E]">Metacognitive Abstention Triggered</span>
                          <p className="text-[#B45309] leading-relaxed">
                            {turn.response.abstention_reason || "Decision exceeded acceptable risk or uncertainty bounds."}
                          </p>
                        </div>
                      </div>
                    )}

                    {/* 3. Candidate Rankings (Softmax Bars) */}
                    {turn.response.candidate_scores && Object.keys(turn.response.candidate_scores).length > 0 && (
                      <div className="space-y-2 pt-2 border-t border-[#F5F3ED]">
                        <div className="flex items-center justify-between text-xs font-semibold text-[#1C1917]">
                          <span>Candidate Comparison & Rankings</span>
                          <span className="text-[10px] font-mono text-[#78716C]">Softmax Probability</span>
                        </div>
                        <div className="space-y-2">
                          {Object.entries(turn.response.candidate_scores).map(([cand, score]) => {
                            const isSelected = String(turn.response?.decision) === cand;
                            const scoreNum = score !== null ? score : 0;
                            return (
                              <div key={cand} className="space-y-1">
                                <div className="flex justify-between text-xs font-mono">
                                  <span className={isSelected ? "font-bold text-[#166534] flex items-center gap-1.5" : "text-[#44403C]"}>
                                    {cand}
                                    {isSelected && (
                                      <span className="text-[9px] px-1.5 py-0.2 rounded-full luxury-pill-green font-semibold">
                                        WINNER
                                      </span>
                                    )}
                                  </span>
                                  <span className="text-[#78716C] font-semibold">
                                    {(scoreNum * 100).toFixed(1)}%
                                  </span>
                                </div>
                                <div className="h-1.5 w-full bg-[#FAF9F5] rounded-full overflow-hidden border border-[#E8E5DC]">
                                  <div
                                    className={`h-full rounded-full transition-all duration-500 ${
                                      isSelected ? "bg-[#166534]" : "bg-[#1C1917]/30"
                                    }`}
                                    style={{ width: `${Math.min(100, Math.max(0, scoreNum * 100))}%` }}
                                  />
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    )}

                    {/* 4. Supporting Reasoning & Rules Evidence */}
                    {turn.response.reasoning_evidence && (
                      <div className="space-y-2 pt-2 border-t border-[#F5F3ED]">
                        <span className="text-xs font-semibold text-[#1C1917] block">
                          Reasoning Trace & Grounded Evidence
                        </span>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs">
                          {/* Active Rules */}
                          <div className="p-3 rounded-lg bg-[#FAF9F5] border border-[#E8E5DC] space-y-1">
                            <span className="text-[10px] font-mono text-[#78716C] uppercase font-semibold">
                              Domain Rules Evaluated
                            </span>
                            {turn.response.reasoning_evidence.rule_activations &&
                            turn.response.reasoning_evidence.rule_activations.length > 0 ? (
                              turn.response.reasoning_evidence.rule_activations.map((r, i) => (
                                <div key={i} className="text-[11px] font-mono text-[#1C1917] flex justify-between">
                                  <span>{r.rule_id}</span>
                                  <span className="text-[#78716C]">wt: {r.weight}</span>
                                </div>
                              ))
                            ) : (
                              <div className="text-[11px] text-[#A8A29E] italic">Standard policy rules valid.</div>
                            )}
                          </div>

                          {/* Safety Checks */}
                          <div className="p-3 rounded-lg bg-[#FAF9F5] border border-[#E8E5DC] space-y-1">
                            <span className="text-[10px] font-mono text-[#78716C] uppercase font-semibold">
                              Safety Constraint Verification
                            </span>
                            {turn.response.reasoning_evidence.safety_violations &&
                            turn.response.reasoning_evidence.safety_violations.length > 0 ? (
                              turn.response.reasoning_evidence.safety_violations.map((v, i) => (
                                <div key={i} className="text-[11px] font-mono text-[#9F1239] flex items-center gap-1">
                                  <AlertCircle className="w-3 h-3 flex-shrink-0" />
                                  <span>{v}</span>
                                </div>
                              ))
                            ) : (
                              <div className="text-[11px] text-[#166534] flex items-center gap-1">
                                <CheckCircle2 className="w-3 h-3 flex-shrink-0" />
                                <span>No policy violations detected.</span>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    )}

                    {/* 5. Proposed Multi-Step Plan (if synthesized) */}
                    {turn.response.selected_plan && turn.response.selected_plan.steps && (
                      <div className="space-y-2 pt-2 border-t border-[#F5F3ED]">
                        <span className="text-xs font-semibold text-[#1C1917] flex items-center gap-1.5">
                          <GitCommit className="w-3.5 h-3.5 text-[#1C1917]" strokeWidth={1.75} />
                          Synthesized Execution Plan ({turn.response.selected_plan.steps.length} sequential steps)
                        </span>
                        <div className="space-y-1.5">
                          {turn.response.selected_plan.steps.map((st, i) => (
                            <div
                              key={i}
                              className="p-2.5 rounded-lg bg-[#FAF9F5] border border-[#E8E5DC] text-xs flex items-center justify-between"
                            >
                              <div className="flex items-center gap-2">
                                <span className="font-mono text-[10px] text-[#78716C]">#{i + 1}</span>
                                <span className="font-mono font-semibold text-[#1C1917]">{st.action}</span>
                                {st.requires_approval && (
                                  <span className="text-[9px] px-1.5 py-0.2 rounded-full luxury-pill-amber font-mono font-semibold">
                                    HUMAN GATE
                                  </span>
                                )}
                              </div>
                              {st.estimated_risk !== undefined && (
                                <span className="font-mono text-[10px] text-[#78716C]">
                                  Risk: {(st.estimated_risk * 100).toFixed(0)}%
                                </span>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* 6. Context Provenance Notice */}
                    <div className="pt-2 border-t border-[#F5F3ED] flex items-center justify-between text-[10px] font-mono text-[#A8A29E]">
                      <span>
                        Verified: {turn.response.used_neural_inference !== false ? "158M ModernBERT Neural Core" : "Deterministic Engine"}
                      </span>
                      <button
                        onClick={() => handleCopyJson(turn.response, turn.id)}
                        className="flex items-center gap-1 text-[#78716C] hover:text-[#1C1917] transition cursor-pointer"
                      >
                        {copiedId === turn.id ? <Check className="w-3 h-3 text-[#166534]" /> : <Copy className="w-3 h-3" />}
                        <span>{copiedId === turn.id ? "Copied Raw JSON" : "Copy Payload"}</span>
                      </button>
                    </div>

                    {/* 7. Suggested Follow-Up Actions */}
                    <div className="pt-2 border-t border-[#F5F3ED] space-y-1.5">
                      <span className="text-[10px] font-mono uppercase text-[#78716C] font-semibold tracking-wider flex items-center gap-1">
                        <CornerDownRight className="w-3 h-3" strokeWidth={1.75} />
                        Suggested Follow-Up Inquiries:
                      </span>
                      <div className="flex flex-wrap gap-1.5">
                        {FOLLOW_UP_SUGGESTIONS.map((sug, i) => (
                          <button
                            key={i}
                            onClick={() => handleSubmit(sug)}
                            className="text-[11px] px-2.5 py-1 rounded-lg bg-white border border-[#E8E5DC] text-[#44403C] hover:border-[#1C1917] hover:text-[#1C1917] transition cursor-pointer"
                          >
                            {sug}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                ) : null}
              </div>
            );
          })
        )}
        <div ref={chatBottomRef} />
      </div>

      {/* Input Box & Drawer Controls */}
      <div className="space-y-2 flex-shrink-0 pt-2 border-t border-[#E8E5DC]">
        {/* Context Drawer Toggle */}
        <div className="flex items-center justify-between text-xs">
          <button
            type="button"
            onClick={() => setShowDrawer(!showDrawer)}
            className="flex items-center gap-1.5 text-xs text-[#78716C] hover:text-[#1C1917] font-medium transition cursor-pointer"
          >
            <SlidersHorizontal className="w-3.5 h-3.5" strokeWidth={1.75} />
            <span>Optional Context & Constraints</span>
            {(candidates.length > 0 || prohibited.length > 0 || goals.length > 0 || constraints.length > 0) && (
              <span className="w-2 h-2 rounded-full bg-[#16A34A]" />
            )}
            {showDrawer ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
          </button>

          {(candidates.length > 0 || prohibited.length > 0 || goals.length > 0 || constraints.length > 0) && (
            <button
              onClick={resetContextInputs}
              className="text-[11px] text-[#A8A29E] hover:text-[#9F1239] transition cursor-pointer"
            >
              Clear Context Filters
            </button>
          )}
        </div>

        {/* Expandable Context Parameters */}
        {showDrawer && (
          <div className="luxury-card p-4 rounded-xl space-y-3 text-xs bg-[#FAF9F5] border border-[#E8E5DC] max-h-56 overflow-y-auto">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {/* Candidate Actions */}
              <div className="space-y-1.5">
                <label className="font-semibold text-[#1C1917] text-[11px] flex justify-between">
                  <span>Candidate Actions ({candidates.length})</span>
                  <span className="text-[#A8A29E] font-normal">Press Enter to add</span>
                </label>
                <div className="flex gap-1.5">
                  <input
                    type="text"
                    value={newCandidate}
                    onChange={(e) => setNewCandidate(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && newCandidate.trim()) {
                        e.preventDefault();
                        setCandidates([...candidates, newCandidate.trim()]);
                        setNewCandidate("");
                      }
                    }}
                    placeholder="e.g. act_autoscale"
                    className="flex-1 text-xs px-2.5 py-1 rounded-md luxury-input font-mono"
                  />
                  <button
                    type="button"
                    onClick={() => {
                      if (newCandidate.trim()) {
                        setCandidates([...candidates, newCandidate.trim()]);
                        setNewCandidate("");
                      }
                    }}
                    className="px-2.5 py-1 rounded-md border border-[#E8E5DC] bg-white text-xs cursor-pointer"
                  >
                    +
                  </button>
                </div>
                <div className="flex flex-wrap gap-1">
                  {candidates.map((c, i) => (
                    <span key={i} className="px-2 py-0.5 rounded bg-white border border-[#E8E5DC] text-[10px] font-mono text-[#1C1917] flex items-center gap-1">
                      {c}
                      <button onClick={() => setCandidates(candidates.filter((_, idx) => idx !== i))} className="text-[#A8A29E] hover:text-[#9F1239]">×</button>
                    </span>
                  ))}
                </div>
              </div>

              {/* Prohibited Actions */}
              <div className="space-y-1.5">
                <label className="font-semibold text-[#9F1239] text-[11px] flex justify-between">
                  <span>Prohibited Constraints ({prohibited.length})</span>
                  <span className="text-[#A8A29E] font-normal">Hard barred</span>
                </label>
                <div className="flex gap-1.5">
                  <input
                    type="text"
                    value={newProhibited}
                    onChange={(e) => setNewProhibited(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && newProhibited.trim()) {
                        e.preventDefault();
                        setProhibited([...prohibited, newProhibited.trim()]);
                        setNewProhibited("");
                      }
                    }}
                    placeholder="e.g. act_hard_reboot"
                    className="flex-1 text-xs px-2.5 py-1 rounded-md luxury-input font-mono"
                  />
                  <button
                    type="button"
                    onClick={() => {
                      if (newProhibited.trim()) {
                        setProhibited([...prohibited, newProhibited.trim()]);
                        setNewProhibited("");
                      }
                    }}
                    className="px-2.5 py-1 rounded-md border border-[#FECDD3] bg-[#FFF1F2] text-[#9F1239] text-xs cursor-pointer"
                  >
                    +
                  </button>
                </div>
                <div className="flex flex-wrap gap-1">
                  {prohibited.map((p, i) => (
                    <span key={i} className="px-2 py-0.5 rounded luxury-pill-rose text-[10px] font-mono flex items-center gap-1">
                      {p}
                      <button onClick={() => setProhibited(prohibited.filter((_, idx) => idx !== i))} className="text-[#9F1239]/70 hover:text-[#9F1239]">×</button>
                    </span>
                  ))}
                </div>
              </div>

              {/* Active Goals */}
              <div className="space-y-1.5">
                <label className="font-semibold text-[#1C1917] text-[11px]">
                  Task Goals ({goals.length})
                </label>
                <div className="flex gap-1.5">
                  <input
                    type="text"
                    value={newGoal}
                    onChange={(e) => setNewGoal(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && newGoal.trim()) {
                        e.preventDefault();
                        setGoals([...goals, newGoal.trim()]);
                        setNewGoal("");
                      }
                    }}
                    placeholder="e.g. Restore service under 60s"
                    className="flex-1 text-xs px-2.5 py-1 rounded-md luxury-input"
                  />
                  <button
                    type="button"
                    onClick={() => {
                      if (newGoal.trim()) {
                        setGoals([...goals, newGoal.trim()]);
                        setNewGoal("");
                      }
                    }}
                    className="px-2.5 py-1 rounded-md border border-[#E8E5DC] bg-white text-xs cursor-pointer"
                  >
                    +
                  </button>
                </div>
              </div>

              {/* Explicit Constraints */}
              <div className="space-y-1.5">
                <label className="font-semibold text-[#1C1917] text-[11px]">
                  Explicit Constraints ({constraints.length})
                </label>
                <div className="flex gap-1.5">
                  <input
                    type="text"
                    value={newConstraint}
                    onChange={(e) => setNewConstraint(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && newConstraint.trim()) {
                        e.preventDefault();
                        setConstraints([...constraints, newConstraint.trim()]);
                        setNewConstraint("");
                      }
                    }}
                    placeholder="e.g. Comply with ISO-27001"
                    className="flex-1 text-xs px-2.5 py-1 rounded-md luxury-input"
                  />
                  <button
                    type="button"
                    onClick={() => {
                      if (newConstraint.trim()) {
                        setConstraints([...constraints, newConstraint.trim()]);
                        setNewConstraint("");
                      }
                    }}
                    className="px-2.5 py-1 rounded-md border border-[#E8E5DC] bg-white text-xs cursor-pointer"
                  >
                    +
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Natural Language Question Form */}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSubmit();
          }}
          className="flex gap-2 items-center"
        >
          <div className="relative flex-1">
            <input
              type="text"
              value={inputQuestion}
              onChange={(e) => setInputQuestion(e.target.value)}
              placeholder="Ask a cognitive question, request a decision, or present an operational dilemma..."
              disabled={submitting}
              className="w-full text-xs px-4 py-3 rounded-xl luxury-input shadow-xs pr-10 leading-relaxed placeholder:text-[#A8A29E]"
            />
          </div>

          <button
            type="submit"
            disabled={!inputQuestion.trim() || submitting}
            className="px-5 py-3 rounded-xl luxury-btn text-xs font-semibold flex items-center gap-2 cursor-pointer disabled:opacity-50 transition shadow-sm"
          >
            {submitting ? (
              <>
                <Brain className="w-4 h-4 animate-spin" strokeWidth={1.75} />
                <span>Thinking...</span>
              </>
            ) : (
              <>
                <span>Submit</span>
                <Send className="w-3.5 h-3.5" strokeWidth={1.75} />
              </>
            )}
          </button>
        </form>
      </div>
    </div>
  );
}

export default function CognitiveQAPage() {
  return (
    <Suspense fallback={<div className="p-8 text-center text-xs font-mono text-[#78716C]">Loading Cognitive Q&A...</div>}>
      <CognitiveQAPageContent />
    </Suspense>
  );
}
