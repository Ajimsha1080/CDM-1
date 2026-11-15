"use client";

import React, { useState, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import {
  cdmApi,
  CognitiveProcessRequest,
  CognitiveProcessResponse,
} from "../../lib/cdm-api";
import { TraceVisualizer } from "../../components/TraceVisualizer";
import {
  Play,
  Sparkles,
  Plus,
  Code2,
  Brain,
  ShieldCheck,
  CheckCircle2,
  AlertCircle,
  Copy,
  Check,
  Layers,
  ShieldAlert,
  ArrowRight,
  ChevronRight,
  Sliders,
  Cpu,
} from "lucide-react";

const EXAMPLES: Record<string, Partial<CognitiveProcessRequest>> = {
  financial_fraud: {
    text: "Urgent wire transfer request of $125,000 to newly registered offshore counterparty account with expedited settlement flag.",
    structured_features: {
      amount_usd: 125000,
      domain: 1, // financial
      urgency_level: 2,
      is_new_counterparty: 1,
      velocity_anomaly_score: 0.88,
    },
    candidates: [
      "act_immediate_settlement",
      "act_request_dual_authorization",
      "act_flag_compliance_investigation",
      "act_reject_wire",
    ],
    prohibited_actions: ["act_immediate_settlement"],
    goals: ["Prevent financial loss", "Maintain settlement SLAs"],
    constraints: ["Comply with AML Rule 42", "Require MFA for amounts > $50,000"],
    task: "choice",
  },
  infra_incident: {
    text: "Production Kubernetes node worker-cluster-us-east-1a experiencing OOM kills on core payment gateway pods. CPU load 98%.",
    structured_features: {
      cpu_percent: 98.4,
      memory_percent: 99.2,
      error_rate: 0.14,
      active_connections: 12400,
    },
    candidates: [
      "act_cordon_and_drain_node",
      "act_restart_payment_pods",
      "act_horizontal_autoscaling_burst",
      "act_reboot_host_machine",
    ],
    prohibited_actions: ["act_reboot_host_machine"],
    goals: ["Restore payment processing within 60s", "Zero data loss"],
    constraints: ["Never reboot host without maintenance window approval"],
    task: "choice",
  },
  medical_triage: {
    text: "Emergency department patient presenting with acute chest pain, dyspnea, and elevated troponin levels. History of hypertension.",
    structured_features: {
      systolic_bp: 165,
      heart_rate: 110,
      troponin_positive: 1,
      oxygen_saturation: 93,
    },
    candidates: [
      "act_immediate_ecg_and_cath_lab_prep",
      "act_administer_aspirin_and_monitor",
      "act_discharge_to_outpatient_clinic",
    ],
    prohibited_actions: ["act_discharge_to_outpatient_clinic"],
    goals: ["Rapid cardiac stabilization", "Prevent ischemic injury"],
    constraints: ["Emergency protocol STAT-4"],
    task: "choice",
  },
};

function CognitivePlaygroundContent() {
  const searchParams = useSearchParams();
  const [text, setText] = useState("");
  const [task, setTask] = useState("choice");
  const [featuresJson, setFeaturesJson] = useState("{\n  \"domain\": 1,\n  \"urgency_level\": 2\n}");
  const [candidates, setCandidates] = useState<string[]>([]);
  const [newCandidate, setNewCandidate] = useState("");
  const [prohibited, setProhibited] = useState<string[]>([]);
  const [newProhibited, setNewProhibited] = useState("");
  const [goals, setGoals] = useState<string[]>([]);
  const [newGoal, setNewGoal] = useState("");
  const [constraints, setConstraints] = useState<string[]>([]);
  const [newConstraint, setNewConstraint] = useState("");

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<CognitiveProcessResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<"trace" | "json">("trace");
  const [copied, setCopied] = useState(false);
  const [activePreset, setActivePreset] = useState("financial_fraud");

  const loadExample = (key: string) => {
    const ex = EXAMPLES[key];
    if (!ex) return;
    setActivePreset(key);
    setText(ex.text || "");
    setTask(ex.task || "choice");
    setFeaturesJson(JSON.stringify(ex.structured_features || {}, null, 2));
    setCandidates(ex.candidates || []);
    setProhibited(ex.prohibited_actions || []);
    setGoals(ex.goals || []);
    setConstraints(ex.constraints || []);
    setResult(null);
    setError(null);
  };

  useEffect(() => {
    const exKey = searchParams.get("example") || "financial_fraud";
    loadExample(exKey);
  }, [searchParams]);

  const handleRun = async () => {
    if (!text.trim()) {
      setError("Please provide a task narrative or query text.");
      return;
    }

    let parsedFeatures: Record<string, any> = {};
    try {
      if (featuresJson.trim()) {
        parsedFeatures = JSON.parse(featuresJson);
      }
    } catch {
      setError("Structured Features must be valid JSON.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const res = await cdmApi.processCognitive({
        text,
        structured_features: parsedFeatures,
        task,
        candidates: candidates.length > 0 ? candidates : undefined,
        prohibited_actions: prohibited.length > 0 ? prohibited : undefined,
        goals: goals.length > 0 ? goals : undefined,
        constraints: constraints.length > 0 ? constraints : undefined,
        session_id: "playground_session_" + Date.now().toString(36),
      });
      setResult(res);
    } catch (err: any) {
      setError(err.message || "Failed to execute cognitive pipeline");
    } finally {
      setLoading(false);
    }
  };

  const handleCopyJson = () => {
    if (!result) return;
    navigator.clipboard.writeText(JSON.stringify(result, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="max-w-7xl mx-auto space-y-6 pb-12">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#E8E5DC] pb-4">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-semibold text-[#1C1917] tracking-tight">
              Cognitive Engine Playground
            </h1>
            <span className="text-[11px] font-mono font-medium px-2 py-0.5 rounded-full luxury-pill-neutral">
              POST /v1/cognitive/process
            </span>
          </div>
          <p className="text-xs text-[#78716C] mt-1.5 leading-relaxed">
            Full 5-stage loop: Perception, Working Memory, Neural Reasoner, Planning Engine, and Safety Verifier.
          </p>
        </div>

        {/* Quick Example Presets */}
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-mono text-[#A8A29E] uppercase tracking-wider">Presets:</span>
          <button
            onClick={() => loadExample("financial_fraud")}
            className={`px-3 py-1.5 text-xs font-medium rounded-lg border transition ${
              activePreset === "financial_fraud"
                ? "bg-[#1C1917] text-white border-[#1C1917]"
                : "bg-white hover:bg-[#F7F5EE] border-[#E8E5DC] text-[#44403C]"
            }`}
          >
            Financial Wire
          </button>
          <button
            onClick={() => loadExample("infra_incident")}
            className={`px-3 py-1.5 text-xs font-medium rounded-lg border transition ${
              activePreset === "infra_incident"
                ? "bg-[#1C1917] text-white border-[#1C1917]"
                : "bg-white hover:bg-[#F7F5EE] border-[#E8E5DC] text-[#44403C]"
            }`}
          >
            Cloud Incident
          </button>
          <button
            onClick={() => loadExample("medical_triage")}
            className={`px-3 py-1.5 text-xs font-medium rounded-lg border transition ${
              activePreset === "medical_triage"
                ? "bg-[#1C1917] text-white border-[#1C1917]"
                : "bg-white hover:bg-[#F7F5EE] border-[#E8E5DC] text-[#44403C]"
            }`}
          >
            Medical Triage
          </button>
        </div>
      </div>

      {/* Main Grid: Left Input Form, Right Execution Trace */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Form (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          {/* Narrative / Context Input */}
          <div className="luxury-card p-4 rounded-xl space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-xs font-semibold text-[#1C1917] flex items-center gap-1.5">
                <span className="w-4 h-4 rounded-full bg-[#F7F5EE] border border-[#E8E5DC] text-[#78716C] text-[10px] font-mono flex items-center justify-center font-bold">1</span>
                Task Narrative & Context
              </label>
              <span className="text-[10px] text-[#A8A29E] font-mono">
                {text.length} / 8192 chars
              </span>
            </div>
            <textarea
              rows={4}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Describe the operational scenario, agent prompt, or incident..."
              className="w-full text-xs p-3 rounded-lg luxury-input placeholder:text-[#A8A29E] leading-relaxed resize-y"
            />
          </div>

          {/* Structured Features (JSON) */}
          <div className="luxury-card p-4 rounded-xl space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-xs font-semibold text-[#1C1917] flex items-center gap-1.5">
                <span className="w-4 h-4 rounded-full bg-[#F7F5EE] border border-[#E8E5DC] text-[#78716C] text-[10px] font-mono flex items-center justify-center font-bold">2</span>
                <Code2 className="w-3.5 h-3.5 text-[#78716C]" strokeWidth={1.75} />
                Structured Features (Tabular JSON)
              </label>
            </div>
            <textarea
              rows={3}
              value={featuresJson}
              onChange={(e) => setFeaturesJson(e.target.value)}
              placeholder='{"urgency_level": 2, "amount_usd": 50000}'
              className="w-full text-xs p-3 rounded-lg luxury-input text-[#1C1917] font-mono placeholder:text-[#A8A29E] leading-relaxed"
            />
          </div>

          {/* Candidates & Constraints */}
          <div className="luxury-card p-4 rounded-xl space-y-4">
            {/* Candidates */}
            <div className="space-y-2">
              <label className="text-xs font-semibold text-[#1C1917] flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <span className="w-4 h-4 rounded-full bg-[#F7F5EE] border border-[#E8E5DC] text-[#78716C] text-[10px] font-mono flex items-center justify-center font-bold">3</span>
                  Candidate Actions
                </span>
                <span className="text-[10px] text-[#78716C] font-mono">
                  {candidates.length} options
                </span>
              </label>
              <div className="flex gap-2">
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
                  placeholder="e.g. act_isolate_node"
                  className="flex-1 text-xs px-3 py-1.5 rounded-lg luxury-input font-mono"
                />
                <button
                  type="button"
                  onClick={() => {
                    if (newCandidate.trim()) {
                      setCandidates([...candidates, newCandidate.trim()]);
                      setNewCandidate("");
                    }
                  }}
                  className="px-3 py-1.5 rounded-lg border border-[#E8E5DC] bg-[#FAF9F5] hover:bg-[#F7F5EE] text-[#1C1917] text-xs font-medium transition cursor-pointer"
                >
                  <Plus className="w-3.5 h-3.5" strokeWidth={1.75} />
                </button>
              </div>
              <div className="flex flex-wrap gap-1.5 max-h-24 overflow-y-auto">
                {candidates.map((cand, idx) => (
                  <span
                    key={idx}
                    className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-[#FAF9F5] border border-[#E8E5DC] text-[11px] font-mono text-[#1C1917]"
                  >
                    {cand}
                    <button
                      type="button"
                      onClick={() => setCandidates(candidates.filter((_, i) => i !== idx))}
                      className="text-[#A8A29E] hover:text-[#9F1239] transition"
                    >
                      ×
                    </button>
                  </span>
                ))}
              </div>
            </div>

            {/* Prohibited Actions */}
            <div className="space-y-2 pt-3 border-t border-[#E8E5DC]">
              <label className="text-xs font-semibold text-[#9F1239] flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <span className="w-4 h-4 rounded-full bg-[#FFF1F2] border border-[#FECDD3] text-[#9F1239] text-[10px] font-mono flex items-center justify-center font-bold">4</span>
                  <ShieldAlert className="w-3.5 h-3.5 text-[#9F1239]" strokeWidth={1.75} />
                  Prohibited Actions (Safety Constraints)
                </span>
                <span className="text-[10px] text-[#9F1239] font-mono">
                  {prohibited.length} barred
                </span>
              </label>
              <div className="flex gap-2">
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
                  placeholder="e.g. act_reboot_host"
                  className="flex-1 text-xs px-3 py-1.5 rounded-lg luxury-input font-mono"
                />
                <button
                  type="button"
                  onClick={() => {
                    if (newProhibited.trim()) {
                      setProhibited([...prohibited, newProhibited.trim()]);
                      setNewProhibited("");
                    }
                  }}
                  className="px-3 py-1.5 rounded-lg border border-[#FECDD3] bg-[#FFF1F2] hover:bg-[#FFE4E6] text-[#9F1239] text-xs font-medium transition cursor-pointer"
                >
                  <Plus className="w-3.5 h-3.5" strokeWidth={1.75} />
                </button>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {prohibited.map((p, idx) => (
                  <span
                    key={idx}
                    className="flex items-center gap-1.5 px-2.5 py-1 rounded-md luxury-pill-rose text-[11px] font-mono"
                  >
                    {p}
                    <button
                      type="button"
                      onClick={() => setProhibited(prohibited.filter((_, i) => i !== idx))}
                      className="text-[#9F1239]/70 hover:text-[#9F1239] transition"
                    >
                      ×
                    </button>
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* Action Button */}
          <button
            onClick={handleRun}
            disabled={loading}
            className="w-full py-3 rounded-xl luxury-btn font-semibold text-xs flex items-center justify-center gap-2 cursor-pointer disabled:opacity-60 transition"
          >
            {loading ? (
              <>
                <Brain className="w-4 h-4 animate-spin" strokeWidth={1.75} />
                Executing Cognitive Engine Loop...
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" strokeWidth={1.75} />
                Execute POST /v1/cognitive/process
              </>
            )}
          </button>

          {error && (
            <div className="p-3 luxury-pill-rose rounded-lg text-xs flex items-start gap-2">
              <AlertCircle className="w-4 h-4 text-[#9F1239] flex-shrink-0 mt-0.5" strokeWidth={1.75} />
              <span className="font-mono">{error}</span>
            </div>
          )}
        </div>

        {/* Right Output Area (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="flex items-center justify-between border-b border-[#E8E5DC] pb-2">
            <div className="flex gap-1.5">
              <button
                onClick={() => setActiveTab("trace")}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition cursor-pointer ${
                  activeTab === "trace"
                    ? "bg-[#1C1917] text-white"
                    : "text-[#78716C] hover:text-[#1C1917] hover:bg-[#F7F5EE]"
                }`}
              >
                Cognitive Execution Trace
              </button>
              <button
                onClick={() => setActiveTab("json")}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium transition cursor-pointer ${
                  activeTab === "json"
                    ? "bg-[#1C1917] text-white"
                    : "text-[#78716C] hover:text-[#1C1917] hover:bg-[#F7F5EE]"
                }`}
              >
                Raw API Response
              </button>
            </div>

            {result && (
              <div className="flex items-center gap-3">
                <span className="text-xs font-mono text-[#78716C]">
                  Latency: <span className="font-semibold text-[#1C1917]">{result.latency_ms} ms</span>
                </span>
                <button
                  onClick={handleCopyJson}
                  className="flex items-center gap-1.5 px-2.5 py-1 rounded-md border border-[#E8E5DC] bg-white text-xs text-[#44403C] hover:bg-[#F7F5EE] transition cursor-pointer"
                >
                  {copied ? <Check className="w-3.5 h-3.5 text-[#166534]" strokeWidth={1.75} /> : <Copy className="w-3.5 h-3.5 text-[#78716C]" strokeWidth={1.75} />}
                  <span>{copied ? "Copied" : "Copy"}</span>
                </button>
              </div>
            )}
          </div>

          {activeTab === "trace" ? (
            <TraceVisualizer result={result} loading={loading} />
          ) : (
            <div className="p-4 bg-white border border-[#E8E5DC] rounded-xl font-mono text-xs overflow-x-auto max-h-[600px] text-[#1C1917] shadow-sm">
              {result ? (
                <pre>{JSON.stringify(result, null, 2)}</pre>
              ) : (
                <div className="text-[#A8A29E] italic py-8 text-center">No execution output yet. Execute a task to view the response payload.</div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function CognitivePlaygroundPage() {
  return (
    <Suspense fallback={<div className="p-8 text-center text-xs font-mono text-[#78716C]">Loading Cognitive Playground...</div>}>
      <CognitivePlaygroundContent />
    </Suspense>
  );
}
