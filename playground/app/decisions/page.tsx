"use client";

import React, { useState } from "react";
import {
  cdmApi,
  DecisionRequest,
  DecisionResponse,
} from "../../lib/cdm-api";
import {
  GitBranch,
  Play,
  RotateCcw,
  CheckCircle2,
  AlertTriangle,
  Scale,
  Sparkles,
  Percent,
  TrendingUp,
  AlertCircle,
  Copy,
  Check,
  Code2,
  Terminal,
  ShieldCheck,
  ShieldAlert,
} from "lucide-react";

type TaskType = "choice" | "binary" | "score" | "predict";

export default function DecisionExplorerPage() {
  const [task, setTask] = useState<TaskType>("choice");
  const [text, setText] = useState("Critical node failure detected in cluster us-west-2 with high memory pressure.");
  const [featuresJson, setFeaturesJson] = useState('{\n  "urgency_level": 2,\n  "domain": 0,\n  "memory_pressure": 0.94\n}');
  const [candidates, setCandidates] = useState<string[]>([
    "act_restart_service",
    "act_drain_traffic",
    "act_quarantine_node",
  ]);
  const [newCand, setNewCand] = useState("");
  const [prohibited, setProhibited] = useState<string[]>(["act_hard_reboot"]);
  const [newProh, setNewProh] = useState("");

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<DecisionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [showCode, setShowCode] = useState(false);

  const handleRun = async () => {
    if (!text.trim()) {
      setError("Please provide input inquiry or narrative text.");
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

    const req: DecisionRequest = {
      text,
      structured_features: parsedFeatures,
      task,
      candidates: task === "choice" ? candidates : undefined,
      prohibited_actions: prohibited.length > 0 ? prohibited : undefined,
    };

    try {
      let res: DecisionResponse;
      if (task === "binary") res = await cdmApi.classify(req);
      else if (task === "choice") res = await cdmApi.choice(req);
      else if (task === "score") res = await cdmApi.score(req);
      else if (task === "predict") res = await cdmApi.predict(req);
      else res = await cdmApi.createDecision(req);

      setResult(res);
    } catch (err: any) {
      setError(err.message || "Decision request failed");
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = () => {
    if (!result) return;
    navigator.clipboard.writeText(JSON.stringify(result, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const curlSnippet = `curl -X POST http://127.0.0.1:8000/v1/${task === "choice" ? "choice" : task === "binary" ? "classify" : task === "score" ? "score" : "decisions"} \\
  -H "Authorization: Bearer cdm_dev_playground_key" \\
  -H "Content-Type: application/json" \\
  -d '${JSON.stringify({ text, structured_features: JSON.parse(featuresJson || "{}"), task, candidates: task === "choice" ? candidates : undefined }, null, 2)}'`;

  return (
    <div className="max-w-7xl mx-auto space-y-6 pb-12">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#E8E5DC] pb-4">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-semibold text-[#1C1917] tracking-tight">
              Decision Intelligence Explorer
            </h1>
            <span className="text-[11px] font-mono font-medium px-2 py-0.5 rounded-full luxury-pill-neutral">
              POST /v1/decisions
            </span>
          </div>
          <p className="text-xs text-[#78716C] mt-1.5 leading-relaxed">
            Multi-task neural inference gates: softmax candidate ranking, binary classification, continuous scoring, and outcome estimation.
          </p>
        </div>

        {/* Task Modality Switcher */}
        <div className="flex rounded-xl bg-white p-1 border border-[#E8E5DC] shadow-xs">
          {(["choice", "binary", "score", "predict"] as TaskType[]).map((t) => (
            <button
              key={t}
              onClick={() => {
                setTask(t);
                setResult(null);
              }}
              className={`px-3 py-1.5 text-xs font-mono font-medium rounded-lg transition cursor-pointer ${
                task === t
                  ? "bg-[#1C1917] text-white shadow-xs font-semibold"
                  : "text-[#78716C] hover:text-[#1C1917]"
              }`}
            >
              {t.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left Input Form (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          <div className="luxury-card p-4 rounded-xl space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-xs font-semibold text-[#1C1917]">
                Inquiry / Narrative Text
              </label>
              <span className="text-[10px] font-mono luxury-pill-neutral px-2 py-0.5 rounded-full font-semibold">
                Modality: {task}
              </span>
            </div>
            <textarea
              rows={4}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Operational observation text..."
              className="w-full text-xs p-3 rounded-lg luxury-input placeholder:text-[#A8A29E] leading-relaxed resize-y"
            />
          </div>

          <div className="luxury-card p-4 rounded-xl space-y-2">
            <label className="text-xs font-semibold text-[#1C1917]">
              Structured Features (Tabular JSON)
            </label>
            <textarea
              rows={3}
              value={featuresJson}
              onChange={(e) => setFeaturesJson(e.target.value)}
              className="w-full text-xs p-3 rounded-lg luxury-input text-[#1C1917] font-mono"
            />
          </div>

          {task === "choice" && (
            <div className="luxury-card p-4 rounded-xl space-y-3">
              <label className="text-xs font-semibold text-[#1C1917] flex items-center justify-between">
                <span>Candidate Actions</span>
                <span className="text-[10px] font-mono text-[#78716C]">{candidates.length} options</span>
              </label>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={newCand}
                  onChange={(e) => setNewCand(e.target.value)}
                  placeholder="e.g. act_reboot_worker"
                  className="flex-1 text-xs px-3 py-1.5 rounded-lg luxury-input font-mono"
                />
                <button
                  type="button"
                  onClick={() => {
                    if (newCand.trim()) {
                      setCandidates([...candidates, newCand.trim()]);
                      setNewCand("");
                    }
                  }}
                  className="px-3 py-1.5 rounded-lg border border-[#E8E5DC] bg-[#FAF9F5] hover:bg-[#F7F5EE] text-[#1C1917] text-xs font-medium cursor-pointer"
                >
                  Add
                </button>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {candidates.map((c, i) => (
                  <span
                    key={i}
                    className="px-2.5 py-1 rounded-md bg-[#FAF9F5] border border-[#E8E5DC] text-[11px] font-mono text-[#1C1917] flex items-center gap-1.5"
                  >
                    {c}
                    <button
                      onClick={() => setCandidates(candidates.filter((_, idx) => idx !== i))}
                      className="text-[#A8A29E] hover:text-[#9F1239] transition cursor-pointer"
                    >
                      ×
                    </button>
                  </span>
                ))}
              </div>
            </div>
          )}

          <div className="luxury-card p-4 rounded-xl space-y-3">
            <label className="text-xs font-semibold text-[#9F1239] flex items-center justify-between">
              <span className="flex items-center gap-1">
                <ShieldAlert className="w-3.5 h-3.5" strokeWidth={1.75} />
                Prohibited Constraints
              </span>
              <span className="text-[10px] font-mono text-[#9F1239]">{prohibited.length} barred</span>
            </label>
            <div className="flex gap-2">
              <input
                type="text"
                value={newProh}
                onChange={(e) => setNewProh(e.target.value)}
                placeholder="e.g. act_delete_database"
                className="flex-1 text-xs px-3 py-1.5 rounded-lg luxury-input font-mono"
              />
              <button
                type="button"
                onClick={() => {
                  if (newProh.trim()) {
                    setProhibited([...prohibited, newProh.trim()]);
                    setNewProh("");
                  }
                }}
                className="px-3 py-1.5 rounded-lg border border-[#FECDD3] bg-[#FFF1F2] hover:bg-[#FFE4E6] text-[#9F1239] text-xs font-medium cursor-pointer"
              >
                Add
              </button>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {prohibited.map((p, i) => (
                <span
                  key={i}
                  className="px-2.5 py-1 rounded-md luxury-pill-rose text-[11px] font-mono flex items-center gap-1.5"
                >
                  {p}
                  <button
                    onClick={() => setProhibited(prohibited.filter((_, idx) => idx !== i))}
                    className="text-[#9F1239]/70 hover:text-[#9F1239] cursor-pointer"
                  >
                    ×
                  </button>
                </span>
              ))}
            </div>
          </div>

          <button
            onClick={handleRun}
            disabled={loading}
            className="w-full py-3 rounded-xl luxury-btn font-semibold text-xs flex items-center justify-center gap-2 cursor-pointer disabled:opacity-60 transition"
          >
            {loading ? (
              <>
                <Scale className="w-4 h-4 animate-spin" strokeWidth={1.75} />
                Running Neural Inference...
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" strokeWidth={1.75} />
                Submit Decision Request ({task.toUpperCase()})
              </>
            )}
          </button>

          {error && (
            <div className="p-3 luxury-pill-rose rounded-xl text-xs flex items-start gap-2">
              <AlertCircle className="w-4 h-4 text-[#9F1239] flex-shrink-0 mt-0.5" strokeWidth={1.75} />
              <span className="font-mono">{error}</span>
            </div>
          )}
        </div>

        {/* Right Output Area (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          <div className="flex items-center justify-between border-b border-[#E8E5DC] pb-2">
            <span className="text-xs font-semibold text-[#1C1917]">
              Inference Decision Output
            </span>
            <div className="flex items-center gap-3">
              <button
                onClick={() => setShowCode(!showCode)}
                className="flex items-center gap-1 text-xs text-[#78716C] hover:text-[#1C1917] font-mono cursor-pointer"
              >
                <Code2 className="w-3.5 h-3.5" strokeWidth={1.75} />
                <span>{showCode ? "Hide cURL" : "View cURL"}</span>
              </button>

              {result && (
                <button
                  onClick={handleCopy}
                  className="flex items-center gap-1 text-xs text-[#78716C] hover:text-[#1C1917] cursor-pointer"
                >
                  {copied ? <Check className="w-3.5 h-3.5 text-[#166534]" strokeWidth={1.75} /> : <Copy className="w-3.5 h-3.5 text-[#78716C]" strokeWidth={1.75} />}
                  <span>{copied ? "Copied" : "Copy JSON"}</span>
                </button>
              )}
            </div>
          </div>

          {/* cURL Snippet Drawer if toggled */}
          {showCode && (
            <div className="p-3.5 rounded-xl bg-white border border-[#E8E5DC] font-mono text-[11px] text-[#1C1917] overflow-x-auto relative space-y-2 shadow-xs">
              <div className="flex justify-between items-center text-[#78716C] text-[10px]">
                <span className="font-semibold">cURL Terminal Command</span>
                <button
                  onClick={() => {
                    navigator.clipboard.writeText(curlSnippet);
                    setCopied(true);
                    setTimeout(() => setCopied(false), 2000);
                  }}
                  className="hover:text-[#1C1917] cursor-pointer"
                >
                  {copied ? "Copied!" : "Copy"}
                </button>
              </div>
              <pre className="text-[#1C1917]">{curlSnippet}</pre>
            </div>
          )}

          {!result ? (
            <div className="p-16 border border-dashed border-[#E8E5DC] rounded-xl text-center space-y-2 text-[#78716C] bg-white">
              <GitBranch className="w-8 h-8 mx-auto text-[#A8A29E]" strokeWidth={1.75} />
              <p className="text-sm font-semibold text-[#1C1917]">Awaiting Decision Query</p>
              <p className="text-xs text-[#78716C] max-w-sm mx-auto leading-relaxed">
                Submit an inquiry to execute real neural inference across the 158M ModernBERT multi-task architecture.
              </p>
            </div>
          ) : (
            <div className="space-y-4">
              {/* Decision Cards */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="luxury-card p-4 rounded-xl">
                  <div className="text-[10px] font-mono text-[#78716C] uppercase font-semibold">Primary Decision</div>
                  <div className="text-base font-bold text-[#1C1917] truncate mt-1 font-mono">
                    {typeof result.decision === "object"
                      ? JSON.stringify(result.decision)
                      : String(result.decision)}
                  </div>
                </div>

                <div className="luxury-card p-4 rounded-xl">
                  <div className="text-[10px] font-mono text-[#78716C] uppercase font-semibold">Confidence</div>
                  <div className="text-base font-bold text-[#166534] mt-1 font-mono">
                    {(result.confidence * 100).toFixed(1)}%
                  </div>
                </div>

                <div className="luxury-card p-4 rounded-xl">
                  <div className="text-[10px] font-mono text-[#78716C] uppercase font-semibold">Assessed Risk</div>
                  <div className={`text-base font-bold mt-1 font-mono ${result.risk > 0.6 ? "text-[#9F1239]" : "text-[#166534]"}`}>
                    {(result.risk * 100).toFixed(1)}%
                  </div>
                </div>

                <div className="luxury-card p-4 rounded-xl">
                  <div className="text-[10px] font-mono text-[#78716C] uppercase font-semibold">Safety Gate</div>
                  <div className={`text-base font-bold mt-1 font-mono ${result.abstain ? "text-[#92400E]" : "text-[#166534]"}`}>
                    {result.abstain ? "ABSTAINED" : "AUTHORIZED"}
                  </div>
                </div>
              </div>

              {/* Candidate Scores Breakdown (if choice task) */}
              {result.candidate_scores && (
                <div className="luxury-card p-5 rounded-xl space-y-3">
                  <div className="flex items-center justify-between">
                    <h4 className="text-xs font-semibold text-[#1C1917]">
                      Candidate Ranking Probabilities
                    </h4>
                    <span className="text-[10px] font-mono text-[#78716C]">Softmax Normalized</span>
                  </div>
                  <div className="space-y-3 pt-1">
                    {Object.entries(result.candidate_scores).map(([cand, score]) => {
                      const isSelected = result.selected_candidate === cand;
                      const scoreVal = score !== null ? score : 0;
                      return (
                        <div key={cand} className="space-y-1.5">
                          <div className="flex justify-between text-xs font-mono">
                            <span className={isSelected ? "text-[#166534] font-bold flex items-center gap-1.5" : "text-[#44403C]"}>
                              {cand} {isSelected && <span className="text-[10px] px-2 py-0.5 rounded-full luxury-pill-green font-semibold">SELECTED</span>}
                            </span>
                            <span className="text-[#78716C] font-mono font-semibold">
                              {(scoreVal * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="h-2 w-full bg-[#FAF9F5] rounded-full overflow-hidden border border-[#E8E5DC]">
                            <div
                              className={`h-full rounded-full transition-all duration-500 ${
                                isSelected ? "bg-[#166534]" : "bg-[#1C1917]/40"
                              }`}
                              style={{ width: `${Math.min(100, Math.max(0, scoreVal * 100))}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Predicted Outcomes */}
              {result.predicted_outcome && (
                <div className="luxury-card p-5 rounded-xl space-y-3">
                  <h4 className="text-xs font-semibold text-[#1C1917]">
                    Predictive Network Estimations
                  </h4>
                  <div className="grid grid-cols-2 gap-3 text-xs">
                    <div className="p-3 bg-[#FAF9F5] rounded-lg border border-[#E8E5DC]">
                      <span className="text-[#78716C] font-mono text-[10px]">Success Probability</span>
                      <div className="text-base font-bold text-[#166534] mt-1 font-mono">
                        {(result.predicted_outcome.binary_outcome_probability * 100).toFixed(1)}%
                      </div>
                    </div>
                    <div className="p-3 bg-[#FAF9F5] rounded-lg border border-[#E8E5DC]">
                      <span className="text-[#78716C] font-mono text-[10px]">Est. Resolution Time</span>
                      <div className="text-base font-bold text-[#1C1917] mt-1 font-mono">
                        {result.predicted_outcome.predicted_resolution_time.toFixed(1)}s
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
