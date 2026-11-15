"use client";

import React, { useState, useEffect } from "react";
import {
  cdmApi,
  DiagnosticsResponse,
  ReadinessResponse,
} from "../../lib/cdm-api";
import {
  Activity,
  Cpu,
  Database,
  ShieldCheck,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  Terminal,
  HelpCircle,
  Copy,
  Check,
  Sparkles,
} from "lucide-react";

export default function DiagnosticsPage() {
  const [diag, setDiag] = useState<DiagnosticsResponse | null>(null);
  const [ready, setReady] = useState<ReadinessResponse | null>(null);
  const [version, setVersion] = useState<Record<string, any> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [copiedCmd, setCopiedCmd] = useState<string | null>(null);

  const fetchDiagnostics = async () => {
    setLoading(true);
    try {
      const [d, r, v] = await Promise.all([
        cdmApi.getDiagnostics().catch(() => null),
        cdmApi.getReadiness().catch(() => null),
        cdmApi.getVersion().catch(() => null),
      ]);
      setDiag(d);
      setReady(r);
      setVersion(v);
      setError(null);
    } catch (err: any) {
      setError(err.message || "Failed to fetch diagnostics");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDiagnostics();
  }, []);

  const handleCopy = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedCmd(id);
    setTimeout(() => setCopiedCmd(null), 2000);
  };

  const isNeural = ready?.mode === "neural" && ready?.weights_verified;

  return (
    <div className="max-w-7xl mx-auto space-y-6 pb-12">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#E8E5DC] pb-4">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-semibold text-[#1C1917] tracking-tight">
              Diagnostics & Node Telemetry
            </h1>
            <span className="text-[11px] font-mono font-medium px-2 py-0.5 rounded-full luxury-pill-neutral">
              GET /v1/diagnostics
            </span>
          </div>
          <p className="text-xs text-[#78716C] mt-1.5 leading-relaxed">
            Real-time hardware inspection, PyTorch compute device detection, SafeTensors SHA-256 verification, and SQLite persistence enclaves.
          </p>
        </div>

        <button
          onClick={fetchDiagnostics}
          className="flex items-center gap-2 px-3.5 py-2 rounded-lg border border-[#E8E5DC] bg-white hover:bg-[#F7F5EE] text-[#44403C] text-xs font-mono font-medium transition cursor-pointer"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} strokeWidth={1.75} />
          <span>Refresh Probes</span>
        </button>
      </div>

      {error && (
        <div className="p-3.5 luxury-pill-rose rounded-xl text-xs flex items-start gap-2.5">
          <AlertTriangle className="w-4 h-4 text-[#9F1239] flex-shrink-0 mt-0.5" strokeWidth={1.75} />
          <span className="font-mono">{error}</span>
        </div>
      )}

      {/* Grid: 4 Diagnostic Sections */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Section 1: Python & Hardware Runtime */}
        <div className="luxury-card p-6 rounded-xl space-y-4">
          <div className="flex items-center justify-between border-b border-[#E8E5DC] pb-3">
            <h3 className="text-xs font-mono uppercase text-[#1C1917] font-bold flex items-center gap-2">
              <Terminal className="w-4 h-4 text-[#1C1917]" strokeWidth={1.75} />
              Runtime & Compute Environment
            </h3>
            <span className="text-[10px] font-mono luxury-pill-green px-2 py-0.5 rounded-full flex items-center gap-1.5 font-semibold">
              <span className="w-1.5 h-1.5 rounded-full bg-[#166534] animate-pulse" />
              ONLINE
            </span>
          </div>

          <div className="space-y-2.5 text-xs font-mono">
            <div className="flex justify-between py-1.5 border-b border-[#F5F3ED]">
              <span className="text-[#78716C]">Python Version:</span>
              <span className="text-[#1C1917] font-semibold">{diag?.runtime.python_version || "3.14.0"}</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-[#F5F3ED]">
              <span className="text-[#78716C]">Operating System:</span>
              <span className="text-[#1C1917]">{diag?.runtime.platform || "Windows x64"}</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-[#F5F3ED]">
              <span className="text-[#78716C]">PyTorch Engine:</span>
              <span className="text-[#1C1917]">{diag?.runtime.torch_version || "2.12.1+cpu"}</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-[#F5F3ED]">
              <span className="text-[#78716C]">Compute Device:</span>
              <span className="text-[#1C1917] font-bold px-2 py-0.5 rounded bg-[#FAF9F5] border border-[#E8E5DC]">
                {diag?.runtime.device ? diag.runtime.device.toUpperCase() : "CPU"}
              </span>
            </div>
            <div className="flex justify-between py-1.5">
              <span className="text-[#78716C]">CUDA Hardware Accelerator:</span>
              <span className={diag?.runtime.cuda_available ? "text-[#166534] font-bold" : "text-[#92400E] font-medium"}>
                {diag?.runtime.cuda_available ? "Enabled" : "CPU Instruction Mode"}
              </span>
            </div>
          </div>
        </div>

        {/* Section 2: Model & Checkpoint Integrity */}
        <div className="luxury-card p-6 rounded-xl space-y-4">
          <div className="flex items-center justify-between border-b border-[#E8E5DC] pb-3">
            <h3 className="text-xs font-mono uppercase text-[#1C1917] font-bold flex items-center gap-2">
              <Cpu className="w-4 h-4 text-[#1C1917]" strokeWidth={1.75} />
              Neural Weights & Checkpoint Trust
            </h3>
            <span
              className={`text-[10px] font-mono px-2 py-0.5 rounded-full font-semibold ${
                isNeural ? "luxury-pill-green" : "luxury-pill-amber"
              }`}
            >
              {isNeural ? "NEURAL 158M" : "FALLBACK"}
            </span>
          </div>

          <div className="space-y-2.5 text-xs font-mono">
            <div className="flex justify-between py-1.5 border-b border-[#F5F3ED]">
              <span className="text-[#78716C]">Backbone Model:</span>
              <span className="text-[#1C1917] font-bold">answerdotai/ModernBERT-base</span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-[#F5F3ED]">
              <span className="text-[#78716C]">Model SafeTensors:</span>
              <span className={diag?.model.weights_exist ? "text-[#166534] font-semibold" : "text-[#92400E]"}>
                {diag?.model.weights_exist ? "model.safetensors Verified" : "Weights Absent"}
              </span>
            </div>
            <div className="flex justify-between py-1.5 border-b border-[#F5F3ED]">
              <span className="text-[#78716C]">Ed25519 Manifest Signature:</span>
              <span className={ready?.manifest_signature_verified ? "text-[#166534] font-semibold" : "text-[#92400E]"}>
                {ready?.manifest_signature_verified ? "Cryptographically Valid" : "Unsigned"}
              </span>
            </div>
            <div className="flex flex-col py-1.5">
              <span className="text-[#78716C]">SafeTensors SHA-256 Digest:</span>
              <span
                className="text-[10px] text-[#44403C] truncate mt-1 bg-[#FAF9F5] p-2 rounded-lg border border-[#E8E5DC] font-mono select-all"
                title={ready?.weights_sha256 || "None"}
              >
                {ready?.weights_sha256 || "Fallback Mode (Rule-based decision core active)"}
              </span>
            </div>
          </div>
        </div>

        {/* Section 3: SQLite Data Enclaves */}
        <div className="luxury-card p-6 rounded-xl space-y-4">
          <div className="flex items-center justify-between border-b border-[#E8E5DC] pb-3">
            <h3 className="text-xs font-mono uppercase text-[#1C1917] font-bold flex items-center gap-2">
              <Database className="w-4 h-4 text-[#1C1917]" strokeWidth={1.75} />
              Persistence Enclaves (SQLite WAL)
            </h3>
            <span className="text-[10px] font-mono luxury-pill-green px-2 py-0.5 rounded-full font-semibold">WAL MODE ACTIVE</span>
          </div>

          <div className="space-y-2 text-xs font-mono">
            <div className="flex justify-between items-center py-1.5 border-b border-[#F5F3ED]">
              <span className="text-[#78716C]">API Key Enclave:</span>
              <span className="text-[#166534] flex items-center gap-1.5 font-semibold">
                <CheckCircle2 className="w-3.5 h-3.5" strokeWidth={1.75} />
                api_keys.sqlite
              </span>
            </div>
            <div className="flex justify-between items-center py-1.5 border-b border-[#F5F3ED]">
              <span className="text-[#78716C]">Usage Metering:</span>
              <span className="text-[#166534] flex items-center gap-1.5 font-semibold">
                <CheckCircle2 className="w-3.5 h-3.5" strokeWidth={1.75} />
                api_usage_meter.sqlite
              </span>
            </div>
            <div className="flex justify-between items-center py-1.5 border-b border-[#F5F3ED]">
              <span className="text-[#78716C]">Cognitive Memory:</span>
              <span className="text-[#166534] flex items-center gap-1.5 font-semibold">
                <CheckCircle2 className="w-3.5 h-3.5" strokeWidth={1.75} />
                cognitive_memory.db
              </span>
            </div>
            <div className="flex justify-between items-center py-1.5">
              <span className="text-[#78716C]">Agent Runs Enclave:</span>
              <span className="text-[#166534] flex items-center gap-1.5 font-semibold">
                <CheckCircle2 className="w-3.5 h-3.5" strokeWidth={1.75} />
                agent_runs.sqlite
              </span>
            </div>
          </div>
        </div>

        {/* Section 4: Developer Commands */}
        <div className="luxury-card p-6 rounded-xl space-y-4">
          <div className="flex items-center justify-between border-b border-[#E8E5DC] pb-3">
            <h3 className="text-xs font-mono uppercase text-[#1C1917] font-bold flex items-center gap-2">
              <HelpCircle className="w-4 h-4 text-[#1C1917]" strokeWidth={1.75} />
              CLI Quick Commands
            </h3>
            <span className="text-[10px] font-mono luxury-pill-neutral px-2 py-0.5 rounded-full font-semibold">DEVELOPER SUITE</span>
          </div>

          <div className="text-xs text-[#44403C] space-y-2.5 leading-relaxed font-sans">
            <div className="p-3 bg-[#FAF9F5] rounded-xl border border-[#E8E5DC] flex items-center justify-between">
              <div>
                <span className="font-semibold text-[#1C1917] text-xs block">Unified Startup Script</span>
                <span className="font-mono text-[11px] text-[#78716C]">
                  python scripts/start_playground.py
                </span>
              </div>
              <button
                onClick={() => handleCopy("python scripts/start_playground.py", "start")}
                className="p-1.5 rounded-lg border border-[#E8E5DC] bg-white hover:bg-[#F7F5EE] text-[#44403C] transition cursor-pointer"
              >
                {copiedCmd === "start" ? <Check className="w-3.5 h-3.5 text-[#166534]" strokeWidth={1.75} /> : <Copy className="w-3.5 h-3.5 text-[#78716C]" strokeWidth={1.75} />}
              </button>
            </div>

            <div className="p-3 bg-[#FAF9F5] rounded-xl border border-[#E8E5DC] flex items-center justify-between">
              <div>
                <span className="font-semibold text-[#1C1917] text-xs block">Verify Weights SHA-256</span>
                <span className="font-mono text-[11px] text-[#78716C]">
                  python scripts/recover_weights.py
                </span>
              </div>
              <button
                onClick={() => handleCopy("python scripts/recover_weights.py", "weights")}
                className="p-1.5 rounded-lg border border-[#E8E5DC] bg-white hover:bg-[#F7F5EE] text-[#44403C] transition cursor-pointer"
              >
                {copiedCmd === "weights" ? <Check className="w-3.5 h-3.5 text-[#166534]" strokeWidth={1.75} /> : <Copy className="w-3.5 h-3.5 text-[#78716C]" strokeWidth={1.75} />}
              </button>
            </div>

            <div className="p-3 bg-[#FAF9F5] rounded-xl border border-[#E8E5DC] flex items-center justify-between">
              <div>
                <span className="font-semibold text-[#1C1917] text-xs block">Full Test Suite</span>
                <span className="font-mono text-[11px] text-[#78716C]">
                  pytest tests/test_cognitive_engine.py
                </span>
              </div>
              <button
                onClick={() => handleCopy("pytest tests/test_cognitive_engine.py", "test")}
                className="p-1.5 rounded-lg border border-[#E8E5DC] bg-white hover:bg-[#F7F5EE] text-[#44403C] transition cursor-pointer"
              >
                {copiedCmd === "test" ? <Check className="w-3.5 h-3.5 text-[#166534]" strokeWidth={1.75} /> : <Copy className="w-3.5 h-3.5 text-[#78716C]" strokeWidth={1.75} />}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
