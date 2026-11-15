"use client";

import React, { useState, useEffect } from "react";
import {
  cdmApi,
  MemoryRecordsResponse,
  WorldStateResponse,
} from "../../lib/cdm-api";
import {
  Database,
  Search,
  BookOpen,
  Workflow,
  Globe,
  RefreshCw,
  AlertTriangle,
  Clock,
  Layers,
  Sparkles,
} from "lucide-react";

type MemoryTab = "episodic" | "rules" | "procedural" | "world_state";

export default function MemoryExplorerPage() {
  const [tab, setTab] = useState<MemoryTab>("episodic");
  const [records, setRecords] = useState<MemoryRecordsResponse | null>(null);
  const [worldState, setWorldState] = useState<WorldStateResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filterText, setFilterText] = useState("");

  const fetchData = async () => {
    setLoading(true);
    try {
      const [recData, wsData] = await Promise.all([
        cdmApi.getMemoryRecords(100).catch(() => null),
        cdmApi.getWorldState("default_session").catch(() => null),
      ]);
      setRecords(recData);
      setWorldState(wsData);
      setError(null);
    } catch (err: any) {
      setError(err.message || "Failed to load memory store data");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const episodicList = records?.episodic || [];
  const rulesList = records?.semantic_rules || [];
  const procList = records?.procedural || [];

  const filteredEpisodic = episodicList.filter(
    (e) =>
      !filterText ||
      e.content?.toLowerCase().includes(filterText.toLowerCase()) ||
      e.task?.toLowerCase().includes(filterText.toLowerCase())
  );

  const filteredRules = rulesList.filter(
    (r) =>
      !filterText ||
      r.rule_id?.toLowerCase().includes(filterText.toLowerCase()) ||
      r.description?.toLowerCase().includes(filterText.toLowerCase())
  );

  return (
    <div className="max-w-7xl mx-auto space-y-6 pb-12">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-[#E8E5DC] pb-4">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="text-xl font-semibold text-[#1C1917] tracking-tight">
              Cognitive Memory & World State
            </h1>
            <span className="text-[11px] font-mono font-medium px-2 py-0.5 rounded-full luxury-pill-neutral">
              GET /v1/memory/records
            </span>
          </div>
          <p className="text-xs text-[#78716C] mt-1.5 leading-relaxed">
            Live inspection of CDM-1 SQLite cognitive persistence: episodic records, semantic domain rules, and world model facts.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={fetchData}
            className="p-2 rounded-lg border border-[#E8E5DC] bg-white hover:bg-[#F7F5EE] text-[#44403C] transition cursor-pointer"
            title="Refresh memory store"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} strokeWidth={1.75} />
          </button>
        </div>
      </div>

      {error && (
        <div className="p-3.5 luxury-pill-rose rounded-xl text-xs flex items-start gap-2.5">
          <AlertTriangle className="w-4 h-4 text-[#9F1239] flex-shrink-0 mt-0.5" strokeWidth={1.75} />
          <span className="font-mono">{error}</span>
        </div>
      )}

      {/* Tabs & Search */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="flex rounded-xl bg-white p-1 border border-[#E8E5DC] shadow-xs">
          <button
            onClick={() => setTab("episodic")}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-mono rounded-lg transition cursor-pointer ${
              tab === "episodic"
                ? "bg-[#1C1917] text-white font-semibold shadow-xs"
                : "text-[#78716C] hover:text-[#1C1917]"
            }`}
          >
            <Layers className="w-3.5 h-3.5" strokeWidth={1.75} />
            <span>Episodic ({episodicList.length})</span>
          </button>
          <button
            onClick={() => setTab("rules")}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-mono rounded-lg transition cursor-pointer ${
              tab === "rules"
                ? "bg-[#1C1917] text-white font-semibold shadow-xs"
                : "text-[#78716C] hover:text-[#1C1917]"
            }`}
          >
            <BookOpen className="w-3.5 h-3.5" strokeWidth={1.75} />
            <span>Rules ({rulesList.length})</span>
          </button>
          <button
            onClick={() => setTab("procedural")}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-mono rounded-lg transition cursor-pointer ${
              tab === "procedural"
                ? "bg-[#1C1917] text-white font-semibold shadow-xs"
                : "text-[#78716C] hover:text-[#1C1917]"
            }`}
          >
            <Workflow className="w-3.5 h-3.5" strokeWidth={1.75} />
            <span>Procedural ({procList.length})</span>
          </button>
          <button
            onClick={() => setTab("world_state")}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-mono rounded-lg transition cursor-pointer ${
              tab === "world_state"
                ? "bg-[#1C1917] text-white font-semibold shadow-xs"
                : "text-[#78716C] hover:text-[#1C1917]"
            }`}
          >
            <Globe className="w-3.5 h-3.5" strokeWidth={1.75} />
            <span>World State</span>
          </button>
        </div>

        {tab !== "world_state" && (
          <div className="relative">
            <Search className="w-3.5 h-3.5 text-[#A8A29E] absolute left-3 top-2.5" strokeWidth={1.75} />
            <input
              type="text"
              value={filterText}
              onChange={(e) => setFilterText(e.target.value)}
              placeholder="Filter memory records..."
              className="text-xs pl-8 pr-3 py-1.5 rounded-lg luxury-input w-64 placeholder:text-[#A8A29E]"
            />
          </div>
        )}
      </div>

      {/* Tab Panels */}
      {tab === "episodic" && (
        <div className="border border-[#E8E5DC] rounded-xl bg-white overflow-hidden shadow-xs">
          {filteredEpisodic.length === 0 ? (
            <div className="p-12 text-center text-xs text-[#78716C]">
              No episodic memory records found. Memory records are automatically persisted during cognitive runs.
            </div>
          ) : (
            <table className="w-full text-xs text-left">
              <thead className="bg-[#FAF9F5] text-[#78716C] border-b border-[#E8E5DC] font-mono">
                <tr>
                  <th className="p-3 font-semibold">Memory ID</th>
                  <th className="p-3 font-semibold">Task</th>
                  <th className="p-3 font-semibold">Content Observation</th>
                  <th className="p-3 font-semibold">Quality</th>
                  <th className="p-3 text-right font-semibold">Created</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#F5F3ED] text-[#44403C]">
                {filteredEpisodic.map((rec) => (
                  <tr key={rec.memory_id} className="hover:bg-[#FAF9F5] transition">
                    <td className="p-3 font-mono font-semibold text-[#1C1917]">{rec.memory_id.slice(0, 10)}...</td>
                    <td className="p-3 font-mono text-[#78716C]">{rec.task || "general"}</td>
                    <td className="p-3 max-w-md font-sans text-[#1C1917] leading-relaxed">{rec.content}</td>
                    <td className="p-3 font-mono font-semibold text-[#166534]">
                      {rec.outcome_quality !== undefined ? rec.outcome_quality.toFixed(2) : "N/A"}
                    </td>
                    <td className="p-3 text-right font-mono text-[#78716C]">
                      {new Date(rec.created_at * 1000).toLocaleTimeString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {tab === "rules" && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {filteredRules.length === 0 ? (
            <div className="col-span-2 p-12 text-center text-xs text-[#78716C] border border-dashed border-[#E8E5DC] bg-white rounded-xl">
              No semantic rules registered.
            </div>
          ) : (
            filteredRules.map((rule, idx) => (
              <div
                key={idx}
                className="luxury-card p-4 rounded-xl space-y-2.5 text-xs"
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono font-bold text-[#1C1917]">{rule.rule_id}</span>
                  <span className="font-mono text-[10px] px-2 py-0.5 rounded-full luxury-pill-neutral font-medium">
                    Weight: {rule.weight}
                  </span>
                </div>
                <p className="text-[#44403C] font-sans leading-relaxed">{rule.description}</p>
                <div className="pt-2.5 border-t border-[#F5F3ED] text-[11px] font-mono space-y-1">
                  <div>
                    <span className="text-[#78716C]">Condition: </span>
                    <span className="text-[#1C1917] font-semibold">{rule.condition}</span>
                  </div>
                  <div>
                    <span className="text-[#78716C]">Action: </span>
                    <span className="text-[#166534] font-semibold">{rule.action}</span>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      )}

      {tab === "procedural" && (
        <div className="space-y-3">
          {procList.length === 0 ? (
            <div className="p-12 text-center text-xs text-[#78716C] border border-dashed border-[#E8E5DC] bg-white rounded-xl">
              No procedural patterns indexed in memory storage.
            </div>
          ) : (
            procList.map((proc, idx) => (
              <div
                key={idx}
                className="luxury-card p-4 rounded-xl space-y-2.5 text-xs"
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono font-bold text-[#1C1917]">{proc.name}</span>
                  <span className="font-mono text-[11px] font-semibold text-[#166534] luxury-pill-green px-2 py-0.5 rounded-full">
                    Success Rate: {(proc.success_rate * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="space-y-1.5 pt-2 border-t border-[#F5F3ED]">
                  {proc.steps.map((s, stepIdx) => (
                    <div key={stepIdx} className="flex items-center gap-2 font-mono text-[11px] text-[#44403C]">
                      <span className="text-[#A8A29E] font-semibold">Step {stepIdx + 1}:</span>
                      <span className="text-[#1C1917]">{s}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))
          )}
        </div>
      )}

      {tab === "world_state" && (
        <div className="space-y-4">
          <div className="luxury-card p-4 rounded-xl space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-semibold text-[#1C1917]">
                Active World State Telemetry (Session: default_session)
              </h3>
              <span className="text-[10px] font-mono luxury-pill-neutral px-2 py-0.5 rounded-full">
                JSON Object
              </span>
            </div>
            {worldState ? (
              <div className="p-4 bg-[#FAF9F5] border border-[#E8E5DC] rounded-lg font-mono text-xs text-[#1C1917] overflow-x-auto max-h-[500px]">
                <pre>{JSON.stringify(worldState.world_state, null, 2)}</pre>
              </div>
            ) : (
              <div className="text-xs text-[#78716C] italic py-8 text-center">No world state retrieved.</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
