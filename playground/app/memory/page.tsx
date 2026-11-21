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
  FileText,
  UploadCloud,
  Trash2,
  CheckCircle2,
} from "lucide-react";

type MemoryTab = "episodic" | "rules" | "procedural" | "world_state" | "documents";

export default function MemoryExplorerPage() {
  const [tab, setTab] = useState<MemoryTab>("episodic");
  const [records, setRecords] = useState<MemoryRecordsResponse | null>(null);
  const [worldState, setWorldState] = useState<WorldStateResponse | null>(null);
  const [sessions, setSessions] = useState<any[]>([]);
  const [selectedSessionId, setSelectedSessionId] = useState<string>("default_session");
  const [documents, setDocuments] = useState<any[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState<string | null>(null);
  const [docSearchQuery, setDocSearchQuery] = useState("");
  const [docSearchResults, setDocSearchResults] = useState<any[] | null>(null);
  const [docSearching, setDocSearching] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [filterText, setFilterText] = useState("");

  const fetchData = async (targetSessionId?: string) => {
    setLoading(true);
    const sessToLoad = targetSessionId || selectedSessionId;
    try {
      const [recData, wsData, sessionList, docList] = await Promise.all([
        cdmApi.getMemoryRecords(100).catch(() => null),
        cdmApi.getWorldState(sessToLoad).catch(() => null),
        cdmApi.listSessions(30).catch(() => []),
        cdmApi.listDocuments().catch(() => []),
      ]);
      setRecords(recData);
      setWorldState(wsData);
      setDocuments(docList || []);
      if (sessionList && sessionList.length > 0) {
        setSessions(sessionList);
        if (sessToLoad === "default_session" && !targetSessionId) {
          setSelectedSessionId(sessionList[0].session_id);
          const activeWs = await cdmApi.getWorldState(sessionList[0].session_id).catch(() => null);
          if (activeWs) setWorldState(activeWs);
        }
      }
      setError(null);
    } catch (err: any) {
      setError(err.message || "Failed to load memory store data");
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadSuccess(null);
    setError(null);
    try {
      const res = await cdmApi.uploadDocument(file);
      setUploadSuccess(`Successfully ingested "${res.filename}" (${res.chunk_count} chunks indexed).`);
      const updatedDocs = await cdmApi.listDocuments();
      setDocuments(updatedDocs);
    } catch (err: any) {
      setError(err.message || "Failed to upload document");
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  };

  const handleDeleteDoc = async (docId: string) => {
    try {
      await cdmApi.deleteDocument(docId);
      const updatedDocs = await cdmApi.listDocuments();
      setDocuments(updatedDocs);
    } catch (err: any) {
      setError(err.message || "Failed to delete document");
    }
  };

  const handleDocSearch = async () => {
    if (!docSearchQuery.trim()) return;
    setDocSearching(true);
    try {
      const results = await cdmApi.searchDocuments(docSearchQuery.trim(), 4);
      setDocSearchResults(results);
    } catch (err: any) {
      setError(err.message || "Search failed");
    } finally {
      setDocSearching(false);
    }
  };

  const handleSessionChange = async (newSessionId: string) => {
    setSelectedSessionId(newSessionId);
    setLoading(true);
    try {
      const ws = await cdmApi.getWorldState(newSessionId).catch(() => null);
      setWorldState(ws);
    } catch (err: any) {
      setError(err.message || "Failed to fetch session world state");
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
          {sessions.length > 0 && (
            <div className="flex items-center gap-1.5 bg-white border border-[#E8E5DC] rounded-lg px-2.5 py-1 text-xs">
              <span className="text-[#78716C] font-medium">Session:</span>
              <select
                value={selectedSessionId}
                onChange={(e) => handleSessionChange(e.target.value)}
                className="bg-transparent text-[#1C1917] font-mono text-[11px] focus:outline-none cursor-pointer"
              >
                {sessions.map((s) => (
                  <option key={s.session_id} value={s.session_id}>
                    {s.title} ({s.session_id.slice(0, 10)})
                  </option>
                ))}
              </select>
            </div>
          )}
          <button
            onClick={() => fetchData()}
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
          <button
            onClick={() => setTab("documents")}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-mono rounded-lg transition cursor-pointer ${
              tab === "documents"
                ? "bg-[#1C1917] text-white font-semibold shadow-xs"
                : "text-[#78716C] hover:text-[#1C1917]"
            }`}
          >
            <FileText className="w-3.5 h-3.5" strokeWidth={1.75} />
            <span>Documents ({documents.length})</span>
          </button>
        </div>

        {tab !== "world_state" && tab !== "documents" && (
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
                  <span className="font-mono font-bold text-[#1C1917]">{rule.rule_id || `RULE-${idx + 1}`}</span>
                  <span className="font-mono text-[10px] px-2 py-0.5 rounded-full luxury-pill-neutral font-medium uppercase">
                    {rule.domain || "general"}
                  </span>
                </div>
                <p className="text-[#44403C] font-sans leading-relaxed">
                  {rule.condition_description || rule.description || rule.condition || "Domain constraint policy"}
                </p>
                <div className="pt-2.5 border-t border-[#F5F3ED] text-[11px] font-mono space-y-1">
                  {rule.permitted_actions && rule.permitted_actions.length > 0 && (
                    <div>
                      <span className="text-[#78716C]">Permitted: </span>
                      <span className="text-[#166534] font-semibold">{rule.permitted_actions.join(", ")}</span>
                    </div>
                  )}
                  {rule.prohibited_actions && rule.prohibited_actions.length > 0 && (
                    <div>
                      <span className="text-[#78716C]">Prohibited: </span>
                      <span className="text-[#9F1239] font-semibold">{rule.prohibited_actions.join(", ")}</span>
                    </div>
                  )}
                  {rule.action && (
                    <div>
                      <span className="text-[#78716C]">Action: </span>
                      <span className="text-[#166534] font-semibold">{rule.action}</span>
                    </div>
                  )}
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
            procList.map((proc, idx) => {
              const steps = proc.action_sequence || proc.steps || proc.subtasks || proc.recipe || [];
              const title = proc.procedure_id || proc.name || proc.pattern_id || `Procedure #${idx + 1}`;
              const successRate = proc.success_rate != null ? `${(proc.success_rate * 100).toFixed(0)}%` : "Verified SOP";
              return (
                <div
                  key={idx}
                  className="luxury-card p-4 rounded-xl space-y-2.5 text-xs"
                >
                  <div className="flex items-center justify-between">
                    <span className="font-mono font-bold text-[#1C1917]">{title}</span>
                    <span className="font-mono text-[11px] font-semibold text-[#166534] luxury-pill-green px-2 py-0.5 rounded-full">
                      {successRate}
                    </span>
                  </div>
                  {proc.trigger_condition && (
                    <div className="text-[11px] text-[#78716C]">
                      <span className="font-semibold text-[#44403C]">Trigger: </span>
                      <span>{proc.trigger_condition}</span>
                    </div>
                  )}
                  <div className="space-y-1.5 pt-2 border-t border-[#F5F3ED]">
                    {steps.map((s: any, stepIdx: number) => (
                      <div key={stepIdx} className="flex items-center gap-2 font-mono text-[11px] text-[#44403C]">
                        <span className="text-[#A8A29E] font-semibold">Step {stepIdx + 1}:</span>
                        <span className="text-[#1C1917]">{typeof s === "string" ? s : JSON.stringify(s)}</span>
                      </div>
                    ))}
                  </div>
                  {proc.required_validations && proc.required_validations.length > 0 && (
                    <div className="pt-2 text-[10px] font-mono text-[#78716C]">
                      <span>Required Validations: </span>
                      <span className="text-[#1C1917] font-semibold">{proc.required_validations.join(", ")}</span>
                    </div>
                  )}
                </div>
              );
            })
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

      {tab === "documents" && (
        <div className="space-y-5">
          {/* Ingestion Dropzone */}
          <div className="luxury-card p-6 rounded-2xl border-2 border-dashed border-[#D6D3C7] bg-[#FAF9F5] text-center space-y-3 hover:border-[#1C1917] transition">
            <div className="w-12 h-12 rounded-full bg-white shadow-xs mx-auto flex items-center justify-center text-[#1C1917]">
              <UploadCloud className="w-6 h-6" strokeWidth={1.75} />
            </div>
            <div>
              <h3 className="text-sm font-semibold text-[#1C1917]">
                Ingest Enterprise Document Knowledge
              </h3>
              <p className="text-xs text-[#78716C] mt-1 max-w-md mx-auto">
                Upload <strong>PDF, DOCX, TXT, MD, CSV, or JSON</strong> documents up to 1 MB. Files are automatically extracted, chunked, and indexed into CDM-1 tenant-isolated vector memory.
              </p>
            </div>

            <div className="flex justify-center items-center gap-3 pt-2">
              <label className="cursor-pointer luxury-button-primary px-4 py-2 rounded-xl text-xs font-medium inline-flex items-center gap-2">
                <UploadCloud className="w-4 h-4" strokeWidth={1.75} />
                <span>{uploading ? "Extracting & Indexing..." : "Select File to Ingest"}</span>
                <input
                  type="file"
                  accept=".pdf,.docx,.doc,.txt,.md,.json,.csv"
                  onChange={handleFileUpload}
                  disabled={uploading}
                  className="hidden"
                />
              </label>
            </div>

            {uploadSuccess && (
              <div className="flex items-center justify-center gap-2 text-xs text-[#166534] font-medium pt-2">
                <CheckCircle2 className="w-4 h-4" />
                <span>{uploadSuccess}</span>
              </div>
            )}
          </div>

          {/* Search Indexed Knowledge */}
          <div className="luxury-card p-4 rounded-xl space-y-3">
            <h4 className="text-xs font-semibold text-[#1C1917]">Test Knowledge Retrieval Search</h4>
            <div className="flex gap-2">
              <input
                type="text"
                value={docSearchQuery}
                onChange={(e) => setDocSearchQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleDocSearch()}
                placeholder="Search knowledge base (e.g. 'refund policy', 'incident steps')..."
                className="text-xs px-3 py-2 rounded-lg luxury-input flex-1"
              />
              <button
                onClick={handleDocSearch}
                disabled={docSearching || !docSearchQuery.trim()}
                className="luxury-button-primary px-4 py-2 rounded-lg text-xs font-medium cursor-pointer"
              >
                {docSearching ? "Searching..." : "Search"}
              </button>
            </div>

            {docSearchResults && (
              <div className="space-y-2 pt-2 border-t border-[#F5F3ED]">
                <div className="text-[11px] font-mono text-[#78716C]">
                  Found {docSearchResults.length} relevant chunk(s):
                </div>
                {docSearchResults.map((hit, idx) => (
                  <div key={idx} className="p-3 bg-[#FAF9F5] border border-[#E8E5DC] rounded-lg text-xs space-y-1">
                    <div className="flex items-center justify-between text-[11px] font-mono text-[#78716C]">
                      <span className="font-semibold text-[#1C1917]">{hit.filename} (Chunk #{hit.chunk_index})</span>
                      <span>Score: {(hit.score * 100).toFixed(1)}%</span>
                    </div>
                    <p className="text-[#44403C] text-[11px] leading-relaxed line-clamp-3">
                      {hit.content}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Document Table */}
          <div className="border border-[#E8E5DC] rounded-xl bg-white overflow-hidden shadow-xs">
            <div className="p-3.5 bg-[#FAF9F5] border-b border-[#E8E5DC] flex items-center justify-between">
              <div className="flex items-center gap-2">
                <FileText className="w-4 h-4 text-[#1C1917]" strokeWidth={1.75} />
                <span className="text-xs font-semibold text-[#1C1917]">Indexed Knowledge Repository</span>
                <span className="text-[10px] font-mono luxury-pill-neutral px-2 py-0.5 rounded-full">
                  {documents.length} files
                </span>
              </div>
            </div>

            {documents.length === 0 ? (
              <div className="p-12 text-center text-xs text-[#78716C] space-y-1">
                <p className="font-medium text-[#1C1917]">No enterprise documents uploaded yet.</p>
                <p>Upload files above to provide factual background context for cognitive answer synthesis.</p>
              </div>
            ) : (
              <table className="w-full text-xs text-left">
                <thead className="bg-[#FAF9F5] text-[#78716C] border-b border-[#E8E5DC] font-mono">
                  <tr>
                    <th className="py-2 px-3.5 font-medium">Filename / Title</th>
                    <th className="py-2 px-3.5 font-medium">Format</th>
                    <th className="py-2 px-3.5 font-medium">Size</th>
                    <th className="py-2 px-3.5 font-medium">Chunks</th>
                    <th className="py-2 px-3.5 font-medium">SHA-256</th>
                    <th className="py-2 px-3.5 font-medium text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#F5F3ED] text-[#44403C]">
                  {documents.map((doc) => (
                    <tr key={doc.doc_id} className="hover:bg-[#FAF9F5] transition">
                      <td className="py-2.5 px-3.5 font-medium text-[#1C1917]">
                        <div className="flex items-center gap-2">
                          <FileText className="w-3.5 h-3.5 text-[#78716C]" />
                          <span>{doc.filename}</span>
                        </div>
                      </td>
                      <td className="py-2.5 px-3.5 font-mono text-[11px] uppercase">
                        {doc.file_type}
                      </td>
                      <td className="py-2.5 px-3.5 font-mono text-[11px]">
                        {(doc.file_size_bytes / 1024).toFixed(1)} KB
                      </td>
                      <td className="py-2.5 px-3.5 font-mono text-[11px] font-semibold text-[#166534]">
                        {doc.chunk_count} chunks
                      </td>
                      <td className="py-2.5 px-3.5 font-mono text-[10px] text-[#A8A29E]">
                        {doc.sha256_hash ? `${doc.sha256_hash.slice(0, 12)}...` : "—"}
                      </td>
                      <td className="py-2.5 px-3.5 text-right">
                        <button
                          onClick={() => handleDeleteDoc(doc.doc_id)}
                          className="text-[#9F1239] hover:bg-[#FFE4E6] p-1.5 rounded-md transition cursor-pointer"
                          title="Delete document and remove vector chunks"
                        >
                          <Trash2 className="w-3.5 h-3.5" strokeWidth={1.75} />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
