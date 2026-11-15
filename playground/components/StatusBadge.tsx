"use client";

import React, { useEffect, useState } from "react";
import { cdmApi, ReadinessResponse } from "../lib/cdm-api";
import { Activity, CheckCircle2, AlertTriangle, XCircle, RefreshCw } from "lucide-react";

export function StatusBadge() {
  const [ready, setReady] = useState<ReadinessResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const checkStatus = async () => {
    setLoading(true);
    try {
      const data = await cdmApi.getReadiness();
      setReady(data);
      setError(null);
    } catch (err: any) {
      setError(err.message || "Offline");
      setReady(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    checkStatus();
    const interval = setInterval(checkStatus, 15000);
    return () => clearInterval(interval);
  }, []);

  if (loading && !ready && !error) {
    return (
      <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#F5F2EB] text-[#78716C] text-xs font-mono border border-[#E7E3D8] animate-pulse">
        <Activity className="w-3.5 h-3.5 animate-spin" strokeWidth={1.75} />
        <span>Connecting node...</span>
      </div>
    );
  }

  if (error || !ready) {
    return (
      <div
        onClick={checkStatus}
        title={`Error: ${error}. Click to retry.`}
        className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#FEF2F2] text-[#991B1B] text-xs font-mono border border-[#FECACA] cursor-pointer hover:bg-[#FEE2E2] transition shadow-xs"
      >
        <XCircle className="w-3.5 h-3.5 text-[#DC2626]" strokeWidth={1.75} />
        <span>Engine Disconnected</span>
        <RefreshCw className="w-3 h-3 ml-0.5 opacity-60" strokeWidth={1.75} />
      </div>
    );
  }

  const isNeural = ready.mode === "neural" && ready.weights_verified;

  return (
    <div
      onClick={checkStatus}
      title={`CDM-1 ${isNeural ? "Neural" : "Fallback"} Mode (${ready.parameters}). Click to refresh.`}
      className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-mono border cursor-pointer transition shadow-xs ${
        isNeural
          ? "bg-[#EDF7F2] text-[#166534] border-[#B7E4C7] hover:bg-[#E2F4EA]"
          : "bg-[#FFFBEB] text-[#92400E] border-[#FDE68A] hover:bg-[#FEF3C7]"
      }`}
    >
      {isNeural ? (
        <CheckCircle2 className="w-3.5 h-3.5 text-[#16A34A]" strokeWidth={1.75} />
      ) : (
        <AlertTriangle className="w-3.5 h-3.5 text-[#D97706]" strokeWidth={1.75} />
      )}
      <span className="font-semibold tracking-tight">{isNeural ? "Neural 158M" : "Fallback"}</span>
      <span className="opacity-70 text-[11px] hidden sm:inline">({ready.status})</span>
    </div>
  );
}
