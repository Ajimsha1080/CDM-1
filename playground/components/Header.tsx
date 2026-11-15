"use client";

import React, { useState, useEffect } from "react";
import { StatusBadge } from "./StatusBadge";
import { cdmApi } from "../lib/cdm-api";
import { Key, ShieldCheck, Check, Terminal, ExternalLink, Menu } from "lucide-react";
import { useMobileNav } from "./MobileNavContext";

export function Header() {
  const { toggle } = useMobileNav();
  const [apiKey, setApiKey] = useState<string>("");
  const [showKeyModal, setShowKeyModal] = useState(false);
  const [keyInput, setKeyInput] = useState("");
  const [testResult, setTestResult] = useState<string | null>(null);
  const [testing, setTesting] = useState(false);

  useEffect(() => {
    cdmApi.getOrFetchDevKey().then((k) => {
      setApiKey(k);
      setKeyInput(k);
    });
  }, []);

  const handleSaveKey = () => {
    if (keyInput.trim()) {
      cdmApi.setApiKey(keyInput.trim());
      setApiKey(keyInput.trim());
      setShowKeyModal(false);
    }
  };

  const handleTestKey = async () => {
    setTesting(true);
    setTestResult(null);
    try {
      const res = await fetch("http://127.0.0.1:8000/v1/models", {
        headers: { Authorization: `Bearer ${keyInput.trim()}` },
      });
      if (res.ok) {
        setTestResult("Authentication succeeded! Tenant authorized.");
      } else {
        const err = await res.json().catch(() => ({}));
        setTestResult(`Auth Failed: ${err.detail || res.statusText}`);
      }
    } catch (e: any) {
      setTestResult(`Connection Failed: ${e.message}`);
    } finally {
      setTesting(false);
    }
  };

  return (
    <header className="h-14 border-b border-[#EAE7DF] bg-[#FBFBFA]/90 backdrop-blur-md px-4 sm:px-6 flex items-center justify-between sticky top-0 z-20 select-none">
      <div className="flex items-center gap-2.5">
        {/* Mobile menu toggle */}
        <button
          onClick={toggle}
          className="md:hidden p-1.5 rounded-lg border border-[#E8E5DC] text-[#44403C] hover:bg-[#F7F5EE] transition cursor-pointer"
          aria-label="Open navigation menu"
        >
          <Menu className="w-4 h-4" strokeWidth={1.75} />
        </button>

        <div className="flex items-center gap-2 text-xs font-mono">
          <span className="text-[#A8A29E] hidden sm:inline">workspace</span>
          <span className="text-[#D6D3D1] hidden sm:inline">/</span>
          <span className="text-[#1C1917] font-semibold flex items-center gap-1.5 font-sans">
            Local Node
            <span className="w-1.5 h-1.5 rounded-full bg-[#16A34A]" />
          </span>
        </div>
      </div>

      <div className="flex items-center gap-3">
        {/* Status indicator */}
        <StatusBadge />

        {/* Auth / Dev Key Button */}
        <button
          onClick={() => {
            setShowKeyModal(true);
            setTestResult(null);
          }}
          className="flex items-center gap-2 text-xs font-mono px-3 py-1 rounded-full bg-[#F5F2EB] hover:bg-[#EFECE3] border border-[#E5E1D6] text-[#44403C] transition cursor-pointer shadow-xs"
          title="Configure API Token"
        >
          <Key className="w-3.5 h-3.5 text-[#78716C]" strokeWidth={1.75} />
          <span className="hidden sm:inline text-[#78716C] font-sans">Token:</span>
          <span className="font-medium text-[#1C1917]">{apiKey ? `${apiKey.slice(0, 10)}...` : "Dev Key"}</span>
        </button>
      </div>

      {/* Modal for setting custom API Key */}
      {showKeyModal && (
        <div className="fixed inset-0 bg-black/30 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-white border border-[#E7E3D8] rounded-2xl p-6 max-w-md w-full shadow-2xl shadow-stone-900/10 space-y-4">
            <div className="flex items-center justify-between border-b border-[#EAE7DF] pb-3">
              <h3 className="font-semibold text-sm text-[#1C1917] flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-[#1C1917]" strokeWidth={1.75} />
                API Token & Tenant Authentication
              </h3>
              <button
                onClick={() => setShowKeyModal(false)}
                className="text-[#A8A29E] hover:text-[#1C1917] text-sm cursor-pointer"
              >
                ✕
              </button>
            </div>

            <p className="text-xs text-[#78716C] leading-relaxed font-sans">
              CDM-1 uses cryptographic Bearer token authentication against SQLite tenant rate limiters. In development mode, the local engine auto-authorizes{" "}
              <code className="text-[#1C1917] font-mono bg-[#F5F2EB] px-1.5 py-0.5 rounded border border-[#E7E3D8]">
                cdm_dev_playground_key
              </code>
              .
            </p>

            <div className="space-y-1.5">
              <label className="block text-xs font-medium text-[#44403C]">
                Bearer Token / Secret
              </label>
              <input
                type="text"
                value={keyInput}
                onChange={(e) => setKeyInput(e.target.value)}
                placeholder="cdm_..."
                className="w-full text-xs font-mono px-3 py-2 bg-[#FAF9F5] border border-[#E7E3D8] rounded-xl text-[#1C1917] focus:outline-none focus:border-[#1C1917]"
              />
            </div>

            {testResult && (
              <div
                className={`p-2.5 rounded-xl text-xs font-mono ${
                  testResult.includes("succeeded")
                    ? "bg-[#EDF7F2] border border-[#B7E4C7] text-[#166534]"
                    : "bg-[#FEF2F2] border border-[#FECACA] text-[#991B1B]"
                }`}
              >
                {testResult}
              </div>
            )}

            <div className="flex items-center justify-between pt-2 border-t border-[#EAE7DF]">
              <button
                type="button"
                onClick={handleTestKey}
                disabled={testing}
                className="px-3 py-1.5 text-xs font-mono text-[#57534E] hover:text-[#1C1917] bg-[#F5F2EB] rounded-xl border border-[#E7E3D8] transition"
              >
                {testing ? "Probing..." : "Test Token"}
              </button>

              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={() => setKeyInput("cdm_dev_playground_key")}
                  className="px-3 py-1.5 text-xs text-[#78716C] hover:text-[#1C1917] cursor-pointer"
                >
                  Reset Default
                </button>
                <button
                  type="button"
                  onClick={handleSaveKey}
                  className="px-4 py-1.5 text-xs font-semibold luxury-btn rounded-xl transition cursor-pointer"
                >
                  Save Token
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </header>
  );
}
