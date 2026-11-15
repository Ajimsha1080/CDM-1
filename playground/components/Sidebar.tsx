"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  Cpu,
  GitBranch,
  Bot,
  Database,
  Activity,
  Terminal,
  BookOpen,
  ChevronRight,
  ShieldCheck,
  CircleDot,
  X,
  MessageSquareText,
} from "lucide-react";
import { useMobileNav } from "./MobileNavContext";

const NAV_ITEMS = [
  {
    name: "Dashboard",
    href: "/",
    icon: LayoutDashboard,
    badge: null,
  },
  {
    name: "Cognitive Q&A",
    href: "/chat",
    icon: MessageSquareText,
    badge: "New",
  },
  {
    name: "Cognitive Loop",
    href: "/cognitive",
    icon: Cpu,
    badge: "Core",
  },
  {
    name: "Decision Explorer",
    href: "/decisions",
    icon: GitBranch,
    badge: "Multi-Task",
  },
  {
    name: "Agent Runs",
    href: "/agent",
    icon: Bot,
    badge: "Control",
  },
  {
    name: "Memory Store",
    href: "/memory",
    icon: Database,
    badge: null,
  },
  {
    name: "Diagnostics",
    href: "/diagnostics",
    icon: Activity,
    badge: null,
  },
];

export function Sidebar() {
  const pathname = usePathname();
  const [latency, setLatency] = useState<number>(3);
  const { isOpen, close } = useMobileNav();

  useEffect(() => {
    const interval = setInterval(() => {
      setLatency(Math.floor(Math.random() * 3) + 2);
    }, 4000);
    return () => clearInterval(interval);
  }, []);

  // Shared Navigation Content Component
  const NavigationContent = () => (
    <>
      {/* Brand Header */}
      <div className="p-4 border-b border-[#EAE7DF] flex items-center justify-between">
        <Link href="/" onClick={close} className="flex flex-col group">
          <div className="font-bold text-sm tracking-wider text-[#1C1917] flex items-center gap-1.5 uppercase font-mono">
            <span>CDM-1</span>
            <span className="text-[9px] px-1.5 py-0.2 rounded font-mono font-semibold bg-[#F2EFE8] text-[#78716C] border border-[#E3DFD5]">
              v1.0
            </span>
          </div>
          <div className="text-[11px] text-[#78716C] font-sans tracking-tight">
            Cognitive Architecture
          </div>
        </Link>
        {/* Mobile close button */}
        <button
          onClick={close}
          className="md:hidden p-1.5 rounded-lg border border-[#E8E5DC] text-[#78716C] hover:text-[#1C1917] hover:bg-[#F7F5EE]"
          aria-label="Close menu"
        >
          <X className="w-4 h-4" strokeWidth={1.75} />
        </button>
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        <div className="px-3 pb-2 text-[10px] font-mono font-semibold uppercase tracking-widest text-[#A8A29E]">
          Workspaces
        </div>
        {NAV_ITEMS.map((item) => {
          const isActive = pathname === item.href;
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              onClick={close}
              className={`flex items-center justify-between px-3 py-2 rounded-xl text-xs font-medium transition-all group ${
                isActive
                  ? "bg-[#EFECE3] text-[#1C1917] font-semibold border border-[#E0DCCE] shadow-xs"
                  : "text-[#57534E] hover:text-[#1C1917] hover:bg-[#F4F1E9]"
              }`}
            >
              <div className="flex items-center gap-2.5">
                <Icon
                  className={`w-4 h-4 transition-colors ${
                    isActive ? "text-[#1C1917]" : "text-[#78716C] group-hover:text-[#1C1917]"
                  }`}
                  strokeWidth={1.75}
                />
                <span className="tracking-tight">{item.name}</span>
              </div>
              <div className="flex items-center gap-1.5">
                {item.badge && (
                  <span
                    className={`text-[9px] px-1.5 py-0.2 rounded font-mono font-medium ${
                      isActive
                        ? "bg-[#E2DDD0] text-[#44403C] border border-[#D1CAB9]"
                        : "bg-[#EFECE4] text-[#78716C] border border-[#E2DDD2]"
                    }`}
                  >
                    {item.badge}
                  </span>
                )}
              </div>
            </Link>
          );
        })}

        <div className="pt-6 px-3 pb-2 text-[10px] font-mono font-semibold uppercase tracking-widest text-[#A8A29E]">
          Documentation
        </div>
        <a
          href="http://127.0.0.1:8000/docs"
          target="_blank"
          rel="noreferrer"
          className="flex items-center justify-between px-3 py-2 rounded-xl text-xs font-medium text-[#57534E] hover:text-[#1C1917] hover:bg-[#F4F1E9] transition"
        >
          <div className="flex items-center gap-2.5">
            <BookOpen className="w-4 h-4 text-[#78716C]" strokeWidth={1.75} />
            <span>FastAPI Reference</span>
          </div>
          <ChevronRight className="w-3.5 h-3.5 text-[#A8A29E]" strokeWidth={1.75} />
        </a>
      </nav>

      {/* Footer Info */}
      <div className="p-3.5 border-t border-[#EAE7DF] bg-[#F7F5EE] text-xs space-y-2">
        <div className="flex items-center justify-between text-[11px] font-mono">
          <span className="flex items-center gap-1.5 text-[#57534E]">
            <span className="w-2 h-2 rounded-full bg-[#16A34A]" />
            127.0.0.1:8000
          </span>
          <span className="text-[#78716C]">{latency}ms latency</span>
        </div>
        <div className="p-2.5 rounded-xl bg-white border border-[#E5E1D6] text-[10px] text-[#57534E] leading-relaxed font-sans flex items-center justify-between shadow-xs">
          <span className="font-medium text-[#292524]">Neural Production 158M</span>
          <ShieldCheck className="w-3.5 h-3.5 text-[#16A34A]" strokeWidth={1.75} />
        </div>
      </div>
    </>
  );

  return (
    <>
      {/* Desktop Sticky Sidebar (Visible md and above) */}
      <aside className="hidden md:flex w-64 flex-shrink-0 bg-[#FBFBFA] border-r border-[#EAE7DF] flex-col h-screen sticky top-0 text-[#1C1917] select-none z-30">
        <NavigationContent />
      </aside>

      {/* Mobile Drawer Overlay (Active when isOpen on mobile) */}
      {isOpen && (
        <div className="fixed inset-0 z-50 md:hidden flex">
          {/* Backdrop blur overlay */}
          <div
            onClick={close}
            className="fixed inset-0 bg-black/40 backdrop-blur-xs transition-opacity"
            aria-hidden="true"
          />

          {/* Drawer Panel */}
          <aside className="relative w-72 max-w-[80vw] bg-[#FBFBFA] flex flex-col h-full z-10 shadow-2xl border-r border-[#EAE7DF] select-none">
            <NavigationContent />
          </aside>
        </div>
      )}
    </>
  );
}
