import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatTime(timestamp: number | string | undefined): string {
  if (!timestamp) return "N/A";
  const num = typeof timestamp === "string" ? parseFloat(timestamp) : timestamp;
  if (isNaN(num)) return "N/A";
  // Check if timestamp in seconds or milliseconds
  const date = new Date(num > 1e11 ? num : num * 1000);
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

export function formatPercent(val: number | undefined): string {
  if (val === undefined || isNaN(val)) return "0.0%";
  return `${(val * 100).toFixed(1)}%`;
}
