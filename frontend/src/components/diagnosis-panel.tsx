"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import { fetchDiagnosis } from "@/lib/api";
import { Card } from "@/components/ui/card";

interface DiagnosisResponse { diagnosis: string; timestamp: string; }
export function DiagnosisPanel() {
  const [result, setResult] = useState<DiagnosisResponse | null>(null); const [error, setError] = useState<string | null>(null); const [loading, setLoading] = useState(false); const [copied, setCopied] = useState(false);
  async function diagnose() { setLoading(true); setError(null); try { setResult((await fetchDiagnosis()) as DiagnosisResponse); } catch (reason) { setError(reason instanceof Error ? reason.message : "AI diagnosis failed"); } finally { setLoading(false); } }
  async function copyDiagnosis() { if (!result) return; await navigator.clipboard.writeText(result.diagnosis); setCopied(true); window.setTimeout(() => setCopied(false), 1500); }
  return <Card className="rounded-xl border border-blue-800/30 bg-linear-to-b from-blue-950/30 to-zinc-900 p-6"><div className="flex flex-wrap items-center justify-between gap-4"><div><h2 className="text-lg font-semibold">🤖 AI Diagnostics</h2><p className="mt-1 text-sm text-zinc-400">Powered by GPT-4o-mini</p></div><button className="rounded-lg bg-blue-600 px-6 py-3 font-medium text-white transition-all hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-50" onClick={diagnose} disabled={loading}>{loading ? <span className="flex items-center gap-2"><span className="size-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />Analyzing drift patterns...</span> : "Get AI Diagnosis"}</button></div>{error && <p className="mt-4 text-sm text-amber-400">{error}</p>}{result && <div className="mt-4 animate-fade-in rounded-lg bg-zinc-950/50 p-6"><div className="mb-4 flex items-center justify-between gap-3 text-xs text-zinc-500"><span>Generated at {new Date(result.timestamp).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}</span><button className="rounded-md border border-zinc-600 px-3 py-1 text-sm transition-colors hover:bg-zinc-800" onClick={copyDiagnosis}>{copied ? "Copied" : "Copy"}</button></div><div className="prose prose-invert prose-sm max-w-none prose-headings:font-semibold prose-headings:text-zinc-100 prose-p:leading-relaxed prose-p:text-zinc-300 prose-li:text-zinc-300 prose-strong:text-white prose-ul:space-y-1"><ReactMarkdown>{result.diagnosis}</ReactMarkdown></div></div>}</Card>;
}
