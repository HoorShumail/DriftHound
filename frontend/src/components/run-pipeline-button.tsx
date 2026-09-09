"use client";

import { useState } from "react";
import { runPipeline } from "@/lib/api";

export function RunPipelineButton() { const [running, setRunning] = useState(false); const [message, setMessage] = useState<string | null>(null); async function run() { setRunning(true); setMessage(null); try { await runPipeline(); setMessage("✓ Pipeline complete"); window.location.reload(); } catch { setMessage("Pipeline failed — check the API server"); } finally { setRunning(false); } } return <div className="flex flex-col items-end gap-1"><button className="rounded-lg border border-zinc-700 bg-zinc-800 px-4 py-2 text-sm font-medium transition-all hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-50" onClick={run} disabled={running}>{running ? <span className="flex items-center gap-2"><span className="size-3 animate-spin rounded-full border-2 border-zinc-400/30 border-t-zinc-100" />Running...</span> : "▶ Run Pipeline"}</button>{message && <span className={`text-xs ${message.startsWith("Pipeline failed") ? "text-red-400" : "text-emerald-400"}`}>{message}</span>}</div>; }
