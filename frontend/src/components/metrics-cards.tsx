"use client";

import { Card } from "@/components/ui/card";

interface Metrics { accuracy?: number; precision?: number; recall?: number; f1_score?: number; roc_auc?: number; }
interface Comparison { [key: string]: { delta?: number; breached?: boolean } | boolean | undefined; }
export function MetricsCards({ metrics, comparison }: { metrics: Metrics | null; comparison?: Comparison | null }) {
  const entries = [["Accuracy", "accuracy"], ["Precision", "precision"], ["Recall", "recall"], ["F1 Score", "f1_score"], ["ROC AUC", "roc_auc"]] as const;
  return <>{entries.map(([label, key]) => { const value = metrics?.[key]; const change = comparison?.[key]; const delta = typeof change === "object" ? change?.delta : undefined; const quality = value == null ? "Awaiting data" : value > 0.95 ? "Excellent" : value > 0.85 ? "Good" : "Review"; const color = quality === "Excellent" ? "text-emerald-400" : quality === "Good" ? "text-yellow-400" : "text-red-400"; return <Card key={key} className="border border-zinc-800 bg-zinc-900/50 p-5 transition-colors hover:border-zinc-700"><p className="text-xs uppercase tracking-[0.14em] text-zinc-500">{label}</p><p className={`mt-2 text-3xl font-bold tracking-tight ${value == null ? "text-zinc-300" : color}`}>{value == null ? "—" : value.toFixed(4)}</p><div className="mt-2 flex items-center justify-between gap-2"><span className={`text-xs font-medium ${value == null ? "text-zinc-500" : color}`}>{quality}</span>{delta != null && <span className={`text-xs ${delta < 0 ? "text-red-400" : "text-emerald-400"}`}>{delta < 0 ? "↓" : "↑"} {Math.abs(delta).toFixed(4)}</span>}</div></Card>; })}</>;
}
