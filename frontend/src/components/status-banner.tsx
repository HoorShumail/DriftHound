"use client";

import { Card } from "@/components/ui/card";

interface Status { overall_drift_detected: boolean; features_drifted: number; total_features: number; }
export function StatusBanner({ status }: { status: Status | null }) {
  if (!status) return <Card className="w-full rounded-xl border border-zinc-800 bg-zinc-900 p-5 text-zinc-400">No data yet — run the pipeline</Card>;
  const drift = status.overall_drift_detected;
  return <Card className={`w-full rounded-xl border p-6 ${drift ? "border-red-800/50 bg-linear-to-r from-red-950 to-red-900" : "border-emerald-800/50 bg-linear-to-r from-emerald-950 to-emerald-900"}`}>
    <div className="flex items-center gap-4"><span className={`size-3 shrink-0 rounded-full ${drift ? "animate-pulse bg-red-400" : "bg-emerald-400"}`} /><span className="text-2xl">{drift ? "⚠️" : "✓"}</span><div><p className="font-bold text-white">{drift ? "Drift Detected" : "All Clear"}</p><p className="mt-1 text-sm text-zinc-300">{drift ? `${status.features_drifted} of ${status.total_features} features affected` : "No drift detected"}</p></div></div>
  </Card>;
}
