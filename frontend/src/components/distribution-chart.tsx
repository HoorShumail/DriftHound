"use client";

import { Card } from "@/components/ui/card";
import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

interface DistributionData { baseline: number[]; current: number[]; }
function histogram(values: number[], bins = 30) { if (!values.length) return []; const min = Math.min(...values); const max = Math.max(...values); const width = (max - min || 1) / bins; return Array.from({ length: bins }, (_, index) => ({ bin: +(min + (index + 0.5) * width).toFixed(2), count: values.filter((value) => value >= min + index * width && (index === bins - 1 || value < min + (index + 1) * width)).length })); }
export function DistributionChart({ feature, data, driftDetected = false }: { feature: string; data: DistributionData | null; driftDetected?: boolean }) {
  if (!data) return <Card className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-5 text-zinc-500">No distribution data</Card>;
  const base = histogram(data.baseline); const current = histogram(data.current); const chart = base.map((item, index) => ({ bin: item.bin, baseline: item.count, current: current[index]?.count || 0 }));
  return <Card className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4 transition-colors hover:border-zinc-700"><div className="mb-3 flex items-center justify-between"><h3 className="text-sm font-medium text-zinc-300">{feature}</h3>{driftDetected && <span className="rounded-full border border-red-500/20 bg-red-500/10 px-2 py-0.5 text-xs text-red-400">Drift ⚠️</span>}</div><div className="mb-3 flex gap-4 text-xs text-zinc-500"><span><i className="mr-1 inline-block size-2 rounded-full bg-blue-500" />Baseline</span><span><i className="mr-1 inline-block size-2 rounded-full bg-orange-500" />Current</span></div><div className="h-56"><ResponsiveContainer width="100%" height="100%"><AreaChart data={chart}><CartesianGrid stroke="#27272a" /><XAxis dataKey="bin" tick={{ fill: "#71717a", fontSize: 10 }} /><YAxis tick={{ fill: "#71717a", fontSize: 10 }} /><Tooltip /><Area type="monotone" dataKey="baseline" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.3} /><Area type="monotone" dataKey="current" stroke="#f97316" fill="#f97316" fillOpacity={0.3} /></AreaChart></ResponsiveContainer></div></Card>;
}
