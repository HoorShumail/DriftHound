"use client";

import { Card } from "@/components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

interface MethodResult { drift_detected?: boolean; }
interface Report { results?: Record<string, Record<string, MethodResult>>; }
export function DriftTable({ report }: { report: Report | null }) {
  if (!report?.results) return <Card className="rounded-xl border border-zinc-800 bg-zinc-900 p-6 text-zinc-400">No drift report available</Card>;
  const rows = Object.entries(report.results);
  const driftedCount = rows.filter(([, methods]) => Object.values(methods).some((item) => item?.drift_detected)).length;
  return <Card className="overflow-hidden rounded-xl border border-zinc-800 bg-zinc-950/30 p-0"><Table><TableHeader className="bg-zinc-900"><TableRow><TableHead className="text-xs uppercase tracking-wider text-zinc-400">Feature</TableHead><TableHead className="text-center text-xs uppercase tracking-wider text-zinc-400">KS Test</TableHead><TableHead className="text-center text-xs uppercase tracking-wider text-zinc-400">PSI</TableHead><TableHead className="text-center text-xs uppercase tracking-wider text-zinc-400">Wasserstein</TableHead><TableHead className="text-xs uppercase tracking-wider text-zinc-400">Status</TableHead></TableRow></TableHeader><TableBody>{rows.map(([feature, methods]) => { const checks = [methods.ks, methods.psi, methods.wasserstein]; const isDrifted = checks.some((item) => item?.drift_detected); return <TableRow key={feature} className="even:bg-zinc-900/30"><TableCell className="font-medium">{feature}</TableCell>{checks.map((item, index) => <TableCell key={index} className={`text-center text-lg ${item?.drift_detected ? "bg-red-950/30 text-red-300" : "text-emerald-400"}`}>{item?.drift_detected ? "⚠️" : "✅"}</TableCell>)}<TableCell><span className={`inline-flex rounded-full border px-3 py-1 text-xs font-medium ${isDrifted ? "border-red-500/20 bg-red-500/10 text-red-400" : "border-emerald-500/20 bg-emerald-500/10 text-emerald-400"}`}>{isDrifted ? "Drifted" : "OK"}</span></TableCell></TableRow>; })}</TableBody></Table><div className="border-t border-zinc-800 px-4 py-3 text-sm text-zinc-500">{driftedCount} of {rows.length} features drifted</div></Card>;
}
