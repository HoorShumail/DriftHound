"use client";

import { useEffect, useState } from "react";
import { DiagnosisPanel } from "@/components/diagnosis-panel";
import { DistributionChart } from "@/components/distribution-chart";
import { DriftTable } from "@/components/drift-table";
import { MetricsCards } from "@/components/metrics-cards";
import { RunPipelineButton } from "@/components/run-pipeline-button";
import { StatusBanner } from "@/components/status-banner";
import { Separator } from "@/components/ui/separator";
import { fetchDistributions, fetchDriftReport, fetchMetrics, fetchMetricsComparison, fetchStatus } from "@/lib/api";

interface DashboardData { loading: boolean; error?: boolean; report?: any; metrics?: any; comparison?: any; distributions?: Record<string, any>; }

export default function Home() {
  const [data, setData] = useState<DashboardData>({ loading: true });
  useEffect(() => {
    Promise.all([fetchStatus(), fetchDriftReport(), fetchMetrics(), fetchMetricsComparison(), fetchDistributions()])
      .then(([status, report, metrics, comparison, distributions]) => setData({ status, report, metrics, comparison, distributions, loading: false } as DashboardData))
      .catch(() => setData({ loading: false, error: true }));
  }, []);
  if (data.loading) return <div className="flex min-h-[70vh] items-center justify-center text-zinc-400">Loading dashboard...</div>;
  const summary = data.report?.summary;
  const status = summary ? { overall_drift_detected: summary.drifted_features?.length > 0, features_drifted: summary.drifted_features?.length || 0, total_features: summary.features_checked || 0 } : null;
  const features = Object.keys(data.distributions || {});
  const updatedAt = data.report?.timestamp ? new Date(data.report.timestamp).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }) : "Not yet";
  const isDrifted = (feature: string) => summary?.per_feature?.[feature]?.drift_detected ?? false;
  return <main className="animate-in fade-in duration-500 space-y-6">
    <nav className="flex flex-wrap items-center justify-between gap-4 border-b border-zinc-800 px-1 pb-5"><div className="flex items-center gap-3"><span className="text-2xl">🐕</span><span className="text-lg font-semibold tracking-tight">DriftHound</span></div><RunPipelineButton /></nav>
    <header className="rounded-xl border border-zinc-800 bg-linear-to-b from-zinc-900 to-zinc-950 px-6 py-7"><p className="text-xs font-semibold uppercase tracking-[0.28em] text-sky-400">Model observability</p><div className="mt-3 flex flex-wrap items-end justify-between gap-4"><div><h1 className="text-4xl font-bold tracking-tight">ML model monitoring</h1><p className="mt-2 text-zinc-400">Drift detection, performance signals, and actionable diagnostics.</p></div><p className="text-xs text-zinc-500">Last updated {updatedAt}</p></div></header>
    {data.error && <p className="rounded-lg border border-red-500/30 bg-red-950/20 p-4 text-red-300">Run pipeline first or check that the API is available.</p>}
    <StatusBanner status={status} />
    <section className="space-y-4"><div><h2 className="text-xl font-semibold">Model Performance</h2><p className="mt-1 text-sm text-zinc-500">Baseline quality across the monitored classification metrics.</p></div><div className="grid grid-cols-2 gap-3 md:grid-cols-5"><MetricsCards metrics={data.metrics} comparison={data.comparison} /></div></section>
    <Separator className="bg-zinc-800" />
    <section className="space-y-4"><div><h2 className="text-xl font-semibold">Feature Drift Analysis</h2><p className="mt-1 text-sm text-zinc-500">Statistical tests compare the latest reference and current samples.</p></div><DriftTable report={data.report} /></section>
    <section className="space-y-4"><div><h2 className="text-xl font-semibold">Distribution Overview</h2><p className="mt-1 text-sm text-zinc-500">Overlaid histograms reveal where current feature shapes diverge.</p></div><div className="grid grid-cols-1 gap-4 md:grid-cols-2">{features.map((feature) => <DistributionChart key={feature} feature={feature} data={data.distributions?.[feature]} driftDetected={isDrifted(feature)} />)}</div></section>
    <section className="space-y-4"><div><h2 className="text-xl font-semibold">AI Diagnostics</h2><p className="mt-1 text-sm text-zinc-500">Investigate root causes and prioritize next actions.</p></div><DiagnosisPanel /></section>
  </main>;
}
