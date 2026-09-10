"use client";

import { useState } from "react";
import { uploadData } from "@/lib/api";

export function UploadPanel({ onUploadComplete }: { onUploadComplete: (result: any) => void }) {
  const [baselineFile, setBaselineFile] = useState<File | null>(null);
  const [currentFile, setCurrentFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleUpload() {
    if (!baselineFile || !currentFile) return;
    setLoading(true);
    setError(null);
    try {
      const result = await uploadData(baselineFile, currentFile);
      onUploadComplete(result);
    } catch (err: any) {
      setError(err?.message || "Upload failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
      <div className="mb-3 flex items-center justify-between gap-3">
        <div>
          <h2 className="text-xl font-semibold">Upload Your Own Data</h2>
          <p className="mt-1 text-sm text-zinc-500">Analyze a baseline and current CSV with the same drift checks.</p>
        </div>
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        <label className="space-y-2 text-sm text-zinc-300">
          <span>Baseline CSV</span>
          <input type="file" accept=".csv" onChange={(e) => setBaselineFile(e.target.files?.[0] || null)} className="block w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-200 file:mr-3 file:rounded file:border-0 file:bg-zinc-800 file:px-2 file:py-1 file:text-zinc-200" />
        </label>
        <label className="space-y-2 text-sm text-zinc-300">
          <span>Current CSV</span>
          <input type="file" accept=".csv" onChange={(e) => setCurrentFile(e.target.files?.[0] || null)} className="block w-full rounded-lg border border-zinc-700 bg-zinc-950 px-3 py-2 text-sm text-zinc-200 file:mr-3 file:rounded file:border-0 file:bg-zinc-800 file:px-2 file:py-1 file:text-zinc-200" />
        </label>
      </div>
      <div className="mt-4 flex items-center gap-3">
        <button
          onClick={handleUpload}
          disabled={loading || !baselineFile || !currentFile}
          className="rounded-lg border border-zinc-700 bg-zinc-800 px-4 py-2 text-sm font-medium text-zinc-100 transition hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? "Analyzing..." : "Analyze My Data"}
        </button>
      </div>
      {error && <p className="mt-2 text-sm text-red-400">{error}</p>}
    </section>
  );
}
