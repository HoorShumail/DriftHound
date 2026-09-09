const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function request(path: string, options?: RequestInit) {
  const response = await fetch(`${API_BASE}${path}`, options);
  if (!response.ok) {
    let detail = `Request failed: ${response.status}`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {}
    throw new Error(detail);
  }
  return response.json();
}

export const fetchStatus = () => request("/api/status");
export const fetchDriftReport = async () => {
  try { return await request("/api/drift-report"); } catch { return null; }
};
export const fetchMetrics = async () => {
  try { return await request("/api/metrics"); } catch { return null; }
};
export const fetchMetricsComparison = async () => {
  try { return await request("/api/metrics/comparison"); } catch { return null; }
};
export const fetchDiagnosis = () => request("/api/diagnose", { method: "POST" });
export const runPipeline = () => request("/api/run-pipeline", { method: "POST" });
export const fetchDistributions = async () => {
  try { return await request("/api/feature-distributions"); } catch { return null; }
};
