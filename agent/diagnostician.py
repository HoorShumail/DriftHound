"""OpenAI-backed diagnosis of DriftHound reports."""

from datetime import datetime, timezone
from pathlib import Path

from agent.client import call_llm, get_client
from agent.prompts import build_diagnosis_prompt


class DriftDiagnostician:
    """Generate concise natural-language explanations of detected drift."""

    def __init__(self, client=None):
        self.client = client if client is not None else get_client()

    def diagnose(self, drift_report, metrics_comparison=None) -> str:
        """Ask the configured language model to explain a drift report."""
        messages = build_diagnosis_prompt(drift_report, metrics_comparison)
        return call_llm(self.client, messages)

    def diagnose_and_save(
        self, drift_report, output_dir, metrics_comparison=None
    ) -> dict:
        """Generate a diagnosis, save it, and return its details."""
        diagnosis = self.diagnose(drift_report, metrics_comparison)
        timestamp = datetime.now(timezone.utc)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        file_path = output / timestamp.strftime("diagnosis_%Y%m%dT%H%M%S%fZ.txt")
        file_path.write_text(diagnosis, encoding="utf-8")
        return {
            "diagnosis": diagnosis,
            "file_path": str(file_path),
            "timestamp": timestamp.isoformat(),
        }
