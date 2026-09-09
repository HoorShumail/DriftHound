"""Prompts used by DriftHound's diagnostic agent."""

import json


SYSTEM_PROMPT = """You are DriftHound's diagnostic agent. Analyze the
drift report and return a well-structured markdown response with:

## 📊 Drift Summary
Brief overview of what drifted, severity level, and impact.

## 🔍 Root Cause Analysis
Bullet list of likely causes, ordered by probability.

## ⚡ Recommended Actions
Numbered list of specific steps to take, ordered by priority.

## 📈 Risk Assessment
Brief assessment of how this drift might affect model predictions.

Use **bold** for emphasis. Keep it concise and actionable.
Use clear section headers."""


def build_diagnosis_prompt(drift_report: dict, metrics_comparison: dict = None) -> list:
    """Build system and user messages from drift and optional metric results."""
    user_content = f"Drift Report:\n{json.dumps(drift_report, indent=2)}"
    if metrics_comparison:
        user_content += (
            "\n\nModel Metrics Comparison:\n"
            f"{json.dumps(metrics_comparison, indent=2)}"
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
