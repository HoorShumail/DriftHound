from unittest.mock import MagicMock

from agent.diagnostician import DriftDiagnostician
from agent.prompts import build_diagnosis_prompt


def make_report():
    return {
        "summary": {
            "features_checked": 2,
            "drifted_features": ["feature_1"],
            "per_feature": {
                "feature_1": {
                    "drift_detected": True,
                    "methods": {
                        "ks": {"drift_detected": True},
                        "psi": {"drift_detected": False},
                    },
                }
            },
        }
    }


def test_build_diagnosis_prompt_contains_feature_names_from_report():
    prompt = build_diagnosis_prompt(make_report())

    assert len(prompt) == 2
    assert prompt[0]["role"] == "system"
    assert prompt[1]["role"] == "user"
    assert "feature_1" in prompt[1]["content"]
    assert "ks" in prompt[1]["content"]


def test_build_diagnosis_prompt_handles_empty_report():
    prompt = build_diagnosis_prompt({})

    assert "{}" in prompt[1]["content"]


def test_diagnose_returns_non_empty_string():
    client = MagicMock()
    client.chat.completions.create.return_value.choices[0].message.content = (
        "Feature 1 drifted; inspect the upstream data source."
    )
    diagnostician = DriftDiagnostician(client=client)

    diagnosis = diagnostician.diagnose(make_report())

    assert diagnosis
    client.chat.completions.create.assert_called_once()
