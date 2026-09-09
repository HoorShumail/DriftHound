from unittest.mock import MagicMock, patch

import pytest

from agent.diagnostic_agent import diagnose_drift


def test_diagnose_drift_returns_mocked_response(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    mocked_response = MagicMock()
    mocked_response.choices[0].message.content = "Mocked drift explanation"

    with patch("agent.diagnostic_agent.OpenAI") as mocked_openai:
        mocked_openai.return_value.chat.completions.create.return_value = (
            mocked_response
        )

        result = diagnose_drift(
            {"mmd": 0.34, "kl_divergence": 1.2, "cosine_drift": 0.15},
            context="The corpus contains machine learning papers.",
        )

    assert result == "Mocked drift explanation"
    mocked_openai.assert_called_once_with(api_key="test-key")
    mocked_openai.return_value.chat.completions.create.assert_called_once()


def test_diagnose_drift_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with patch("agent.diagnostic_agent.load_dotenv"):
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY.*\.env"):
            diagnose_drift({"mmd": 0.34})