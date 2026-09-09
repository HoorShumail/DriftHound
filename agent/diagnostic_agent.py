"""Use OpenAI to explain drift scores in plain English."""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI


def diagnose_drift(drift_scores: dict, context: str = "") -> str:
    """Ask GPT-4o-mini to explain drift and recommend practical next steps.

    The function sends the supplied detector scores and optional corpus context
    to OpenAI, then returns the model's plain-English explanation.

    Raises:
        RuntimeError: If ``OPENAI_API_KEY`` is not available in the environment
            or a ``.env`` file.
    """
    # Load local development variables before reading the API key.
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Set OPENAI_API_KEY in a .env file "
            "before calling diagnose_drift()."
        )

    # Keep the request focused so the result is useful to a beginner.
    prompt = f"""Analyze these document drift scores for a RAG/document-QA system:

Drift scores: {json.dumps(drift_scores, sort_keys=True)}
Additional context: {context or "No additional context was provided."}

Please:
1. Explain in simple, non-technical language whether significant drift is present.
2. Explain what this likely means for the RAG/document-QA system.
3. Give 2-3 concrete, actionable recommendations.
"""

    # Create the official client with the key loaded from the environment.
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful diagnostic assistant for document drift.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content