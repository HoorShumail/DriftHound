"""Small helpers for creating and calling the OpenAI client."""

import os

from dotenv import load_dotenv
from openai import OpenAI


def get_client() -> OpenAI:
    """Create an OpenAI client using ``OPENAI_API_KEY`` from ``.env``."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not found in .env")
    return OpenAI(api_key=api_key)


def call_llm(client, messages, model="gpt-4o-mini", temperature=0.3) -> str:
    """Send chat messages to OpenAI and return the response text."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content
