"""Download and cache all-MiniLM-L6-v2 for reliable offline test runs.

Run this script once before running tests if Hugging Face downloads are flaky.
After the model is cached locally, the embedding tests can run without another
model download.
"""

from sentence_transformers import SentenceTransformer


def main() -> None:
    """Download the local embedding model and confirm that caching succeeded."""
    SentenceTransformer("all-MiniLM-L6-v2")
    print("Successfully downloaded and cached all-MiniLM-L6-v2.")


if __name__ == "__main__":
    main()