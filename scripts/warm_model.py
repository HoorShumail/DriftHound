"""Download and warm the local all-MiniLM-L6-v2 embedding model.

Run this once with network access before using the embedding tests offline.
"""

from sentence_transformers import SentenceTransformer


def main() -> None:
    """Load the model, encode one sentence, and confirm the cached model works."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embedding = model.encode(
        ["drifthound model warm-up"],
        convert_to_numpy=True,
    )
    print(f"Embedding shape: {embedding.shape}")
    print("MODEL CACHED OK")


if __name__ == "__main__":
    main()
