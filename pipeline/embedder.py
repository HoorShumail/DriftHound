"""Create free, local text embeddings with sentence-transformers."""

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """Turn text into numbers that capture aspects of its meaning.

    The numbers are called embeddings. Similar pieces of text tend to have
    similar vectors, which can later be compared for drift detection.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Load the local Sentence Transformers model once for this instance."""
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Convert a list of strings into one embedding row per string."""
        if not texts:
            get_dimension = getattr(self.model, "get_embedding_dimension", None)
            embedding_dim = int(
                get_dimension()
                if callable(get_dimension)
                else self.model.get_sentence_embedding_dimension()
            )
            return np.empty((0, embedding_dim), dtype=np.float32)

        # Encode the whole batch once and request a NumPy array directly.
        return np.asarray(
            self.model.encode(texts, convert_to_numpy=True), dtype=np.float32
        )

    def embed_one(self, text: str) -> np.ndarray:
        """Convert one string into a single one-dimensional embedding vector."""
        # Encode one item as a batch, then remove the one-item batch dimension.
        return self.embed([text])[0]

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Compatibility alias for the existing batch embedding API."""
        return self.embed(texts)

    def embed_file(self, filepath: str) -> np.ndarray:
        """Embed each non-empty line in a UTF-8 text file as one document."""
        lines = [
            line.strip()
            for line in Path(filepath).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return self.embed(lines)


TextEmbedder = Embedder


if __name__ == "__main__":
    embedder = Embedder()
    sample_texts = [
        "Neural networks learn patterns from data.",
        "Drift detection compares two distributions.",
        "Local models can create embeddings without paid APIs.",
    ]
    print(embedder.embed(sample_texts).shape)