"""Generate local text embeddings with a free sentence-transformers model."""

import numpy as np
from sentence_transformers import SentenceTransformer


# Load the model once when this module is imported, then reuse it for all calls.
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed_texts(texts: list[str]) -> np.ndarray:
    """Convert a list of strings into a 2D NumPy embedding array."""
    return np.asarray(
        model.encode(texts, convert_to_numpy=True),
        dtype=np.float32,
    )


def embed_text(text: str) -> np.ndarray:
    """Convert one string into a 1D NumPy embedding vector."""
    # Encode as a one-item batch, then remove the batch dimension.
    return embed_texts([text])[0]


class EmbeddingPipeline:
    """Compatibility wrapper around the module-level embedding functions."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Use the already-loaded local model for this pipeline instance."""
        self.model = model if model_name == "all-MiniLM-L6-v2" else SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of text strings as a two-dimensional array."""
        return np.asarray(self.model.encode(texts, convert_to_numpy=True), dtype=np.float32)

    def get_embedding_dim(self) -> int:
        """Return the number of values in each embedding vector."""
        get_dimension = getattr(self.model, "get_embedding_dimension", None)
        if callable(get_dimension):
            return int(get_dimension())
        return int(self.model.get_sentence_embedding_dimension())


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the cosine similarity between two one-dimensional vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0.0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.dot(a, b) / denominator)