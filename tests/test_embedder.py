from pathlib import Path

import numpy as np

from pipeline.embedder import Embedder


def test_embed_texts_shape():
    embedder = Embedder()
    embeddings = embedder.embed_texts(["model training", "neural networks", "AI data"])

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, 384)


def test_embed_file_reads_lines():
    embedder = Embedder()
    filepath = Path("data/sample_reference.txt")
    expected_lines = [line for line in filepath.read_text(encoding="utf-8").splitlines() if line.strip()]

    embeddings = embedder.embed_file(str(filepath))

    assert embeddings.shape == (len(expected_lines), 384)


def test_empty_input():
    embeddings = Embedder().embed_texts([])

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (0, 384)