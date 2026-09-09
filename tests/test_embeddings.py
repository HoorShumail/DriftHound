import numpy as np
import pytest

from pipeline.embeddings import EmbeddingPipeline, cosine_similarity


@pytest.fixture(scope="module")
def embedding_pipeline():
    return EmbeddingPipeline()


def test_embedding_shape(embedding_pipeline):
    embeddings = embedding_pipeline.embed(["hello world", "test sentence"])

    assert embeddings.shape == (2, 384)


def test_embedding_dim_method(embedding_pipeline):
    assert embedding_pipeline.get_embedding_dim() == 384


def test_embedding_not_zeros(embedding_pipeline):
    embeddings = embedding_pipeline.embed(["This is a real sentence."])

    assert not np.allclose(embeddings, 0.0)


def test_cosine_similarity(embedding_pipeline):
    embeddings = embedding_pipeline.embed(
        [
            "cat sits on mat",
            "cat is sitting on the mat",
            "quantum physics equations",
        ]
    )

    similar_score = cosine_similarity(embeddings[0], embeddings[1])
    different_score = cosine_similarity(embeddings[0], embeddings[2])

    assert similar_score > 0.7
    assert different_score < 0.5