"""High-level helpers for comparing text distributions for drift."""

from core.drift_detectors import cosine_drift, kl_divergence_gaussian, mmd
from pipeline.embedder import Embedder


# These defaults reflect the different scales of the three detector outputs.
DEFAULT_THRESHOLDS = {
    "mmd": 0.01,
    "kl": 1_000_000.0,
    "cosine": 0.4,
}


def check_drift(
    reference_texts: list[str],
    current_texts: list[str],
    thresholds: dict[str, float] | None = None,
) -> dict:
    """Compare baseline and current text and report scores, flags, and drift.

    Both text lists are converted to embeddings with the local sentence-
    transformer model. Each detector score is then compared with its matching
    threshold; the overall result is drifted when any detector flags drift.
    """
    # Start with calibrated defaults and allow callers to override selected values.
    thresholds_used = DEFAULT_THRESHOLDS.copy()
    if thresholds is not None:
        thresholds_used.update(thresholds)

    # Embed both distributions with the same model before comparing them.
    embedder = Embedder()
    reference_embeddings = embedder.embed_texts(reference_texts)
    current_embeddings = embedder.embed_texts(current_texts)

    # Run all three detectors and use the requested public score names.
    scores = {
        "mmd": mmd(reference_embeddings, current_embeddings),
        "kl": kl_divergence_gaussian(reference_embeddings, current_embeddings),
        "cosine": cosine_drift(reference_embeddings, current_embeddings),
    }
    flags = {
        metric: scores[metric] > thresholds_used[metric]
        for metric in scores
    }

    return {
        "scores": scores,
        "thresholds": thresholds_used,
        "flags": flags,
        "drift_detected": any(flags.values()),
    }


class DriftMonitor:
    """Compatibility wrapper for the earlier stateful monitor interface."""

    def __init__(self, reference_texts: list[str]) -> None:
        self.reference_texts = reference_texts

    def check_drift(self, current_texts: list[str]) -> dict[str, float]:
        """Return the three detector scores for the stored reference texts."""
        result = check_drift(self.reference_texts, current_texts)
        return {
            "mmd": result["scores"]["mmd"],
            "kl_divergence": result["scores"]["kl"],
            "cosine_drift": result["scores"]["cosine"],
        }

    def is_drifting(self, current_texts: list[str]) -> bool:
        """Return whether any default detector threshold is exceeded."""
        return check_drift(self.reference_texts, current_texts)["drift_detected"]