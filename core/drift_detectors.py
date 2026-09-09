"""Core statistical measures for comparing embedding distributions."""

import numpy as np


def mmd(X: np.ndarray, Y: np.ndarray, gamma: float | None = None) -> float:
	"""Measure distribution drift with the squared MMD estimate.

	This compares every sample in ``X`` and ``Y`` using a Gaussian-shaped
	similarity kernel. A score near zero means the distributions look alike;
	larger scores indicate drift.
	When ``gamma`` is omitted, it is set to one divided by the embedding
	dimension.
	"""
	X = np.asarray(X, dtype=float)
	Y = np.asarray(Y, dtype=float)
	if X.ndim != 2 or Y.ndim != 2 or X.shape[0] == 0 or Y.shape[0] == 0:
		raise ValueError("X and Y must be non-empty two-dimensional arrays")
	if X.shape[1] != Y.shape[1]:
		raise ValueError("X and Y must have the same embedding dimension")
	if not np.isfinite(X).all() or not np.isfinite(Y).all():
		raise ValueError("X and Y must contain only finite values")
	if gamma is None:
		gamma = 1.0 / X.shape[1]
	if gamma <= 0:
		raise ValueError("gamma must be positive")

	# Compute the Gaussian kernel matrix from pairwise squared distances.
	def rbf_kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
		left_squared = np.sum(left * left, axis=1)[:, None]
		right_squared = np.sum(right * right, axis=1)[None, :]
		squared_distances = np.maximum(
			left_squared + right_squared - 2.0 * left @ right.T, 0.0
		)
		return np.exp(-gamma * squared_distances)

	# Average within-set and cross-set similarities for MMD squared.
	score = (
		np.mean(rbf_kernel(X, X))
		+ np.mean(rbf_kernel(Y, Y))
		- 2.0 * np.mean(rbf_kernel(X, Y))
	)
	return float(score)


def kl_divergence_gaussian(X: np.ndarray, Y: np.ndarray) -> float:
	"""Approximate the Gaussian KL divergence from distribution ``X`` to ``Y``.

	Each array is summarized by its mean and covariance, then the closed-form
	multivariate Gaussian KL formula is applied. A score near zero means little
	drift; larger values mean that ``X`` differs more from ``Y``.
	"""
	X = np.asarray(X, dtype=float)
	Y = np.asarray(Y, dtype=float)
	if X.ndim != 2 or Y.ndim != 2 or X.shape[0] == 0 or Y.shape[0] == 0:
		raise ValueError("X and Y must be non-empty two-dimensional arrays")
	if X.shape[1] != Y.shape[1]:
		raise ValueError("X and Y must have the same embedding dimension")
	if not np.isfinite(X).all() or not np.isfinite(Y).all():
		raise ValueError("X and Y must contain only finite values")
	if X.shape[0] < 2 or Y.shape[0] < 2:
		raise ValueError("X and Y need at least two samples for covariance")

	# Estimate each distribution's mean vector and sample covariance matrix.
	mean_X = np.mean(X, axis=0)
	mean_Y = np.mean(Y, axis=0)
	covariance_X = np.atleast_2d(np.cov(X, rowvar=False))
	covariance_Y = np.atleast_2d(np.cov(Y, rowvar=False))

	# Add a small diagonal ridge so covariance matrices are invertible.
	epsilon = 1e-6
	identity = np.eye(X.shape[1])
	covariance_X += epsilon * identity
	covariance_Y += epsilon * identity

	# Compute KL(N_X || N_Y) without explicitly forming a matrix inverse.
	mean_difference = mean_Y - mean_X
	solved_covariance = np.linalg.solve(covariance_Y, covariance_X)
	solved_mean = np.linalg.solve(covariance_Y, mean_difference)
	trace_term = np.trace(solved_covariance)
	quadratic_term = mean_difference @ solved_mean
	log_determinant_term = (
		np.linalg.slogdet(covariance_Y)[1] - np.linalg.slogdet(covariance_X)[1]
	)
	dimension = X.shape[1]
	score = 0.5 * (
		trace_term + quadratic_term - dimension + log_determinant_term
	)
	return float(max(score, 0.0))


def cosine_drift(X: np.ndarray, Y: np.ndarray) -> float:
	"""Measure cosine distance between the two arrays' mean vectors.

	The centroid is the mean vector for each set. The result is ``1 - cosine
	similarity``: near zero means matching directions, while values closer to
	two mean stronger directional drift.
	"""
	X = np.asarray(X, dtype=float)
	Y = np.asarray(Y, dtype=float)
	if X.ndim != 2 or Y.ndim != 2 or X.shape[0] == 0 or Y.shape[0] == 0:
		raise ValueError("X and Y must be non-empty two-dimensional arrays")
	if X.shape[1] != Y.shape[1]:
		raise ValueError("X and Y must have the same embedding dimension")
	if not np.isfinite(X).all() or not np.isfinite(Y).all():
		raise ValueError("X and Y must contain only finite values")

	# Reduce each distribution to its average embedding vector.
	centroid_X = np.mean(X, axis=0)
	centroid_Y = np.mean(Y, axis=0)
	norm_X = np.linalg.norm(centroid_X)
	norm_Y = np.linalg.norm(centroid_Y)

	# Handle zero centroids explicitly because cosine similarity is undefined.
	if norm_X == 0.0 and norm_Y == 0.0:
		return 0.0
	if norm_X == 0.0 or norm_Y == 0.0:
		return 1.0

	# Convert cosine similarity into cosine distance.
	similarity = np.dot(centroid_X, centroid_Y) / (norm_X * norm_Y)
	return float(1.0 - np.clip(similarity, -1.0, 1.0))
