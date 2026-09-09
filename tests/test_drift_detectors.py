"""Deterministic tests for the three pure-math drift detectors."""

import numpy as np

from core.drift_detectors import cosine_drift, kl_divergence_gaussian, mmd


def test_mmd_similar_distributions():
	random_state = np.random.RandomState(1)
	X = random_state.normal(0, 1, size=(100, 16))
	Y = random_state.normal(0, 1, size=(100, 16))

	assert mmd(X, Y) < 0.05


def test_mmd_different_distribution():
	random_state = np.random.RandomState(2)
	X = random_state.normal(0, 1, size=(100, 16))
	Y = random_state.normal(5, 1, size=(100, 16))
	similar_Y = random_state.normal(0, 1, size=(100, 16))
	similar_score = mmd(X, similar_Y)
	different_score = mmd(X, Y)

	assert different_score > 0.1
	assert different_score > similar_score


def test_mmd_returns_float():
	random_state = np.random.RandomState(3)

	assert isinstance(
		mmd(random_state.normal(size=(10, 4)), random_state.normal(size=(10, 4))),
		float,
	)


def test_kl_similar_distributions():
	random_state = np.random.RandomState(4)
	X = random_state.normal(0, 1, size=(100, 16))
	Y = random_state.normal(0, 1, size=(100, 16))

	assert kl_divergence_gaussian(X, Y) < 10.0


def test_kl_different_distribution():
	random_state = np.random.RandomState(5)
	X = random_state.normal(0, 1, size=(100, 16))
	Y = random_state.normal(5, 1, size=(100, 16))
	similar_Y = random_state.normal(0, 1, size=(100, 16))
	similar_score = kl_divergence_gaussian(X, similar_Y)
	different_score = kl_divergence_gaussian(X, Y)

	assert different_score > 10.0
	assert different_score > similar_score


def test_kl_returns_float():
	random_state = np.random.RandomState(6)

	assert isinstance(
		kl_divergence_gaussian(
			random_state.normal(size=(10, 4)), random_state.normal(size=(10, 4))
		),
		float,
	)


def test_cosine_similar_distributions():
	random_state = np.random.RandomState(7)
	X = random_state.normal(2, 1, size=(100, 16))
	Y = random_state.normal(2, 1, size=(100, 16))

	assert cosine_drift(X, Y) < 0.1


def test_cosine_different_distribution():
	random_state = np.random.RandomState(8)
	X = random_state.normal(0, 1, size=(100, 16))
	Y = random_state.normal(5, 1, size=(100, 16))
	similar_X = random_state.normal(2, 1, size=(100, 16))
	similar_Y = random_state.normal(2, 1, size=(100, 16))
	similar_score = cosine_drift(similar_X, similar_Y)
	different_score = cosine_drift(X, Y)

	assert different_score > 0.1
	assert different_score > similar_score


def test_cosine_returns_float():
	random_state = np.random.RandomState(9)

	assert isinstance(
		cosine_drift(
			random_state.normal(size=(10, 4)), random_state.normal(size=(10, 4))
		),
		float,
	)
