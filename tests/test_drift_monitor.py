import pytest

from monitoring.drift_monitor import check_drift


CAT_REFERENCE = [
    "The cat sleeps quietly on the sofa.",
    "A kitten plays with a ball of yarn.",
    "The family feeds their friendly pet every morning.",
    "Cats enjoy warm places near the window.",
    "The veterinarian checks the cat's health.",
]

CAT_CURRENT = [
    "A playful kitten naps beside its owner.",
    "The household pet enjoys chasing a toy mouse.",
    "A cat rests comfortably in a sunny room.",
    "The family gives their kitten fresh food.",
    "A healthy pet visits the animal doctor.",
]

COOKING_REFERENCE = [
    "Fresh vegetables are chopped for a healthy salad.",
    "The chef seasons the soup with herbs and spices.",
    "Bread dough needs time to rise before baking.",
    "Pasta is boiled until it is tender and served with sauce.",
    "A hot pan is useful for sauteing onions and garlic.",
]

SPACE_CURRENT = [
    "Astronauts conduct experiments aboard the space station.",
    "The telescope observes distant galaxies and stars.",
    "A rocket launches a satellite into orbit around Earth.",
    "Planets travel around their star in predictable orbits.",
    "Scientists study the formation of black holes.",
]


@pytest.fixture(scope="module")
def similar_result():
    return check_drift(CAT_REFERENCE, CAT_CURRENT)


@pytest.fixture(scope="module")
def different_result():
    return check_drift(COOKING_REFERENCE, SPACE_CURRENT)


def test_no_drift_for_similar_text(similar_result):
    assert similar_result["drift_detected"] is False


def test_drift_for_different_topics(different_result):
    assert different_result["drift_detected"] is True


def test_return_structure(similar_result):
    assert set(similar_result) == {
        "scores",
        "thresholds",
        "flags",
        "drift_detected",
    }
    assert set(similar_result["scores"]) == {"mmd", "kl", "cosine"}