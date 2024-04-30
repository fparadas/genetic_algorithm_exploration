import pytest
import numpy as np
from ga.selection import tournament_selection, select, SelectionType
from ga.population import create_individual_dtype


@pytest.fixture
def population():
    dtype, err = create_individual_dtype(2)
    if err is not None:
        raise ValueError(err)
    return np.array(
        [(i, np.random.random(2), np.random.random()) for i in range(20)],
        dtype=dtype,
    )


@pytest.fixture
def rng():
    return np.random.default_rng(seed=42)  # Fixed seed for reproducibility


def test_tournament_selection_success(rng, population):
    """Tests tournament selection successfully selects the correct number of individuals."""
    result = tournament_selection(rng, population, competition_size=8, n_competitors=2)
    assert result.is_ok(), "Should successfully select individuals"
    assert (
        len(result.value) == 4
    ), "Should select exactly 4 winners for 8 individuals with 2 competitors each"


def test_tournament_selection_failure_too_few_competitors(rng, population):
    """Tests tournament selection fails if there are not enough competitors."""
    result = tournament_selection(rng, population, competition_size=8, n_competitors=10)
    assert result.is_error(), "Should return an error due to insufficient competitors"


def test_tournament_selection_failure_competition_size_exceeds_population(
    rng, population
):
    """Tests tournament selection fails if the competition size exceeds the population size."""
    result = tournament_selection(rng, population, competition_size=25, n_competitors=2)
    assert (
        result.is_error()
    ), "Should return an error due to competition size exceeding population size"


def test_select_function_valid_tournament(rng, population):
    """Tests the select function twih a valid tournament selection type."""
    result = select(
        rng=rng,
        population=population,
        competition_size=10,
        selection_type=SelectionType.TOURNAMENT,
        n_competitors=3,
    )
    assert result.is_ok(), "Selection should be successful"
    assert len(result.value) == 3, "Should select exactly 3 winners"


def test_select_function_invalid_selection_type(rng, population):
    """Tests the select function with an invalid selection type."""
    result = select(
        rng=rng,
        population=population,
        competition_size=10,
        selection_type=SelectionType.ROULETTE,  # Assuming not implemented
    )
    assert result.is_error(), "Should return an error for unimplemented selection type"
