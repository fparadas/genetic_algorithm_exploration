import pytest
import numpy as np
from ga.population import create_individual_dtype
from ga.types.result import Result
from ga.recombination import recombine, RecombinationMethod


# Fixture to create a population array
@pytest.fixture
def population():
    dtype = create_individual_dtype(2).value
    return np.array([(0, [1, 2], 0.0), (1, [3, 4], 0.0), (2, [5, 6], 0.0)], dtype=dtype)


@pytest.fixture
def rng():
    return np.random.default_rng()


# Test for valid recombination methods
@pytest.mark.parametrize(
    "method",
    [RecombinationMethod.MEAN, RecombinationMethod.MAX, RecombinationMethod.MIN],
)
def test_valid_recombination_methods(rng, population, method):
    result = recombine(rng, method, population, 2)
    assert result.is_ok()
    assert isinstance(result.value, np.ndarray)
    assert len(result.value) > 0


# Test for invalid recombination method
def test_invalid_recombination_method(rng, population):
    result = recombine(rng, None, population, 2)
    assert result.is_error()
    assert "Invalid recombination method" in result.error


# Test for exceeding number of parents
def test_exceeding_parents(rng, population):
    result = recombine(rng, RecombinationMethod.MEAN, population, 5)
    assert result.is_error()
    assert "Number of parents exceeds selected population size" in result.error


# Test for insufficient number of parents
@pytest.mark.parametrize("n_parens", [0, 1])
def test_insufficient_parents(rng, population, n_parens):
    result = recombine(rng, RecombinationMethod.MEAN, population, n_parens)
    assert result.is_error()
    assert "Number of parents must be at least 2" in result.error


# Test for population too small
def test_small_population(rng):
    small_population = np.array(
        [(0, [1, 2], 0.0)], dtype=create_individual_dtype(2).value
    )
    result = recombine(rng, RecombinationMethod.MEAN, small_population, 2)
    assert result.is_error()
    assert "Selected population must contain at least 2 individuals" in result.error
