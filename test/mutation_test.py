import pytest
import numpy as np
from ga.types.result import Result
from ga.population import create_individual_dtype
from ga.mutation import (
    MutationMethod,
    mutate,
)


@pytest.fixture
def dtype():
    return create_individual_dtype(2).value


@pytest.fixture
def population(dtype):
    return np.array([(0, [1.0, 2.0], 0.0), (1, [3.0, 4.0], 0.0)], dtype=dtype)


@pytest.fixture
def rng():
    return np.random.default_rng()


# Test valid mutation methods
@pytest.mark.parametrize(
    "method", [MutationMethod.NORMAL, MutationMethod.STANDART, MutationMethod.UNIFORM]
)
def test_valid_mutation_methods(rng, population, method):
    mutation_rate = 0.5
    result = mutate(rng, method, population, mutation_rate)
    assert result.is_ok()
    assert len(result.value) > 0  # Checks if some mutations occurred


# Test for invalid mutation method
def test_invalid_mutation_method(rng, population):
    mutation_rate = 0.5
    result = mutate(rng, "invalid_method", population, mutation_rate)
    assert result.is_error()
    assert "Invalid mutation method" in result.error


# Test for invalid mutation rate
@pytest.mark.parametrize("mutation_rate", [-0.1, 1.1])
def test_invalid_mutation_rate(rng, population, mutation_rate):
    result = mutate(rng, MutationMethod.NORMAL, population, mutation_rate)
    assert result.is_error()
    assert "Invalid mutation rate" in result.error


# Test for empty population
def test_empty_population(rng, dtype):
    empty_population = np.array([], dtype=dtype)
    mutation_rate = 0.5
    result = mutate(rng, MutationMethod.NORMAL, empty_population, mutation_rate)
    assert result.is_error()
    assert "Population must contain at least one individual" in result.error


# Optional: Test actual mutation logic
def test_mutation_logic(rng, population):
    mutation_rate = 1.0  # 100% mutation rate to test all individuals
    result = mutate(rng, MutationMethod.UNIFORM, population, mutation_rate)
    assert result.is_ok()
    original_points = population["point"]
    mutated_points = result.value["point"]
    # Ensure that all points have been mutated
    assert not np.array_equal(original_points, mutated_points)
