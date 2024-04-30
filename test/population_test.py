import numpy as np
import pytest
from ga.population import create_individual_dtype, initialize_population


def test_create_individual_dtype():
    # Test for 2-dimensional points
    dtype_2d = create_individual_dtype(2).value
    assert dtype_2d["index"].kind == "i"  # integer for 'index'
    assert dtype_2d["index"].itemsize == 8  # 32 bits
    assert dtype_2d["point"].shape == (2,)  # shape of 2-dimensional point
    assert dtype_2d["evaluation"].kind == "f"  # double precision float for 'evaluation'

    # Test for 3-dimensional points
    dtype_3d = create_individual_dtype(3).value
    assert dtype_3d["point"].shape == (3,)

    # Test for 0-dimensional points
    assert create_individual_dtype(0).is_error()


def test_initialize_population():
    # Define a simple uniform distribution function for testing
    def uniform_distribution(min_val, max_val, size):
        return np.random.uniform(min_val, max_val, size)

    # Test with a 2-dimensional space and population size of 5
    dim = 2
    boundaries = (0, 1)
    population_size = 5
    population = initialize_population(
        uniform_distribution, dim, boundaries, population_size
    ).value

    assert len(population) == population_size  # Check population size
    assert population["index"].dtype == int  # Check index dtype
    assert population["point"].shape == (population_size, dim)  # Check shape of points
    assert population["evaluation"].dtype == float  # Check evaluation dtype
    assert np.all(
        population["evaluation"] == 0
    )  # Check all evaluations initialized to 0

    # Test if points are within the specified boundaries
    assert np.all(population["point"] >= boundaries[0])
    assert np.all(population["point"] <= boundaries[1])

    # Test with zero population size
    population = initialize_population(uniform_distribution, dim, boundaries, 0).value
    assert len(population) == 0

    # Test with zero dimensions
    assert initialize_population(
        uniform_distribution, 0, boundaries, population_size
    ).is_error()
