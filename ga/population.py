import numpy as np
from typing import Tuple, Callable
from ga.types.result import Result

Population = np.ndarray


def create_individual_dtype(dim: int) -> Result[np.dtype, str]:
    """
    Create a dtype for individuals in a genetic algorithm based on the specified dimension of points.

    Parameters:
    dim : int
        The dimensionality of the point in each individual.

    Returns:
    dtype : np.dtype
        A NumPy dtype for individuals.
    """
    if dim <= 0:
        return Result.Error("Dimension must be greater than zero")

    dtype = np.dtype([("index", int), ("point", float, (dim,)), ("evaluation", float)])
    return Result.Ok(dtype)


def initialize_population(
    distribution_function: Callable,
    dim: int,
    boundaries: Tuple[float, float],
    population_size: int,
) -> Result[Population, str]:
    """
    Initializes a population for a genetic algorithm.

    Generates an initial population of individuals for a genetic algorithm, where each individual is represented by a point vector within a specified dimensional space and boundaries.
    Each individual also has an index and an evaluation value initially set to zero.

    Parameters:
    distribution_function : callable
        The function used to generate points within the search space.
        This function should accept boundaries and size as parameters to produce a matrix of points.
    dim : int
        The dimensionality of the search space.
    boundaries : tuple
        A tuple containing the minimum and maximum boundaries of the search space (min, max).
    population_size : int
        The number of individuals in the initial population.

    Returns:
    np.ndarray
        A numpy structured array containing the initial population, where each element is a tuple consisting of:
        - 'index': the index of the individual in the population,
        - 'point': a point vector in the search space,
        - 'eval': the evaluation value of the individual, initially set to zero.

    Example:
    >>> import numpy as np
    >>> def custom_distribution(min_val, max_val, size):
    ...     return np.random.uniform(min_val, max_val, size)
    >>> population = setup_initial_population(custom_distribution, 2, (0, 1), 10)
    >>> print(population['point'])
    [[0.1234, 0.2345],
    [0.3456, 0.4567],
    ...
    [0.8912, 0.9321]]
    """
    if boundaries[0] >= boundaries[1]:
        return Result.Error("Invalid boundaries: min must be less than max")
    if population_size < 0:
        return Result.Error("Population size must be non-negative")

    dtype, err = create_individual_dtype(dim)
    if err != None:
        return Result.Error(err)

    population = np.array(
        [
            (i, u, 0)
            for i, u in enumerate(
                distribution_function(*boundaries, size=(population_size, dim))
            )
        ],
        dtype=dtype,
    )

    return Result.Ok(population)
