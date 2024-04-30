"""
A Genetic Algorithm Module

This module provides the necessary functions to execute a genetic algorithm simulation,
focusing on solving optimization problems using evolutionary principles. The genetic
algorithm leverages selection, recombination, and mutation to evolve a population
towards better solutions over successive generations.

The module integrates various components such as selection, recombination, and mutation
strategies, which are configurable to adapt to different problem domains.

Dependencies:
- numpy: Used for numerical operations and data structure management.
- ga.selection: Provides selection mechanisms like rank and tournament.
- ga.recombination: Supports recombination methods.
- ga.mutation: Implements mutation operations.
- ga.types: Includes custom types like Result for error handling and response.

The module expects all supporting sub-modules (selection, recombination, mutation, types)
to be appropriately configured and available in the 'ga' package.

Examples and usage of the module are intended for those familiar with genetic algorithms
and require a custom setup per the problem's domain specifics.
"""

import numpy as np
from ga.selection import SelectionType, select
from ga.population import initialize_population, update_population
from ga.recombination import RecombinationMethod, recombine
from ga.mutation import MutationMethod, mutate

from typing import Tuple, Callable
from ga.types.result import Result


def run(
    func: Callable,
    dim: int,
    boundaries: Tuple[float, float],
    population_size: int,
    mutation_rate: float,
    recombination_rate: float,
    n_generations: int,
    selection_type: SelectionType = SelectionType.TOURNAMENT,
    recombination_method: RecombinationMethod = RecombinationMethod.MEAN,
    mutation_method: MutationMethod = MutationMethod.NORMAL,
    **kwargs
):
    """
    Executes a genetic algorithm with specified parameters.

    This function initializes a population and then iteratively applies genetic
    operations like selection, recombination, and mutation across a specified number
    of generations to evolve solutions.

    Parameters:
    func : Callable
        The fitness function to evaluate individuals. It should take an array of values
        and return a fitness score.
    dim : int
        The dimensionality of the problem space.
    boundaries : Tuple[float, float]
        A tuple specifying the minimum and maximum values of the search space.
    population_size : int
        The number of individuals in the population.
    mutation_rate : float
        The probability of mutation per individual.
    recombination_rate : float
        The proportion of the population to select for recombination.
    n_generations : int
        The number of generations to simulate.
    kwargs : dict
        Additional keyword arguments for advanced configurations such as the number of
        competitors in tournament selection or the number of parents in recombination.

    Returns:
    Result[Population, str]
        A Result object containing the final population or an error message.

    Example:
    >>> final_population = run(
            your_fitness_function, 3, (0, 10), 100, 0.01, 0.25, 50,
            n_competitors=5, n_parens=2
        )
    """

    rng = np.random.default_rng()
    vec_func = np.vectorize(func, signature="(n)->()")
    population, err = initialize_population(
        rng.uniform, dim, boundaries, population_size
    )
    if err != None:
        return Result.Error(err)

    # evaluate and cut population
    population["evaluation"] = vec_func(population["point"])
    population, err = select(rng, population, population_size, SelectionType.RANK)
    if err != None:
        return Result.Error(err)

    for i in range(n_generations):

        # perform recombination
        parents, err = select(
            rng,
            population,
            int(np.floor(population_size * recombination_rate)),
            selection_type,
            **kwargs,
        )
        if err != None:
            return Result.Error(err)

        children, err = recombine(rng, recombination_method, parents, **kwargs)
        if err != None:
            return Result.Error(err)

        population = np.concatenate((population, children), axis=0)

        # perform mutation

        mutated_children, err = mutate(rng, mutation_method, population, mutation_rate)
        if err != None:
            return Result.Error(err)

        population, err = update_population(population, mutated_children)
        if err != None:
            return Result.Error(err)

        # evaluate and cut population
        population["evaluation"] = vec_func(population["point"])
        population, err = select(rng, population, population_size, SelectionType.RANK)
        if err != None:
            return Result.Error(err)

    return Result.Ok(population)
