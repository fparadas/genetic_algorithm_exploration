import numpy as np
from ga.types.result import Result
from ga.population import Population
from enum import Enum


class MutationMethod(Enum):
    NORMAL = "NORMAL"
    STANDART = "STANDART"
    UNIFORM = "UNIFORM"


def mutate(
    rng: np.random.Generator,
    method: MutationMethod,
    population: Population,
    mutation_rate: float,
) -> Result[Population, str]:
    """
    Mutates individuals in a population according to a mutation rate.

    Parameters:
    rng : np.random.Generator
        A random number generator instance.
    population : np.ndarray
        A list of individuals to mutate.
    mutation_rate : float
        The probability of mutation for each individual.

    Returns: np.ndarray
        A numpy structured array containing the mutated population.
    """
    match method:
        case MutationMethod.NORMAL:
            mutation_func = lambda x: x + rng.normal(0, 1, x.shape)
        case MutationMethod.STANDART:
            mutation_func = lambda x: x + rng.standard_t(1, x.shape)
        case MutationMethod.UNIFORM:
            mutation_func = lambda x: x + rng.uniform(-1, 1, x.shape)
        case _:
            return Result.Error("Invalid mutation method")

    if mutation_rate < 0 or mutation_rate > 1:
        return Result.Error("Invalid mutation rate")
    if len(population) <= 0:
        return Result.Error("Population must contain at least one individual")

    mutation_size = int(np.floor(len(population) * mutation_rate))
    mutated_population = rng.choice(population, mutation_size, replace=False)
    for i in range(len(mutated_population)):
        mutated_population[i]["point"] = mutation_func(mutated_population[i]["point"])
        mutated_population[i]["evaluation"] = 0.0

    return Result.Ok(mutated_population)
