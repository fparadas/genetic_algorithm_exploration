"""
selection.py

This module provides various selection mechanisms for a genetic algorithm library. 
It defines selection methods such as tournament selection, rank-based selection, and 
placeholders for other types like roulette wheel selection. Each selection method is 
designed to operate on a population of individuals, identifying and returning a subset 
of individuals based on their fitness and the specified selection strategy.

Selection types are managed through an enumeration that allows users to specify the 
desired method dynamically. This flexibility facilitates the exploration of different 
genetic algorithm configurations and their impact on the algorithm's performance.

Functions:
- select: Orchestrates the selection process by invoking the appropriate method based 
  on the specified SelectionType.
- tournament_selection: Implements tournament selection, where individuals compete 
  directly in randomly formed groups.

The module ensures that each selection process adheres to the genetic algorithm's requirements,
providing a robust framework for evaluating and evolving solutions through generational iterations.

Examples of how to use these functions are provided in the docstrings, illustrating their 
usage within genetic algorithm workflows.
"""

from enum import Enum
import numpy as np
from utils.result import Result


class SelectionType(Enum):
    """
    Enumeration of selection types for a genetic algorithm.

    Attributes:
        TOURNAMENT: Represents tournament selection method.
        ROULETTE: Represents roulette wheel selection method.
        RANK: Represents rank-based selection method.
    """

    TOURNAMENT = "tournament"
    ROULETTE = "roulette"
    RANK = "rank"


def select(
    rng: np.random.Generator,
    population: np.ndarray,
    competition_size: int,
    selection_type: SelectionType,
    **kwargs,
) -> Result:
    """
    Selects individuals from a population according to the specified selection method in a genetic algorithm.

    Parameters:
    rng : np.random.Generator
        A random number generator instance.
    population : np.ndarray
        A list of individuals from which a subset will be selected.
    competition_size : int
        The number of individuals to select from the population.
    selection_type : SelectionType
        The type of selection to perform.
    selection_params : dict, optional
        Additional parameters for the selection method.

    Returns:
    Result
        A Result object containing either the selected population or an error message.
    """
    match selection_type:
        case SelectionType.TOURNAMENT:
            return tournament_selection(
                rng=rng,
                population=population,
                competition_size=competition_size,
                **kwargs,
            )
        case SelectionType.RANK:
            return rank_selection(population, competition_size)
        case SelectionType.ROULETTE:
            return Result.Error("Roulette wheel selection not implemented yet")
        case _:
            return Result.Error(f"Invalid selection type: {selection_type}")


def tournament_selection(
    rng: np.random.Generator,
    population: np.ndarray,
    competition_size: int,
    n_competitors: int = 2,
    **kwargs,
) -> Result:
    """
    Performs a tournament selection for a genetic algorithm.

    Parameters:
    rng :
        The random number generator used to select individuals.
    population :
        An array of individuals, each with a 'eval' attribute representing their fitness.
    competition_size :
        The number of individuals selected to participate in the tournament.
    n_competitors :
        The number of competitors in each tournament round.

    Returns:
    Result
        A Result object containing either the selected population or an error message.
    """
    if competition_size > len(population):
        return Result.Error("Competition size larger than population")
    if competition_size < n_competitors:
        return Result.Error("Not enough competitors to form a single group")

    selected_indices = rng.choice(
        range(len(population)), competition_size, replace=False
    )
    n_groups = int(np.floor(competition_size / n_competitors))
    groups = rng.choice(selected_indices, (n_groups, n_competitors), replace=False)

    winners = np.array(
        [
            population[min(group, key=lambda idx: population[idx]["evaluation"])]
            for group in groups
        ]
    )
    return Result.Ok(winners)


def rank_selection(population: np.ndarray, competition_size: int) -> Result:
    """
    Performs rank-based selection for a genetic algorithm.

    Parameters:
    rng :
        The random number generator used to select individuals.
    population :
        An array of individuals, each with a 'eval' attribute representing their fitness.
    competition_size :
        The number of individuals selected to participate in the tournament.

    Returns:
    Result
        A Result object containing either the selected population or an error message.
    """
    return Result.Ok(np.sort(population, order="evaluation")[:competition_size])
