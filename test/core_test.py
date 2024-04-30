import pytest
import numpy as np
from unittest.mock import patch
from ga.core import run  # Update with the correct path to your run function
from ga.types.result import Result
from ga.selection import SelectionType
from ga.recombination import RecombinationMethod
from ga.mutation import MutationMethod
from ga.population import create_individual_dtype


# Define a simple fitness function for testing
def simple_fitness(points):
    return np.sum(points**2)


# Test setup
@pytest.fixture
def basic_population():
    dtype = create_individual_dtype(2).value
    return np.array([(i, [i, i + 1], 0.0) for i in range(10)], dtype=dtype)


### Tests


def test_successful_run():
    result = run(simple_fitness, 2, (0, 10), 10, 0.1, 0.8, 5)
    assert result.is_ok()
    assert len(result.value) == 10  # Ensuring population size remains constant


@patch("ga.core.initialize_population")
def test_initialization_failure(mock_init):
    mock_init.return_value = Result.Error("Initialization failed")
    result = run(simple_fitness, 2, (0, 10), 10, 0.1, 0.8, 5)
    assert result.is_error()
    assert "Initialization failed" == result.error


def test_selection_failure(basic_population):
    """Tests if the selection process correctly handles and reports a failure."""
    # Setup the environment with patches
    with patch(
        "ga.core.initialize_population", return_value=(basic_population, None)
    ) as mock_init, patch(
        "ga.core.select", return_value=(None, "Selection failed")
    ) as mock_select:

        # Execute the genetic algorithm run function
        result = run(simple_fitness, 2, (0, 10), 10, 0.1, 0.2, 5)

        # Assert that the result indicates an error and contains the correct message
        assert result.is_error(), "Expected an error result from the run function"
        assert (
            "Selection failed" in result.error
        ), "The error message should indicate a selection failure"

        # Optionally, ensure that mocks are called as expected
        mock_init.assert_called_once()
        mock_select.assert_called_once()


def test_recombination_failure(basic_population):
    with patch(
        "ga.core.initialize_population", return_value=(basic_population, None)
    ), patch("ga.selection.select", return_value=(basic_population, None)), patch(
        "ga.core.recombine", return_value=(None, "Recombination failed")
    ):
        result = run(simple_fitness, 2, (0, 10), 10, 0.1, 0.2, 5)
        assert result.is_error()
        assert "Recombination failed" in result.error


def test_mutation_failure(basic_population):
    with patch(
        "ga.core.initialize_population", return_value=(basic_population, None)
    ), patch("ga.core.select", return_value=(basic_population, None)), patch(
        "ga.core.recombine", return_value=(basic_population, None)
    ), patch(
        "ga.core.mutate", return_value=(None, "Mutation failed")
    ):
        result = run(simple_fitness, 2, (0, 10), 10, 0.1, 0.2, 5)
        assert result.is_error()
        assert "Mutation failed" in result.error


# ### More Tests
# # Add more tests to cover additional scenarios, such as handling of incorrect parameters, boundary conditions, etc.
