"""
result.py

Defines a generic Result type used throughout the genetic algorithm library.

This module introduces a Result class that encapsulates the outcome of operations, 
allowing for explicit handling of successes and failures without raising exceptions.
The Result type is designed to be a robust mechanism for function return values, 
facilitating error handling and the propagation of error information alongside normal 
results without disrupting the flow of the program.

The Result class can hold either a value (indicating success) or an error (indicating failure),
but not both simultaneously. This design ensures that functions using this type are explicit
about their success or failure states, improving the readability and maintainability of the code.

Classes:
- Result: A generic type that holds a value in the case of success or an error in the case of failure.

Usage:
The Result type is intended to be used for returning and propagating outcomes from functions 
within the genetic algorithm that may encounter recoverable errors or need to indicate failure
without raising exceptions. It provides methods to check the status of the result and to easily
access the contained data.

Example of creating and using a Result object:
>>> res = Result.Ok(123)  # Represents success with a value
>>> if res.is_ok():
...     print(res.value)  # Outputs: 123
>>> err = Result.Error("Failure occurred")  # Represents an error
>>> if err.is_error():
...     print(err.error)  # Outputs: Failure occurred
"""

from typing import Generic, TypeVar, Union, Any

T = TypeVar("T")  # Success type
E = TypeVar("E")  # Error type


class Result(Generic[T, E]):
    def __init__(self, value: Union[T, None] = None, error: Union[E, None] = None):
        self.value = value
        self.error = error
        if value is not None and error is not None:
            raise ValueError("Result cannot have both value and error")
        if value is None and error is None:
            raise ValueError("Result must have either value or error")

    def is_ok(self):
        return self.value is not None

    def is_error(self):
        return self.error is not None

    @classmethod
    def Ok(cls, value: T):
        return cls(value=value)

    @classmethod
    def Error(cls, error: E):
        return cls(error=error)
