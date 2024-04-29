"""
@title spec.py
@author Felipe Paradas
@date 04/2024

Defines custom types used throughout the genetic algorithm library.
"""

import numpy as np


def individual_dtype(dim: int) -> np.dtype:
    """
    Returns the structured data type for an individual with the specified dimensionality.

    Parameters:
        dim (int): The dimensionality of the individual's point vector.

    Returns:
        np.dtype: A structured data type for an individual with the specified dimensionality.
    """
    return [("index", int), ("point", float, (dim,)), ("fitness", float)]
