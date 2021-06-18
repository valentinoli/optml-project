from __future__ import annotations

import numpy as np
from typing import Any


def random_ball(num_points: int, dimension: int, radius: int = 1) -> np.ndarray:
    """
    Generate "num_points" random points in "dimension" that have uniform
    probability over the unit ball scaled by "radius" (length of points
    are in range [0, "radius"]).
    https://stackoverflow.com/a/54544972/8238129
    Args:
        num_points: number of points to generate
        dimension: dimensionality of the points
        radius: the radius of the ball to generate from within
    Returns:
        An array of the generated points
    """
    # First generate random directions by normalizing the length of a
    # vector of random-normal values (these distribute evenly on ball).
    random_directions = np.random.normal(size=(dimension, num_points))
    random_directions /= np.linalg.norm(random_directions, axis=0)
    # Second generate a random radius with probability proportional to
    # the surface area of a ball with a given radius.
    random_radii = np.random.random(num_points) ** (1 / dimension)
    # Return the list of random (direction & length) points.
    return radius * (random_directions * random_radii).T


def Z_k_1(gamma: float, M: int, d: int) -> float:
    """
    Args:
        gamma: scaling parameter
        M: number of mixtures
        d: dimension
    Returns:
        Gaussian noise for the ULA and MALA algorithms
    """
    return np.random.default_rng().normal(0, 2*gamma, (M, d))


def get_gamma(d: int, gamma: Optional[float] = None) -> float:
    """
    Args:
        gamma: scaling parameter
        d: dimension
    Returns:
        Returns gamma parameter for the ULA and MALA algorithms
    """
    if gamma is not None:
        return gamma
    # default gamma
    return 252/(d**2) - 13
