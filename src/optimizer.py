from .mixture import Mixture
import numpy as np


def em(model: Mixture, max_iterations, convergence_likelihood=None, convergence_threshold=1e-6) -> int:
    """Expectation Maximization Algorithm"""
    for i in range(max_iterations):
        model.e_step()
        model.m_step()

        # Optional convergence test
        likelihood_est = model.objective()
        if convergence_likelihood is not None \
                and likelihood_est - convergence_likelihood < convergence_threshold:
            return i + 1

    return max_iterations


def mala(model: Mixture):
    """Metropolis-adjusted Langevin algorithm"""
    raise NotImplementedError
