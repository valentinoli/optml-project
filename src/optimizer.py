from .mixture import Mixture


def em(model: Mixture):
    """Expectation Maximization Algorithm"""
    steps = 0
    converged = False
    while not converged:
        steps += 1
        converged = model.e_step()
        model.m_step()
        if steps == 100:
            converged = True
    return steps


def mala(model: Mixture):
    """Metropolis-adjusted Langevin algorithm"""
    raise NotImplementedError
