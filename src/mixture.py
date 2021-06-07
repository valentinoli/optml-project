import math
import numpy as np
from scipy.special import gamma
from collections.abc import Callable


# https://stackoverflow.com/a/54544972/8238129
# Generate "num_points" random points in "dimension" that have uniform
# probability over the unit ball scaled by "radius" (length of points
# are in range [0, "radius"]).
def random_ball(num_points: int, dimension: int, radius: int = 1):
    # First generate random directions by normalizing the length of a
    # vector of random-normal values (these distribute evenly on ball).
    random_directions = np.random.normal(size=(dimension, num_points))
    random_directions /= np.linalg.norm(random_directions, axis=0)
    # Second generate a random radius with probability proportional to
    # the surface area of a ball with a given radius.
    random_radii = np.random.random(num_points) ** (1 / dimension)
    # Return the list of random (direction & length) points.
    return radius * (random_directions * random_radii).T


class Mixture:
    def __init__(self, d: int):
        # dimension
        self.d = d

        # number of mixtures
        self.M = int(math.log(d, 2))

        # radius containing the data
        self.R = 2 * self.M

    def sample(self) -> np.ndarray:
        """Create synthetic dataset with sparse entries for GMM experiment"""
        rng = np.random.default_rng()
        d = self.d
        # number of data points
        N = 2 ** d

        # number of nonzero entries of each point
        num_nonzero = self.M

        # create ndarray of permuted indices of each data point
        idx = np.array([
            rng.permutation(i) for i in np.tile(np.arange(d), (N, 1))
        ])

        # M nonzero entries, selected uniformly at random
        idx_nonzero = idx[:, :num_nonzero]

        # initialize points array with zeros
        points = np.zeros(idx.shape)

        # all nonzero entries follow a uniform distribution on [-1, 1]
        for i, indices in enumerate(idx_nonzero):
            points[i, indices] = rng.uniform(low=-1, high=1, size=num_nonzero)
        return points

    def pdf(self):
        raise NotImplementedError

    def _log_pdf_recipe(self, samples: np.ndarray, distribution: Callable[[np.ndarray], float]) -> float:
        pass


class GaussianMixture(Mixture):
    """Class for the Gaussian Mixture Model experiment"""

    def __init__(self, d: int, init_from_data: bool = True):
        super().__init__(d)

        # variance
        self.var = 1 / d

        # covariance matrix (isotropic and uniform)
        # self.cov = np.diag(np.repeat(self.var, d))

        # normalization constant for each Gaussian is the same
        # self.Z = math.sqrt(((2 * math.pi) ** d) * np.linalg.det(self.cov))
        self.Z = math.sqrt(((2 * math.pi * self.var) ** d))
        # lambda_i's
        self.lambda_ = self.Z * self.var / 1000

        self.points = self.sample()
        self.mu = self.init_params(init_from_data)

    def init_params(self, from_data: bool) -> np.ndarray:
        if from_data:
            # initialize cluster centers from data
            rng = np.random.default_rng()
            return rng.choice(self.points, size=self.M)

        # initialize in ball of radius R
        return random_ball(num_points=self.M, radius=self.R, dim=self.d)

    def pdf_constant_mixture(self):
        # Computes the constant mixture
        # Assume data are distributed in a bounded region
        # and take this constant mixture to describe that observation
        # Z_0 == Z ?
        return (np.linalg.norm(self.points, axis=1) <= self.R).astype(int) / self.Z

    def pdf(self) -> float:
        norm_diff = np.linalg.norm(self.points[:, None, :] - self.mu[None, :, :], axis=2)
        exponent = -0.5 * norm_diff ** 2 / self.var
        const_mix = (1 - self.M * self.lambda_) * self.pdf_constant_mixture()
        return (self.var / 1000) * np.exp(exponent).sum(axis=1) + const_mix

    def prior(self):
        # Computes the prior term
        sqrt_M_times_R = math.sqrt(self.M) * self.R
        # Frobenius norm of means matrix
        fro = np.linalg.norm(self.mu)
        return np.exp(-self.M * (fro - sqrt_M_times_R) ** 2 * int(fro >= sqrt_M_times_R))

    def objective(self):
        return -np.log(self.prior()) - np.log(self.pdf()).sum()


class DirichletMixture(Mixture):
    """Class for the Dirichlet Mixture Model experiment"""

    def __init__(self, d: int, init_from_data: bool = True):
        super().__init__(d)

        # Create indices for parameters
        self.idx_alpha = np.arange(self.M)
        self.idx_pi = np.arange(self.M, self.M*2)
        self.theta = self.init_params(init_from_data)

    def init_params(self, from_data: bool) -> np.ndarray:
        # NOTE: adapted from Gaussian!!! Needs to change if not using shifted-scaled variant
        if from_data:
            # initialize cluster centers from data
            rng = np.random.default_rng()
            return rng.choice(self.points, size=self.M)

        # initialize in ball of radius R
        return random_ball(num_points=self.M, radius=self.R, dim=self.d)

    def pdf(self) -> float:
        alpha = self.theta[self.idx_alpha]
        pi = self.theta[self.idx_pi]
        fraction = gamma(np.sum(alpha)) / np.prod(gamma(alpha))
        product = np.prod(np.power(self.points, alpha - 1))
        inner_sum = np.sum(fraction * np.multiply(product, pi))
        return math.log(inner_sum)

    def objective(self):
        return
