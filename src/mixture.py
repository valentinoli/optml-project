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
        
        # number of samples
        self.N = 2 ** d

        # radius containing the data
        self.R = 2 * self.M
        
        # cluster assignments -> (N, M)
        self.assignments = np.full((self.N, self.M), np.nan)

    def sample(self) -> np.ndarray:
        """
        Create synthetic dataset with sparse entries for GMM experiment
        :returns: numpy array of shape (N, d)
        """
        rng = np.random.default_rng()

        # number of nonzero entries of each point
        num_nonzero = self.M

        # for each data point, create an array of permuted indices
        # -> (N, d)
        idx = np.array([
            rng.permutation(i) for i in np.tile(np.arange(self.d), (self.N, 1))
        ])

        # M nonzero entries, selected uniformly at random
        # -> (N, M)
        idx_nonzero = idx[:, :num_nonzero]

        # initialize points array with zeros 
        # -> (N, d)
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
        
        # normalization constant for the constant mixture term
        self.Z0 = ((2 * math.pi) ** (d/2)) / math.gamma(d/2 + 1)

        # normalization constant for the Gaussians: it is the same for all components
        # because the covariance matrix is isotropic and uniform accross components
        self.Z = (2 * math.pi * self.var) ** (d/2)
        
        # consequently, lambda_i's are uniform
        self.lambda_ = self.Z * self.var / 1000
        
        # constant mixture term
        # -> assume data are distributed in a bounded region (norm(point) <= R)
        # and take this constant mixture to describe that observation
        self.C = (1 - self.M * self.lambda_) / self.Z0
        
        # strong convexity parameter
        self.m = 1/64

        # sample points (N, d)
        self.points = self.sample()
        
        # cluster centers (M, d)
        self.mu = self.init_params(init_from_data)

    def init_params(self, from_data: bool) -> np.ndarray:
        """
        Computes initial mean parameters of the GMM
        :returns: numpy array of shape (M, d)
        """
        if from_data:
            # initialize cluster centers from data
            rng = np.random.default_rng()
            return rng.choice(self.points, size=self.M)

        # initialize in ball of radius R
        return random_ball(num_points=self.M, radius=self.R, dim=self.d)

    def pdf_main(self) -> np.ndarray:
        """
        Computes the regular GMM PDF term for all components
        :returns: numpy array of shape (N, M)
        """
        # compute norm of pairwise differences between data points and cluster centers -> (N, M)
        norm_diff = np.linalg.norm(self.points[:, None, :] - self.mu[None, :, :], axis=2)
        exponent = -0.5 * norm_diff ** 2 / self.var
        return (self.var / 1000) * np.exp(exponent)
    
    def pdf(self) -> np.ndarray:
        """
        Computes the PDF value of the mixture at all points
        :returns: numpy array of shape (N,)
        """
        return self.pdf_main().sum(axis=1) + self.C  # (N,)

    def prior(self) -> np.float64:
        """Computes the prior term"""
        sqrt_M_times_R = math.sqrt(self.M) * self.R
        # Frobenius norm of means matrix
        fro = np.linalg.norm(self.mu)
        return np.exp(-self.m * (fro - sqrt_M_times_R) ** 2 * int(fro >= sqrt_M_times_R))

    def objective(self) -> np.float64:
        """Computes the objective function value"""
        log_prior = -np.log(self.prior)
        # sum log of pdf at all points
        log_pdf = -np.log(self.pdf()).sum()
        return log_prior + log_pdf
    
    def e_step(self):
        """Expectation step of EM-algorithm"""
        f = self.pdf_main()  # (N, M)
        f_sum = f.sum(axis=1)[:, None]  # (N, 1)
        self.assignments = f / (f_sum + self.C)  # (N, M)
        
    def m_step(self):
        """Maximization step of EM-algorithm"""
        assignments_sum = self.assignments.sum(axis=0)[:, None]  # (M, 1)
        clusters_weighted_sum = self.assignments.T @ self.points  # (M, N) x (N, d) -> (M, d)
        # update cluster centers -> (M, d)
        self.mu = clusters_weighted_sum / assignments_sum



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
