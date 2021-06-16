from __future__ import annotations

import math
import numpy as np
from scipy.special import gamma
from typing import Callable

from .helpers import random_ball


class Mixture:
    """Superclass for mixture models in the context of the experiment"""
    def __init__(self, d: int):
        # dimension
        self.d = d

        # number of mixtures
        self.M = int(math.log(d, 2))
        
        # number of samples
        self.N = 2 ** d 

        # radius containing the data
        self.R = 2 * self.M
        
        # cluster assignments -> (M, N)
        self.assignments = np.full((self.M, self.N), np.nan)

        # Parameters of model
        self.params = None

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


class GaussianMixture(Mixture):
    """Class for the Gaussian Mixture Model experiment"""

    def __init__(self, d: int, init_from_data: bool = True):
        super().__init__(d)

        # variance
        self.var = 1 / math.sqrt(d)
        
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
        self.params = self.init_params(init_from_data)

    def reset(self, from_data: bool, resample: bool = False):
        """
        Reset parameters and points for GMM
        """
        if resample:
            self.points = self.sample()
        self.params = self.init_params(from_data)

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
        return random_ball(num_points=self.M, dimension=self.d, radius=self.R)

    def pdf_main(self, params: np.ndarray) -> np.ndarray:
        """
        Computes the regular GMM PDF term for all components
        :returns: numpy array of shape (M, N)
        """
        # compute norm of pairwise differences between data points and cluster centers -> (M, N)
        norm_diff = np.linalg.norm(self.points[None, :, :] - params[:, None, :], axis=2)
        exponent = -0.5 * norm_diff ** 2 / self.var
        return (self.var / 1000) * np.exp(exponent)
    
    def pdf(self, params: np.ndarray) -> np.ndarray:
        """
        Computes the PDF value of the mixture at all points
        :returns: numpy array of shape (N,)
        """
        return self.pdf_main(params).sum(axis=0) + self.C  # (N,)

    def prior(self, params: np.ndarray) -> np.float64:
        """Computes the prior term"""
        sqrt_M_times_R = math.sqrt(self.M) * self.R
        # Frobenius norm of means matrix
        fro = np.linalg.norm(params)
        return np.exp(-self.m * (fro - sqrt_M_times_R) ** 2 * int(fro >= sqrt_M_times_R))

    def objective(self, params: Optional[np.ndarray] = None) -> np.float64:
        """Computes the objective function value"""
        params = params if params is not None else self.params
        log_prior = -np.log(self.prior(params))
        # sum log of pdf at all points
        log_pdf = -np.log(self.pdf(params)).sum()
        return log_prior + log_pdf
    
    def gradient(self, x: np.ndarray, precision: float = 1e-4) -> np.ndarray:
        """
        Computes an approximation of the gradient of the objective function for the ULA algorithm
        Principle: grad[i] = (f(x_1, ... , x_i + e,..., x_n) - f(x_1, ... , x_i,..., x_n))/e
        x of size (nb_experiments, M, d)
        """
        gradient = np.zeros(x.shape)
        for exp in range(x.shape[0]):
            f_x = self.objective(x[exp])
            for mean in range(self.M):
                for dim in range(self.d):
                    h = np.zeros(x[exp].shape)
                    h[mean, dim] = precision
                    gradient[exp, mean, dim] = (self.objective(x[exp] + h) - f_x)/precision
        return gradient
    
    def e_step(self):
        """Expectation step of EM-algorithm"""
        f = self.pdf_main(self.params)  # (M, N)
        f_sum = f.sum(axis=0)  # (N,)
        self.assignments = f / (f_sum + self.C)  # (M, N)
        
    def m_step(self):
        """Maximization step of EM-algorithm"""
        assignments_sum = self.assignments.sum(axis=1, keepdims=True)  # (M, 1)
        clusters_weighted_sum = self.assignments @ self.points  # (M, N) x (N, d) -> (M, d)
        # update cluster centers -> (M, d)
        self.params = clusters_weighted_sum / assignments_sum


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
        return random_ball(num_points=self.M, dimension=self.d, radius=self.R)

    def pdf(self) -> float:
        alpha = self.theta[self.idx_alpha]
        pi = self.theta[self.idx_pi]
        fraction = gamma(np.sum(alpha)) / np.prod(gamma(alpha))
        product = np.prod(np.power(self.points, alpha - 1))
        inner_sum = np.sum(fraction * np.multiply(product, pi))
        return math.log(inner_sum)

    def objective(self):
        return
