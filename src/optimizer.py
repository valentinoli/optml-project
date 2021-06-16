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


def ula(model: Mixture, nb_iters, nb_exps, error, gamma = None):
    """
    Unadjusted Langevin Algorithm
    Args:
        model: Mixture model with all the objective function paramters and gradient
        gamma (float): step size
        nb_iters (int): number of iterations
        nb_exps: number of samples computed in parallel
    Returns:
        x_k: the final sample
    """
    
    d = model.d
    L = model.L
    R = model.R
    M = model.M
    m = model.m
    if gamma == None:
        gamma = 252/(d**2) - 13   #Found via trial and errors, and according to the paper inversely proportionate to d**2
    
    x_k = np.random.multivariate_normal(np.zeros(d), (1/L)*np.identity(d), (nb_exps,M))
    
    #Gaussian noise
    def Z_k_1(): return np.random.normal(0,2*gamma,(M,d))
    
    for i in range(nb_iters - 1):
        #Samples the diffusion paths, using Euler-Maruyama scheme:
        x_k_1 = x_k - gamma * model.gradient(x_k, error) + Z_k_1()
        
        x_k = x_k_1

    return  x_k

def mala(model, nb_iters, nb_exps, error, gamma = None):
    """
    Metropolis-Adjusted Langevin Algorithm
    Args:
        model: Mixture model with all the objective function paramters and gradient
        gamma (float): step size
        nb_iters (int): number of iterations
        nb_exps: number of samples computed in parallel
    Returns:
        x_k: the final sample
    """

    d = model.d
    L = model.L
    R = model.R
    M = model.M
    m = model.m
    if gamma == None:
        gamma = 252/(d**2) - 13
    
    x_k = np.random.multivariate_normal(np.zeros(d), (1/L)*np.identity(d), (nb_exps,M))
    
    #grad_x_k = model.gradient(x_k, error)
    
    #Gaussian noise
    def Z_k_1(): return np.random.normal(0,2*gamma,(M,d))
    
    def transition_density(x,y,grad_x): return np.exp(-np.sum((y - x - gamma*grad_x)**2) / (4*gamma))
    
    for i in range(nb_iters - 1):
        grad_x_k = model.gradient(x_k, error)
        
        #Samples the diffusion paths, using Euler-Maruyama scheme:
        x_k_1 = x_k - gamma * grad_x_k 
        + np.sqrt(2*gamma) * Z_k_1()
        
        grad_x_k_1 = model.gradient(x_k_1, error)
        
        acceptance_ratio = np.zeros(x_k.shape[0])
        for i in range(x_k.shape[0]):
            acceptance_ratio[i] = (model.objective(x_k_1[i])*transition_density(x_k_1[i],x_k[i],grad_x_k_1[i])) 
            / (model.objective(x_k[i])*transition_density(x_k[i],x_k_1[i],grad_x_k[i]))
        uniform_distr = np.random.rand(nb_exps)
        
        index_forward = np.where(uniform_distr <= acceptance_ratio)[0]

        x_k[index_forward, ] = x_k_1[index_forward, ]
        

        
    return x_k
