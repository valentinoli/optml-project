from .mixture import Mixture
import numpy as np


def em(model: Mixture, max_iterations, convergence_likelihood=None, convergence_threshold=1e-6) -> int:
    """Expectation Maximization Algorithm"""
    for i in range(max_iterations):
        model.e_step()
        likelihood_est = model.objective(model.params)
        model.m_step()
        
        # Optional convergence test
        if convergence_likelihood is not None \
                and likelihood_est - convergence_likelihood < convergence_threshold:
            return i + 1

    return max_iterations


def ula(model, nb_iters, nb_exps, error, gamma = None, exp_mu = None, exp_U = None, timeout = 50000):
    """
    If exp_mu and exp_U are left blank, the algorithm will run for nb_iters. Otherwise, it will run until
    it has fullfilled the paper's convergence criteria, or until it timeouts.
    Args:
        model: Mixture model with all the objective function paramters and gradient
        gamma (float): step size
        nb_iters (int): number of iterations
        nb_exps: number of samples computed in parallel
        exp_mu and exp_U: Expectation of the best means, and objective value of the best mean
        timeout: timeout for convergence to exp_mu and exp_U
    Returns:
        x_k: the final sample
    """
    
    d = model.d
    R = model.R
    M = model.M
    m = model.m
    if gamma is None:
        lr = {
            2:90,
            3:20,
            4:9.25,
            5:2,
            6:1.75,
            7:0.5,
            8:0.45
        }
        gamma = lr[d]
    print(gamma)
    x_k = np.zeros((nb_exps,M,d))
    for i in range(nb_exps):
        x_k[i] = model.init_params(True)
    x_list = [x_k]
    U_list = [model.objective(x_k[0])]
    #Gaussian noise
    def Z_k_1(): return np.random.normal(0,2*gamma,(M,d))
    #Running for nb_iters if exp_mu and exp_u are left blank
    if exp_mu is None and exp_U is None:
        for i in range(nb_iters - 1):
            grad_x_k = model.gradient(x_k, error)

            #Samples the diffusion paths, using Euler-Maruyama scheme:
            x_k_1 = x_k - gamma * grad_x_k 
            +  Z_k_1()
            x_k = x_k_1
            x_list.append(x_k)
            U_list.append(model.objective(x_k[0]))
    #Running until convergence or timeout if exp_mu and exp_u are passed as parameters
    else:
        U_acc = model.objective(x_k[0])
        x_k_acc = x_k[0]
        iteration = 1
        while np.absolute(U_acc/iteration - exp_U) >= 10**(-6) and np.linalg.norm(x_k_acc/iteration - exp_mu) >= 10**(-3):
            grad_x_k = model.gradient(x_k, error)
            #Samples the diffusion paths, using Euler-Maruyama scheme:
            x_k_1 = x_k - gamma * grad_x_k 
            +  Z_k_1()
            
            x_k = x_k_1
            x_k_acc += x_k[0]
            U_acc += model.objective(x_k[0])
            x_list.append(x_k)
            U_list.append(model.objective(x_k[0]))
            iteration += 1
            if iteration > timeout:
                print("Timeout")
                break
        print("Converged at iteration: " + str(iteration))    

    return  U_list, x_list

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
            numerator = (model.objective(x_k_1[i])*transition_density(x_k_1[i],x_k[i],grad_x_k_1[i]))
            denominator = (model.objective(x_k[i])*transition_density(x_k[i],x_k_1[i],grad_x_k[i]))
            acceptance_ratio[i] = numerator / denominator
        uniform_distr = np.random.rand(nb_exps)
        
        index_forward = np.where(uniform_distr <= acceptance_ratio)[0]

        x_k[index_forward, ] = x_k_1[index_forward, ]
        
    return x_k
