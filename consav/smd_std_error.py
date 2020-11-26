# -*- coding: utf-8 -*-
"""smd_std_error

Provides a funciton for calculating standard errors.
"""

import numpy as np
from scipy import linalg

def compute_std_error(g,theta,W,Omega,Nobs,Nsim=1.0e+10,step=1.0e-5,args=()):
    """ calculate standard errors from minimum-distance type estimation
        g should return a vector with:

            data moments - simulated moments as a function of theta

        Args:

            g (callable): moment function (return vector of length J)
            theta (np.ndarray): parameter vector (length K)
            W (np.ndarray): weigting matrix (dim J-by-J)
            Omega (np.ndarray): covaraince matrix of empirical moments (dim J-by-J)
            Nobs (scalar): number of observations
            Nsim (scalar,optional): number of simulations
            step (scalar,optional): finite step in numerical gradients
            args (tupple,optinal): additional arguments passed to g
        
    """

    # a. dimensions
    K = len(theta)
    J = len(W[0])

    # b. numerical gradient.
    grad = np.empty((J,K))
    for p in range(K):

        theta_now = theta.copy() 

        step_now  = np.zeros(K)
        step_now[p] = np.fmax(step,step*np.abs(theta_now[p]))

        g_forward = g(theta_now + step_now,*args)
        g_backward = g(theta_now - step_now,*args)

        grad[:,p] = (g_forward - g_backward)/(2.0*step_now[p])

    # c. asymptotic variance
    GW  = grad.T @ W
    GWG = GW @ grad
    Avar = np.linalg.inv(GWG) @ ( GW @ Omega @ GW.T ) @ np.linalg.inv(GWG)
    
    # d. return asymptotic standard errors
    fac  = (1.0 + 1.0/Nsim)/Nobs 
    std_error = np.sqrt(fac*np.diag(Avar))
    
    return std_error