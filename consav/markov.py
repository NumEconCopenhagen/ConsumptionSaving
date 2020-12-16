# -*- coding: utf-8 -*-
""" markov

Functions for working with Markov-processes.
"""

import numpy as np
from scipy.stats import norm
from numpy.linalg import svd
from numba import njit
from scipy import optimize

###########
# tauchen #
###########

def tauchen(mu,rho,sigma,m=3,n=7,cutoff=None):
    """ tauchen approximation of autoregressive process

    Args:

        mu (double): mean of shock
        rho (double): AR(1) coefficient
        sigma (double): std. of shock
        m (int): scale factor for width of grid
        n (int): number of grid points
        cutoff (double,optional): remove transition probabilities less than cutoff

    Returns:

        grid (numpy.ndarray): grid
        trans (numpy.ndarray): transition matrix
        ergodic (numpy.ndarray): ergodic distribution
        trans_cumsum (numpy.ndarray): transition matrix (cumsum)
        ergodic_cumsum (numpy.ndarray): ergodic distribution (cumsum)

    """
     
    # a. allocate
    grid = np.zeros(n)
    trans = np.zeros((n,n))
     
    # b. discretize x
    std_grid = np.sqrt(sigma**2/(1-rho**2))
  
    grid[0] = mu/(1-rho)-m*std_grid
    grid[n-1] = mu/(1-rho)+m*std_grid
  
    step = (grid[n-1]-grid[0])/(n-1)
  
    for i in range(1,n-1):
        grid[i] = grid[i-1] + step
         
    # c. generate transition matrix
    for j in range(n):

        trans[j,0] = norm.cdf((grid[0] - mu - rho*grid[j] + step/2) / sigma)
        trans[j,n-1] = 1 - norm.cdf((grid[n-1] - mu - rho*grid[j] - step/2) / sigma)
        
        for k in range(1,n-1):
            trans[j,k] = norm.cdf((grid[k] - mu - rho*grid[j] + step/2) / sigma) - \
                         norm.cdf((grid[k] - mu - rho*grid[j] - step/2) / sigma)
                       
    # d. find the ergodic distribution
    ergodic = find_ergodic(trans)

    # e. apply cutoff
    if not cutoff is None:  
        trans[trans < cutoff] = 0         

    # f. find cumsums
    trans_cumsum = np.array([np.cumsum(trans[i,:]) for i in range(n)])
    ergodic_cumsum = np.cumsum(ergodic)

    return grid,trans,ergodic,trans_cumsum,ergodic_cumsum

###############
# rouwenhorst #
###############

def rouwenhorst(mu,rho,sigma,n=7):
    """Rouwenhorst method to discretize autoregressie process

    Args:

        mu (double): mean
        rho (double): AR(1) coefficient
        sigma (double): std. of shock
        n (int): number of grid points

    Returns:

        grid (numpy.ndarray): grid
        trans (numpy.ndarray): transition matrix
        ergodic (numpy.ndarray): ergodic distribution
        trans_cumsum (numpy.ndarray): transition matrix (cumsum)
        ergodic_cumsum (numpy.ndarray): ergodic distribution (cumsum)

    """

    # a. parametrize Rouwenhorst for n=2
    p = (1+rho)/2
    trans = np.array([[p,1-p],[1-p,p]])

    # b. implement recursion to build from n = 3 to n = N
    for n in range(3,n+1):
        P1,P2,P3,P4 = (np.zeros((n, n)) for _ in range(4))
        P1[:-1,:-1] = p*trans
        P2[:-1,1:] = (1-p)*trans
        P3[1:,:-1] = (1-p)*trans
        P4[1:,1:] = p*trans
        trans = P1 + P2 + P3 + P4
        trans[1:-1] /= 2

    # c. invariant distribution
    ergodic = find_ergodic(trans)

    # d. scaling
    grid_sd = np.sqrt(sigma**2/(1-rho**2))*np.sqrt(n-1)
    grid = np.linspace(mu/(1-rho)-grid_sd,mu/(1-rho)+grid_sd,n)

    # e. find cumsums
    trans_cumsum = np.array([np.cumsum(trans[i,:]) for i in range(n)])
    ergodic_cumsum = np.cumsum(ergodic)

    return grid,trans,ergodic,trans_cumsum,ergodic_cumsum

def log_rouwenhorst(rho,sigma,n=7):
    """Rouwenhorst method to discretize autoregressie process in logs 
    with mean of one

    Args:

        rho (double): AR(1) coefficient
        sigma (double): std. of shock
        n (int): number of grid points

    Returns:

        grid (numpy.ndarray): grid
        trans (numpy.ndarray): transition matrix
        ergodic (numpy.ndarray): ergodic distribution
        trans_cumsum (numpy.ndarray): transition matrix (cumsum)
        ergodic_cumsum (numpy.ndarray): ergodic distribution (cumsum)

    """

    # a. standard
    grid,trans,ergodic,trans_cumsum,ergodic_cumsum = rouwenhorst(0.0,rho,sigma,n)
    
    # b. take exp and ensure exact mean of one
    grid = np.exp(grid)
    grid /= np.sum(ergodic*grid)

    return grid,trans,ergodic,trans_cumsum,ergodic_cumsum

###########
# general #
###########

def find_ergodic(trans,atol=1e-13,rtol=0):
    """ find ergodic distribution from transition matrix 
    
    Args:

        trans (numpy.ndarray): transition matrix
        atol (double): absolute tolerance
        rtol (double): relative tolerance
    
    Returns:

        (nump.ndarray): ergodic distribution

    """

    I = np.identity(len(trans))
    A = np.atleast_2d(np.transpose(trans)-I)
    _u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T

    return (ns/(sum(ns))).ravel()

@njit
def choice(r,p_cumsum):
    """ select from cumulated probilities 

    Args:

        r (double): uniform random number
        p_cumsum (numpy.ndarray): vector of cumulated probabilities, [x,y,z,...,1] where z > y > x > 0

    Returns:

        i (int): selection index

    """

    i = 0
    while r > p_cumsum[i] and i+1 < p_cumsum.size:
        i = i + 1

    return i
