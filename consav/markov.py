# -*- coding: utf-8 -*-
""" grids

Functions for working with Markov-processes.
"""

import numpy as np
from scipy.stats import norm
from numpy.linalg import svd
from numba import njit

###########
# tauchen #
###########

def tauchen(mu,rho,sigma,m=3,N=7,cutoff=None):
    """ tauchen approximation of autoregressive process

    Args:

        mu (double): mean
        rho (double): AR(1) coefficient
        sigma (double): std. of shock
        m (int): scale factor for width of grid
        N (int): number of grid points
        cutoff (double,optional):  

    Returns:

        x (numpy.ndarray): grid
        trans (numpy.ndarray): transition matrix
        ergodic (numpy.ndarray): ergodic distribution
        trans_cumsum (numpy.ndarray): transition matrix (cumsum)
        ergodic_cumsum (numpy.ndarray): ergodic distribution (cumsum)

    """
     
    # a. allocate
    x = np.zeros(N)
    trans = np.zeros((N,N))
     
    # b. discretize x
    std_x = np.sqrt(sigma**2/(1-rho**2))
  
    x[0] = mu/(1-rho) - m*std_x
    x[N-1] = mu/(1-rho) + m*std_x
  
    step = (x[N-1]-x[0])/(N-1)
  
    for i in range(1,N-1):
        x[i] = x[i-1] + step
         
    # c. generate transition matrix
    for j in range(N):

        trans[j,0] = norm.cdf((x[0] - mu - rho*x[j] + step/2) / sigma)
        trans[j,N-1] = 1 - norm.cdf((x[N-1] - mu - rho*x[j] - step/2) / sigma)
        
        for k in range(1,N-1):
            trans[j,k] = norm.cdf((x[k] - mu - rho*x[j] + step/2) / sigma) - \
                         norm.cdf((x[k] - mu - rho*x[j] - step/2) / sigma)
                       
    # d. find the ergodic distribution
    ergodic = find_ergodic(trans)

    # e. apply cutoff
    if not np.isnan(cutoff):  
        trans[trans < cutoff] = 0         

    # f. find cumsums
    trans_cumsum = np.array([np.cumsum(trans[i, :]) for i in range(N)])
    ergodic_cumsum = np.cumsum(ergodic)

    return x,trans,ergodic,trans_cumsum,ergodic_cumsum

###############
# rouwenhorst #
###############

def markov_rouwenhorst(rho,sigma,N=7):
    """Rouwenhorst method analog to markov_tauchen

    Args:

        rho (double): AR(1) coefficient
        sigma (double): std. of shock
        N (int): number of grid points

    Returns:

        y (numpy.ndarray): grid
        trans (numpy.ndarray): transition matrix
        ergodic (numpy.ndarray): ergodic distribution

    """

    # a. parametrize Rouwenhorst for n=2
    p = (1+rho)/2
    trans = np.array([[p,1-p],[1-p,p]])

    # b. implement recursion to build from n = 3 to n = N
    for n in range(3, N + 1):
        P1, P2, P3, P4 = (np.zeros((n, n)) for _ in range(4))
        P1[:-1, :-1] = p * trans
        P2[:-1, 1:] = (1 - p) * trans
        P3[1:, :-1] = (1 - p) * trans
        P4[1:, 1:] = p * trans
        trans = P1 + P2 + P3 + P4
        trans[1:-1] /= 2

    # c. invariant distribution
    ergodic = find_ergodic(trans)

    # d. scaling
    s = np.linspace(-1, 1, N)
    mean = np.sum(ergodic*s)
    sigma_ = np.sqrt(np.sum(ergodic*(s-mean)**2))
    s *= (sigma / sigma_)

    y = np.exp(s) / np.sum(ergodic * np.exp(s))

    return y, trans, ergodic

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
