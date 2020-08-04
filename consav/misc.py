# -*- coding: utf-8 -*-
"""newton_raphson

This module provides misc functions.
"""

import time
import math
import numpy as np
from scipy.stats import norm
from numpy.linalg import svd
from numba import njit

def elapsed(t0,t1=None):
    """ time elapsed since t0 with in nice format

    Args:

        t0 (double): start time
        t1 (double,optional): end time (else now)

    Return:

        (str): elapsed time in nice format

    """ 

    if t1 is None:
        secs = time.time()-t0
    else:
        secs = t1-t0

    days = secs//(60*60*24)
    secs -= 60*60*24*days

    hours = secs//(60*60)
    secs -= 60*60*hours

    mins = secs//(60)
    secs -= 60*mins
   
    text = ''
    if days > 0: text += f'{days} days '
    if hours > 0: text += f'{hours} hours '
    if mins > 0: text += f'{mins} mins '

    if days > 0 or hours > 0:
        pass
    elif mins > 0:
        text += f'{secs:.0f} secs '
    else:
        text = f'{secs:.1f} secs '

    return text[:-1]
    
def nonlinspace(x_min, x_max, n, phi):
    """ like np.linspace. but with unequal spacing

    Args:

        x_min (double): maximum value
        x_max (double): minimum value
        n (int): number of points
        phi (double): phi = 1 -> eqaul spacing, phi up -> more points closer to minimum
    
    Returns:

        y (list): grid with unequal spacing

    """

    assert x_max > x_min
    assert n >= 2
    assert phi >= 1
 
    # 1. recursion
    y = np.empty(n)
 
    y[0] = x_min
    for i in range(1, n):
        y[i] = y[i-1] + (x_max-y[i-1]) / (n-i)**phi
    
    # 3. assert increaing
    assert np.all(np.diff(y) > 0)
 
    return y
 
def equilogspace(x_min,x_max,n):
    """ like np.linspace. but (close to)  equidistant in logs

    Args:

        x_min (double): maximum value
        x_max (double): minimum value
        n (int): number of points
    
    Returns:

        y (list): grid with unequal spacing

    """

    pivot = np.abs(x_min) + 0.25
    y = np.geomspace(x_min + pivot, x_max + pivot, n) - pivot
    y[0] = x_min  # make sure *exactly* equal to x_min
    return y

@njit
def nonlinspace_jit(x_min, x_max, n, phi):
    """ like nonlinspace, but can be used in numba """
        
    y = np.zeros(n)

    y[0] = x_min
    for i in range(1,n):
        y[i] = y[i-1] + (x_max-y[i-1]) / (n-i)**phi 
    
    return y

def gauss_hermite(n):
    """ gauss-hermite nodes

    Args:

        n (int): number of points

    Returns:

        x (numpy.ndarray): nodes of length n
        w (numpy.ndarray): weights of length n

    """

    # a. calculations
    i = np.arange(1,n)
    a = np.sqrt(i/2)
    CM = np.diag(a,1) + np.diag(a,-1)
    L,V = np.linalg.eig(CM)
    I = L.argsort()
    V = V[:,I].T

    # b. nodes and weights
    x = L[I]
    w = np.sqrt(math.pi)*V[:,0]**2

    return x,w

def normal_gauss_hermite(sigma, n=7, mu=None, exp=True):
    """ normal gauss-hermite nodes

    Args:

        sigma (double): standard deviation
        n (int): number of points
        mu (double,optinal): mean
        exp (bool,optinal): take exp and correct mean (if not specified)

    Returns:

        x (numpy.ndarray): nodes of length n
        w (numpy.ndarray): weights of length n

    """

    if sigma == 0.0 or n == 1:
        x = np.ones(n)
        if mu is not None:
            x += mu
        w = np.ones(n)
        return x,w

    # a. GaussHermite
    x,w = gauss_hermite(n)
    x *= np.sqrt(2)*sigma 

    # b. log-normality
    if exp:
        if mu is None:
            x = np.exp(x - 0.5*sigma**2)
        else:
            x = np.exp(x + mu)
    else:
        if mu is None:
            x = x 
        else:
            x = x + mu

    w /= np.sqrt(math.pi)

    return x,w

def create_shocks(sigma_psi,Npsi,sigma_xi,Nxi,pi=0,mu=None):
    """ log-normal gauss-hermite nodes for permanent transitory model

    Args:

        sigma_psi (double): standard deviation of permanent shock
        Npsi (int): number of points for permanent shock
        sigma_xi (double): standard deviation of transitory shock
        Nxi (int): number of points for transitory shock        
        pi (double): probability of low income shock
        mu (double): value of low income shock

    Returns:

        psi (numpy.ndarray): nodes for permanent shock of length Npsi*Nxi+1
        psi_w (numpy.ndarray): weights for permanent shock of length Npsi*Nxi+1
        xi (numpy.ndarray): nodes for transitory shock of length Npsi*Nxi+1
        xi_w (numpy.ndarray): weights for transitory shock of length Npsi*Nxi+1
        Nshocks (int): number of nodes = Npsi*Nxi+1

    """

    # a. gauss hermite
    psi, psi_w = normal_gauss_hermite(sigma_psi, Npsi)
    xi, xi_w = normal_gauss_hermite(sigma_xi, Nxi)

    # b. add low inncome shock
    if pi > 0:
         
        # a. weights
        xi_w *= (1.0-pi)
        xi_w = np.insert(xi_w,0,pi)

        # b. values
        xi = (xi-mu*pi)/(1.0-pi)
        xi = np.insert(xi,0,mu)
    
    # c. tensor product
    psi,xi = np.meshgrid(psi,xi,indexing='ij')
    psi_w,xi_w = np.meshgrid(psi_w,xi_w,indexing='ij')

    return psi.ravel(), psi_w.ravel(), xi.ravel(), xi_w.ravel(), psi.size

def tauchen(mu,rho,sigma,m=3,N=7,cutoff=np.nan):
    """ tauchen approximation of autoregressive process

    Args:

        mu (double): mean
        rho (double): AR(1) coefficient
        sigma (double): std. of shock
        m (int): scale factor for width of grid
        N (int): number of grid points
        cutoff (double): 

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
  
    x[0]    = mu/(1-rho) - m*std_x
    x[N-1]  = mu/(1-rho) + m*std_x
  
    step    = (x[N-1]-x[0])/(N-1)
  
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
    ergodic = _find_ergodic(trans)

    # e. apply cutoff
    if not np.isnan(cutoff):  
        trans[trans < cutoff] = 0         

    # f. find cumsums
    trans_cumsum = np.array([np.cumsum(trans[i, :]) for i in range(N)])
    ergodic_cumsum = np.cumsum(ergodic)

    return x, trans, ergodic, trans_cumsum, ergodic_cumsum

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
    ergodic = _find_ergodic(trans)

    # d. scaling
    s = np.linspace(-1, 1, N)
    mean = np.sum(ergodic*s)
    sigma_ = np.sqrt(np.sum(ergodic*(s-mean)**2))
    s *= (sigma / sigma_)

    y = np.exp(s) / np.sum(ergodic * np.exp(s))

    # e. find cumsums
    trans_cumsum = np.array([np.cumsum(trans[i, :]) for i in range(N)])
    ergodic_cumsum = np.cumsum(ergodic)

    return y, trans, ergodic, trans_cumsum, ergodic_cumsum

def _find_ergodic(trans,atol=1e-13,rtol=0):
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
