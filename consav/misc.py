# -*- coding: utf-8 -*-
"""newton_raphson

This module provides misc functions.
"""

import math
import numpy as np

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
 
def gauss_hermite(n):
    """ gauss-hermite nodes

    Args:

        n (int): number of points

    Returns:

        x (numpy.ndarray): nodes of length n
        w (numpy.ndarray): weights of length n

    """

    # a. calculations
    i = np.arange(1,5)
    a = np.sqrt(i/2)
    CM = np.diag(a,1) + np.diag(a,-1)
    L,V = np.linalg.eig(CM)
    I = L.argsort()
    V = V[:,I].T

    # b. nodes and weights
    x = L[I]
    w = np.sqrt(math.pi)*V[:,0]**2

    return x,w

def log_normal_gauss_hermite(sigma, n):
    """ log-normal gauss-hermite nodes

    Args:

        sigma (double): standard deviation
        n (int): number of points

    Returns:

        x (numpy.ndarray): nodes of length n
        w (numpy.ndarray): weights of length n

    """

    if sigma == 0.0 or n == 1:
        x = np.ones(n)
        w = np.ones(n)
        return x,w

    # a. GaussHermite
    x,w = gauss_hermite(n)

    # b. log-normality
    x = np.exp(x*np.sqrt(2)*sigma-0.5*sigma**2)
    w = w/np.sqrt(math.pi)

    # c. assert a mean of one
    assert(1.0-np.sum(w*x) < 1e-8)

    return x,w

def create_shocks(sigma_psi,Npsi,sigma_xi,Nxi,pi,mu):
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
    psi, psi_w = log_normal_gauss_hermite(sigma_psi, Npsi)
    xi, xi_w = log_normal_gauss_hermite(sigma_xi, Nxi)
 
    # b. add low inncome shock
    if pi > 0:
         
        # a. weights
        xi_w *= (1.0-pi)
        xi_w = np.insert(xi_w,0,pi)

        # b. values
        xi = (xi-mu*pi)/(1.0-pi)
        xi = np.insert(xi,0,mu)

    assert(np.allclose(np.sum(psi_w),1))
    assert(np.allclose(np.sum(xi_w),1))
    assert(np.allclose(np.sum(psi_w*psi),1))
    assert(np.allclose(np.sum(xi_w*xi),1))
    
    # c. tensor product
    psi,xi = np.meshgrid(psi,xi,indexing='ij')
    psi_w,xi_w = np.meshgrid(psi_w,xi_w,indexing='ij')

    assert(np.allclose(np.sum(psi_w*xi_w),1))

    return psi.ravel(), psi_w.ravel(), xi.ravel(), xi_w.ravel(), psi.size