# -*- coding: utf-8 -*-
""" grids

Functions for creating grids.
"""

import numpy as np
from numba import njit

def nonlinspace(x_min,x_max,n,phi):
    """ like np.linspace. but with unequal spacing

    Args:

        x_min (double): minimum value
        x_max (double): maximum value
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
 
@njit
def nonlinspace_jit(x_min,x_max,n,phi):
    """ like nonlinspace, but can be used in numba """
        
    y = np.zeros(n)

    y[0] = x_min
    for i in range(1,n):
        y[i] = y[i-1] + (x_max-y[i-1]) / (n-i)**phi 
    
    return y

def equilogspace(x_min,x_max,n):
    """ like np.linspace. but (close to) equidistant in logs

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
