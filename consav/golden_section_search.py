# -*- coding: utf-8 -*-
"""golden_section_search

This module provides a Numba JIT compilled golden section search optimizer for a custom objective.

"""

import math
from numba import njit

# constants
inv_phi = (math.sqrt(5) - 1) / 2 # 1/phi                                                                                                                
inv_phi_sq = (3 - math.sqrt(5)) / 2 # 1/phi^2                                                                                                                

# create
def create_optimizer(obj):
    """ create golden section search optimizer for the function obj
    
    Args:

        obj (callable): objective function with *args (must be decorated with @njit)

    Returns:

        optimizer (callable): optimizer called as (a,b,tol,*args) where [a,b] is the starting bracket
    
    """

    @njit
    def golden_section_search(a,b,tol,*args):
        """ optimizer
        
        Args:

            a (double): minimum of starting bracket
            b (double): maximum of starting bracket
            tol (tok): tolerance
            *args: additional arguments to the objective function

        Returns:

            optimizer (callable): optimizer called as (a,b,tol,*args) where [a,b] is the starting bracket
        
        """
            
        # a. distance
        dist = b - a
        if dist <= tol: 
            return (a+b)/2

        # b. number of iterations
        n = int(math.ceil(math.log(tol/dist)/math.log(inv_phi)))

        # c. potential new mid-points
        c = a + inv_phi_sq * dist
        d = a + inv_phi * dist
        yc = obj(c,*args)
        yd = obj(d,*args)

        # d. loop
        for _ in range(n-1):
            if yc < yd:
                b = d
                d = c
                yd = yc
                dist = inv_phi*dist
                c = a + inv_phi_sq * dist
                yc = obj(c,*args)
            else:
                a = c
                c = d
                yc = yd
                dist = inv_phi*dist
                d = a + inv_phi * dist
                yd = obj(d,*args)

        # e. return
        if yc < yd:
            return (a+d)/2
        else:
            return (c+b)/2

    return golden_section_search