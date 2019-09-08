# -*- coding: utf-8 -*-
"""linear_interp

Functions for multi-linear interpolation.
"""

import numpy as np
from numba import njit, int32, double

#################
# binary search #
#################

@njit(int32(int32,int32,double[:],double),fastmath=True)
def binary_search(imin,Nx,x,xi):
        
    # a. checks
    if xi <= x[0]:
        return 0
    elif xi >= x[Nx-2]:
        return Nx-2
    
    # b. binary search
    half = Nx//2
    while half:
        imid = imin + half
        if x[imid] <= xi:
            imin = imid
        Nx -= half
        half = Nx//2
        
    return imin

#############
# functions #
#############

from .linear_interp_1d import *
from .linear_interp_2d import *
from .linear_interp_3d import *
from .linear_interp_4d import *