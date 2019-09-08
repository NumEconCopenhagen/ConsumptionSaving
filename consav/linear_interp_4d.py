import numpy as np
from numba import njit, boolean, int32, double, void
from .linear_interp import binary_search

@njit(double(double[:],double[:],double[:],double[:],double[:,:,:,:],double,double,double,double,int32,int32,int32,int32),fastmath=True)
def _interp_4d(grid1,grid2,grid3,grid4,value,xi1,xi2,xi3,xi4,j1,j2,j3,j4):
    """ 4d interpolation for one point with known location
        
    Args:

        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        grid3 (numpy.ndarray): 1d grid
        grid4 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (4d)
        xi1 (double): input point
        xi2 (double): input point
        xi3 (double): input point
        xi4 (double): input point
        j1 (int): location in grid 
        j2 (int): location in grid
        j3 (int): location in grid
        j4 (int): location in grid

    Returns:

        yi (double): output

    """

    # a. left/right
    nom_1_left = grid1[j1+1]-xi1
    nom_1_right = xi1-grid1[j1]

    nom_2_left = grid2[j2+1]-xi2
    nom_2_right = xi2-grid2[j2]

    nom_3_left = grid3[j3+1]-xi3
    nom_3_right = xi3-grid3[j3]

    nom_4_left = grid4[j4+1]-xi4
    nom_4_right = xi4-grid4[j4]

    # b. interpolation
    denom = (grid1[j1+1]-grid1[j1])*(grid2[j2+1]-grid2[j2])*(grid3[j3+1]-grid3[j3])*(grid4[j4+1]-grid4[j4])
    nom = 0
    for k1 in range(2):
        nom_1 = nom_1_left if k1 == 0 else nom_1_right
        for k2 in range(2):
            nom_2 = nom_2_left if k2 == 0 else nom_2_right       
            for k3 in range(2):
                nom_3 = nom_3_left if k3 == 0 else nom_3_right  
                for k4 in range(2):
                    nom_4 = nom_4_left if k4 == 0 else nom_4_right  
                    nom += nom_1*nom_2*nom_3*nom_4*value[j1+k1,j2+k2,j3+k3,j4+k4]

    return nom/denom

@njit(double(double[:],double[:],double[:],double[:],double[:,:,:,:],double,double,double,double),fastmath=True)
def interp_4d(grid1,grid2,grid3,grid4,value,xi1,xi2,xi3,xi4):
    """ 4d interpolation for one point
        
    Args:

        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        grid3 (numpy.ndarray): 1d grid
        grid4 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (4d)
        xi1 (double): input point
        xi2 (double): input point
        xi3 (double): input point
        xi4 (double): input point

    Returns:

        yi (double): output

    """

    # a. search in each dimension
    j1 = binary_search(0,grid1.size,grid1,xi1)
    j2 = binary_search(0,grid2.size,grid2,xi2)
    j3 = binary_search(0,grid3.size,grid3,xi3)
    j4 = binary_search(0,grid4.size,grid4,xi4)

    return _interp_4d(grid1,grid2,grid3,grid4,value,xi1,xi2,xi3,xi4,j1,j2,j3,j4)