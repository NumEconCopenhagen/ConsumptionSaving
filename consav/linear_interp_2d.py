import numpy as np
from numba import njit, boolean, int32, double, void
from .linear_interp import binary_search

@njit(double(double[:],double[:],double[:,:],double,double,int32,int32),fastmath=True)
def _interp_2d(grid1,grid2,value,xi1,xi2,j1,j2):
    """ 2d interpolation for one point with known location
        
    Args:

        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (2d)
        xi1 (double): input point
        xi2 (double): input point
        j1 (int): location in grid 
        j2 (int): location in grid

    Returns:

        yi (double): output

    """

    # a. left/right
    nom_1_left = grid1[j1+1]-xi1
    nom_1_right = xi1-grid1[j1]

    nom_2_left = grid2[j2+1]-xi2
    nom_2_right = xi2-grid2[j2]

    # b. interpolation
    denom = (grid1[j1+1]-grid1[j1])*(grid2[j2+1]-grid2[j2])
    nom = 0
    for k1 in range(2):
        nom_1 = nom_1_left if k1 == 0 else nom_1_right
        for k2 in range(2):
            nom_2 = nom_2_left if k2 == 0 else nom_2_right                    
            nom += nom_1*nom_2*value[j1+k1,j2+k2]

    return nom/denom

@njit(double(double[:],double[:],double[:,:],double,double),fastmath=True)
def interp_2d(grid1,grid2,value,xi1,xi2):
    """ 2d interpolation for one point
        
    Args:

        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (2d)
        xi1 (double): input point
        xi2 (double): input point

    Returns:

        yi (double): output

    """

    # a. search in each dimension
    j1 = binary_search(0,grid1.size,grid1,xi1)
    j2 = binary_search(0,grid2.size,grid2,xi2)

    return _interp_2d(grid1,grid2,value,xi1,xi2,j1,j2)

@njit(double(int32[:],double[:],double[:],double[:,:],double,double),fastmath=True)
def interp_2d_to_rep(js,grid1,grid2,value,xi1,xi2):
    """ 2d interpolation for one point (with later repition without search)
        
    Args:

        js (numpy.nddarray): information on search
        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (2d)
        xi1 (double): input point
        xi2 (double): input point

    Returns:

        yi (double): output

    """

    # a. search in each dimension
    js[0] = j1 = binary_search(0,grid1.size,grid1,xi1)
    js[1] = j2 = binary_search(0,grid2.size,grid2,xi2)

    return _interp_2d(grid1,grid2,value,xi1,xi2,j1,j2)

@njit(double(int32[:],double[:],double[:],double[:,:],double,double),fastmath=True)
def interp_2d_from_rep(js,grid1,grid2,value,xi1,xi2):
    """ 2d interpolation for one point without search
        
    Args:

        js (numpy.nddarray): information on search
        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (2d)
        xi1 (double): input point
        xi2 (double): input point

    Returns:

        yi (double): output

    """

    # a. search in each dimension
    j1 = js[0]
    j2 = js[1]

    return _interp_2d(grid1,grid2,value,xi1,xi2,j1,j2)

@njit(void(double[:],double[:],double[:,:],double[:],double[:],double[:]),fastmath=True)
def interp_2d_vec(grid1,grid2,value,xi1,xi2,yi):
    """ 2d interpolation for vector of points
        
    Args:

        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (2d)
        xi1 (numpy.ndarray): input vector
        xi2 (numpy.ndarray): input vector
        yi (numpy.ndarray): output vector

    """

    for i in range(yi.size):
        yi[i] = interp_2d(grid1,grid2,value,xi1[i],xi2[i])

@njit(int32[:](double[:],double,int32),fastmath=True)
def interp_2d_prep(grid1,xi1,Nyi):
    """ preperation for 2d interpolation of only last dimension
        
    Args:

        grid1 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (2d)
        xi1 (double): input point
        Nyi (int): number of points to be evaluated

    Returns:

        prep (numpy.ndarray): information for remaining operations

    """

    # a. search in non-last dimensions
    j1 = binary_search(0,grid1.size,grid1,xi1)
    
    # b. prep
    prep = np.zeros((1+Nyi,),dtype=np.int32)
    prep[Nyi+0] = j1

    return prep

@njit(void(int32[:],double[:],double[:],double[:,:],double,double[:],double[:],boolean,boolean),fastmath=True)
def _interp_2d_only_last_vec(prep,grid1,grid2,value,xi1,xi2,yi,monotone,search):
    """ 2d interpolation for vector of points
        
    Args:

        prep (numpy.ndarray): information for remaining operations
        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (2d)
        xi1 (double): input point
        xi2 (numpy.ndarray): vector of points
        yi (numpy.ndarray): output vector
        monotone (bool): indicator for whether xi2 is monotone
        search (bool): indicator for whether search is needed at all

    """

    # a. search in last dimension
    Nyi = yi.size
    j1 = prep[Nyi + 0]
    if search:
        for i in range(Nyi):
            if monotone and i > 0:
                j2 = prep[i-1]
                while xi2[i] >= grid2[j2+1] and j2 < grid2.size-2:
                    j2 += 1
                prep[i] = j2
            else:
                prep[i] = binary_search(0,grid2.size,grid2,xi2[i])

    # b. initialize
    for i in range(yi.size):
        yi[i] = 0.0

    # c. interpolation
    nom_1_left = grid1[j1+1]-xi1
    nom_1_right = xi1-grid1[j1]

    denom = (grid1[j1+1]-grid1[j1])
    for k1 in range(2):
        nom_1 = nom_1_left if k1 == 0 else nom_1_right   
        for i in range(yi.size):
            for k2 in range(2):
                j2 = prep[i]
                nom_2 = grid2[j2+1]-xi2[i] if k2 == 0 else xi2[i]-grid2[j2]            
                yi[i] += nom_1*nom_2*value[j1+k1,j2+k2]

    for i in range(Nyi):
        j2 = prep[i]
        yi[i] /= denom*(grid2[j2+1]-grid2[j2])

@njit(void(int32[:],double[:],double[:],double[:,:],double,double[:],double[:]),fastmath=True)
def interp_2d_only_last_vec(prep,grid1,grid2,value,xi1,xi2,yi):
    """ 2d interpolation for vector of points
        
    Args:

        prep (numpy.ndarray): information for remaining operations
        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (2d)
        xi1 (double): input point
        xi2 (numpy.ndarray): vector of points
        yi (numpy.ndarray): output vector

    """

    _interp_2d_only_last_vec(prep,grid1,grid2,value,xi1,xi2,yi,False,True)

@njit(void(int32[:],double[:],double[:],double[:,:],double,double[:],double[:]),fastmath=True)
def interp_2d_only_last_vec_mon(prep,grid1,grid2,value,xi1,xi2,yi):
    """ 2d interpolation for vector of points where xi2 is monotone
        
    Args:

        prep (numpy.ndarray): information for remaining operations
        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (2d)
        xi1 (double): input point
        xi2 (numpy.ndarray): vector of points
        yi (numpy.ndarray): output vector

    """

    _interp_2d_only_last_vec(prep,grid1,grid2,value,xi1,xi2,yi,True,True)    

@njit(void(int32[:],double[:],double[:],double[:,:],double,double[:],double[:]),fastmath=True)
def interp_2d_only_last_vec_mon_rep(prep,grid1,grid2,value,xi1,xi2,yi):
    """ 2d interpolation for vector of points where xi2 is monotone and search is not needed
        
    Args:

        prep (numpy.ndarray): information for remaining operations
        grid1 (numpy.ndarray): 1d grid
        grid2 (numpy.ndarray): 1d grid
        value (numpy.ndarray): value array (2d)
        xi1 (double): input point
        xi2 (numpy.ndarray): vector of points
        yi (numpy.ndarray): output vector

    """

    _interp_2d_only_last_vec(prep,grid1,grid2,value,xi1,xi2,yi,True,False)       