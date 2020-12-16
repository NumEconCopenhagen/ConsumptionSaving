# -*- coding: utf-8 -*-
"""newton_raphson

Provides a Numba jit compilled newton raphson optimizer for a custom objective function.
"""

import numpy as np
from numba import njit

@njit
def optimizer(obj,x0,args=(),max_iter=5000,grad_step=1.0e-5,tol_f=1.0e-6,tol_x=1.0e-5):
    """ newton-raphson optimizer
    
    Args:

        obj (callable): function to optimize over
        x0 (ndarray): vector of initial guesses
        args: additional arguments to the objective function
        max_inter (int,optional): maximum number of interation
        grad_step (double,optional): step size for numerical gradients
        tol_f (double,optional): tolerance for function
        tol_x (double,optional): tolerance for x

    Returns:

        (ndarray): optimization result

    """
    
    dim = x0.size

    # allocate
    x_min = np.zeros(dim)
    x_now = np.zeros(dim)
    x_grad = np.zeros(dim)
    x_hess = np.zeros(dim)
    grad = np.zeros(dim)
    grad_hess = np.zeros(dim)
    hess = np.zeros((dim,dim))

    # initial values
    for i in range(dim):
        x_min[i] = x0[i]
    f_min = obj(x_min,*args)

    # iterate
    for it in range(max_iter):

        # a. gradient and hess
        num_grad(obj,x_min,grad_step,f_min,grad,x_grad,*args)
        num_hess(obj,x_min,grad_step,grad,hess,x_grad,x_hess,grad_hess,*args) 

        # b. direction
        hess_det = np.linalg.det(hess)
        d = - np.sign(hess_det)*(np.linalg.inv(hess) @ np.transpose(grad))
            
        # c. update
        x_now[:] = x_min[:] + d
        f_now = obj(x_now,*args)

        # d. check for convergence
        if np.abs(f_now-f_min) < tol_f or np.amax(np.abs(x_now-x_min)) < tol_x or it >= max_iter-1:
            
            x_min[:] = x_now[:]
            f_min = f_now
            break
                
        # e. line-search
        if f_now >= f_min: # worsening, take avg of the last two iterations

            x_step = 0.5*(x_min[:] + x_now[:])
            f_step = obj(x_step,*args)

            if f_step < f_now:
                f_now = f_step
                x_now[:] = x_step[:]

        # f. update values
        x_min[:] = x_now[:]
        f_min = f_now

    return x_min

@njit    
def num_grad(obj,x,grad_step,f0,grad,x_grad,*args):
    """ calculate numerical gradient
    
    Args:

        obj (callable): input, objective function with *args (must be decorated with @njit)
        x (ndarray): input, starting point
        grad_step (double): input, step size for gradient
        f0 (double): function input, value at stating point
        grad (ndarray): output, resulting gradient
        x_grad (ndarray): input, working memory
        *args: additional arguments to the objective function

    """

    for i, xi in enumerate(x):

        step_now = np.fmax(grad_step*xi,grad_step) # step
        
        x_grad = x.copy()
        x_grad[:] = x[:] # baseline
        x_grad[i] = xi + step_now # change
        grad[i] = (obj(x_grad,*args)-f0)/step_now # gradient

@njit 
def num_hess(obj,x,grad_step,grad,hess,x_grad,x_hess,grad_hess,*args):
    """ calculate numerical hessian
    
    Args:

        obj (callable): input, objective function with *args (must be decorated with @njit)
        x (ndarray): input, starting point
        grad_step (double): input, step size for gradient
        hess (ndarray): output, resulting hessian
        x_grad (ndarray): input, working memory
        x_hess (ndarray): input, working memory
        grad_hess (ndarray): input, working memory
        *args: additional arguments to the objective function

    """

    for i, xi in enumerate(x):
        
        step_now = np.fmax(grad_step*xi,grad_step) # step

        x_hess[:] = x[:] # baseline
        x_hess[i] = xi + step_now # change
        f0 = obj(x_hess,*args) # value

        num_grad(obj,x_hess,grad_step,f0,grad_hess,x_grad,*args) # gradient
        hess[:,i] = (grad_hess - grad)/step_now # change in gradient
