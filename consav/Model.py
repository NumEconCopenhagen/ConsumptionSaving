# -*- coding: utf-8 -*-
"""Model

This module provides a class for consumption-saving models with methods for saving and loading
and interfacing with C++. 

"""

import os
import ctypes as ct
import pickle
import numpy as np
from numba import jitclass, int32, double, boolean

from . import cpptools

# for save/load
def _filename(model):    
    if hasattr(model,'solmethod'):
        return f'{model.name}_{model.solmethod}'
    else:
        return f'{model.name}'

def _convert_to_dict(someclass,somelist):

    if somelist is None:
        return {}
    elif len(somelist) > 0 and type(somelist[0]) is str:
        keys = [var for var in somelist]
    else:
        keys = [var[0] for var in somelist]
    values = [getattr(someclass,key) for key in keys]
    return {key:val for key,val in zip(keys,values)}

# main
class ModelClass():
    
    def __init__(self):
        """ defines default attributes """

        self.name = None # name of parametrization
        self.solmethod = None # solution methods
        self.compiler = None # compiler
        self.parlist = [] # list of parameters, (name,numba type)
        self.par = None # jitted class with fields as in parlist
        self.sollist = [] # list of parameters, (name,numba type)        
        self.sol = None # jitted class with fields as in sollist
        self.simlist = [] # list of parameters, (name,numba type)        
        self.sim = None # jitted class with fields as in sollist        
        self.log = None # log text
        self.vs_path = 'C:/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Auxiliary/Build/'
        self.intel_path = 'C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2018.5.274/windows/bin/'
        self.intel_vs_version = 'vs2017'
        self.savelist = []

    def create_subclasses(self,parlist,sollist,simlist):
        """ create jitted subclasses par, sol, sim

        Args:

            parlist (list): list of parameters, grids etc. with elements (name, numba type)
            sollist (list): list of solution variables with elements (name, numba type)
            simlist (list): list of simulation variables with elements (name, numba type)
        
        Returns:

            par (class): class with parameters, grids etc.
            sol (class): class with solution variables
            sim (class): class with simulation variables

        """
    
        self.parlist = parlist
        self.sollist = sollist
        self.simlist = simlist

        @jitclass(self.parlist) # numba class with variables in parlist
        class ParClass():
            def __init__(self):
                pass

        @jitclass(self.sollist) # numba jit class with variables in sollist
        class SolClass():
            def __init__(self):
                pass

        @jitclass(self.simlist) # numba jit class with variables in simlist
        class SimClass():
            def __init__(self):
                pass

        return ParClass(),SolClass(),SimClass()

    def get_par_dict(self):
        """ get dictionary for the par subclass """
        
        return _convert_to_dict(self.par,self.parlist)

    def save(self):
        """ save the model parameters, the solution simulation variables """
        
        if not os.path.exists('data'):
            os.makedirs('data')

        # a. save parameters pickle
        par_dict = _convert_to_dict(self.par,self.parlist)
        with open(f'data/{_filename(self)}_par.p', 'wb') as f:
            pickle.dump(par_dict, f)

        # b. solution
        sol_dict = _convert_to_dict(self.sol,self.sollist)
        np.savez(f'data/{_filename(self)}_sol.npz', **sol_dict)
    
        # c. simulation
        sim_dict = _convert_to_dict(self.sim,self.simlist)
        np.savez(f'data/{_filename(self)}_sim.npz', **sim_dict)

        # d. additional
        if hasattr(self,'savelist'):
            addi_dict = _convert_to_dict(self,self.savelist + ['savelist'])
            with open(f'data/{_filename(self)}.p', 'wb') as f:
                pickle.dump(addi_dict, f)        

    def load(self):
        """ load the model parameters and solution and simulation variables"""

        # a. parameters
        with open(f'data/{_filename(self)}_par.p', 'rb') as f:
            self.par_dict = pickle.load(f)
        for key,val in self.par_dict.items():
            setattr(self.par,key,val)

        # b. solution
        with np.load(f'data/{_filename(self)}_sol.npz') as data:
            for key in data.files:
                setattr(self.sol,key,data[key])

        # c. solution
        with np.load(f'data/{_filename(self)}_sim.npz') as data:
            for key in data.files:
                setattr(self.sim,key,data[key])

        # d. additional
        filesavelist = f'data/{_filename(self)}.p'
        if os.path.isfile(filesavelist):
            with open(filesavelist, 'rb') as f:
                addi_dict = pickle.load(f)
            for key,val in addi_dict.items():
                setattr(self,key,val)

    def __str__(self):
        """ called when model is printed """ 
        
        # a. keys and values in parlist
        keys = [var[0] for var in self.parlist]
        values = [getattr(self.par,key) for key in keys]

        # b. create description
        description = f'Modelclass: {self.__class__.__name__}\n'
        description += 'Parameters:\n'
        for var,val in zip(self.parlist,values):
            if var[1] in [int32,double]:
                description += f' {var[0]} = {val}\n'
            elif var[1] == boolean:
                if var[1]:
                    description += f' {var[0]} = True\n'
                else:
                    description += f' {var[0]} = False\n'
            elif var[1] in [double[:],double[:,:],double[:,:,:],double[:,:,:,:]]:
                description += f' {var[0]} = [array of doubles]\n'
            else:
                description += f' {var[0]} = ?\n'

        return description 

    #######################
    ## interact with cpp ##
    #######################
    
    def setup_cpp(self,use_nlopt=False):
        """ setup interface to cpp files 
        
        Args:

            compiler (str,optional): compiler choice (vs or intel)
            use_nlopt (bool,optional): use NLopt optimizer

        """

        # a. setup NLopt
        if not os.path.isfile(f'{os.getcwd()}/libnlopt-0.dll'):
            cpptools.setup_nlopt(vs_path=self.vs_path)
            
        # b. dictionary of cppfiles
        self.cppfiles = dict()

        # c. ctypes version of par and sol classes
        self.parcpp = cpptools.setup_struct(self.parlist,'par_struct','cppfuncs//par_struct.cpp')
        self.solcpp = cpptools.setup_struct(self.sollist,'sol_struct','cppfuncs//sol_struct.cpp')
        self.simcpp = cpptools.setup_struct(self.simlist,'sim_struct','cppfuncs//sim_struct.cpp')

    def link_cpp(self,filename,funcnames,do_compile=True,do_print=False):
        """ link c++ library
        
        Args:

            filename (str): path to .dll file (no .dll extension!)
            funcames (list): list of function names
            do_compile (bool): compile from .cpp to .dll
            do_print (bool): print if succesfull

        """

        use_openmp_with_vs = False

        if do_compile:
            cpptools.compile('cppfuncs//' + filename,
                compiler=self.compiler,
                vs_path=self.vs_path,
                intel_path=self.intel_path, 
                intel_vs_version=self.intel_vs_version, 
                do_print=do_print)

        funcs = [(name,[ct.POINTER(self.parcpp),ct.POINTER(self.solcpp),ct.POINTER(self.simcpp)]) for name in funcnames]
        if self.compiler == 'vs': 
            funcs.append(('setup_omp',[]))
            use_openmp_with_vs = True

        self.cppfiles[filename] = cpptools.link(filename,funcs,use_openmp_with_vs=use_openmp_with_vs,do_print=do_print)
                
    def delink_cpp(self,filename,do_print=False,do_remove=True):
        """ delink cpp library
        
        Args:
            filename (str): path to .dll file (no .dll extension!).
            do_print (bool,optional): print if successfull    
            do_remove (bool,optional): remove dll file after delinking
                    
         """

        cpptools.delink(self.cppfiles[filename],filename,do_print=do_print,do_remove=do_remove)

    def call_cpp(self,filename,funcname):
        """ call c++ function in linked c++ library
        
        Args:
        
            filename (str): path to .cpp file (no .cpp extension!).
            funcname (str): print if successfull    
        
        """ 
            
        p_par = cpptools.get_struct_pointer(self.par,self.parcpp)
        p_sol = cpptools.get_struct_pointer(self.sol,self.solcpp)
        p_sim = cpptools.get_struct_pointer(self.sim,self.simcpp)
        
        funcnow = getattr(self.cppfiles[filename],funcname)
        funcnow(p_par,p_sol,p_sim)
    
    ##############
    ## run file ##
    ##############

    def write_run_file(self,filename='run.py',name='',solmethod='',method='',**kwargs):
        """ call c++ function in linked c++ library
        
        Args:
        
            filename (str,optional): name of run fil
            name (str,optional): name of instance
            solmethod (str,optional): solutionmethod
            method (str,optional): method to call in run file
            **kwargs: parameters to be updated from baseline
        
        """ 

        modulename = self.__class__.__module__
        modelname = self.__class__.__name__

        if name == '':
            name = self.name

        if solmethod == '':
            solmethod = self.solmethod

        if method == '':
            method = 'solve'

        with open(f'{filename}', 'w') as txtfile:
            
            txtfile.write(f'from {modulename} import {modelname}\n')
            txtfile.write('updpar = dict()\n')
            for key,val in kwargs.items():
                txtfile.write(f'updpar["{key}"] = {val}\n')

            txtfile.write(f'model = {modelname}(name="{name}",solmethod="{solmethod}",**updpar)\n')
            txtfile.write(f'model.{method}()\n')