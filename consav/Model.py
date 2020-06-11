# -*- coding: utf-8 -*-
"""Model

This module provides a class for consumption-saving models with methods for saving and loading
and interfacing with numba functions C++.

Modles 

"""

import os
import traceback
import time
import ctypes as ct
import copy
import pickle
from types import SimpleNamespace
from collections import namedtuple

import numpy as np

from . import cpptools

# main
class ModelClass():
    
    def __init__(self,name=None,load=False,**kwargs):
        """ defines default attributes """

        # a. name
        if name is None: raise Exception('name must be specified')
        self.name = name

        # b. C++ information
        self.cppinfo = {}
        self.cppinfo['compiler'] = 'vs'
        self.cppinfo['vs_path'] = 'C:/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Auxiliary/Build/'
        self.cppinfo['intel_path'] = 'C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2018.5.274/windows/bin/'
        self.cppinfo['intel_vs_version'] = 'vs2017'

        # c. new or load
        if not load: # new

            # i. empty lists
            self.not_float_list = []
            self.savelist = []

            # ii. create par, sol and sim 
            self.par = SimpleNamespace()
            self.sol = SimpleNamespace()
            self.sim = SimpleNamespace()      

            # ii. set independent variables in par, sol and sim
            self.setup() # baseline
            for key,val in kwargs.items(): setattr(self.par,key,val)
            
            # iii. set independent variables in par, sol and sim
            self.allocate()

        else: # load

            self.load()
            for key,val in kwargs.items(): setattr(self.par,key,val)

        # d. infrastructure
        self.setup_infrastructure()
        self.update_jit()

    def setup(self):
        """ set independent variables in par, sol and sim """

        raise Exception('The model must have defined an .setup() method')

    def allocate(self):
        """ set independent variables in par, sol and sim """

        raise Exception('The model must have defined an .allocate() method')

    ####################
    ## infrastructure ##
    ####################

    def setup_infrastructure(self):
        """ setup infrastructure to call numba jit functions and C++ functions """
        
        # a. convert to dictionaries
        par_dict = self.par.__dict__
        sol_dict = self.sol.__dict__
        sim_dict = self.sim.__dict__

        # b. type check
        def check(key,val):

            scalar_or_ndarray = np.isscalar(val) or type(val) is np.ndarray
            assert scalar_or_ndarray, f'{key} is not scalar or numpy array'
            
            listed_non_float = not np.isscalar(val) or type(val) is str or type(val) is np.float or key in self.not_float_list
            assert listed_non_float, f'{key} is {type(val)}, not float, but not on the list'

        for key,val in par_dict.items(): check(key,val)
        for key,val in sol_dict.items(): check(key,val)
        for key,val in sim_dict.items(): check(key,val)

        # c. namedtuple (definitions)
        self.ParClass = namedtuple(f'ParClass',[key for key in par_dict.keys()])
        self.SolClass = namedtuple(f'SolClass',[key for key in sol_dict.keys()])        
        self.SimClass = namedtuple(f'SimClass',[key for key in sim_dict.keys()])        

    def update_jit(self):
        """ update values and references in par_jit, sol_jit, sim_jit """

        self.par_jit = self.ParClass(**self.par.__dict__)
        self.sol_jit = self.SolClass(**self.sol.__dict__)
        self.sim_jit = self.SimClass(**self.sim.__dict__)

    ####################
    ## save-copy-load ##
    ####################

    def save(self,drop_sol=False,drop_sim=False):
        """ save the model parameters, the solution simulation variables (in /data) """
        
        if not os.path.exists('data'):
            os.makedirs('data')

        # a. save parameters
        with open(f'data/{self.name}_par.p', 'wb') as f:
            pickle.dump(self.par, f)

        # b. solution
        if drop_sol:
            sol_dict = {}
        else:
            sol_dict = self.sol.__dict__
        np.savez(f'data/{self.name}_sol.npz', **sol_dict)
    
        # c. simulation
        if drop_sim:
            sim_dict = {}
        else:        
            sim_dict = self.sim.__dict__
        np.savez(f'data/{self.name}_sim.npz', **sim_dict)

        # d. additional
        internal_savelist = [key for key in ['cppinfo','not_float_list','savelist'] if hasattr(self,key)]
        savelist = self.savelist + internal_savelist
        savelist_dict = {key:getattr(self,key) for key in savelist}
        with open(f'data/{self.name}.p', 'wb') as f:
            pickle.dump(savelist_dict, f)     

    def copy(self,name=None,**kwargs):
        """ copy the model parameters, the solution simulation variables """
        
        if name is None: name = f'{self.name}_copy' # if not
        other = self.__class__(name=name)

        # a. parameters
        for key,val in self.par.__dict__.items():
            setattr(other.par,key,copy.copy(val))

        # b. solution
        for key,val in self.sol.__dict__.items():
            setattr(other.sol,key,copy.copy(val))
    
        # c. simulation
        for key,val in self.sim.__dict__.items():
            setattr(other.sim,key,copy.copy(val))

        # d. savelist
        internal_savelist = [key for key in ['cppinfo','not_float_list','savelist'] if hasattr(self,key)]
        savelist = self.savelist + internal_savelist
        for key in savelist:
            setattr(other,key,copy.deepcopy(getattr(self,key)))

        # e. update
        for key,val in kwargs.items(): setattr(other.par,key,val)

        return other

    def load(self):
        """ load par, sol, sim and anything in .savelist """

        # a. parameters
        with open(f'data/{self.name}_par.p', 'rb') as f:
            self.par = pickle.load(f)

        # b. solution
        self.sol = SimpleNamespace()
        with np.load(f'data/{self.name}_sol.npz') as data:
            for key in data.files:
                setattr(self.sol,key,data[key])

        # c. simulation
        self.sim = SimpleNamespace()            
        with np.load(f'data/{self.name}_sim.npz') as data:
            for key in data.files:
                setattr(self.sim,key,data[key])

        # d. additional
        filesavelist = f'data/{self.name}.p'
        if os.path.isfile(filesavelist):
            
            with open(filesavelist, 'rb') as f:
                savelist_dict = pickle.load(f)

            for key,val in savelist_dict.items():
                setattr(self,key,val)

    ##########
    ## print #
    ##########

    def __str__(self):
        """ called when model is printed """ 
        
        def print_items(sn):
            """ print items in SimpleNamespace """

            description = ''
            nbytes = 0

            for key,val in sn.__dict__.items():

                if np.isscalar(val) and not type(val) is np.bool:
                    description += f' {key} = {val} [{type(val).__name__}]\n'
                elif type(val) is np.bool:
                    if val:
                        description += f' {key} = True\n'
                    else:
                        description += f' {key} = False\n'
                elif type(val) is np.ndarray:
                    description += f' {key} = ndarray with shape = {val.shape} [dtype: {val.dtype}]\n'            
                    nbytes += val.nbytes
                else:                
                    description += f' {key} = ?\n'

            description += f'memory, gb: {nbytes/(10**9):.1f}\n' 
            return description

        description = f'Modelclass: {self.__class__.__name__}\n\n'

        description += 'Parameters:\n'
        description += print_items(self.par)
        description += '\n'

        description += 'Solution:\n'
        description += print_items(self.sol)
        description += '\n'

        description += 'Simulation:\n'
        description += print_items(self.sim)

        return description 

    #######################
    ## interact with cpp ##
    #######################
    
    def setup_cpp(self,use_nlopt=False):
        """ setup interface to cpp files """

        # a. setup NLopt
        if use_nlopt and not os.path.isfile(f'{os.getcwd()}/libnlopt-0.dll'):
            cpptools.setup_nlopt(vs_path=self.cppinfo['vs_path'])
            
        # b. dictionary of cppfiles
        self.cppfiles = dict()

        # c. ctypes version of par, sol and sim classes
        self.par_cpp = cpptools.setup_struct(self.par,'par_struct','cppfuncs//par_struct.cpp')
        self.sol_cpp = cpptools.setup_struct(self.sol,'sol_struct','cppfuncs//sol_struct.cpp')
        self.sim_cpp = cpptools.setup_struct(self.sim,'sim_struct','cppfuncs//sim_struct.cpp')

    def link_cpp(self,filename,funcspecs,do_compile=True,do_print=False):
        """ link C++ library
        
        Args:

            filename (str): path to .dll file (no .dll extension!)
            funcspecs (list): list of function names and (optionally) arguments
            do_compile (bool): compile from .cpp to .dll
            do_print (bool): print if succesfull

        """

        use_openmp_with_vs = False

        # a. compile
        if do_compile:

            cpptools.compile('cppfuncs//' + filename,
                compiler=self.cppinfo['compiler'],
                vs_path=self.cppinfo['vs_path'],
                intel_path=self.cppinfo['intel_path'], 
                intel_vs_version=self.cppinfo['intel_vs_version'], 
                do_print=do_print)

        # b. function list
        funcs = []
        for funcspec in funcspecs:
            if type(funcspec) == str:
                funcs.append( (funcspec,[ct.POINTER(self.par_cpp),ct.POINTER(self.sol_cpp),ct.POINTER(self.sim_cpp)]) )
            else:
                funcs.append(funcspec)
        
        if self.cppinfo['compiler'] == 'vs': 
            funcs.append(('setup_omp',[]))
            use_openmp_with_vs = True

        # c. link
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
        """ call C++ function in linked C++ library
        
        Args:
        
            filename (str): path to .cpp file (no .cpp extension!).
            funcname (str): print if successfull    
        
        """ 
            
        p_par = cpptools.get_struct_pointer(self.par,self.par_cpp)
        p_sol = cpptools.get_struct_pointer(self.sol,self.sol_cpp)
        p_sim = cpptools.get_struct_pointer(self.sim,self.sim_cpp)
        
        funcnow = getattr(self.cppfiles[filename],funcname)
        funcnow(p_par,p_sol,p_sim)
    
    ##############
    ## run file ##
    ##############

    def write_run_file(self,filename='run.py',name='',method='',**kwargs):
        """ write and run a .py file
        
        Args:
        
            filename (str,optional): name of run fil
            name (str,optional): name of instance
            solmethod (str,optional): solutionmethod
            method (str,optional): method to call in run file
            **kwargs: parameters to be updated from baseline
        
        """ 

        modulename = self.__class__.__module__
        modelname = self.__class__.__name__

        if name == '': name = self.name
        if method == '': method = 'solve'

        with open(f'{filename}', 'w') as txtfile:
            
            txtfile.write(f'from {modulename} import {modelname}\n')
            txtfile.write('updpar = dict()\n')
            for key,val in kwargs.items():
                if type(val) is str:
                    txtfile.write(f'updpar["{key}"] = "{val}"\n')
                else:
                    txtfile.write(f'updpar["{key}"] = {val}\n')

            txtfile.write(f'model = {modelname}(name="{name}",**updpar)\n')
            txtfile.write(f'model.{method}()\n')

#######
# jit #
#######
class jit(): 

    def __init__(self,model): 
        
        self.model = model
        self.par = model.par
        self.sol = model.sol
        self.sim = model.sim
      
    def __enter__(self): 

        self.model.update_jit()
        self.model.par = self.model.par_jit
        self.model.sol = self.model.sol_jit
        self.model.sim = self.model.sim_jit

        return self.model
  
    def __exit__(self, exc_type, exc_value, tb):

        if exc_type is not None:
            traceback.print_exception(exc_type, exc_value, tb)

        self.model.par = self.par
        self.model.sol = self.sol
        self.model.sim = self.sim

        del self.model.par_jit
        del self.model.sol_jit
        del self.model.sim_jit

        return True