# -*- coding: utf-8 -*-
"""Model

This module provides a class for consumption-saving models with methods for saving and loading
and interfacing with C++. 

"""

import os
import copy
import ctypes as ct
import pickle
from types import SimpleNamespace
import numpy as np
import numba as nb

from . import cpptools

# for save/load
def _filename(model):    
    if model.solmethod is None:
        return f'{model.name}'
    else:
        return f'{model.name}_{model.solmethod}'

def _convert_to_dict(someclass,somelist):

    if somelist is None:
        return {}
    elif len(somelist) > 0 and type(somelist[0]) is str:
        keys = [var for var in somelist]
    else:
        keys = [var[0] for var in somelist]
    values = [getattr(someclass,key) for key in keys]
    return {key:val for key,val in zip(keys,values)}

global_savelist =  ['compiler','not_float_list','parlist','sollist','simlist','vs_path','intel_path','intel_vs_version','savelist']

# main
class ModelClass():
    
    def __init__(self,name=None,
                      solmethod=None,
                      load=False,
                      compiler='vs',**kwargs):
        
        """ defines default attributes """

        if name is None: raise Exception('name must be specified') 

        # a. basic information
        self.name = name # name of parametrization
        self.solmethod = solmethod # solution methods
        self.compiler = compiler # compiler

        # b. basic data structre
        self.par = SimpleNamespace()
        self.sol = SimpleNamespace()
        self.sim = SimpleNamespace()        

        # c. compiler information
        self.vs_path = 'C:/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Auxiliary/Build/'
        self.intel_path = 'C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2018.5.274/windows/bin/'
        self.intel_vs_version = 'vs2017'

        # d. misc
        self.not_float_list = None
        self.savelist = []

        # e. setup (including type inference)
        self.setup()
        for key,val in kwargs.items(): setattr(self.par,key,val)
        self.allocate()
        self.setup_subclasses()

        # f. load
        if load: 
            self.load()
            for key,val in kwargs.items(): setattr(self.par,key,val)

    def setup(self):
        """ set independent variables in par, sol and sim """

        raise Exception('The model must have defined an .setup() method')

    def allocate(self):
        """ set dependent variables in par, sol and sim """

        raise Exception('The model must have defined an .allocate() method')

    def setup_subclasses(self):
        """ setup jitted subclasses par, sol and sim with automatic type inference """

        # a. convert to dictionaries
        par_dict = self.par.__dict__
        sol_dict = self.sol.__dict__
        sim_dict = self.sim.__dict__
        
        # b. infer types
        def check(key,val):

            assert np.isscalar(val) or type(val) is np.ndarray, f'{key} is not scalar or numpy array'
            if self.not_float_list is None: raise Exception('The model must have a par.not_float_list = []')
            if np.isscalar(val):
                assert type(val) is np.float or key in self.not_float_list, f'{key} is {type(val)}, not float, but not on the list'

            return val

        parlist = [(key,nb.typeof(check(key,val))) for key,val in par_dict.items()]
        sollist = [(key,nb.typeof(check(key,val))) for key,val in sol_dict.items()]
        simlist = [(key,nb.typeof(check(key,val))) for key,val in sim_dict.items()]

        # c. create subclasses
        self.par,self.sol,self.sim = self.create_subclasses(parlist,sollist,simlist)

        # d. set values
        for key,val in par_dict.items(): setattr(self.par,key,val)
        for key,val in sol_dict.items(): setattr(self.sol,key,val)
        for key,val in sim_dict.items(): setattr(self.sim,key,val)

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

        @nb.experimental.jitclass(self.parlist) # numba class with variables in parlist
        class ParClass():
            def __init__(self):
                pass

        @nb.experimental.jitclass(self.sollist) # numba jit class with variables in sollist
        class SolClass():
            def __init__(self):
                pass

        @nb.experimental.jitclass(self.simlist) # numba jit class with variables in simlist
        class SimClass():
            def __init__(self):
                pass

        return ParClass(),SolClass(),SimClass()

    def get_par_dict(self):
        """ get dictionary for the par subclass """
        
        return _convert_to_dict(self.par,self.parlist)

    def get_sol_dict(self):
        """ get dictionary for the sol subclass """
        
        return _convert_to_dict(self.sol,self.sollist)

    def get_sim_dict(self):
        """ get dictionary for the sim subclass """
        
        return _convert_to_dict(self.sim,self.simlist)

    def save(self,drop_sol=False,drop_sim=False):
        """ save the model parameters, the solution simulation variables """
        
        if not os.path.exists('data'):
            os.makedirs('data')

        # a. save parameters
        par_dict = _convert_to_dict(self.par,self.parlist)
        with open(f'data/{_filename(self)}_par.p', 'wb') as f:
            pickle.dump(par_dict, f)

        # b. solution
        if drop_sol:
            sol_dict = {}
        else:
            sol_dict = _convert_to_dict(self.sol,self.sollist)
        np.savez(f'data/{_filename(self)}_sol.npz', **sol_dict)
    
        # c. simulation
        if drop_sim:
            sim_dict = {}
        else:        
            sim_dict = _convert_to_dict(self.sim,self.simlist)
        np.savez(f'data/{_filename(self)}_sim.npz', **sim_dict)

        # d. additional
        for x in global_savelist: 
            if not hasattr(self,x): setattr(self,x,None)
            
        additional_dict = _convert_to_dict(self,self.savelist + global_savelist)
        with open(f'data/{_filename(self)}.p', 'wb') as f:
            pickle.dump(additional_dict, f)        

    def copy(self,name=None,**kwargs):
        """ copy the model parameters, the solution simulation variables """
        
        if name is None: name = f'{self.name}_copy' # if not
        other = self.__class__(name=name,solmethod=self.solmethod,compiler=self.compiler)

        # a. save parameters
        par_dict = _convert_to_dict(self.par,self.parlist)
        for key,val in par_dict.items():
            setattr(other.par,key,copy.copy(val))

        # b. solution
        sol_dict = _convert_to_dict(self.sol,self.sollist)
        for key,val in sol_dict.items():
            setattr(other.sol,key,copy.copy(val))
    
        # c. simulation
        sim_dict = _convert_to_dict(self.sim,self.simlist)
        for key,val in sim_dict.items():
            setattr(other.sim,key,copy.copy(val))

        # d. additional
        for x in global_savelist: 
            if not hasattr(self,x): setattr(self,x,None)

        for key in self.savelist + global_savelist:
            setattr(other,key,copy.deepcopy(getattr(self,key)))

        # e. update
        for key,val in kwargs.items():
            setattr(other.par,key,val) # like par.key = val

        return other

    def load(self):
        """ load the model parameters and solution and simulation variables"""

        # a. parameters
        with open(f'data/{_filename(self)}_par.p', 'rb') as f:
            par_dict = pickle.load(f)
        for key,val in par_dict.items():
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
                additional_dict = pickle.load(f)
            for key,val in additional_dict.items():
                setattr(self,key,val)

    def __str__(self):
        """ called when model is printed """ 
        
        def print_list(class_,list_):

            description = ''

            keys = [var[0] for var in list_]
            values = [getattr(class_,key) for key in keys]

            nbytes = 0
            for var,val in zip(list_,values):    
                if np.isscalar(val) and not var[1] == nb.boolean:
                    description += f' {var[0]} = {val} [{var[1]}]\n'
                elif var[1] == nb.boolean:
                    if val:
                        description += f' {var[0]} = True\n'
                    else:
                        description += f' {var[0]} = False\n'
                elif type(var[1]) is nb.types.npytypes.Array:
                    description += f' {var[0]} = {var[1]} with shape = {val.shape}\n'            
                    nbytes += val.nbytes
                else:                
                    description += f' {var[0]} = ?\n'

            description += f'memory, gb: {nbytes/(10**9):.1f}\n' 
            return description

        description = f'Modelclass: {self.__class__.__name__}\n\n'

        description += 'Parameters:\n'
        description += print_list(self.par,self.parlist)
        description += '\n'

        description += 'Solution:\n'
        description += print_list(self.sol,self.sollist)
        description += '\n'

        description += 'Simulation:\n'
        description += print_list(self.sim,self.simlist)

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