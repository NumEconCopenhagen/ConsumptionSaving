# -*- coding: utf-8 -*-
"""Model

Provides a class for consumption-saving models with methods for saving and loading
and interfacing with numba jitted functions and C++.

"""

import os
from copy import deepcopy
import pickle
from types import SimpleNamespace
from collections import namedtuple

import numpy as np

from .cpptools import link_to_cpp

# main
class ModelClass():
    
    def __init__(self,name=None,
            load=False,from_dict=None,
            setup_infrastruct=True,**kwargs):
        """ defines default attributes """

        if load: assert from_dict is None, 'dictionary should be be specified when loading'
        assert not hasattr(self,'cpp'), 'the model can not have .cpp method'

        # a. name
        if name is None: raise Exception('name must be specified')
        self.name = name

        # list of internal of attributes (used when saving)
        self.internal_attrs = [
            'savefolder','namespaces','not_floats','other_attrs',
            'cpp_filename','cpp_options','cpp_structsmap']

        # b. new or load
        self.savefolder = 'saved'

        if (not load) and (from_dict is None): # new
            
            # i. empty containers
            self.namespaces = []
            self.not_floats = []
            self.other_attrs = []
            self.cpp = SimpleNamespace() # overwritten later, helps linter (not internal attribute)
            self.cpp_filename = None
            self.cpp_options = {}
            self.cpp_structsmap = {}

            # ii. settings
            assert hasattr(self,'settings'), 'The model must have defined an .settings() method'
            self.settings()

            # default to none
            for attr in self.other_attrs:
                if not hasattr(self,attr): setattr(self,attr,None)

            self.par = SimpleNamespace() # -> helps linter
            self.sol = SimpleNamespace() # -> helps linter
            self.sim = SimpleNamespace() # -> helps linter

            for ns in self.namespaces:
                setattr(self,ns,SimpleNamespace())

            self.namespaces = list(set(self.namespaces + ['par','sol','sim']))

            # iii setup
            assert hasattr(self,'setup'), 'The model must have defined an .setup() method'
            self.setup()

            # iv. update
            self.update(kwargs)
            
            # vi. allocate
            assert hasattr(self,'allocate'), 'The model must have defined an .allocate() method'
            self.allocate()

        elif load: # load
            
            self.settings()
            self.load()
            self.update(kwargs)

        else:
            
            self.from_dict(from_dict)

        # c. infrastructure
        if setup_infrastruct:
            self.setup_infrastructure()
    
    def update(self,upd_dict):
        """ update """

        for nskey,values in upd_dict.items():
            assert nskey in self.namespaces, f'{nskey} is not a namespace'
            assert type(values) is dict, f'{nskey} must be dict'
            for key,value in values.items():
                assert hasattr(getattr(self,nskey),key), f'{key} is not in {nskey}' 
                setattr(getattr(self,nskey),key,value)        

    ####################
    ## infrastructure ##
    ####################

    def setup_infrastructure(self):
        """ setup infrastructure to call numba jit functions """
        
        # a. convert to dictionaries
        ns_dict = {}
        for ns in self.namespaces:
            ns_dict[ns] = getattr(self,ns).__dict__

        # b. type check
        def check(key,val):

            _scalar_or_ndarray = np.isscalar(val) or type(val) is np.ndarray or type(val) is np.memmap
            assert _scalar_or_ndarray, f'{key} is not scalar or numpy arraym but {type(val)}'
            
            _non_float = not np.isscalar(val) or type(val) is str or type(val) is np.float or type(val) is np.float64 or key in self.not_floats
            assert _non_float, f'{key} is {type(val)}, not float, but not on the list'

        for ns in self.namespaces:
            for key,val in ns_dict[ns].items():
                check(key,val)

        # c. namedtuple (definitions)
        self.ns_jit_def = {}
        for ns in self.namespaces:
            self.ns_jit_def[ns] = namedtuple(f'{ns.capitalize()}Class',[key for key in ns_dict[ns].keys()])

    def update_jit(self):
        """ update values and references in par_jit, sol_jit, sim_jit """

        self.ns_jit = {}
        for ns in self.namespaces:
            self.ns_jit[ns] = self.ns_jit_def[ns](**getattr(self,ns).__dict__)

    ####################
    ## save-copy-load ##
    ####################
    
    def all_attrs(self):
        """ return all attributes """

        return self.namespaces + self.other_attrs + self.internal_attrs

    def as_dict(self,drop=[]):
        """ return a dict version of the model """
        
        model_dict = {}
        for attr in self.all_attrs():
            if not attr in drop: model_dict[attr] = getattr(self,attr)

        return model_dict

    def from_dict(self,model_dict,do_copy=False):
        """ construct the model from a dict version of the model """

        self.namespaces = model_dict['namespaces']
        self.other_attrs = model_dict['other_attrs']
        for attr in self.all_attrs():
            if attr in model_dict:
                if do_copy:
                    setattr(self,attr,deepcopy(model_dict[attr]))
                else:
                    setattr(self,attr,model_dict[attr])
            else:
                setattr(self,attr,None)

    def save(self,drop=[]):
        """ save the model """

        # a. ensure path        
        if not os.path.exists(self.savefolder):
            os.makedirs(self.savefolder)

        # b. create model dict
        model_dict = self.as_dict(drop=drop)

        # b. save to disc
        with open(f'{self.savefolder}/{self.name}.p', 'wb') as f:
            pickle.dump(model_dict, f)

    def load(self):
        """ load the model """

        # a. load
        with open(f'{self.savefolder}/{self.name}.p', 'rb') as f:
            model_dict = pickle.load(f)

        self.cpp = SimpleNamespace()
        
        # b. construct
        self.from_dict(model_dict)

    def copy(self,name=None,**kwargs):
        """ copy the model """
        
        # a. name
        if name is None: name = f'{self.name}_copy'
        
        # b. model dict
        model_dict = self.as_dict()

        # c. initialize
        other = self.__class__(name=name)

        # d. fill
        other.from_dict(model_dict,do_copy=True)
        other.update(kwargs)
        other.ns_jit_def = self.ns_jit_def
        if not type(self.cpp) is SimpleNamespace:
            other.link_to_cpp(force_compile=False)

        return other

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

            description += f' memory, gb: {nbytes/(10**9):.1f}\n' 
            return description

        description = f'Modelclass: {self.__class__.__name__}\n'
        description += f'Name: {self.name}\n\n'

        description += 'namespaces: ' + str(self.namespaces) + '\n'
        description += 'other_attrs: ' + str(self.other_attrs) + '\n'
        description += 'savefolder: ' + str(self.savefolder) + '\n'
        description += 'not_floats: ' + str(self.not_floats) + '\n'

        for ns in self.namespaces:
            description += '\n'
            description += f'{ns}:\n'
            description += print_items(getattr(self,ns))

        return description 

    #######################
    ## interact with cpp ##
    #######################

    def link_to_cpp(self,force_compile=True,do_print=False):
        """ link to C++ file """

        # a. unpack
        filename = self.cpp_filename
        options = self.cpp_options
        
        def structname(ns):

            if ns in self.cpp_structsmap:
                return self.cpp_structsmap[ns]
            else:
                return f'{ns}_struct'
              
        structsmap = {structname(ns):getattr(self,ns) for ns in self.namespaces}

        # b. link to C++
        self.cpp = link_to_cpp(filename,force_compile=force_compile,options=options,
            structsmap=structsmap,do_print=do_print)

    ############
    # clean-up #
    ############

    def __del__(self):

        if hasattr(self,'cpp') and (not type(self.cpp) is SimpleNamespace):
            self.cpp.delink()            