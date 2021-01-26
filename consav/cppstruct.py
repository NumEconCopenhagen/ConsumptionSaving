# -*- coding: utf-8 -*-
""" cppstruct

Functions for using dict-like objects in Python in C++.
"""

import ctypes as ct
import numpy as np

def get_fields(pythonobj):
    """ construct ctypes list of fields from pythonobj
    
    Args:
    
        pythonobj: e.g. class, SimpleNamespace or namedtuple

    Returns:
    
        ctlist (list): list of fields with elements (name,ctypes type) 
        cttxt (str): string with content of C++ struct

    """

    ctlist = []
    cttxt = ''

    for key,val in pythonobj.__dict__.items():

        # a. scalars
        if np.isscalar(val):

            if type(val) in [np.int,np.int_]:
        
                ctlist.append((key,ct.c_long))
                cttxt += f' int {key};\n'
        
            elif type(val) in [np.float,np.float_]:
            
                ctlist.append((key,ct.c_double))          
                cttxt += f' double {key};\n'

            elif type(val) is np.bool:
        
                ctlist.append((key,ct.c_bool))
                cttxt += f' bool {key};\n'
            
            elif type(val) is str:

                ctlist.append((key,ct.c_char_p))
                cttxt += f' char *{key};\n' 

            else:

                raise ValueError(f'unknown scalar type for {key}, type is {type(val)}')
        
        # b. arrays
        else:

            assert hasattr(val,'dtype'), f'{key} is neither scalar nor np.array'
            
            if val.dtype == np.int_:

                ctlist.append((key,ct.POINTER(ct.c_long)))               
                cttxt += f' int* {key};\n'
                     
            elif val.dtype == np.float_:
            
                ctlist.append((key,ct.POINTER(ct.c_double)))
                cttxt += f' double* {key};\n'

            elif val.dtype == np.bool_:
            
                ctlist.append((key,ct.POINTER(ct.c_bool)))
                cttxt += f' bool* {key};\n'

            else:
                
                raise ValueError(f'unknown array type for {key}, dtype is {val.dtype}')
    
    return ctlist,cttxt

def setup_struct(pythonobj,structname,structfile,do_print=False):
    """ create ctypes struct from setup_struct
    
    Args:
    
        pythonobj: e.g. class, SimpleNamespace or namedtuple (with __dict__ method)
        structname (str): name of C++ struct
        strucfile (str): name of of filename for C++ struct
        do_print (bool): print contents of structs

    Write strutfile with Cc++ struct called structname.

    Returns:

        ctstruct (class): ctypes struct type with elements from pythonobj

     """

    assert hasattr(pythonobj,'__dict__'), f'python object does not have a dictionary interface'
    assert type(structname) is str
    assert type(structfile) is str

    # a. get fields
    ctlist, cttxt = get_fields(pythonobj)

    if do_print: print(cttxt)
    
    # d. write cpp file with struct
    with open(structfile, 'w') as cppfile:

        cppfile.write(f'typedef struct {structname}\n') 
        cppfile.write('{\n')
        cppfile.write(cttxt)
        cppfile.write('}')
        cppfile.write(f' {structname};\n\n')

    # c. ctypes struct
    class ctstruct(ct.Structure):
        _fields_ = ctlist

    return ctstruct

def get_pointers(pythonobj,ctstruct):
    """ construct ctypes struct class with pointers from pythonobj
    
    Args:
    
        pythonobj: e.g. class, SimpleNamespace or namedtuple
        ctstruct (class): ctypes struct type

    Returns:
    
        p_ctstruct (class): ctypes struct with pointers to pythonobj
        
    """

    # a. setup
    p_ctstruct = ctstruct()
    
    # b. add fields
    for field in ctstruct._fields_:
        
        key = field[0]                
        val = getattr(pythonobj,key)
       
        if isinstance(field[1](),ct.c_long):
            try:
                setattr(p_ctstruct,key,val)
            except:
                raise Exception(f'{key} is not an integer')

        elif isinstance(field[1](),ct.POINTER(ct.c_long)):
            assert np.issubdtype(val.dtype, np.int_), f'field = {field}'
            setattr(p_ctstruct,key,np.ctypeslib.as_ctypes(val.ravel()[0:1])) 
            # why [0:1]? hack to avoid bug for arrays with more elements than highest int32

        elif isinstance(field[1](),ct.c_double):
            try:
                setattr(p_ctstruct,key,val)   
            except:
                raise Exception(f'{key} is not a floating point')

        elif isinstance(field[1](),ct.POINTER(ct.c_double)):
            assert np.issubdtype(val.dtype, np.float_), f'field = {field}'
            setattr(p_ctstruct,key,np.ctypeslib.as_ctypes(val.ravel()[0:1]))
            # why [0:1]? hack to avoid bug for arrays with more elements than highest int32
        
        elif isinstance(field[1](),ct.c_bool):
            setattr(p_ctstruct,key,val)

        elif isinstance(field[1](),ct.POINTER(ct.c_bool)):
            assert np.issubdtype(val.dtype, np.bool_), f'field = {field}'
            setattr(p_ctstruct,key,np.ctypeslib.as_ctypes(val.ravel()[0:1]))
            # why [0:1]? hack to avoid bug for arrays with more elements than highest int32            
        
        elif isinstance(field[1](),ct.c_char_p):
            assert type(val) is str, f'field = {field}'
            setattr(p_ctstruct,key,val.encode())

        else:

            raise ValueError(f'no such type, variable {key}')
    
    return p_ctstruct

def get_struct_pointer(pythonobj,ctstruct):
    """ return pointer to ctypes struct
    
    Args:
    
        pythonobj: e.g. class, SimpleNamespace or namedtuple
        ctstruct (class): ctypes struct type

    Returns:
    
        pointer: pointer to ctypes struct with pointers to pythonobj
    
    """

    return ct.byref(get_pointers(pythonobj,ctstruct))