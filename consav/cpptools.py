# -*- coding: utf-8 -*-
"""cpptools

This module provides a simplified interface for calling C++ functions in Python. 

The compile() function takes a .cpp as input, writes a .bat file, and run it 
to compile a .dll file. The link() function loads the .dll file and defines 
the required functions. The .dll can be unloaded with unlink().

In the sub-section #struct# an interface for passing a Python object to a C++ struct 
is included. The order of fields must be exactly the same in Python and C++. To 
avoid errors a function writing the C++ struct from the Python object is provided.

"""

import os
import time
import zipfile
import urllib.request
import ctypes as ct
import numpy as np

import os, zipfile

def find_vs_path():

    paths = [   
        'C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Auxiliary/Build/',
        'C:/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Auxiliary/Build/'
    ]

    for path in paths:
        if os.path.isdir(path): return path

    raise Exception('no Visual Studio installation found')

def setup_nlopt(vs_path=None,download=True,nloptfolder='cppfuncs',do_print=True):
    """download and setup nlopt

    Args:

        vs_path (str,optional): path to vs compiler

    """

    vs_path = vs_path if not vs_path is None else find_vs_path()

    # a. download
    if download:
        url = 'http://ab-initio.mit.edu/nlopt/nlopt-2.4.2-dll64.zip'
        nloptzip = f'{os.getcwd()}/{nloptfolder}/nlopt-2.4.2-dll64.zip'
        urllib.request.urlretrieve(url, nloptzip)

    # b. unzip
    if download:
        filename = os.path.abspath(f'{os.getcwd()}/{nloptfolder}/nlopt-2.4.2-dll64.zip') 
        with zipfile.ZipFile(filename) as file:
            file.extractall(f'{os.getcwd()}/{nloptfolder}/nlopt-2.4.2-dll64/')

    # c. setup string
    pwd_str = f'cd /d "{os.getcwd()}/{nloptfolder}/nlopt-2.4.2-dll64/"\n'    
    path_str = f'cd /d "{vs_path}"\n'
    version_str = 'call vcvarsall.bat x64\n'
    setup_str = 'lib /def:libnlopt-0.def /machine:x64'
    
    # d. write .bat
    lines = [path_str,version_str,pwd_str,setup_str]
    with open('compile.bat', 'w') as txtfile:
        txtfile.writelines(lines)

    # e. call .bat
    result = os.system('compile.bat')
    if result == 0:
        if do_print: print('nlopt setup done')
    else: 
        raise ValueError('nlopt setup failed')
    os.remove('compile.bat')

    # f. copy
    dst = f'{os.getcwd()}/libnlopt-0.dll'
    if os.path.isfile(dst):
        os.remove(dst)
    os.rename(f'{os.getcwd()}/{nloptfolder}/nlopt-2.4.2-dll64/libnlopt-0.dll',dst)

    # g. remove zip file
    if download: os.remove(nloptzip) 

def compile(filename,compiler='vs',
            vs_path = None,
            intel_path = 'C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2018.5.274/windows/bin/',
            intel_vs_version = 'vs2017',
            nlopt_lib = 'cppfuncs/nlopt-2.4.2-dll64/libnlopt-0.lib',
            additional_cpp = '',
            dllfilename='',do_print=True):      
    """compile cpp file to dll

    Args:

        filename (str): path to .cpp file (no .cpp extensions!)
        compiler (str,optional): compiler choice (vs or intel)
        vs_path (str,optional): path to vs compiler
        intel_path (str,optional): path to intel compiler
        intel_vs_version (str,optional): vs version used by intel compiler
        dllfilename (str,optional): filename of resulting dll file 
        do_print (bool,optional): print if succesfull

    """

    if compiler == 'vs':
        vs_path = vs_path if not vs_path is None else find_vs_path()

    if os.path.isfile(nlopt_lib):
        use_nlopt = True
    else:
        use_nlopt = False

    # a. compile string
    if compiler == 'vs':
        
        pwd_str = 'cd /d "' + os.getcwd() + '"\n'    
        path_str = f'cd /d "{vs_path}"\n'
        version_str = 'call vcvarsall.bat x64\n'
        
        compile_str = f'cl'
        if use_nlopt: compile_str += f' {nlopt_lib}'
        compile_str += f' /LD /EHsc /Ox /openmp {filename}.cpp {additional_cpp}\n'

        lines = [path_str,version_str,pwd_str,compile_str]

    elif compiler == 'intel':
        
        pwd_str = 'cd /d "' + os.getcwd() + '"\n'            
        path_str = f'cd /d "{intel_path}"\n'
        version_str = f'call ipsxe-comp-vars.bat intel64 {intel_vs_version}\n'
        
        compile_str = f'icl'
        if use_nlopt: compile_str += f' {nlopt_lib}'
        compile_str += ' /LD /O3 /arch:CORE-AVX512 /openmp {filename}.cpp {additional_cpp}\n'

        lines = [path_str,version_str,pwd_str,compile_str]
        
    # b. write .bat
    with open('compile.bat', 'w') as txtfile:
        txtfile.writelines(lines)
                               
    # c. compile
    result = os.system('compile.bat')
    if result == 0:
        if do_print:
            print('cpp files compiled')
    else: 
        raise ValueError('cpp files can not be compiled')

    # d. rename dll
    filename_raw = os.path.splitext(os.path.basename(filename))[0]
    if dllfilename != '':
        os.replace(f'{filename_raw}.dll',dllfilename + '.dll')

    # e. clean up
    os.remove('compile.bat')
    if compiler == 'vs':
        os.remove(f'{filename_raw}.obj')
        os.remove(f'{filename_raw}.lib')
        os.remove(f'{filename_raw}.exp')    
    elif compiler == 'intel':
        os.remove(f'{filename_raw}.obj')
        os.remove(f'{filename_raw}.lib')
        os.remove(f'{filename_raw}.exp')

def set_argtypes(cppfile,funcs):
    """ set argument types

    Args:
        cppfile (ctypes.CDLL): c++ library (result of ct.cdll.LoadLibrary('cppfile.dll'))
        funcs (list): list of functions with elements (functionname,[argtype1,argtype2,etc.])
        
    """

    for func in funcs:
        name = func[0]
        argtypes = func[1]        
        funcnow = getattr(cppfile,name)
        funcnow.restype = None
        funcnow.argtypes = argtypes

def link(filename,funcs,use_openmp_with_vs=False,do_print=True,
         nlopt_lib = 'cppfuncs/nlopt-2.4.2-dll64/libnlopt-0.lib'): 
    """ link cpp library
        
    Args:

        filename (str): path to .dll file (no .dll extension!)
        funcs (list): list of functions with elements (functionname,[argtype1,argtype2,etc.])
        use_openmp_with_vs (bool,optional): use openmp with vs as sompiler
        do_print (str,optional): print if successfull

    Return:
    
        cppfile (ctypes.CDLL): C++ library (result of ct.cdll.LoadLibrary('cppfile.dll'))
    
    """

    # a. link
    if os.path.isfile(nlopt_lib):
        nloptfile = ct.cdll.LoadLibrary(f'{os.getcwd()}/libnlopt-0.dll')

    cppfile = ct.cdll.LoadLibrary(f'{os.getcwd()}/{filename}.dll')
    if do_print:
        print('cpp files loaded')
    
    # b. functions
    set_argtypes(cppfile,funcs)
    if use_openmp_with_vs: # needed with openmp
        cppfile.setup_omp() # must exist
        delink(cppfile,filename,do_print=False,do_remove=False)
        cppfile = ct.cdll.LoadLibrary(f'{os.getcwd()}/{filename}.dll')
        set_argtypes(cppfile,funcs)

    if os.path.isfile(nlopt_lib):
        delink(nloptfile,do_print=False,do_remove=False)
        
    return cppfile

def delink(cppfile,filename=None,do_print=True,do_remove=True):
    """ delink cpp library
        
    Args:
    
        cppfile (ctypes.CDLL): C++ library (result of ct.cdll.LoadLibrary('cppfile.dll'))
        filename (str): path to .dll file (no .dll extension!).
        do_print (bool,optional): print if successfull    
        do_remove (bool,optional): remove dll file after delinking

    """

    # a. get handle
    handle = cppfile._handle

    # b. delete linking variable
    del cppfile

    # c. free handle
    ct.windll.kernel32.FreeLibrary.argtypes = [ct.wintypes.HMODULE]
    ct.windll.kernel32.FreeLibrary(handle)
    if do_print:
        print('cpp files delinked')

    # d. remove dll file
    if do_remove:
        os.remove(filename + '.dll')

##########
# struct #
##########

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

            if type(val) is str:

                ctlist.append((key,ct.POINTER(ct.c_char_p)))
                cttxt += f' char *{key};\n' 

            elif type(val) is np.int:
        
                ctlist.append((key,ct.c_long))
                cttxt += f' int {key};\n'
        
            elif type(val) is np.float:
            
                ctlist.append((key,ct.c_double))          
                cttxt += f' double {key};\n'

            elif type(val) is np.bool:
        
                ctlist.append((key,ct.c_bool))
                cttxt += f' bool {key};\n'
            
            else:

                raise ValueError(f'unknown scalar type for {key}, type is {type(val)}')
        
        # b. arrays
        else:
            
            if val.dtype == np.int_:

                ctlist.append((key,ct.POINTER(ct.c_long)))               
                cttxt += f' int *{key};\n'
                     
            elif val.dtype == np.float_:
            
                ctlist.append((key,ct.POINTER(ct.c_double)))
                cttxt += f' double *{key};\n'

            elif val.dtype == np.bool_:
            
                ctlist.append((key,ct.POINTER(ct.c_bool)))
                cttxt += f' bool *{key};\n'

            else:
                
                raise ValueError(f'unknown type for {key}')
    
    return ctlist,cttxt

def setup_struct(pythonobj,structname=None,structfile=None):
    """ create ctypes struct from setup_struct
    
    Args:
    
        pythonobj: e.g. class, SimpleNamespace or namedtuple
        structname (str): name of C++ struct
        strucfile (str): name of of filename for C++ struct

    Write strutfile with Cc++ struct called structname.

    Returns:

        ctstruct (class): ctypes struct type with elements from pythonobj

     """

    # a. get fields
    ctlist, cttxt = get_fields(pythonobj)
    
    # d. write cpp file with struct
    if structname is None: assert structfile is None
    if structfile is None: assert structname is None
    if not structname is None:
        
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

    p_ctstruct = ctstruct()
    
    for field in ctstruct._fields_:
        
        key = field[0]                
        val = getattr(pythonobj,key)

        if isinstance(field[1](),ct.POINTER(ct.c_char_p)):
            pass # not implemented
        
        elif isinstance(field[1](),ct.c_long):
            setattr(p_ctstruct,key,val)
        elif isinstance(field[1](),ct.POINTER(ct.c_long)):
            assert np.issubdtype(val.dtype, np.int_)
            setattr(p_ctstruct,key,np.ctypeslib.as_ctypes(val.ravel()[0:1])) 
            # why [0:1]? hack to avoid bug for arrays with more elements than highest int32

        elif isinstance(field[1](),ct.c_double):
            setattr(p_ctstruct,key,val)            
        elif isinstance(field[1](),ct.POINTER(ct.c_double)):
            assert np.issubdtype(val.dtype, np.float_)
            setattr(p_ctstruct,key,np.ctypeslib.as_ctypes(val.ravel()[0:1]))
            # why [0:1]? hack to avoid bug for arrays with more elements than highest int32
        
        elif isinstance(field[1](),ct.c_bool):
            setattr(p_ctstruct,key,val)
        elif isinstance(field[1](),ct.POINTER(ct.c_bool)):
            assert np.issubdtype(val.dtype, np.bool_)
            setattr(p_ctstruct,key,np.ctypeslib.as_ctypes(val.ravel()[0:1]))
            # why [0:1]? hack to avoid bug for arrays with more elements than highest int32            
        
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