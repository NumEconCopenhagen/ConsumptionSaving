# -*- coding: utf-8 -*-
""" cppcompile

Functions for compiling C++ files to use in Python.
"""

import os
import zipfile
import urllib.request

############
# auxilary #
############

def find_vs_path():
    """ find path to visual studio """

    paths = [   
        'C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Auxiliary/Build/',
        'C:/Program Files (x86)/Microsoft Visual Studio/2017/Community/VC/Auxiliary/Build/'
    ]

    for path in paths:
        if os.path.isdir(path): return path

    raise Exception('no Visual Studio installation found')

def write_setup_omp():
    """ write C++ file to setup OpenMP with Visual Studio """

    assert not os.path.isfile('setup_omp.cpp'), f'setup_omp.cpp already exists'

    with open(f'setup_omp.cpp', 'w') as cppfile:
        
        cppfile.write('#include <windows.h>\n') 
        cppfile.write('#define EXPORT extern "C" __declspec(dllexport)\n')
        cppfile.write('EXPORT void setup_omp(){\n')
        cppfile.write('SetEnvironmentVariable("OMP_WAIT_POLICY", "passive");\n')
        cppfile.write('}\n')

def setup_nlopt(vs_path=None,download=True,unzip=False,folder='cppfuncs/',force_copy=True,do_print=False):
    """download and setup nlopt

    Args:

        vs_path (str,optional): path to vs compiler
        download (bool,optional): download nlopt 2.4.2
        unzip (bool,optional): unzip even if not downloaded
        folder (str,optional): folder to put nlopt to
        force_copy (bool,optional): force overwrite of .dll
        do_print (bool,optional): print progress

    """

    vs_path = vs_path if not vs_path is None else find_vs_path()
    dst = f'{os.getcwd()}/libnlopt-0.dll'
    lib = f'{os.getcwd()}/{folder}nlopt-2.4.2-dll64/libnlopt-0.lib'

    if os.path.isfile(dst) and os.path.isfile(lib):
        if do_print: print('nlopt already installed')
        return

    # a. download
    zipfilename = os.path.abspath(f'{os.getcwd()}/{folder}nlopt-2.4.2-dll64.zip')
    if download:
        url = 'http://ab-initio.mit.edu/nlopt/nlopt-2.4.2-dll64.zip'
        urllib.request.urlretrieve(url,zipfilename)

    # b. unzip
    if download or unzip:
        with zipfile.ZipFile(zipfilename) as file:
            file.extractall(f'{os.getcwd()}/{folder}nlopt-2.4.2-dll64/')

    # c. setup string
    pwd_str = f'cd /d "{os.getcwd()}/{folder}nlopt-2.4.2-dll64/"\n'    
    path_str = f'cd /d "{vs_path}"\n'
    version_str = 'call vcvarsall.bat x64\n'
    setup_str = 'lib /def:libnlopt-0.def /machine:x64'
    
    # d. write .bat
    lines = [path_str,version_str,pwd_str,setup_str]
    with open('compile_nlopt.bat', 'w') as txtfile:
        txtfile.writelines(lines)

    # e. call .bat
    result = os.system('compile_nlopt.bat')
    if result == 0:
        if do_print: print('nlopt successfully installed')
    else: 
        raise ValueError('nlopt installation failed')

    os.remove('compile_nlopt.bat')

    # f. copy
    if os.path.isfile(dst) and force_copy: os.remove(dst)
    if not os.path.isfile(dst) or force_copy:
        os.rename(f'{os.getcwd()}/{folder}nlopt-2.4.2-dll64/libnlopt-0.dll',dst)

    # g. remove zip file
    if download or unzip: os.remove(zipfilename) 

def setup_tasmanian(vs_path=None,download=True,unzip=False,folder='cppfuncs/',force_copy=True,do_print=False):
    """download and setup nlopt

    Args:

        vs_path (str,optional): path to vs compiler
        download (bool,optional): download Tasmanian 5.1
        unzip (bool,optional): unzip even if not downloaded
        folder (str,optional): folder to put Tasmanian to
        force_copy (bool,optional): force overwrite of .dll
        do_print (bool,optional): print progress

    """

    vs_path = vs_path if not vs_path is None else find_vs_path()
    dst = f'{os.getcwd()}/libtasmaniansparsegrid.dll'
    lib = f'{os.getcwd()}/{folder}TASMANIAN-5.1/libtasmaniansparsegrid.lib'

    if os.path.isfile(dst) and os.path.isfile(lib):
        if do_print: print('Tasmanian already installed')
        return

    # a. download
    zipfilename = os.path.abspath(f'{os.getcwd()}/{folder}TASMANIAN-5.1.zip') 
    if download:
        url = 'https://github.com/ORNL/TASMANIAN/archive/v5.1.zip'
        urllib.request.urlretrieve(url,zipfilename)
        
    # b. unzip
    if download or unzip:
        with zipfile.ZipFile(zipfilename) as file:
            file.extractall(f'{os.getcwd()}/{folder}')        

    # c. setup string
    pwd_str = f'cd /d "{os.getcwd()}/{folder}TASMANIAN-5.1/"\n'    
    path_str = f'cd /d "{vs_path}"\n'
    version_str = 'call vcvarsall.bat x64\n'
    setup_str = 'call WindowsMake.bat'

    # d. write .bat
    lines = [path_str,version_str,pwd_str,setup_str]
    with open('compile_tasmanian.bat', 'w') as txtfile:
        txtfile.writelines(lines)

    # e. call .bat
    result = os.system('compile_tasmanian.bat')
    if result == 0:
        if do_print: print('tasmanian successfully installed')
    else: 
        raise ValueError('tasmanian installation failed')

    os.remove('compile_tasmanian.bat')

    # f. copy
    if os.path.isfile(dst) and force_copy: os.remove(dst)
    if not os.path.isfile(dst) or force_copy:
        os.rename(f'{os.getcwd()}/{folder}Tasmanian-5.1/libtasmaniansparsegrid.dll',dst)

    # g. remove zip file
    if download or unzip: os.remove(zipfilename)

def setup_alglib(download=True,unzip=False,folder='cppfuncs/',do_print=False):
    """download and setup alglib

    Args:

        download (bool,optional): download Tasmanian 5.1
        unzip (bool,optional): unzip even if not downloaded
        folder (str,optional): folder to put Tasmanian to
        do_print (bool,optional): print progress

    """

    if os.path.isdir(f'{os.getcwd()}/{folder}alglib-3.17.0'):
        if do_print: print('alglib already installed')
        return

    # a. download
    zipfilename = os.path.abspath(f'{os.getcwd()}/{folder}alglib-3.17.0.cpp.gpl.zip')
    if download:
        url = 'https://www.alglib.net/translator/re/alglib-3.17.0.cpp.gpl.zip'
        urllib.request.urlretrieve(url,zipfilename)
        
    # b. unzip
    if download or unzip:
        with zipfile.ZipFile(zipfilename) as file:
            file.extractall(f'{os.getcwd()}/{folder}alglib-3.17.0')        

    # c. remove zip file
    if download or unzip: os.remove(zipfilename)

    if do_print: print('alglib succesfully installed')

def add_macros(macros):
    """ add macros to compile string
        
    Args:

        macros (dict/list): preprocessor macros

    Returns:

        compile_str (str): macro string 

    """

    compile_str = ''

    if type(macros) is dict:
    
        for k,v in macros.items():
            if v is None:
                compile_str += f' /D{k}'
            else:
                compile_str += f' /D{k}={v}'
    
    elif type(macros) is list:

        for k in macros: 
            compile_str += f' /D{k}'

    return compile_str

###########
# compile #
###########

def set_default_options(options):
    """

    Args:

        options (dict): compiler options with 

            compiler (str): compiler choice (vs or intel)
            vs_path (str): path to vs compiler (if None then newest version found is used, e.g. C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Auxiliary/Build/)
            intel_path (str): path to intel compiler
            intel_vs_version (str): vs version used by intel compiler
            nlopt_lib (str): path to NLopt library (included if exists, default is cppfuncs/nlopt-2.4.2-dll64/libnlopt-0.lib)
            tasmanian_lib (str): path to Tasmanian library (included if exists, default is cppfuncs/TASMANIAN-5.1/libtasmaniansparsegrid.lib')
            additional_cpp (str): additional cpp files to include ('' default)
            dllfilename (str): filename of resulting dll file (if None (default) based on .cpp file)
            macros (dict/list): preprocessor macros

    """

    options.setdefault('compiler','vs')
    
    if options['compiler'] == 'vs':
        options.setdefault('vs_path',find_vs_path())
    else:
        options.setdefault('vs_path',None)

    options.setdefault('intel_path','C:/Program Files (x86)/IntelSWTools/compilers_and_libraries_2018.5.274/windows/bin/')
    options.setdefault('intel_vs_version','vs2017')
    options.setdefault('nlopt_lib','cppfuncs/nlopt-2.4.2-dll64/libnlopt-0.lib')
    options.setdefault('tasmanian_lib','cppfuncs/TASMANIAN-5.1/libtasmaniansparsegrid.lib')
    options.setdefault('additional_cpp','')
    options.setdefault('macros',None)
    options.setdefault('dllfilename',None)

    assert options['compiler'] in ['vs','intel'], f'unknown compiler {options["compiler"]}'

def compile(filename,options={},do_print=False):      
    """compile cpp file to dll

    Args:

        filename (str): path to .cpp file
        options (dict,optional): compiler options
        do_print (bool,optional): print if succesfull

    """
    
    set_default_options(options)

    if options['compiler'] == 'vs' and options['vs_path'] is None:
        options['vs_path'] = find_vs_path()

    compiler = options['compiler']
    vs_path = options['vs_path']
    intel_path = options['intel_path']
    intel_vs_version = options['intel_vs_version']
    nlopt_lib = options['nlopt_lib']
    tasmanian_lib = options['tasmanian_lib']
    additional_cpp = options['additional_cpp']
    macros = options['macros']
    dllfilename = options['dllfilename']
    
    # a. check filename
    assert os.path.isfile(filename), f'"{filename}" does not exist'

    basename = os.path.basename(filename)
    dirname = os.path.dirname(filename)

    # b. prepare visual studio
    if compiler == 'vs':
        write_setup_omp()

    # c. check for nlopt
    if os.path.isfile(nlopt_lib):
        use_nlopt = True
    else:
        use_nlopt = False

    if os.path.isfile(tasmanian_lib):
        use_tasmanian = True
    else:
        use_tasmanian = False

    # d. compile string
    pwd_str = 'cd /d "' + os.getcwd() + '"\n'    
    
    if compiler == 'vs':
        
        path_str = f'cd /d "{vs_path}"\n'
        version_str = 'call vcvarsall.bat x64\n'
        
        compile_str = f'cl'
        if use_nlopt: compile_str += f' {nlopt_lib}'
        if use_tasmanian: compile_str += f' {tasmanian_lib}'

        compile_str += f' /LD /EHsc /Ox /openmp {filename} setup_omp.cpp {additional_cpp} {add_macros(macros)}\n' 

        lines = [path_str,version_str,pwd_str,compile_str]

    elif compiler == 'intel':
        
        path_str = f'cd /d "{intel_path}"\n'
        version_str = f'call ipsxe-comp-vars.bat intel64 {intel_vs_version}\n'
        
        compile_str = f'icl'
        if use_nlopt: compile_str += f' {nlopt_lib}'
        if use_tasmanian: compile_str += f' {tasmanian_lib}'

        compile_str += f' /LD /EHsc /O3 /arch:CORE-AVX512 /openmp {filename} {additional_cpp} {add_macros(macros)}\n' 

    lines = [path_str,version_str,pwd_str,compile_str]
        
    # e. write .bat
    with open('compile.bat', 'w') as txtfile:
        txtfile.writelines(lines)
                               
    # f. compile
    result = os.system('compile.bat')
    if compiler == 'vs': 
        os.remove(f'setup_omp.cpp')
        os.remove(f'setup_omp.obj')

    if result == 0:
        if do_print: print('C++ files compiled')
    else: 
        raise Exception('C++ files can not be compiled')

    # g. rename dll
    filename_raw = os.path.splitext(basename)[0]
    if dllfilename is None: 
        dllfilename = f'{filename_raw}.dll'  
    else:
        os.replace(f'{filename_raw}.dll',dllfilename)

    # h. clean up
    os.remove('compile.bat')
    if compiler == 'vs':
        os.remove(f'{filename_raw}.obj')
        os.remove(f'{filename_raw}.lib')
        os.remove(f'{filename_raw}.exp')    
    elif compiler == 'intel':
        os.remove(f'{filename_raw}.obj')
        os.remove(f'{filename_raw}.lib')
        os.remove(f'{filename_raw}.exp')