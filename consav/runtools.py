# -*- coding: utf-8 -*-
"""runtools

This module provides misc run tools.
"""


def write_numba_config(disable=0,threads=1,threading_layer='omp',parfor_max_tuple_size=None):
    """ write .numba_config.yaml file
        
    Args:
    
        disable_jit (int,optional): numba disabled if = 1.
        num_threads (int,optional): number of threads used in numba
        threading_layers (str,optional): type of parallization, 'omp' or 'tbb'
        parfor_max_tuple_size (int,optional): length of tuple allowed in prange loop
    
    """

    with open(f'.numba_config.yaml', 'w') as txtfile:
        txtfile.write(f'disable_jit: {disable}\n')
        txtfile.write(f'num_threads: {threads}\n')
        txtfile.write(f'threading_layer: {threading_layer}\n')
        if not parfor_max_tuple_size is None: txtfile.write(f'parfor_max_tuple_size: {parfor_max_tuple_size}\n')