# -*- coding: utf-8 -*-
"""runtools

This module provides misc run tools.
"""


def write_numba_config(disable=0,threads=1,threading_layer='omp'):
    """ write .numba_config.yaml file
        
    Args:
    
        disable_jit (int): numba disabled if = 1.
        num_threads (int): number of threads used in numba
        threading_layers (str): type of parallization, 'omp' or 'tbb'
    
    """

    with open(f'.numba_config.yaml', 'w') as txtfile:
        txtfile.write(f'disable_jit: {disable}\n')
        txtfile.write(f'num_threads: {threads}\n')
        txtfile.write(f'threading_layer: {threading_layer}\n')