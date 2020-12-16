# -*- coding: utf-8 -*-
""" misc

Misc. function.
"""

import time

def elapsed(t0,t1=None):
    """ time elapsed since t0 with in nice format

    Args:

        t0 (double): start time
        t1 (double,optional): end time (else now)

    Return:

        (str): elapsed time in nice format

    """ 

    if t1 is None:
        secs = time.time()-t0
    else:
        secs = t1-t0

    days = secs//(60*60*24)
    secs -= 60*60*24*days

    hours = secs//(60*60)
    secs -= 60*60*hours

    mins = secs//(60)
    secs -= 60*mins
   
    text = ''
    if days > 0: text += f'{days} days '
    if hours > 0: text += f'{hours} hours '
    if mins > 0: text += f'{mins} mins '

    if days > 0 or hours > 0:
        pass
    elif mins > 0:
        text += f'{secs:.0f} secs '
    else:
        text = f'{secs:.1f} secs '

    return text[:-1]
