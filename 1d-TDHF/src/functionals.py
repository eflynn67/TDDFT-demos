import numpy as np
from init import *
from fields import *
def h_BKN(rho):
    '''
    

    Parameters
    ----------
    rhoArr : nd array
        DESCRIPTION.
    r : float
        DESCRIPTION.

    Returns
    -------
    returns the functional value of h at a particular point r.

    '''
    h = .75*t1*rho + 0.1875*t3*rho**2 + yukArr(rho)
    
    return h

def h_Skyrme():
    return "No Skyrme interaction yet."

