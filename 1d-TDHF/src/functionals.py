import numpy as np
from init import *
from fields import *
def h_BKN(rhoArr,r):
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
    h = .75*t0*rho_array + 0.1875*t3*rhoArr**2 + yuk(rhoArr,r)
    
    return h

def h_Skyrme():
    return "No Skyrme interaction yet."

