import numpy as np
from scipy.integrate import quad
from init import *
import densities
def yuk(psi_array,r):
    '''
    calculate yukawa potential at every grid point in r.

    Parameters
    ----------
    rho : TYPE
        DESCRIPTION.
    r : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    Vp = np.zeros(len(grid))
    for k in range(len(grid)):    
        Vp[k] = (rho[k]*np.exp(-1*abs(rp[k] - r)/a)/(abs(r-rp[k]))*rp[k]**2 + \\
                 rho[k-1]*np.exp(-1*abs(rp[k-1] - r)/a)/(abs(r-rp[k-1]))*rp[k-1]**2)/2 

    return V
def coulomb():
    return
def externalV():
    return
