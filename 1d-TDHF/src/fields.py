import numpy as np
from numba import jit
from init import *
import densities

@jit(nopython=True)
def yuk(rhoArr,r):
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
    rp = grid.copy()
    if r == grid[0] or r == grid[-1]:
        Vp = 0
    else:
        Vp = .5*(rhoArr[0]*(rp[0]**2)*np.exp(-abs(rp[0]-r)/a)/(abs(r-rp[0]))) + \
        .5*(rhoArr[-1]*(rp[-1]**2)*np.exp(-abs(rp[-1]-r)/a)/(abs(r-rp[-1])))
    for k in np.arange(1,len(grid)-1,1): 
        if rp[k] - r < 10**-25:
            Vp += 0.0
        else:
            Vp += rhoArr[k+1]*(rp[k+1]**2)*np.exp(-abs(rp[k+1]-r)/a)/(abs(r-rp[k+1]))
    return Vp*aV0
@jit(nopython=True)
def coulomb(rhoArr,r):
    rp = grid.copy()
    if r == grid[0] or r == grid[-1]:
        Vp = 0
    else:
        Vp = .5*(rhoArr[0]*(rp[0]**2)/(abs(r-rp[0]))) + .5*(rhoArr[-1]*(rp[-1]**2)/(abs(r-rp[-1])))
    for k in np.arange(1,len(grid)-1,1):
        if rp[k] - r < 10**-25:
            Vp += 0.0
        else:
            Vp += rhoArr[k+1]*(rp[k+1]**2)/(abs(r-rp[k+1]))
    return Vp*e2
def externalV():
    return
