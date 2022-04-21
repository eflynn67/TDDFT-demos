import numpy as np
from numba import jit
from init import *
import densities
'''
@jit(nopython=True,parallel=True)
def yuk(rhoArr,r):
    
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

    
    rp = grid.copy()
    if r == grid[0] or r == grid[-1]:
        Vp = 0
    else:
        Vp = .5*(rhoArr[0]*np.exp(-abs(rp[0]-r)/a)/(abs(r-rp[0]))) + \
        .5*(rhoArr[-1]*np.exp(-abs(rp[-1]-r)/a)/(abs(r-rp[-1])))
    for k in np.arange(1,len(grid)-1,1): 
        if rp[k] - r < 10**-25:
            Vp += 0.0
        else:
            Vp += rhoArr[k+1]*np.exp(-abs(rp[k+1]-r)/a)/(abs(r-rp[k+1]))
    return Vp*aV0
'''
@jit(nopython=True,parallel=True)
def yukArr(rhoArr):
    Vyuk = np.zeros(len(grid))
    for i,r in enumerate(grid):
        I = 0.0 
        for j in range(i):
            rp = grid[j]
            rs = max(r,rp)
            I += aV0*(rhoArr[j]*rp/(rs*r))*np.exp((r**2 + rp**2)/aV0)*np.sinh(2*r*rp/aV0)*step_size
        Vyuk[i] = I
    return Vyuk*aV0

@jit(nopython=True,parallel=True)
def coulombArr(rhoArr):
    Vc = np.zeros(len(grid))
    for i,r in enumerate(grid):
        I = 0.0 
        for j in range(i):
            rp = grid[j]
            rs = max(r,rp)
            I += (rhoArr[j]*(rp**2)/rs)*step_size
        Vc[i] = I
    return Vc*4*np.pi*e2
        
'''
@jit(nopython=True,parallel=True)
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

@jit(nopython=True)
def coulombArr(rho):


    Parameters
    ----------
    rhoArr : nd Array
        array representation of the function rho.

    Returns
    -------
    Arr : TYPE
        DESCRIPTION.


    Arr = np.zeros(len(grid))
    for i,r in enumerate(grid): 
        Arr[i] = coulomb(rho,r)
    return Arr
'''
'''
@jit(nopython=True)
def yukArr(rhoArr):


    Parameters
    ----------
    rhoArr : nd Array
        array representation of the function rho.

    Returns
    -------
    Arr : TYPE
        DESCRIPTION.

   Arr = np.zeros(len(grid))
    for i,r in enumerate(grid): 
        Arr[i] = yuk(rhoArr,r)
    return Arr
'''
def externalV(r,t):
    
    return None

def harm(x,alpha):
    '''
    1-d harmonic Oscillator potential. Used for testing

    Parameters
    ----------
    x : float or nd array
        position.
    alpha : float
        oscillator length parameter.

    Returns
    -------
    float or ndarray
        value of potential evaluated at x.

    '''
    return alpha*x**2