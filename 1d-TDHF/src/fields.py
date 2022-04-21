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
            I += rhoArr[j]*(np.exp(-abs(r-rp)/a)/abs(r-rp))*step_size
        Vyuk[i] = I
    return Vyuk*aV0

@jit(nopython=True,parallel=True)
def yukArr2(rhoArr):
    Vyuk = np.zeros(len(grid))
    for i,r in enumerate(grid):
        I = 0.0
        for thetap in np.arange(0,np.pi,step_size):
            for j in range(i):
                rp = grid[j]
                num = np.exp(-np.sqrt(r**2 + rp**2 - 2*r*rp*np.cos(thetap))/a)*np.sin(thetap)*rp**2
                denom = np.sqrt(r**2 + rp**2 - 2*r*rp*np.cos(thetap))
                I += rhoArr[j]*(num/denom)*step_size
        Vyuk[i] = I
    return Vyuk*aV0
@jit(nopython=True,parallel=True)
def coulombArr(rhoArr):
    Vc12 = np.zeros(len(grid))
    Vcinf = 0.0
    for i,r in enumerate(grid):
        I = 0.0
        for j in range(i):
            rp = grid[j]
            I += (rhoArr[j]*(rp**2)/r)*step_size - rhoArr[j]*rp*step_size
        Vc12[i] = I 
    for i in range(0,len(grid)):
        Vcinf += rhoArr[i]*grid[i]*step_size
    
    Vc = 4*np.pi*e2*(Vc12 + Vcinf)
    return Vc*4*np.pi*e2

def centriforceArr(l):
    result = l*(l+1)/grid**2
    result[0] = 0.0
    return result
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