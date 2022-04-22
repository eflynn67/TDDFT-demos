import numpy as np
from numba import jit
from init import *
import densities

@jit(nopython=True)
def yukArr(rhoArr):
    Vyuk = np.zeros(len(grid))
    for i,r in enumerate(grid):
        I = 0.0 
        for j in range(i):
            rp = grid[j]
            I += rhoArr[j]*(np.exp(-abs(r-rp)/a)/abs(r-rp))*rp**2 * step_size
        Vyuk[i] = I
    return Vyuk*aV0*4*np.pi

@jit(nopython=True)
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
    return Vyuk*aV0*2*np.pi
@jit(nopython=True)
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
    
    Vc = 4*np.pi*e2*(Vc12 + Vcinf) #- e2*(3.0/np.pi)**(1.0/3.0)*rhoArr**(1.0/3.0)
    return Vc

def centriforceArr(l):
    result = l*(l+1)/grid**2
    result[0] = 0.0
    return result

'''
def externalV(r,t):
    
    return None
'''