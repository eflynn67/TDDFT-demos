import numpy as np
from init import *
from inputs import *
def rho(psiArr,psiArrStar,x):
    '''
    Calculates the density at points r for protons and neutrons

    Parameters
    ----------
    psiArr : nd array
        array containing the wavefunctions for a particular q. should have size Z x len(grid) or N x len(grid)
    psiStarArr : nd array
        array containing the conjugate wavefunctions for a particular q. should have size Z x len(grid) or N x len(grid). These 
        wavefunctions can either be conj(psi(x,t)) for real time sols or psi(x,-t) for tunneling sols. 
    x : nd array
        spatial grid.

    Returns
    -------
    rho: ndArray.

    '''
    rho = np.zeros(len(grid))
    for q in range(2):
        for n in range(nmax+1):
            for s in range(len(spin)):
                rho += psiArrStar[q][n][s]*psiArr[q][n][s]
    return rho
    
    