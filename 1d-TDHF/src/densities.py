import numpy as np
from numba import jit
from init import * 

def rho(psi_array):
    '''
    Calculates the density at points r for protons and neutrons

    Parameters
    ----------
    psi : nd array
        array containing the wavefunctions for a particular q. should have size Z x len(grid) or N x len(grid)
    r : nd array
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    rho_p = np.zeros(len(grid))
    rho_n = np.zeros(len(grid))
    for n in range(nmax):
        for l in range(lmax+1):
            for s in range(len(spin)):
                j = l + spin[s]
                #print('n',n,'l',l,'s',s,'j',j)
                rho_p += 1/(4*np.pi*grid**2)*(2*j+1)*psi_array[0][n][l][s]**2
                rho_n += 1/(4*np.pi*grid**2)*(2*j+1)*psi_array[1][n][l][s]**2
    rho_tot = rho_p + rho_n
    return rho_tot,rho_p,rho_n

