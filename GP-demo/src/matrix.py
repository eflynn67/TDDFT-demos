import scipy as sci
import potentials
from init import *
def construct_H_GP(psi,psiStar,mass,alpha,q):
    '''
    Uses 2nd order finite difference scheme to construct a discretized differential 
    H operator for the GP potential.

    Parameters
    ----------
    psi : ndArray
        wavefunction.
    psiStar : ndArray
        wavefunction conjugate dual.
    mass : TYPE
        DESCRIPTION.
    alpha : TYPE
        DESCRIPTION.
    q : TYPE
        DESCRIPTION.

    Returns
    -------
    H : TYPE
        DESCRIPTION.

    '''
    dim = len(grid)
    off_diag = np.zeros(dim)
    off_diag[1] = 1
    H = -1*(-2*np.identity(dim) + sci.linalg.toeplitz(off_diag))/(mass*step_size**2) + np.diag(potentials.V_HO(grid,alpha)) \
        + np.diag(potentials.V_rho(psi,psiStar,q))
    return H