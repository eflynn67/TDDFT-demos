import scipy as sci
import numpy as np
import potentials
import sys
from init import *
sys.path.insert(0, '../../src/methods')
import spec

class FD_H_func:
    def HO(psi,psiStar,mass,alpha,q):
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
        H = -1*(-2*np.identity(dim) + sci.linalg.toeplitz(off_diag))/(step_size**2) + np.diag(potentials.V_HO(grid,alpha)) \
            +np.diag(potentials.V_rho(psi,psiStar,q))
        return H
    def quartic(psi,psiStar,mass,alpha,q):
        '''
        Uses 2nd order finite difference scheme to construct a discretized differential
        H operator for the GP with quartic potential.

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
        H = -1*(-2*np.identity(dim) + sci.linalg.toeplitz(off_diag))/(mass*step_size**2) + np.diag(potentials.V_quartic(grid,alpha)) \
            + np.diag(potentials.V_rho(psi,psiStar,q))
        H = np.array(H,dtype='complex')
        return H
    def gaussian(psi,psiStar,mass,alpha,q):
        '''
        Uses 2nd order finite difference scheme to construct a discretized differential
        H operator for the GP potential with gaussian potential.

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
        H = -1*(-2*np.identity(dim) + sci.linalg.toeplitz(off_diag))/(mass*step_size**2) + np.diag(potentials.V_gaussian(grid)) \
            + np.diag(potentials.V_rho(psi,psiStar,q))
        return H
class spec_H_func:
    def __init__(self, N,CPnts,D_1):
        self.N = N
        self.CPnts = CPnts
        self.D_1 = D_1
        self.D_2 = np.matmul(D_1,D_1)
    def HO(self,psi,psiStar,mass,alpha,q,BC=True):
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
        H = -1.0*self.D_2/mass + np.diag(potentials.V_HO(self.CPnts,alpha))
        # enforce boundary conditions by setting end points to zero. This means
        # we can remove two rows and two cols
        if BC == True:
            H = np.delete(H,0,0)
            H = np.delete(H,-1,-1)
            H = np.delete(H,0,-1)
            H = np.delete(H,-1,0)
        return H
    def gaussian(self,psi,psiStar,mass,alpha,q,BC=True):
        H = -1.0*self.D_2/mass + np.diag(potentials.V_gaussian(self.CPnts))
        # enforce boundary conditions by setting end points to zero. This means
        # we can remove two rows and two cols
        if BC == True:    
            H = np.delete(H,0,0)
            H = np.delete(H,-1,-1)
            H = np.delete(H,0,-1)
            H = np.delete(H,-1,0)
        return H
    def GP_HO(self,psi,psiStar,mass,alpha,q,BC=True):
        H = -1.0*self.D_2/mass + np.diag(potentials.V_HO(self.CPnts,alpha)) \
            + np.diag(potentials.V_rho(psi,psiStar,q))
        # enforce boundary conditions by setting end points to zero. This means
        # we can remove two rows and two cols
        if BC == True:
            H = np.delete(H,0,0)
            H = np.delete(H,-1,-1)
            H = np.delete(H,0,-1)
            H = np.delete(H,-1,0)
        return H
    def GP_gaussian(self,psi,psiStar,mass,alpha,q,BC=True):
        H = -1.0*self.D_2/mass + np.diag(potentials.V_gaussian(self.CPnts)) \
            + np.diag(potentials.V_rho(psi,psiStar,q))
        # enforce boundary conditions by setting end points to zero. This means
        # we can remove two rows and two cols
        if BC == True:
            H = np.delete(H,0,0)
            H = np.delete(H,-1,-1)
            H = np.delete(H,0,-1)
            H = np.delete(H,-1,0)
        return H

