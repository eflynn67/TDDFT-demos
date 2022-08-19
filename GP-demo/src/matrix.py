import scipy as sci
import potentials
from init import *
def get_H_func(name='HO'):
    if name =='HO':
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
                + np.diag(potentials.V_rho(psi,psiStar,q))
            return H
        return HO
    elif name=='quartic':
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
        return quartic
    elif name=='gaussian':
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
        return gaussian
    if name =='double':
        def double(psi,psiStar,mass,alpha,q):
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
            dim = len(largegrid)
            off_diag = np.zeros(dim)
            off_diag[1] = 1
            H = -1*(-2*np.identity(dim) + sci.linalg.toeplitz(off_diag))/(step_size**2) + np.diag(potentials.V_HO(largegrid,alpha)) \
                + np.diag(potentials.V_rho(psi,psiStar,q))
            return H
        return double
    else:
        raise(Exception('No Hamiltonian with that name.'))