import numpy as np
import scipy as sci
from scipy import optimize
from scipy import special
import matplotlib.pyplot as plt
import sys
from init import *
from inputs import *
import matrix
import solver

sys.path.insert(0, '../../src/methods')
import spec
def getExactLambda(n,mass,alpha):
    '''
    Exact eigenvalues of the HO equation. -f''(x) + k x^2 f(x) = 2 m E f(x)
    Lambda = 2 m E
    E = (n + .5) \omega
    \alpha = m^2 \omega^2
    \omega = \sqrt{alpha/m^2}
    Parameters
    ----------
    n : float
        principle quantum number. float integers
    omega : float
        oscillator frequency.

    Returns
    -------
    float
        oscillator energy 2mE.
    '''
    return 2*mass*(.5 + n)*np.sqrt(alpha/mass**2)
def getPsi_x(n,k):
    '''
    Definition of exact HO wavefunction taken from Zettili page 240.

    Parameters
    ----------
    n : TYPE
        principle quantum number for SE equation
    k : TYPE
        harmonic oscillator parameter (mass*omega)^2 from the potential term \mu^2 \omega^{2} x^{2} in the SE.
    Returns
    -------
    wf : function
        1-d wavefunction for the 1d-harmonic oscillator as a function of position x.
    '''
    herm = special.hermite(n)
    def wf(x):
        result = (1/np.sqrt(np.sqrt(np.pi)*2**(n)*np.math.factorial(n)*(k)**(.25)))*np.exp(-x**(2)*np.sqrt(k)/2)*herm(x*k**(.25))
        return(result)
    return wf
#N = 300
interval = (-10,10)
## Define GaussLobatto points in the interval [-1,1]
CPnts = spec.GaussLobatto().chebyshev(N)

# To reduce the stability problems with the GL points, we map these points using
# the arcsin transform
beta = 1.0 # the stretching parameter for the arcsin coordinate transform

transform = spec.coord_transforms(interval=interval)
CPnts_shift,dgdy_arc = transform.arcTransform(CPnts,beta)

# Define derivative matrix at the new Cpnts
A = np.diag(1.0/dgdy_arc)

#print(A)
CPnts_mapped,dgdy_affine = transform.inv_affine(CPnts_shift)
D_1 = np.matmul(A,spec.DerMatrix().getCheb(CPnts))/dgdy_affine
## map the Cpnts on [-1,1] to the truncated interval defined by the variable




H_constructor = matrix.spec_H_func(N, CPnts_mapped,D_1)

H = H_constructor.HO(alpha,mass)

evals, evects = np.linalg.eig(H)
idx = evals.argsort()
evals = evals[idx]
evects = evects[:,idx]
gs_exact = getExactLambda(0,mass,alpha)
exact_psi = getPsi_x(0,.5)
exact_evect = exact_psi(CPnts_mapped[1:N])/np.linalg.norm(exact_psi(CPnts_mapped[1:N]))
psi_0 = evects[:,0]/np.linalg.norm(evects[:,0])
print('ground state energy error',abs(evals[0]-gs_exact))
print('ground state evect error',np.linalg.norm(abs(psi_0)-exact_evect))
plt.plot(CPnts_mapped[1:N],abs(psi_0),label='Numerical')
plt.plot(CPnts_mapped[1:N],abs(exact_evect),label='Exact')
plt.legend()
plt.show()



################################################################################
### Time Propagation
################################################################################
psi_series = np.zeros((nt_steps+2,len(psi_0)),dtype=complex)
psi_series[0] = psi_0.copy()
bndyErr=[]
print(nt_steps*delta_t)
for i in range(nt_steps+1):
    psi_series[i+1] = solver.prop_cheb(psi_series[i],H,dt=delta_t,prop_order=prop_order)
    err = np.abs(psi_series[i+1][0]-psi_series[i][0]) \
    + np.abs(psi_series[i+1][-1]-psi_series[i][-1])
    bndyErr.append(err)

m,b = np.polyfit(np.arange(nt_steps+1), np.log(bndyErr), 1)
print('b',b)
print('m',m)
plt.loglog(np.arange(nt_steps+1),bndyErr)
plt.xlabel('Number of time Steps')
plt.ylabel('Boundary Error')
plt.show()
plt.plot(CPnts_mapped[1:N],np.real(psi_series[-1]),label='real')
plt.plot(CPnts_mapped[1:N],np.imag(psi_series[-1]),label='real')
plt.show()