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
beta = 0 # the stretching parameter for the arcsin coordinate transform

transform = spec.coord_transforms(interval=interval)
CPnts_shift,dgdy_arc = transform.arcTransform(CPnts,beta)

# Define derivative matrix at the new Cpnts
A = np.diag(1.0/dgdy_arc)

#print(A)
CPnts_mapped,dgdy_affine = transform.inv_affine(CPnts_shift)
D_1 = np.matmul(A,spec.DerMatrix().getCheb(CPnts))/dgdy_affine

## initialize psi and psiStar with HO solutions

psi = np.zeros(len(CPnts))
psiStar = np.zeros(len(CPnts))


H_constructor = matrix.spec_H_func(N, CPnts_mapped,D_1)
H_func = H_constructor.HO

H = H_func(psi,psiStar,mass,alpha,q)
energies,evects = sci.linalg.eig(H)
idx = energies.argsort()
energies = energies[idx]
evects = evects[:,idx]
evects = evects.T
print(energies[0])
psi = evects[0]
psi = np.concatenate([[0],psi,[0]])
psiStar = np.conjugate(psi)
exact_psi = getPsi_x(0,alpha)
exact_evect = exact_psi(CPnts_mapped)/np.linalg.norm(exact_psi(CPnts_mapped))
exact_GS = getExactLambda(0,mass,alpha)
print('Eigenvalue Error: ', np.abs(energies[0]-exact_GS))
print('EigenFunction Error ', np.linalg.norm(abs(psi)-abs(exact_evect)))

plt.plot(CPnts_mapped,abs(psi),label='Numerical')
plt.plot(CPnts_mapped,exact_evect,label='Exact')
plt.title(f'New N = {N}')
plt.show()

'''



### Now solve the nonlinear problem self-consistently 
H_func = H_constructor.GP_HO
E_gs, psi = solver.MatrixSolve_SC(H_func,psi,psiStar)
print(E_gs)
plt.plot(CPnts_mapped,abs(psi))
plt.ylim([0,.3])
plt.show()

################################################################################
### Time Propagation
################################################################################
psi_series_forward = np.zeros((nt_steps+1,len(CPnts)),dtype=complex)
psi_series_backward = np.zeros((nt_steps+1,len(CPnts)),dtype=complex)
psi_series_forward[0] = psi[:].copy()
psi_series_backward[0] = psi[:].copy()
H =  H_func(psi_series_forward[0],psi_series_forward[0],mass,alpha,q,BC=False)

for i in range(nt_steps):
    #print(i*delta_t)
    psi_series_forward[i+1] = solver.prop(psi_series_forward[i],H,dt=0.5*delta_t,prop_order=prop_order)
    H = H_func(psi_series_forward[i+1],psi_series_forward[i+1],mass,alpha,q,BC=False)
    
    psi_series_forward[i+1] = solver.prop(psi_series_forward[i],H,dt=delta_t,prop_order=prop_order)
    
    if i % 200 == 0:
        plt.plot(CPnts_mapped,np.abs(psi_series_forward[i+1]))
        plt.title(f'Iteration {i}')
        plt.show()
    #err = np.abs(psi_series[i+1][0]-psi_series[i][0]) \
    #+ np.abs(psi_series[i+1][-1]-psi_series[i][-1])
    #bndyErr.append(err)

plt.plot(CPnts_mapped,np.abs(psi_series_forward[0]))
plt.plot(CPnts_mapped,np.abs(psi_series_forward[-1]))
plt.show()

m,b = np.polyfit(np.arange(nt_steps+1), np.log(bndyErr), 1)
print('b',b)
print('m',m)
plt.loglog(np.arange(nt_steps+1),bndyErr)
plt.xlabel('Number of time Steps')
plt.ylabel('Boundary Error')
plt.show()

'''