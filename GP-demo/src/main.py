import numpy as np
import scipy as sci
from scipy import optimize
from scipy import special
import matplotlib.pyplot as plt

from init import *
from inputs import *
import matrix
import wf
import potentials as pots
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
N = 300
interval = (-100,100)
CPnts = spec.GaussLobatto(interval=interval).chebyshev(N) #Note these are NOT mapped to [-1,1] 
#CPnts_unit = spec.coord_transforms(interval=interval).affine(CPnts)


H_constructor = matrix.spec_H_func(N, CPnts,interval)
H = H_constructor.HO(alpha)

evals, evects = np.linalg.eig(H)
idx = evals.argsort()  
evals = evals[idx]
evects = evects[:,idx]
gs_exact = getExactLambda(0,mass,alpha)
print('ground state energy error',evals[0]-gs_exact)
psi_0 = evects[:,0]/np.linalg.norm(evects[:,0])
plt.plot(CPnts[1:N],psi_0)

plt.show()
Hk = np.matmul(H,H)
print(np.max(Hk))
test_psi_0 = np.matmul(Hk,psi_0)
#print(test_psi_0)
test_evals ,test_evects= np.linalg.eig(Hk)
test_idx = test_evals.argsort()  
test_evals = test_evals[idx]
test_evects = test_evects[:,idx]
#print(test_evals)
plt.plot(CPnts[1:N],test_psi_0)
plt.show()
#plt.plot(CPnts[1:N],test_evects[:,0])
#plt.xlim([-10,10])
#plt.show()

'''
psi_series = np.zeros((nt_steps+2,len(CPnts)-2),dtype=complex)
psi_series[0] = evects[:,0]/np.linalg.norm(evects[:,0]).copy()
for i in range(nt_steps+1):
    psi_series[i+1] = solver.prop(psi_series[i],H,delta_t)
    #print(max(np.real(psi_series[i+1])))
    plt.plot(CPnts[1:N],np.real(psi_series[i+1]))
    plt.show()



psi_func = wf.getPsi_x(0,1,0) ## GS harmonic oscillator.

psiArr = psi_func(grid)
norm_psi = 1.0/np.linalg.norm(psiArr)
psiArr = psiArr*norm_psi
psiStarArr = np.conjugate(psiArr) 

H_func = matrix.get_H_func(name='gaussian')
E_GS, evect_GS = solver.MatrixSolve_SC(H_func, psiArr, psiStarArr)
evect_GS = evect_GS/np.linalg.norm(evect_GS)

## remake the Hamiltonian with GS used in the non-linear interaction
H = H_func(evect_GS,np.conjugate(evect_GS),mass,alpha,q)
print('Numerical GP:', E_GS)
plt.plot(grid,np.real(evect_GS))
plt.plot(grid ,np.imag(evect_GS))
plt.title('Ground state')
plt.show()

V = pots.V_gaussian(grid)
plt.plot(grid,V,label='V')
plt.plot(grid,np.real(evect_GS),label='GS real')
plt.plot(grid,np.imag(evect_GS),label='GS imag')

#plt.xlim([-10,10])
#plt.ylim([-1,1])
plt.title('Potential')
plt.legend()
plt.show()


### Propagation 
psi_series_forward = np.zeros((nt_steps+1,len(grid)),dtype=complex)
psi_series_backward = np.zeros((nt_steps+1,len(grid)),dtype=complex)
psi_series_forward[0] = evect_GS.copy()
psi_series_backward[0] = evect_GS.copy()
for j in range(nt_steps):
    # predictor step
    psi_series_forward[j+1] = solver.prop(psi_series_forward[j],H,0.5*1.0j*delta_t)
    psi_series_backward[j+1] = solver.prop(psi_series_backward[j],H,-0.5*1.0j*delta_t)
    ## recalculate the H using the predictor GS wavefunction
    H = H_func(psi_series_forward[j+1],psi_series_backward[j+1],mass,alpha,q)
    # now calculate the corrector step with the newly calculated H
    psi_series_forward[j+1] = solver.prop(psi_series_forward[j],H,1.0j*delta_t)
    psi_series_backward[j+1] = solver.prop(psi_series_backward[j],H,-1.0j*delta_t)
    ## plot each time step
    #plt.plot(grid,np.real(psi_series_backward[j+1]),label='$Re(\psi)$ backward')
    #plt.plot(grid,np.imag(psi_series_backward[j+1]),label='$Im(\psi)$ backward')
    #plt.plot(grid,np.real(psi_series_forward[j+1]),label='$Re(\psi)$ forward')
    #plt.plot(grid,np.imag(psi_series_forward[j+1]),label='$Im(\psi)$ forward')
    plt.plot(grid,np.real(np.conjugate(psi_series_forward[j])*psi_series_forward[j]),label='$\psi^{*}\psi$')
    plt.plot(grid,V,label='V')
    plt.title(f't = {np.around(delta_t + delta_t*j,4)} (step {j})')
    plt.ylim([-.1,.2 ])
    plt.legend()
    plt.show()
    
print('End')
'''