import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import matrix
#import wf
sys.path.insert(0, '../solvers')
import spec
import solvers
def getExactLambda(n,mass,alpha):
    '''
    Exact eigenvalues of the HO equation. -f''(x) + k x^2 f(x) = 2 m E f(x)
    Used for checking the solution method
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
    Used for checking the solution method
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

###############################################################################
## Self Consistent Solve parameters
niter = 50 # number of self consistent iterations
sigma = .5 #sigma = [0,1]. 0.0 means nothing happens

###############################################################################
## Propagation parameters
prop_order = 6 #Expansion order of the propagator e^(-i \Delta t h(t)).
delta_t = 10**(-4) # time step length rule of thumb is delta_t < 1/N^(1.5)
nt_steps = 10 #number of time steps

###############################################################################
## Domain properties
#step_size = .2 ## for finite difference schemes
N = 20 # number of collocation points.
lb = -5 # left boundary
rb = 5 # right boundary

###############################################################################
## Interaction parameters
mass = 1.0
kappa = 1.0 # interaction strength for HO potential 
q = -0.5# interaction strength for |psi|^2 term in GP Hamiltonian

###############################################################################
e2 = 1.4399784 # e^2 charge in MeV fm
hb2m0 = 20.735530
kappa = 1.0
mass = 1.0
params = {'mass':mass,'nt_steps':nt_steps,'niter':niter,'hb2m0':hb2m0,'kappa':kappa,'q':q}


###############################################################################
currentInterval = (-1,1) # interval the derivatives are defined on. (we start here)
targetInterval = (lb,rb) ## interval on real space to solve on
## Define Chebyshev GaussLobatto points in the interval [-1,1]
GL_func = spec.GaussLobatto()
CPnts,weights = GL_func.chebyshev(N)

# To reduce the stability problems with the GL points, we map these points using
# the arcsin transform
beta = 0.5 # the stretching parameter for the arcsin coordinate transform

coord_trans = spec.coord_transforms(currentInterval,targetInterval)
CPnts_shift,dgdy_arc = coord_trans.arcTransform(CPnts,beta)


# Define derivative matrix at the new Cpnts
A = np.diag(1.0/dgdy_arc)

#print(A)
CPnts_mapped,dgdy_affine = transform.inv_affine(CPnts_shift)
D_1 = np.matmul(A,spec.DerMatrix().getCheb(CPnts))/dgdy_affine

# define integration weights
int_weights = spec.GaussLobatto().getDx(CPnts_shift)*dgdy_affine

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

psi_norm = wf.normalize(psi**2,int_weights)
psi = psi/psi_norm
psiStar = np.conjugate(psi)/psi_norm

exact_psi = getPsi_x(0,alpha)
exact_evect = exact_psi(CPnts_mapped)
exact_norm = wf.normalize(exact_evect**2,int_weights)
exact_evect = exact_evect/exact_norm
exact_GS = getExactLambda(0,mass,alpha)

'''

print('Eigenvalue Error: ', np.abs(energies[0]-exact_GS))
print('EigenFunction Error ', np.linalg.norm(abs(psi)-abs(exact_evect)))

plt.plot(CPnts_mapped,abs(psi),label='Numerical')
plt.plot(CPnts_mapped,exact_evect,label='Exact')
plt.legend()
plt.show()
'''



### Now solve the nonlinear problem self-consistently
H_func = H_constructor.GP_HO
E_gs, psi = solver.MatrixSolve_SC_hermitian(H_func,psi,psiStar,int_weights)
print(E_gs)
plt.plot(CPnts_mapped,abs(psi))
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
    psi_series_forward[i+1] = solver.prop_cheb(psi_series_forward[i],H,dt=0.5*delta_t,prop_order=prop_order,weights=int_weights)
    H = H_func(psi_series_forward[i+1],psi_series_forward[i+1],mass,alpha,q,BC=False)

    psi_series_forward[i+1] = solver.prop_cheb(psi_series_forward[i],H,dt=delta_t,prop_order=prop_order,weights=int_weights)
    #psi_series_forward[i+1] = psi_series_forward[i+1]/wf.normalize(psi_series_forward[i+1])
    if i % 500 == 0:
        plt.plot(CPnts_mapped,np.real(psi_series_forward[i+1]),label='real')
        plt.plot(CPnts_mapped,np.imag(psi_series_forward[i+1]),label='imag')
        plt.plot(CPnts_mapped,np.abs(psi_series_forward[i+1]),label='rho')
        plt.title(f't = {round(i*delta_t,2)}')
        plt.legend()
        plt.show()
    #err = np.abs(psi_series[i+1][0]-psi_series[i][0]) \
    #+ np.abs(psi_series[i+1][-1]-psi_series[i][-1])
    #bndyErr.append(err)

#plt.plot(CPnts_mapped,np.abs(psi_series_forward[0]))
#plt.plot(CPnts_mapped,np.abs(psi_series_forward[-1]))
plt.show()

