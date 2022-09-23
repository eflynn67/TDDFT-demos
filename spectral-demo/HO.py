import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
from scipy import special
import math
import h5py
import time
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
def derFactors1(L,x):
    f1 = (-3*L**2 * x)/(L**2 + x**2)**(5/2)
    f2 = L**2 / (L**2 + x**2)**(3/2)
    return f1,f2
def arcTransform(Cpnts,beta):
    '''
    Transformation for Chebyshev Gauss_Lobatto points. Transformation taken from
    Kosloff and Tal-Ezer (1991)
    this function maps f: [-1,1]-> [-1,1]
    Cpnts must be in the interval [-1,1].
    Parameters
    ----------
    Cpnts : ndarray
        grid points to transform.
    beta : float
        parameter to control the grid spacing.
    Returns
    -------
    Cpnts_mapped : ndarray
        transformed grid points
    dgdy: derivative of the coordinate transformation with respect to y (Cpnts)
    '''
    beta = float(beta)
    if beta == 0.0:
        Cpnts_mapped = Cpnts
        dgdy = np.full(Cpnts.shape,1.0)
    else:
        Cpnts_mapped = np.arcsin((beta)*Cpnts)/np.arcsin(beta)
        dgdy = (beta/np.arcsin(beta))* 1.0/np.sqrt(1 - (beta*Cpnts)**2 )
    return Cpnts_mapped, dgdy

def cotTransform(Cpnts,L):
    '''
    Coordinate transform that maps points from [0,2pi]. Taken from Spectral Methods
    in Fluid Dynamics (1988)
    this function maps f: [0,2pi]-> (-inf,inf)
    Cpnts must be in the interval [0,2pi].
    Parameters
    ----------
    Cpnts : ndarray
        grid points to transform.
    L : float
        parameter to control the grid spacing scale.
    Returns
    -------
    result : ndarray
        transformed grid points
    '''
    result = -L* 1.0/np.tan(Cpnts*.5)
    return result
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
def getGaussLobatto(N):
    '''
    Gets the Gauss-Lobatto points of the Mth chebyshev polynomials.
    Formula taken from Spectral Methods in Fluid Dynamics (1988)
    N: integer
    max number of basis elements
    '''
    GSPnts = np.zeros(N+1)
    w = np.zeros(N+1)
    for l in range(N+1):
        GSPnts[l] =  np.cos(l*np.pi/N)
        if (l == 0) or (l == N):
            w[l] = np.pi/(2.0*N)
        else:
            w[l] = np.pi/N
    return GSPnts, w

def c_coeff(l,N):
    if l == 0 or l == N:
        return 2.0
    else:
        return 1.0

def getDerMatrix(CPnts):
    '''
    Analytic solution taken from Spectral Methods in Fluid Dynamics (1988) pg 69 (or 84 in pdf).
    Assumes you are taking the collocation points at the Gauss-Lobatto points

    i labels the collocation point, j labels C
    Parameters
    ----------
    Cpnts: array
        array of collocation points to evaluate the derivative matrix at
    BC: string
        set the type of boundary conditions you want.
        dirichlet: this returns a matrix of size N - 2. This is because we remove
        two rows and two columns. This is because the dirichlet BCs allows us to move
        the boundary terms in the non-homogenous part
    Returns
    -------
    None.
    '''
    N = len(CPnts) - 1
    D = np.zeros((N+1,N+1))
    for i in range(N+1):
        for j in range(N+1):
            if i == 0 and j == 0:
                D[i][j] = -(2 * N**2 + 1)/6
            elif i == N  and j == N:
                D[i][j] = (2 * N**2 + 1)/6
            elif i == j and j <= N and j >= 1 and i <= N  and i >= 1 :
                D[i][j] = - CPnts[j]/(2*(1 - CPnts[j]**2))
            else:
                D[i][j] = c_coeff(i,N)*(-1)**(i+j) /(c_coeff(j,N)*(CPnts[i] - CPnts[j]))
    return D

def getCollocation_fourier(N):
    '''
    Gets the Gauss-Lobatto points of the Mth chebyshev polynomials.
    Formula taken from Spectral Methods in Fluid Dynamics (1988)
    N: integer
    max number of basis elements
    '''

    CPnts = np.zeros(N+1)
    for j in range(N+1):
        CPnts[j] = 2.0*j*np.pi/N
    return CPnts

def getDerMatrix_fourier(CPnts):
    '''
    Analytic solution taken from Spectral Methods in Fluid Dynamics (1988) pg 69 (or 84 in pdf).
    Assumes you are taking the collocation points at the Gauss-Lobatto points

    i labels the collocation point, j labels C
    Parameters
    ----------
    Cpnts: array
        array of collocation points to evaluate the derivative matrix at
    BC: string
        set the type of boundary conditions you want.
        dirichlet: this returns a matrix of size N - 2. This is because we remove
        two rows and two columns. This is because the dirichlet BCs allows us to move
        the boundary terms in the non-homogenous part
    Returns
    -------
    None.
    '''
    N = len(CPnts)
    S = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i != j:
                S[i][j] = .5*(-1)**(i+j) * 1.0/np.tan((i-j)*np.pi/N)
    return S
def prop_cheb(psi,H,dt,weights):

    dpsi = np.zeros(psi.shape,dtype='complex')
    dpsi_buffer = np.zeros((prop_order,len(psi)),dtype='complex')
    H_norm = np.linalg.norm(H,ord=1)
    H_normalized = H/H_norm
    tau = dt*H_norm
    dpsi_buffer[0] = psi
    dpsi_buffer[1] = np.matmul(H_normalized,psi)
    for k in np.arange(0,prop_order):
        if k  == 0:
            dpsi += (- 1.0j)**k * special.jv(k,tau)*dpsi_buffer[k]
        elif k ==1 :
            dpsi += 2.0*(- 1.0j)**k * special.jv(k,tau)*dpsi_buffer[k]
        else:
            dpsi_buffer[k] = 2.0*np.matmul(H_normalized,dpsi_buffer[k-1]) - dpsi_buffer[k-2]
            dpsi += (2.0*(- 1.0j)**k) * special.jv(k,tau)*dpsi_buffer[k]

    norm = 1.0/normalize(np.abs(dpsi),weights)
    dpsi = dpsi*norm
    return dpsi

def prop(psi,H,dt,weights):
    dpsi = np.zeros(psi.shape,dtype='complex')
    dpsi_buffer = np.zeros((prop_order,len(psi)),dtype='complex')
    H_norm = np.linalg.norm(H,ord=1)
    H_normalized = H/H_norm
    tau = dt*H_norm
    dpsi_buffer[0] = psi
    dpsi_buffer[1] = np.matmul(H_normalized,psi)
    for k in np.arange(0,prop_order):
        if k  == 0:
            dpsi += ((- 1.0j)**k)*dpsi_buffer[k]
        elif k ==1 :
            dpsi += ((- 1.0j*tau)**k)*dpsi_buffer[k]
        else:
            dpsi_buffer[k] = np.matmul(H_normalized,dpsi_buffer[k-1]) - dpsi_buffer[k-2]
            dpsi += ((- 1.0j*tau)**k)*dpsi_buffer[k]/math.factorial(k)
    norm = 1.0/normalize(np.abs(dpsi),weights)
    dpsi = dpsi*norm
    return dpsi
def normalize(funcArr,weights):
    norm = 0.0
    for i in range(len(weights)-1):
        norm += weights[i]*funcArr[i]
    norm = np.sqrt(norm)
    return norm

N = 200 # number of basis functions .
gamma = 1.0
interval = (-10,10)
Cpnts, weights= getGaussLobatto(N)
grid = np.linspace(interval[0],interval[1],200)
#Cpnts = getCollocation_fourier(N)
#print(CPnts)
## define affine transformation parameters
a = min(interval)
b = max(interval)
Cpnts_shift,dgdy = arcTransform(Cpnts, gamma ) # for chebyshev
#Cpnts_shift = cotTransform(Cpnts,1.0) # for fourier
#Cpnts_shift = Cpnts

#alpha1 = (b - a)/(2.0*np.pi) # for the fourier basis
#alpha2 = a

alpha1 = .5*(b-a) # for the chebyshev basis
alpha2 = .5*(a+b)

#affine transform the coordinates to the interval defined by interval variable
Cpnts_mapped = Cpnts_shift*alpha1  + alpha2
#CPnts_mapped = CPnts  + alpha2
## define derivative matrix on the interval (0,2pi)
A = np.diag(1.0/dgdy)

D_1 = np.matmul(A,getDerMatrix(Cpnts)/(alpha1))
#D_1 = getDerMatrix_fourier(Cpnts)


D_2 = np.matmul(D_1,D_1)

# Now that we have the derivative matrix, we just need to construct the matrix
# for the ODE on the new interval.
# the alpha1 factor comes from chain rule.

L = -D_2+ .5*np.diag(Cpnts_mapped**2)

#print(L)
# enforce BCs by removing first and last rows and columns. This is the boundary
# bordering technique.

L = np.delete(L,0,0)
L = np.delete(L,-1,-1)
L = np.delete(L,0,-1)
L = np.delete(L,-1,0)

n = 0
exact_psi = getPsi_x(n,.5)
evals,evects = np.linalg.eig(L)

idx = evals.argsort()
evals= evals[idx]
#print(eigs)
evects = evects[:,idx]

exact_GS = getExactLambda(n,1,.5)
exact_evect = exact_psi(Cpnts_mapped)
exact_evect_uniform = exact_psi(grid)

psi_0 = evects[:,n]
psi_0 = np.concatenate([[0],psi_0,[0]])
delta_x = np.zeros(len(Cpnts) - 1)
delta_x2 = np.zeros(len(grid)-1)
for i in range(len(Cpnts) - 1):
    delta_x[i] = Cpnts_shift[i-1] - Cpnts_shift[i]
for i in range(len(grid) - 1):
    delta_x2[i] = grid[i+1] - grid[i]
norm1 = 0
norm2 = 0
norm3 = 0
int_weights = delta_x*alpha1
for i in range(len(Cpnts) -1):
    norm1 += (int_weights[i])*psi_0[i]**2
    norm2 += (int_weights[i])*exact_evect[i]**2

for i in range(len(grid) -1):
    norm3 += (delta_x2[i])*exact_evect_uniform[i]**2
norm1 = np.sqrt(norm1)
norm2 = np.sqrt(norm2)
norm3 = np.sqrt(norm3)
print(norm1)
print(norm2)
print(norm3)
print('Eigenvalue Error: ', np.abs(evals[n]-exact_GS))
print('EigenFunction Error ', np.linalg.norm(abs(psi_0/norm1)-abs(exact_evect/norm2)))
#plt.plot(Cpnts_mapped,psi_0**2,label='Numerical N='+str(N))
plt.plot(Cpnts_mapped,abs(psi_0)/norm1,label='Numerical N='+str(N))
plt.plot(Cpnts_mapped,exact_evect/norm2,label='exact N='+str(N))
plt.plot(grid,exact_evect_uniform/norm3,label='uniform')
#plt.plot(Cpnts_mapped[1:N],psi_0-exact_evect,label='exact')
plt.legend()
plt.show()


###############################
start_time = time.time()
prop_order = 6
nt_steps = 10000
dt = 10**(-4)
print(dt)
print(dt*nt_steps)
psi_series = np.zeros((nt_steps+2,len(Cpnts)-2),dtype=complex)
psi_series[0] = evects[:,0]/np.linalg.norm(evects[:,0]).copy()
bndyErr=[]
for i in range(nt_steps+1):
    psi_series[i+1] = prop(psi_series[i],L,dt=dt,weights =int_weights)
    err = np.abs(psi_series[i+1][0]-psi_series[i][0]) \
    + np.abs(psi_series[i+1][-1]-psi_series[i][-1])
    bndyErr.append(err)
    if i % 100 == 0:
        plt.plot(Cpnts_mapped[1:N],np.real(psi_series[i+1]))
        plt.plot(Cpnts_mapped[1:N],np.imag(psi_series[i+1]))
        plt.title(f't = {round(i*dt,3)}')
        plt.show()
end_time = time.time()

print('Time: ',end_time - start_time)
'''
m,b = np.polyfit(np.arange(nt_steps+1), np.log(bndyErr), 1)
print('b',b)
print('m',m)
plt.loglog(np.arange(nt_steps+1),bndyErr)

plt.xlabel('Number of time Steps')
plt.ylabel('psi[i+1][0] - psi[i][0]')
plt.savefig('error_buildup.pdf')
plt.show()
plt.plot(Cpnts_mapped[1:N],np.real(psi_series[-1]),label='real')
plt.plot(Cpnts_mapped[1:N],np.imag(psi_series[-1]),label='real')
plt.show()
'''