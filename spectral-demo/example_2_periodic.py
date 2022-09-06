import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_chebyt
'''
Here we use Fourier basis to solve a periodic boundary condition problem.
First we solve a very simple one
1) y' = - y
But, we do this in a strange way. We first use the Fourier Basis and the 
associated Gauss-Lobatto Points. Then we use an analytic expression for the 
derivative matrix. 

The first deriviative matrix is given on pg 44 of Spectral Methods in Fluid Dynamics (1988)
The second derivative matric is given on 
'''

def getCollocation(N):
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
    N = len(CPnts)  
    S = np.zeros((N,N),dtype='complex')
    for i in range(N):
        for j in range(N):
            if i != j:
                S[i][j] = .5*(-1)**(i+j) * 1.0/np.tan((i-j)*np.pi/N)
    return S



N = 501 # number of basis functions. for odd number, this is includes 0 and the end boundary 
##############################################################################
### Part 1


CPnts = getCollocation(N)
D_1 = getDerMatrix(CPnts)

f = np.sin(2*CPnts)
plt.plot(CPnts,f)
plt.show()

dfdx = np.matmul(D_1,f)

plt.plot(CPnts,dfdx)
plt.plot(CPnts,f)
plt.show()

'''

I = np.identity(len(D_1),dtype='complex')
L = D_1
eigs,evects = np.linalg.eig(L)
abs_eigs = np.abs(eigs)

idx = abs_eigs.argsort()   
eigs= eigs[idx]
#print(eigs)
evects = evects[:,idx]
F = np.zeros(len(D_1))

#sol = np.linalg.solve(L,F)

#plt.plot(CPnts,sol)
#plt.show()
#print(evects[:,2][0])
#print(evects[:,2][-1])
plt.plot(CPnts,np.real(evects[:,2])/np.abs(evects[:,2]))
plt.plot(CPnts,np.imag(evects[:,2])/np.abs(evects[:,2]))
plt.plot(CPnts+2*np.pi,np.real(evects[:,2])/np.abs(evects[:,2]))
plt.plot(CPnts+2*np.pi,np.imag(evects[:,2])/np.abs(evects[:,2]))
#plt.plot(CPnts,np.real(evects[:,3])/np.abs(evects[:,3]))
#plt.plot(CPnts,np.imag(evects[:,3])/np.abs(evects[:,3]))
plt.show()
'''