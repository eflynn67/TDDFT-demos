import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_chebyt
'''
Here we use Fourier basis to solve a periodic boundary condition problem.
First we solve a very simple one
1) y'' = - y
with Periodic boundary conditions y(0) = y(2 pi) = 0
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
        CPnts[i] = 2*j*np.pi/(N+1)
    return GSPnts

def c_coeff(l,N):
    if l == 0 or l == N:
        return 2.0
    else: 
        return 1.0
def getDerMatrix(CPnts,BC = 'dirichlet'):
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
    S = np.zeros((N+1,N+1))
    for i in range(N+1):
        for j in range(N+1):
            if i != j:
                S[i][j] = .5*(-1)**(i+j) * np.cot((i-j)*np.pi/N)
    return S

N = 100 # number of basis functions .

##############################################################################
### Part 1

interval = (0,1)
## boundary values y(-1) = y0 = a and y(1)=y1 = b
y0 = 0
y1 = 0

