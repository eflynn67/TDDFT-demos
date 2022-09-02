import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_chebyt
'''
Here we use Fourier basis to solve a periodic boundary condition problem.

1) y'' + x y' = x 
with Dirichelt BCs

Next, we tackle a non-linear equation: Painleve I

y'' = 6y^2 + x
with Dirichlet BCS y(0) = 0 and y(1) = 0. This is a different beast. We use the self consistent approach
to solving this equation. We start our initial guess using x_{j}(x_{j}-1)
'''

def getGaussLobatto(N,interval=(-1,1)):
    '''
    Gets the Gauss-Lobatto points of the Mth chebyshev polynomials.
    Formula taken from Spectral Methods in Fluid Dynamics (1988)
    N: integer
    max number of basis elements
    '''
    GSPnts = np.zeros(N+1)
    for l in range(N+1):
        GSPnts[l] = .5*(interval[0] + interval[1]) - .5*(interval[1] -interval[0]) \
            *np.cos(l*np.pi/N)
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

N = 100 # number of basis functions .

##############################################################################
### Part 1

interval = (0,1)
## boundary values y(-1) = y0 = a and y(1)=y1 = b
y0 = 0
y1 = 0

