import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_chebyt

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

N = 30 # number of basis functions
h = .1

for N in np.arange(20,100,20):
    interval = (0,2*np.pi)
    grid = np.arange(interval[0],interval[1]+h,h)
    ## boundary values y(-1) = y0 = a and y(1)=y1 = b
    y0 = 0
    y1 = 0
    collocationPnts = getGaussLobatto(N,interval=interval)
    
    ## Rescale so the points are in [-1,1]
    alpha1 = .5*(max(collocationPnts) - min(collocationPnts))
    alpha2 = .5*(max(collocationPnts) + min(collocationPnts))
    #transform the coordinates 
    collocationPnts = collocationPnts/alpha1  - alpha2/alpha1
    
    D_1 = getDerMatrix(collocationPnts)
    D_2 = np.matmul(D_1,D_1)
    
    I = np.identity(N+1)
    # Now that we have the derivative matrix, we just need to construct the matrix 
    # for the ODE
    
    L = D_2/alpha1**2
    
    # construct F
    F = np.full(len(L),0) #- y0*L[:,0] - y1*L[:,-1]
    
    # enforce BCs by removing first and last rows and columns. This is the boundary
    # bordering technique.
    L = np.delete(L,0,0)
    L = np.delete(L,-1,-1)
    L = np.delete(L,0,-1)
    L = np.delete(L,-1,0)
    
    
    
    
    #sol = np.linalg.solve(L,F)
    # rescale the solution and coordinates
    #collocationPnts = collocationPnts*alpha1  + alpha2
    #sol = np.concatenate([[y0],sol,[y1]])
    
    
    eigs,evects = np.linalg.eig(L)
    collocationPnts = collocationPnts*alpha1  + alpha2
    print(eigs)
    plt.plot(collocationPnts[1:N],np.real(evects[:,-1]))
    plt.show()
    #plt.plot(collocationPnts,sol)
    #plt.show()