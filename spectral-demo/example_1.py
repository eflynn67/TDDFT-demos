import numpy as np
import matplotlib.pyplot as plt
'''
This is a numerical implementation of the worked example from pg 1094 of 
Numerical Recipes. 
Here we solve the ODE

y'' + y' -2y + 2 = 0 

on the interval [-1,1] with Dirichlet BCs y(-1) = y(1) = 0.
This is just practice implementing a pseudo-spectral method to solve a simple ODE.
We use Chebyshev polynomials as a basis.

Since we have Dirchlet BCs, we use Gauss-Lobbato points as the collocation points.

'''


def exact_sol(x):
    '''
    Exact solution to the ODE.
    '''
    result = 1.0 - (1.0/np.sinh(3))*(np.exp(x)*np.sinh(2) + np.exp(-2)*np.sinh(1))
    return result
def getGaussLobatto(NcPnts,N):
    '''
    Gets the Gauss-Lobatto points of the Mth chebyshev polynomials.
    Formula taken from Spectral Methods in Fluid Dynamics (1988)
    cPnts: integer
    The number of collocation points you want
    N: integer
    max number of basis elements
    '''
    GSPnts = np.zeros(NcPnts)
    for l in range(NcPnts):
        GSPnts[l] = -np.cos(l*np.pi/N)
    return GSPnts
def getWeights(z,xPnts,m,n):
    '''
    Taken from Numerical recipes
    Parameters
    ----------
    z : TYPE
        location where the matrices are evaluated
    xPnts : TYPE
        set of collocation points
    m : TYPE
        max order of derivative wanted

    Returns
    -------
    None.

    '''
    C = np.zeros((n,m))
    c1 = 1.0
    c4 = xPnts[0] - z 
    C[0][0] = 1.0
    for i in np.arange(1,n-1,1):
        mn = min(i,m-1)
        c2 = 1.0
        c5 = c4
        c4 = xPnts[i]-z
        for j in range(i):
            c3 = xPnts[i]-xPnts[j]
            c2 = c2*c3
            if j == i-1:
                for k in np.arange(1,mn,1)[::-1]:
                    C[i][k] = c1*(k*C[i-1][k-1] - c5*C[i-1][k])/c2
                C[i][0] = -c1*c5*C[i-1][0]/c2
            else:
                for k in np.arange(1,mn,1)[::-1]:
                    C[j][k] = (c4*C[j][k] - k*C[j][k-1])/c3
                C[j][0] = c4*C[j][0]/c3
        c1=c2
    return C     
def get2nd_Derivative(x):
    n = len(x)
    m = 2
    d1 = np.zeros((n,n))
    d2 = np.zeros((n,n))
    for i in range(n):
        C = getWeights(x[i],x,m,n)
        for j in range(n):
            d1[i][j] = C[j][0]
            d2[i][j] = C[j][1]
    return d1,d2
            
h = .1   
grid = np.arange(-1,1,h)

N = 4 # number of chebyshev polynomials to use.
k = 5 # number of collocation ponts
collocationPnts = getGaussLobatto(k,N)
#print(getWeights(collocationPnts[1],collocationPnts,2,k))
D_1,D_2 = get2nd_Derivative(collocationPnts)
print(D_1)

#C_mat = np.zeros()



