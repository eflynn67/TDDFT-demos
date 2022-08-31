import numpy as np
import matplotlib.pyplot as plt

'''
1) This is a numerical implementation of the worked example from pg 1094 of 
Numerical Recipes but with Periodic boundary conditions. 
Here we solve the ODE

y'' + y' -2y + 2 = 0 

on the interval [-1,1] with periodic BCs y(-1) = y(1).
Since we have periodic BCs we use the Fourier basis.

Collocation points of the Fourier Series are given by analytic expression found on .

2) we play with different regions to see how the chebyshev polynomials work on
intervals besides [-1,1] where they are defined. We take the domain [0,2].

we take the same BCs on the boundary y(0) = y(2) = 0.

3) Try [-2,4] with y(-2) = y(4) = 0.

'''

def exact_sol1(x):
    '''
    Exact solution to the ODE in example 1.
    '''
    result = 1.0 - (1.0/np.sinh(3))*(np.exp(x)*np.sinh(2) + np.exp(-2*x)*np.sinh(1))
    return result

def exact_sol2(x):
    '''
    Exact solution to the ODE example 2. Found exact sol with mathematica
    '''
    result = (1.0 + np.exp(2) + np.exp(4) - np.exp(4-2*x)-np.exp(x)*(1+np.exp(2))) \
    * (1.0/(1.0 + np.exp(2) + np.exp(4)))
    return result

def exact_sol3(x):
    result = (1.0 + np.exp(6) + np.exp(12) - np.exp(8-2*x)-np.exp(2+x)*(1+np.exp(6))) \
    * (1.0/(1.0 + np.exp(6) + np.exp(12)))
    return result

def getGaussLobatto(NcPnts,N,interval=(-1,1)):
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
        GSPnts[l] = .5*(interval[0] + interval[1]) - .5*(interval[1] -interval[0]) \
            *np.cos(l*np.pi/N)
    return GSPnts

def c_coeff(l,N):
    if l == 0 or l == N:
        return 2.0
    else: 
        return 1.0
def getDerMatrix(N,CPnts):
    '''
    Analytic solution taken from Spectral Methods in Fluid Dynamics (1988) pg 69 (or 84 in pdf).
    Assumes you are taking the collocation points at the Gauss-Lobatto points
    
    i labels the collocation point, j labels C
    Parameters
    ----------
    N : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
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
h = .1

grid = np.arange(-1,1+h,h) 
N = 200 # number of knots.
k = 201 # number of collocation ponts