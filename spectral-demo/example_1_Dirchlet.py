import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_chebyt
'''
This is a numerical implementation of the worked example from pg 1094 of 
Numerical Recipes with some additional material. 
Here we solve the ODE

y'' + y' -2y + 2 = 0 

on various intervals and Dirichlet BCs. 
This is just practice implementing a pseudo-spectral method to solve a simple ODE.
We use Chebyshev polynomials as a basis.

Since we have Dirchlet BCs, we use Gauss-Lobbato extrema points as the collocation 
points and we use the "boundary bordering" technique.

The solution is assumed to be of the form 

y(x) = \sum_{n}^{N} y(x_{n})T_{n}(x)

where x_{n} are the extremea Gauss-Lobbato points of the Chebyshev polynomials

1) Solve on the interval [-1,1] with Dirichlet BCs y(-1) = y(1) = 0.

2) We play with different regions to see how the chebyshev polynomials work on
intervals outside [-1,1] where they are defined. We take the domain [0,2].

We map the domain [0,2] to [-1,1] via an affine transformation.
BCs on the boundary are y(0) = y(2) = 0.

3) Try [-2,4] with y(-2) = 2,  y(4) = 5.

'''

def exact_sol1(x):
    '''
    Exact solution to the ODE in example 1. From Numerical Recipes.
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
    '''
    Exact solution for ODE example 3. Found using Mathematica
    '''
    result = (1 - np.exp(18) + np.exp(2+x) - 4*np.exp(14+x) - np.exp(8-2*x)*(np.exp(6)-4))/(1 - np.exp(18))
    return result

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
h = .1
grid = np.arange(-1,1+h,h) 
N = 20 # number of basis functions .

##############################################################################
### Part 1

interval = (-1,1)
## boundary values y(-1) = y0 = a and y(1)=y1 = b
y0 = 0
y1 = 0




collocationPnts = getGaussLobatto(N,interval=interval)

D_1 = getDerMatrix(collocationPnts)
D_2 = np.matmul(D_1,D_1) # CAREFUL: this doesn't always hold.

I = np.identity(N+1)
# Now that we have the derivative matrix, we just need to construct the matrix 
# for the ODE
L = D_1 + D_2 -2*I 

# construct force function
F = np.full(len(L),-2) - y0*L[:,0] - y1*L[:,-1]


L = np.delete(L,0,0)
L = np.delete(L,-1,-1)
L = np.delete(L,0,-1)
L = np.delete(L,-1,0)

sol = np.linalg.solve(L,F[1:N])
sol = np.concatenate([[y0],sol,[y1]])
error = np.abs(sol - exact_sol1(collocationPnts))
print('Problem 1 Max error:',max(error))


plt.plot(collocationPnts,sol,label='Numerical N='+str(N))
plt.plot(grid,exact_sol1(grid),label='Exact')
plt.title('Problem 1')
plt.legend()
plt.show()

##############################################################################
### part 2 
interval = (0,2)
grid = np.arange(interval[0],interval[1]+h,h)
## boundary values y(-1) = y0 = a and y(1)=y1 = b
y0 = 0
y1 = 0
collocationPnts = getGaussLobatto(N,interval=interval)
## Rescale so the points are in [-1,1]
alpha1 = .5*abs(max(collocationPnts) - min(collocationPnts))
alpha2 = .5*abs(max(collocationPnts) + min(collocationPnts))
#transform the coordinates 
collocationPnts = collocationPnts/alpha1  - alpha2/alpha1

D_1 = getDerMatrix(collocationPnts)
D_2 = np.matmul(D_1,D_1)

I = np.identity(N+1)
# Now that we have the derivative matrix, we just need to construct the matrix 
# for the ODE

L = D_1/alpha1 + D_2/alpha1**2 -2*I

# construct F
F = np.full(len(L),-2) - y0*L[:,0] - y1*L[:,-1]

# enforce BCs by removing first and last rows and columns. This is the boundary
# bordering technique.
L = np.delete(L,0,0)
L = np.delete(L,-1,-1)
L = np.delete(L,0,-1)
L = np.delete(L,-1,0)



sol = np.linalg.solve(L,F[1:N])
# rescale the solution and coordinates
collocationPnts = collocationPnts*alpha1 + 1
sol = np.concatenate([[y0],sol,[y1]])
error = np.abs(sol - exact_sol2(collocationPnts))
print('Problem 2 Max error:', max(error))


plt.plot(collocationPnts,sol,label='Numerical N='+str(N))
plt.plot(grid,exact_sol2(grid),label='Exact')
plt.legend()
plt.title('Problem 2')
plt.show()

##############################################################################
# Part 3

interval = (-2,4)
grid = np.arange(interval[0],interval[1]+h,h)
## boundary values y(-1) = y0 = a and y(1)=y1 = b
y0 = 2
y1 = 5
collocationPnts = getGaussLobatto(N,interval=interval)
## Rescale so the points are in [-1,1]
alpha1 = .5*abs(max(collocationPnts) - min(collocationPnts))
alpha2 = .5*abs(max(collocationPnts) + min(collocationPnts))
#transform the coordinates 
collocationPnts = collocationPnts/alpha1  - alpha2/alpha1

D_1 = getDerMatrix(collocationPnts)
D_2 = np.matmul(D_1,D_1)

I = np.identity(N+1)
# Now that we have the derivative matrix, we just need to construct the matrix 
# for the ODE

L = D_1/alpha1 + D_2/alpha1**2 -2*I

# construct F
F = np.full(len(L),-2) - y0*L[:,0] - y1*L[:,-1]

# enforce BCs by removing first and last rows and columns. This is the boundary
# bordering technique.
L = np.delete(L,0,0)
L = np.delete(L,-1,-1)
L = np.delete(L,0,-1)
L = np.delete(L,-1,0)



sol = np.linalg.solve(L,F[1:N])
# rescale the solution and coordinates
collocationPnts = collocationPnts*alpha1 + 1
sol = np.concatenate([[y0],sol,[y1]])

error = np.abs(sol - exact_sol3(collocationPnts))
print('Problem 3 Max Error:', max(error))


plt.plot(collocationPnts,sol,label='Numerical N='+str(N))
plt.plot(grid,exact_sol3(grid),label='Exact')
plt.legend()
plt.title('Problem 3')
plt.show()
