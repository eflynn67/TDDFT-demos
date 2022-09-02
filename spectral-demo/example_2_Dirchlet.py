import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_chebyt
'''
Here we cover two examples:

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




collocationPnts = getGaussLobatto(N,interval=interval)

D_1 = getDerMatrix(collocationPnts)
D_2 = np.matmul(D_1,D_1) # CAREFUL: this doesn't always hold.

I = np.identity(N+1)
# Now that we have the derivative matrix, we just need to construct the matrix 
# for the ODE
L = D_2 + D_1*collocationPnts 

# construct force function
F = collocationPnts - y0*L[:,0] - y1*L[:,-1]


L = np.delete(L,0,0)
L = np.delete(L,-1,-1)
L = np.delete(L,0,-1)
L = np.delete(L,-1,0)

sol = np.linalg.solve(L,F[1:N])
sol = np.concatenate([[y0],sol,[y1]])


plt.plot(collocationPnts,sol,label='Numerical N='+str(N))

plt.title('Problem 1')
plt.legend()
plt.show()

##############################################################################
### Part 2

interval = (0,1)
## boundary values y(-1) = y0 = a and y(1)=y1 = b
y0 = 0
y1 = 0

alpha = .2 # mixing parameter

nIters = 10

collocationPnts = getGaussLobatto(N,interval=interval)

D_1 = getDerMatrix(collocationPnts)
D_2 = np.matmul(D_1,D_1) # CAREFUL: this doesn't always hold.

I = np.identity(N+1)
# Now that we have the derivative matrix, we just need to construct the matrix 
# for the ODE
L = D_2 
# construct force function
F = np.zeros((nIters+1,len(L)))
sol_n = np.zeros((nIters+1,len(L)))
sol_n[0] = collocationPnts*(collocationPnts - 1)
F[0] = 6*sol_n[0]**2 + collocationPnts - y0*L[:,0] - y1*L[:,-1]


L_buffer = L.copy()
L_buffer = np.delete(L_buffer,0,0)
L_buffer = np.delete(L_buffer,-1,-1)
L_buffer = np.delete(L_buffer,0,-1)
L_buffer = np.delete(L_buffer,-1,0)
for i in np.arange(0,nIters,1):
    sol_buf = np.linalg.solve(L_buffer,F[i][1:N])
    sol_buf = np.concatenate([[y0],sol_buf,[y1]])
    sol_n[i+1] = alpha*sol_n[i] + (1-alpha)*sol_buf
    
    F[i+1] =   6*sol_n[i+1]**2 + collocationPnts - y0*L[:,0] - y1*L[:,-1]
    if i != 0:
        print('Error:',np.linalg.norm(sol_n[i+1]-sol_n[i]))



plt.plot(collocationPnts,sol,label='Problem 1 Numerical N='+str(N))
plt.plot(collocationPnts,sol_n[-1],label=' Problem 2 Numerical N='+str(N))
plt.title('Problem 2')
plt.legend()
plt.show()
