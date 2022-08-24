import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs
from scipy.linalg import toeplitz
from scipy.linalg import eig
from scipy.linalg import eigh
from scipy.linalg import svd
import matplotlib.pyplot as plt

## This is a demo solution of the 1-1 diffusion equation 
#### u_t +D u_xx = \lambda u
##with Dirchlet 
# D is some real number 
def exact_uk(t,n):
    return np.exp((2*np.pi*1j/T)*t)
def exact_lambda(n):
    return(2.0*np.pi*n/T)

D = -1.0
## boundary conditions on the spatial domain and periodic in time.
## keep this a square grid
L = 1
T = 1

h = .001

x_grid = np.arange(0,L,h)
nx = len(x_grid)

t_grid = np.arange(0,T,h)
nt = len(t_grid)


print(nx)
print(nt)
print('number of matrix entries: ',nx**2 * nt**2)

#First construct the A matrix for the second derivative in space
diag = np.ones([nx])
diags = np.array([diag,-2*diag,diag])

A = sparse.spdiags(diags,np.array([-1,0,1]),nx,nx).toarray()/h**2


# now construct the I matrix for the first order time derivative
I = np.identity(nx)
first_col = np.zeros(nx)
first_col[1] = 1
first_col[-1] = -1
first_row = np.zeros(nx)
first_row[1] = -1
first_row[-1] = 1

## Now make the block matrix. 
# Make the pattern we need for the time derivative with periodic BC
I_blocks = (D/(2.0*h))*toeplitz(first_col,first_row)
# add up the block matrices. The np.kron(I,A) puts the A matrix on the diagonal
# of the larger block matrix 
#L = np.kron(I_blocks,I) + np.kron(I,A)



# find evals and evects

"""
lambda_k,u_k = eig(I_blocks)
idx = np.imag(lambda_k).argsort()[::-1]  
lambda_k = lambda_k[idx]
u_k = u_k[:,idx]
mid = int(len(lambda_k)/2-1)
print(mid)
lc = u_k[:,mid+4] + u_k[:,mid+5]
plt.plot(x_grid,np.imag(lc))
"""

#start=1
#shift=2
#plt.plot(x_grid[start::shift],np.imag(u_k[start::shift,mid])    )
#plt.plot(x_grid[:],np.real(u_k[:,mid+2])    )
#plt.plot(x_grid,np.imag(u_k[:,mid+2])  )
#plt.plot(x_grid,np.imag(u_k[:,mid+4])  )
#plt.xlim([.1,.12])
plt.show()
#idx = np.real(lambda_k).argsort()[::-1]  
#lambda_k = lambda_k[idx]
#u_k = u_k[:,idx]
 
#np.savetxt('evect.txt',u_k,delimiter='\t',fmt='%.10f%+.10fj')
#np.savetxt('evals.txt',[np.real(lambda_k),np.imag(lambda_k)],delimiter=',')
# remember our evects are flattened u_{ij} matrices of the form U = (..,.u^{[j]},..)
# where j labels a fixed row.
# Reshape the solution to a 2-d grid
'''
u_array = []
for i in range(len(u_k)):
    u_array.append(u_k[:,i].reshape(nx,nt))

xx,tt = np.meshgrid(x_grid,t_grid)
#k = np.argmin(np.real(lambda_k))# select the kth eigenvector/eigenvalue
#m = 0 # select the mth time slice

for k in range(len(u_array)):
    plt.plot(x_grid,np.real(u_array[k][0]),label='real')
    plt.plot(x_grid,np.imag(u_array[k][-1]),label='last real')
    plt.ylim([-.5,.5])
    plt.legend()
    plt.title(str(k))
    plt.show()

plt.plot(x_grid,np.imag(u_array[29][0]),label='imag')
plt.plot(x_grid,np.imag(u_array[29][-1]),label='last imag')
plt.legend()
plt.show()

plt.contourf(xx,tt,np.real(u_array[29]))
plt.title('real part')
plt.show()

plt.contourf(xx,tt,np.imag(u_array[29]))
plt.title('imag part')
plt.show()
'''