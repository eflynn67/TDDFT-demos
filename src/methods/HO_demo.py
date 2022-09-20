import numpy as np
import matplotlib.pyplot as plt
import spec

N = 200

interval = (-10,10)
CPnts = spec.GaussLobatto(interval=interval).chebyshev(N)

transform = spec.coord_transforms(interval=interval)
CPnts_unit = transform.affine(CPnts)
print(CPnts_unit)

D = spec.DerMatrix(interval)
D_1 = D.getCheb(CPnts_unit)
D_2 = np.matmul(D_1,D_1)
alpha1 = D.alpha1
L = -D_2/(alpha1**2) + .5*np.diag(CPnts**2)
L = np.delete(L,0,0)
L = np.delete(L,-1,-1)
L = np.delete(L,0,-1)
L = np.delete(L,-1,0)

evals,evects = np.linalg.eigh(L)

print(evals)

plt.plot(CPnts[1:N],evects[:,2],label='Numerical N='+str(N))
plt.xlim([-10,10])
plt.legend()
plt.show()