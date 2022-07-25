import numpy as np
import scipy as sci
import math

from inputs import *
from init import *
import matrix

def MatrixSolve_SC(H,name='GP1'):
    energies,evects = sci.linalg.eigh(H,subset_by_index = [0,1])
    evects = evects.T
    norm = 1/(np.linalg.norm(np.dot(evects[0],evects[0])))
    conj_evect = np.conjugate(evects[0])
    
    for l in range(niter): 
        if name=='GP1':
            H = matrix.construct_H_GP(evects[0],conj_evect,mass,alpha,q)
        elif name=='GP2':
            H = matrix.construct_H_GP_quartic(evects[0],conj_evect,mass,alpha,q)
        else:
            print('need a good name. wtf')
        energies,evects = sci.linalg.eigh(H,subset_by_index = [0,1])
        evects = evects.T
        norm = 1/(np.linalg.norm(np.dot(evects[0],evects[0])))
        conj_evect = np.conjugate(evects[0])
        print(f'Energies {l}: {energies}')
    norm = 1/(np.linalg.norm(evects[0]))
    evects[0] = evects[0]*norm
    return energies[0],evects[0]

def timeEvolve(psi,H):
    dpsi = np.zeros(psi.shape)
    for k in range(prop_order):
        dpsi += (-1.0j * delta_t)**k / math.factorial(k) * np.matmul(np.linalg.matrix_power(H, k),psi) 
    return dpsi