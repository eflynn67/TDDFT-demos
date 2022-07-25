import numpy as np
import scipy as sci

from inputs import *
from init import *
from matrix import construct_H_GP
def MatrixSolve_SC(H):
    energies,evects = sci.linalg.eigh(H,subset_by_index = [0,1])
    evects = evects.T
    norm = 1/(np.linalg.norm(np.dot(evects[0],evects[0])))
    conj_evect = np.conjugate(evects[0])
    
    for l in range(niter): 
        H = construct_H_GP(evects[0],conj_evect,mass,alpha,q)
        energies,evects = sci.linalg.eigh(H,subset_by_index = [0,1])
        evects = evects.T
        norm = 1/(np.linalg.norm(np.dot(evects[0],evects[0])))
        conj_evect = np.conjugate(evects[0])
        print(f'Energies {l}: {energies}')
    norm = 1/(np.linalg.norm(evects[0]))
    evects[0] = evects[0]*norm
    return energies[0],evects[0]