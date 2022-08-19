import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import math
from inputs import *
from init import *
import matrix

def MatrixSolve_SC(H_func,psiArr,psiStarArr):
    H = H_func(psiArr,psiStarArr,mass,alpha,q)
    
    energies,evects = sci.linalg.eigh(H,subset_by_index = [0,0])
    evects = evects.T
    
    norm = 1/(np.linalg.norm(np.dot(evects[0],evects[0])))
    conj_evect = np.conjugate(evects[0])
    
    for l in range(niter): 
        H = H_func(evects[0],conj_evect,mass,alpha,q)
        energies,evects = sci.linalg.eigh(H,subset_by_index = [0,0])
        #energies,evects = sci.linalg.eig(H)
        evects = evects.T
        norm = 1.0/np.linalg.norm(evects[0])
        conj_evect = np.conjugate(evects[0])
        print(f'Energies {l}: {energies[0]}')
    norm = 1/(np.linalg.norm(evects[0]))
    evects[0] = evects[0]*norm
    return energies[0],evects[0]

def prop(psi,H,dt):
    #action of propagator on psi
    dpsi = np.zeros(psi.shape,dtype='complex')
    for k in range(prop_order):
        dpsi += ((-1.0j *dt)**k / math.factorial(k)) * np.matmul(np.linalg.matrix_power(H, k),psi)
    norm = 1.0/np.linalg.norm(dpsi)
    dpsi = dpsi*norm
    return dpsi

                        
                        
                        
                        