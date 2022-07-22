import numpy as np
import wf
import densities as dens
import static
import dynamic
from init import *
from inputs import *


#################################
# Initialize the single particle wavefunctions
#################################

psiArr,energiesArr = wf.initWfs(name='HO')
rhoArr = rho(psiArr,np.conjugate(psiArr),grid)



###############################################################################
# STATIC SOLVE. This section we solve the TISE for the evals and evects at t =0
###############################################################################

#################################
# Main Loop over all the single particle wavefunctions 
#################################

for q in range(2):
    for n in range(nmax+1):
        for s in range(len(spin)):
            psi = psiArr[q][n][s]