import numpy as np 
from inputs import *
### initilization file for arrays that are used throughout the package
#exec(open("~/TDDFT-demos/1d-TDHF/src/inputs.py").read())
e2 = 1.4399784 # e^2 charge in MeV fm
hb2m0 = 20.735530
Rp = 1.0 #0.831 # radius of the proton in fm. used for density saturation density

spin = np.array([.5,-.5])
nt = N + Z
vws = np.zeros(2)
vws[0] = -51.+33.*(N-Z)/nt
vws[1] = -51.-33.*(N-Z)/nt
grid = np.arange(lb,rb+step_size,step_size)
Rp_ind = np.where(grid <= Rp)[0][-1]
nbox = len(grid)