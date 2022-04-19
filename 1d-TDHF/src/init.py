import numpy as np 
from inputs import *
### initilization file for arrays that are used throughout the package
#exec(open("~/TDDFT-demos/1d-TDHF/src/inputs.py").read())
e2 = 1.4399784 # e^2 charge in MeV fm
mp = 938.28 # mass of proton in MeV/c^2
mn = 939.57## mass of neutron in MeV/c^2
spin = np.array([.5,-.5])
nt = N + Z
vws = np.zeros(2)
vws[0] = -51.+33.*(N-Z)/nt
vws[1] = -51.-33.*(N-Z)/nt
grid = np.arange(lb,rb+step_size,step_size)
