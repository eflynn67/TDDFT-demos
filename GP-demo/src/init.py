import numpy as np 
from inputs import *
### initilization file for arrays that are used throughout the package
#exec(open("~/TDDFT-demos/1d-TDHF/src/inputs.py").read())
e2 = 1.4399784 # e^2 charge in MeV fm
hb2m0 = 20.735530

#spin = np.array([.5,-.5])
grid = np.arange(lb,rb+step_size,step_size)
largegrid = np.arange(lb,rb+step_size,step_size)
t_steps = np.linspace(0,2,nt_steps)
nbox = len(grid)