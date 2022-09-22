import numpy as np
import sys
from inputs import *

### initilization file for arrays that are used throughout the package
#exec(open("~/TDDFT-demos/1d-TDHF/src/inputs.py").read())
e2 = 1.4399784 # e^2 charge in MeV fm
hb2m0 = 20.735530

#spin = np.array([.5,-.5])
step_size = .2
grid = np.arange(lb,rb+step_size,step_size)
nbox = len(grid)