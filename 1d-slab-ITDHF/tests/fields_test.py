import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../src')
import fields
from init import *

VArr = fields.gaussian(grid)
## check x = 0 
V_0 = fields.gaussian(0.0)
V_0_exact = -0.397472
print('Numerical V(0):',V_0)
print('Analytic V(0):',V_0_exact)
plt.plot(grid,VArr)
plt.show()


