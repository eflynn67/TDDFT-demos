import numpy 
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../src')
import wf
from init import *
from inputs import *

psi1 = wf.getHO(2, .5)
plt.plot(grid,psi1(grid))
plt.show()