import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.special import assoc_laguerre
sys.path.insert(0, '../src')
import wf
import utilities 
R_p = .831
r_array = utilities.getGrid(0, 10)
psi = wf.initWf(name='hydrogen',n=1,l=0)
norm = np.linalg.norm(psi(r_array))
plt.plot(r_array/R_p,psi(r_array)/norm)
plt.show()



