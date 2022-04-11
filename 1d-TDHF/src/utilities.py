import numpy as np

def getGrid(r0,r1,h=10**-2):
    grid = np.arange(r0,r1+h,h)
    return grid 