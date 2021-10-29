import os, sys
#Taken from https://stackoverflow.com/a/49375740. Modeled off of PyNEB`
dirName = os.path.dirname(os.path.realpath(__file__))
if dirName not in sys.path:
    sys.path.append(dirName)

from SE_solvers import *
from DFT_solvers import *
from utilities import *
