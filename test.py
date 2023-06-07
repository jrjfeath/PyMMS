import os
import yaml
from PyMMS_Functions import pymms
from PyQt6 import uic, QtWidgets, QtCore, QtGui

#Before we proceed check if we can load Parameter file
#Directory containing this file
fd = os.path.dirname(__file__)
try:
    with open(f'{fd}/PyMMS_Defaults.yaml', 'r') as stream:
        defaults = yaml.load(stream,Loader=yaml.SafeLoader)
except FileNotFoundError:
    print("Cannot find the Parameters file (PyMMS_Defaults.yaml), where did it go?")
    exit()

#Create PyMMS object
pymms = pymms()

#Create operation modes, these could change in the future
defaults = pymms.operation_modes(defaults)

print(pymms.idflex.open_dll())
print(pymms.idflex.init_device())
print(pymms.idflex.readImage(5))