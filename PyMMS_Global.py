import os
import yaml
from PyMMS_Functions import pymms

#Directory containing this file
fd = os.path.dirname(__file__)
#Directory containing dlls needed to run the camera
os.add_dll_directory(fd)

try:
    with open(f'{fd}/PyMMS_Defaults.yaml', 'r') as stream:
        defaults = yaml.load(stream,Loader=yaml.SafeLoader)
except FileNotFoundError:
    print("Cannot find the Parameters file (PyMMS_Defaults.yaml), where did it go?")
    exit()

#Create PyMMS object
pymm = pymms()
#Connect to camera
pymm.idflex.open_dll()
pymm.idflex.init_device()
#Send GlobalInitialize command to camera
byte = (bytes.fromhex('23 FF E3 0D')).decode('latin-1')
print(byte)
ret, dat = pymm.idflex.writeread_device(byte,0,1000)
print(ret,dat)
if ret != 0:
    print('Could not connect to camera, is it plugged in?')
pymm.idflex.close_device()
