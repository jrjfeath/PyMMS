import os
import sys
import time
import yaml

from PyMMS_Functions import pymms

pymms = pymms()
if pymms.error == 1: sys.exit() #Kill the process if unable to find dll

ret = pymms.init_device()
if ret != 0:
    print('Cannot connect to camera, check USB connection!')
    sys.exit()

#Before we proceed check if we can load Parameter file
#Directory containing this file
fd = os.path.dirname(__file__)
try:
    with open(f'{fd}/PyMMS_Defaults.yaml', 'r') as stream:
        defaults = yaml.load(stream,Loader=yaml.SafeLoader)
except FileNotFoundError:
    print("Cannot find the Parameters file (PyMMS_Defaults.yaml), where did it go?")
    sys.exit()

#Obtain the hardware settings for the PIMMS (hex -> binary), decode("latin-1") for str
for name, details in defaults['HardwareInitialization'].items():
    byte = bytes.fromhex(details[0])
    if len(details) == 2:
        ret, dat = pymms.writeread_device(byte,details[1])
    else:
        ret, dat = pymms.writeread_device(byte,details[1],details[2])
    if ret != 0:
        print(f'Could not write {name}, have you changed the value?')
        sys.exit()
    print(f'Setting: {name}, Sent: {byte}, Returned: {dat}')

#Combine the DAC settings to form the initialization string for PIMMS (int -> hex)
dac_hex = '#PC'+''.join([format(x,'X').zfill(4) for x in defaults['dac_settings'].values()])
ret, dat = pymms.writeread_device(dac_hex,len(dac_hex))
if ret != 0:
    print(f'Could not write DAC settings, have you changed a value?')
    sys.exit()
print(f'Setting: DAC, Sent: {dac_hex}, Returned: {dat}')

'''
ret, dat = pymms.writeread_device('#0@1F6E\r',7)
print(ret, dat)
time.sleep(0.1)
ret, dat = pymms.writeread_device('#0A\r',5)
print(ret, dat)
'''

ret = pymms.close_device()
if ret != 0:
    print('Cannot connect to camera, check USB connection!')
    sys.exit()