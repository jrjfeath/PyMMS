import os
import sys
import time
import yaml

from PyMMS_Functions import pymms

#Before we proceed check if we can load Parameter file
#Directory containing this file
fd = os.path.dirname(__file__)
try:
    with open(f'{fd}/PyMMS_Defaults.yaml', 'r') as stream:
        defaults = yaml.load(stream,Loader=yaml.SafeLoader)
except FileNotFoundError:
    print("Cannot find the Parameters file (PyMMS_Defaults.yaml), where did it go?")
    sys.exit()

#Create PyMMS object
pymms = pymms()
#Create operation modes, these could change in the future
defaults = pymms.operation_modes(defaults)
#Connect to the camera
pymms.init_device()
#Send Hardware, DAC Settings, and Control settings
pymms.hardware_settings(defaults)
#Set MSB and LSB to non-zero values and resend control values
defaults['ControlSettings']['iCompTrimMSB_DAC'] = [162,[42]]
defaults['ControlSettings']['iCompTrimLSB_DAC'] = [248,[43]]
pymms.program_bias_dacs(defaults)
#Set correct values since camera is loaded with 0 values initially
defaults['dac_settings']['iSenseComp'] = 1204
defaults['dac_settings']['iTestPix'] = 1253
pymms.dac_settings(defaults)
#Disconnect from the camera
pymms.close_device()


