import os
import sys
import time

import yaml
import numpy as np
import matplotlib.pyplot as plt

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

print('Program initial voltages')
#Send Hardware, DAC Settings, and Control settings
pymms.hardware_settings(defaults)

print('Program running voltages')
#Set MSB and LSB to non-zero values and resend control values
defaults['ControlSettings']['iCompTrimMSB_DAC'] = [162,[42]]
defaults['ControlSettings']['iCompTrimLSB_DAC'] = [248,[43]]
pymms.program_bias_dacs(defaults)

#Set correct values since camera is loaded with 0 values initially
defaults['dac_settings']['iSenseComp'] = 1204
defaults['dac_settings']['iTestPix'] = 1253
pymms.dac_settings(defaults)

print('Start  up')
#Now that parameters are set start-up camera for image acquisition
pymms.send_operation(defaults['operation_hex']['Start Up'])

#Make a default 15 intensity trim file for testing
trim = pymms.write_trim()
pymms.send_trim_to_pimms(trim)

#Set camera to take analogue picture along with exp bins
print('Grab frame')
pymms.send_operation(defaults['operation_hex']['Experimental w. Analogue Readout'])
pymms.writeread_str(['#1@0001\r'])

#Set timeout for reading from camera
pymms.setTimeOut()

#Get image from camera
test = pymms.readImage()
for i in range(5):
    img = test[i]
    #The analogue image comes back with massive intensities
    if i == 0:
        img = img - np.min(img)
    img = ((img / np.max(img)) * 255)
    plt.imshow(img.astype(np.int32), cmap='gray', vmin=0, vmax=255)
    plt.show()

#Disconnect from the camera
pymms.close_device()


