import os
import sys
import time
import numpy as np
from Classes.PyMMS_trim_data import TrimData

#########################################################################################
#This file is used to test the UI without actually communicating with PImMS
#All communication functions have been replaced with dummy functions 
#########################################################################################

#Directory containing this file
fd = os.path.dirname(__file__)
fd = f'{os.path.dirname(fd)}//DLLs'
#Directory containing dlls needed to run the camera
if sys.platform == "win32": os.add_dll_directory(fd)

class idflexusb():
    '''
    Controls interactions with the 64bit idFLEX_USB shared library (.dll)
    
    Functions should not be altered unless instructions have been given by aSpect.
    
    If camera id is required call func.camera_id
    '''

    def __init__(self) -> None:
        self.camera_id = 0
        self.message : str = f""
        self.error : int = 0

    def init_device(self) -> None:
        '''
        Return of 0 means the camera successfully connected \n
        Return of 7 means there was an error connection to the camera
        '''
        self.message = f'Device connected.'

    def close_device(self) -> None:
        '''
        Return of 0 means the camera successfully disconnected\n
        Return of anything else means there was an error disconnecting camera
        '''
        self.message = f'Device disconnected.'

    def writeread_device(self,data,bytestoread,timeout=1000,sleep=0.01) -> None:
        '''
        Write data to the PIMMS camera.
        Format: data [string], byte size [int], Timeout(ms) [int]
        '''
        print(data)
        time.sleep(sleep) #Wait X ms between each send

    def setAltSetting(self,altv=0) -> None:
        '''
        Writing trim data requires the camera to be set to alt = 1.
        '''
        self.message = f'Set alt setting.'

    def setTimeOut(self,eps='0x82') -> None:
        '''
        Set camera timeout for taking images. 
        
        EPS is changed when writing trim and taking images.
        '''
        self.message = f'Timeout set.'

    def write_trim_device(self,trim,eps='0x2') -> None:
        '''
        Write trim data to the PIMMS camera. Used for writing trim data.
        Format:  trim data [array]
        '''
        self.message = f'Trim data uploaded.'

    def readImage(self,size=5) -> np.ndarray:
        '''
        Read image array off camera. Array is columns wide by number of outputs
        multiplied by the number of rows. i.e 324*324 experimental (4) would be
        (324,1296).
        '''

        #Get the images as a numpy array and then reshape
        samples = [0, 50, 100, 150, 200, 255]
        probablility = [0.9, 0.04, 0.025, 0.02, 0.01, 0.005]
        img  = np.random.choice(samples, size=(size,324,324), p = probablility)

        return img

class pymms():
    '''
    Object for communicating with PIMMS camera.

    Functions parse data to be passed along to idflexusb dll.
    '''

    def __init__(self, settings) -> None:
        self.idflex = idflexusb()
        #Create operation modes, these could change in the future
        self.settings : dict = self.operation_modes(settings)

    def operation_modes(self,settings) -> dict:
        operation_hex = {}
        for key, value in settings['OperationModes'].items():
            hexes = ['#1@0004\r']
            reg = 26 #The first register value is always 26
            for subroutine in value:
                adr, regn = list(settings['SubRoutines'].values())[subroutine]
                res = reg << 8 | adr
                reg = regn
                hexes.append(f'#1@{hex(res)[2:].zfill(4)}\r')
            operation_hex[key] = hexes
        settings['operation_hex'] = operation_hex
        return settings

    def writeread_str(self,hex_list = []) -> None:
        '''
        Function takes a list and writes data to camera.
        '''
        for hexs in hex_list:
            self.idflex.writeread_device(hexs,len(hexs))

    def dac_settings(self):
        '''
        Combine the DAC settings to form the initialization string for PIMMS (int -> hex)
        Called whenever vThN & vThP are changed
        '''
        dac_hex = '#PC'+''.join([format(x,'X').zfill(4) for x in self.settings['dac_settings'].values()])+'\r'
        self.writeread_str([dac_hex])

    def program_bias_dacs(self):
        '''
        Programm the PIMMS2 DACs
        '''
        hex_str = self.settings['operation_hex']['Programme PImMS2 Bias DACs']
        for data in self.settings['ControlSettings'].values():
            value  = data[0]
            reg = data[1]
            if len(reg) == 1:
                res = (reg[0] << 8) | value
                hex_str.append(f'#1@{hex(res)[2:].zfill(4)}\r')
            else:
                q, r = divmod(value, 256) #Calculate 8 bit position
                hi = (reg[0] << 8) | q
                lo = (reg[1] << 8) | r
                hex_str.append(f'#1@{hex(hi)[2:].zfill(4)}\r')
                hex_str.append(f'#1@{hex(lo)[2:].zfill(4)}\r')
        return self.writeread_str(hex_str)
    
    def turn_on_pimms(self):
        '''
        Send PIMMS the initial start-up commands.

        Defaults are read from the PyMMS_Defaults.

        All important voltages are initially set to 0mV.
        '''
        #Connect to the camera
        self.idflex.init_device()

        print('Test')

        #Obtain the hardware settings for the PIMMS (hex -> binary), decode("latin-1") for str
        for name, details in self.settings['HardwareInitialization'].items():
            byte = (bytes.fromhex(details[0])).decode('latin-1')
            if len(details) == 2:
                self.idflex.writeread_device(byte,details[1])
            else:
                self.idflex.writeread_device(byte,details[1],details[2])
                if name == 'GlobalInitialize':
                    time.sleep(3)

        #Program dac settings
        self.dac_settings()

        #Program control settings
        self.program_bias_dacs()

        #Write stop header at end
        self.writeread_str(['#1@0001\r'])

        #If all connection commands successful return 0
        self.idflex.message = f'Connected to PIMMS!'

    def send_trim_to_pimms(self,trim):
        '''
        Sends trim data to camera.
        '''
        #Write the stop command to PIMMS
        self.writeread_str(['#1@0000\r'])

        #Change the camera to setting 1
        self.idflex.setAltSetting(altv=1)

        #Tell camera that we are sending it trim data
        self.writeread_str(['#0@0D01\r','#1@0002\r'])

        #Set timeout for reading the trim file
        self.idflex.setTimeOut(eps='0x2')
        
        #Send trim data to camera.
        self.idflex.write_trim_device(trim)
        
        #Tell camera to stop expecting trim data
        self.writeread_str(['#1@0000\r','#0@0D00\r'])

        #Change the camera to setting 0
        self.idflex.setAltSetting(altv=0)

        #Write stop header at end
        self.writeread_str(['#1@0001\r'])

        #If no errors return pass
        self.idflex.message = f'Trim data sent!'

    def send_output_types(self,function=0,trigger=0,rows=5):
        #Set camera to take analogue picture along with exp bins
        if function == 1:
            self.writeread_str(self.settings['operation_hex']['Experimental w. Analogue Readout'])
        #Set camera to take experiment bins only
        else:
            self.writeread_str(self.settings['operation_hex']['Experimental'])
        #0001 is free runnning, and 0081 is triggered
        if trigger == 0:
            self.writeread_str(['#1@0001\r'])
        else:
            self.writeread_str(['#1@0081\r'])

        #Set timeout for reading from camera
        self.idflex.setTimeOut()

        self.idflex.message = f'Updated camera view.'

    def start_up_pimms(self,trim_file="",function=0,trigger=0,rows=5):
        '''
        This function sends the updated DAC and start-up commands to PIMMS.

        The order of operations are IMPORTANT do not change them.
        '''

        # Set correct values since camera is loaded with 0 values initially
        self.settings['dac_settings']['iSenseComp'] = 1204
        self.settings['dac_settings']['iTestPix'] = 1253

        # Program dac settings
        self.dac_settings()

        #S end command strings to prepare PIMMS for image acquisition
        self.writeread_str(self.settings['operation_hex']['Start Up'])

        # Set MSB and LSB to non-zero values and resend control values
        self.settings['ControlSettings']['iCompTrimMSB_DAC'] = [162,[42]]
        self.settings['ControlSettings']['iCompTrimLSB_DAC'] = [248,[43]]
        self.program_bias_dacs()

        # After these commands are sent we now send the trim file to PIMMS
        # Generate an empty trim array
        trim = TrimData.write_trim(value=0)
        if (os.path.isfile(trim_file) & trim_file.endswith('.bin')):
            trim = TrimData.read_trim(trim_file)
        if (os.path.isfile(trim_file) & trim_file.endswith('.csv')):
            trim = TrimData.write_trim(filename=trim_file)
        
        self.send_trim_to_pimms(trim)
        self.send_output_types(function,trigger)

        #If all DAC setting and startup commands successful
        self.idflex.message = f'Updated PIMMS DACs, trim, and readout!'

    def calibrate_pimms(self,update=False,vThN=450,vThP=450,value=15,iteration=0) -> None:
        '''
        This function controls calibration of the camera, updating the pixel mask and trim values. 
        
        It can optionally select experiment mode and update the voltages if required.
        '''
        # Update with new threshold values
        if update:
            self.settings['dac_settings']['vThN'] = vThN
            self.settings['dac_settings']['vThP'] = vThP
            self.dac_settings()
            self.send_output_types(1,0,4)
        else:
            # Write trim data
            trim = TrimData.write_trim(value=value,iteration=iteration,calibration=True)
            self.send_trim_to_pimms(trim)
            self.writeread_str(['#À@0005\r'])