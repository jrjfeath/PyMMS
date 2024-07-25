import ctypes
import os
import sys
import numpy as np
import time
from Classes.PyMMS_trim_data import TrimData
#Important notes about ctypes 
#Pointers are made using: ptr = ctypes.c_void_p()
#When you want to pass a pointer to a C function use ctypes.byref(ptr) in the function call

#DLL functions should have their arguments and return types defined
#func.restype = ctypes.c_int (i.e function returns an int)
#func.argtypes = [ctypes.POINTER(ctypes.c_void_p),ctypes.c_int] (takes a pointer and int)

#Directory containing this file
fd = os.path.dirname(__file__)
fd = f'{os.path.dirname(fd)}//DLLs'
#Directory containing dlls needed to run the camera
if sys.platform == "win32": os.add_dll_directory(fd)
dll_path = os.path.join(fd,'idFLEX_USB.dll')

class idflexusb():
    '''
    Controls interactions with the 64bit idFLEX_USB shared library (.dll)
    
    Functions should not be altered unless instructions have been given by aSpect.
    
    If camera id is required call func.camera_id
    '''

    def __init__(self) -> None:
        self.camera_id = ctypes.c_void_p() #Set camera value to none
        self.message : str = f""
        self.error : int = 0
        try:
            self.pimms = ctypes.cdll.LoadLibrary(dll_path)
            self.message = f'Connect to camera!'
        except FileNotFoundError:
            self.message = f'Cannot find: {dll_path}'
            self.error = 1

    def error_encountered(self, message = f"") -> None:
        '''
        If an error is encountered while running code closes connection to the camera.
        '''
        if self.camera_id is not None:
            self.close_device()
            self.camera_id = 0
            self.message = f'Error: {message}'
            self.error = 1

    def init_device(self) -> None:
        '''
        Return of 0 means the camera successfully connected \n
        Return of 7 means there was an error connection to the camera
        '''
        open_cam = self.pimms.Init_Device
        open_cam.restype = ctypes.c_int
        open_cam.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

        camera_id = ctypes.c_void_p()
        ret = open_cam(ctypes.byref(camera_id))
        self.camera_id = camera_id
        if ret != 0:
            self.error = 1
            self.message = f'Error: Cannot connect to camera, check USB connection!'
        time.sleep(0.5)

    def close_device(self) -> None:
        '''
        Return of 0 means the camera successfully disconnected\n
        Return of anything else means there was an error disconnecting camera
        '''
        close_cam = self.pimms.Close_Device
        close_cam.restype = ctypes.c_int 
        close_cam.argtypes = [ctypes.c_void_p]

        ret = close_cam(self.camera_id)
        self.error = ret
        self.message = f'Disconnected device : {ret}'
        print(self.message)

    def writeread_device(self,data,bytestoread,timeout=1000,sleep=0.05) -> None:
        '''
        Write data to the PIMMS camera.
        Format: data [string], byte size [int], Timeout(ms) [int]
        '''
        wr_cam = self.pimms.serialWriteRead

        data_in_ba = bytearray()
        data_in_ba.extend(map(ord, data))
        char_array = ctypes.c_char * len(data_in_ba)
        bytestowrite = ctypes.c_int32(ctypes.sizeof(char_array))
        timeout = ctypes.c_int32(timeout)

        wr_cam.restype = ctypes.c_int
        wr_cam.argtypes = [ctypes.c_void_p, 
                           ctypes.POINTER(ctypes.c_int32), 
                           ctypes.POINTER(ctypes.c_char * ctypes.sizeof(char_array)), 
                           ctypes.POINTER(ctypes.c_int32),  
                           ctypes.POINTER(ctypes.c_char * bytestoread), 
                           ctypes.c_int32] 

        data_out = ctypes.create_string_buffer(bytestoread)
        bytestoread = ctypes.c_int32(ctypes.sizeof(data_out))

        ret = wr_cam(self.camera_id, 
                     ctypes.byref(bytestowrite), 
                     char_array.from_buffer(data_in_ba), 
                     ctypes.byref(bytestoread), 
                     data_out, 
                     timeout)

        if len(data) > 200: 
            print(f'Sent trim, Returned: {ret}')
        else:
            print(f'{ret}, Sent: {data[:-1]}, Returned: {data_out.raw[:bytestoread.value]}')
        if ret != 0: self.error_encountered(f"Error writing data to camera.")
        time.sleep(sleep) #Wait X ms between each send

    def setAltSetting(self,altv=0) -> None:
        '''
        Writing trim data requires the camera to be set to alt = 1.
        '''

        alt = self.pimms.setAltSetting
        alt.restype = ctypes.c_int
        alt.argtypes = [ctypes.c_void_p, 
                        ctypes.c_uint8]

        value = ctypes.c_uint8(altv)

        ret = alt(self.camera_id, value)
        if ret != 0: self.error_encountered(f"Could not change camera register.")

    def setTimeOut(self,eps='0x82') -> None:
        '''
        Set camera timeout for taking images. 
        
        EPS is changed when writing trim and taking images.
        '''

        sto = self.pimms.setTimeOut
        sto.restype = ctypes.c_int
        sto.argtypes = [ctypes.c_void_p, 
                        ctypes.c_uint8,
                        ctypes.c_int32]

        ep = ctypes.c_uint8(int(eps,16))
        timeout = ctypes.c_int32(5000)

        ret = sto(self.camera_id, ep, timeout)

        if ret != 0: self.error_encountered(f"Could not set camera timeout for trim data.")

    def write_trim_device(self,trim,eps='0x2') -> None:
        '''
        Write trim data to the PIMMS camera. Used for writing trim data.
        Format:  trim data [array]
        '''

        w_cam = self.pimms.writeData

        ep = ctypes.c_uint8(int(eps,16))
        arr = ctypes.c_uint8 * trim.size #Make an empty array
        bytestowrite = ctypes.c_int32(ctypes.sizeof(arr)) #get size of array

        w_cam.restype = ctypes.c_int
        w_cam.argtypes = [ctypes.c_void_p,
                          ctypes.c_uint8,
                          ctypes.POINTER(ctypes.c_int32),
                          ctypes.POINTER(ctypes.c_uint8 * ctypes.sizeof(arr))]

        ret = w_cam(self.camera_id, ep, ctypes.byref(bytestowrite), arr.from_buffer(trim))

        if ret != 0: self.error_encountered(f"Could not send camera trim data.")

    def readImage(self,size=5) -> np.ndarray:
        '''
        Read image array off camera. Array is columns wide by number of outputs
        multiplied by the number of rows. i.e 324*324 experimental (4) would be
        (324,1296).
        '''

        arrayType = ((ctypes.c_uint16 * 324) * (324 * size))
        array = arrayType()
        buffer = ctypes.c_int32(ctypes.sizeof(array))

        rda = self.pimms.readDataAsync
        rda.restype = ctypes.c_int
        rda.argtypes = [ctypes.c_void_p, 
                        ctypes.POINTER(ctypes.c_int32), 
                        ctypes.POINTER(arrayType)]
        
        ret = rda(self.camera_id,
                  ctypes.byref(buffer),
                  array)

        #Get the images as a numpy array and then reshape
        img = np.ctypeslib.as_array(array)
        img = img.reshape(size,324,324)

        return img

class pymms():
    '''
    Object for communicating with PIMMS camera.

    Functions parse data to be passed along to idflexusb dll.
    '''

    def __init__(self, settings : dict) -> None:
        self.idflex = idflexusb()
        self.trim = TrimData()
        #Create operation modes, these could change in the future
        self.settings : dict = self.operation_modes(settings)

    def operation_modes(self,settings) -> dict:
        operation_hex = {}
        for key, value in settings['OperationModes'].items():
            hexes = ['#1@0000\r']
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
            if self.idflex.error != 0: return

    def dac_settings(self) -> None:
        '''
        Combine the DAC settings to form the initialization string for PIMMS (int -> hex)
        Called whenever vThN & vThP are changed
        '''
        dac_hex = '#PC'+''.join([format(x,'X').zfill(4) for x in self.settings['dac_settings'].values()])+'\r'
        self.writeread_str([dac_hex])

    def program_bias_dacs(self) -> None:
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
        self.writeread_str(hex_str)

    def turn_on_pimms(self) -> None:
        '''
        Send PIMMS the initial start-up commands.

        Defaults are read from the PyMMS_Defaults.

        All important voltages are initially set to 0mV.
        '''
        #Connect to the camera
        ret = self.idflex.init_device()
        if self.idflex.error != 0: return

        #Obtain the hardware settings for the PIMMS (hex -> binary), decode("latin-1") for str
        for name, details in self.settings['HardwareInitialization'].items():
            byte = (bytes.fromhex(details[0])).decode('latin-1')
            if len(details) == 2:
                self.idflex.writeread_device(byte,details[1])
            else:
                self.idflex.writeread_device(byte,details[1],details[2])
                if name == 'GlobalInitialize':
                    time.sleep(3)
                else:
                    time.sleep(0.1)
            if self.idflex.error != 0: return

        #Program dac settings
        self.dac_settings()
        if self.idflex.error != 0: return

        #Program control settings
        self.program_bias_dacs()
        if self.idflex.error != 0: return

        #Write stop header at end
        self.writeread_str(['#1@0001\r'])
        if self.idflex.error != 0: return

        #If all connection commands successful return 0
        self.idflex.message = 'Connected to PIMMS!'

    def send_trim_to_pimms(self,trim) -> None:
        '''
        Sends trim data to camera.
        '''
        #Write the stop command to PIMMS
        self.writeread_str(['#1@0000\r'])
        if self.idflex.error != 0: return 

        #Change the camera to setting 1
        self.idflex.setAltSetting(altv=1)
        if self.idflex.error != 0: return 

        #Tell camera that we are sending it trim data
        self.writeread_str(['#0@0D01\r','#1@0002\r'])
        if self.idflex.error != 0: return 

        #Set timeout for reading the trim file
        self.idflex.setTimeOut(eps='0x2')
        if self.idflex.error != 0: return 
        
        #Send trim data to camera.
        self.idflex.write_trim_device(trim)
        if self.idflex.error != 0: return 
        
        #Tell camera to stop expecting trim data
        self.writeread_str(['#1@0000\r','#0@0D00\r'])
        if self.idflex.error != 0: return 

        #Change the camera to setting 0
        self.idflex.setAltSetting(altv=0)
        if self.idflex.error != 0: return 

        #Write stop header at end
        self.writeread_str(['#1@0001\r'])
        if self.idflex.error != 0: return

    def send_output_types(self,function=0,trigger=0,rows=5) -> None:
        #Set camera to take analogue picture along with exp bins
        if function == 1:
            self.writeread_str(self.settings['operation_hex']['Experimental w. Analogue Readout'])
        #Set camera to take experiment bins only
        else:
            self.writeread_str(self.settings['operation_hex']['Experimental'])
        if self.idflex.error != 0: return
        #0001 is free runnning, and 0081 is triggered
        if trigger == 0:
            self.writeread_str(['#1@0001\r'])
        else:
            self.writeread_str(['#1@0081\r'])
        if self.idflex.error != 0: return

        #Set timeout for reading from camera
        ret = self.idflex.setTimeOut()
        if self.idflex.error != 0: return

        self.idflex.message = 'Updated camera view.'

    def start_up_pimms(self,trim_file="",function=0,trigger=0,rows=5) -> None:
        '''
        This function sends the updated DAC and start-up commands to PIMMS.

        The order of operations are IMPORTANT do not change them.
        '''

        #Set correct values since camera is loaded with 0 values initially
        self.settings['dac_settings']['iSenseComp'] = 1204
        self.settings['dac_settings']['iTestPix'] = 1253

        #Program dac settings
        self.dac_settings()
        if self.idflex.error != 0: return

        #Send command strings to prepare PIMMS for image acquisition
        self.writeread_str(self.settings['operation_hex']['Start Up'])
        if self.idflex.error != 0: return

        #Set MSB and LSB to non-zero values and resend control values
        self.settings['ControlSettings']['iCompTrimMSB_DAC'] = [162,[42]]
        self.settings['ControlSettings']['iCompTrimLSB_DAC'] = [248,[43]]
        self.program_bias_dacs()
        if self.idflex.error != 0: return

        #After these commands are sent we now send the trim file to PIMMS
        # Generate an empty trim array
        trim = self.trim.write_trim(value=0)
        if (os.path.isfile(trim_file) & trim_file.endswith('.bin')):
            trim = self.trim.read_binary_trim(trim_file)
        if (os.path.isfile(trim_file) & trim_file.endswith('.csv')):
            trim = self.trim.write_trim(filename=trim_file)
        
        self.send_trim_to_pimms(trim)
        if self.idflex.error != 0: return
        self.send_output_types(function,trigger)
        if self.idflex.error != 0: return

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
            if self.idflex.error != 0: return
            self.send_output_types(0,0)
        else:
            # Write trim data
            trim = self.trim.write_trim(value=value,iteration=iteration,calibration=True)
            self.send_trim_to_pimms(trim)