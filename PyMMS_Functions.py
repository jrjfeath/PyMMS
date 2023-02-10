import ctypes
import os
import math
import sys
import numpy as np
import time
#Important notes about ctypes 
#Pointers are made using: ptr = ctypes.c_void_p()
#When you want to pass a pointer to a C function use ctypes.byref(ptr) in the function call

#DLL functions should have their arguments and return types defined
#func.restype = ctypes.c_int (i.e function returns an int)
#func.argtypes = [ctypes.POINTER(ctypes.c_void_p),ctypes.c_int] (takes a pointer and int)

#Directory containing this file
fd = os.path.dirname(__file__)
#Directory containing dlls needed to run the camera
os.add_dll_directory(fd)

class pymms():
    '''
    Controls interactions with the 64bit idFLEX_USB shared library (.dll)
    
    Functions should not be altered unless instructions have been given by aSpect.
    
    If camera id is required call func.camera_id

    Functions controlling calibration and trim files are also found here.
    '''

    def __init__(self) -> None:
        self.camera_id = ctypes.c_void_p() #Set camera value to none
        self.error = 0
        try:
            self.pimms = ctypes.cdll.LoadLibrary(f'{fd}/idFLEX_USB.dll')
            print('Sucessfully loaded dll.')
        except FileNotFoundError:
            print(f'Cannot find: {fd}/idFLEX_USB.dll')
            self.error_encountered()

    def error_encountered(self):
        '''
        If an error is encountered while running code closes connection to the camera.
        '''
        if self.camera_id.value is not None:
            self.close_device()
            self.camera_id = 0
        sys.exit()

    def init_device(self):
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
            print('Cannot connect to camera, check USB connection!')
            self.error_encountered()

    def close_device(self):
        '''
        Return of 0 means the camera successfully disconnected\n
        Return of anything else means there was an error disconnecting camera
        '''
        close_cam = self.pimms.Close_Device
        close_cam.restype = ctypes.c_int 
        close_cam.argtypes = [ctypes.c_void_p]

        ret = close_cam(self.camera_id)
        if ret != 0:
            print('Cannot disconnect from camera, check USB connection!')
            self.error_encountered()

    def writeread_device(self,data,bytestoread,timeout=1000):
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

        return ret, data_out.raw

    def setAltSetting(self,altv=0):
        '''
        Writing trim data requires the camera to be set to alt = 1.
        '''

        alt = self.pimms.setAltSetting
        alt.restype = ctypes.c_int
        alt.argtypes = [ctypes.c_void_p, 
                        ctypes.c_uint8]

        value = ctypes.c_uint8(altv)

        ret = alt(self.camera_id, value)
        if ret != 0:
            print('Could not change camera register')
            self.error_encountered()

    def setTimeOut(self,eps='0x82'):
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

        if ret != 0:
            print('Could not set camera timeout for trim data')
            self.error_encountered()

    def write_trim_device(self,trim,eps='0x2'):
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

        if ret != 0:
            print('Could not send camera trim data')
            self.error_encountered()

    def readImage(self,size=5):
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

        if ret != 0:
            print('Could not get image data')
            self.error_encountered()

        #Get the images as a numpy array and then reshape
        img = np.ctypeslib.as_array(array)
        img = img.reshape(size,324,324)

        return img

    #End of dll functions and beginning of trim/calibration functions.

    def writeread_str(self,hex_list):
        '''
        Function takes a list and writes data to camera.
        '''
        for hexs in hex_list:
            ret, dat = self.writeread_device(hexs,len(hexs))
            print(f'Sent: {hexs[:-1]}, Returned: {dat}')
            if ret != 0:
                print(f'Could not write settings, have you changed a value?')
                self.error_encountered()

    def send_trim_to_pimms(self,trim):
        '''
        Sends trim data to camera.
        '''
        #Write the stop command to PIMMS
        self.writeread_str(['#1@0000\r'])

        #Wait 10ms before going to next step
        time.sleep(0.01)

        #Change the camera to setting 1
        self.setAltSetting(altv=1)

        #Tell camera that we are sending it trim data
        self.writeread_str(['#0@0D01\r','#1@0002\r'])

        #Set timeout for reading the trim file
        self.setTimeOut(eps='0x2')
        
        #Send trim data to camera.
        self.write_trim_device(trim)
        
        #Tell camera to stop expecting trim data
        self.writeread_str(['#1@0000\r','#0@0D00\r'])

        time.sleep(0.1)

        #Change the camera to setting 0
        self.setAltSetting(altv=0)

        #Write stop header at end
        self.writeread_str(['#1@0001\r'])

    def operation_modes(self,settings):
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

    def send_operation(self,hex_str,MemReg=4,UpdateReg=False):
        '''
        Send various operation modes as defined in defaults file.
        '''
        #If the user wants to change the number of memory registers for the camera
        if UpdateReg == True:
            res = (83 << 8) | MemReg
            hex_str.append(f'#1@{hex(res)[2:].zfill(4)}\r')

        for hexs in hex_str:
            ret, dat = self.writeread_device(hexs,len(hexs))
            print(f'Sent: {hexs[:-1]}, Returned: {dat}')
            if ret != 0:
                print(f'Could not write settings, have you changed a value?')
                self.error_encountered()

    def hardware_settings(self,settings):
        #Obtain the hardware settings for the PIMMS (hex -> binary), decode("latin-1") for str
        for name, details in settings['HardwareInitialization'].items():
            byte = (bytes.fromhex(details[0])).decode('latin-1')
            if len(details) == 2:
                ret, dat = self.writeread_device(byte,details[1])
            else:
                ret, dat = self.writeread_device(byte,details[1],details[2])
            print(f'{ret}, Setting: {name}, Sent: {byte[:-1]}, Returned: {dat}')
            if ret != 0:
                print(f'Could not write {name}, have you changed the value?')
                self.error_encountered()

        #Program dac settings
        self.dac_settings(settings)

        #Program control settings
        self.program_bias_dacs(settings)

        #Write stop header at end
        self.writeread_str(['#1@0001\r'])

    def dac_settings(self,settings):
        '''
        Combine the DAC settings to form the initialization string for PIMMS (int -> hex)
        Called whenever vThN & vThP are changed
        '''
        dac_hex = '#PC'+''.join([format(x,'X').zfill(4) for x in settings['dac_settings'].values()])+'\r'
        self.writeread_str([dac_hex])

    def program_bias_dacs(self,settings):
        '''
        Programm the PIMMS2 DACs, called after hardware settings
        '''
        hex_str = settings['operation_hex']['Programme PImMS2 Bias DACs']
        for data in settings['ControlSettings'].values():
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

    def calibration(self):
        '''
        How the calibration works:

        Initially all pixels are set to the maximum allowed voltage (15) and the 
        threshold (vThP-vThN) is scanned from -100mV to 100mV to uniformly observe 
        no intensity to the maximum allowed voltage on every pixel.

        vThN is set to 500mV

        vThP is scanned from 400mV to 600mV

        When high thresholds (>75mV) are set pixel masking is required due to the sheer 
        volume of noise being experienced (Power Droop). In this region only one in
        nine pixels is sampled to ensure adjacent pixels are not interfering with
        one another.
        '''

    def read_trim(self,filename=None):
        '''
        This function reads a binary calibration file for PIMMS2 made using labview.
        '''
        file_arr = np.fromfile(filename,dtype=np.uint8)
        return file_arr

    def write_trim(self,filename=None,cols=324,rows=324,value=15):
        '''
        This function generates a calibration string for PIMMS2 either using a text file
        or through manual generation. If no filename is specified the entire calibration
        will default to the specified value (15) unless another is specified.
        '''
        if filename == None:
            arr =  np.full((cols, rows),value, dtype='>i')
        else:
            arr = np.loadtxt(filename,dtype=np.uint8)
            cols, rows = arr.shape

        file_arr = np.zeros((1,math.ceil((cols*rows*5)/8)),dtype=np.uint8)[0]

        #A function to convert 0-15 into a boolean list
        def int_to_bool_list(num):
            return [bool(num & (1<<n)) for n in range(4)]

        #A dictionary containing the boolean lists for 0-15 to reduce runtime
        ba = {}
        for i in range(16):
            ba[i] = int_to_bool_list(i)

        #Generating the trim is a fairly convoluted process
        #First the loop starts with the last column and last row going backwards to 0,0
        #Confusingly we investigate the first index of the boolean array of each row
        #before we continue onto the next index.
        #Every time i increments by 8 we move an index in the file_array
        i = 0
        for a in range(cols-1,-1,-1):
            for b in range(5):
                for c in range(rows-1,-1,-1):
                    if b == 4:
                        i += 1
                        continue
                    q, r = divmod(i, 8)
                    v = 2**(7-r)
                    file_arr[q] += (ba[arr[c,a]][b] * v)
                    i += 1
        return file_arr