import ctypes
import os
import math
import sys
import numpy as np
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

    def write_device(self, data, timeout=1000):
        '''
        Write data to the PIMMS camera.
        Format:  Data in [string], Timeout(ms) [int]
        '''
        w_cam = self.pimms.serialWrite

        data_in_ba = bytearray()
        data_in_ba.extend(map(ord, str(data)))
        char_array = ctypes.c_char * len(data_in_ba)
        bytestowrite = ctypes.c_int32(ctypes.sizeof(char_array))
        timeout = ctypes.c_int32(timeout)

        w_cam.restype = ctypes.c_int
        w_cam.argtypes = [ctypes.c_void_p, 
                          ctypes.POINTER(ctypes.c_int32), 
                          ctypes.POINTER(ctypes.c_char * ctypes.sizeof(char_array)), 
                          ctypes.c_int32] 

        ret = w_cam(self.camera_id, ctypes.byref(bytestowrite), char_array.from_buffer(data_in_ba), timeout)
        return ret

    def read_device(self,bytestoread,timeout=1000):
        '''
        Write data to the PIMMS camera.
        Format: byte size [int], Timeout(ms) [int]
        '''
        r_cam = self.pimms.serialRead
        r_cam.restype = ctypes.c_int
        r_cam.argtypes = [ctypes.c_void_p, 
                          ctypes.POINTER(ctypes.c_int32), 
                          ctypes.POINTER(ctypes.c_char * bytestoread), 
                          ctypes.c_int32] 

        data_out = ctypes.create_string_buffer(bytestoread)
        bytestoread = ctypes.c_int32(ctypes.sizeof(data_out))
        timeout = ctypes.c_int32(timeout)
        
        ret = r_cam(self.camera_id, ctypes.byref(bytestoread), data_out, timeout)
        return ret, data_out.raw

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

    def setTimeOut(self):
        '''
        Set camera timeout for taking images. Values are always the same except for ID.
        '''

        sto = self.pimms.setTimeOut
        sto.restype = ctypes.c_int
        sto.argtypes = [ctypes.c_void_p, 
                        ctypes.c_uint8,
                        ctypes.c_int32]

        ep = ctypes.c_uint8(int('0x82',16))
        timeout = ctypes.c_int32(5000)

        ret = sto(self.camera_id,
                  ep,
                  timeout)

        return ret

    #End of dll functions and beginning of trim/calibration functions.

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

    def dac_settings(self,settings):
        '''
        Combine the DAC settings to form the initialization string for PIMMS (int -> hex)
        Called whenever vThN & vThP are changed
        '''
        dac_hex = '#PC'+''.join([format(x,'X').zfill(4) for x in settings['dac_settings'].values()])+'\r'
        ret, dat = self.writeread_device(dac_hex,len(dac_hex))
        print(f'Setting: DAC, Sent: {dac_hex[:-1]}, Returned: {dat}')
        if ret != 0:
            print(f'Could not write DAC settings, have you changed a value?')
            self.error_encountered()

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
        hex_str.append('#1@0001\r')
        for hexs in hex_str:
            ret, dat = self.writeread_device(hexs,len(hexs))
            print(f'Setting: DAC, Sent: {hexs[:-1]}, Returned: {dat}')
            if ret != 0:
                print(f'Could not write DAC settings, have you changed a value?')
                self.error_encountered()

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