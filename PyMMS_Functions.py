import ctypes
import os
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
    Controls interactions with the 64bit idFLEX_USB shared library (.dll) \n
    Functions should not be altered unless instructions have been given by aSpect. \n
    If camera id is required call func.camera_id
    '''

    def __init__(self) -> None:
        self.camera_id = 0
        self.error = 0
        try:
            self.pimms = ctypes.cdll.LoadLibrary(f'{fd}/idFLEX_USB.dll')
            print('Sucessfully loaded dll.')
        except FileNotFoundError:
            print(f'Cannot find: {fd}/idFLEX_USB.dll')
            self.error = 1

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
        return ret

    def close_device(self):
        '''
        Return of 0 means the camera successfully disconnected\n
        Return of anything else means there was an error disconnecting camera
        '''
        close_cam = self.pimms.Close_Device
        close_cam.restype = ctypes.c_int 
        close_cam.argtypes = [ctypes.c_void_p]

        ret = close_cam(self.camera_id)
        return ret

    def write_device(self, data, timeout=1000):
        '''
        Write data to the PIMMS camera.
        Format:  Data in [string], Timeout(ms) [int]
        '''
        w_cam = self.pimms.serialWrite

        data_in_ba = bytearray()
        data_in_ba.extend(map(ord, data))
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