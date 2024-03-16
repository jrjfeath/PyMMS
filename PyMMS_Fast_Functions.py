import ctypes
import os
import sys
import numpy as np
import time
from PyMMS_TrimData import TrimData
#Important notes about ctypes 
#Pointers are made using: ptr = ctypes.c_void_p()
#When you want to pass a pointer to a C function use ctypes.byref(ptr) in the function call

#DLL functions should have their arguments and return types defined
#func.restype = ctypes.c_int (i.e function returns an int)
#func.argtypes = [ctypes.POINTER(ctypes.c_void_p),ctypes.c_int] (takes a pointer and int)

#Directory containing this file
fd = os.path.dirname(__file__)
#Directory containing dlls needed to run the camera
if sys.platform == "win32": os.add_dll_directory(fd)
pleora_sdk = r'C:\Program Files\Common Files\Pleora\eBUS SDK'
os.add_dll_directory(pleora_sdk)
dll_path = os.path.join(fd,'PImMS.dll')

class idflexusb():
    '''
    Controls interactions with the 64bit PImMS shared library for fast PImMS (.dll)
    
    Functions should not be altered unless instructions have been given by aSpect.
    '''
    def __init__(self) -> None:
        self.camera_id = ctypes.c_void_p() #Set camera value to none
        self.message : str = f""
        self.error : int = 0
        try:
            self.pimms = ctypes.cdll.LoadLibrary(dll_path)
            self.message = f"Loaded PyMMS dll."
        except FileNotFoundError:
            self.message =  f'Cannot find?: {dll_path}\n Cannot find?: {pleora_sdk}'
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
        '''

        open_cam = self.pimms.InitDevice
        open_cam.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),  # deviceHdl
            ctypes.POINTER(ctypes.c_char),    # deviceDescriptor
            ctypes.POINTER(ctypes.c_uint32),  # internalResult
            ctypes.POINTER(ctypes.c_char),    # errorCodeString
            ctypes.POINTER(ctypes.c_uint32),  # errorCodeStringLength
            ctypes.POINTER(ctypes.c_char),    # errorDescription
            ctypes.POINTER(ctypes.c_uint32)   # errorDescriptionLength
        ]
        open_cam.restype = ctypes.c_int32

        # Define variables
        deviceDescriptor = ctypes.create_string_buffer(1024) #This always returns empty
        internalResult = ctypes.c_uint32()
        errorCodeString = ctypes.create_string_buffer(1024)  # Allocate buffer for error code string
        errorCodeStringLength = ctypes.c_uint32(256)  # Set initial size of the buffer
        errorDescription = ctypes.create_string_buffer(1024)  # Allocate buffer for error description
        errorDescriptionLength = ctypes.c_uint32(256)  # Set initial size of the buffer

        # Call the function
        result = open_cam(
            ctypes.byref(self.camera_id),
            deviceDescriptor,
            ctypes.byref(internalResult),
            errorCodeString,
            ctypes.byref(errorCodeStringLength),
            errorDescription,
            ctypes.byref(errorDescriptionLength)
        )
        print(f'Open Camera: {result}')
        print(f'Device ID: {self.camera_id}')
        print(f'Error: {errorCodeString.value.decode("utf-8")}')
        print(f'Error Description: {errorDescription.value.decode("utf-8")}')

        if result != 0:
            self.error = 1
            self.message = f'Error: Cannot connect to camera, check USB connection!'
            return 

        class PortConfig(ctypes.Structure):
            _fields_ = [
                ("SerialPort", ctypes.c_uint),
                ("StopBits", ctypes.c_char_p),
                ("Parity", ctypes.c_char_p),
                ("LoopBack", ctypes.c_bool),
                ("Mode", ctypes.c_char_p),
                ("BaudRate", ctypes.c_char_p),
                ("BaudRateFactor", ctypes.c_uint),
                ("BaudRateValue", ctypes.c_double),
                ("SystemClockDivider", ctypes.c_char_p),
                ("OutputClockFrequency", ctypes.c_double)
            ]

        portConfig = PortConfig(
            2,                                        # SerialPort
            ctypes.c_char_p(b"One"),                  # StopBits
            ctypes.c_char_p(b"None"),                 # Parity
            False,                                    # LoopBack
            ctypes.c_char_p(b"UART"),                 # Mode
            ctypes.c_char_p(b"Baud38400"),            # BaudRate
            0,                                        # BaudRateFactor
            0.0,                                      # BaudRateValue
            ctypes.c_char_p(b"By2"),                  # SystemClockDivider
            0.0                                       # OutputClockFrequency
        )

        OpenSerialPort = self.pimms.OpenSerialPort
        OpenSerialPort.argtypes = [
            ctypes.c_void_p, # deviceHdl
            ctypes.c_uint8,  # appPortIdx
            ctypes.c_void_p, # portConfig
            ctypes.c_uint32, # rxBufferSize
            ctypes.c_uint8,  # useTermChar
            ctypes.c_uint8   # termChar
        ]
        OpenSerialPort.restype = ctypes.c_int32

        rxBufferSize = 2048
        useTermChar = True
        termChar = 0xD  # Termination character: \r 

        result = OpenSerialPort(
            self.camera_id,
            ctypes.c_uint8(0),
            ctypes.byref(portConfig),
            ctypes.c_uint32(rxBufferSize),
            ctypes.c_uint8(useTermChar),
            ctypes.c_uint8(termChar)
        )
        print(f'Open Serial Port 1: {result}')
        if result != 0:
            self.error_encountered(f'Cannot connect to serial port, check USB connection!')
            return

        portConfig.SerialPort = 3
        portConfig.Mode = ctypes.c_char_p(b"USRT")
        useTermChar = False
        termChar = 0x0  # Termination character

        result = OpenSerialPort(
            self.camera_id,
            ctypes.c_uint8(1),
            ctypes.byref(portConfig),
            ctypes.c_uint32(rxBufferSize),
            ctypes.c_uint8(useTermChar),
            ctypes.c_uint8(termChar)
        )

        print(f'Open Serial Port 2: {result}')
        print("OutputClockFrequency:", portConfig.OutputClockFrequency)
        if result != 0:
            self.error_encountered(f'Cannot connect to serial port, check USB connection!')
            return 

        open_stream = self.pimms.OpenStream
        open_stream.restype = ctypes.c_int 
        open_stream.argtypes = [ctypes.c_void_p]

        result = open_stream(self.camera_id)
        print(f'Open stream: {result}')
        if result != 0:
            self.error_encountered(f'Cannot connect to stream, check USB connection!')
    
    def close_pipeline(self) -> None:
        '''
        Return of 0 means the pipeline was closed\n
        '''
        close_pipeline = self.pimms.ClosePipeline
        close_pipeline.restype = ctypes.c_int 
        close_pipeline.argtypes = [ctypes.c_void_p]

        ret = close_pipeline(self.camera_id)
        print(f'Closing pipeline: {ret}')
    
    def close_device(self) -> None:
        '''
        Return of 0 means the camera successfully disconnected\n
        '''
        self.close_pipeline()

        close_stream = self.pimms.CloseStream
        close_stream.restype = ctypes.c_int 
        close_stream.argtypes = [ctypes.c_void_p]

        ret = close_stream(self.camera_id)

        close_device = self.pimms.ExitDevice
        close_device.restype = ctypes.c_int 
        close_device.argtypes = [ctypes.c_void_p]

        ret = close_device(self.camera_id)
        self.error = ret
        self.message = f'Disconnected device: {ret}'
        print(self.message)

    def writeread_device(self,data,bytestoread,port=0,timeout=1000,sleep=0.05) -> None:
        '''
        Write data to the PIMMS camera.
        Format: data [string], byte size [int], port [int], Timeout(ms) [int]
        '''

        wr_cam = self.pimms.SerialPortWriteRead

        wr_cam.restype = ctypes.c_int
        wr_cam.argtypes = [
            ctypes.c_void_p,                              # handle
            ctypes.c_uint8,                               # applicationPortIdx
            ctypes.POINTER(ctypes.c_uint8),               # bufferW
            ctypes.c_uint32,                              # bytesToWrite
            ctypes.POINTER(ctypes.c_uint32),              # bytesWritten
            ctypes.POINTER(ctypes.c_char * bytestoread),  # bufferR
            ctypes.c_uint32,                              # bytesToRead
            ctypes.POINTER(ctypes.c_uint32),              # bytesRead
            ctypes.c_uint32                               # timeOut
        ]

        # Setup data for transfer. 
        # Data is passed as hex to function, convert to ascii characters
        #Data passed is either a list of strings or numpy array of ascii characters
        if not isinstance(data, np.ndarray):
            byte_data = bytearray()
            byte_data.extend(map(ord, data)) #i.e convert A to ascii (65)
        else:
            byte_data = bytearray(data)
        ascii_data = ctypes.c_char * len(byte_data)
        ascii_data = ascii_data.from_buffer(byte_data)
        bytesToWrite = len(data)
        bytesWritten = ctypes.c_uint32()

        #Setup output data
        data_out = ctypes.create_string_buffer(bytestoread)
        bytesRead = ctypes.c_uint32()

        ret = wr_cam(
            self.camera_id,
            ctypes.c_uint8(port),
            ctypes.cast(ascii_data, ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_uint32(bytesToWrite),
            ctypes.byref(bytesWritten),
            data_out,
            ctypes.c_uint32(bytestoread),
            ctypes.byref(bytesRead),
            ctypes.c_uint32(timeout)
        )

        if len(data) > 200: 
            print(f'Sent trim, Returned: {ret}')
        else:
            print(f'{ret}, Sent: {data[:-1]}, Returned: {data_out.raw[:bytesRead.value]}')
        if ret != 0: self.error_encountered(f"Error writing data to camera.")
        time.sleep(sleep) #Wait X ms between each send
    
    def SetFrameTimeOut(self) -> None:
        '''
        Set the timeout for fetching a frame from PIMMS.
        '''
        sfto = self.pimms.SetFrameTimeOut
        sfto.restype = ctypes.c_int
        # handle, timeout
        sfto.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        ret = sfto(self.camera_id,ctypes.c_uint32(5000))
        if ret != 0: self.error_encountered(f"Error setting frame timeout.")
        print(f'Setting Frame Timeout: {ret}')
        time.sleep(0.1)
    
    def SetFrameFormatControl(self,rows=5, offset_x = 0, offset_y = 0) -> None:
        '''
        Set the parameters for each frame, i.e width, height, etc.
        '''

        sffc = self.pimms.SetFrameFormatControl

        sffc.restype = ctypes.c_int
        sffc.argtypes = [
            ctypes.c_void_p,  # handle
            ctypes.c_char_p,  # Pixel Format
            ctypes.c_uint64,  # Width
            ctypes.c_uint64,  # Height
            ctypes.c_uint64,  # Offset X
            ctypes.c_uint64,  # Offset Y
            ctypes.c_char_p,  # Sensor Taps
            ctypes.c_char_p,  # Test Pattern
            ctypes.c_uint8    # useDVAL
        ]

        ret = sffc(
            self.camera_id,
            ctypes.c_char_p(b"Mono16"),
            ctypes.c_uint64(324),
            ctypes.c_uint64(rows*324),
            ctypes.c_uint64(offset_x),
            ctypes.c_uint64(offset_y),
            ctypes.c_char_p(b"One"),
            ctypes.c_char_p(b"Off"),
            ctypes.c_uint8(1)
        )
        if ret != 0: self.error_encountered(f"Error setting frames.")
        print(f'Setting up frame for image capture: {ret}')
        time.sleep(0.1) #Wait X ms between each send
    
    def CreatePipeline(self) -> None:
        '''
        Open pipeline for image transfer.
        '''
        create_pipeline = self.pimms.CreatePipeline
        create_pipeline.restype = ctypes.c_int 
        create_pipeline.argtypes = [
            ctypes.c_void_p, # handle
            ctypes.c_uint32, # bufferCount
            ctypes.c_uint32, # trasnferBufferCount
            ctypes.c_uint32  # transferBufferFrameCount
        ]

        ret = create_pipeline(
            self.camera_id,
            ctypes.c_uint32(32),
            ctypes.c_uint32(1),
            ctypes.c_uint32(1),
        )
        if ret != 0: self.error_encountered(f"Error setting up pipeline.")
        print(f'Creating pipeline: {ret}')

    def StartAcquisition(self) -> None:
        '''
        Return of 0 means the acquisition was started\n
        '''
        start_aqcuisition = self.pimms.StartImageAcquisition
        start_aqcuisition.restype = ctypes.c_int 
        start_aqcuisition.argtypes = [ctypes.c_void_p]

        ret = start_aqcuisition(self.camera_id)
        if ret != 0: self.error_encountered(f"Error starting acquisition.")
        print(f'Starting image acquisition: {ret}')

    def readImage(self, size=5) -> np.ndarray:
        '''
        Read image array off camera. Array is columns wide by number of outputs
        multiplied by the number of rows. i.e 324*324 experimental (4) would be
        (324,1296).
        '''

        arrayType = ((ctypes.c_uint16 * 324) * (324 * size))
        array = arrayType()
        frame_count = ctypes.c_uint32(1) 

        AcquireFrame = self.pimms.GetNextImage
        AcquireFrame.restype = ctypes.c_int 
        AcquireFrame.argtypes = [
            ctypes.c_void_p,                  # handle
            ctypes.POINTER(ctypes.c_uint32),  # Frame Count
            ctypes.POINTER(arrayType),        # Buffer
            ctypes.c_uint32                   # TriggerTimeOut
        ]

        ret = AcquireFrame(
            self.camera_id,
            ctypes.byref(frame_count),
            array,
            ctypes.c_uint32(5000),
        )

        #Get the images as a numpy array and then reshape
        img = np.ctypeslib.as_array(array)
        img = img.reshape(size,324,324)

        return img

    def StopAcquisition(self) -> None:
        '''
        Return of 0 means the acquisition was stopped\n
        '''
        stop_aqcuisition = self.pimms.StopImageAcquisition
        stop_aqcuisition.restype = ctypes.c_int 
        stop_aqcuisition.argtypes = [ctypes.c_void_p]

        ret = stop_aqcuisition(self.camera_id)
        if ret != 0: self.error_encountered(f"Error stopping acquisition.")

class pymms():
    '''
    Object for communicating with PIMMS camera.

    Functions parse data to be passed along to idflexusb dll.
    '''

    def __init__(self, settings : dict) -> None:
        self.idflex = idflexusb()
        if self.idflex.error != 0: return
        #Create operation modes, these could change in the future
        self.settings : dict = self.operation_modes(settings)

    def operation_modes(self,settings) -> dict:
        operation_hex = {}
        for key, value in settings['OperationModes'].items():
            hexes = ['#À@0004\r']
            reg = 26 #The first register value is always 26
            for subroutine in value:
                adr, regn = list(settings['SubRoutines'].values())[subroutine]
                res = reg << 8 | adr
                reg = regn
                hexes.append(f'#À@{hex(res)[2:].zfill(4)}\r')
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
        dac_hexes = [format(x,'X').zfill(4) for x in self.settings['dac_settings'].values()]
        indices = [16,8,0,1,2,3,30,31,23,24,22,21,20,19,18,28]
        dac_hex = '#ÀT'+''.join([dac_hexes[x] for x in indices])+"\r"
        self.idflex.writeread_device(dac_hex,255,port=0,timeout=2000,sleep=0.5)

    def program_bias_dacs(self) -> None:
        '''
        Programm the PIMMS2 DACs
        '''
        hex_str = self.settings['operation_hex']['Programme PImMS2 Bias DACs']
        # Unique to fast pimms, add it to settings after loading defaults
        self.settings['ControlSettings']['Fast Read Out'] = [0,[160]]
        for data in self.settings['ControlSettings'].values():
            value  = data[0]
            reg = data[1]
            if len(reg) == 1:
                res = (reg[0] << 8) | value
                hex_str.append(f'#À@{hex(res)[2:].zfill(4)}\r')
            else:
                q, r = divmod(value, 256) #Calculate 8 bit position
                hi = (reg[0] << 8) | q
                lo = (reg[1] << 8) | r
                hex_str.append(f'#À@{hex(hi)[2:].zfill(4)}\r')
                hex_str.append(f'#À@{hex(lo)[2:].zfill(4)}\r')
        self.writeread_str(hex_str)

    def turn_on_pimms(self) -> None:
        '''
        Send PIMMS the initial start-up commands.

        Defaults are read from the PyMMS_Defaults.

        All important voltages are initially set to 0mV.
        '''
        #Connect to the camera
        self.idflex.init_device()
        if self.idflex.error != 0: return 

        #Obtain the hardware settings for the PIMMS (hex -> binary), decode("latin-1") for str
        for name, details in self.settings['HardwareInitializationGen2'].items():
            byte = (bytes.fromhex(details[0])).decode('latin-1')
            self.idflex.writeread_device(byte,details[1],timeout=details[2])
            if self.idflex.error != 0: return 

        #Program dac settings
        self.dac_settings()
        if self.idflex.error != 0: return 

        #Program control settings
        self.program_bias_dacs()
        if self.idflex.error != 0: return 

        #Write stop header at end
        self.writeread_str(['#À@0005\r'])
        if self.idflex.error != 0: return 

        #If all connection commands successful return 0
        self.idflex.message = f'Connected to PIMMS!'
    
    def send_trim_to_pimms(self,trim) -> None:
        '''
        Sends trim data to camera.
        '''
        #Write the stop command to PIMMS
        self.writeread_str(['#À@0000\r'])
        if self.idflex.error != 0: return 
        #Tell camera that we are sending it trim data
        ret = self.writeread_str(['#À@0004\r'])
        if self.idflex.error != 0: return 
        #Send trim data to camera.
        self.idflex.writeread_device(trim,0,port=1,timeout=5000,sleep=0.5)

    def send_output_types(self,function=0,trigger=0,rows=5) -> None:
        # Set camera to take analogue picture along with exp bins
        if function == 0:
            self.writeread_str(self.settings['operation_hex']['Experimental w. Analogue Readout'])
        # Set camera to take experiment bins only
        else:
            self.writeread_str(self.settings['operation_hex']['Experimental'])
        if self.idflex.error != 0: return 
        
        # Set trigger: 0005 is free runnning, and 0085 is triggered
        if trigger == 0:
            self.writeread_str(['#À@0005\r'])
        else:
            self.writeread_str(['#À@0085\r'])
        if self.idflex.error != 0: return 

        self.idflex.close_pipeline()
        if self.idflex.error != 0: return 
        self.idflex.SetFrameTimeOut()
        if self.idflex.error != 0: return 
        self.idflex.SetFrameFormatControl(rows)
        if self.idflex.error != 0: return 
        self.idflex.CreatePipeline()
        if self.idflex.error != 0: return 
        self.idflex.message = 'Updated readout!'

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
        trim = TrimData.write_trim(value=0)
        if (os.path.isfile(trim_file) & trim_file.endswith('.bin')):
            trim = TrimData.read_trim(trim_file)
        if (os.path.isfile(trim_file) & trim_file.endswith('.csv')):
            trim = TrimData.write_trim(filename=trim_file)
        
        self.send_trim_to_pimms(trim)
        if self.idflex.error != 0: return 

        self.send_output_types(function,trigger,rows)
        if self.idflex.error != 0: return 

        #If all DAC setting and startup commands successful
        self.idflex.message = f'Updated PIMMS DACs, trim, and readout!'