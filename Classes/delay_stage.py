'''
This library is used for communication with delay stages being integrated into the PyMMS software.
If a new class is added make sure to follow the function calls, see newport class, to ensure no code is broken.
The Newport.DLS.CommandInterfaceDLS.dll can be found in the following directory after installing their software:
C:/Windows/Microsoft.NET/assembly/GAC_64/Newport.DLS.CommandInterface
'''
import os
import sys
import serial.tools.list_ports #pyserial

#########################################################################################
# Class used for communicating with the Newport delay stage
#########################################################################################
class NewportDelayStage():
    '''
    Controls communication with Newport delay stages.\n
    If the name of the dll is different pass that when the function is called.\n
    Imported at the end of the script to prevent conflicts with PyQt library.\n
    Requires pyserial library.
    '''

    def __init__(self,directory,hardware_id='PID=104D:3009',filename='Newport.DLS.CommandInterfaceDLS'):
        if os.path.isfile(filename):
            self.hardware_id = hardware_id
            import clr #pythonnet
            #Load in newport c++ library
            sys.path.append(directory)
            clr.AddReference(filename)
            from CommandInterfaceDLS import DLS
            self.myDLS = DLS()
            self.dls_files_present = True
        else:
            self.dls_files_present = False
            print('Warning: delay stage control software not present.')

    def get_com_ports(self):
        '''
        List the available devices on the computer.\n
        The Newport stage used in B2 has hardware ID PID=104D:3009\n
        If the hardware id is different, figure out which id belongs\n 
        to your stage and pass that variable to the class call.
        '''
        ports = serial.tools.list_ports.comports()
        com_list = []
        if self.dls_files_present:
            for port, desc, hwid in sorted(ports):
                if self.hardware_id in hwid:
                    com_list.append(f"{port}; Delay Stage ; {hwid}")
                else:
                    com_list.append(f"{port}; {desc} ; {hwid}")
        return com_list

    def connect_stage(self,value):
        '''Connect to the delay stage by providing a COM port.'''
        return self.myDLS.OpenInstrument(value)

    def get_position(self):
        '''
        Returns the position of the delay stage.
        TP returns a tuple, 0 index is error code, 1 index is the value
        '''
        return str(self.myDLS.TP()[1])
    
    def get_minimum_position(self):
        '''Get the minimum position of the delay stage (mm).'''
        return str(self.myDLS.SL_Get()[1])
    
    def get_maximum_position(self):
        '''Get the maximum position of the delay stage (mm).'''
        return str(self.myDLS.SR_Get()[1])
    
    def set_position(self,value):
        '''Set the position of the delay stage.'''
        self.myDLS.PA_Set(value)

    def set_velocity(self,value):
        '''Set the velocity.\n Maximum velocity is 300 mm/s'''
        self.myDLS.VA_Set(value)

    def get_velocity(self):
        '''Get the velocity.'''
        return str(self.myDLS.VA_Get()[1])
    
    def set_acceleration(self,value):
        '''Set the acceleration.'''
        self.myDLS.AC_Set(value)

    def get_acceleration(self):
        '''Get the acceleration.'''
        return str(self.myDLS.AC_Get()[1])

    def disconnect_stage(self):
        '''
        Disconnect from the delay stage
        '''
        self.myDLS.CloseInstrument()