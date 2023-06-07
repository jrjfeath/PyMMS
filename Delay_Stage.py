import os
import sys
import ctypes

path = r'C:\Users\mb\OneDrive - Nexus365\Documents\Newport'
os.add_dll_directory(path)
dls = ctypes.cdll.LoadLibrary(f'{path}\\Newport.DLS.CommandInterfaceDLS.dll')

cmn = getattr(dls,'DLS')