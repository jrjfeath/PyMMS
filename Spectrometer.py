import os
import ctypes
import time
import numpy as np
import matplotlib.pyplot as plt

int_time = 200  # integration time in milliseconds

assembly_path = r"Q:\Cameras\PIMMS"
Driver = ctypes.cdll.LoadLibrary(os.path.join(assembly_path,'Driver_64.dll'))

def get_wavelengths(num):    
    spec = Driver.getWavelength
    spec.restype = ctypes.POINTER(ctypes.c_float * num)
    ret = spec()
    return ret.contents[:]

def get_spectrum(num):
    class Point(ctypes.Structure):
        _fields_ = [("Spectrum", ctypes.POINTER(ctypes.c_int)),
                    ("Flag", ctypes.c_int)]

    spec = Driver.ReadSpectrum
    spec.restype = Point
    ret = spec()
    nums = []
    for i in range(num):
        try: nums.append(ret.Spectrum[i])
        except: print(i)
    return np.array(nums)

Driver.openSpectraMeter()
Driver.initialize()
#Start a wavelength aquisition
Driver.getSpectrum(ctypes.c_int(int_time))
#Loop until the spectrum is collected
while Driver.getSpectrumDataReadyFlag() != 1:
    time.sleep(0.001)
count = Driver.getPixelCount()
X = get_wavelengths(count)
Y = get_spectrum(count)
Driver.closeSpectraMeter()

plt.plot(X,Y)
plt.show()
