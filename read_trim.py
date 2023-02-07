import math
import numpy as np
import time
import h5py

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def write_trim(filename=None,cols=324,rows=324,value=15):
    '''
    This function generates a calibration file for PIMMS2 either using a text file
    or through manual generation. If no filename is specified the entire calibration
    will default to the specified value (15) unless another is specified.

    How the calibration works:\n
    -Initially all pixels are set to the maximum allowed voltage (15) and the threshold
    (vThP-vThN) is scanned from -100mV to 200mV to uniformly observe absolutely no 
    intensity to the maximum allowed voltage on every pixel.
    '''
    if filename == None:
        arr =  np.full((cols, rows),value, dtype='>i')
    else:
        arr = np.loadtxt(filename,dtype=np.uint8)
        cols, rows = arr.shape

    file_arr = np.zeros((1,math.ceil((cols*rows*5)/8)),dtype=np.uint8)[0]

    def int_to_bool_list(num):
        return [bool(num & (1<<n)) for n in range(4)]

    ba = {}
    for i in range(16):
        ba[i] = int_to_bool_list(i)

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

def read_trim():
    '''
    This function reads a binary calibration file for PIMMS2 made using labview.
    '''
    file_arr = np.fromfile(f'{fd}\\045.bin',dtype=np.uint8)
    return file_arr

fd = r'Q:\Cameras\PIMMS'

string = '2331 4031 4146 460D'
#print([int(x,16) for x in string.split()])
hi = 26
lo = 255
res = hi << 8 | lo

import ctypes

a = bytes.fromhex('23 31 40 30 30 30 30 0D')
b = a.decode('latin-1')
data_in_ba = bytearray()
data_in_ba.extend(map(ord, b))
#print(a, str(a), data_in_ba)

h = ctypes.c_uint8(int('0x82',16))
print(h)