import math
import numpy as np
import time
import sys
import ctypes
import itertools

from PyMMS_Functions import idflexusb

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_agg import FigureCanvasAgg

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

fd = r'C:\Users\mb\Documents\GitHub\PyMMS'

file_arr = np.fromfile(r'C:\Users\mb\Downloads\045_10.bin',dtype=np.uint8)

a = np.ones((4,2,2))
print(np.insert(a,0,np.zeros((2,2)),axis=0))
'''d = np.zeros((100,))
b, c = np.unique(a, return_counts=True)
d[b] = c'''

'''for i in range(4):
    a = np.random.randint(100,size=324*324)
    d = np.zeros((100,))
    b, c = np.unique(a, return_counts=True)
    d[b] = c'''
#ax.plot(d)

'''canvas.draw()
buf = canvas.buffer_rgba()
X = np.asarray(buf)'''

print(time.time() - start)

'''a = np.random.randint(4000,size=1000)
counts, bins = np.histogram(a,bins=400)
d = np.diff(bins) / 2
g = np.zeros((10,))
x = np.hstack((d+bins[:-1],d+bins[:-1]-1,d+bins[:-1]+1))
y = np.hstack((counts,g,g))

x, y = zip(*sorted(zip(x, y)))
print(x,y)

plt.plot(x,y)
plt.show()'''