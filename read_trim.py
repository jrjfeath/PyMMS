import math
import numpy as np

def write_trim(filename=None,cols=324,rows=324,value=15):
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
    print(file_arr[:45])
    #with open(f'{fd}\\045.bin','wb') as opf:
    #    opf.write(file_arr)

def open_trim():
    data = np.fromfile(f'{fd}\\045.bin',dtype=np.uint8)
    print(data)
    print(len(data)*8)

fd = r'C:\Users\mb\Downloads\045.txt'
write_trim(value=5)
