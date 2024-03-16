import math
import numpy as np

class TrimData():
    '''
    This class contains functions for reading or writing trim files.
    '''
    def read_binary_trim(filename=None) -> np.ndarray:
        '''
        This function reads a binary calibration file for PIMMS2 made using labview.
        '''
        file_arr = np.fromfile(filename,dtype=np.uint8)
        return file_arr
    
    def write_trim(filename=None,cols=324,rows=324,value=0) -> np.ndarray:
        '''
        This function generates a calibration string for PIMMS2 either using a text file
        or through manual generation. If no filename is specified the entire calibration
        will default to the specified value (0) unless another is specified.
        '''
        #If the trim data is stored in csv format
        if filename:
            arr = np.fromfile(filename,dtype=np.uint8)
        else:
            arr = np.full((cols, rows),value, dtype='>i')

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