import math
import numpy as np
import time

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
    
    def write_trim(filename=None,cols=324,rows=324,value=15,iteration=0,calibration=False) -> np.ndarray:
        '''
        This function generates a calibration string for PIMMS2 either using a text file
        or through manual generation. If no filename is specified the entire calibration
        will default to the specified value (0) unless another is specified.
        '''
        # Create an array for controlling the pixel mask, 1 is off, 0 is on
        pixels_enabled = np.zeros((rows,cols), dtype=int)
        # If the trim data is stored in csv format
        if filename:
            arr = np.fromfile(filename,dtype=np.uint8)
        else:
            arr = np.full((rows,cols),value, dtype='>i')
            # When calibrating only set values for every 9th pixel and disable all other pixels
            # See Jason Lee's PhD thesis page 94 onwards on power droop
            if calibration:
                row, col = iteration // 9, iteration % 9
                pixels_enabled = np.ones((rows,cols), dtype=int)
                pixels_enabled[row::9, col::9] = 0

        # Create the trim array, each index represents a byte, every 5 bits
        # corresponds to a pixel.
        file_arr = np.zeros((math.ceil((cols*rows*5)/8)),dtype=np.uint8)

        # Convert the array of trim integer values into an array of bit values.
        # 0 = [False, False, False, False], 15 = [True, True, True, True]
        boolean_array = np.empty((rows, cols, 4), dtype=object)
        for i in range(16):
            boolean_array[arr==i] = [bool(i & (1<<n)) for n in range(4)]

        # Flatten (cols * rows) arrays and reverse the order, pimms reads from bottom right to top left
        pixels_enabled = pixels_enabled.flatten()[::-1]
        boolean_array = np.array(boolean_array.flatten()[::-1],dtype=int)
        
        # Calculate the values of 2^n
        powers_of_two = [2**(7-x) for x in range(8)]
        # Calculate the quotient and the remainder for the total number of iterations
        ql = [x // 8 for x in range(cols * rows * 5)]
        rl = [x % 8 for x in range(cols * rows * 5)]
        # Calculate the power multiplier for each remainder (128,64,32,16,8,4,2,1)
        pl = [powers_of_two[x] for x in rl]
        # The trim bits corresponds to the data every x * 5 * rows (i.e 0 to 1296 for 324*324)
        trim_indices = np.array([ql[x*rows*5:(x+1)*5*rows-rows] for x in range(0,cols)]).flatten()
        trim_values = np.array([pl[x*rows*5:(x+1)*5*rows-rows] for x in range(0,cols)]).flatten()
        trim_values *= boolean_array
        np.add.at(file_arr,trim_indices,trim_values)
        # The pixel mask bit corresponds to the data every x * 4 * rows (i.e 1296 to 1620 for 324*324)
        pixel_mask_indices = np.array([ql[x*5*rows-rows:x*5*rows] for x in range(1,cols+1)]).flatten()
        pixel_mask_values  = np.array([pl[x*5*rows-rows:x*5*rows] for x in range(1,cols+1)]).flatten()
        pixel_mask_values *= pixels_enabled
        np.add.at(file_arr,pixel_mask_indices ,pixel_mask_values)

        return file_arr
