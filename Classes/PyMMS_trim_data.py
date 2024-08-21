import math
import numpy as np

class TrimData():
    '''
    This class contains functions for reading or writing trim files.
    '''
    def read_binary_trim(self,filename=None) -> np.ndarray:
        '''
        This function reads a binary calibration file for PIMMS2 made using labview.
        '''
        file_arr = np.fromfile(filename,dtype=np.uint8)
        return file_arr
    
    def write_trim(self,filename=None,cols=324,rows=324,value=15,iteration=0,calibration=False) -> np.ndarray:
        '''
        This function generates a calibration string for PIMMS2 either using a text file
        or through manual generation. If no filename is specified the entire calibration
        will default to the specified value (0) unless another is specified.
        '''
        # Create an array for controlling the pixel mask, 1 is off, 0 is on
        pixels_enabled = np.zeros((rows,cols), dtype=int)
        # If the trim data is stored in csv format
        if filename:
            arr = np.loadtxt(filename,dtype=np.uint8,delimiter=',')
        else:
            arr = np.full((rows,cols),value, dtype='>i')
            # When calibrating only set values for every 4th pixel and disable all other pixels
            # See Jason Lee's PhD thesis page 94 onwards on power droop
            if calibration:
                row, col = iteration // 3, iteration % 3
                pixels_enabled = np.ones((rows,cols), dtype=int)
                pixels_enabled[row::3, col::3] = 0

        # Create the trim array, each index represents a byte, every 5 bits
        # corresponds to a pixel.
        file_arr = np.zeros((math.ceil((cols*rows*5)/8)),dtype=np.uint8)

        # Convert the array of trim integer values into an array of bit values.
        # 0 = [False, False, False, False], 15 = [True, True, True, True]
        boolean_array = np.empty((rows, cols, 4), dtype=object)
        for i in range(16):
            boolean_array[arr==i] = [bool(i & (1<<n)) for n in range(4)]

        # Make arrays 1D for quicker indexing
        pixels_enabled = pixels_enabled.flatten()
        boolean_array = boolean_array.reshape((-1, 4))

        # Determine the quotient of each index
        ql = np.arange(cols * rows * 5) // 8
        # Determine the remainder of each index
        rl = np.arange(cols * rows * 5) % 8
        # Calculate the power multiplier for each remainder
        pl = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint8)[rl]
      
        i = 0
        trim_multiplier_indices = []
        trim_position_indices = []
        trim_boolean_indices = []
        mask_multiplier_indices = []
        mask_position_indices = []
        # PImMS is column-major sorted, so we have to iterate row by column
        for row in range(cols-1,-1,-1):
            for b in range(5): # Iterate over each bit
                for col in range(rows-1,-1,-1):
                    index = row + col * cols
                    # Filter out values for the boolean array (first four bits)
                    if b != 4:
                        trim_multiplier_indices.append(i)
                        trim_position_indices.append(index)
                        trim_boolean_indices.append(b)
                    # Filter out values for the enabled array (last bit)
                    else:
                        mask_multiplier_indices.append(i)
                        mask_position_indices.append(index)
                    i += 1
        # Sort the arrays so the match the bitness of pimms
        sorted_boolean = boolean_array[trim_position_indices, trim_boolean_indices]
        sorted_enabled = pixels_enabled[mask_position_indices]

        # Write the trim value data to the trim file array
        arr_indices = np.array(ql[trim_multiplier_indices]).flatten()
        arr_multiplier = np.array(pl[trim_multiplier_indices]).flatten()
        sorted_boolean *= arr_multiplier
        np.add.at(file_arr, arr_indices, sorted_boolean)

        # Write the mask value data to the trim file array
        pixel_mask_indices = np.array(ql[mask_multiplier_indices]).flatten()
        pixel_mask_multiplier  = np.array(pl[mask_multiplier_indices]).flatten()
        sorted_enabled *= pixel_mask_multiplier
        np.add.at(file_arr, pixel_mask_indices, sorted_enabled)
        return file_arr

    def binary_trim_to_csv(self,filename, cols=324, rows=324):
        '''
        This function reverses the process of storing data to bindary and reconstructs
        the original trim and enabled arrays.
        '''
        file_arr = self.read_binary_trim(filename)
        # Initialize arrays
        arr = np.zeros((rows, cols), dtype=np.uint8)
        pixels_enabled = np.zeros((rows * cols), dtype=int)
        
        # Determine the quotient of each index
        ql = np.arange(cols * rows * 5) // 8
        # Determine the remainder of each index
        rl = np.arange(cols * rows * 5) % 8
        # Calculate the power multiplier for each remainder
        pl = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint8)[rl]

        i = 0
        trim_multiplier_indices = []
        trim_position_indices = []
        trim_boolean_indices = []
        mask_multiplier_indices = []
        mask_position_indices = []

        for row in range(cols-1, -1, -1):
            for b in range(5):
                for col in range(rows-1, -1, -1):
                    index = row + col * cols
                    if b != 4:
                        trim_multiplier_indices.append(i)
                        trim_position_indices.append(index)
                        trim_boolean_indices.append(b)
                    else:
                        mask_multiplier_indices.append(i)
                        mask_position_indices.append(index)
                    i += 1

        trim_indices = np.array(ql[trim_multiplier_indices]).flatten()
        trim_multipliers = np.array(pl[trim_multiplier_indices]).flatten()
        
        mask_indices = np.array(ql[mask_multiplier_indices]).flatten()
        mask_multipliers = np.array(pl[mask_multiplier_indices]).flatten()
        
        # Retrieve and decode trim values
        trim_values = (file_arr[trim_indices] & trim_multipliers).astype(bool)
        boolean_array = np.zeros((rows * cols, 4), dtype=bool)
        boolean_array[trim_position_indices, trim_boolean_indices] = trim_values
        
        # Map boolean arrays to 0-15 integer values
        for i in range(16):
            bits = [bool(i & (1<<n)) for n in range(4)]
            mask = (boolean_array == bits).all(axis=-1)
            arr[mask.reshape(rows, cols)] = i
        
        # Retrieve and decode mask values
        mask_values = (file_arr[mask_indices] & mask_multipliers).astype(bool)
        pixels_enabled[mask_position_indices] = mask_values
        pixels_enabled = pixels_enabled.reshape((rows, cols))
        
        np.savetxt(f'{filename[:-4]}.csv', arr, delimiter=',', fmt='%i')
        np.savetxt(f'{filename[:-4]}-enabled.csv', pixels_enabled, delimiter=',', fmt='%i')