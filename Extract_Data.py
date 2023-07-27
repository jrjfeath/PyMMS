filename = r'C:\Users\chem-chem1612\Documents\Test\Temp_0000.h5'

#########################################################################################
# Extracts the Data stored by PymMS, format is: Shot ID, bin ID, ToF, X, Y
#########################################################################################
import h5py
import numpy as np

def Extract_Data(filename):
    data = []
    with h5py.File(filename, 'r') as hf:
        # The data is stored as arrays of five second increments
        # This was done to reduce read/write operations on the hard drive
        # shot_total tracks how many shots have been seen in each loop
        shot_total = 0
        for key in hf.keys():
            # Open the datasets by key value as a numpy array
            shot_data = hf[key][()]
            # ~ inverts the array and sets 0 to -1, any checks if the row is all -1
            shot_indices = np.where(~shot_data.any(axis=1))[0]
            # Make an empty array to store the shot id information
            shot_id = np.zeros((shot_data.shape[0]),dtype=np.int32)
            # set all values from 0 to first index to current shot total
            shot_id[0:shot_indices[0]] = shot_total
            shot_total+=1
            # loop through all indices and add shot_total
            for lid in range(len(shot_indices)):
                if lid == len(shot_indices) - 1: 
                    shot_id[shot_indices[lid]:] = shot_total
                else: 
                    shot_id[shot_indices[lid]:shot_indices[lid+1]] = shot_total
                    shot_total+=1
            # add shot id to front of array
            shot_data = np.hstack((shot_id[np.newaxis].T,shot_data))
            # Remove rows that are 0
            shot_data = np.delete(shot_data,shot_indices,0)        
            data.append(shot_data)
    # Combine all the data into one master array
    data = np.vstack(data)
    # Rearrange the columns such that the data format is: Shot id, bin id, ToF, X, Y
    data = data[:, [0, 4, 3, 1, 2]]
    return data

if __name__ == '__main__':
    print(Extract_Data(filename))