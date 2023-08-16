filename = r'Q:\Joe\Temp_0003.h5'
filename2 = r'Q:\Joe\Temp_0003_position.h5'

#########################################################################################
# Extracts the Data stored by PymMS, format is: Shot ID, bin ID, ToF, X, Y
#########################################################################################
import h5py
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

def Extract_Data(filename,pos_data=None):
    data = []
    with h5py.File(filename, 'r') as hf:
        # The data is stored as arrays of five second increments
        # This was done to reduce read/write operations on the hard drive
        # shot_total tracks how many shots have been seen in each loop
        shot_total = 0
        keys = list(sorted([int(x) for x in hf.keys()]))
        for key in keys:
            # Open the datasets by key value as a numpy array
            shot_data = hf[str(key)][()]
            # ~ inverts the array and sets 0 to -1, any checks if the row is all -1
            shot_indices = np.where(~shot_data.any(axis=1))[0]
            # Make an empty array to store the shot id information
            shot_id = np.zeros((shot_data.shape[0]),dtype=np.int32)
            # set all values from 0 to first index to current shot total
            shot_id[0:shot_indices[0]] = shot_total
            if pos_data is not None:
                # Make an empty array to store the shot id information
                pos_id = np.zeros((shot_data.shape[0]),dtype=np.float32)
                pos_id[0:shot_indices[0]] = pos_data[shot_total]
            # loop through all indices and add shot_total
            shot_total+=1
            for lid in range(len(shot_indices)):
                if lid == len(shot_indices) - 1: 
                    shot_id[shot_indices[lid]:] = shot_total
                    if pos_data is not None:
                        try: pos_id[shot_indices[lid]:] = pos_data[shot_total]
                        except: pass
                else: 
                    shot_id[shot_indices[lid]:shot_indices[lid+1]] = shot_total
                    if pos_data is not None:
                        pos_id[shot_indices[lid]:shot_indices[lid+1]] = pos_data[shot_total]
                    shot_total+=1
            # add shot id to front of array
            shot_data = np.hstack((shot_id[np.newaxis].T,shot_data))
            if pos_data is not None: shot_data = np.hstack((shot_data,pos_id[np.newaxis].T))
            # Remove rows that are 0
            shot_data = np.delete(shot_data,shot_indices,0)        
            data.append(shot_data)
    # Combine all the data into one master array
    data = np.vstack(data)
    # Rearrange the columns such that the data format is: Shot id, bin id, ToF, X, Y, Delay Stage Position
    if pos_data is not None: data = data[:, [0, 4, 3, 1, 2, 5]]
    else: data = data[:, [0, 4, 3, 1, 2]]
    return data

def extract_delay_position(filename):
    data = []
    with h5py.File(filename, 'r') as hf:
        keys = list(sorted([int(x) for x in hf.keys()]))
        for key in keys:
            shot_data = hf[str(key)][()]
            for position in shot_data:
                data.append(position)
    return np.array(data)

