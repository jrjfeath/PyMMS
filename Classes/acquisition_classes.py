# Native python library imports
import os
import time
import queue
# External Library imports
import h5py
import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore

class ImageAcquisitionThread(QtCore.QThread):
    '''
    Thread for getting images from PIMMS.

    Size and pymms class passed after thread initialised
    '''
    def __init__(self, parent=None) -> None:
        QtCore.QThread.__init__(self, parent)
        self.running : bool = True
        self.waiting : bool = True
        self.size : int = 5
        self.image_queue = None # queue for storing images
        self.pymms = None # Class for camera communication

    def run(self) -> None:
        '''Ask the camera for image arrays.'''
        while self.running:
            if self.waiting: continue # If thread not active continue
            image = self.pymms.idflex.readImage(size = self.size)
            try: self.image_queue.put_nowait(image)
            except queue.Full: pass

    def pause(self) -> None:
        '''When Calibrating we do not want to be constantly creating and destroying threads.'''
        self.waiting = True
        # Empty image_queue on pause to prevent memory issues
        with self.image_queue.mutex:
            self.image_queue.queue.clear()

    def stop(self) -> None:
        '''When we close the application make sure the thread is closed.'''
        self.running = False

class RunCameraThread(QtCore.QThread):
    '''
    Main function for analysing frames from the camera.
    '''
    finished = QtCore.pyqtSignal()                  # Emit signal when thread finishes
    limit = QtCore.pyqtSignal()                     # Emit signal when number of frames is reached
    progress = QtCore.pyqtSignal(str)               # Emit current progress on delay stage position
    acquisition_fps = QtCore.pyqtSignal(str, str)   # Emit current fps
    tof_counts = QtCore.pyqtSignal(int, np.ndarray) # Emit ToF counts
    ui_image = QtCore.pyqtSignal(np.ndarray)        # Emit the image array
    move_position = QtCore.pyqtSignal(float)        # Emit new delay stage position
    request_position = QtCore.pyqtSignal()          # Request current delay stage position

    def __init__(self) -> None:
        QtCore.QThread.__init__(self)
        self.image_queue = None                     # Queue for storing images
        self.running     : bool = True              # Used for killing thread
        self.waiting     : bool = True              # Used to analyse images from camera
        self.analogue    : bool = True              # Is camera outputting analogue images
        self.save_data   : bool = False             # Should camera save the data
        self.stop_frames : bool = False             # Should camera stop after n frames
        self.bins        : int = 4                  # Number of camera bins
        self.cml_number  : int = 0                  # Current number of cumulative shots
        self.shot_number : int = 1                  # Current number of shots
        self.fps         : int = 0                  # Current frame rate
        self.stop_count  : int = 0                  # Stop after n frames
        self.filename    : str = 'None'             # Name of file to save
        self.fps_timer   : float = time.time()      # Starting time on fps timer
        self.save_timer  : float = time.time()      # Starting time on save timer
        self.frm_image   : list = []                # Analysed frames to save
        self.pos_data    : list = []                # Position data to save
        self.cml_image   : np.ndarray = np.zeros((324,324),dtype=np.float32)

        # Variables used to track if user is calibrating the camera
        self.calibration : bool = False             # Is the user running a calibration
        self.calibration_array : np.ndarray = np.zeros((4,324,324),dtype=np.uint16)

        # Variables used to track progress of the delay stage
        self.delay_connected : bool = False         # Is the delay stage connected
        self.delay_shots     : int = 0              # How many shots per delay stage position
        self.delay_start     : float = 0.0          # What is the starting position
        self.delay_step      : float = 0.0          # What is the step size
        self.delay_end       : float = 0.0          # What is the end position
        self.delay_position  : float = 0.0          # What is the current delay stage position

    def reset_variables(self) -> None:
        pass

    def update_variables(self) -> None:
        self.fps_timer   : float = time.time()      # Starting time on fps timer
        self.save_timer  : float = time.time()      # Starting time on save timer 

    def __init__(self, parent=None) -> None:
        QtCore.QThread.__init__(self, parent)
        self.image_queue = None # queue for storing images
        self.running : bool = True
        self.analogue : bool = True
        self.delay_connected : bool = False
        self.bins : int = 4
        self.cml_number : int = 0
        self.shot_number : int = 1
        self.fps : int = 0
        self.fps_timer : float = time.time()
        self.save_timer : float = time.time()
        self.frm_image : list = []
        self.pos_data : list = []
        self.cml_image : np.ndarray = np.zeros((324,324),dtype=np.float32)
        self.save_data : bool = parent._save_box.isChecked()
        self.number_of_frames : int = parent._n_of_frames.value()
        self.stop_after_n_frames : bool = self.number_of_frames != 0
        self.calibration : bool = parent.run_calibration_
        self.calibration_array : np.ndarray = np.zeros((4,324,324),dtype=np.uint16)

        #Create a filename for saving data
        filename = os.path.join(parent._dir_name.text(),parent._file_name.text())
        filename += '_0000.h5'
        #Check if file already exists so you dont overwrite data
        fid = 1
        while os.path.exists(filename):
            filename = f'{filename[:-8]}_{fid:04d}.h5'
            fid+=1
        self.filename = filename

        if self.calibration:
            self.number_of_frames = parent._cal_frames.value()
            self.stop_after_n_frames = True

        #Delay stage end position
        if parent.dly_connected:
            self.delay_start : float = parent._delay_t0.value() + (parent._dly_start.value() / 6671.2819)
            self.delay_step : float = parent._dly_step.value() / 6671.2819
            self.delay_end : float = parent._delay_t0.value() + (parent._dly_stop.value() / 6671.2819)
            self.delay_current : float = self.delay_start
            self.delay_shots : int = 0
            self.position : float = 0.0
            self.delay_end += self.delay_step #include the delay end in the position search
            self.delay_connected = True

        self.window_ = parent

    def extract_tof_data(self,mem_regs : np.ndarray) -> list:
        '''
        Parse images and get the X,Y,T data from each mem reg
        '''
        tof_data : list = []
        for id, reg in enumerate(mem_regs):
            try: Y,X = np.nonzero(reg)
            except ValueError:
                print(len(reg))
                return tof_data
            ToF = np.nan_to_num(reg[Y,X]) # Replace any nan values with 0
            ids = np.full((ToF.shape),id)
            #These are arrays that are flat, we want to stack them and then transpose
            st = np.transpose(np.vstack((X,Y,ToF,ids)))
            tof_data.append(np.array(st,dtype=np.int16))
        return tof_data

    def parse_analogue(self,img : np.ndarray) -> np.ndarray:
        '''
        Convert PIMMS array into an img format for displaying data
        '''
        #Default min is 5000 and max is 40000
        img[img > 40000] = 40000
        img[img < 5000] = 5000
        img = img.flatten()
        #Loop through each row, data storage is strange
        rows = []
        for i in range(324):
            row = img[i*324:(i+1)*324]
            a = row[0::4]
            b = row[1::4]
            c = row[2::4]
            d = row[3::4]
            rows.append(np.hstack((a,b,c,d)))
        img = rows
        img = img - np.min(img)
        return img

    def save_data_to_file(self) -> None:
        '''
        Save each frame stored in frm_image to the pickled file
        '''
        with h5py.File(self.filename, 'a') as hf:
            try: hf.create_dataset(f'{self.shot_number}',data=np.vstack(self.frm_image),compression='gzip')
            except ValueError: print('No data to save?')
        #If there is delay stage data save it
        if len(self.pos_data) == 0: return
        with h5py.File(f'{self.filename[:-3]}_position.h5', 'a') as hf:
            try: hf.create_dataset(f'{self.shot_number}',data=np.vstack(self.pos_data),compression='gzip')
            except ValueError: print('No data to save?')

    def delay_stage_position(self, position) -> None:
        self.delay_position = position
        self.delay_shots += 1
        #If we have reached the max number of images for this position
        if self.delay_shots == self.window_._dly_img.value():
            self.progress.emit(f'Moved to: {self.position:.4f}, End at: {self.delay_end:.4f}')
            self.delay_position+=self.delay_step
            self.move_position.emit(self.delay_position)
            self.delay_shots = 0
        #If we have passed the end point stop camera
        if (self.delay_start > self.delay_end) and (self.delay_position <= self.delay_end):
            self.stop_limit()
        if (self.delay_start < self.delay_end) and (self.delay_position >= self.delay_end):
            self.stop_limit()

    def run(self) -> None:       
        print('Starting camera.')
        self.acquisition_fps.emit('0','  0 fps')

        # Continue checking the image queue until we kill the camera.
        while self.running == True:
            # Update the fps count every second
            if time.time() - self.fps_timer > 1:
                self.acquisition_fps.emit(f'{self.shot_number}',f'{self.fps:3d} fps')
                self.fps_timer = time.time()
                self.fps = 0

            # Save data every 30 seconds to reduce disk writes
            if time.time() - self.save_timer > 5:
                if self.save_data and not self.calibration:
                    self.save_data_to_file()
                self.frm_image = [] #Remove all frames from storage
                self.pos_data = []
                self.save_timer = time.time()

            # Stop acquiring images when limit is reached
            if self.shot_number >= self.number_of_frames and self.stop_after_n_frames:
                self.acquisition_fps.emit(f'{self.shot_number}',f'  0 fps')
                self.stop_limit()

            # Wait for image to come through queue
            if self.image_queue.empty(): continue
            images = self.image_queue.get_nowait()

            # Update various counts
            self.cml_number+=1
            self.shot_number+=1
            self.fps+=1

            # Make sure PIMMS has passed back a data frame of the correct size
            if images.shape[0] != (self.bins + self.analogue) : continue

            # During calibration there is no need to output any data
            if self.calibration:
                temp = np.zeros((images.shape),dtype=int)
                temp[images != 0] = 1
                self.calibration_array = np.add(self.calibration_array,temp)
                self.cml_number+=1
                self.shot_number+=1
                self.fps+=1
                continue

            # Check if the user is saving delay position data
            if self.delay_connected:
                self.delay_stage_position()
                self.pos_data.append(self.position)
                                    
            '''
            Grab TOF data, then get unique values and update plot.
            '''
            img_index = self.window_._window_view.currentIndex()
            # Get the X, Y, ToF data from each mem reg
            if self.analogue: 
                tof_data = self.extract_tof_data(images[1:])
            else:
                tof_data = self.extract_tof_data(images)
            # Make a combined array for saving to file
            stacked_tof = np.vstack(tof_data)

            # Append the ToF Data to the save frame
            self.frm_image.append(stacked_tof)
            self.frm_image.append(np.zeros((4,),dtype=np.int16))

            # If the user wants to refresh the cumulative image clear array
            if self.window_.reset_cml_ == True:
                self.cml_image = np.zeros((324,324),dtype=np.float32)
                self.cml_number = 1
                self.window_.reset_cml_ = False

            # If the user selected cumulative do nothing
            if self.window_._tof_view.currentIndex() == 4: 
                tof_data = stacked_tof
            # Return the specific ToF the user wants
            elif self.window_._tof_view.currentIndex() < self.bins:
                tof_data = tof_data[self.window_._tof_view.currentIndex()]
            # If the user is selecting a bin that doesnt exist
            else:
                tof_data = np.zeros((tof_data[0].shape), dtype=np.uint8)

            tof = np.zeros(4096)
            uniques, counts = np.unique(tof_data[:,-2].flatten(), return_counts=True) 
            tof[uniques] = counts
            total_ions = int(np.sum(tof))
            self.tof_counts.emit(total_ions, tof)

            '''
            After TOF data has been plotted, convert into img format.
            '''
            #If the user wants to plot the analogue image
            if self.analogue and img_index == 4:
                image = self.parse_analogue(images[0])
            else:
                try:
                    if self.analogue: image = images[1:][img_index]
                    else: image = images[img_index]
                except IndexError: image = np.zeros((324,324))

            #Remove any ToF outside of range
            image[image > self.window_._max_x.value()] = 0
            image[image < self.window_._min_x.value()] = 0
            image = ((image / np.max(image)))
            image = np.rot90(image, self.window_.rotation_)
            #If no ions in shot
            if np.isnan(np.sum(image)): image = np.zeros((324,324))
            self.cml_image  = ((self.cml_image * (self.cml_number - 1)) / self.cml_number) + (image / self.cml_number)

            if self.window_._view.currentIndex() == 1:
                image = self.cml_image 
                image = ((image / np.max(image)))
            
            # Scale the image based off the slider
            if self.window_._vmax.value() != 100:
                image[image > (self.window_._vmax.value()*0.01)] = 0
                image = ((image / np.max(image)))

            colourmap = self.window_._colourmap.currentText()
            if colourmap != "None": 
                cm = pg.colormap.get(colourmap,source="matplotlib")
                image = cm.map(image)
            else:
                image = np.array(image * 255, dtype=np.uint8)

            self.ui_image.emit(image)
            # Delete large arrays to conserve memory
            del tof, tof_data, image

        if self.save_data:
            self.save_data_to_file()

        if self.calibration:
            self.window_.calibration_array_ = self.calibration_array

        # Delete large arrays to conserve memory
        del self.frm_image, self.calibration_array, self.cml_image
        self.finished.emit()
        print('Camera stopping.')

    def stop_limit(self) -> None:
        '''Used to stop the thread internally'''
        self.running = False
        self.limit.emit()

    def stop(self) -> None:
        '''Used to stop the thread externally'''
        self.running = False