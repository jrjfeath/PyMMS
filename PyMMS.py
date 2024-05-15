#####################################################################################
# Variables found throughout this file are defined as:
# 1) UI variables from image.ui are: _name
# 2) UI variables generated in this file are: name_
# 3) All other variables are lowercase: name
#####################################################################################

#Native python library imports
import datetime
import os
import pathlib
import queue
import sys
import time
#Pip install library imports, found in install folder
#Pythonnet clr library is imported at bottom of file in __main__
#this library conflicts with PyQt if loaded before mainwindow
import h5py
import numpy as np
import pyqtgraph as pg
import warnings
import yaml
from PyQt6 import uic, QtWidgets, QtCore, QtGui
#Import the newport class function
from Delay_Stage import newport_delay_stage
#Supress numpy divide and cast warnings
warnings.simplefilter('ignore', RuntimeWarning)

#Directory containing this file
fd = os.path.dirname(__file__)
#filename for the ui file
uifn = "image.ui"
#create path to ui file
uifp = os.path.join(fd,uifn)
if os.path.isfile(uifp) == False:
    print("Cannot find the ui file (image.ui), where did it go?")
    exit() 

#Before we proceed check if we can load Parameter file
try:
    with open(f'{fd}/PyMMS_Defaults.yaml', 'r') as stream:
        defaults = yaml.load(stream,Loader=yaml.SafeLoader)
except FileNotFoundError:
    print("Cannot find the Parameters file (PyMMS_Defaults.yaml), where did it go?")
    exit()

#########################################################################################
# Thread Classes for controlling PImMS
#########################################################################################
class ImageAcquisitionThread(QtCore.QThread):
    '''
    Thread for getting images from PIMMS.

    Size and pymms class passed after thread initialised
    '''
    def __init__(self, parent=None) -> None:
        QtCore.QThread.__init__(self, parent)
        self.running : bool = True
        self.size : int = 5
        self.image_queue = None # queue for storing images
        self.pymms = None # Class for camera communication

    def run(self) -> None:
        #Get image from camera
        while self.running:
            image = self.pymms.idflex.readImage(size = self.size)
            try: self.image_queue.put_nowait(image)
            except queue.Full: pass

    def stop(self) -> None:
        self.running = False

class RunCameraThread(QtCore.QThread):
    '''
    Main function for analysing frames from the camera.
    '''
    finished = QtCore.pyqtSignal()
    limit = QtCore.pyqtSignal()
    progress = QtCore.pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        QtCore.QThread.__init__(self, parent)
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
        self.calibration_array : np.ndarray = np.zeros((4,324,324),dtype=int)

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

    def delay_stage_position(self) -> None:
        self.position = float(delay_stage.get_position())
        self.delay_shots += 1
        #If we have reached the max number of images for this position
        if self.delay_shots == self.window_._dly_img.value():
            self.progress.emit(f'Moved to: {self.position:.4f}, End at: {self.delay_end:.4f}')
            delay_current+=self.delay_step
            delay_stage.set_position(delay_current)
            self.delay_shots = 0
        #If we have passed the end point stop camera
        if (self.delay_start > self.delay_end) and (delay_current <= self.delay_end): 
            print(delay_current, self.delay_end)
            self.stop_limit()
        if (self.delay_start < self.delay_end) and (delay_current >= self.delay_end):
            print(delay_current, self.delay_end)
            self.stop_limit()

    def run(self) -> None:       
        print('Starting camera.')
        self.window_._frame_count.setText('0')

        # Continue checking the image queue until we kill the camera.
        while self.running == True:
            # Update the fps count every second
            if time.time() - self.fps_timer > 1:
                self.window_._fps_1.setText(f'{self.fps} fps')
                self.window_._frame_count.setText(f'{self.shot_number}')
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
                self.window_._frame_count.setText(f'{self.shot_number}')
                self.stop_limit()

            # If queue is empty the program will crash if you try to check it
            if self.image_queue.empty():
                pass
            else:
                # Get image array from queue
                images = self.image_queue.get_nowait()
                if images.shape[0] != (self.bins + self.analogue) : continue

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

                # Append the ToF Data to the save frame
                self.frm_image.append(np.vstack(tof_data))
                self.frm_image.append(np.zeros((4,),dtype=np.int16))

                # If the user wants to refresh the cumulative image clear array
                if self.window_.reset_cml_ == True:
                    self.cml_image = np.zeros((324,324),dtype=np.float32)
                    self.cml_number = 1
                    self.window_.reset_cml_ = False

                #Get tof plot data before getting live view image
                tof = np.zeros(4096)
                # If the user selected cumulative do nothing
                if self.window_._tof_view.currentIndex() == 4: 
                    pass
                # Return the specific ToF the user wants
                elif self.window_._tof_view.currentIndex() < self.bins:
                    tof_data = tof_data[self.window_._tof_view.currentIndex()]
                # If the user is selecting a bin that doesnt exist
                else:
                    tof_data = np.zeros((tof_data[0].shape), dtype=np.uint8)
                
                uniques, counts = np.unique(np.vstack(tof_data)[:,-2].flatten(), return_counts=True) 
                tof[uniques] = counts
                total_ions = int(np.sum(tof))
                self.window_.ion_count_displayed = total_ions
                self.window_.tof_counts_ = tof
                #Update the ion count
                del self.window_.ion_counts_[0]
                self.window_.ion_counts_.append(total_ions)

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
                self.cml_image  = ((self.cml_image   * (self.cml_number - 1)) / self.cml_number) + (image / self.cml_number)

                if self.window_._view.currentIndex() == 1:
                    image = self.cml_image 
                    image = ((image / np.max(image)))
                
                # Scale the image based off the slider
                if self.window_._vmax.value() != 100:
                    image[image > (self.window_._vmax.value()*0.01)] = 0
                    image = ((image / np.max(image)))

                #self.window_.image_ = np.array(image * 255,dtype=np.uint8)
                colourmap = self.window_._colourmap.currentText()
                if colourmap != "None": 
                    cm = pg.colormap.get(colourmap,source="matplotlib")
                    self.window_.image_ = cm.map(image)
                else:
                    self.window_.image_ = np.array(image * 255, dtype=np.uint8)

                #Update image canvas
                self.cml_number+=1
                self.shot_number+=1
                self.fps+=1

        if self.save_data:
            self.save_data_to_file()

        if self.calibration:
            self.window_.calibration_array_ = self.calibration_array

        self.finished.emit()
        print('Camera stopping.')

    def stop_limit(self) -> None:
        '''Used to stop the thread internally'''
        self.running = False
        self.limit.emit()

    def stop(self) -> None:
        '''Used to stop the thread externally'''
        self.running = False

class CameraCalibrationThread(QtCore.QThread):
    '''
    Function for calibrating the camera.
    '''
    take_images = QtCore.pyqtSignal()
    finished = QtCore.pyqtSignal()
    progress = QtCore.pyqtSignal(list)
    def __init__(self, parent=None) -> None:
        QtCore.QThread.__init__(self, parent)
        self.trim_value = parent._cal_trim.value()
        self.running = True
        self.initial = parent._cal_vthp.value()
        self.end = parent._cal_vthp_stop.value()
        self.inc = parent._cal_inc.value()
        self.current = 0
        self.static = parent._cal_static.isChecked()
        self.directory = parent._file_dir_2.text()
        self.window_ = parent

    def create_filename(self) -> None:
        '''
        As the calibration runs files are created to store the data.\n
        Ensure the user specified file is valid and dont overwrite existing files.
        '''
        # The user may try writing to somewhere they shouldn't
        if os.access(self.directory, os.W_OK) is not True:
            print('Invalid directory, writing to Documents')
            self.directory = str(pathlib.Path.home() / 'Documents')

        #Create a filename for saving data
        filename = f'{os.path.join(self.directory)}/P{self.current}_N{self.window_._cal_vthn.value()}'
        if self.static: filename += '_static'
        filename += '_0000.csv'
        #Check if file already exists so you dont overwrite data
        fid = 1
        while os.path.exists(filename):
            filename = f'{filename[:-9]}_{fid:04d}.csv'
            fid+=1
        self.filename = filename
    
    def run(self) -> None:
        number_of_runs = (self.trim_value+1) * 81
        if self.static: number_of_runs = 81
        # Set the start and stop labels
        self.window_._cal_start_l.setText(f'{self.initial}')
        self.window_._cal_end_l.setText(f'{self.end}')
        for vthp in range(self.initial, self.end+self.inc, self.inc):
            if not self.running: break
            self.current = vthp
            self.create_filename()
            # Before calibration set VthP and VthN
            self.window_.pymms.calibrate_pimms(
                update=True,
                vThN=self.window_._cal_vthn.value(),
                vThP=vthp,
            )
            # Update the current voltage label
            self.window_._self_cur_l.setText(f'{vthp}')
            counter = 0
            current = 0
            calibration = np.zeros((16,4,324,324)) # Calibration Array
            start = time.time()
            # Scan values 14 through 4, 15 used to find mean, 0,1,2,3 are too intense to use
            for v in range(self.trim_value, 3, -1):
                if self.static and self.trim_value != v: break
                if not self.running: break
                # We need to scan 324*324 pixels in steps of 9 pixels, thus 81 steps
                for i in range(0, 81):
                    if not self.running: break
                    print(f'Value, Run: {v, i}\n\n')
                    self.window_.pymms.calibrate_pimms(value=v,iteration=i)
                    time.sleep(0.1)
                    # Tell the acquisition to begin
                    if self.running: self.take_images.emit()
                    # On the first instance Images thread may not exist
                    while 'Images' not in self.window_.threads_ and self.running:
                        time.sleep(0.01)
                    # Check to see if thread is running
                    while not self.window_.threads_['Images'].running and self.running:
                        time.sleep(0.01)
                    # Check to see if the thread has completed
                    while self.window_.threads_['Images'].running and self.running:
                        time.sleep(0.01)
                    # Write data to the calibration array
                    calibration[v] = np.add(calibration[v],self.window_.calibration_array_)
                    # Update the progress bar for how far along the process is
                    percent_complete = int(np.floor((counter/number_of_runs) * 100))
                    if percent_complete > current:
                        time_remaining = int(((time.time() - start) / counter) * (number_of_runs - counter))
                        time_converted = f'{datetime.timedelta(seconds=time_remaining)}'
                        self.progress.emit([time_converted, percent_complete])
                        current = percent_complete
                    counter+=1
                with open(self.filename, "a") as opf:
                    np.savetxt(opf, np.sum(calibration[v],axis=0,dtype=np.int16), delimiter=',', fmt='%i')
                self.progress.emit(['00:00:00', 0])
        # After all threshold values have finished let the UI know the process is done
        if self.running: self.finished.emit()

class CameraCommandsThread(QtCore.QThread):
    '''
    Thread controls camera updates so the UI is not locked.
    This allows the UI to update the progress bar for the user.
    '''
    console_message = QtCore.pyqtSignal(str)
    turn_on_completed = QtCore.pyqtSignal()
    run_camera = QtCore.pyqtSignal()
    finished = QtCore.pyqtSignal()

    def __init__(self, parent=None) -> None:
        QtCore.QThread.__init__(self, parent)
        self.trim_file : str = ""
        self.output : int = 0
        self.trigger : int = 0
        self.rows : int = 5
        self.function : str = ""
        self.pymms = None # Class for camera communication

    def run(self) -> None:
        # The turn_on function has no arguments
        function = getattr(self.pymms, self.function)
        if self.function == 'turn_on_pimms': function()
        elif self.function == 'start_up_pimms': function(self.trim_file,self.output,self.trigger,self.rows)
        else: function(self.output,self.trigger,self.rows)
        self.console_message.emit(self.pymms.idflex.message)
        if self.function == 'turn_on_pimms': self.turn_on_completed.emit()
        else: self.run_camera.emit()
        self.finished.emit()

#########################################################################################
# Thread Classes for threading various elements of the UI
#########################################################################################
class UpdatePlotsThread(QtCore.QThread):
    update = QtCore.pyqtSignal()

    def __init__(self, parent=None) -> None:
        QtCore.QThread.__init__(self, parent)
        self.running = True

    def run(self) -> None:
        while self.running == True:
            self.update.emit()
            QtCore.QThread.msleep(100)

    def stop(self) -> None:
        self.running = False

class ProgressBarThread(QtCore.QThread):
    progressChanged = QtCore.pyqtSignal(int)
    started = QtCore.pyqtSignal()
    finished = QtCore.pyqtSignal()

    def __init__(self,parent=None) -> None:
        QtCore.QThread.__init__(self, parent)
        self.running = True

    def run(self) -> None:
        self.started.emit()
        while self.running == True:
            self.progressChanged.emit(np.random.randint(low=1,high=100))
            time.sleep(0.2)
        self.progressChanged.emit(0)
        self.finished.emit()

    def stop(self) -> None:
        self.running = False

class GetDelayPositionThread(QtCore.QThread):
    '''
    Thread reading position data of delay stage for UI
    '''
    progressChanged = QtCore.pyqtSignal(float)
    finished = QtCore.pyqtSignal()

    def __init__(self,parent=None) -> None:
        QtCore.QThread.__init__(self, parent)
        self.running = True

    def run(self) -> None:
        while self.running == True:
            value = delay_stage.get_position()
            try: self.progressChanged.emit(float(value))
            except: print('Cannot read position data.')
            time.sleep(0.5)
        self.finished.emit()

    def stop(self) -> None:
        self.running = False
#########################################################################################
# Class used for spawning threads
#########################################################################################
class UI_Threads():
    '''
    Threads for running the UI. 

    Work by creating a pyqt thread object and passing that object to a worker.

    When the worker terminates it passes a finished signal which terminates the thread.
    '''    
    def __init__(self, main_window) -> None:
        self.window_ = main_window

    def camera_commands_thread(self,function : str) -> None:
        '''
        This function controls updating camera settings. \n
        It terminates on its own after settings are updated.
        '''
        # Setup a progressbar to indicate camera communication
        thread = ProgressBarThread()
        thread.started.connect(lambda: self.window_._progressBar.setFormat('Communicating with camera.'))
        thread.finished.connect(lambda: self.window_._progressBar.setFormat(''))
        thread.progressChanged.connect(self.window_.update_pb)
        self.window_.threads_['ProgressBar'] = thread
        self.window_.threads_['ProgressBar'].start()
        # Start the camera communication thread
        thread = CameraCommandsThread()
        thread.output = self.window_._exp_type.currentIndex()
        thread.trim_file = self.window_._trim_dir.text()
        thread.trigger = self.window_._trigger.currentIndex()
        thread.rows = self.window_._bins.value() + 1 - self.window_._exp_type.currentIndex()
        thread.pymms = self.window_.pymms
        thread.function = function
        thread.console_message.connect(self.window_.update_console)
        thread.turn_on_completed.connect(self.window_.lock_camera_connect)
        thread.run_camera.connect(self.window_.unlock_run_camera)
        thread.finished.connect(self.window_.threads_['ProgressBar'].stop)
        self.window_.threads_['Camera'] = thread
        self.window_.threads_['Camera'].start()

    def acquisition_threads(self) -> None:
        '''
        This function controls image processing threads.\n
        self.threads_['Acquisition'] = Get images from camera \n 
        self.window_.threads_['Plots'] = Update UI plots \n
        self.threads_['Images'] = Process images from camera \n
        These threads are terminated by calling the stop() function.
        '''
        # Check if user is connected to the delay stage and saving data
        if self.window_.dly_connected and self.window_._save_box.isChecked():
            self.move_to_starting_position()
        #Fast Pymms has to start the acquisition
        if hasattr(self.window_.pymms.idflex, 'StartAcquisition'):
            self.window_.pymms.idflex.StartAcquisition()
        # Calculate the number of bins to pass to pymms
        bins = self.window_._bins.value()
        analogue = True
        if self.window_._exp_type.currentIndex() == 1:
            analogue = False
        size = bins + analogue
        # Generate a shared image queue object
        image_queue = queue.Queue(maxsize=2)
        # Start the processing thread for images
        thread = RunCameraThread(self.window_)
        thread.analogue = analogue
        thread.bins = bins
        thread.image_queue = image_queue
        thread.progress.connect(self.window_.update_console)
        thread.limit.connect(self.window_.start_and_stop_camera)
        self.window_.threads_['Images'] = thread
        self.window_.threads_['Images'].start()
        # Start the plot update thread
        thread = UpdatePlotsThread()
        thread.update.connect(self.window_.update_plots)
        self.window_.threads_['Plots'] = thread
        self.window_.threads_['Plots'].start()
        # Start the acquisition thread 
        thread = ImageAcquisitionThread()
        thread.size = size
        thread.image_queue = image_queue
        thread.pymms = self.window_.pymms
        self.window_.threads_['Acquisition'] = thread
        self.window_.threads_['Acquisition'].start()

    def stop_acquisition_threads(self) -> None:
        '''Stop all three acquisition threads'''
        for thread_name in ['Acquisition','Plots','Images']:
            thread = self.window_.threads_[thread_name]
            thread.running = False
            thread.wait()
            del thread
        # Fast Pymms has to close the acquisition
        if hasattr(self.window_.pymms.idflex, 'StopAcquisition'):
            self.window_.pymms.idflex.StopAcquisition()

    def get_delay_position_thread(self) -> None:
        '''Starts the thread that updates the delay generator position'''
        thread = GetDelayPositionThread()
        thread.progressChanged.connect(self.window_.update_pos)
        self.window_.threads_['Delay'] = thread
        self.window_.threads_['Delay'].start()

    def close_threads(self) -> None:
        '''This function terminates all active threads.'''
        for thread_name in self.window_.threads_:
            thread = self.window_.threads_[thread_name]
            thread.running = False
            thread.wait()
        # Fast Pymms has to close the acquisition
        if hasattr(self.window_.pymms.idflex, 'StopAcquisition'):
            self.window_.pymms.idflex.StopAcquisition()

    def move_to_starting_position(self) -> None:
        '''
        This function moves the delay stage to the starting position.
        '''
        #Check if the delay stage is connected and user is saving data
        start = self.window_._delay_t0.value() + (self.window_._dly_start.value() / 6671.2819)
        delay_stage.set_position(start)
        position = delay_stage.get_position()
        while (start-0.01 < float(position) < start+0.01) != True:
            self.window_.update_pb(np.random.randint(0,100))
            try: self.window_.update_console(f'Moving to start position: {round(start,4)}, Current position: {round(position,4)}')
            except TypeError: print(start,position)
            position = delay_stage.get_position()
            time.sleep(0.01)
        self.window_.update_console(f'Finished moving to start position: {position}')
        self.window_.update_pb(0)

    def camera_calibration_thread(self) -> None:
        print('Setting up for calibration.\n\n')
        thread = CameraCalibrationThread(self.window_)
        thread.progress.connect(self.window_.update_calibration_progress)
        thread.take_images.connect(self.window_.ui_threads.acquisition_threads)
        thread.finished.connect(lambda: self.window_.start_and_stop_camera('Calibration'))
        self.window_.threads_['Calibration'] = thread
        self.window_.threads_['Calibration'].start()

#########################################################################################
# Class used for modifying the plots
#########################################################################################
class UI_Plots():
    def __init__(self, main_window):
        self.window_ = main_window
        self.generate_plots()
    
    def generate_plots(self):
        '''
        Setup plot for TOF data

        Plot only updates the data and does not redraw each update.

        Axes are updated by looking at value changes.
        '''
        
        '''
        Setup plot for image readout
        '''
        self.window_.image_widget_origpos = self.window_._image_widget.pos()
        self.window_.image_widget_origwidth = self.window_._image_widget.width()
        self.window_.image_widget_origheight = self.window_._image_widget.height()
        self.window_._image_widget.setWindowTitle('Readout')
        self.window_._image_widget.installEventFilter(self.window_)
        self.grid = QtWidgets.QGridLayout(self.window_._image_widget)
        self.window_.graphics_view_ = pg.RawImageWidget(self.window_._image_widget,scaled=True)
        self.grid.addWidget(self.window_.graphics_view_, 0, 0, 1, 1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.window_.graphics_view_.sizePolicy().hasHeightForWidth())
        self.window_.graphics_view_.setSizePolicy(sizePolicy)
        self.window_.graphics_view_.setImage(self.window_.image_, levels=[0,255])
        

        self.window_.tof_plot_origpos = self.window_._tof_plot.pos()
        self.window_.tof_plot_origwidth = self.window_._tof_plot.width()
        self.window_.tof_plot_origheight = self.window_._tof_plot.height()
        self.window_.tof_plot_line = self.window_._tof_plot.plot(self.window_.ion_counts_,pen=pg.mkPen(color=(255, 0, 0)))
        self.window_.tof_selection_line_min = self.window_._tof_plot.plot(self.window_.ion_counts_,pen=pg.mkPen(color=(0, 0, 0)))
        self.window_.tof_selection_line_max = self.window_._tof_plot.plot(self.window_.ion_counts_,pen=pg.mkPen(color=(0, 0, 0)))
        self.window_._tof_plot.setWindowTitle('ToF Spectrum')
        self.window_._tof_plot.setBackground('w')
        self.window_._tof_plot.installEventFilter(self.window_)
        self.window_._tof_plot.setLabel('bottom', "Time Bin")
        self.window_._tof_plot.setLabel('left', "Counts")
        
        self.window_.ion_count_plot_origpos = self.window_._ion_count_plot.pos()
        self.window_.ion_count_plot_origwidth = self.window_._ion_count_plot.width()
        self.window_.ion_count_plot_origheight = self.window_._ion_count_plot.height()
        self.window_.ion_count_plot_line = self.window_._ion_count_plot.plot(self.window_.ion_counts_,pen=pg.mkPen(color=(255, 0, 0)))
        self.window_._ion_count_plot.setWindowTitle('Ion Count Spectrum')
        self.window_._ion_count_plot.setBackground('w')
        self.window_._ion_count_plot.installEventFilter(self.window_)
        self.window_._ion_count_plot.setLabel('bottom', "Number of Frames")
        self.window_._ion_count_plot.setLabel('left', "Total Ion Count")

        self.change_axes()

    def plot_selection(self):
        '''Updates the selection of the ToF displayed on the plot.'''
        x = np.zeros(4096)
        x[self.window_._min_x.value()] = self.window_._max_y.value()
        x[self.window_._max_x.value()] = self.window_._max_y.value()
        self.window_.tof_selection_line_min.setData(x,pen=pg.mkPen(color=(0, 0, 0), style=QtCore.Qt.PenStyle.DotLine))
        self.window_.tof_selection_line_max.setData(x,pen=pg.mkPen(color=(0, 0, 0), style=QtCore.Qt.PenStyle.DotLine))

    def change_axes(self):
        '''
        Updates the TOF axes to make it easier to view data.
        '''
        self.window_._tof_plot.setXRange(self.window_._tof_range_min.value(), self.window_._tof_range_max.value(), padding=0)
        self.window_._tof_plot.setYRange(self.window_._min_y.value(), self.window_._max_y.value(), padding=0)
        self.plot_selection()

    def rotate_camera(self,option):       
        #Rotate clockwise
        if option == 0:
            self.window_.rotation_ += 1
            if self.window_.rotation_ == 4: self.window_.rotation_ = 0
        #Rotate counter clockwise
        else:
            self.window_.rotation_ -= 1
            if self.window_.rotation_ < 0: self.window_.rotation_ = 3

    def pop_out_window(self,option):
        if option == 0: #If user is popping out tof
            self.window_.tof_expanded_ = True
            self.window_._tof_plot.setParent(None)
            self.window_._tof_plot.move(int(self.window_.width()/2),int(self.window_.height()/2))
            self.window_._tof_plot.show()
        elif option == 1: #if user is popping out image
            self.window_.img_expanded_ = True
            self.window_._image_widget.setParent(None)
            self.window_._image_widget.move(int(self.window_.width()/2),int(self.window_.height()/2))
            self.window_._image_widget.show()
        else:
            self.window_.ionc_expanded_ = True
            self.window_._ion_count_plot.setParent(None)
            self.window_._ion_count_plot.move(int(self.window_.width()/2),int(self.window_.height()/2))
            self.window_._ion_count_plot.show()             

#########################################################################################
# Mainwindow used for displaying UI
#########################################################################################
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super(MainWindow, self).__init__()

        #Load the ui file
        uic.loadUi(uifp,self)

        #Add default save locations
        self._file_dir_2.setText(str(pathlib.Path.home() / 'Documents'))

        #Add colourmaps
        colourmaps = pg.colormap.listMaps("matplotlib")
        self._colourmap.addItems(colourmaps)

        #Make some UI variables
        self.pymms = None
        self.threads_ = {} # Dictionary containing threads
        self.rotation_ = 0 # Rotation angle of the image
        self.connected_ = False #Is camera connected?
        self.dly_connected = False #Is the delay stage connected?
        self.camera_running_ = False #Is camera running?
        self.run_calibration_ = False
        self.calibration_array_ = np.zeros((4,324,324))
        self.tof_expanded_ = False
        self.img_expanded_ = False
        self.ionc_expanded_ = False
        self.reset_cml_ = False
        self.error_ = False #Is there an error?
        self.ion_count_displayed = 0
        self.ion_counts_ = [np.nan for x in range(self._nofs.value())]
        self.tof_counts_ = np.zeros(4096)
        self.image_ = np.zeros((324,324))
        self._vthp.setValue(defaults['dac_settings']['vThP'])
        self._vthn.setValue(defaults['dac_settings']['vThN'])
        self._bins.setValue(defaults['ControlSettings']['Mem Reg Read'][0])

        #Call the class responsible for plot drawing and functions
        self.ui_plots = UI_Plots(self)
        #Class the class responsible for handling threads
        self.ui_threads = UI_Threads(self)

        '''
        Various commands for buttons found on the ui
        '''
        self._camera_connect_button.clicked.connect(self.camera_control)
        self._trim_button.clicked.connect(lambda: self.open_file_dialog(1))
        self._path_button.clicked.connect(lambda: self.open_file_dialog(0))
        self._rotate_c.clicked.connect(lambda: self.ui_plots.rotate_camera(0))
        self._rotate_cc.clicked.connect(lambda: self.ui_plots.rotate_camera(1))
        self._update_camera.clicked.connect(self.update_camera)
        self._update_camera_2.clicked.connect(self.update_output)
        self._vthp.valueChanged.connect(self.update_dafaults)
        self._vthn.valueChanged.connect(self.update_dafaults)
        self._bins.valueChanged.connect(self.update_dafaults)
        self._window_view.currentIndexChanged.connect(self.reset_images)
        self._view.currentIndexChanged.connect(self.reset_images)
        self._rotate_c_2.clicked.connect(self.reset_images)

        self._button.clicked.connect(lambda: self.start_and_stop_camera('Acquisition'))
        self._cal_run.clicked.connect(lambda: self.start_and_stop_camera('Calibration'))

        #Update the plots when they are clicked on
        self._nofs.valueChanged.connect(self.update_nofs)
        self._tof_range_min.valueChanged.connect(self.ui_plots.change_axes)
        self._tof_range_max.valueChanged.connect(self.ui_plots.change_axes)
        self._min_y.valueChanged.connect(self.ui_plots.change_axes)
        self._max_y.valueChanged.connect(self.ui_plots.change_axes)
        self._min_x.valueChanged.connect(self.ui_plots.plot_selection)
        self._max_x.valueChanged.connect(self.ui_plots.plot_selection)
        self._pop_tof.clicked.connect(lambda: self.ui_plots.pop_out_window(0))
        self._pop_ion_count.clicked.connect(lambda: self.ui_plots.pop_out_window(2))
        self._pop_image.clicked.connect(lambda: self.ui_plots.pop_out_window(1))

        #Sometimes while idle the code tries to quit, prevent closing
        quit = QtGui.QAction("Quit", self)
        quit.triggered.connect(self.closeEvent)

        #Delay stage UI interactions
        self.update_coms()
        self._dly_refresh.clicked.connect(self.update_coms)
        self._dly_connect_button.clicked.connect(self.dly_control)
        self._dly_send_to.clicked.connect(self.dly_position)
        self._dly_vel_but.clicked.connect(self.dly_velocity)
        if not delay_stage.dls_files_present:
            self._dly_connect_button.setEnabled(False)

    def start_and_stop_camera(self, button=None) -> None:
        '''
        This function starts and stops the camera acquisition.\n
        It disables all buttons except for plot pop-outs and the start/stop.
        '''
        available_buttons = ['_pop_tof','_pop_ion_count','_pop_image',
                             '_rotate_c','_rotate_cc','_rotate_c_2']
        # The button returns none when the frame limit is reached
        if button != None:
            enable = True
            # If the user is doing a simple acquisition
            if button == 'Acquisition':
                available_buttons.append('_button')
                if self.camera_running_:
                    self.ui_threads.stop_acquisition_threads()
                    self.camera_running_ = False
                    self._button.setText("Start")
                else:
                    self.camera_running_ = True
                    self._button.setText("Stop")
                    self.ui_threads.acquisition_threads()
                    enable = False
            # If the user is looking to run a calibration
            if button == 'Calibration':
                available_buttons.append('_cal_run')
                if self.run_calibration_:
                    self.threads_['Calibration'].running = False
                    self.threads_['Calibration'].wait()
                    self._cal_run.setText("Start")
                    self._exp_type.setEnabled(True)
                    self.run_calibration_ = False
                else:
                    self.run_calibration_ = True
                    self._cal_run.setText("Stop")
                    self._exp_type.setCurrentIndex(1)
                    self._exp_type.setEnabled(False)
                    self.ui_threads.camera_calibration_thread()
                    enable = False
            for button in self.findChildren(QtWidgets.QPushButton):
                if button.objectName() in available_buttons: continue
                button.setEnabled(enable)
        else:
            # When the user is not running calibration re-enable UI
            self.ui_threads.stop_acquisition_threads()
            if self.camera_running_:
                for button in self.findChildren(QtWidgets.QPushButton):
                    if button.objectName() in available_buttons: continue
                    button.setEnabled(True)
                self._button.setText("Start")
                self.camera_running_ = False

    def update_nofs(self) -> None:
        self.ion_counts_ = [np.nan for x in range(self._nofs.value())]
        self.ion_count_plot_line.setData(self.ion_counts_)

    def update_plots(self) -> None:
        self.ion_count_plot_line.setData(self.ion_counts_)
        self.tof_plot_line.setData(self.tof_counts_)
        self.graphics_view_.setImage(self.image_, levels=[0,255])
        self._ion_count.setText(str(self.ion_count_displayed))

    def open_file_dialog(self,option : int) -> None:
        '''Open the window for finding files/directories'''
        if option == 0:
            dir_path = QtWidgets.QFileDialog.getExistingDirectory(
                parent=self,
                caption="Select directory",
                directory=os.getenv("HOME"),
                options=QtWidgets.QFileDialog.Option.ShowDirsOnly,
            )
            self._dir_name.setText(dir_path)
        else:
            file_path = QtWidgets.QFileDialog.getOpenFileName(
                parent=self,
                caption="Open File",
                directory=os.getenv("HOME")
            )
            self._trim_dir.setText(file_path[0])

    def update_console(self,string : str) -> None:
        '''
        Update console log with result of command passed to PIMMS.
        '''
        if 'Error' in string:
            self.threads_['ProgressBar'].stop
            self.error_ = True
            self._plainTextEdit.setPlainText(string)
            self._plainTextEdit.setStyleSheet(
                """QPlainTextEdit {color: #FF0000;}"""
            )
            error = QtWidgets.QMessageBox()
            error.setText(string)
            error.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Ok)
            error = error.exec()
        else:
            self._plainTextEdit.setPlainText(string)

    def eventFilter(self, source, event):
        '''
        Capture close events to pop the windows back into the main body
        '''
        if (event.type() == QtCore.QEvent.Type.Close and isinstance(source, QtWidgets.QWidget)):
            if source.windowTitle() == self._tof_plot.windowTitle():
                self.tof_expanded_ = False
                self._tof_plot.setParent(self._tab)
                self._tof_plot.setGeometry(
                    self.tof_plot_origpos.x(),
                    self.tof_plot_origpos.y(),
                    self.tof_plot_origwidth,
                    self.tof_plot_origheight
                )
                self._tof_plot.show()
            elif source.windowTitle() == self._image_widget.windowTitle():
                self.img_expanded_ = False
                self._image_widget.setParent(self._tab)
                self._image_widget.setGeometry(
                    self.image_widget_origpos.x(),
                    self.image_widget_origpos.y(),
                    self.image_widget_origwidth,
                    self.image_widget_origheight
                )
                self._image_widget.show()
            else:
                self.ionc_expanded_ = False
                self._ion_count_plot.setParent(self._tab)
                self._ion_count_plot.setGeometry(
                    self.ion_count_plot_origpos.x(),
                    self.ion_count_plot_origpos.y(),
                    self.ion_count_plot_origwidth,
                    self.ion_count_plot_origheight
                )
                self._ion_count_plot.show()                          
            event.ignore() #Ignore close
            return True #Tell qt that process done
        else:
            return super(MainWindow,self).eventFilter(source,event)

    def closeEvent(self, event) -> None:
        '''If the application is closed do the following'''
        close = QtWidgets.QMessageBox()
        close.setText("Would you like to quit?")
        close.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Yes | 
                                    QtWidgets.QMessageBox.StandardButton.No)
        close = close.exec()

        if close == QtWidgets.QMessageBox.StandardButton.Yes.value:
            if self.dly_connected:
                delay_stage.disconnect_stage()
            if self.pymms != None: 
                self.ui_threads.close_threads()
            if self.connected_: 
                self.pymms.idflex.close_device()
            self._image_widget.close()
            event.accept()
        else:
            event.ignore()

    def reset_images(self) -> None:
        '''Reset the cumulative images'''
        if self.reset_cml_ == False:
            self.reset_cml_ = True

    def update_pb(self, value : int) -> None:
        '''Set the progressbar value'''
        self._progressBar.setValue(value)

    def update_cal_pb(self, value : int) -> None:
        '''Set the progressbar value'''
        self._cal_progress.setValue(value)

    def update_calibration_progress(self, value : list) -> None:
        self._cal_remaining.setText(value[0])
        self._cal_progress.setValue(value[1])

    #####################################################################################
    # Delay Stage Functions
    #####################################################################################
    def update_pos(self,value : float) -> None:
        '''Updates the position of the delay stage on the UI'''
        self._dly_pos.setText(str(value))

    def update_coms(self) -> None:
        '''Updates the available coms on the computer'''
        self._com_ports.clear()
        com_list = delay_stage.get_com_ports()
        self._com_ports.addItems(com_list)

    def dly_control(self) -> None:
        '''Connect to and disconnect from the delay stage.'''
        if self.dly_connected == False:
            port, name, _ = self._com_ports.currentText().split(';')
            if 'Delay Stage' in name:
                delay_stage.connect_stage(port)
                self._dly_vel.setText(delay_stage.get_velocity())
                self._dly_pos_min.setText(delay_stage.get_minimum_position())
                self._dly_pos_max.setText(delay_stage.get_maximum_position())
                self.dly_connected = True
                self._dly_connect_button.setText('Disconnect')
                self.ui_threads.dly_pos_thread()
        else:
            self.ui_threads.dly_worker.stop()
            delay_stage.disconnect_stage()
            self._dly_connect_button.setText('Connect')
            self.dly_connected = False

    def dly_velocity(self) -> None:
        '''Change the velocity of the delay stage'''
        delay_stage.set_velocity(self._dly_vel_set.value())
        self._dly_vel.setText(delay_stage.get_velocity())

    def dly_position(self) -> None:
        '''Move delay stage to position specified.'''
        delay_stage.set_position(self._delay_t0.value())

    #####################################################################################
    # PIMMS Functions
    #####################################################################################
    def lock_camera_connect(self) -> None:
        '''
        Only connect to the camera and enable sending commands if there are no issues
        '''
        if self.connected_ == False and self.error_ == False:
            self._camera_connect.setText(f'Connected: {self._function_selection.currentText()}')
            self._camera_connect.setStyleSheet("color: green")
            self._camera_connect_button.setText(f'Disconnect')
            self._update_camera.setDisabled(False)
            self.connected_ = True
            self.threads_['ProgressBar'].stop()
        else:
            self.pymms = None
            self._camera_connect.setText(f'Disconnected')
            self._camera_connect.setStyleSheet("color: red")
            self._camera_connect_button.setText(f'Connect')
            self._button.setDisabled(True)
            self._cal_run.setDisabled(True)
            self._update_camera.setDisabled(True)
            self._update_camera_2.setDisabled(True)
            self.connected_ = False
            self.error_ = False
        self._camera_connect_button.setDisabled(False)

    def update_dafaults(self) -> None:
        '''
        Update user controllable default values
        '''
        if self.pymms == None:
            defaults['dac_settings']['vThP'] = self._vthp.value()
            defaults['dac_settings']['vThN'] = self._vthn.value()
            defaults['ControlSettings']['Mem Reg Read'][0] = self._bins.value()
        else:
            self.pymms.settings['dac_settings']['vThP'] = self._vthp.value()
            self.pymms.settings['dac_settings']['vThN'] = self._vthn.value()
            self.pymms.settings['ControlSettings']['Mem Reg Read'][0] = self._bins.value()

    def camera_control(self) -> None:
        '''
        Connect to and disconnect from the camera.
        '''
        if self.connected_ == False:
            # Find which option the user has selected 
            option_index = self._function_selection.currentIndex()
            #Functions to import that relate to PIMMS ideflex.dll
            if option_index == 0:
                from PyMMS_Functions import pymms
            if option_index == 1:
                from PyMMS_Fast_Functions import pymms
            if option_index == 2:
                from PyMMS_Test_Functions import pymms
            #Create PyMMS object
            self.pymms = pymms(defaults)
            self._plainTextEdit.setPlainText(f'Connecting to Camera, please wait!')
            self._camera_connect_button.setDisabled(True)
            self.ui_threads.camera_commands_thread('turn_on_pimms')
        else:
            #Disconnect from the camera
            self._plainTextEdit.setPlainText(f'Disconnecting from Camera, please wait!')
            self.pymms.idflex.close_device()
            self._plainTextEdit.setPlainText(self.pymms.idflex.message)
            self.lock_camera_connect()
            self.pymms = None

    def unlock_run_camera(self) -> None:
        self._button.setDisabled(False)
        self._cal_run.setDisabled(False)
        self._update_camera.setDisabled(False)
        self._update_camera_2.setDisabled(False)
        self._camera_connect_button.setDisabled(False)    

    def update_output(self) -> None:
        '''
        If user just wants to update the readout type

        Check that camera has been connected, updated and is not running.
        '''
        self._camera_connect_button.setDisabled(True)
        self._button.setDisabled(True)
        self._cal_run.setDisabled(True)
        self._update_camera.setDisabled(True)
        self._update_camera_2.setDisabled(True)
        self._plainTextEdit.setPlainText(f'Updating Camera output type!')
        self.ui_threads.camera_commands_thread('send_output_types')

    def update_camera(self) -> None:
        '''
        Update the DAC, camera settings, and TRIM file.
        '''
        if self.connected_ == False: 
            self._plainTextEdit.setPlainText(f'Please connect to the camera before updating output!')
            return
        self._camera_connect_button.setDisabled(True)
        self._button.setDisabled(True)
        self._cal_run.setDisabled(True)
        self._update_camera.setDisabled(True)
        self._update_camera_2.setDisabled(True)
        if os.path.isfile(self._trim_dir.text()) == True:
            self._plainTextEdit.setPlainText(f'Updating Camera DACs with trim file!')
        else:
            self._plainTextEdit.setPlainText(f'Updating Camera DACs without trim file!')
        self.ui_threads.camera_commands_thread('start_up_pimms') #Connect to camera

def except_hook(cls, exception, traceback):
    '''If this is not called the UI will not return errors'''
    sys.__excepthook__(cls, exception, traceback)

if __name__ == '__main__':
    sys.excepthook = except_hook
    app = QtWidgets.QApplication(sys.argv)
    #Call the class for your respective delay stage
    delay_stage = newport_delay_stage(fd)

    #Create window
    w = MainWindow()
    #Load the app
    w.show()

    app.exec()
