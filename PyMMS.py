#####################################################################################
# Variables found throughout this file are defined as:
# 1) UI variables from image.ui are: _name
# 2) UI variables generated in this file are: name_
# 3) All other variables are lowercase: name
#####################################################################################

#Native python library imports
import os
import queue
import sys
import time
#Pip install library imports, found in install folder
#Pythonnet clr library is imported at bottom of file in __main__
#this library conflicts with PyQt if loaded before mainwindow
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
import yaml
import warnings
from PyQt6 import uic, QtWidgets, QtCore, QtGui
#Functions to import that relate to PIMMS ideflex.dll
from PyMMS_Functions_Spoof import pymms
#Import the newport class function
from Delay_Stage import newport_delay_stage
#Supress numpy divide and cast warnings
warnings.simplefilter('ignore', RuntimeWarning)

#Before we proceed check if we can load Parameter file
#Directory containing this file
fd = os.path.dirname(__file__)
try:
    with open(f'{fd}/PyMMS_Defaults.yaml', 'r') as stream:
        defaults = yaml.load(stream,Loader=yaml.SafeLoader)
except FileNotFoundError:
    print("Cannot find the Parameters file (PyMMS_Defaults.yaml), where did it go?")
    exit()

#filename for the ui file
uifn = "image.ui"
#create path to ui file
uifp = os.path.join(fd,uifn)
if os.path.isfile(uifp) == False:
    print("Cannot find the ui file (image.ui), where did it go?")
    exit() 

#Create PyMMS object
pymms = pymms()

#Create operation modes, these could change in the future
defaults = pymms.operation_modes(defaults)

#########################################################################################
# Classes for grabbing and processing images from PImMS
#########################################################################################
class ImageAcquisitionThread(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    def __init__(self, size=5):
        super(ImageAcquisitionThread, self).__init__()
        self.running = True
        self.size = size
        self.image_queue = queue.Queue(maxsize=2)

    def run(self):
        #Get image from camera
        while self.running:
            a = pymms.idflex.readImage(size = self.size)
            try: self.image_queue.put_nowait(a)
            except queue.Full: pass
        print("Image acquisition has stopped")
        self.finished.emit()

    def stop(self):
        self.running = False

class run_camera(QtCore.QObject):
    '''
    Main function for analysing frames from the camera.
    '''
    finished = QtCore.pyqtSignal()
    limit = QtCore.pyqtSignal() #indicate when frames == limit
    progress = QtCore.pyqtSignal(str)

    def __init__(self, window):
        super(run_camera, self).__init__()
        self.running = True
        self.window_ = window

    def extract_tof_data(self,mem_regs):
        '''
        Parse images and get the X,Y,T data from each mem reg
        '''
        #Skip analogue array if it exists
        tof_data = []
        for id, reg in enumerate(mem_regs):
            Y,X = np.nonzero(reg)
            ToF = reg[Y,X]
            ids = np.full((ToF.shape),id)
            #These are arrays that are flat, we want to stack them and then transpose
            st = np.transpose(np.vstack((X,Y,ToF,ids)))
            tof_data.append(np.array(st,dtype=np.int16))
        return tof_data

    def parse_analogue(self,img):
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

    def save_data(self,filename,frm_image,shot_number,pos_data):
        '''
        Save each frame stored in frm_image to the pickled file
        '''
        with h5py.File(filename, 'a') as hf:
            try: hf.create_dataset(f'{shot_number}',data=np.vstack(frm_image),compression='gzip')
            except ValueError: print('No data to save?')
        #If there is delay stage data save it
        if len(pos_data) == 0: return
        with h5py.File(f'{filename[:-3]}_position.h5', 'a') as hf:
            try: hf.create_dataset(f'{shot_number}',data=np.vstack(pos_data),compression='gzip')
            except ValueError: print('No data to save?')

    def run(self):       
        print('Starting camera.')
        #Make a queue to store image data
        image_queue = self.window_.threads_[0][1].image_queue

        #Check what type of images are being output from the camera, 0 anlg + exp, 1 exp
        img_types = self.window_.camera_function_

        #If the user is saving data create the file
        if self.window_._save_box.isChecked():
            #Create a filename for saving data
            filename = os.path.join(self.window_._dir_name.text(),self.window_._file_name.text())
            filename += '_0000.h5'

            #Check if file already exists so you dont overwrite data
            fid = 1
            while os.path.exists(filename):
                filename = f'{filename[:-8]}_{"{:04d}".format(fid)}.h5'
                fid+=1

        #Variables to keep track of fps and averaging
        cml_number = 0
        shot_number = 0
        fps=0
        timer = time.time()
        save_timer = time.time()
        frm_image = []
        pos_data = []
        cml_image = np.zeros((324,324),dtype=np.float32)
        stopped = False
        self.window_._frame_count.setText('0')

        #Delay stage end position
        if self.window_.dly_connected:
            delay_start = self.window_._delay_t0.value() + (self.window_._dly_start.value() / 6671.2819)
            delay_step = self.window_._dly_step.value() / 6671.2819
            delay_end = self.window_._delay_t0.value() + (self.window_._dly_stop.value() / 6671.2819)
            delay_end += delay_step #include the delay end in the position search
            delay_current = delay_start
            delay_shots = 0

        #Continue checking the image queue until we kill the camera.
        while self.running == True:
            #Update the fps count every second
            if time.time() - timer > 1:
                self.window_._fps_1.setText(f'{fps} fps')
                self.window_._frame_count.setText(f'{shot_number}')
                timer = time.time()
                fps = 0

            #Save data every 30 seconds to reduce disk writes
            if time.time() - save_timer > 5:
                if self.window_._save_box.isChecked():
                    self.save_data(filename,frm_image,shot_number,pos_data)
                frm_image = [] #Remove all frames from storage
                pos_data = []
                save_timer = time.time()

            #Stop acquiring images when limit is reached
            if shot_number >= self.window_._n_of_frames.value() and self.window_._n_of_frames.value() != 0 and self.window_._save_box.isChecked() and not stopped:
                self.window_._frame_count.setText(f'{shot_number}')
                stopped = True
                shot_number = 0
                self.limit.emit()

            #If queue is empty the program will crash if you try to check it
            if image_queue.empty():
                pass
            else:
                shot_number+=1
                fps+=1

                #Get image array from queue
                images = image_queue.get_nowait()

                #Check if the user is saving delay position data
                if self.window_.dly_connected and self.window_._save_box.isChecked():
                    position = float(delay_stage.get_position())
                    delay_shots+=1
                    #If we have reached the max number of images for this position
                    if delay_shots == self.window_._dly_img.value():
                        self.progress.emit(f'Moved to: {round(position,4)}, End at: {round(delay_end,4)}')
                        delay_current+=delay_step
                        delay_stage.set_position(delay_current)
                        delay_shots = 0
                    #If we have passed the end point stop camera
                    if (delay_start > delay_end) and (delay_current <= delay_end) and (not stopped): 
                        print(delay_current, delay_end)
                        stopped = True
                        self.limit.emit()
                    if (delay_start < delay_end) and (delay_current >= delay_end) and (not stopped):
                        print(delay_current, delay_end)
                        stopped = True
                        self.limit.emit()
                                        
                img_index = self.window_._window_view.currentIndex()

                '''
                Grab TOF data, then get unique values and update plot.
                '''
                #Get the X, Y, ToF data from each mem reg
                if img_types == 0: 
                    tof_data = self.extract_tof_data(images[1:])
                else:
                    tof_data = self.extract_tof_data(images)
                    img_index-=1 #There is no analogue image here so remove one index

                frm_image.append(np.vstack(tof_data))
                frm_image.append(np.zeros((4,),dtype=np.int16))
                #After frame data is saved save position data
                if self.window_.dly_connected and self.window_._save_box.isChecked(): 
                    pos_data.append(position)

                #If the user wants to refresh the cumulative image clear array
                if self.window_.reset_cml_ == True:
                    cml_image = np.zeros((324,324),dtype=np.float32)
                    cml_number = 1
                    self.window_.reset_cml_ = False

                #Get tof plot data before getting live view image
                tof = np.zeros(4096)
                #Get the ToF data from the tof_data list
                if self.window_._tof_view.currentIndex() != 4: tof_data = tof_data[self.window_._tof_view.currentIndex()]
                #If the camera doesn't have the voltages set correctly it will sometimes see nothing
                try:
                    uniques, counts = np.unique(np.vstack(tof_data)[:,-2].flatten(), return_counts=True) 
                    tof[uniques] = counts
                except (IndexError,ValueError):
                    pass
                total_ions = int(np.sum(tof))
                self.window_.ion_count_displayed = total_ions
                self.window_.tof_counts_ = tof

                #Update the ion count
                del self.window_.ion_counts_[0]
                self.window_.ion_counts_.append(total_ions)

                '''
                After TOF data has been plotted, convert into img format.
                '''
                #Make an empty array so there is always something to convert
                image = np.zeros((324,324))
                #If the user wants to plot the analogue image
                if img_types == 0 and self.window_._window_view.currentIndex() == 0: 
                    image = self.parse_analogue(images[0])
                else:
                    #Remove any ToF outside of range
                    image = images[img_index]
                    image[image > self.window_._max_x.value()] = 0
                    image[image < self.window_._min_x.value()] = 0
                image = ((image / np.max(image)))
                image = np.rot90(image, self.window_.rotation_)

                if self.window_._view.currentIndex() == 1:
                    cml_image = ((cml_image  * (cml_number - 1)) / cml_number) + (image / cml_number)
                    image = cml_image
                    image = ((image / np.max(image)))
                
                # Scale the image based off the slider
                if self.window_._vmax.value() != 100:
                    image[image > (self.window_._vmax.value()*0.01)] = 0
                    image = ((image / np.max(image)))

                self.window_.image_ = np.array(image * 255,dtype=np.uint8)

                #Update image canvas
                cml_number+=1

        if self.window_._save_box.isChecked():
            self.save_data(filename,frm_image,shot_number,pos_data)

        self.finished.emit()
        print('Camera stopping.')

    def stop(self):
        self.running = False
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
        
        self.window_.ion_count_plot_origpos = self.window_._ion_count_plot.pos()
        self.window_.ion_count_plot_origwidth = self.window_._ion_count_plot.width()
        self.window_.ion_count_plot_origheight = self.window_._ion_count_plot.height()
        self.window_.ion_count_plot_line = self.window_._ion_count_plot.plot(self.window_.ion_counts_,pen=pg.mkPen(color=(255, 0, 0)))
        self.window_._ion_count_plot.setWindowTitle('Ion Count Spectrum')
        self.window_._ion_count_plot.setBackground('w')
        self.window_._ion_count_plot.installEventFilter(self.window_)

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
# Classes used for threading various elements of the UI
#########################################################################################
class update_plots(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    update = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        QtCore.QThread.__init__(self, parent)
        self.running = True

    def run(self):
        while self.running == True:
            self.update.emit()
            QtCore.QThread.msleep(100)
        self.finished.emit()

    def stop(self):
        self.running = False

class get_dly_position(QtCore.QObject):
    '''
    Thread reading position data of delay stage for UI
    '''
    progressChanged = QtCore.pyqtSignal(float)
    finished = QtCore.pyqtSignal()

    def __init__(self,parent=None):
        QtCore.QThread.__init__(self, parent)
        self.running = True

    def run(self):
        while self.running == True:
            value = delay_stage.get_position()
            try: self.progressChanged.emit(float(value))
            except: print('Cannot read position data.')
            time.sleep(0.5)
        self.finished.emit()

    def stop(self):
        self.running = False

class progress_bar(QtCore.QObject):
    progressChanged = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal()

    def __init__(self,parent=None):
        QtCore.QThread.__init__(self, parent)
        self.running = True

    def run(self):
        while self.running == True:
            self.progressChanged.emit(np.random.randint(low=1,high=100))
            time.sleep(0.2)
        self.progressChanged.emit(0)
        self.finished.emit()

    def stop(self):
        self.running = False

class camera_commands_thread(QtCore.QObject):
    '''
    Thread controls camera updates so the UI is not locked.
    This allows the UI to update the progress bar for the user.
    '''
    progressChanged = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()

    def __init__(self, function=0, trim_file=None, trigger=0, parent=None):
        QtCore.QThread.__init__(self, parent)
        self.function = function
        self.trim_file = trim_file
        self.trigger = trigger

    def find_dll(self):
        #Load the PIMMS dll file and report if it was successful or not
        ret = pymms.idflex.open_dll()
        self.progressChanged.emit(ret)
        self.finished.emit()

    def turn_on(self):
        ret = pymms.turn_on_pimms(defaults)
        self.progressChanged.emit(ret)
        self.finished.emit()

    def update_dacs(self):
        ret = pymms.start_up_pimms(defaults,self.trim_file,self.function,self.trigger)
        self.progressChanged.emit(ret)
        self.finished.emit()

    def update_readout(self):
        ret = pymms.send_output_types(defaults,self.function,self.trigger)
        self.progressChanged.emit(ret)
        self.finished.emit()

class UI_Threads():
    '''
    Threads for running the UI. 

    Work by creating a pyqt thread object and passing that object to a worker.

    When the worker terminates it passes a finished signal which terminates the thread.
    '''    
    def __init__(self, main_window):
        self.window_ = main_window

    def update_plot_thread(self):
        self.udpl_thread = QtCore.QThread()
        self.udpl_worker = update_plots()
        self.udpl_worker.moveToThread(self.udpl_thread)
        self.udpl_worker.update.connect(self.window_.update_plots)
        self.udpl_worker.finished.connect(self.udpl_thread.quit)
        self.udpl_worker.finished.connect(self.udpl_worker.deleteLater)
        self.udpl_thread.started.connect(self.udpl_worker.run)
        self.udpl_thread.finished.connect(self.udpl_thread.deleteLater)
        self.udpl_thread.start()

    def dly_pos_thread(self):
        # Step 1: Create a QThread object
        self.dly_thread = QtCore.QThread()
        # Step 2: Create a worker object
        self.dly_worker = get_dly_position()
        # Step 3: Move worker to the thread
        self.dly_worker.moveToThread(self.dly_thread)
        self.dly_thread.started.connect(self.dly_worker.run)
        self.dly_worker.finished.connect(self.dly_thread.quit)
        self.dly_worker.finished.connect(self.dly_worker.deleteLater)
        self.dly_thread.finished.connect(self.dly_thread.deleteLater)
        # Step 4: When progress changed tell the UI
        self.dly_worker.progressChanged.connect(self.window_.update_pos)
        # Step 5: Start the thread
        self.dly_thread.start()

    def pb_thread(self):
        self.window_._progressBar.setFormat('Communicating with camera.')
        # Step 1: Create a QThread object
        self._pb_thread = QtCore.QThread()
        # Step 2: Create a worker object
        self._pb_worker = progress_bar()
        # Step 3: Move worker to the thread
        self._pb_worker.moveToThread(self._pb_thread)
        self._pb_thread.started.connect(self._pb_worker.run)
        self._pb_worker.finished.connect(lambda: self.window_._progressBar.setFormat(''))
        self._pb_worker.finished.connect(self._pb_thread.quit)
        self._pb_worker.finished.connect(self._pb_worker.deleteLater)
        self._pb_thread.finished.connect(self._pb_thread.deleteLater)
        # Step 4: When progress changed tell the UI
        self._pb_worker.progressChanged.connect(self.window_.update_pb)
        # Step 5: Start the thread
        self._pb_thread.start()

    def camera_thread(self,function,trim_file=None,subfunction=0,trigger=0):
        '''
        This thread handles communication for updates with the camera.
        '''
        # Step 1: Create a QThread object
        self._up_thread = QtCore.QThread()
        # Step 2: Create a worker object
        self._up_worker = camera_commands_thread(subfunction,trim_file,trigger)
        # Step 3: Move worker to the thread
        self._up_worker.moveToThread(self._up_thread)
        # Step 4: Connect signals and slots
        if function == 0: self._up_thread.started.connect(self._up_worker.find_dll)
        if function == 1: 
            self._up_thread.started.connect(self._up_worker.turn_on)
            self._up_thread.finished.connect(self.window_.lock_camera_connect)
        if function == 2: 
            self._up_thread.started.connect(self._up_worker.update_dacs)
            self._up_thread.finished.connect(self.window_.unlock_run_camera)
        if function == 3: 
            self._up_thread.started.connect(self._up_worker.update_readout)
            self._up_thread.finished.connect(self.window_.unlock_run_camera)
        self._up_worker.progressChanged.connect(self.window_.update_console)
        #Tell the progress bar to stop running
        self._up_worker.finished.connect(lambda: self._pb_worker.stop())
        #Kill the worker and thread
        self._up_worker.finished.connect(self._up_thread.quit)
        self._up_worker.finished.connect(self._up_worker.deleteLater)
        self._up_thread.finished.connect(self._up_thread.deleteLater)
        # Step 5: Start the thread
        self._up_thread.start()

    def image_processing_thread(self,function,size=5):
        '''
        Controls the camera readout threads.

        Function 0 creates a thread for getting frames from the camera

        Function 1 creates a thread for processing the frames
        '''
        # Step 1: Create a QThread object
        thread = QtCore.QThread()
        # Step 2: Create a worker object
        if function == 0: 
            worker = ImageAcquisitionThread(size)
        else: 
            worker = run_camera(self.window_)
            worker.progress.connect(self.window_.update_console)
            worker.limit.connect(self.run_camera_threads)
        # Step 3: Move worker to the thread
        worker.moveToThread(thread)
        # Step 4: Connect signals and slots
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        worker.finished.connect(thread.deleteLater)
        return [thread, worker]

    def run_camera_threads(self):
        '''
        This function generates the threads required for running the camera
        '''
        if not self.window_.camera_running_:
            #Check if the delay stage is connected and user is saving data
            if self.window_.dly_connected and self.window_._save_box.isChecked():
                start = self.window_._delay_t0.value() - (self.window_._dly_start.value() / 6671.2819)
                delay_stage.set_position(start)
                position = delay_stage.get_position()
                while (start-0.01 < float(position) < start+0.01) != True:
                    self.window_.update_pb(np.random.randint(0,100))
                    self.window_.update_console(f'Moving to start position: {round(start,4)}, Current position: {round(position,4)}')
                    position = delay_stage.get_position()
                    time.sleep(0.01)
                self.window_.update_console(f'Finished moving to start position: {position}')
                self.window_.update_pb(0)
            self.update_plot_thread()
            #Check if the user is running experimental with analogue or not
            size = 5
            if self.window_._exp_type.currentIndex() == 1: size = 4
            #Disable camera options so user doesnt get confused or break things
            self.window_.camera_running_ = True
            self.window_.camera_function_ = self.window_._exp_type.currentIndex()
            self.window_._button.setText("Stop")
            self.window_._update_camera.setDisabled(True)
            self.window_._update_camera_2.setDisabled(True)
            self.window_._camera_connect_button.setDisabled(True)
            self.window_._dly_connect_button.setDisabled(True)
            #Empty any pre-existing threads
            self.window_.threads_.clear()
            #Threads 1&2 have been removed due to issues communicating with the PIMMS
            #camera and can be readded once communication is sorted out
            self.window_.threads_ = [self.image_processing_thread(0,size),
                                    self.image_processing_thread(1)]
            #The objects in self.threads_ are the thread and worker, start the thread
            for thread in self.window_.threads_:
                thread[0].start()
        else:
            self.window_.camera_running_ = False
            #stop the plot update thread
            self.udpl_worker.stop()
            self.udpl_thread.quit()
            self.udpl_thread.wait()
            #When we want to stop a thread we need to tell the worker (index 1) to stop
            for thread in self.window_.threads_:
                thread[1].stop()
                thread[0].quit()
                thread[0].wait()
            self.window_._update_camera.setDisabled(False)
            self.window_._update_camera_2.setDisabled(False)
            self.window_._camera_connect_button.setDisabled(False)
            self.window_._dly_connect_button.setDisabled(False)
            self.window_._button.setText("Start")

#########################################################################################
# Mainwindow used for displaying UI
#########################################################################################
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        #Load the ui file
        uic.loadUi(uifp,self)

        #Make some UI variables
        self.threads_ = []
        self.rotation_ = 0 # Rotation angle of the image
        self.connected_ = False #Is camera connected?
        self.dly_connected = False #Is the delay stage connected?
        self.camera_running_ = False #Is camera running?
        self.camera_programmed_ = False
        self.tof_expanded_ = False
        self.img_expanded_ = False
        self.ionc_expanded_ = False
        self.reset_cml_ = False
        self.error_ = False #Is there an error?
        self.camera_function_ = 0
        self.ion_count_displayed = 0
        self.ion_counts_ = [np.nan for x in range(50)]
        self.tof_counts_ = np.zeros(4096)
        self.image_ = np.zeros((324,324,3))
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

        self._button.clicked.connect(self.ui_threads.run_camera_threads)

        #Update the plots when they are clicked on
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

        self.ui_threads.pb_thread() #Progress bar for connecting to camera
        self.ui_threads.camera_thread(0)

    def update_plots(self):
        self.ion_count_plot_line.setData(self.ion_counts_)
        self.tof_plot_line.setData(self.tof_counts_)
        self.graphics_view_.setImage(self.image_, levels=[0,255])
        self._ion_count.setText(str(self.ion_count_displayed))

    def open_file_dialog(self,option):
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

    def update_console(self,string):
        '''
        Update console log with result of command passed to PIMMS.
        '''
        if 'Cannot' in string:
            self.ui_threads._pb_worker.stop
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

    def closeEvent(self, event):
        '''If the application is closed do the following'''
        close = QtWidgets.QMessageBox()
        close.setText("Would you like to quit?")
        close.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Yes | 
                                    QtWidgets.QMessageBox.StandardButton.No)
        close = close.exec()

        if close == QtWidgets.QMessageBox.StandardButton.Yes.value:
            if self.dly_connected:
                delay_stage.disconnect_stage()
            #If camera is connected disconnect it before closing
            if self.connected_:
                if self.camera_running_:
                    self.run_camera_threads()
                    time.sleep(0.5) #Sleep between commands or it crashes
                pymms.close_pimms()
            self._image_widget.close()
            event.accept()
        else:
            event.ignore()

    def reset_images(self):
        '''Reset the cumulative images'''
        if self.reset_cml_ == False:
            self.reset_cml_ = True

    def update_pb(self, value):
        '''Set the progressbar value'''
        self._progressBar.setValue(value)

    #####################################################################################
    # Delay Stage Functions
    #####################################################################################
    def update_pos(self,value):
        '''Updates the position of the delay stage on the UI'''
        self._dly_pos.setText(str(value))

    def update_coms(self):
        '''Updates the available coms on the computer'''
        self._com_ports.clear()
        com_list = delay_stage.get_com_ports()
        self._com_ports.addItems(com_list)

    def dly_control(self):
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

    def dly_velocity(self):
        '''Change the velocity of the delay stage'''
        delay_stage.set_velocity(self._dly_vel_set.value())
        self._dly_vel.setText(delay_stage.get_velocity())

    def dly_position(self):
        '''Move delay stage to position specified.'''
        delay_stage.set_position(self._delay_t0.value())

    #####################################################################################
    # PIMMS Functions
    #####################################################################################
    def lock_camera_connect(self):
        '''
        Only connect to the camera and enable sending commands if there are no issues
        '''
        if self.connected_ == False and self.error_ == False:
            self._camera_connect.setText(f'Connected to Camera')
            self._camera_connect.setStyleSheet("color: green")
            self._camera_connect_button.setText(f'Disconnect')
            self._update_camera.setDisabled(False)
            self.connected_ = True      
        else:
            self._camera_connect.setText(f'Disconnected from camera')
            self._camera_connect.setStyleSheet("color: red")
            self._camera_connect_button.setText(f'Connect')
            self._button.setDisabled(True)
            self._update_camera.setDisabled(True)
            self._update_camera_2.setDisabled(True)
            self.connected_ = False
            self.camera_programmed_ = False
        self._camera_connect_button.setDisabled(False)

    def update_dafaults(self):
        '''
        Update user controllable default values
        '''
        defaults['dac_settings']['vThP'] = self._vthp.value()
        defaults['dac_settings']['vThN'] = self._vthn.value()
        defaults['ControlSettings']['Mem Reg Read'][0] = self._bins.value()

    def camera_control(self):
        '''
        Connect to and disconnect from the camera.
        '''
        if self.connected_ == False:
            self._plainTextEdit.setPlainText(f'Connecting to Camera, please wait!')
            self._camera_connect_button.setDisabled(True)
            self.ui_threads.pb_thread() #Progress bar for connecting to camera
            self.ui_threads.camera_thread(1) #Connect to camera
        else:
            self._plainTextEdit.setPlainText(f'Disconnecting from Camera, please wait!')
            #Disconnect from the camera
            ret = pymms.close_pimms()
            if ret != 0:
                self._plainTextEdit.setPlainText(f'{ret}')
            else:
                self._plainTextEdit.setPlainText(f'Disconnected from Camera!')
            self.lock_camera_connect()

    def unlock_run_camera(self):
        self._button.setDisabled(False)
        self._update_camera.setDisabled(False)
        self._update_camera_2.setDisabled(False)
        self._camera_connect_button.setDisabled(False)
        self.camera_programmed_ = True     

    def update_output(self):
        '''
        If user just wants to update the readout type

        Check that camera has been connected, updated and is not running.
        '''
        self._camera_connect_button.setDisabled(True)
        self._button.setDisabled(True)
        self._update_camera.setDisabled(True)
        self._update_camera_2.setDisabled(True)
        self._plainTextEdit.setPlainText(f'Updating Camera output type!')
        trigger = self._trigger.currentIndex()
        function = self._exp_type.currentIndex()
        self.ui_threads.pb_thread() #Progress bar for connecting to camera
        self.ui_threads.camera_thread(3,subfunction=function,trigger=trigger) #Connect to camera

    def update_camera(self):
        '''
        Update the DAC, camera settings, and TRIM file.
        '''
        if self.connected_ == False: 
            self._plainTextEdit.setPlainText(f'Please connect to the camera before updating output!')
            return
        self._camera_connect_button.setDisabled(True)
        self._button.setDisabled(True)
        self._update_camera.setDisabled(True)
        self._update_camera_2.setDisabled(True)
        trim_file  = self._trim_dir.text()
        function = self._exp_type.currentIndex()
        trigger = self._trigger.currentIndex()
        if os.path.isfile(trim_file) == True:
            self.ui_threads.pb_thread() #Progress bar for connecting to camera
            self.ui_threads.camera_thread(2,trim_file,function,trigger) #Connect to camera
            self._plainTextEdit.setPlainText(f'Updating Camera DACs with trim file!')
        else:
            self.ui_threads.pb_thread() #Progress bar for connecting to camera
            self.ui_threads.camera_thread(2,subfunction=function,trigger=trigger) #Connect to camera
            self._plainTextEdit.setPlainText(f'Updating Camera DACs without trim file!')

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
