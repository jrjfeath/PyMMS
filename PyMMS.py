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
# Classes for grabbing and processing images from PImMS
#########################################################################################
class ImageAcquisitionThread(QtCore.QObject):
    '''
    Thread for getting images from PIMMS.

    Size and pymms class passed after thread initialised
    '''
    finished = QtCore.pyqtSignal()
    def __init__(self) -> None:
        super(ImageAcquisitionThread, self).__init__()
        self.running : bool = True
        self.size : int = 5
        self.image_queue = queue.Queue(maxsize=2)

    def run(self) -> None:
        #Get image from camera
        while self.running:
            image = self.pymms.idflex.readImage(size = self.size)
            try: self.image_queue.put_nowait(image)
            except queue.Full: pass
        print("Image acquisition has stopped")
        self.finished.emit()

    def stop(self) -> None:
        self.running = False

class run_camera(QtCore.QObject):
    '''
    Main function for analysing frames from the camera.
    '''
    finished = QtCore.pyqtSignal()
    limit = QtCore.pyqtSignal() #indicate when frames == limit
    progress = QtCore.pyqtSignal(str)

    def __init__(self, window) -> None:
        super(run_camera, self).__init__()
        self.running : bool = True
        self.analogue : bool = True    
        self.bins : int = 4
        self.window_ = window

        #Delay stage end position
        if self.window_.dly_connected:
            self.delay_start : float = self.window_._delay_t0.value() + (self.window_._dly_start.value() / 6671.2819)
            self.delay_step : float = self.window_._dly_step.value() / 6671.2819
            self.delay_end : float = self.window_._delay_t0.value() + (self.window_._dly_stop.value() / 6671.2819)
            self.delay_current : float = self.delay_start
            self.delay_shots : int = 0
            self.stopped : bool = False
            self.position : float = 0.0
            self.delay_end += self.delay_step #include the delay end in the position search

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

    def save_data(self,filename,frm_image,shot_number,pos_data) -> None:
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
        if (self.delay_start > self.delay_end) and (delay_current <= self.delay_end) and (not self.stopped): 
            print(delay_current, self.delay_end)
            self.stopped = True
            self.limit.emit()
        if (self.delay_start < self.delay_end) and (delay_current >= self.delay_end) and (not self.stopped):
            print(delay_current, self.delay_end)
            self.stopped = True
            self.limit.emit()

    def run(self) -> None:       
        print('Starting camera.')

        #Variables to keep track of fps and averaging
        cml_number : int = 0
        shot_number : int = 0
        fps : int = 0
        timer : float = time.time()
        save_timer : float = time.time()
        frm_image : list = []
        pos_data : list = []
        cml_image : np.ndarray = np.zeros((324,324),dtype=np.float32)
        self.window_._frame_count.setText('0')

        #Make a queue to store image data
        image_queue = self.window_.threads_[0][1].image_queue

        #If the user is saving data create the file
        if self.window_._save_box.isChecked():
            #Create a filename for saving data
            filename = os.path.join(self.window_._dir_name.text(),self.window_._file_name.text())
            filename += '_0000.h5'

            #Check if file already exists so you dont overwrite data
            fid = 1
            while os.path.exists(filename):
                filename = f'{filename[:-8]}_{fid:04d}.h5'
                fid+=1

        # Continue checking the image queue until we kill the camera.
        while self.running == True:
            # Update the fps count every second
            if time.time() - timer > 1:
                self.window_._fps_1.setText(f'{fps} fps')
                self.window_._frame_count.setText(f'{shot_number}')
                timer = time.time()
                fps = 0

            # Save data every 30 seconds to reduce disk writes
            if time.time() - save_timer > 5:
                if self.window_._save_box.isChecked():
                    self.save_data(filename,frm_image,shot_number,pos_data)
                frm_image = [] #Remove all frames from storage
                pos_data = []
                save_timer = time.time()

            # Stop acquiring images when limit is reached
            if shot_number >= self.window_._n_of_frames.value() and self.window_._n_of_frames.value() != 0 and self.window_._save_box.isChecked() and not stopped:
                self.window_._frame_count.setText(f'{shot_number}')
                stopped = True
                self.limit.emit()

            # If queue is empty the program will crash if you try to check it
            if image_queue.empty():
                pass
            else:
                shot_number+=1
                fps+=1

                # Get image array from queue
                images = image_queue.get_nowait()
                if images.shape[0] != (self.bins + self.analogue) : continue

                # Check if the user is saving delay position data
                if self.window_.dly_connected and self.window_._save_box.isChecked():
                    self.delay_stage_position()
                                        
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
                frm_image.append(np.vstack(tof_data))
                frm_image.append(np.zeros((4,),dtype=np.int16))

                # After frame data is saved save position data
                if self.window_.dly_connected and self.window_._save_box.isChecked(): 
                    pos_data.append(self.position)

                # If the user wants to refresh the cumulative image clear array
                if self.window_.reset_cml_ == True:
                    cml_image = np.zeros((324,324),dtype=np.float32)
                    cml_number = 1
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

                if self.window_._view.currentIndex() == 1:
                    cml_image = ((cml_image  * (cml_number - 1)) / cml_number) + (image / cml_number)
                    image = cml_image
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
                cml_number+=1

        if self.window_._save_box.isChecked():
            self.save_data(filename,frm_image,shot_number,pos_data)
            shot_number = 0

        self.finished.emit()
        print('Camera stopping.')

    def stop(self) -> None:
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

class progress_bar(QtCore.QObject):
    progressChanged = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal()

    def __init__(self,parent=None) -> None:
        QtCore.QThread.__init__(self, parent)
        self.running = True

    def run(self) -> None:
        while self.running == True:
            self.progressChanged.emit(np.random.randint(low=1,high=100))
            time.sleep(0.2)
        self.progressChanged.emit(0)
        self.finished.emit()

    def stop(self) -> None:
        self.running = False

class camera_commands_thread(QtCore.QObject):
    '''
    Thread controls camera updates so the UI is not locked.
    This allows the UI to update the progress bar for the user.
    '''
    progressChanged = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()

    def __init__(self, parent=None) -> None:
        QtCore.QThread.__init__(self, parent)
        self.trim_file : str = ""
        self.function : int = 0
        self.trigger : int = 0
        self.rows : int = 5

    def turn_on(self) -> None:
        self.pymms.turn_on_pimms()
        self.progressChanged.emit(self.pymms.idflex.message)
        self.finished.emit()

    def update_dacs(self) -> None:
        self.pymms.start_up_pimms(self.trim_file,self.function,self.trigger,self.rows)
        self.progressChanged.emit(self.pymms.idflex.message)
        self.finished.emit()

    def update_readout(self) -> None:
        self.pymms.send_output_types(self.function,self.trigger,self.rows)
        self.progressChanged.emit(self.pymms.idflex.message)
        self.finished.emit()

class UI_Threads():
    '''
    Threads for running the UI. 

    Work by creating a pyqt thread object and passing that object to a worker.

    When the worker terminates it passes a finished signal which terminates the thread.
    '''    
    def __init__(self, main_window) -> None:
        self.window_ = main_window

    def update_plot_thread(self) -> None:
        self.udpl_thread = QtCore.QThread()
        self.udpl_worker = update_plots()
        self.udpl_worker.moveToThread(self.udpl_thread)
        self.udpl_worker.update.connect(self.window_.update_plots)
        self.udpl_worker.finished.connect(self.udpl_thread.quit)
        self.udpl_worker.finished.connect(self.udpl_worker.deleteLater)
        self.udpl_thread.started.connect(self.udpl_worker.run)
        self.udpl_thread.finished.connect(self.udpl_thread.deleteLater)
        self.udpl_thread.start()

    def pb_thread(self) -> None:
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

    def dly_pos_thread(self) -> None:
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

    def camera_thread(self,function : int) -> None:
        '''
        This thread handles communication for updates with the camera.
        '''
        # Step 1: Create a QThread object
        self.threadct = QtCore.QThread()
        # Step 2: Create a worker object
        self.workerct = camera_commands_thread()
        # Step 3: Move worker to the thread
        self.workerct.moveToThread(self.threadct)
        #Pass variables to worker
        self.workerct.function = self.window_._exp_type.currentIndex()
        self.workerct.trim_file = self.window_._trim_dir.text()
        self.workerct.trigger = self.window_._trigger.currentIndex()
        self.workerct.rows = self.window_._bins.value() + 1 - self.window_._exp_type.currentIndex()
        self.workerct.pymms = self.window_.pymms
        # Step 4: Connect signals and slots
        if function == 1: 
            self.threadct.started.connect(self.workerct.turn_on)
            self.threadct.finished.connect(self.window_.lock_camera_connect)
        if function == 2: 
            self.threadct.started.connect(self.workerct.update_dacs)
            self.threadct.finished.connect(self.window_.unlock_run_camera)
        if function == 3: 
            self.threadct.started.connect(self.workerct.update_readout)
            self.threadct.finished.connect(self.window_.unlock_run_camera)
        self.workerct.progressChanged.connect(self.window_.update_console)
        #Tell the progress bar to stop running
        self.workerct.finished.connect(lambda: self._pb_worker.stop())
        #Kill the worker and thread
        self.workerct.finished.connect(self.threadct.quit)
        self.workerct.finished.connect(self.workerct.deleteLater)
        self.threadct.finished.connect(self.threadct.deleteLater)
        # Step 5: Start the thread
        self.threadct.start()

    def image_processing_thread(self,function : int) -> list:
        '''
        Controls the camera readout threads.

        Function 0 creates a thread for getting frames from the camera

        Function 1 creates a thread for processing the frames
        '''

        #Check if the user is running experimental with analogue or not
        bins = self.window_._bins.value()
        analogue = True
        if self.window_._exp_type.currentIndex() == 1:
            analogue = False
        size = bins + analogue

        # Step 1: Create a QThread object
        thread = QtCore.QThread()
        # Step 2: Create a worker object
        if function == 0: 
            worker = ImageAcquisitionThread()
            worker.pymms = self.window_.pymms
            worker.size = size
        else: 
            worker = run_camera(self.window_)
            worker.analogue = analogue
            worker.bins = bins
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

    def run_camera_threads(self) -> None:
        '''
        This function generates the threads required for running the camera
        '''
        if not self.window_.camera_running_:
            #Check if the delay stage is connected and user is saving data
            if self.window_.dly_connected and self.window_._save_box.isChecked():
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
            self.update_plot_thread()
            #Fast Pymms has to start the acquisition
            if hasattr(self.window_.pymms.idflex, 'StartAcquisition'):
                self.window_.pymms.idflex.StartAcquisition()
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
            self.window_.threads_ = [self.image_processing_thread(0),
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
            #Fast Pymms has to close the acquisition
            if hasattr(self.window_.pymms.idflex, 'StopAcquisition'):
                self.window_.pymms.idflex.StopAcquisition()

#########################################################################################
# Mainwindow used for displaying UI
#########################################################################################
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super(MainWindow, self).__init__()

        #Load the ui file
        uic.loadUi(uifp,self)

        #Add colourmaps
        colourmaps = pg.colormap.listMaps("matplotlib")
        self._colourmap.addItems(colourmaps)

        #Make some UI variables
        self.pymms = None
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

        self._button.clicked.connect(self.ui_threads.run_camera_threads)

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
            #If camera is connected disconnect it before closing
            if self.connected_:
                if self.camera_running_:
                    self.run_camera_threads()
                    time.sleep(0.5) #Sleep between commands or it crashes
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
        else:
            self.pymms = None
            self._camera_connect.setText(f'Disconnected')
            self._camera_connect.setStyleSheet("color: red")
            self._camera_connect_button.setText(f'Connect')
            self._button.setDisabled(True)
            self._update_camera.setDisabled(True)
            self._update_camera_2.setDisabled(True)
            self.connected_ = False
            self.camera_programmed_ = False
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
            self.ui_threads.pb_thread() #Progress bar for connecting to camera
            self.ui_threads.camera_thread(1) #Connect to camera
        else:
            self.pymms = None
            self._plainTextEdit.setPlainText(f'Disconnecting from Camera, please wait!')
            #Disconnect from the camera
            self.pymms.idflex.close_device()
            self._plainTextEdit.setPlainText(self.pymms.idflex.message)
            self.lock_camera_connect()

    def unlock_run_camera(self) -> None:
        self._button.setDisabled(False)
        self._update_camera.setDisabled(False)
        self._update_camera_2.setDisabled(False)
        self._camera_connect_button.setDisabled(False)
        self.camera_programmed_ = True     

    def update_output(self) -> None:
        '''
        If user just wants to update the readout type

        Check that camera has been connected, updated and is not running.
        '''
        self._camera_connect_button.setDisabled(True)
        self._button.setDisabled(True)
        self._update_camera.setDisabled(True)
        self._update_camera_2.setDisabled(True)
        self._plainTextEdit.setPlainText(f'Updating Camera output type!')
        self.ui_threads.pb_thread() #Progress bar for connecting to camera
        self.ui_threads.camera_thread(3) #Connect to camera

    def update_camera(self) -> None:
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
        if os.path.isfile(self._trim_dir.text()) == True:
            self._plainTextEdit.setPlainText(f'Updating Camera DACs with trim file!')
        else:
            self._plainTextEdit.setPlainText(f'Updating Camera DACs without trim file!')
        self.ui_threads.pb_thread() #Progress bar for connecting to camera
        self.ui_threads.camera_thread(2) #Connect to camera

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
