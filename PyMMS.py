import os
import queue
import sys
import time

import h5py
import numpy as np
import yaml
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('QtAgg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyMMS_Functions import pymms
from PyQt6 import uic, QtWidgets, QtCore, QtGui

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
# Thread and Functions for grabbing images from PImMS
#########################################################################################
class ImageAcquisitionThread(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    def __init__(self, size, parent=None):
        QtCore.QThread.__init__(self, parent)
        self._running = True
        self._size = size
        self.image_queue = queue.Queue(maxsize=2)
        #self.plot_queue = queue.Queue(maxsize=2)
        #self.tof_queue = queue.Queue(maxsize=2) 

    def run(self):
        while self._running:
            #Get image from camera
            a = pymms.idflex.readImage(size = self._size)
            #If any of the queues are full skip them
            try: self.image_queue.put_nowait(a)
            except queue.Full: pass
            #Extra threads are currently unusable
            #try: self.plot_queue.put_nowait(a)
            #except queue.Full: pass
            #try: self.tof_queue.put_nowait(a)
            #except queue.Full: pass
        print("Image acquisition has stopped")
        self.finished.emit()

    def stop(self):
        self._running = False

class Plot_Image(QtCore.QObject):
    '''
    Class responsible for updating the image output
    '''
    finished = QtCore.pyqtSignal()

    def __init__(self, window):
        super(Plot_Image, self).__init__()
        self.isRunning_ = True
        self.window_ = window

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

    def run(self):
        plot_queue = self.window_.threads_[0][1].plot_queue
        fps = 0
        cml_number = 1
        cml_image = np.zeros((324,324),dtype=np.float32)
        #Check what type of images are being output from the camera, 0 anlg + exp, 1 exp
        img_types = self.window_.camera_function_
        timer = time.time()

        while self.isRunning_ == True:

            #Update the fps count every second
            if time.time() - timer > 1:
                self.window_._fps_2.setText(f'{fps} fps')
                timer = time.time()
                fps = 0

            #Get processed image data
            try: images = plot_queue.get(timeout=1)
            except queue.Empty: continue

            #If the user wants to refresh the cumulative image clear array
            if self.window_.reset_cml_ == True:
                cml_image = np.zeros((324,324),dtype=np.float32)
                cml_number = 1
                self.window_.reset_cml_ = False

            #Determine colourmap
            cm_text = self.window_._colourmap.currentText()
            img_index = self.window_._window_view.currentIndex()
            if cm_text == 'None':
                cm = plt.get_cmap('gray')
            else:
                cm = plt.get_cmap(cm_text)

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
            image = np.rot90(image, self.window_._rotation)

            if self.window_._view.currentIndex() == 1:
                cml_image = ((cml_image  * (cml_number - 1)) / cml_number) + (image / cml_number)
                image = cml_image
                cml_number+=1

            #Create an RGB image from the array
            colour_image = (cm(image)[:,:,:3])
            self.window_._image_ref.set_data(colour_image)

            #Update image canvas
            self.window_._imgc.draw()

            fps+=1

    def stop(self):
        self.isRunning_ = False

class Plot_Tof(QtCore.QObject):
    '''
    Class responsible for updating the ToF output
    '''
    finished = QtCore.pyqtSignal()

    def __init__(self, window):
        super(Plot_Tof, self).__init__()
        self.isRunning_ = True
        self.window_ = window

    def extract_tof_data(self,mem_regs):
        '''
        Parse images and get the X,Y,T data from each mem reg
        '''
        #Skip analogue array if it exists
        tof_data = []
        for reg in mem_regs:
            Y,X = np.nonzero(reg)
            ToF = reg[Y,X]
            st = np.transpose(np.vstack((X,Y,ToF)))
            tof_data.append(np.array(st,dtype=np.int16))
        return tof_data

    def run(self):
        tof_queue = self.window_.threads_[0][1].tof_queue
        tof_index = self.window_._tof_view.currentIndex()
        #Check what type of images are being output from the camera, 0 anlg + exp, 1 exp
        img_types = self.window_.camera_function_
        timer = time.time()
        fps=0
        while self.isRunning_ == True:
            #Update the fps count every second
            if time.time() - timer > 1:
                self.window_._fps_3.setText(f'{fps} fps')
                timer = time.time()
                fps = 0

            if tof_queue.empty():
                continue
            try: images = tof_queue.get(timeout=1)
            except queue.Empty: continue

            if img_types == 0: 
                tof_data = self.extract_tof_data(images[1:])
            else:
                tof_data = self.extract_tof_data(images)

            #Get tof plot data before getting live view image
            tof = np.zeros(4096)
            #Get the ToF data from the tof_data list
            if tof_index != 4:
                uniques, counts = np.unique(tof_data[tof_index].flatten(), return_counts=True)
            else:
                uniques, counts = np.unique(np.vstack(tof_data)[:,-1].flatten(), return_counts=True)
            tof[uniques] = counts
            self.window_._ion_count.setText(str(int(np.sum(tof))))
            self.window_._plot_ref.set_ydata(tof)
            self.window_._tofp.draw()
            fps+=1

    def stop(self):
        self.isRunning_ = False

class run_camera(QtCore.QObject):
    '''
    Main function for analysing frames from the camera.
    '''
    finished = QtCore.pyqtSignal()
    limit = QtCore.pyqtSignal() #indicate when frames == limit

    def __init__(self, window):
        super(run_camera, self).__init__()
        self._isRunning = True
        self._window = window

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

    def save_data(self,filename,frm_image,shot_number):
        '''
        Save each frame stored in frm_image to the pickled file
        '''
        with h5py.File(filename, 'a') as hf:
            try: hf.create_dataset(f'{shot_number}',data=np.vstack(frm_image),compression='gzip')
            except ValueError: print('No data to save?')

    def run(self):       
        print('Starting camera.')
        #Make a queue to store image data
        image_queue = self._window.threads_[0][1].image_queue

        #Check what type of images are being output from the camera, 0 anlg + exp, 1 exp
        img_types = self._window.camera_function_

        #If the user is saving data create the file
        if self._window._save_box.isChecked():
            #Create a filename for saving data
            filename = os.path.join(self._window._dir_name.text(),self._window._file_name.text())
            filename += '_0000.h5'

            #Check if file already exists so you dont overwrite data
            fid = 1
            while os.path.exists(filename):
                filename = f'{filename[:-8]}_{"{:04d}".format(fid)}.h5'
                fid+=1

        #Variables to keep track of fps and averaging
        cml_number = 0
        shot_number = 0
        tracker = 0
        fps=0
        timer = time.time()
        save_timer = time.time()
        frm_image = []
        cml_image = np.zeros((324,324),dtype=np.float32)
        self._window._frame_count.setText('0')

        #Continue checking the image queue until we kill the camera.
        while self._isRunning == True:
            #Update the fps count every second
            if time.time() - timer > 1:
                self._window._fps_1.setText(f'{fps} fps')
                self._window._frame_count.setText(f'{shot_number}')
                timer = time.time()
                fps = 0
                tracker = 0

            #Save data every 30 seconds to reduce disk writes
            if time.time() - save_timer > 5:
                if self._window._save_box.isChecked():
                    self.save_data(filename,frm_image,shot_number)
                frm_image = [] #Remove all frames from storage
                save_timer = time.time()

            #Stop acquiring images when limit is reached
            if shot_number >= self._window._n_of_frames.value() and self._window._n_of_frames.value() != 0 and self._window._save_box.isChecked() == True:
                self._window._frame_count.setText(f'{shot_number}')
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
                img_index = self._window._window_view.currentIndex()

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

                if (time.time() - timer) > (0.1 * tracker):
                    tracker+=1

                    #Determine colourmap
                    cm_text = self._window._colourmap.currentText()
                    if cm_text == 'None':
                        cm = plt.get_cmap('gray')
                    else:
                        cm = plt.get_cmap(cm_text)

                    #If the user wants to refresh the cumulative image clear array
                    if self._window.reset_cml_ == True:
                        cml_image = np.zeros((324,324),dtype=np.float32)
                        cml_number = 1
                        self._window.reset_cml_ = False

                    #Get tof plot data before getting live view image
                    tof = np.zeros(4096)
                    #Get the ToF data from the tof_data list
                    uniques, counts = np.unique(np.vstack(tof_data)[:,-2].flatten(), return_counts=True)
                    tof[uniques] = counts
                    self._window._ion_count.setText(str(int(np.sum(tof))))
                    self._window._plot_ref.set_ydata(tof)
                    self._window._tofp.draw()

                    '''
                    After TOF data has been plotted, convert into img format.
                    '''
                    #Make an empty array so there is always something to convert
                    image = np.zeros((324,324))
                    #If the user wants to plot the analogue image
                    if img_types == 0 and self._window._window_view.currentIndex() == 0: 
                        image = self.parse_analogue(images[0])
                    else:
                        #Remove any ToF outside of range
                        image = images[img_index]
                        image[image > self._window._max_x.value()] = 0
                        image[image < self._window._min_x.value()] = 0
                    image = ((image / np.max(image)))
                    image = np.rot90(image, self._window._rotation)

                    if self._window._view.currentIndex() == 1:
                        cml_image = ((cml_image  * (cml_number - 1)) / cml_number) + (image / cml_number)
                        image = cml_image

                    #Create an RGB image from the array
                    colour_image = (cm(image)[:,:,:3])
                    self._window._image_ref.set_data(colour_image)

                    #Update image canvas
                    self._window._imgc.draw()
                    cml_number+=1

        if self._window._save_box.isChecked():
            self.save_data(filename,frm_image,cml_number)

        self.finished.emit()
        print('Camera stopping.')

    def stop(self):
        self._isRunning = False

#########################################################################################
# Canvas for plotting ToF and image data
#########################################################################################
class MplCanvas(FigureCanvasQTAgg):
    '''
    Matplotlib canvas for TOF readout data.

    Added to the _grid_matplotlib object in the image.ui
    '''

    def __init__(self, parent=None, width=7, height=3, dpi=72):
        self._window = parent
        plt.rc('axes', labelsize=2)# fontsize of the x and y labels
        fig = Figure(figsize=(width, height), dpi=dpi)
        #Adjusts the constraints of the plot so it best fits the grid
        fig.subplots_adjust(
            top=0.896,
            bottom=0.135,
            left=0.125,
            right=0.964,
            hspace=0.1,
            wspace=0.115
        )
        self.axes = fig.add_subplot(111)
        #Set default grid limits
        self.axes.set_xlim(0, 4095)
        self.axes.set_ylim(0, 100000)
        #Change the font size so it is within bounds of grid
        for label in (self.axes.get_xticklabels() + self.axes.get_yticklabels()):
            label.set_fontsize(7)
        super(MplCanvas, self).__init__(fig)

    def mouseDoubleClickEvent(self, event):
        '''
        If user double clicks plot expand it
        '''
        event.ignore() #Ignore click
        if self._window._tof_expanded == False:
            self._window.ui_plots.pop_out_window(0) 

class image_canvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=1, height=1, dpi=72):
        self._window = parent
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.subplots_adjust(
            top=1.0,
            bottom=0,
            left=0,
            right=1.0,
            hspace=0,
            wspace=0
        )
        self.axes = fig.add_subplot(111)
        self.axes.set_axis_off()
        super(image_canvas, self).__init__(fig)

    def mouseDoubleClickEvent(self, event):
        '''
        If user double clicks plot expand it
        '''
        event.ignore() #Ignore click
        if self._window._img_expanded == False:
            self._window.ui_plots.pop_out_window(1) 

#########################################################################################
# Class used for modifying the plots
#########################################################################################
class UI_Plots():
    def __init__(self, main_window):
        self.m_w = main_window
        self.generate_plots()
    
    def generate_plots(self):
        '''
        Setup plot for TOF data

        Plot only updates the data and does not redraw each update.

        Axes are updated by looking at value changes.
        '''
        self.m_w._tofp = MplCanvas(self.m_w, width=5, height=4)
        toolbar = NavigationToolbar(self.m_w._tofp, self.m_w)
        layout = QtWidgets.QGridLayout()
        layout.addWidget(toolbar)
        layout.addWidget(self.m_w._tofp)
        self.m_w._grid_matplotlib.setLayout(layout) #Add plot to grid
        self.m_w._grid_matplotlib.origpos = self.m_w._grid_matplotlib.pos()
        self.m_w._grid_matplotlib.origwidth = self.m_w._grid_matplotlib.width()
        self.m_w._grid_matplotlib.origheight = self.m_w._grid_matplotlib.height()
        self.m_w._grid_matplotlib.installEventFilter(self.m_w)
        self.m_w._grid_matplotlib.setWindowTitle('ToF Spectrum')
        plot_refs = self.m_w._tofp.axes.plot(np.zeros(4096,), 'r') #Create an empty plot
        self.m_w._plot_ref = plot_refs[0] #Create a reference to overwrite
        self.m_w._min_x.valueChanged.connect(self.change_axes)
        self.m_w._max_x.valueChanged.connect(self.change_axes)
        self.m_w._min_y.valueChanged.connect(self.change_axes)
        self.m_w._max_y.valueChanged.connect(self.change_axes)
        '''
        Setup plot for image readout
        '''
        self.m_w._imgc = image_canvas(self.m_w)
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.m_w._imgc)
        self.m_w._image_widget.setLayout(layout)
        self.m_w._image_widget.setWindowTitle('Readout')
        self.m_w._image_widget.origpos = self.m_w._image_widget.pos()
        self.m_w._image_widget.origwidth = self.m_w._image_widget.width()
        self.m_w._image_widget.origheight = self.m_w._image_widget.height()
        self.m_w._image_widget.installEventFilter(self.m_w)
        self.m_w._image_ref = self.m_w._imgc.axes.imshow(np.zeros((324,324,3)),origin='lower') #Create an empty img

    def change_axes(self):
        '''
        Updates the TOF axes to make it easier to view data.
        '''
        self.m_w._tofp.axes.set_xlim(self.m_w._min_x.value(), self.m_w._max_x.value())
        self.m_w._tofp.axes.set_ylim(self.m_w._min_y.value(), self.m_w._max_y.value())
        self.m_w._tofp.draw()

    def rotate_camera(self,option):       
        #Rotate clockwise
        if option == 0:
            self.m_w._rotation += 1
            if self.m_w._rotation == 4: self.m_w._rotation = 0
        #Rotate counter clockwise
        else:
            self.m_w._rotation -= 1
            if self.m_w._rotation < 0: self.m_w._rotation = 3

    def pop_out_window(self,option):
        if option == 0: #If user is popping out tof
            self.m_w._tof_expanded = True
            self.m_w._grid_matplotlib.setParent(None)
            self.m_w._grid_matplotlib.move(int(self.m_w.width()/2),int(self.m_w.height()/2))
            self.m_w._grid_matplotlib.show()
        else: #if user is popping out image
            self.m_w._img_expanded = True
            self.m_w._image_widget.setParent(None)
            self.m_w._image_widget.move(int(self.m_w.width()/2),int(self.m_w.height()/2))
            self.m_w._image_widget.show()

#########################################################################################
# Classes used for threading various elements of the UI
#########################################################################################
class progress_bar(QtCore.QObject):
    progressChanged = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal()

    def __init__(self,parent=None):
        QtCore.QThread.__init__(self, parent)
        self._isRunning = True

    def run(self):
        while self._isRunning == True:
            self.progressChanged.emit(np.random.randint(low=1,high=100))
            time.sleep(0.2)
        self.progressChanged.emit(0)
        self.finished.emit()

    def stop(self):
        self._isRunning = False

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
        self.m_w = main_window

    def pb_thread(self):
        self.m_w._progressBar.setFormat('Communicating with camera.')
        # Step 1: Create a QThread object
        self._pb_thread = QtCore.QThread()
        # Step 2: Create a worker object
        self._pb_worker = progress_bar()
        # Step 3: Move worker to the thread
        self._pb_worker.moveToThread(self._pb_thread)
        self._pb_thread.started.connect(self._pb_worker.run)
        self._pb_worker.finished.connect(lambda: self.m_w._progressBar.setFormat(''))
        self._pb_worker.finished.connect(self._pb_thread.quit)
        self._pb_worker.finished.connect(self._pb_worker.deleteLater)
        self._pb_thread.finished.connect(self._pb_thread.deleteLater)
        # Step 4: When progress changed tell the UI
        self._pb_worker.progressChanged.connect(self.m_w.update_pb)
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
            self._up_thread.finished.connect(self.m_w.lock_camera_connect)
        if function == 2: 
            self._up_thread.started.connect(self._up_worker.update_dacs)
            self._up_thread.finished.connect(self.m_w.unlock_run_camera)
        if function == 3: 
            self._up_thread.started.connect(self._up_worker.update_readout)
            self._up_thread.finished.connect(self.m_w.unlock_run_camera)
        self._up_worker.progressChanged.connect(self.m_w.update_console)
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
        if function == 0: worker = ImageAcquisitionThread(size)
        elif function == 1: worker = Plot_Image(self.m_w)
        elif function == 2: worker = Plot_Tof(self.m_w)
        else: 
            worker = run_camera(self.m_w)
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
        if self.m_w._camera_running == False:
            #Check if the user is running experimental with analogue or not
            size = 5
            if self.m_w._exp_type.currentIndex() == 1: size = 4
            #Disable camera options so user doesnt get confused or break things
            self.m_w._camera_running = True
            self.m_w.camera_function_ = self.m_w._exp_type.currentIndex()
            self.m_w._button.setText("Stop")
            self.m_w._update_camera.setDisabled(True)
            self.m_w._update_camera_2.setDisabled(True)
            self.m_w._camera_connect_button.setDisabled(True)
            #Empty any pre-existing threads
            self.m_w.threads_.clear()
            #Threads 1&2 have been removed due to issues communicating with the PIMMS
            #camera and can be readded once communication is sorted out
            self.m_w.threads_ = [self.image_processing_thread(0,size),
                                 #self.image_processing_thread(1),
                                 #self.image_processing_thread(2),
                                 self.image_processing_thread(3)]
            #The objects in self.threads_ are the thread and worker, start the thread
            for thread in self.m_w.threads_:
                thread[0].start()
        else:
            #When we want to stop a thread we need to tell the worker (index 1) to stop
            for thread in self.m_w.threads_:
                thread[1].stop()
                thread[0].quit()
                thread[0].wait()
            self.m_w._update_camera.setDisabled(False)
            self.m_w._update_camera_2.setDisabled(False)
            self.m_w._camera_connect_button.setDisabled(False)
            self.m_w._camera_running = False
            self.m_w._button.setText("Start")

#########################################################################################
# Mainwindow used for displaying UI
#########################################################################################
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        #Load the ui file
        uic.loadUi(uifp,self)

        self.threads_ = []
        self._rotation = 0 # Rotation angle of the image
        self._connected = False #Is camera connected?
        self._camera_running = False #Is camera running?
        self._camera_programmed = False
        self._tof_expanded = False
        self._img_expanded = False
        self.reset_cml_ = False
        self._error = False #Is there an error?
        self.camera_function_ = 0
        self._vthp.setValue(defaults['dac_settings']['vThP'])
        self._vthn.setValue(defaults['dac_settings']['vThN'])
        self._bins.setValue(defaults['ControlSettings']['Mem Reg Read'][0])
        [self._colourmap.addItem(x) for x in plt.colormaps()] #Add colourmaps

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
        self._button.clicked.connect(self.ui_threads.run_camera_threads)
        self._vthp.valueChanged.connect(self.update_dafaults)
        self._vthn.valueChanged.connect(self.update_dafaults)
        self._bins.valueChanged.connect(self.update_dafaults)
        self._window_view.currentIndexChanged.connect(self.reset_images)
        self._view.currentIndexChanged.connect(self.reset_images)
        self._rotate_c_2.clicked.connect(self.reset_images)
        #Sometimes during idle the code tries to quit, prevent it
        quit = QtGui.QAction("Quit", self)
        quit.triggered.connect(self.closeEvent)

        self.ui_threads.pb_thread() #Progress bar for connecting to camera
        self.ui_threads.camera_thread(0)

    def reset_images(self):
        if self.reset_cml_ == False:
            self.reset_cml_ = True

    def open_file_dialog(self,option):
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
        print(string)
        if 'Cannot' in string:
            self.ui_threads._pb_worker.stop
            self._error = True
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

    def update_pb(self, value):
        self._progressBar.setValue(value)

    def lock_camera_connect(self):
        '''
        Only connect to the camera and enable sending commands if there are no issues
        '''
        if self._connected == False and self._error == False:
            self._camera_connect.setText(f'Connected to Camera')
            self._camera_connect.setStyleSheet("color: green")
            self._camera_connect_button.setText(f'Disconnect')
            self._update_camera.setDisabled(False)
            self._connected = True      
        else:
            self._camera_connect.setText(f'Disconnected from camera')
            self._camera_connect.setStyleSheet("color: red")
            self._camera_connect_button.setText(f'Connect')
            self._button.setDisabled(True)
            self._update_camera.setDisabled(True)
            self._update_camera_2.setDisabled(True)
            self._connected = False
        self._camera_connect_button.setDisabled(False)

    def eventFilter(self, source, event):
        '''
        Capture close events to pop the windows back into the main body
        '''
        if (event.type() == QtCore.QEvent.Type.Close and isinstance(source, QtWidgets.QWidget)):
            if source.windowTitle() == self._grid_matplotlib.windowTitle():
                self._tof_expanded = False
                self._grid_matplotlib.setParent(self)
                self._grid_matplotlib.setGeometry(
                    self._grid_matplotlib.origpos.x(),
                    self._grid_matplotlib.origpos.y()+20,
                    self._grid_matplotlib.origwidth,
                    self._grid_matplotlib.origheight
                )
                self._grid_matplotlib.show()
            else:
                self._img_expanded = False
                self._image_widget.setParent(self)
                self._image_widget.setGeometry(
                    self._image_widget.origpos.x(),
                    self._image_widget.origpos.y()+20,
                    self._image_widget.origwidth,
                    self._image_widget.origheight
                )
                self._image_widget.show()               
            event.ignore() #Ignore close
            return True #Tell qt that process done
        else:
            return super(MainWindow,self).eventFilter(source,event)

    def closeEvent(self, event):
        close = QtWidgets.QMessageBox()
        close.setText("Would you like to quit?")
        close.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Yes | 
                                    QtWidgets.QMessageBox.StandardButton.No)
        close = close.exec()

        if close == QtWidgets.QMessageBox.StandardButton.Yes.value:
            #If camera is connected disconnect it before closing
            if self._connected == True:
                if self._camera_running == True:
                    self.run_camera_threads()
                    time.sleep(0.5) #Sleep between commands or it crashes
                pymms.close_pimms()
            self._grid_matplotlib.close() #Close tof widget
            self._image_widget.close()
            event.accept()
        else:
            event.ignore()

    '''
    Various functions that control the PImMS camera
    '''

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
        if self._connected == False:
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
        self._camera_programmed = True     

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
        if self._connected == False: 
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
    sys.__excepthook__(cls, exception, traceback)

if __name__ == '__main__':
    sys.excepthook = except_hook
    app = QtWidgets.QApplication(sys.argv)
    #Create window
    w = MainWindow()
    #Load the app
    w.show()

    app.exec()
