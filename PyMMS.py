#####################################################################################
# Variables found throughout this file are defined as:
# 1) UI variables from image.ui are: _name
# 2) UI variables generated in this file are: name_
# 3) All other variables are lowercase: name
#####################################################################################

#Native python library imports
import os
import queue
import serial.tools.list_ports
import sys
import time
#Pip install library imports, found in install folder
#Pythonnet clr library is imported at bottom of file in __main__
#this library conflicts with PyQt if loaded before mainwindow
import h5py
import numpy as np
import yaml
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyQt6 import uic, QtWidgets, QtCore, QtGui
#Functions to import that relate to PIMMS ideflex.dll
from PyMMS_Functions import pymms

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
    def __init__(self, size, dly_connected, dly_steps, parent=None):
        QtCore.QThread.__init__(self, parent)
        self.running = True
        self.dly_connected = dly_connected
        self.size = size
        self.image_queue = queue.Queue(maxsize=2)
        self.dly_steps = dly_steps

    def run(self):
        current_shot = 0
        while self.running:
            if self.dly_connected == True:
                #Get position from delay stage
                pos = myDLS.TP()[1]
            else:
                pos = False
            #Get image from camera
            a = pymms.idflex.readImage(size = self.size)
            #If any of the queues are full skip them
            try: self.image_queue.put_nowait([pos,a])
            except queue.Full: pass
            if self.dly_connected == True and self.dly_steps['saving'] == True:
                current_shot+=1
                if current_shot >= self.dly_steps['frames']:
                    self.dly_steps['start']+=self.dly_steps['step']
                    myDLS.PA_Set(self.dly_steps['start'])
                    current_shot = 0
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
        tracker = 0
        fps=0
        timer = time.time()
        save_timer = time.time()
        frm_image = []
        pos_data = []
        cml_image = np.zeros((324,324),dtype=np.float32)
        stopped = False
        self.window_._frame_count.setText('0')

        #Continue checking the image queue until we kill the camera.
        while self.running == True:
            #Update the fps count every second
            if time.time() - timer > 1:
                self.window_._fps_1.setText(f'{fps} fps')
                self.window_._frame_count.setText(f'{shot_number}')
                timer = time.time()
                fps = 0
                tracker = 0

            #Save data every 30 seconds to reduce disk writes
            if time.time() - save_timer > 5:
                if self.window_._save_box.isChecked():
                    self.save_data(filename,frm_image,shot_number,pos_data)
                frm_image = [] #Remove all frames from storage
                pos_data = []
                save_timer = time.time()

            #Stop acquiring images when limit is reached
            if shot_number >= self.window_._n_of_frames.value() and self.window_._n_of_frames.value() != 0 and self.window_._save_box.isChecked() == True:
                self.window_._frame_count.setText(f'{shot_number}')
                shot_number = 0
                self.limit.emit()

            #If queue is empty the program will crash if you try to check it
            if image_queue.empty():
                pass
            else:
                shot_number+=1
                fps+=1

                #Get image array from queue
                position, images = image_queue.get_nowait()
                if position: 
                    pos_data.append(position)
                    #Stop the process if the camera passes the stop point
                    if self.window_._dly_step.value() > 0 and self.window_._save_box.isChecked() == True:
                        if position > (self.window_._dly_stop.value()+self.window_._dly_step.value()) and stopped == False:
                            stopped = True
                            self.limit.emit()
                    else:
                        if position < (self.window_._dly_stop.value()+self.window_._dly_step.value()) and stopped == False:
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

                if (time.time() - timer) > (0.1 * tracker):
                    tracker+=1

                    #Determine colourmap
                    cm_text = self.window_._colourmap.currentText()
                    if cm_text == 'None':
                        cm = plt.get_cmap('gray')
                    else:
                        cm = plt.get_cmap(cm_text)

                    #If the user wants to refresh the cumulative image clear array
                    if self.window_.reset_cml_ == True:
                        cml_image = np.zeros((324,324),dtype=np.float32)
                        cml_number = 1
                        self.window_.reset_cml_ = False

                    #Get tof plot data before getting live view image
                    tof = np.zeros(4096)
                    #Get the ToF data from the tof_data list
                    if self.window_._tof_view.currentIndex() != 4: tof_data = tof_data[self.window_._tof_view.currentIndex()]
                    try:
                        uniques, counts = np.unique(np.vstack(tof_data)[:,-2].flatten(), return_counts=True) 
                        tof[uniques] = counts
                    except (IndexError,ValueError):
                        pass
                        # print(tof_data)
                        # self.window_.update_console('Cannot understand data from camera, is it being triggered correctly?')
                        # self.stop()
                    total_ions = int(np.sum(tof))
                    self.window_._ion_count.setText(str(total_ions))
                    self.window_._tofp.ln.set_ydata(tof)

                    #Update the ion count
                    del self.window_.ion_counts_[0]
                    self.window_.ion_counts_.append(total_ions)
                    self.window_.ionc_.ln.set_ydata(np.array(self.window_.ion_counts_))
                    #If it is not the first image
                    self.window_.ionc_.axes.set_ylim(np.nanmin(self.window_.ion_counts_)-1, np.nanmax(self.window_.ion_counts_)+1)
                    self.window_.ionc_.ymin.set_text(np.nanmin(self.window_.ion_counts_)-1)
                    self.window_.ionc_.yavg.set_text(int(np.nanmean(self.window_.ion_counts_)))
                    self.window_.ionc_.ymax.set_text(np.nanmax(self.window_.ion_counts_)+1)

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
                        image[image > (self.window_._vmax.value()*0.01)] = 1.0

                    #Create an RGB image from the array
                    colour_image = (cm(image)[:,:,:3])
                    self.window_._image_ref.set_data(colour_image)

                    self.window_._imgc.draw()
                    self.window_._tofp.draw()

                    #Update image canvas
                    cml_number+=1

        if self.window_._save_box.isChecked():
            self.save_data(filename,frm_image,shot_number,pos_data)

        self.finished.emit()
        print('Camera stopping.')

    def stop(self):
        self.running = False

#########################################################################################
# Canvas for plotting ToF and image data
#########################################################################################
class BlitManager:
    def __init__(self, canvas, animated_artists=()):
        """
        Parameters
        ----------
        canvas : FigureCanvasAgg
            The canvas to work with, this only works for subclasses of the Agg
            canvas which have the `~FigureCanvasAgg.copy_from_bbox` and
            `~FigureCanvasAgg.restore_region` methods.

        animated_artists : Iterable[Artist]
            List of the artists to manage

        Copied from matplotlib's website:
        https://matplotlib.org/stable/tutorials/advanced/blitting.html
        """
        self.canvas = canvas
        self._bg = None
        self._artists = []

        for a in animated_artists:
            self.add_artist(a)
        # grab the background on every draw
        self.cid = canvas.mpl_connect("draw_event", self.on_draw)

    def on_draw(self, event):
        """Callback to register with 'draw_event'."""
        cv = self.canvas
        if event is not None:
            if event.canvas != cv:
                raise RuntimeError
        self._bg = cv.copy_from_bbox(cv.figure.bbox)
        self._draw_animated()

    def add_artist(self, art):
        """
        Add an artist to be managed.

        Parameters
        ----------
        art : Artist

            The artist to be added.  Will be set to 'animated' (just
            to be safe).  *art* must be in the figure associated with
            the canvas this class is managing.

        """
        if art.figure != self.canvas.figure:
            raise RuntimeError
        art.set_animated(True)
        self._artists.append(art)

    def _draw_animated(self):
        """Draw all of the animated artists."""
        fig = self.canvas.figure
        for a in self._artists:
            fig.draw_artist(a)

    def update(self):
        """Update the screen with animated artists."""
        cv = self.canvas
        fig = cv.figure
        # paranoia in case we missed the draw event,
        if self._bg is None:
            self.on_draw(None)
        else:
            # restore the background
            cv.restore_region(self._bg)
            # draw all of the animated artists
            self._draw_animated()
            # update the GUI state
            cv.blit(fig.bbox)

class MplCanvas(FigureCanvasQTAgg):
    '''
    Matplotlib canvas for TOF readout data.

    Added to the _grid_matplotlib object in the image.ui
    '''
    def __init__(self, parent=None, width=7, height=3, dpi=72):
        super(MplCanvas, self).__init__(Figure(figsize=(width, height), dpi=dpi))
        self.window_ = parent
        plt.rc('axes', labelsize=2)# fontsize of the x and y labels
        #Adjusts the constraints of the plot so it best fits the grid
        self.figure.subplots_adjust(
            top=0.896,
            bottom=0.135,
            left=0.125,
            right=0.964,
            hspace=0.1,
            wspace=0.115
        )
        self.axes = self.figure.add_subplot(111)
        #Set default grid limits
        self.axes.set_xlim(self.window_._min_x.value(), self.window_._max_x.value())
        self.axes.set_ylim(self.window_._min_y.value(), self.window_._max_y.value())
        #Change the font size so it is within bounds of grid
        for label in (self.axes.get_xticklabels() + self.axes.get_yticklabels()):
            label.set_fontsize(7)
        (self.ln,) = self.axes.plot(np.zeros(4096,),'r', animated=True) #putting it in () returns actor
        self.bm = BlitManager(self.figure.canvas, [self.ln])

    def mouseDoubleClickEvent(self, event):
        '''
        If user double clicks plot expand it
        '''
        event.ignore() #Ignore click
        if self.window_.tof_expanded_ == False:
            self.window_.ui_plots.pop_out_window(0) 

class IonCountCanvas(FigureCanvasQTAgg):
    '''
    Matplotlib canvas for TOF readout data.

    Added to the _ion_count_matplotlib object in the image.ui
    '''
    def __init__(self, parent=None, width=7, height=3, dpi=72):
        super(IonCountCanvas, self).__init__(Figure(figsize=(width, height), dpi=dpi))
        self.window_ = parent
        plt.rc('axes', labelsize=2)# fontsize of the x and y labels
        #Adjusts the constraints of the plot so it best fits the grid
        self.figure.subplots_adjust(
            top=0.896,
            bottom=0.135,
            left=0.125,
            right=0.964,
            hspace=0.1,
            wspace=0.115
        )
        self.axes = self.figure.add_subplot(111)
        #Set default grid limits
        self.axes.set_xlim(0, len(self.window_.ion_counts_))
        self.axes.get_yaxis().set_ticks([])
        self.ymin = self.axes.annotate('0',(-35,-5),xycoords='axes points')
        bbox = self.figure.get_window_extent().transformed(self.figure.dpi_scale_trans.inverted())
        width, height = bbox.width*self.figure.dpi, bbox.height*self.figure.dpi
        self.yavg = self.axes.annotate('0.5',(-35,height/4),xycoords='axes points')
        self.ymax = self.axes.annotate('1',(-35,height/2),xycoords='axes points')
        #Change the font size so it is within bounds of grid
        for label in (self.axes.get_xticklabels() + self.axes.get_yticklabels()):
            label.set_fontsize(7)
        (self.ln,) = self.axes.plot(self.window_.ion_counts_,'r', animated=True) #putting it in () returns actor
        self.bm = BlitManager(self.figure.canvas, [self.ln,self.ymin,self.yavg,self.ymax])

    def mouseDoubleClickEvent(self, event):
        '''
        If user double clicks plot expand it
        '''
        event.ignore() #Ignore click
        if self.window_.ionc_expanded_ == False:
            self.window_.ui_plots.pop_out_window(2) 

class image_canvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=1, height=1, dpi=72):
        super(image_canvas, self).__init__(Figure(figsize=(width, height), dpi=dpi))
        self.window_ = parent
        self.figure.subplots_adjust(
            top=1.0,
            bottom=0,
            left=0,
            right=1.0,
            hspace=0,
            wspace=0
        )
        self.axes = self.figure.add_subplot(111)
        self.axes.set_axis_off()

    def mouseDoubleClickEvent(self, event):
        '''
        If user double clicks plot expand it
        '''
        event.ignore() #Ignore click
        if self.window_.img_expanded_ == False:
            self.window_.ui_plots.pop_out_window(1) 

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
        self.window_._tofp = MplCanvas(self.window_, width=5, height=4)
        toolbar = NavigationToolbar(self.window_._tofp, self.window_)
        layout = QtWidgets.QGridLayout()
        layout.addWidget(toolbar)
        layout.addWidget(self.window_._tofp)
        self.window_._grid_matplotlib.setLayout(layout) #Add plot to grid
        self.window_._grid_matplotlib.origpos = self.window_._grid_matplotlib.pos()
        self.window_._grid_matplotlib.origwidth = self.window_._grid_matplotlib.width()
        self.window_._grid_matplotlib.origheight = self.window_._grid_matplotlib.height()
        self.window_._grid_matplotlib.installEventFilter(self.window_)
        self.window_._grid_matplotlib.setWindowTitle('ToF Spectrum')
        self.window_._min_x.valueChanged.connect(self.change_axes)
        self.window_._max_x.valueChanged.connect(self.change_axes)
        self.window_._min_y.valueChanged.connect(self.change_axes)
        self.window_._max_y.valueChanged.connect(self.change_axes)
        '''
        Setup plot for image readout
        '''
        self.window_._imgc = image_canvas(self.window_)
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self.window_._imgc)
        self.window_._image_widget.setLayout(layout)
        self.window_._image_widget.setWindowTitle('Readout')
        self.window_._image_widget.origpos = self.window_._image_widget.pos()
        self.window_._image_widget.origwidth = self.window_._image_widget.width()
        self.window_._image_widget.origheight = self.window_._image_widget.height()
        self.window_._image_widget.installEventFilter(self.window_)
        self.window_._image_ref = self.window_._imgc.axes.imshow(np.zeros((324,324,3)),origin='lower') #Create an empty img
        '''
        Setup plot for ion count
        '''
        self.window_.ionc_ = IonCountCanvas(self.window_, width=5, height=4)
        toolbar = NavigationToolbar(self.window_.ionc_, self.window_)
        layout = QtWidgets.QGridLayout()
        layout.addWidget(toolbar)
        layout.addWidget(self.window_.ionc_)
        self.window_._ion_count_matplotlib.setLayout(layout) #Add plot to grid
        self.window_._ion_count_matplotlib.origpos = self.window_._ion_count_matplotlib.pos()
        self.window_._ion_count_matplotlib.origwidth = self.window_._ion_count_matplotlib.width()
        self.window_._ion_count_matplotlib.origheight = self.window_._ion_count_matplotlib.height()
        self.window_._ion_count_matplotlib.installEventFilter(self.window_)
        self.window_._ion_count_matplotlib.setWindowTitle('Ion Count')

    def change_axes(self):
        '''
        Updates the TOF axes to make it easier to view data.
        '''
        self.window_._tofp.axes.set_xlim(self.window_._min_x.value(), self.window_._max_x.value())
        self.window_._tofp.axes.set_ylim(self.window_._min_y.value(), self.window_._max_y.value())
        self.window_._tofp.draw()

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
            self.window_._grid_matplotlib.setParent(None)
            self.window_._grid_matplotlib.move(int(self.window_.width()/2),int(self.window_.height()/2))
            self.window_._grid_matplotlib.show()
        elif option == 1: #if user is popping out image
            self.window_.img_expanded_ = True
            self.window_._image_widget.setParent(None)
            self.window_._image_widget.move(int(self.window_.width()/2),int(self.window_.height()/2))
            self.window_._image_widget.show()
        else:
            self.window_.ionc_expanded_ = True
            self.window_._ion_count_matplotlib.setParent(None)
            self.window_._ion_count_matplotlib.move(int(self.window_.width()/2),int(self.window_.height()/2))
            self.window_._ion_count_matplotlib.show()           

#########################################################################################
# Classes used for threading various elements of the UI
#########################################################################################
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
            value = myDLS.TP()
            try: self.progressChanged.emit(value[1])
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
            dly_step = {
                'frames': self.window_._dly_img.value(),
                'start': self.window_._dly_start.value(),
                'step': self.window_._dly_step.value(),
                'saving':  self.window_._save_box.isChecked()
                }
            worker = ImageAcquisitionThread(size, self.window_.dly_connected, dly_step)
        else: 
            worker = run_camera(self.window_)
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
        if self.window_.camera_running_ == False:
            #Check if the delay stage is connected, if it is move to starting position
            if self.window_.dly_connected == True:
                self.window_.dly_position()
                while (self.window_._dly_start.value()-0.01 < myDLS.TP()[1] < self.window_._dly_start.value()+0.01) != True:
                    time.sleep(0.001)
            #self.window_.timer.start(50)
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
            #self.window_.timer.stop()
            #When we want to stop a thread we need to tell the worker (index 1) to stop
            for thread in self.window_.threads_:
                thread[1].stop()
                thread[0].quit()
                thread[0].wait()
            self.window_._update_camera.setDisabled(False)
            self.window_._update_camera_2.setDisabled(False)
            self.window_._camera_connect_button.setDisabled(False)
            self.window_._dly_connect_button.setDisabled(False)
            self.window_.camera_running_ = False
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
        self.ion_counts_ = [np.nan for x in range(50)]
        self._vthp.setValue(defaults['dac_settings']['vThP'])
        self._vthn.setValue(defaults['dac_settings']['vThN'])
        self._bins.setValue(defaults['ControlSettings']['Mem Reg Read'][0])
        [self._colourmap.addItem(x) for x in plt.colormaps()] #Add colourmaps

        #Call the class responsible for plot drawing and functions
        self.ui_plots = UI_Plots(self)
        #Class the class responsible for handling threads
        self.ui_threads = UI_Threads(self)

        #Update plots on a timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateplots)

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

    def updateplots(self):
        #self.ionc_.bm.update()
        pass

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
            if source.windowTitle() == self._grid_matplotlib.windowTitle():
                self.tof_expanded_ = False
                self._grid_matplotlib.setParent(self._tab)
                self._grid_matplotlib.setGeometry(
                    self._grid_matplotlib.origpos.x(),
                    self._grid_matplotlib.origpos.y(),
                    self._grid_matplotlib.origwidth,
                    self._grid_matplotlib.origheight
                )
                self._grid_matplotlib.show()
            elif source.windowTitle() == self._image_widget.windowTitle():
                self.img_expanded_ = False
                self._image_widget.setParent(self._tab)
                self._image_widget.setGeometry(
                    self._image_widget.origpos.x(),
                    self._image_widget.origpos.y(),
                    self._image_widget.origwidth,
                    self._image_widget.origheight
                )
                self._image_widget.show()
            else:
                self.ionc_expanded_ = False
                self._ion_count_matplotlib.setParent(self._tab)
                self._ion_count_matplotlib.setGeometry(
                    self._ion_count_matplotlib.origpos.x(),
                    self._ion_count_matplotlib.origpos.y(),
                    self._ion_count_matplotlib.origwidth,
                    self._ion_count_matplotlib.origheight
                )
                self._ion_count_matplotlib.show()                          
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
            #If camera is connected disconnect it before closing
            if self.connected_ == True:
                if self.camera_running_ == True:
                    self.run_camera_threads()
                    time.sleep(0.5) #Sleep between commands or it crashes
                pymms.close_pimms()
            self._grid_matplotlib.close() #Close tof widget
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
        ports = serial.tools.list_ports.comports()
        com_list = []
        for port, desc, hwid in sorted(ports):
            if 'PID=104D:3009' in hwid:
                com_list.append(f"{port}; Delay Stage ; {hwid}")
            else:
                com_list.append(f"{port}; {desc} ; {hwid}")
        self._com_ports.addItems(com_list)

    def dly_control(self):
        '''Connect to and disconnect from the delay stage.'''
        if self.dly_connected == False:
            port, name, _ = self._com_ports.currentText().split(';')
            if 'Delay Stage' in name:
                myDLS.OpenInstrument(port)
                self._dly_vel.setText(str(myDLS.VA_Get()[1]))
                self._dly_pos_min.setText(str(myDLS.SL_Get()[1]))
                self._dly_pos_max.setText(str(myDLS.SR_Get()[1]))
                self.dly_connected = True
                self._dly_connect_button.setText('Disconnect')
                self.ui_threads.dly_pos_thread()
        else:
            self.ui_threads.dly_worker.stop()
            myDLS.CloseInstrument()
            self._dly_connect_button.setText('Connect')
            self.dly_connected = False

    def dly_velocity(self):
        '''Change the velocity of the delay stage'''
        myDLS.VA_Set(self._dly_vel_set.value()) #Maximum velocity is 300 mm/s
        self._dly_vel.setText(str(myDLS.VA_Get()[1]))

    def dly_position(self):
        '''Move delay stage to position specified.'''
        myDLS.PA_Set(self._dly_start.value())

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

    #Delay stage dll
    #This must be imported after QApplication for some reason???
    import clr #pip install pythonnet, NOT clr
    file_name = 'Newport.DLS.CommandInterfaceDLS'
    sys.path.append(fd)
    clr.AddReference(file_name)
    from CommandInterfaceDLS import DLS
    myDLS = DLS() #DLS is imported from CommandInterfaceDLS

    #Create window
    w = MainWindow()
    #Load the app
    w.show()

    app.exec()
