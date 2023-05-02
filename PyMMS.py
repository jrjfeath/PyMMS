import os
import queue
import sys
import time

import yaml
import numpy as np
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

class ImageAcquisitionThread(QtCore.QObject):
    finished = QtCore.pyqtSignal()
    def __init__(self, size, parent=None):
        QtCore.QThread.__init__(self, parent)
        self._running = True
        self._size = size
        self._image_queue = queue.Queue(maxsize=2)

    def get_output_queue(self):
        return self._image_queue  

    def run(self):
        while self._running:
            try:
                #Get image from camera
                a = pymms.idflex.readImage(size = self._size)
                self._image_queue.put_nowait(a)
            except queue.Full:
                # No point in keeping this image around when the queue is full, let's skip to the next one
                pass
            except Exception as error:
                print("Encountered error: {error}, image acquisition will stop.".format(error=error))
                break
        print("Image acquisition has stopped")
        self.finished.emit()

    def stop(self):
        self._running = False

class run_camera(QtCore.QObject):
    '''
    Main function for analysing frames from the camera.
    '''
    finished = QtCore.pyqtSignal()

    def __init__(self, window):
        super(run_camera, self).__init__()
        self._isRunning = False
        self._window = window

    def parse_images(self,img_array):
        '''
        Convert PIMMS output into an img format for displaying data
        '''
        x = 0
        #If the user is looking at analogue and digital images
        if len(img_array) == 5:
            img = img_array[0]
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
            img = ((img / np.max(img)) * 255)
            img_array[0] = img
            x+=1
        #After processing or skipping analogue modify digital
        for a in range(4):
            img = img_array[x+a]
            img_array[x+a] = ((img / np.max(img)) * 255)
        return img_array

    def run(self):
        if not self._isRunning:
            self._isRunning = True
        
        print('Starting camera.')
        #Make a queue to store image data
        image_queue = self._window.threads_[0][1].get_output_queue()
        #Variables to keep track of fps and averaging
        shot_number = 0
        fps = 0
        start = time.time()
        save_timer = time.time()

        #Continue checking the image queue until we kill the camera.
        while self._isRunning == True:
            #Update the fps count every second
            if time.time() - start > 1:
                self._window._fps.setText(f'{fps} fps')
                start = time.time()
                fps = 0

            #If queue is empty the program will crash if you try to check it
            if image_queue.empty():
                pass
            else:
                fps+=1
                #Determine colourmap
                cm_text = self._window._colourmap.currentText()
                if cm_text == 'None':
                    cm = plt.get_cmap('gray')
                else:
                    cm = plt.get_cmap(cm_text)

                #Get image array from queue
                images = image_queue.get_nowait()

                '''
                Grab TOF data, get unique values and update plot.
                '''
                #Get tof plot data before getting live view image
                tof = np.zeros((4096,))
                #If user is doing exp w. anal
                if len(images) == 5:
                    #If user wants cumulative ToF Spectrum
                    if self._window._tof_view.currentIndex() == 4:
                        tof_i = images[1:]
                    else: #Plot specific ToF Spectrum
                        tof_i = images[self._window._tof_view.currentIndex()+1]
                else:
                    if self._window._tof_view.currentIndex() == 4:
                        tof_i = images
                    tof_i = images[self._window._tof_view.currentIndex()]
                uniques, counts = np.unique(tof_i.flatten(), return_counts=True)
                tof[uniques] = counts
                self._window._plot_ref.set_ydata(tof)
                self._window._tofp.draw()

                '''
                After TOF data has been plotted, convert into img format.
                '''
                images = self.parse_images(images)
                if len(images) == 4:
                    images = np.insert(images,0,np.zeros((324,324)),axis=0)
                image = images[self._window._window_view.currentIndex()]
                if self._window._window_view.currentIndex() != 0:
                    image[image > self._window._max_x.value()] = 0
                    image[image < self._window._min_x.value()] = 0
                image = np.rot90(image, self._window._rotation)

                #Create an RGB image from the array
                if (shot_number % 2) == 0:
                    colour_image = (cm(image)[:,:,:3] * 255).astype(np.uint8)
                    self._window._image_ref.set_data(colour_image)

                    #Update image canvas
                    self._window._imgc.draw()
                shot_number+=1

        self.finished.emit()
        print('Camera stopping.')

    def stop(self):
        self._isRunning = False

class camera_commands_thread(QtCore.QObject):
    '''
    Thread controls camera updates so the UI is not locked.
    This allows the UI to update the progress bar for the user.
    '''
    progressChanged = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()

    def __init__(self, function=0, trim_file=None, parent=None):
        QtCore.QThread.__init__(self, parent)
        self.function = function
        self.trim_file = trim_file

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
        ret = pymms.start_up_pimms(defaults,self.trim_file,self.function)
        self.progressChanged.emit(ret)
        self.finished.emit()

    def update_readout(self):
        ret = pymms.send_output_types(defaults,self.function)
        self.progressChanged.emit(ret)
        self.finished.emit()

class progress_bar(QtCore.QObject):
    progressChanged = QtCore.pyqtSignal(int)
    finished = QtCore.pyqtSignal()

    def __init__(self, delay,parent=None):
        QtCore.QThread.__init__(self, parent)
        self.delay = delay
        self._isrunning = True

    def run(self):
        delay = self.delay
        delay *= 10
        count = 1
        while count <= delay:
            if self._isrunning == False: break
            self.progressChanged.emit(int((count / delay) * 100))
            count += 1
            time.sleep(0.1)
        self.progressChanged.emit(0)
        self.finished.emit()

class MplCanvas(FigureCanvasQTAgg):
    '''
    Matplotlib canvas for TOF readout data.

    Added to the _grid_matplotlib object in the image.ui
    '''

    def __init__(self, parent=None, width=7, height=3, dpi=400):
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
            self._window.pop_out_window(0) 

class image_canvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=1, height=1, dpi=100):
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
            self._window.pop_out_window(1) 

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
        self._error = False #Is there an error?
        self._programming = False #Is the camera updating?
        self._vthp.setValue(defaults['dac_settings']['vThP'])
        self._vthn.setValue(defaults['dac_settings']['vThN'])
        self._bins.setValue(defaults['ControlSettings']['Mem Reg Read'][0])
        [self._colourmap.addItem(x) for x in plt.colormaps()] #Add colourmaps

        '''
        Setup plot for TOF data

        Plot only updates the data and does not redraw each update.

        Axes are updated by looking at value changes.
        '''
        self._tofp = MplCanvas(self, width=5, height=4, dpi=100)
        toolbar = NavigationToolbar(self._tofp, self)
        layout = QtWidgets.QGridLayout()
        layout.addWidget(toolbar)
        layout.addWidget(self._tofp)
        self._grid_matplotlib.setLayout(layout) #Add plot to grid
        self._grid_matplotlib.origpos = self._grid_matplotlib.pos()
        self._grid_matplotlib.origwidth = self._grid_matplotlib.width()
        self._grid_matplotlib.origheight = self._grid_matplotlib.height()
        self._grid_matplotlib.installEventFilter(self)
        self._grid_matplotlib.setWindowTitle('ToF Spectrum')
        plot_refs = self._tofp.axes.plot(np.zeros(4096,), 'r') #Create an empty plot
        self._plot_ref = plot_refs[0] #Create a reference to overwrite
        self._min_x.valueChanged.connect(self.change_axes)
        self._max_x.valueChanged.connect(self.change_axes)
        self._min_y.valueChanged.connect(self.change_axes)
        self._max_y.valueChanged.connect(self.change_axes)
        '''
        Setup plot for image readout
        '''
        self._imgc = image_canvas(self)
        layout = QtWidgets.QGridLayout()
        layout.addWidget(self._imgc)
        self._image_widget.setLayout(layout)
        self._image_widget.setWindowTitle('Readout')
        self._image_widget.origpos = self._image_widget.pos()
        self._image_widget.origwidth = self._image_widget.width()
        self._image_widget.origheight = self._image_widget.height()
        self._image_widget.installEventFilter(self)
        self._image_ref = self._imgc.axes.imshow(np.zeros((324,324,3))) #Create an empty img
        '''
        Various commands for buttons found on the ui
        '''
        self._camera_connect_button.clicked.connect(self.camera_control)
        self._rotate_c.clicked.connect(lambda: self.rotate_camera(0))
        self._rotate_cc.clicked.connect(lambda: self.rotate_camera(1))
        self._update_camera.clicked.connect(self.update_camera)
        self._button.clicked.connect(self.run_camera_threads)
        self._vthp.valueChanged.connect(self.update_dafaults)
        self._vthn.valueChanged.connect(self.update_dafaults)
        self._bins.valueChanged.connect(self.update_dafaults)
        #Sometimes during idle the code tries to quit, prevent it
        quit = QtGui.QAction("Quit", self)
        quit.triggered.connect(self.closeEvent)

        self.camera_thread(0)

    def change_axes(self):
        '''
        Updates the TOF axes to make it easier to view data.
        '''
        self._tofp.axes.set_xlim(self._min_x.value(), self._max_x.value())
        self._tofp.axes.set_ylim(self._min_y.value(), self._max_y.value())
        self._tofp.draw()

    def rotate_camera(self,option):
        #Rotate clockwise
        if option == 0:
            self._rotation += 1
            if self._rotation == 4: self._rotation = 0
        #Rotate counter clockwise
        else:
            self._rotation -= 1
            if self._rotation < 0: self._rotation = 3

    def open_file_dialog(self,option):
        options = QtWidgets.QFileDialog.Options()
        options |= QtWidgets.QFileDialog.DontUseNativeDialog
        dirname = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory")
        if dirname:
            if option == 1:
                self._dir_name.setText(dirname)
            else:
                self._trim_name.setText(dirname)

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

    def pop_out_window(self,option):
        if option == 0: #If user is popping out tof
            self._tof_expanded = True
            self._grid_matplotlib.setParent(None)
            self._grid_matplotlib.move(int(self.width()/2),int(self.height()/2))
            self._grid_matplotlib.show()
        else: #if user is popping out image
            self._img_expanded = True
            self._image_widget.setParent(None)
            self._image_widget.move(int(self.width()/2),int(self.height()/2))
            self._image_widget.show()

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

    def update_console(self,string):
        '''
        Update console log with result of command passed to PIMMS.
        '''
        print(string)
        if 'Cannot' in string:
            self._pb_worker._isrunning = False
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
            self._connected = False
        self._camera_connect_button.setDisabled(False)

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
            self.camera_thread(1) #Connect to camera
            self.pb_thread(6) #Progress bar for connecting to camera
        else:
            self._plainTextEdit.setPlainText(f'Disconnecting from Camera, please wait!')
            #Disconnect from the camera
            ret = pymms.close_pimms()
            if ret != 0:
                self._plainTextEdit.setPlainText(f'{ret}')
            else:
                self._plainTextEdit.setPlainText(f'Disconnected from Camera!')
            self.lock_camera_connect()

    def update_output(self):
        '''
        If user just wants to update the readout type

        Check that camera has been connected, updated and is not running.
        '''
        if self._connected == False: return
        if self._camera_programmed == False: return
        if self._camera_running == False: return
        if self._exp_type.currentIndex() == 0:
            pymms.send_output_types(defaults,0)
        else:
            pymms.send_output_types(defaults,1)

    def update_camera(self):
        '''
        Update the DAC, camera settings, and TRIM file.
        '''
        trim_file  = self._trim_dir.text()
        function = self._exp_type.currentIndex()
        if os.path.isfile(trim_file) == True:
            self.camera_thread(2,trim_file,function) #Connect to camera
            self.pb_thread(3) #Progress bar for connecting to camera
            self._plainTextEdit.setPlainText(f'Updating Camera DACs with trim file!')
        else:
            self.camera_thread(2,subfunction=function) #Connect to camera
            self.pb_thread(3) #Progress bar for connecting to camera
            self._plainTextEdit.setPlainText(f'Updating Camera DACs without trim file!')
        self._button.setDisabled(False)
        self._camera_programmed = True

    '''
    Threads for running the UI. 

    Work by creating a pyqt thread object and passing that object to a worker.

    When the worker terminates it passes a finished signal which terminates the thread.
    '''
    def pb_thread(self,delay):
        # Step 1: Create a QThread object
        self._pb_thread = QtCore.QThread()
        # Step 2: Create a worker object
        self._pb_worker = progress_bar(delay)
        # Step 3: Move worker to the thread
        self._pb_worker.moveToThread(self._pb_thread)
        self._pb_thread.started.connect(self._pb_worker.run)
        self._pb_worker.finished.connect(self._pb_thread.quit)
        self._pb_worker.finished.connect(self._pb_worker.deleteLater)
        self._pb_thread.finished.connect(self._pb_thread.deleteLater)
        self._pb_worker.progressChanged.connect(self.update_pb)
        # Step 5: Start the thread
        self._pb_thread.start()        

    def camera_thread(self,function,trim_file=None,subfunction=0):
        '''
        This thread handles communication for updates with the camera.
        '''
        # Step 1: Create a QThread object
        self._up_thread = QtCore.QThread()
        # Step 2: Create a worker object
        self._up_worker = camera_commands_thread(subfunction,trim_file)
        # Step 3: Move worker to the thread
        self._up_worker.moveToThread(self._up_thread)
        # Step 4: Connect signals and slots
        if function == 0: self._up_thread.started.connect(self._up_worker.find_dll)
        if function == 1: 
            self._up_thread.started.connect(self._up_worker.turn_on)
            self._up_thread.finished.connect(self.lock_camera_connect)
        if function == 2: self._up_thread.started.connect(self._up_worker.update_dacs)
        if function == 3: self._up_thread.started.connect(self._up_worker.update_readout)
        self._up_worker.progressChanged.connect(self.update_console)
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
        else: worker = run_camera(self)
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
        if self._camera_running == False:
            #Check if the user is running experimental with analogue or not
            size = 5
            if self._exp_type.currentIndex() == 1: size = 4
            #Disable camera options so user doesnt get confused or break things
            self._camera_running = True
            self._button.setText("Stop")
            self._update_camera.setDisabled(True)
            #Empty any pre-existing threads
            self.threads_.clear()
            self.threads_ = [self.image_processing_thread(0,size),self.image_processing_thread(1)]
            #The objects in self.threads_ are the thread and worker, start the thread
            for thread in self.threads_:
                thread[0].start()
        else:
            #When we want to stop a thread we need to tell the worker (index 1) to stop
            for thread in self.threads_:
                thread[1].stop()
                thread[0].quit()
                thread[0].wait()
            self._update_camera.setDisabled(False)
            self._camera_running = False
            self._button.setText("Start")

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
