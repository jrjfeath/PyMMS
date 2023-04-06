import os
import queue
import sys
import time
import threading

import yaml
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PyMMS_Functions import pymms
from PyQt5 import uic, QtWidgets, QtCore

#Before we proceed check if we can load Parameter file
#Directory containing this file
fd = os.path.dirname(__file__)
try:
    with open(f'{fd}/PyMMS_Defaults.yaml', 'r') as stream:
        defaults = yaml.load(stream,Loader=yaml.SafeLoader)
except FileNotFoundError:
    print("Cannot find the Parameters file (PyMMS_Defaults.yaml), where did it go?")
    sys.exit()

#filename for the ui file
uifn = "image.ui"
#create path to ui file
uifp = os.path.join(fd,uifn)
if os.path.isfile(uifp) == False:
    print("Cannot find the ui file (image.ui), where did it go?")
    sys.exit() 

#Create PyMMS object
pymms = pymms()

#Create operation modes, these could change in the future
defaults = pymms.operation_modes(defaults)

class ImageAcquisitionThread(threading.Thread):
    def __init__(self, size=5):
        super(ImageAcquisitionThread, self).__init__()
        self._running = True
        self._size = size
        self._image_queue = queue.Queue(maxsize=2)

    def get_output_queue(self):
        return self._image_queue

    def stop(self):
        self._running = False

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

class run_camera(QtCore.QObject):
    '''
    Main function for analysing frames from the camera.

    ImageAcquisition thread grabs the data from the camera and passes it.
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
        for a in range(4):
            img = img_array[x+a]
            img_array[x+a] = ((img / np.max(img)) * 255)
        return img_array

    def task(self):
        if not self._isRunning:
            self._isRunning = True
        
        print('Starting camera.')
        #Determine size of image array pimms should pass back
        size = 5
        if self._window._exp_type.currentIndex() == 1: size = 4
        #Make a worker for the thread (QObject)
        image_acquisition_thread = ImageAcquisitionThread(size)
        #Make a queue to store image data
        image_queue = image_acquisition_thread.get_output_queue()
        #Start taking pictures after camera setup
        image_acquisition_thread.start()

        #Variables to keep track of fps and averaging
        shot_number = 1
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
                    #If user wants cumulative images
                    if self._window._tof_view.currentIndex() == 4:
                        tof_i = images[1:]
                    else:
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
                colour_image = (cm(image)[:,:,:3] * 255).astype(np.uint8)
                self._window._image_ref.set_data(colour_image)

                #Update image canvas
                self._window._imgc.draw()
        
        image_acquisition_thread.stop()
        image_acquisition_thread.join()

        self.finished.emit()
        print('Camera stopping.')

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

class Progress_Bar_Worker(QtCore.QObject):
    progressChanged = QtCore.pyqtSignal(int)

    def __init__(self, window):
        super(Progress_Bar_Worker, self).__init__()
        self._window = window

    def task(self):
        while True:
            if self._window._pb == False:
                time.sleep(0.1)
            else:
                wait = self._window._timeout / 100
                progressbar_value = 0
                while progressbar_value < 101:
                    self.progressChanged.emit(progressbar_value)
                    progressbar_value+=1
                    time.sleep(wait)
                self._window._pb = False
                self.progressChanged.emit(0)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        #Load the ui file
        uic.loadUi(uifp,self)

        self._rotation = 0 # Rotation angle of the image
        self._connected = False #Is camera connected?
        self._camera_running = False #Is camera running?
        self._camera_programmed = False
        self._tof_expanded = False
        self._img_expanded = False
        self._timeout = 5 #Timeout for progressbar
        self._pb = False #Should progressbar update?
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
        self._button.clicked.connect(self.run_camera)
        #Sometimes during idle the code tries to quit, prevent it
        quit = QtWidgets.QAction("Quit", self)
        quit.triggered.connect(self.closeEvent)

        '''
        Progress bar included to inform users of runtime left
        '''
        self._pb_thread = QtCore.QThread()
        # Step 2: Create a worker object
        self._pb_worker = Progress_Bar_Worker(self)
        # Step 3: Move worker to the thread
        self._pb_worker.moveToThread(self._pb_thread)
        # Step 4: Connect signals and slots
        self._pb_thread.started.connect(self._pb_worker.task)
        self._pb_worker.progressChanged.connect(self._progressBar.setValue)
        #Start thread
        self._pb_thread.start()

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
        close.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel)
        close = close.exec()

        if close == QtWidgets.QMessageBox.Yes:
            #If camera is connected disconnect it before closing
            if self._connected == True:
                self.run_camera()
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
        if (event.type() == QtCore.QEvent.Close and isinstance(source, QtWidgets.QWidget)):
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

    '''
    From this point down are functions that control the PImMS camera
    '''

    def camera_control(self):
        if self._connected == False:
            self._timeout = 10
            self._pb = True
            self._plainTextEdit.setPlainText(f'Connecting to Camera, please wait!')
            #Connect to the camera and run the initial commands
            pymms.turn_on_pimms(defaults)
            self._camera_connect.setText(f'Connected to Camera')
            self._camera_connect.setStyleSheet("color: green")
            self._camera_connect_button.setText(f'Disconnect')
            self._update_camera.setDisabled(False)
            self._connected = True
        else:
            #Disconnect from the camera
            pymms.close_pimms()
            self._camera_connect.setText(f'Disconnected from camera')
            self._camera_connect.setStyleSheet("color: red")
            self._camera_connect_button.setText(f'Connect')
            self._button.setDisabled(True)
            self._update_camera.setDisabled(True)
            self._connected = False

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
            pymms.start_up_pimms(defaults,trim_file,function=function)
            print('Trim file!')
        else:
            #Update DAC settings and values, upload trim data
            pymms.start_up_pimms(defaults,function=function)
            print('No trim file!')
        self._button.setDisabled(False)
        self._camera_programmed = True

    def run_camera(self):
        '''
        Controls the camera thread.

        Works by creating a pyqt thread object and passing that object a worker.

        When the worker terminates it passes a finished signal which terminates the thread.
        '''
        def set_false(self):
            # Set all values to false when user stops reading data
            self._worker._isRunning = False
            self._camera_running = False
            self._button.setText("Start")

        if self._camera_running == False:
            print('Camera starting')
            #Disable camera options so user doesnt get confused
            self._camera_running = True
            self._button.setText("Stop")
            # Step 1: Create a QThread object
            self._thread = QtCore.QThread()
            # Step 2: Create a worker object
            self._worker = run_camera(self)
            # Step 3: Move worker to the thread
            self._worker.moveToThread(self._thread)
            # Step 4: Connect signals and slots
            self._thread.started.connect(self._worker.task)
            self._worker.finished.connect(self._thread.quit)
            # Step 5: Start the thread
            self._thread.start()
            # Step 6: Restore initial values so camera can run again
            self._thread.finished.connect(
                lambda: set_false(self)
            )

        else:
            self._worker._isRunning = False

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
